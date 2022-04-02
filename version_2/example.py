import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import svd
import networkx as nx
import heapq
from collections import defaultdict


def surface_squares(x_min, x_max, y_min, y_max, steps):
    x = np.linspace(x_min, x_max, steps)
    y = np.linspace(y_min, y_max, steps)
    xx, yy = np.meshgrid(x, y)
    zz = xx**2 + yy**2
    return xx, yy, zz


def get_meshgrid_ax(x, y, z):
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_surface(X=x, Y=y, Z=z)
    # return ax
    fig = mlab.figure()
    su = mlab.surf(x.T, y.T, z.T, warp_scale=0.1)


def get_knn(flattened_points, num_neighbors):
    # need the +1 because each point is its own nearest neighbor
    knn = NearestNeighbors(num_neighbors+1)
    # normalize flattened points when finding neighbors
    neighbor_flattened = (flattened_points - np.min(flattened_points, axis=0)) / (np.max(flattened_points, axis=0) - np.min(flattened_points, axis=0))
    knn.fit(neighbor_flattened)
    dist, indices = knn.kneighbors(neighbor_flattened)
    return dist, indices


def rotmatrix(axis, costheta):
    """ Calculate rotation matrix

    Arguments:
    - `axis`     : Rotation axis
    - `costheta` : Rotation angle
    """
    x, y, z = axis
    c = costheta
    s = np.sqrt(1-c*c)
    C = 1-c
    return np.matrix([[x*x*C+c,    x*y*C-z*s,  x*z*C+y*s],
                      [y*x*C+z*s,  y*y*C+c,    y*z*C-x*s],
                      [z*x*C-y*s,  z*y*C+x*s,  z*z*C+c]])


def plane(Lx, Ly, Nx, Ny, n, d):
    """ Calculate points of a generic plane 

    Arguments:
    - `Lx` : Plane Length first direction
    - `Ly` : Plane Length second direction
    - `Nx` : Number of points, first direction
    - `Ny` : Number of points, second direction
    - `n`  : Plane orientation, normal vector
    - `d`  : distance from the origin
    """

    x = np.linspace(-Lx/2, Lx/2, Nx)
    y = np.linspace(-Ly/2, Ly/2, Ny)
    # Create the mesh grid, of a XY plane sitting on the orgin
    X, Y = np.meshgrid(x, y)
    Z = np.zeros([Nx, Ny])
    n0 = np.array([0, 0, 1])

    # Rotate plane to the given normal vector
    if any(n0 != n):
        costheta = np.dot(n0, n)/(np.linalg.norm(n0)*np.linalg.norm(n))
        axis = np.cross(n0, n)/np.linalg.norm(np.cross(n0, n))
        rotMatrix = rotmatrix(axis, costheta)
        XYZ = np.vstack([X.flatten(), Y.flatten(), Z.flatten()])
        X, Y, Z = np.array(rotMatrix*XYZ).reshape(3, Nx, Ny)

    eps = 0.000000001
    dVec = d #abs((n/np.linalg.norm(n)))*d#np.array([abs(n[i])/np.linalg.norm(n)*val if abs(n[i]) > eps else val for i, val in enumerate(d)]) #
    X, Y, Z = X+dVec[0], Y+dVec[1], Z+dVec[2]
    return X, Y, Z


def build_proxy_graph(proxy_n_dist, proxy_n_indices):
    G = nx.Graph()

    for distance_list, neighbor_list in zip(proxy_n_dist, proxy_n_indices):
        # first element is always point
        current_node = neighbor_list[0]
        neighbor_list = neighbor_list[1:]
        distance_list = distance_list[1:]
        for neighbor, dist in zip(neighbor_list, distance_list):
            G.add_edge(current_node, neighbor, weight=dist)
    return G


def get_plane_points(normal_vec, initial_point, min_range=-10, max_range=10, steps=1000):
    steps_for_plane = np.linspace(min_range, max_range, steps)
    xx, yy = np.meshgrid(steps_for_plane, steps_for_plane)
    d = -initial_point.dot(normal_vec)
    eps = 0.000000001
    if abs(normal_vec[2]) < eps and abs(normal_vec[1]) > eps:
        zz = (-xx*normal_vec[2] - yy*normal_vec[0] - d)/normal_vec[1]
    else:
        zz = (-xx*normal_vec[0] - yy*normal_vec[1] - d)/normal_vec[2]
    return xx, yy, zz


# def plot_tangent_plane_at_point(pointset, flattened_points, node, normal_vec):
#     ax = get_meshgrid_ax(x=pointset[:, :, 0], y=pointset[:, :, 1], z=pointset[:, :, 2])
#     node_loc = flattened_points[node]
#     print("Node loc: {}".format(node_loc))
#     xx, yy, zz = plane(10, 10, 500, 500, normal_vec, node_loc)
#     # xx, yy, zz = get_plane_points(normal_vec, node_loc)
#     print("Normal Vec: {}".format(normal_vec))
#     ax.plot_surface(X=xx, Y=yy, Z=zz)
#     ax.plot([node_loc[0]], [node_loc[1]], [node_loc[2]], markerfacecolor='k', markeredgecolor='k', marker='o', markersize=10)
#     plt.show()


def generate_tangent_spaces(proxy_graph, flattened_points):
    # This depth should gaurantee at least 16 neighbors
    tangent_spaces = {}
    for node in proxy_graph.nodes():
        neighbors = list(nx.neighbors(proxy_graph, node))
        node_point = flattened_points[node]
        zero_mean_mat = np.zeros((len(neighbors)+1, len(node_point)))
        for i, neighbor in enumerate(neighbors):
            zero_mean_mat[i] = flattened_points[neighbor]
        zero_mean_mat[-1] = node_point

        zero_mean_mat = zero_mean_mat - np.mean(zero_mean_mat, axis=0)
        u, s, v = svd(zero_mean_mat.T)
        # smat = np.zeros(u.shape[0], v.shape[0])
        # smat[:s.shape[0], :s.shape[0]] = np.diag(s)
        tangent_spaces[node] = u
    return tangent_spaces


def geodesic_single_path_dijkstra(flattened_points, proximity_graph, tangent_frames, start, end):
    # short circuit
    if start == end:
        return []
    # Create min priority queue
    minheap = []
    pred = {}
    dist = defaultdict(lambda: 1.0e+100)
    # for i, point in enumerate(flattened_points):
    R = {}
    t_dist = {}
    geo_dist = {}
    R[start] = np.eye(3)
    t_dist[start] = np.ones((3,))
    dist[start] = 0
    start_vector = flattened_points[start]
    for neighbor in nx.neighbors(proxy_graph, start):
        pred[neighbor] = start
        dist[neighbor] = np.linalg.norm(start_vector - flattened_points[neighbor])
        heapq.heappush(minheap, (dist[neighbor], neighbor))
    while minheap:
        r_dist, r_ind = heapq.heappop(minheap)
        if r_ind == end:
            break
        q_ind = pred[r_ind]
        u, s, v = svd(tangent_frames[q_ind].T*tangent_frames[r_ind])
        R[r_ind] = np.dot(R[q_ind], u * v.T)
        t_dist[r_ind] = t_dist[q_ind]+np.dot(R[q_ind], tangent_frames[q_ind].T * (r_dist - dist[q_ind]))
        geo_dist[r_ind] = np.linalg.norm(t_dist[r_ind])
        for neighbor in nx.neighbors(proxy_graph, r_ind):
            temp_dist = dist[r_ind] + np.linalg.norm(flattened_points[neighbor] - flattened_points[r_ind])
            if temp_dist < dist[neighbor]:
                dist[neighbor] = temp_dist
                pred[neighbor] = r_ind
                heapq.heappush(minheap, (dist[neighbor], neighbor))
    # found ending index, now loop through preds for path
    current_ind = end
    node_path = [end]
    while current_ind != start:
        node_path.append(pred[current_ind])
        current_ind = pred[current_ind]

    return node_path


def plot_path_on_surface(pointset, flattened_points, path):
    # ax = get_meshgrid_ax(x=pointset[:, :, 0], y=pointset[:, :, 1], z=pointset[:, :, 2])
    # ax.plot(points_in_path[:, 0], points_in_path[:, 1], points_in_path[:, 2], linewidth=10.0)
    # plt.show()
    get_meshgrid_ax(x=pointset[:, :, 0], y=pointset[:, :, 1], z=pointset[:, :, 2])
    points_in_path = flattened_points[path]
    mlab.plot3d(points_in_path[:, 0], points_in_path[:, 1], points_in_path[:, 2] *.1)
    mlab.show()


"""
    True geodesic of graph.
    Build proximity graph
    Find tangent space using geodisic neighborhood at each point in graph
    Parallel transport vectors between tangent space points
    Use this as your distance metric
    Dijkstra's Algorithm
"""
if __name__ == "__main__":
    x, y, z = surface_squares(-5, 5, -5, 5, 500)
    # plot_meshgrid(x, y, z)
    pointset = np.stack([x, y, z], axis=2)
    proxy_graph_num_neighbors = 16
    flattened_points = pointset.reshape(pointset.shape[0]*pointset.shape[1], pointset.shape[2])
    flattened_points = flattened_points
    proxy_n_dist, proxy_n_indices = get_knn(flattened_points, proxy_graph_num_neighbors)
    # Generate a proximity graph using proxy_graph_num_neighbors
    # Nodes = number of points, max # of edges = number of points * num_neighbors
    proxy_graph = build_proxy_graph(proxy_n_dist, proxy_n_indices)
    # Now, using the geodesic_num_neighbors, get geodesic neighborshood for tangent space construction
    tangent_spaces = generate_tangent_spaces(proxy_graph, flattened_points)
    node_to_use = 2968
    # 3rd vector of tangent space is normal to plane
    # plot_tangent_plane_at_point(pointset, flattened_points, node_to_use, tangent_spaces[node_to_use][:, 2])
    path = geodesic_single_path_dijkstra(flattened_points, proxy_graph, tangent_spaces, 250, 249750)
    plot_path_on_surface(pointset, flattened_points, path)

    