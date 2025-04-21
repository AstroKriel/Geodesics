## ###############################################################
## DEPENDANCIES
## ###############################################################
from itertools import combinations


## ###############################################################
## FUNCTIONS
## ###############################################################
def plot_mesh_wireframe(ax, vertices, faces):
  num_faces = faces.shape[0]
  for face_index in range(num_faces-1):
    for comb_pair in list(combinations(faces[face_index], 2)):
      ax.plot(
        [ vertices[comb_pair[0],0], vertices[comb_pair[1],0] ],
        [ vertices[comb_pair[0],1], vertices[comb_pair[1],1] ],
        [ vertices[comb_pair[0],2], vertices[comb_pair[1],2] ],
        color="black", ls="-", lw=0.1, zorder=1
      )

def plot_mesh_and_solution(
    ax,
    vertices,
    faces,
    soln_path      : list[float] | None = [],
    start_coord    : tuple[float, float] | None = (),
    end_coord      : tuple[float, float] | None = (),
    plot_wireframe : bool = False,
  ):
  if plot_wireframe: plot_mesh_wireframe(ax, vertices, faces)
  ax.plot(vertices[:,0], vertices[:,1], vertices[:,2], "k.", ms=0.1, alpha=0.5) # plot point-surface
  if len(soln_path) > 0:   ax.plot(soln_path[0], soln_path[1], soln_path[2], color="black", ls="-", lw=2, zorder=3)
  if len(start_coord) > 0: ax.plot(start_coord[0], start_coord[1], color="blue", marker=".", ms=5, zorder=3)
  if len(end_coord) > 0:   ax.plot(end_coord[0], end_coord[1],   color="red", marker=".", ms=5, zorder=3)
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")


## END OF MODULE