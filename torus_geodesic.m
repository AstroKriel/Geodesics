clc

k = 5;   % shape of Torus
c = k/5; % nature of geodesic

% generating torus
theta_f = linspace(0,2*pi,50);
phi_f   = linspace(0,2*pi,50);
[T,P]   = meshgrid(theta_f,phi_f);

% plotting torus
xf = (k+cos(P)).*cos(T);
yf = (k+cos(P)).*sin(T);
zf = sin(P);
figure
surf(xf,yf,zf)
axis equal
shading interp

if c == 0
    disp("if-statement: 1.0")
    theta1 = 0;
    z = sin(P).*cos(T-T);
    x = (k+cos(P))*cos(theta1);
    y = (k+cos(P))*sin(theta1);
    z1 = sin(P).*cos(T-T);
    x1 = (k+cos(P))*cos(theta1+(pi/6));
    y1 = (k+cos(P))*sin(theta1+(pi/6));
    z2 = sin(P).*cos(T-T);
    x2 = (k+cos(P))*cos(theta1+(pi));
    y2 = (k+cos(P))*sin(theta1+(pi));
    hold on
    plot3(x,y,z,'r',x1,y1,z1,'r',x2,y2,z2,'r')
    axis equal
elseif (0 < c) && (c < (k-1))
    disp("if-statement: 2.0")
    theta1=0;
    phi1=0;
    phi2=360;
    % ODE45 helps in solving differential equations using RUNGA-KETTA method
    dydt = @(t,y) c / ((k+cos(t)) * sqrt(((k+cos(t))^2 -c^2)));
    [t,y]  =  ode45(dydt , [phi1*pi/180,phi2*pi/180], theta1);
    p  =  y(end);
    a  =  floor(2*pi/p);
    hold on
    for i=1:a+1
        xt1  =  (k+cos(t)).*cos(y+(i-1)*p);
        yt1  =  (k+cos(t)).*sin(y+(i-1)*p);
        zt1  =  sin(t);
        plot3(xt1,yt1,zt1,'r','linewidth',1)
    end
%     % plotting only upto phi  =  720 degrees becuase it is a repetitive plot
%     figure
%     plot(t*180/pi,y*180/pi,'b',(t*180/pi)+360,(p+y)*180/pi,'b')
%     xlabel('\phi')
%     ylabel('\theta')
%     title('\theta vs \phi')
elseif c == k-1
    theta1 = 0;
    Phi1 = 2;
    if phi1 == pi
        disp("if-statement: 3.1")
        z =sin(P-P); x  =  (k-1)*cos(P);y  = (k-1)*sin(P);
        hold on
        plot3(x,y,z,'r','linewidth',1.1)
        axis equal
    else
        disp("if-statement: 3.2")
        phi1=-pi+1e-3;phi2=pi-1e-3;
        dydt  =  @(t,y) c/((k+cos(t))*sqrt(((k+cos(t))^2 -c^2)));
        [t,y]  =  ode45(dydt,[phi1,phi2],theta1);
        xg  =  (k+cos(t)).*cos(y);
        yg  =  (k+cos(t)).*sin(y);
        zg  =  sin(t);
        z1  =  sin(P-P);
        x1  =  (k-1)*cos(P);
        y1  =  (k-1)*sin(P);
        hold on
        plot3(xg,yg,zg,'r',x1,y1,z1,'b')
        axis equal
%         figure
%         plot(t,y,'b')
%         grid on
%         xlabel('\phi')
%         ylabel('\theta')
%         title('\theta vs \phi')
    end
elseif ((k-1)<c)&&(c<(k+1))
    disp("if-statement: 4.0")
    phi11  =  -acosd(c-k)*pi/180+(1e-6);
    phi21  =   acosd(c-k)*pi/180-(1e-6);
    theta1  =  0;
    dydt  =  @(t,y) c/((k+cos(t))*sqrt(((k+cos(t))^2 -c^2)));
    [t,y]  =  ode45(dydt , [phi11,phi21],theta1);
    xg  =  (k+cos(t)).*cos(y);yg  =  (k+cos(t)).*sin(y);zg  =  sin(t);
    p  =  y(end);
    a  =  floor(2*pi/p);
    hold on
    for i=1:a+1
        xt1  =  (k+cos(t)).*cos(y+(i-1)*p);
        yt1  =  (k+cos(t)).*sin(y+(i-1)*p);
        zt1  =  sin(((-1)^(i-1))*t);
        plot3(xt1,yt1,zt1,'r','linewidth',1)
    end
    hold on
    z2  =   sin(acosd(c-k)*pi/180)*cos(T-T);
    z3  =  -sin(acosd(c-k)*pi/180)*cos(T-T);
    x2  =   c*cos(P);y2 =c*sin(P);
    plot3(x2,y2,z2,'b',x2,y2,z3,'b')
    axis equal
%     figure
%     for i=1:a+1
%         hold on
%         plot(((-1)^(i-1))*t*180/pi,(y+p*(i-1))*180/pi,'b')
%         grid on
%         xlabel('\phi')
%         ylabel('\theta')
%         title('\theta vs \phi. Oscillation pattern i.e., phi is bounded')
%     end
elseif c  ==  k+1
    disp("if-statement: 5.0")
    z  =  0*cos(P-P);
    x  =  c*cos(P);
    y  =  c*sin(P);
    hold on
    plot3(x,y,z,'r','linewidth',1)
    axis equal
end
