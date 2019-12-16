clc
clearvars
%parameters that can be modified
nx=40;                              %number of points
ny=30;
xmin=0; xmax=1.5;                    %domain dimentions
ymin=0; ymax=1;
dt=0.001;
nstep=5050;   
mu=0.1; 
un=1;us=0;ve=0;vw=0;             % b.c for tangential velocity
maxit=65000;                        
pit=100;                            %plot each 'pit' iterations
plotvorticity=1;                    % 1 to plot vorticity 0 to not plot
%----------------------------------
%--------initializing--------------
%----------------------------------
time=0.0;
dx=(xmax-xmin)/nx;
dy=(ymax-ymin)/ny;
[X,Y]=meshgrid(xmin:dx:xmax,ymin:dy:ymax);
u=zeros(nx+1,ny+2);
v=zeros(nx+2,ny+1);
p=zeros(nx+2,ny+2);
pp=zeros(nx+1,ny+1);
ut=zeros(nx+1,ny+2);
vt=zeros(nx+2,ny+1);
uu=zeros(nx+1,ny+1);
vv=zeros(nx+1,ny+1);
if plotvorticity==1
    w=zeros(nx+1,ny+1);
end
 
for is=1:nstep
    %enforcing b.c by interpolation
    u(1:nx+1,1)=2*us-u(1:nx+1,2);
    u(1:nx+1,ny+2)=2*un-u(1:nx+1,ny+1);
    v(1,1:ny+1)=2*vw-v(2,1:ny+1);
    v(nx+2,1:ny+1)=2*ve-v(nx+1,1:ny+1);
 
    i=2:nx; j=2:ny+1;      % temporary u-velocity
      ut(i,j)=u(i,j)+dt*((-0.25)* ...
           (((u(i+1,j)+u(i,j)).^2-(u(i,j)+u(i-1,j)).^2)/dx ... %DUUDX
           +((u(i,j+1)+u(i,j)).*(v(i+1,j)+v(i,j)) ...          %DUVDY 1
           -(u(i,j)+u(i,j-1)).*(v(i+1,j-1)+v(i,j-1)))/dy)...   %DUVDY 2
           +(mu)*((u(i+1,j)-2*u(i,j)+u(i-1,j))/dx^2 ...        %D2UDX
           +(u(i,j+1)-2*u(i,j)+u(i,j-1))/dy^2));               %D2UDY 
 
    i=2:nx+1; j=2:ny;       % temporary v-velocity
      vt(i,j)=v(i,j)+dt*((-0.25)*...
          (((u(i,j+1)+u(i,j)).*(v(i+1,j)+v(i,j))...          %DUVDX 1
          -(u(i-1,j+1)+u(i-1,j)).*(v(i,j)+v(i-1,j)))/dx...   %DUVDX 2
          +((v(i,j+1)+v(i,j)).^2-(v(i,j)+v(i,j-1)).^2)/dy)... %DVVDY
          +(mu)*((v(i+1,j)-2*v(i,j)+v(i-1,j))/dx^2 ...       %D2VDX
          +(v(i,j+1)+-2*v(i,j)+v(i,j-1))/dy^2));             %D2VDY
 
    pt=p;
    for it=1:maxit                 % solve for pressure
      % neumann b.c for pressure
      pt(1,:)=pt(2,:);
      pt(nx+2,:)=pt(nx+1,:);
      pt(:,1)=pt(:,2);
      pt(:,ny+2)=pt(:,ny+1);
       
      i=2:nx+1; j=2:ny+1; 
         pt(i,j)=(0.5/(dx^2+dy^2))*((dy^2)*(pt(i+1,j)+pt(i-1,j))...
          +(dx^2)*(pt(i,j+1)+pt(i,j-1))...
          -(dx*dy/dt)*(dy*(ut(i,j)-ut(i-1,j))...
          +dx*(vt(i,j)-vt(i,j-1))));
       
        Er=max(max(pt-p));
        p=pt;
        if Er<10^-6
            break;   
        end
    end
                                   % correct the velocity
    u(2:nx,2:ny+1)=...
           ut(2:nx,2:ny+1)-(dt/dx)*(p(3:nx+1,2:ny+1)-p(2:nx,2:ny+1));
    v(2:nx+1,2:ny)=...
           vt(2:nx+1,2:ny)-(dt/dy)*(p(2:nx+1,3:ny+1)-p(2:nx+1,2:ny));
 
    time=time+dt ;                  % plot the results
    if is==pit*ceil(is/pit)
        
        uu(1:nx+1,1:ny+1)=0.5*(u(1:nx+1,2:ny+2)+u(1:nx+1,1:ny+1));
        vv(1:nx+1,1:ny+1)=0.5*(v(2:nx+2,1:ny+1)+v(1:nx+1,1:ny+1));        
        pp=0.5*(p(1:nx+1,1:ny+1)+p(2:nx+2,2:ny+2));
        hold off,h=pcolor(X,Y,flipud(rot90(pp)));
        set(h,'FaceAlpha',0.5,'EdgeAlpha',0);
        hold on;
        quiver(X,Y,flipud(rot90(uu)),flipud(rot90(vv)),2,'color',[0 0 0]);
        title(time);
        if plotvorticity==1
            w(1:nx+1,1:ny+1)=(u(1:nx+1,2:ny+2)-u(1:nx+1,1:ny+1))/(2*dx)...
                   +(v(1:nx+1,1:ny+1)-v(2:nx+2,1:ny+1))/(2*dy); 
          contour(X,Y,flipud(rot90(w)),10);
        end
        axis equal,pause(0.005),drawnow
    end
 
end