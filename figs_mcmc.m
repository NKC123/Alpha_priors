col = 0.8*[1,1,1]; %std shading color
load('run_pcn.mat');

% Truth and observations
figure(1);
clf
subplot(131);
plot(md.XX,UT,'LineWidth',2,'Color','k');
axis([-1,1,-2,1]);
hold on
scatter(md.XX(obsInd),y,'x','LineWidth',2);
hold off
title('True state and observations','FontSize',16);
ylabel('$u^\dagger(x)$','Interpreter','latex','FontSize',20);
xlabel('$x$','Interpreter','latex','FontSize',20);
grid on
box on

% Cauchy MCMC output
subplot(132);
U_std = sqrt(UVar-UMean.^2);
axis([-1,1,-2,1]);
hold on
fill([md.XX',fliplr(md.XX')],[UMean'+U_std',fliplr(UMean'-U_std')],col)
plot(md.XX,UMean-U_std,'Color',col);
plot(md.XX,UMean+U_std,'Color',col);
plot(md.XX,UMean,'LineWidth',2,'Color','k');
hold off

title('Cauchy NN Prior','FontSize',16);
ylabel('$\mathrm{E}(u(x)) \pm \mathrm{std}(u(x))$','Interpreter','latex','FontSize',20);
xlabel('$x$','Interpreter','latex','FontSize',20);
grid on
box on

% Gaussian regression output
subplot(133);
% Define Gaussian prior
tau = 20;
nu = 4;
sig = 5;

[~,~,L] = laplacian(md.N,{'NN'});
L = L*(md.N-1)^2;
P = (L+tau^2*speye(md.N))^nu/(tau^(2*nu))/sig^2;

UMean = (G'*G/md.gm^2 + P)\(G'*y/md.gm^2);
U_std = sqrt(diag(inv(G'*G/md.gm^2 + P)));

axis([-1,1,-2,1]);
hold on
fill([md.XX',fliplr(md.XX')],[UMean'+U_std',fliplr(UMean'-U_std')],col)
plot(md.XX,UMean-U_std,'Color',col);
plot(md.XX,UMean+U_std,'Color',col);
plot(md.XX,UMean,'LineWidth',2,'Color','k');
hold off

title('Gaussian Prior','FontSize',16);
ylabel('$\mathrm{E}(u(x)) \pm \mathrm{std}(u(x))$','Interpreter','latex','FontSize',20);
xlabel('$x$','Interpreter','latex','FontSize',20);
grid on
box on

% Maybe useless figure (dependence of Gaussian mean on parameter tau)
figure(2);
clf
tauMin = 5;
tauMax = 40;
tauNum = 10;
tauSet = linspace(tauMin,tauMax,tauNum);
UMeans = zeros(md.N,tauNum);

for j=1:tauNum
    tau = tauSet(j);
    P = (L+tau^2*speye(md.N))^nu/(tau^(2*nu))/sig^2;
    UMeans(:,j) = (G'*G/md.gm^2 + P)\(G'*y/md.gm^2);
end


[X,Y] = meshgrid(tauSet,md.XX);
plot3(X,Y,UMeans,'linewidth',2)
grid on
xlabel('$\tau$','Interpreter','latex','FontSize',20)
ylabel('$x$','Interpreter','latex','FontSize',20)
zlabel('$\mathrm{E}(u(x))$','Interpreter','latex','FontSize',20)