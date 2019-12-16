%rng(69);
clf

%% MCMC parameters
mcmc.beta = 0.01;       % pCN jump parameter
mcmc.h = 0.0000001;                 % MALA step size parameter
mcmc.local = 500;        % how many steps to calculate local acceptance rate?
mcmc.thinning = 100;    % how often to update figure?
mcmc.burnin = 1000;

%% Model parameters
md.d = 1;              % Spatial dimension 
md.H = [100,10,10,10,10];    % Layer widths
md.L = 3;              % Number of hidden layers
md.N = 2^7;            % Sample on N or N*N grid
md.J = 50;
md.gm = 0.05;

% Calculate total number of weights
md.NW = md.d*md.H(1) + md.H(1);
for l=2:md.L
    md.NW = md.NW + md.H(l)*md.H(l-1) + md.H(l);
end
md.NW = md.NW + md.H(md.L);

% Define the spatial grid
md.XX = linspace(-1,1,md.N)';

% Create the truth, forward map, data
%UT = (0.5*(sign(md.XX+0.5)-sign(md.XX-0.5))-0.5)+cos(4*pi*md.XX).*(md.XX>0.5);
UT = (0.5*(sign(md.XX+0.5)-sign(md.XX-0.5))-0.5)...
        - 2*(0.5*(sign(md.XX-0.5)-sign(md.XX-0.7))-0.5).*(md.XX>0.5)...
        + 1*(0.5*(sign(md.XX-0.5)-sign(md.XX-0.7))-0.5).*(md.XX>0.7);

obsInd = round(linspace(1,md.N,md.J));
basis = speye(md.N);

B = zeros(md.N);
for j=1:md.N
    B(:,j) = idct(dct(full(basis(:,j))).*exp(-0.05*(1:md.N)'));
end

A = basis(obsInd,:);
G = A*B;

y = G*UT + normrnd(0,1,md.J,1)*md.gm;

% Objective function
J = @(xi) JFull(xi,md);

% Cost function
md.cost = @(U) norm(G*U-y)^2/(2*md.gm^2);
md.dcost = @(U) G'*(G*U-y)/md.gm^2;

% Activation function
md.sig = @(z) erf(z);
md.dsig = @(z) 2*sqrt(2)*normpdf(z*sqrt(2));

%{
md.tt = 100;
md.sig = @(z) log(1+exp(md.tt*z))./md.tt;
md.dsig = @(z) exp(md.tt*z)./(1+exp(md.tt*z));
%}

% Non-centring function
%
md.lambda = @(z) 0.5*tan(pi*normcdf(z)-pi/2);
md.dlambda = @(z) 0.5*pi*sec(pi*normcdf(z)-pi/2).^2.*normpdf(z);
%

%md.lambda = @(z) z;
%md.dlambda = @(z) z.^0;

subplot(233);
plot(md.XX,UT);axis([-1,1,-2,1]);
hold on
scatter(md.XX(obsInd),y,'x');
hold off


 
% Finite Difference test
%{
xi = 0.1*normrnd(0,1,md.NW,1);
U = normrnd(0,1,md.N,1);
%U = build_net(xi,md);

basis = speye(md.NW);
basisN = speye(md.N);
[~,PhiXi,DPhi_adj] = build_net(xi,md);
Cost0 = md.cost(U);
DCost_adj = md.dcost(U);
DPhi_fd = zeros(md.NW,1);
DCost_fd = zeros(md.N,1);

ep = 1e-8;
for j=1:md.NW
    [~,PhiXiP] = build_net(xi+ep*basis(:,j),md);
    DPhi_fd(j) = (PhiXiP-PhiXi)/ep;
end
for j=1:md.N
    DCost_fd(j) = (md.cost(U+ep*basisN(:,j))-Cost0)/ep;
end

%{
subplot(131)
plot(DCost_adj);
subplot(132)
plot(DCost_fd);
subplot(133)
semilogy(abs(DCost_adj-DCost_fd)./abs(DCost_fd));
%}

%
subplot(131)
plot(DPhi_adj);
subplot(132)
plot(DPhi_fd);
subplot(133)
semilogy(abs(DPhi_adj-DPhi_fd));
%

return
%}

%% Solve Langevin equation
%
S = 1e6;
UMean = zeros(md.N,1);
UVar = zeros(md.N,1);
xiMean = zeros(md.NW,1);

%xi = 0.1*normrnd(0,1,md.NW,1);
a0 = 2e-5;
maxIt = 100;

PhiTrace = zeros(S,1);
DPhiTrace = zeros(S,1);
nXiTrace = zeros(S,1);
PhiTrace0 = zeros(S,1);

for j=1:S
    [U,PhiXi,DPhiXi] = build_net(xi,md);
    PhiTrace(j) = PhiXi + norm(xi)^2/2;
    DPhiTrace(j) = norm(DPhiXi + xi);
    nXiTrace(j) = norm(xi)^2/2;
    PhiTrace0(j) = PhiXi;
    
    if mod(j,mcmc.thinning) == 0
        fprintf('Phi = %f\t DPhi = %f\n',PhiTrace(j),DPhiTrace(j));
        
        subplot(231)
        semilogy(1:j,PhiTrace(1:j));
        
        subplot(232);
        semilogy(1:j,DPhiTrace(1:j));
        
        %{
        subplot(234);
        plot(1:j,PhiTrace0(1:j));
        subplot(235);
        plot(1:j,nXiTrace(1:j));
        %}
        
        %
        ax3 = subplot(236);
        cla(ax3);
        plot(md.XX,UT);
        hold on
        plot(md.XX,U);axis([-1,1,-2,1]);
        hold off        
        pause(0.01);
        
        if j > mcmc.burnin
            q = (j-mcmc.burnin)/mcmc.thinning;
            UMean = ((q-1)*UMean + U)/q;
            UVar = ((q-1)*UVar + U.^2)/q;
            xiMean = ((q-1)*xiMean + xi)/q;

            subplot(234);
            plot(md.XX,[UMean,UMean+sqrt(UVar-UMean.^2),UMean-sqrt(UVar-UMean.^2)]);axis([-1,1,-2,1]);
            subplot(235);
            plot(md.XX,build_net(xiMean,md));axis([-1,1,-2,1]);
            pause(0.01);
        end
        
    end
    
    %[ac,~] = LineSearch(xi,-DPhiXi-xi,a0,maxIt,J);
    ac = a0;
    xi = xi - ac*(DPhiXi + xi);% + sqrt(2*ac)*normrnd(0,1,md.NW,1);
    
    if ac < 1e-16
        maxIt = min(1e3,maxIt*2);
    end

end
disp(size(U))
disp(PhiXi);
plot(md.XX,U)



function z = JFull(xi,model)
    [~,PhiXi] = build_net(xi,model);
    z = PhiXi + norm(xi)^2/2;
end

function [U,PhiXi,DPhiXi] = build_net(xi,model)

    N = model.N;
    L = model.L;
    H = model.H;
    XX = model.XX;
    
    xi0 = xi;
    xi = model.lambda(xi0);
    % Extract weights/biases from xi
    W = cell(L+1,1);
    B = cell(L,1);
    
    curr_ind = 1;
    next_ind = model.d*H(1);
    W{1} = reshape(xi(curr_ind:next_ind),H(1),model.d);
    curr_ind = next_ind;
    next_ind = curr_ind + H(1);
    B{1} = xi(curr_ind+1:next_ind);
    
    for l=2:L
        curr_ind = next_ind;
        next_ind = curr_ind + H(l-1)*H(l);
        W{l} = reshape(xi(curr_ind+1:next_ind),H(l),H(l-1));
        curr_ind = next_ind;
        next_ind = curr_ind + H(l);
        B{l} = xi(curr_ind+1:next_ind);
    end
    curr_ind = next_ind;
    W{L+1} = reshape(xi(curr_ind+1:end),1,H(L));
    
    % Calculate U
    U = W{1}*XX' + B{1};
    U = model.sig(U);
    for l=2:L
        U = model.sig(W{l}*U + B{l});
    end
    U = (W{L+1}*U)'/H(L);
    
    % Return Phi(U)
    if nargout > 1
        PhiXi = model.cost(U);
    end
    
    % Return DPhi(U)  
    if nargout > 2
    
        dcost = model.dcost(U);
        for n = 1:N
            % Build network
            z = cell(L+1,1);
            a = cell(L+1,1);

            z{1} = W{1}*XX(n) + B{1};
            a{1} = model.sig(z{1});
            for l=2:L
                z{l} = W{l}*a{l-1} + B{l};
                a{l} = model.sig(z{l});
            end
            z{L+1} = W{L+1}*a{L};
            a{L+1} = z{L+1};

            % Calculate cost derivative (backprop);
            delta = cell(L+1,1);
            delta{L+1} = 1;

            for l=L:-1:1
               delta{l} = (W{l+1}'*delta{l+1}).*model.dsig(z{l}); 
            end

            temp = delta{1}.*XX(n)';
            dC{n} = temp(:);
            dC{n} = [dC{n};delta{1}];
            for l=2:L
               temp = delta{l}*a{l-1}';
               dC{n} = [dC{n};temp(:)];
               dC{n} = [dC{n};delta{l}];
            end
            temp = delta{L+1}*a{L}';
            dC{n} = [dC{n};temp(:)];  
            dC{n} = model.dlambda(xi0).*dC{n};
        end
        dCF = zeros(model.NW,N);
        for n=1:N
            dCF(:,n) = dC{n};
        end
        DPhiXi = dCF*dcost/H(L);

    end

end

    

