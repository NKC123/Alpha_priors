%rng(69);

%% MCMC parameters
mcmc.method = 'pcn';    % MCMC method (pcn, mala)
mcmc.mix_ratio = 0;   % Ratio of MALA steps to pCN steps (mixed method)
S = 2000;      % MCMC samples
mcmc.beta = 0.01;       % pCN jump parameter
mcmc.h = 0.0001;                 % MALA step size parameter
mcmc.local = 100;        % how many steps to calculate local acceptance rate?
mcmc.thinning = 100;    % how often to update figure?
mcmc.burnin = 300000;
mcmc.b_acc_target = 0.2;
mcmc.h_acc_target = 0.2;

%% Model parameters
md.d = 1;              % Spatial dimension 
md.H = [100,10,10];    % Layer widths
md.L = 2;              % Number of hidden layers
md.N = 2^7;            % Sample on N or N*N grid
md.J = 50;
md.gm = 0.2;

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

%UT = (0.5*(sign(md.XX+0.5))) + (cos(2*pi*md.XX)-1).*(md.XX>0);

obsInd = round(linspace(1,md.N,md.J));
basis = speye(md.N);

B = zeros(md.N);
for j=1:md.N
    B(:,j) = idct(dct(full(basis(:,j))).*exp(-0.0*(1:md.N)'));
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
md.sig = @(z) tanh(z);
md.dsig = @(z) 2*sqrt(2)*normpdf(z*sqrt(2));

% Non-centring function
md.lambda = @(z) tan(pi*normcdf(z)-pi/2);
md.dlambda = @(z) pi*sec(pi*normcdf(z)-pi/2).^2.*normpdf(z);

subplot(141);
plot(md.XX,UT);axis([-1,1,-2,1]);
hold on
scatter(md.XX(obsInd),y,'x');
hold off

%{
tic
for j=1:1000
xi = normrnd(0,1,md.NW,1);
[~,PhiXi,DPhiXi] = build_net(xi,md);
end
toc
return
%}

% Finite Difference test
%
xi = normrnd(0,1,md.NW,1);
U = normrnd(0,1,md.N,1);

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

subplot(131)
plot(DPhi_adj);
subplot(132)
plot(DPhi_fd);
subplot(133)
plot(DPhi_adj./DPhi_fd);
return
%

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

        % Build network
        z = cell(L+1,1);
        a = cell(L+1,1);

        z{1} = W{1}*XX' + B{1};
        a{1} = model.sig(z{1});
        for l=2:L
            z{l} = W{l}*a{l-1} + B{l};
            a{l} = model.sig(z{l});
        end
        z{L+1} = W{L+1}*a{L};
        a{L+1} = z{L+1};  

        % Calculate cost derivative (backprop);
        delta = cell(L+1,1);
        delta{L+1} = ones(1,N);

        for l=L:-1:1
           delta{l} = (W{l+1}'*delta{l+1}).*model.dsig(z{l}); 
        end

        temp = delta{1}.*XX';
        dC = temp;
        dC = [dC;delta{1}];
        for l=2:L+1
            temp2 = delta{l}(:,1)*a{l-1}(:,1)';
            temp = temp2(:);
            for n=2:N
                temp2 = delta{l}(:,n)*a{l-1}(:,n)';
                temp = [temp,temp2(:)];          
            end
            dC = [dC;temp];
            if l < L+1
                dC = [dC;delta{l}];
            end
        end

        dC = model.dlambda(xi0).*dC;
        DPhiXi = dC*dcost/H(L);

    end
end

    

