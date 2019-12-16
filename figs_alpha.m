clf
%rng(420);

d = 2;              % Spatial dimension
S = 1;

L = 3;              % Number of layers
H0 = 8*[100,100,10];   % Base Layer widths
N = 2^(7);            % Sample on N or N*N grid

% Stable parameters
par.alpha = 1;
par.beta = 0;
par.gamma = 1;
par.delta = 0;

%myVideo = VideoWriter('myVideoFile'); 
%myVideo.FrameRate = 10; 
%open(myVideo)


if d==1
    XX = linspace(-1,1,N);
else
    [X,Y] = meshgrid(linspace(0,1,N));
    XX = [X(:),Y(:)]';
end

sig = @(z) ReLU(z);  % Activation function
sig_relu = @(z) ReLU(z);  % Activation function

KMax = 100;

alph = linspace(1.01,2,KMax);

% assume beta = delta = 0
U1 = @(xi) pi*normcdf(xi)-pi/2;
W1 = @(xi) -log(normcdf(xi));
L1 = @(xi1,xi2,alpha) sin(alpha*U1(xi1))./cos(U1(xi1)).^(1/alpha).*(cos((1-alpha)*U1(xi1))./W1(xi2)).^(1/alpha-1);

xi1W = cell(L,1);
xi2W = cell(L,1);
xi1A = cell(L,1);
xi2A = cell(L,1);

% Fix normal realizations
H = H0;

W = cell(L,1);
A = cell(L,1);

xi1W{1} = normrnd(0,1,d,H(1));
xi2W{1} = normrnd(0,1,d,H(1));
xi1A{1} = normrnd(0,1,H(1),1);
xi2A{1} = normrnd(0,1,H(1),1);

NW = d*H(1) + H(1);
for l=2:L
    xi1W{l} = normrnd(0,1,H(l-1),H(l));
    xi2W{l} = normrnd(0,1,H(l-1),H(l));
    xi1A{l} = normrnd(0,1,H(l),1);
    xi2A{l} = normrnd(0,1,H(l),1);
    NW = NW + H(l)*H(l-1) + H(l);
end

xi1V = normrnd(0,1,H(L),1);
xi2V = normrnd(0,1,H(L),1);


% Calculate realizations varying alpha
for k=1:KMax
    par.alpha = alph(k);
    H = H0*2^(k-1);
    U = zeros(1,N);
    U2 = zeros(1,N);

    W = cell(L,1);
    A = cell(L,1);

    W{1} = sp(L1(xi1W{1},xi2W{1},par.alpha));
    A{1} = sp(L1(xi1A{1},xi2A{1},par.alpha));

    NW = d*H(1) + H(1);
    for l=2:L
        W{l} = sp(L1(xi1W{l},xi2W{l},par.alpha));
        A{l} = sp(L1(xi1A{l},xi2A{l},par.alpha));
        NW = NW + H(l)*H(l-1) + H(l);
    end

    V = sp(L1(xi1V,xi2V,par.alpha));
    NW = NW + H(L);

    h = sig(A{1}+W{1}'*XX);
    for l=2:(L-1)
        h = sig(A{l} + W{l}'*h);
    end

    h = sig(A{L} + W{L}'*h);

    f = (V'*h)/H(L)^(1/par.alpha);
    %U = ((s-1)*U + abs(gradient(f)))/s;
    %U2 = ((s-1)*U2 +gradient(f).^2)/s;


    %subaxis(1,KMax,k,'Padding',0,'SpacingHoriz',0,'Margin',0.1);
    if d==1
        plot(XX,f);
    else
        surf(X,Y,reshape(f,N,N),'EdgeColor','None');view(2);axis square;
        %colorbar;
        axis off
        colormap jet;
        title(sprintf('alpha = %.2f',par.alpha));
    end
    pause(0.01);
%    frame = getframe(gcf);
%    writeVideo(myVideo, frame);
end

%close(myVideo)


function B = sp(A)
    B = A;%sparse(A.*(abs(A)>0.0));
end