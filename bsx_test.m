rng(1);
N1 = 10;
N2 = 1;
I = 3;

A = normrnd(0,1,N1,I);
B = normrnd(0,1,N2,I);
G = zeros(N1*N2,I);

flat = @(Z) Z(:);
for j=1:I
    G(:,j) = flat(A(:,j)*B(:,j)');
end

H = zeros(N1*N2,I);
H = bsxfun(@minus, A,B')