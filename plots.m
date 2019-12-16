

%Plot the different activiation functions

x=linspace(-3,3,400);
sigmoid = 1./(1+exp(-x));
plot(x,g,'m','LineWidth',2);
hold on
plot(x,sigmoid,'r','LineWidth',2);
hold on
plot(x,tanh(x),'b','LineWidth',2);
lgd=legend('ReLU','sigmoid','tanh');
lgd.FontSize = 18;
set(gca,'FontSize',14)
