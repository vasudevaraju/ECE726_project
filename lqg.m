%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%authors: Sreechakra Vasudeva Raju Rachavelpula, Mohammad Sareeb Hakak

%This code first generates a control for the linearized plant model of the 
%quadcoptor by solving the finite-horizon tracking LQR problem. Then it
%calulates gain for an LQG observer by solving its observer LQR
%counterpart. The covariance matrices for process noise and measurement
%noise are selected based on the plant and sensors. Here they were set
%very low to make the simulation fast

%The states of the system are so :
%[x,x.,y,y.,z,z.,p,q,r,ph,th,ps] where
%x,y,z are the x,y,z position coordinates of center of quadcoptor in IRF
%x.,y.,z. are the velocity components in the x,y,z directions in IRF
%p, q, r are the angular velocities about x,y,z axes in body frame
%ph ,th, ps are the euler angles defining orientation of body frame wrt IRF
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% This section solves the DRE to get the FHLQT gain 
clear;clc;
g = 9.81; m = 1.4; Ix=0.03;Iy =0.03; Iz =0.04;

A = [ ...
[ 0, 1, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0]
[ 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, g, 0]
[ 0, 0, 0, 1, 0, 0, 0, 0, 0,  0, 0, 0]
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, -g, 0, 0]
[ 0, 0, 0, 0, 0, 1, 0, 0, 0,  0, 0, 0]
[ 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0]
[ 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0]
[ 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0]
[ 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0]
[ 0, 0, 0, 0, 0, 0, 1, 0, 0,  0, 0, 0]
[ 0, 0, 0, 0, 0, 0, 0, 1, 0,  0, 0, 0]
[ 0, 0, 0, 0, 0, 0, 0, 0, 1,  0, 0, 0]];
 
B = [...
[   0,    0,    0,    0]
[   0,    0,    0,    0]
[   0,    0,    0,    0]
[   0,    0,    0,    0]
[   0,    0,    0,    0]
[ 1/m,    0,    0,    0]
[   0, 1/Ix,    0,    0]
[   0,    0, 1/Iy,    0]
[   0,    0,    0, 1/Iz]
[   0,    0,    0,    0]
[   0,    0,    0,    0]
[   0,    0,    0,    0]];

C = [1,zeros(1,11);0,0,1,zeros(1,9);0,0,0,0,1,zeros(1,7);zeros(1,11),1];
Q = diag([10;10;1000;100]);
R = 1000*eye(4);
F = Q; %F is terminal state cost
ft = 10; %simulation time
tspan = [0:0.01:ft];

%initial conditions are the terminal conditions 
K0 = reshape(C'*F*C,[144,1]); 
S0 = -C'*F*y_des(ft);
%K and S are solutions of the DRE

[T,Kaug] = ode45(@(t,x) proj_dre(t,x,y_des(ft-t),A,B,C,Q,R), tspan, [K0;S0]);

%flip the time and gain matrices because the DRE is solved in backward time
t = ft-flip(T);  
Kaug = flip(Kaug); 
%Kaug is a time-series matrix containing row vectors with augmented K and S elements

%% This section solves the observer LQR problem to get the LQG observer gain
%Very low covariances are chosen to make the simulation faster
clc
W2 = 0.01*eye(12); %sensor noise covariance matrix
W1 = 0.001*eye(12); %process noise covariance matrix
C1 = eye(12);

[K1,G,P1] = lqr(A',C1',W1,W2);

L= inv(W2)*C1*G;

L= L';

%% This section simulates the FHLQT controller with an LQG observer
%The simulation is very slow with a variable-step solver such as ode45
%It is recommended to simulate this a fixed-step solver or convert the
%continuous closed-loop system dynamics to a discrete system.
x0 = [zeros(1,12) 10*randn(1,12)];
[t,x] = ode45(@(T,X)proj_k(T,X,A,B,C1,L,t,Kaug,R),[0,20],x0);

for i=1:size(t)
z(i,:) = y_des(t(i))';
end
figure;plot(t,z(:,1),t,x(:,1),t,x(:,13),'--')
figure;plot(t,z(:,2),t,x(:,3),t,x(:,15),'--')
figure;plot(t,z(:,3),t,x(:,5),t,x(:,17),'--')
figure;plot(t,z(:,4),t,x(:,12),t,x(:,24),'--')

%% This section generates the reference trajectories for x,y,z and yaw
function z = y_des(t)
z = zeros(4,1);
z(1) = 5*sin(t);
z(2) = 5*cos(t);
z(3) = 5;
z(4)=pi;
end

%% This function contains the DRE equation

function dxdt = proj_dre(t,x,z,A,B,C,Q,R)
K =  reshape(x(1:144,1),[12,12]);
S = reshape(x(145:156,1),[12,1]);
dKdt = (K*A+ A'*K + C'*Q*C - K*B*inv(R)*B'*K);
dSdt = (A'*S - K*B*inv(R)*B'*S - C'*Q*z);
dKdt = reshape(dKdt,[144,1]);
dxdt = [dKdt;dSdt];

end

%% This function contains the closed-loop dynamics with the LQG observer 
function dxdt = proj_k(t,x,A,B,C1,L,T,Kaug,R)
Kint = zeros(1,156);
for j =1:size(Kaug,2)
    Kint(j) = interp1(T, Kaug(:,j),t,'spline');
end
K = reshape(Kint(1,1:144),[12,12]);
S = Kint(1,145:end)';
dxdt(1:12,1) = A*x(1:12) - B*inv(R)*B'*K*x(13:24) - B*inv(R)*B'*S + sqrt(0.001)*randn(12,1);
y = C1*x(1:12) + sqrt(0.01)*randn(12,1);
dxdt(13:24,1) = L*y + (A-B*inv(R)*B'*K-L*C1)*x(13:24) - B*inv(R)*B'*S;
end