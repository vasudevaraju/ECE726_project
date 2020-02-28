%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%authors: Sreechakra Vasudeva Raju Rachavelpula, Mohammad Sareeb Hakak

%This code generates a control for the linearized plant model of the 
%quadcoptor for trajectory tracking by solving the Differential Ricatti 
%Equation (DRE). This is done by augmenting the trajectory dynamics to
%the plant dynamics.

%The states of the system are so :
%[x,x.,y,y.,z,z.,p,q,r,ph,th,ps] where
%x,y,z are the x,y,z position coordinates of center of quadcoptor in IRF
%x.,y.,z. are the velocity components in the x,y,z directions in IRF
%p, q, r are the angular velocities about x,y,z axes in body frame
%ph ,th, ps are the euler angles defining orientation of body frame wrt IRF

%Here the system is made to track in 10secs:
%xref = sin(t)
%yref = cos(t)
%zref = 5
%psi = pi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%T matrix contains the trajectory dynamics
T = [0,1,0,0,0,0;...
    -1,0,0,0,0,0;...
    0,0,0,1,0,0;...
    0,0,-1,0,0,0;...
    0,0,0,0,0,0;...
    0,0,0,0,0,0];
Aaug = [A,zeros(12,6);zeros(6,12),T];
Baug = [B;zeros(6,4)];
E = [1,zeros(1,5);0,0,1,0,0,0;0,0,0,0,1,0;0,0,0,0,0,1];
Caug = [-C, E];
Faug = Caug'*F*Caug;
Qaug = Caug'*Q*Caug;

%initial conditions are the terminal conditions 
Kf = reshape(Faug,[18*18,1]);
%K is the solution of the DRE
tspan = [0:0.01:10];
[T,Kaug] = ode45(@(t,x) proj_augdre(t,x,Aaug,Baug,Qaug,R), tspan, Kf);
%flip the time and gain matrices because the DRE is solved in backward time
t = 10-flip(T); 
Kaug = flip(Kaug);
%Kaug is a time-series matrix containing row vectors which are elements of
%solution K of the DRe

%% This section simulates the FHLQT controller
x0 = zeros(1,12);
z0 = [0,1,0,1,5,pi];
[t,x] = ode45(@(T,X)proj_augsys(T,X,Aaug,Baug,t,Kaug,R),t',[x0,z0]);
figure;plot(t,x(:,13),t,x(:,1)); %state 13 is xref
figure; plot(t,x(:,15),t,x(:,3));%state 15 is yref
figure; plot(t,x(:,17),t,x(:,5));%state 17 is zref
figure; plot(t,x(:,18),t,x(:,12));%state 18 is psref

%% This function contains the augmented DRE in backward time
function dxdt = proj_augdre(t,x,A,B,Q,R)
K =  reshape(x,[18,18]);
dKdt = (K*A+ A'*K + Q - K*B*inv(R)*B'*K);
dxdt = reshape(dKdt,[18*18,1]);
end

%% This function contains the closed-loop equations
function dxdt = proj_augsys(t,x,A,B,T,Kaug,R)
Kint = zeros(1,18*18);
for j =1:size(Kaug,2)
    Kint(j) = interp1(T, Kaug(:,j),t,'spline');
end
K = reshape(Kint(1,1:18*18),[18,18]);
dxdt = (A - B*inv(R)*B'*K)*x;
end