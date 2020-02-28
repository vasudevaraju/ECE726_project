%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%authors: Sreechakra Vasudeva Raju Rachavelpula, Mohammad Sareeb Hakak

%This code generates the control for the linearized plant model of the 
%quadcoptor using an actor-critic based reinforcement learning algorithm
%with a disounted reward. The plant dynamics and the trajectory generator
% dynamics are assumed to be unknown beforehand. For more information about
%this RL algorithm used for generating the optimal control,
%refer to the following paper: https://doi.org/10.1109/TAC.2014.2317301

%Our implementation of the algorithm seemed to be very sensitive to various
%factors like convergence criteria, discount value, number of samples etc.
%However, MATLAB's implementation of a similar RL algorithm is very robust.

%This code only generates the control for the y-coordinate of the
%quadcoptor center. The y-coordinate is made to track the trajectory 
%yref = cos(t) for simplicity. The linearized model of the plant allows for 
%decoupling of the x, y, z and yaw dynamics. Hence the control for x-coord,
%z-coord and yaw of the quadcoptor can be designed similarly

%The states related to the dynamics in the x-direction are so:
%[y,y.,phi,p] where 
%y is the y-coordinate
%y. is the velocity in the y-direction
%phi is the rotation about the x-axis
%p is the angular velocity about the x-axis

%The control input is the torque about x-axis 'u2'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;clc;
m = 1.4; Ix = 0.03;g=9.81;

%The systems dynamics with the augmented trajectory dynamics are defined
%but are assumed to be unavailable to the RL algorithm. These dynamics are
%assumed to be available only through sensor measurements in real-life.
A = [0,1,0,0;...
    0,0,-g,0;...
    0,0,0,1;...
    0,0,0,0];
Aaug = [A,zeros(4,2);zeros(2,4),[0,1;-1,0]];
B = [0;0;0;1/Ix];
Baug = [B;zeros(2,1)];
C = [1,0,0,0];
Caug = [C, -1,0];
D = 0;

i = 0; %iteration number

Q= 1*eye(1);%costs are defined for calculating the reward
R= 1; %These are to be picked by the designer before running the RL algo
Qaug = Caug'*Q*Caug;
S = place(A,B,-[2,5,3,4]); %Initial guess of the control gain S
%The closed-loop with this control gain should be stable
K = pinv(inv(R)*B')*S; %Solve for solution of the ARE (K) from control gain S
K(5,5) = 0;K(6,6) = 0;
%Augment a 5th, 6th row and column to which corresponds to the tajectory dynamics

sample_num = 60; %no of samples to be collected in one iteration
sample_time = 0.05; %sampling time
t0 = 0;%initial iteration time
x0 = [1,0,0,0,1,0];
%initial conditions of the quacoptor x-dynamics with x-ref 
V0 = 0;
x0 = [x0,V0];%augment cost to the dynamics
eps = 1e-4; %small value to check convergence
S = inv(R)*Baug'*K; %get the initial control gain S
gam = 0.01; %discount factor

while 1
    i = i+1; %increment iteration number
    
    X(1,:) = x0; % X collects state data after each sampling time like sensor 
    reward = zeros(size(X,1)-1,1);
    T = t0;
    
    %obtain samples within the iteration and calculate the reward
    for j = 1:sample_num
    [t,X_dum] = ode45(@(t,X)get_state_cost(t,X,S,Qaug,R),[t0,t0+sample_time],x0);
    reward(j,:) = X_dum(end,end)-X_dum(1,end); %rewards
    t0 = t0+sample_time;
    x0 = X_dum(length(t),:);
    T(j+1) = t(end);
    X(j+1,:) = X_dum(length(t),:);
    
    plot(t,X_dum(:,1),'b','LineWidth',2);hold on
    plot(t,X_dum(:,2),'g','LineWidth',2);
    plot(t,X_dum(:,3),'m');
    plot(t,X_dum(:,4),'k');
    plot(t,X_dum(:,5),'r','LineWidth',2);
    
    end
    
    x = X(:,1:6);
    plot(t,x(end,1),'k*');hold on
    plot(t,x(end,2),'k*');
    plot(t,x(end,3),'k*');
    plot(t,x(end,4),'k*');
    plot(t,x(end,5),'k*');
    
    dummy_act = zeros(size(X,1),21); 
    activation = zeros(size(X,1)-1,21); %activations

    % This for loop takes states x at each sample time t in an iteration 
    % and reshapes the elements in x*transpose(x) into a row vector.
    % It does this for all samples in the iteration step
    
    for j = 1:size(X,1)
        x_shaped = x(j,:)'*x(j,:);
        current_x = [];
        for k = 1:size(x_shaped,1)
            for l =k:size(x_shaped,2)
                if l ~= k
                    current_x = [current_x,(x_shaped(k,l) + x_shaped(l,k))/2];
                else
                     current_x = [current_x,x_shaped(k,l)];
                end
            end
        end
        dummy_act(j,:) = current_x;
    end
    for j =1:size(X,1)-1
        activation(j,:) = dummy_act(j,:) - exp(-gam*sample_time)*dummy_act(j+1,:);
        %calculate the activations by taking appropriate differences
    end
    K_dummy = pinv(activation)*reward;
    
    %This for loop constructs the symmetric K matrix which is solution of
    %ARE from the K_dummy row vector working in the backward direction 
   
    m=0;
    for k = size(x_shaped,1):-1:1
        for l =size(x_shaped,2):-1:k
            if k == l
                K_new(k,l) = K_dummy(end-m);
                m = m+1;
            else
                K_new(k,l) = K_dummy(end-m)/2;
                K_new(l,k) = K_dummy(end-m)/2;
                m = m+1;
            end
        end
    end
        if abs(reward(end))<=eps %check convergence
        break;
    end    
    S = inv(R)*Baug'*K_new;% update control gain
    K = K_new; %update the obtained K matrix
end


legend('x1','x2','x3','x4','ref');
title('Reinforcement learning state evolution');xlabel('t');ylabel('x');
[t,x]=ode45(@(t,x)(Aaug-Baug*inv(R)*Baug'*K)*x,[0,10],[5,0,0,0,1,0]);
figure;plot(t,x(:,5),'b','LineWidth',3);
hold on;plot(t,x(:,1),'r','LineWidth',2);
title('Learned Control Law Performace'); xlabel('t'); ylabel('x');



%% This ode file is like a sensor which measures the states. 
% It also returns the cost V at each time step in the same state vector
function dXdt = get_state_cost(t, X,S,Q,R)
m = 1.4; Ix = 0.03; g=9.81;
A = [0,1,0,0;0,0,-g,0;0,0,0,1;0,0,0,0];
Aaug = [A,zeros(4,2);zeros(2,4),[0,1;-1,0]];
B = [0;0;0;1/Ix];
Baug = [B;zeros(2,1)];
C = [1,0,0,0];
Caug = [C, -1,0];
D = 0;
i = 0;
Q= 1*eye(1);
R= 1;
Qaug = Caug'*Q*Caug;

x = X(1:6,1);
V = X(7);
u = -S*x  ;
dxdt = Aaug*x + Baug*u;
dVdt = x'*Qaug*x + u'*R*u;
dXdt = [dxdt;dVdt]; 
end