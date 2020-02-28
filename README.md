# ECE726_project

This repository contains all the code files for the implementation of an FHLQT controller with an LQG observer and a discounted IHLQT controllers using an actor-critic reinforcement learning algorithm on a linearized plant-model of the quadcoptor system.

The FHLQT controller is implemented both ways - by augmenting the trajectory generator dynamics to the plant dynamics and solving the FHLQR problem; and also by solving the regular FHLQT problem without augmentation. The LQG observer is implemented by solving the observer ARE.

The IHLQT controller is implemented by augmenting the trajectory dynamics to the plant model and solving the discounted ARE. The IHLQT controller is also derived by implementing a actor-critic reinforcement learning algorithm with a discounted cost function as described in the paper : https://doi.org/10.1109/TAC.2014.2317301



