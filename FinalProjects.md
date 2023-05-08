- Instructor **Chen Ding**
- TA **Owen Wacha**

# Public Release of Final Projects

Students had from Monday, March 27 to Friday April 28 to propose, discuss, and complete a final project on a topic of their choosing.

## Yifan Zhu, Parallel Neural Network Framework in Rust

The project implements a composable and customizable neural network framework in Rust. Utilizing Rayon, ArrayFire and MPI, 
the framework upports data parallelism, model parallelism and it can be scaled to multi-node clusters.
It covers fully-connected neural network layers, including several variants of activition functions, loss functions and gradient optimizers. 
Besides traditional backward propagation, the framework supports likelihood ratio method, which provides a new embarrasing parallel training 
paradigm for ANNs. In the evaluation part, this project provides performance and accuracy data under general training situations with different 
settings covering both traditional and new methods implemented in our framework.

Sides: 
