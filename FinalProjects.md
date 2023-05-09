- Instructor **Chen Ding**
- TA **Owen Wacha**

# Public Release of Final Projects

Students had from Monday, March 27 to Friday April 28 to propose, discuss, and complete a final project on a topic of their choosing.

- CPU parallelizatin in Rust (6 projects)
- GPU parallelization (3)
- Distributed parallelization (2)

## CPU Parallelizaiton Using Rust

### Yifan Zhu, Parallel Neural Network Framework in Rust

The project implements a composable and customizable neural network framework in Rust. Utilizing Rayon, ArrayFire and MPI, 
the framework upports data parallelism, model parallelism and it can be scaled to multi-node clusters.
It covers fully-connected neural network layers, including several variants of activition functions, loss functions and gradient optimizers. 
Besides traditional backward propagation, the framework supports likelihood ratio method, which provides a new embarrasing parallel training 
paradigm for ANNs. In the evaluation part, this project provides performance and accuracy data under general training situations with different 
settings covering both traditional and new methods implemented in our framework.

Sides: https://github.com/dcompiler/258s23/blob/main/slides/Yifan%20Zhu.key

### Will Nguyen and Woody Wu, Enhancing N-Body Simulation Performance through Parallelism and Rust-based Quadtree Data Structure

This project aims to enhance the performance of an N-body simulation by incorporating parallelism and a quadtree data structure designed in Rust. N-body simulations play a critical role in astrophysics, as they model the movements of celestial bodies; however, these simulations can be computationally intensive. This investigation focuses on utilizing OpenMP and Rayon for parallel processing and implementing a quadtree data structure for efficient spatial partitioning. The project employs a benchmarking script designed to assess the efficiency of a variety of parallelism techniques, such as Pthreads, OpenMP, and Rayon. Data is gathered through a range of sequential and parallel test cases executed iteratively across multiple celestial body scales. Rust’s implementation offers type safety, memory safety, and concurrency, contributing to effective and dependable performance. This project showcases the potential of parallelism and the Rust-based quadtree data structure in improving the computational efficacy of N-body simulations. Code is available here: https://github.com/YanghuiWu/nbody_rust_simu.

### Yiming (Pete) Leng, Parallel A*

A* algorithm is a search algorithm for finding optimal paths in graphs that's widely used in networking, gaming, robotics, and most recently in AI. The motivation for parallel A* arises as problems size continue to grow. In addition to the benefit of speedup, parallelization of A* enables solving problems otherwise unsolvable by the sequential counterpart (due to its memory limitation) by potentially utilizing the aggregate memory of several computing clusters. This project focuses on studying the speedup from parallelizing A*.  It surveys state of the art methods in parallel A*, which are implemented and benchmarked in Rust.

### Ziqi Feng, Parallel SAT solvers

The Davis-Putnam-Logemann-Loveland (DPLL) algorithm is a complete, backtracking-based search algorithm for solving the satisfiability of propositional logic formulas. The project develops and evaluates parallel DPLL.  Both the sequential and parallel versions of the algorithm are implemented in Rust.

### Shaotong Sun and Jionghao Han, PLUSS Sampler Optimization

Programs and algorithms must be carefully designed to minimize data movement, which raises the importance of data locality analysis.  Parallel Locality Analysis using Static Sampling (PLUSS) is an on-going project led by Fangzhou Liu.  The analysis happens at compile time and is implemented with LLVM. This project translated and reimplemented PLUSS’s solution for the matrix multiplication test in Rust. We programmed in Rust to analyze the shared cache performance and parallelized the analysis program.  Our implementation focuses on ensuring thread safety, preventing undefined behavior, and improving code quality. 

### Jiakun Fan, MVCC Software Transactional Memory in Rust

Multiversion concurrency control (MCC or MVCC) is a widely used concurrency control method in database management systems.  This project uses MVCC to improve the implementation of software transactional memory (STM).  It introduces a new concept of ”space” and applies MVCC on spaces to enhance performance of STM.

## GPU Parallelization

### Aayush Poudel and Matt Nappo, Parallelizing the Traveling Salesman Problem Using Ant Colony Optimization on a GPU

This project develops a GPU implementation of the Traveling Salesman Problem (TSP) using CUDA. We compare the performance of our GPU implementation against a CPU implementation and an OpenMP implementation on the CPU. We compare the running times of both implementations, as we’ve verified the correctness of the algorithms. Our experiments show that the GPU implementation is significantly faster than the CPU and OpenMP implementations. We also validate the optimality of our solution against the linear programming based Concorde TSP solver. Overall, our results strongly show that a GPU implementation of the TSP can provide significant speedup over CPU implementations, making it a promising approach for solving large-scale TSP instances.

### Zeliang Zhang, Highly Parallel Gradient Estimation using GPUs

The gradient-based algorithm has achieved great success in solving convex optimization problems, such as linear regression, perceptron, and deep neural networks. Conventional methods use the chain-rule principle to backpropagate the exact gradient for optimization, while it is usually tricky to process complex and deep systems due to their data dependence. On the other hand, the like- lihood ratio-based method provides a novel view for gradient es- timation, which is more computation-friendly compared with the conventional chain-rule-based back-propagation method. It can also be applied to non-differential optimization problems which the conventional gradient-based methods can not handle. In our work, we parallelize the likelihood ratio method for gradient estimation. We achieve high-performance gradient estimation to support neural network training. It consists of two strategies, namely the layer-level parallel strategy and the model-level parallel strategy. We evaluate the proposed method by training neural networks on MNIST and Ag-News datasets with the comparison between the classification accuracy and training time.

### Luchuan Song and Zeliang Zhang, Highly Parallel Tensor Computation for Classical Simulation of Quantum Circuits using GPUs

We develop a reinforcement learning method for optimise the order of tensor contraction in the tensor network and parallelize the reinforcement learning training.

## Distributed Parallelization

### Donovan Zhong and Muhammad Qasim, A Raft-based key-value storage implementation

When it comes to distributed systems consistency among a network of distinct
servers is a key problem. That is, the problem of making sure that all servers
or devices agree on the data that the server has on storage and all servers treat
new data in the same manner. To this end, PAXOS provided a great solution
and became the industry standard. However, some researchers have been worried
about the degree of complexity of PAXOS as the number of different protocols
and types of server configurations possible make it hard to debug PAXOS on most
large-scale system. To address this issue, Diego Angaro and John Ousterhout, in
their paper titled ’In search of an understandable consensus algorithm’, introduced
RAFT a consensus algorithm meant to serve as an alternative to PAXOS. RAFT
provides a great replacement since it is functionally equivalent to multi-PAXOS
and as efficient as PAXOS while also providing a very simply server communication
protocol. In this research, we explored RAFT and asked how suitable RAFT would
be as a replacement for PAXOS. More specifically, we want to test Angaro et al.’s 
claim that RAFT is more suitable for most practical server applications. To
do this, we implement RAFT is java using the gRPC communication protocol and
test how efficient it is, how it scales and how it performs in comparison to PAXOS.

### Suumil Roy, Parallel Video Compression using MPI and OpenMP

The project develops parallel video compression using MPI for interframe compression 
and OpenMP for intra-frame compression.


