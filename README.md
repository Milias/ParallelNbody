# ParallelNbody
Solving the N-body problem using CUDA.

This is a project for Parallel Algorithms, a course at Utrecht Univerisity in the Computer Science
specialization of Mathematics Master's Program. The objective is to write a parallel program that
calculates individual trajectories of a number of bodies, taking into account gravitational
interactions.

Since the straightforward approach of direct sums over all the bodies in the system has a compu-
tational cost that grows as O(N^2), we are going to implement Barnes-Hut algorithm, which grows
as O(N*log(N)). This is specially important, since we want to simulate trajectories for a number
of bodies > 1000.

Finally, the program is going to be written in C++ using CUDA, so its main workload will be
done in a Nvidia GPU. As for plotting the final results, we are going to use Unreal Engine 4,
a 3D render framework written in C++.

# Summary
- N-Body simulation program, for N > 1000.
- Using CUDA for computing.
- Barnes-Hut algorithm for less complexity.
- 3D representation of final results using Unreal Engine 4.
