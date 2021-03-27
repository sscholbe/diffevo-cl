# DiffEvoCL - User-programmable Differential Evolution for OpenCL
This is a parallelized implementation of the [Differential Evolution](https://en.wikipedia.org/wiki/Differential_evolution) algorithm written in OpenCL 2.0 and C. It is implemented as a dynamic library, which allows for easy installation and exports a single function `diffevo_solve`. It is a **general purpose library** as it lets you swap out the cost function invisibly from the rest of the algorithm in a modular fashion. Meaning all you have to do is to implement your cost function, and let the library do the rest. It also offers you more control about how it is evaluated (if needed) and easily lets you transfer additional data (e.g. custom parameters) to the function.

I have successfully used Differential Evolution before in my student project [Optimizing Acoustic Properties of Microperforated Panels](https://gitlab.ethz.ch/sscholbe/optimization-of-microperforated-panels). This library should be a versatile framework for future optimization problems.

### What is it?

Differential Evolution (DE) is a non-deterministic **optimization method** that tries to minimize a given cost function by searching for candidates in a possibly higher-dimensional space. While commonly known optimization methods like the Newton Iteration are based on the gradient of the function, DE works stochastically in a **population-based manner** introducing **mutation and genetic crossover** and does not rely on a differentiable, or continuous problem. That is, it also **works well for very noisy problems**. An example for a noisy problem is the Schaffer Function N. 4 (see below), which is hilariously bumpy optimization problem and has 4 global minima hidden within many local optima. This is one of many so called "test functions for optimization", which aim to push optimizers to their limits. Purely gradient-based optimizers likely cannot find a global minimum, as they get stuck in one of the many local minima. On the other hand, DE solves these kinds of problems very well.

![](/schaffer.png?raw=true)

*Schaffer Function N. 4 - A typical optimization challenge having 4 global minima hidden within many local optima ([HQ version](/schaffer.pdf)).*

![](/schaffer_func.png?raw=true)


DiffEvoCL correctly identifies one of the four symmetric global minima `minimum: x: 0.000000 y: -1.253132 with cost 0.292579` ([code](/tester/eval_schaffer.cl)).

### How does it work?

DE introduces mutation and genetic crossover in a population-based approach. An initial population of candidates (which are just vectors that the define the inputs to the cost function) is randomly generated. In each iteration, a candidate population is generated consisting of members geometrically constructed by mutating three population members and crossing them with their respective previous member. One could consider them a “child” of four population members. The new population is then chosen by selecting the better members of both population (based on the cost function).

### Why does it work?

Many optimization methods do not have proofs, as they “just work or don’t” based on metaheuristics. They do not guarantee convergence. DE also does not always converge.

### How is it parallelized?

All phases of the algorithm have been **parallelized on the granularity of population members**. As noticeable in the procedure of the algorithm, all actions are applied on a per-member basis. This is exploited for the efficient implementation. A finer granularity would only increase overhead, as the actions per se are rather simple (and thus work per group too small), a coarser granularity would introduce unnecessary serialization.

### How do I use it?

The usage is very easy.
```c
int diffevo_solve(const char *path, const diffevo_params_t *params, double *best, double *cost);
```

* The first argument `path` is a path to an .cl file containing your implementation of the cost function (see the next question).
* `params` is of type `diffevo_params_t` and lets you configure the DE parameters like number of iterations, population size, attributes, crossover probability, etc. For every parameter there is  an example value or range given that works well for a broad class of problems.
* `best` is a pointer to a `double` array and is the location where DiffEvoCL will return the attributes of the best candidate after execution.
* `cost` will store the associated cost of the best population member.

The return code is 0 if the execution was successful, otherwise a non-zero value.

To install it, copy *diffevo.dll* and *diffevo.h* to your project directory and link *diffevo.lib*.

### How do I implement my cost function?

To give the most freedom to the user and get most performance in the `eval()` step, i.e. evaluating the cost function for all population members, it will be defined a OpenCL kernel. The signature of the kernel should be
```c
__kernel void eval(
__constant double *restrict pop,
__global double *restrict costs,
unsigned num_pop,
unsigned num_attr,
__constant double *restrict eval_data,
__local double *restrict local_data
);
```

While this might look scary, it is actually very straight forward. The `pop` buffer contains all population members (all attributes stacked). Meaning `pop[n * num_attr + 0]` stores the first attribute of the n-th population member, `pop[n * num_attr + 1]` the second etc. To get the id of the current population member your kernel is working on, you should use `get_global_id(0)`. The `costs` buffer stores the cost (single value) per population member.

**Your task** is to implement the cost function in a C-like syntax: read the current candidate from `pop`, evaluate it and write the corresponding cost into the `costs` buffer. I strongly suggest you looking at the example Schaffer implementation.

### What is the kernel argument `eval_data` used for?

In case you want to communicate some static data to your kernel (e.g. some additional application-fixed parameters to the cost function) you can set `const_data_ptr` in `eval_params` and `const_data_size`. At launch DiffEvoCL will copy this data into read-only memory onto your device and allow you to read from it in your kernel through the parameter `eval_data`. Note, that you can change the type of `eval_data`, as long as you always make sure that `const_data_size` is of proper size in bytes (also, it always has to be a pointer).

### What is the kernel argument `local_data` used for?

In case you want to parallelize your cost function even further, you can set the following `eval_params` in your `diffevo_params_t` struct.

* `local_work_size` sets the number of calls to your `eval` kernel per population member (the number of work items per work group), i.e. the kernel will be called `local_work_size` times in parallel (thus, **barriers are possible**). Note, that the upper limit by the hardware is usually 256. To get your population id, you must use `get_group_id(0)` instead of `get_global_id(0)`, and the sub-id using `get_local_id(0)`.
* `local_data_size` allows you to allocate local storage, which all of your kernels that are called on the same population member can read from and write to. You are responsible for ensuring consistency, I strongly suggest the use of barriers. An example for this being used is if your cost function is applied on a lot of data that afterwards gets summed together. Then you can use `local_data` within your kernel to communicate their "parts" and have it summed up by the kernel with local id 0 (root) and written into `costs`.

### What is missing?

A feature I want to implement in the future is allowing the user to specify constraints on the attributes themselves, e.g. a candidate `(x, y)` with `x > y` may never exist (and is thus never generated, not even by mutation or crossover).
