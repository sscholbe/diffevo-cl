#pragma once

typedef struct {
    // Fixed number of iterations the algorithm will execute.
    // e.g. 250; 100 - 10000.
    unsigned num_iter;

    // Number of members in the population. 
    // e.g. 40, 20 - 100
    unsigned num_pop;

    // Number of attributes per entity member, ideally the number of parameters of your problem.
    // e.g. 3
    unsigned num_attr;

    // Initial candidates are Normal(mu, sigma^2) distributed (i.i.d. per member and attribute).
    // e.g. 0
    double mu;

    // Initial candidates are Normal(mu, sigma^2) distributed (i.i.d. per member and attribute).
    // e.g. 1
    double sigma;

    // Geometric shrink factor of the cuboid.
    // e.g. 0.6; 0.4 - 0.9
    double shrink;

    // Probability of a mutation occuring.
    // e.g. 0.5; 0.1 - 0.9
    double crossover;

    // Allows you to further configure the eval() kernel.
    struct {
        // In case the eval() kernel needs some constant globally shared data (meaning same for all 
        // members during all iterations), here you can set it.
        // This could be e.g. additional parameters set by your application.
        // Pointer to the data that should be copied to read-only kernel address space.
        // NULL, if not needed.
        void *const_data_ptr;

        // Number of bytes of constant globally shared data, see const_data_ptr;
        // 0, if not needed.
        unsigned const_data_size;

        // In case the eval() function can be even further parallelized (up to 256), you can set
        // the number of work groups that will execute in parallel per population member. Note, that
        // you are responsible for aggregating the data properly.
        // 0, if not needed.
        unsigned local_work_size;

        // In case you chose to futher parallelize your eval() function, this number of bytes that
        // will be made available in local kernel address space, meaning this way you can share
        // data between work groups.
        // 0, if not needed.
        unsigned local_data_size;
    } eval_params;
} diffevo_params_t;

#ifdef _diffevo_export
#define _dll __declspec(dllexport)
#else
#define _dll __declspec(dllimport)
#endif

// Solves a minimization problem using the Differential Evolution (DE) algorithm. Based on the given
// parameters it will try to solve the problem in a highly parallelized OpenCL context.
// This function is not safe for multi threading.
//
// - path: Path of the file containing the eval() kernel source (i.e. your minimization problem).
// - params: Pointer to the parameters of the algorithm.
// - best: Pointer to where the best candidate (i.e. all its attributes) will be written to.
// - cost: Pointer to where the cost of the best candidate will be written to.
//
// Returns 0, if the execution was successful, otherwise a non-zero value.
_dll int diffevo_solve(const char *path, const diffevo_params_t *params, double *best, double *cost);
