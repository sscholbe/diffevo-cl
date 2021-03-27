#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>
#include <direct.h>
#include <string.h>

#define _diffevo_export

#include "diffevo.h"

int last_error;

void report_error(char *msg) {
    last_error = -1;
    fprintf(stderr, "%s\n", msg);
}

void report_error_code(char *msg, int code) {
    last_error = -1;
    fprintf(stderr, "%s [code: %d]\n", msg, code);
}

#define _s(x) #x
const char *algo_src =
#include "diffevo.cl"
;

cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;

struct {
    cl_mem rng, seeds, pop[3], costs[3];
    cl_mem eval_data;
} buffers;

struct {
    cl_kernel init, eval, mutate, select;
} kernels;

#define _if_err_ret(msg) if (CL_SUCCESS != err) { report_error_code(msg, err); return -1; }

int init_cl(void) {
    cl_int err;

    // TODO: This code just picks the first device it can find. Maybe we want to check for
    // a suitable GPU/CPU first? Especially check that CL version >= 2.0...

    cl_uint num_plat;
    err = clGetPlatformIDs(0, NULL, &num_plat);
    _if_err_ret("clGetPlatformIDs() failed");
    if (0 == num_plat) {
        report_error("No platforms available");
        return -1;
    }

    cl_platform_id plat;
    err = clGetPlatformIDs(1, &plat, NULL);
    _if_err_ret("clGetPlatformIDs() failed");

    cl_uint num_dev;
    err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &device, &num_dev);
    _if_err_ret("clGetDeviceIDs() failed");
    if (0 == num_dev) {
        report_error("No devices available");
        return -1;
    }

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    _if_err_ret("clCreateContext() failed");
    queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
    _if_err_ret("clCreateCommandQueueWithProperties() failed");

    return 0;
}

int destroy_cl(void) {
    cl_int err;

    if (NULL != buffers.rng) {
        err = clReleaseMemObject(buffers.rng);
        _if_err_ret("clReleaseMemObject() failed");
    }
    if (NULL != buffers.seeds) {
        err = clReleaseMemObject(buffers.seeds);
        _if_err_ret("clReleaseMemObject() failed");
    }

    for (unsigned i = 0; i < 3; i++) {
        if (NULL != buffers.pop[i]) {
            err = clReleaseMemObject(buffers.pop[i]);
            _if_err_ret("clReleaseMemObject() failed");
        }
        if (NULL != buffers.costs[i]) {
            err = clReleaseMemObject(buffers.costs[i]);
            _if_err_ret("clReleaseMemObject() failed");
        }
    }
    if (NULL != buffers.eval_data) {
        err = clReleaseMemObject(buffers.eval_data);
        _if_err_ret("clReleaseMemObject() failed");
    }

    if (NULL != kernels.init) {
        err = clReleaseKernel(kernels.init);
        _if_err_ret("clReleaseKernel() failed");
    }
    if (NULL != kernels.eval) {
        err = clReleaseKernel(kernels.eval);
        _if_err_ret("clReleaseKernel() failed");
    }
    if (NULL != kernels.mutate) {
        err = clReleaseKernel(kernels.mutate);
        _if_err_ret("clReleaseKernel() failed");
    }
    if (NULL != kernels.select) {
        err = clReleaseKernel(kernels.select);
        _if_err_ret("clReleaseKernel() failed");
    }

    if (NULL != program) {
        err = clReleaseProgram(program);
        _if_err_ret("clReleaseProgram() failed");
    }
    if (NULL != queue) {
        err = clReleaseCommandQueue(queue);
        _if_err_ret("clReleaseCommandQueue() failed");
    }
    if (NULL != device) {
        err = clReleaseDevice(device);
        _if_err_ret("clReleaseDevice() failed");
    }
    if (NULL != context) {
        err = clReleaseContext(context);
        _if_err_ret("clReleaseContext() failed");
    }

    return 0;
}

int read_file(const char *path, char **contents, size_t *length) {
    FILE *fp = fopen(path, "rb");
    if (NULL == fp) {
        return -1;
    }

    if (0 != fseek(fp, 0, SEEK_END)) {
        fclose(fp);
        return -1;
    }

    unsigned long len = ftell(fp);
    if (-1 == len) {
        fclose(fp);
        return -1;
    }
    rewind(fp);

    char *str = malloc(len + 1);
    if (NULL == str) {
        report_error("Out of memory");
        fclose(fp);
        return -1;
    }
    str[len] = '\0';

    if (len != fread(str, 1, len, fp)) {
        free(str);
        str = NULL;
        fclose(fp);
        return -1;
    }

    fclose(fp);

    *contents = str;
    *length = len;

    return 0;
}

int create_program(const char *eval_path) {
    char *eval_src;
    size_t eval_src_len;

    if (0 != read_file(eval_path, &eval_src, &eval_src_len)) {
        report_error("Failed to open eval() source file");
        return -1;
    }

    cl_int err;

    //
    // Compile both sources (DE algorithm and user-defined eval()) into one program.
    //

    const char *srcs[2];
    size_t lens[2];

    srcs[0] = eval_src;
    lens[0] = eval_src_len;
    srcs[1] = algo_src;
    lens[1] = strlen(algo_src);

    program = clCreateProgramWithSource(context, 2, srcs, lens, &err);
    free(eval_src);
    eval_src = NULL;
    _if_err_ret("clCreateProgramWithSource() failed");

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (CL_SUCCESS != err) {
        if (CL_BUILD_PROGRAM_FAILURE != err) {
            report_error("clBuildProgram() failed without a build failure");
            return -1;
        }

        //
        // In case of a program build error, read and print the build log to the user.
        //

        size_t log_len;
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_len);
        _if_err_ret("clGetProgramBuildInfo() failed");
        char *log = malloc(log_len);
        if (NULL == log) {
            report_error("Out of memory");
            return -1;
        }

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_len, log, NULL);
        _if_err_ret("clGetProgramBuildInfo() failed");
        if (CL_SUCCESS != err) {
            free(log);
            log = NULL;
            return -1;
        }

        report_error(log);
        free(log);
        log = NULL;

        return -1;
    }

    return 0;
}

#define _if_err_die(msg) if(CL_SUCCESS != err) { report_error_code(msg, err); goto __CleanUp; }

int diffevo_solve(const char *path, const diffevo_params_t *params, double *best, double *cost) {
    int err;

    if (NULL == path) {
        report_error("eval() path not specified");
        return -1;
    }

    err = init_cl();
    if (0 != err) {
        report_error("Error while creating OpenCL context");
        goto __CleanUp;
    }

    err = create_program(path);
    if (0 != err) {
        report_error("Error while loading program");
        goto __CleanUp;
    }

    //
    // Use one RNG per population member. A finer granularity does not make sense, because the
    // overhead would be too high (since mutation and crossover is rather quick), and a coarser
    // granularity would drastically decrease the parallelism (would not be per member anymore).
    //

    buffers.rng = clCreateBuffer(context, CL_MEM_READ_WRITE, params->num_pop * 0x10, NULL, &err);
    _if_err_die("clCreateBuffer() failed");

    //
    // Generate the seeds that will be used in the init() kernel to initialize the RNGs.
    // 

    srand((unsigned) time(NULL));

    unsigned *seeds = malloc(params->num_pop * sizeof(unsigned));
    if (NULL == seeds) {
        report_error("Out of memory");
        goto __CleanUp;
    }

    for (unsigned i = 0; i < params->num_pop; i++) {
        seeds[i] = rand();
    }

    buffers.seeds = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        params->num_pop * sizeof(unsigned), seeds, &err);
    free(seeds);
    seeds = NULL;
    _if_err_die("clCreateBuffer() failed");

    //
    // Use three population and cost buffers, so that we can select() from two into one separate.
    // Altough increasing the memory consumption, it allows to make the input buffers cacheable.
    //

    for (unsigned i = 0; i < 3; i++) {
        buffers.pop[i] = clCreateBuffer(context, CL_MEM_READ_WRITE,
            params->num_pop * params->num_attr * sizeof(double), NULL, &err);
        _if_err_die("clCreateBuffer() failed");
        buffers.costs[i] = clCreateBuffer(context, CL_MEM_READ_WRITE,
            params->num_pop * sizeof(double), NULL, &err);
        _if_err_die("clCreateBuffer() failed");
    }

    //
    // If the user wishes, we copy their data into a read-only buffer, so it can be used
    // during eval(). This could be for example some dynamic parameters.
    //

    if (NULL != params->eval_params.const_data_ptr) {
        buffers.eval_data = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            params->eval_params.const_data_size, params->eval_params.const_data_ptr, &err);
        _if_err_die("clCreateBuffer() failed");
    }

    kernels.init = clCreateKernel(program, "init", &err);
    _if_err_die("Failed to create init() kernel");
    kernels.eval = clCreateKernel(program, "eval", &err);
    _if_err_die("Failed to create eval() kernel");
    kernels.mutate = clCreateKernel(program, "mutate", &err);
    _if_err_die("Failed to create mutate() kernel");
    kernels.select = clCreateKernel(program, "select", &err);
    _if_err_die("Failed to create select() kernel");

    const size_t pop_dim = params->num_pop;

    //
    // Initialize the RNGs and population.
    //

    err = clSetKernelArg(kernels.init, 0, sizeof(cl_mem), &buffers.rng);
    _if_err_die("init!clSetKernelArg(0) failed");
    err = clSetKernelArg(kernels.init, 1, sizeof(cl_mem), &buffers.seeds);
    _if_err_die("init!clSetKernelArg(1) failed");
    err = clSetKernelArg(kernels.init, 2, sizeof(cl_mem), &buffers.pop[0]);
    _if_err_die("init!clSetKernelArg(2) failed");
    err = clSetKernelArg(kernels.init, 3, sizeof(cl_uint), &params->num_pop);
    _if_err_die("init!clSetKernelArg(3) failed");
    err = clSetKernelArg(kernels.init, 4, sizeof(cl_uint), &params->num_attr);
    _if_err_die("init!clSetKernelArg(4) failed");
    err = clSetKernelArg(kernels.init, 5, sizeof(cl_double), &params->mu);
    _if_err_die("init!clSetKernelArg(5) failed");
    err = clSetKernelArg(kernels.init, 6, sizeof(cl_double), &params->sigma);
    _if_err_die("init!clSetKernelArg(6) failed");

    cl_event init_evt;
    err = clEnqueueNDRangeKernel(queue, kernels.init, 1, NULL, &pop_dim, NULL,
        0, NULL, &init_evt);
    _if_err_die("init!clEnqueueNDRangeKernel() failed");

    //
    // Evaluate the initial population.
    //

    err = clSetKernelArg(kernels.eval, 0, sizeof(cl_mem), &buffers.pop[0]);
    _if_err_die("eval!clSetKernelArg(0) failed");
    err = clSetKernelArg(kernels.eval, 1, sizeof(cl_mem), &buffers.costs[0]);
    _if_err_die("eval!clSetKernelArg(1) failed");
    err = clSetKernelArg(kernels.eval, 2, sizeof(cl_uint), &params->num_pop);
    _if_err_die("eval!clSetKernelArg(2) failed");
    err = clSetKernelArg(kernels.eval, 3, sizeof(cl_uint), &params->num_attr);
    _if_err_die("eval!clSetKernelArg(3) failed");
    err = clSetKernelArg(kernels.eval, 4, sizeof(cl_mem), &buffers.eval_data);
    _if_err_die("eval!clSetKernelArg(4) failed");
    err = clSetKernelArg(kernels.eval, 5, params->eval_params.local_data_size, NULL);
    _if_err_die("eval!clSetKernelArg(5) failed");

    size_t eval_glb_work, eval_loc_work;
    size_t *eval_loc_work_ptr;

    if (0 < params->eval_params.local_work_size) {
        // The user explicitly set the number of work groups that are used per population member.
        // The global work is picked so that for every population member we have as much eval()
        // calls (in parallel) as the user chose.
        // TODO: Notify the user if the number of work group exceeds hardware limitations.
        eval_loc_work = params->eval_params.local_work_size;
        eval_loc_work_ptr = &eval_loc_work;
        eval_glb_work = eval_loc_work * pop_dim;
    } else {
        // Passing NULL as local_work_size tells the compiler find the ideal number of work groups.
        // Note, that the global work is reduced, because eval() is called only once per member.
        eval_loc_work_ptr = NULL;
        eval_glb_work = pop_dim;
    }

    cl_event eval_evt;
    err = clEnqueueNDRangeKernel(queue, kernels.eval, 1, NULL, &eval_glb_work,
        eval_loc_work_ptr, 1, &init_evt, &eval_evt);
    _if_err_die("eval!clEnqueueNDRangeKernel() failed");

    cl_event select_evt = NULL;

    for (unsigned i = 0; i < params->num_iter; i++) {
        // Since we use three buffers for population and costs each for memory efficiency reasons
        // but actually only deal with two populations per iteration (current and mutated), we will
        // always swap two of them.
        const unsigned p_cand = (i % 2 == 0) ? 0 : 2, p_res = 2 - p_cand;

        //
        // Mutate the population.
        //

        err = clSetKernelArg(kernels.mutate, 0, sizeof(cl_mem), &buffers.rng);
        _if_err_die("mutate!clSetKernelArg(0) failed");
        err = clSetKernelArg(kernels.mutate, 1, sizeof(cl_mem), &buffers.pop[p_cand]);
        _if_err_die("mutate!clSetKernelArg(1) failed");
        err = clSetKernelArg(kernels.mutate, 2, sizeof(cl_mem), &buffers.pop[1]);
        _if_err_die("mutate!clSetKernelArg(2) failed");
        err = clSetKernelArg(kernels.mutate, 3, sizeof(cl_uint), &params->num_pop);
        _if_err_die("mutate!clSetKernelArg(3) failed");
        err = clSetKernelArg(kernels.mutate, 4, sizeof(cl_uint), &params->num_attr);
        _if_err_die("mutate!clSetKernelArg(4) failed");
        err = clSetKernelArg(kernels.mutate, 5, sizeof(cl_double), &params->shrink);
        _if_err_die("mutate!clSetKernelArg(5) failed");
        err = clSetKernelArg(kernels.mutate, 6, sizeof(cl_double), &params->crossover);
        _if_err_die("mutate!clSetKernelArg(6) failed");

        cl_event mutate_evt;
        err = clEnqueueNDRangeKernel(queue, kernels.mutate, 1, NULL, &pop_dim, NULL, 1,
            (i == 0) ? &init_evt : &select_evt, &mutate_evt);
        _if_err_die("mutate!clEnqueueNDRangeKernel() failed");

        //
        // Evaluate the mutated population.
        //

        err = clSetKernelArg(kernels.eval, 0, sizeof(cl_mem), &buffers.pop[1]);
        _if_err_die("eval!clSetKernelArg(0) failed");
        err = clSetKernelArg(kernels.eval, 1, sizeof(cl_mem), &buffers.costs[1]);
        _if_err_die("eval!clSetKernelArg(1) failed");
        err = clSetKernelArg(kernels.eval, 2, sizeof(cl_uint), &params->num_pop);
        _if_err_die("eval!clSetKernelArg(2) failed");
        err = clSetKernelArg(kernels.eval, 3, sizeof(cl_uint), &params->num_attr);
        _if_err_die("eval!clSetKernelArg(3) failed");
        err = clSetKernelArg(kernels.eval, 4, sizeof(cl_mem), &buffers.eval_data);
        _if_err_die("eval!clSetKernelArg(4) failed");
        err = clSetKernelArg(kernels.eval, 5, params->eval_params.local_data_size, NULL);
        _if_err_die("eval!clSetKernelArg(5) failed");

        cl_event eval_mut_evt;
        err = clEnqueueNDRangeKernel(queue, kernels.eval, 1, NULL, &eval_glb_work,
            eval_loc_work_ptr, 1, &mutate_evt, &eval_mut_evt);
        _if_err_die("eval!clEnqueueNDRangeKernel() failed");

        //
        // Select the better members out of both populations.
        //

        err = clSetKernelArg(kernels.select, 0, sizeof(cl_mem), &buffers.pop[p_cand]);
        _if_err_die("select!clSetKernelArg(0) failed");
        err = clSetKernelArg(kernels.select, 1, sizeof(cl_mem), &buffers.costs[p_cand]);
        _if_err_die("select!clSetKernelArg(1) failed");
        err = clSetKernelArg(kernels.select, 2, sizeof(cl_mem), &buffers.pop[1]);
        _if_err_die("select!clSetKernelArg(2) failed");
        err = clSetKernelArg(kernels.select, 3, sizeof(cl_mem), &buffers.costs[1]);
        _if_err_die("select!clSetKernelArg(3) failed");
        err = clSetKernelArg(kernels.select, 4, sizeof(cl_mem), &buffers.pop[p_res]);
        _if_err_die("select!clSetKernelArg(4) failed");
        err = clSetKernelArg(kernels.select, 5, sizeof(cl_mem), &buffers.costs[p_res]);
        _if_err_die("select!clSetKernelArg(5) failed");
        err = clSetKernelArg(kernels.select, 6, sizeof(cl_uint), &params->num_pop);
        _if_err_die("select!clSetKernelArg(6) failed");
        err = clSetKernelArg(kernels.select, 7, sizeof(cl_uint), &params->num_attr);
        _if_err_die("select!clSetKernelArg(7) failed");

        cl_event wait_for_eval[2];
        wait_for_eval[0] = eval_evt;
        wait_for_eval[1] = eval_mut_evt;

        err = clEnqueueNDRangeKernel(queue, kernels.select, 1, NULL, &pop_dim, NULL, 2,
            wait_for_eval, &select_evt);
        _if_err_die("select!clEnqueueNDRangeKernel() failed");
    }

    clFlush(queue);
    clFinish(queue);

    //
    // Determine the index of the best population member (i.e. the one with least cost).
    //

    // As with p_cand and p_res, depending on the number of iterations the last iterations output
    // buffer could be swapped.
    const unsigned p_fin = (params->num_iter % 2 == 0) ? 0 : 2;

    double *costs = malloc(params->num_pop * sizeof(double));
    if (NULL == costs) {
        report_error("Out of memory");
        goto __CleanUp;
    }

    err = clEnqueueReadBuffer(queue, buffers.costs[p_fin], CL_TRUE, 0,
        params->num_pop * sizeof(double), costs, 0, NULL, NULL);

    if (CL_SUCCESS != err) {
        free(costs);
        costs = NULL;

        report_error("clEnqueueReadBuffer() failed");
        goto __CleanUp;
    }

    double best_c = costs[0];
    unsigned best_i = 0;

    for (unsigned i = 1; i < params->num_pop; i++) {
        if (costs[i] < best_c) {
            best_c = costs[i];
            best_i = i;
        }
    }

    free(costs);
    costs = NULL;

    //
    // Read the attributes of the best member back from the buffer into the application memory.
    //

    err = clEnqueueReadBuffer(queue, buffers.pop[p_fin], CL_TRUE, 0,
        params->num_attr * sizeof(double), best, 0, NULL, NULL);
    _if_err_die("clEnqueueReadBuffer() failed");

    *cost = best_c;

__CleanUp:

    err = destroy_cl();
    if (0 != err) {
        report_error("Error while destroying OpenCL context");
        return -1;
    }

    return last_error;
}
