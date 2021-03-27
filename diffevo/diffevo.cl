// Trick that allows us to directly include this file into diffevo.c as string.
#ifndef _s
#define _s(x)
#endif
_s(

//
// RFC 8682: TinyMT32 Pseudorandom Number Generator (PRNG)
//

typedef struct {
    unsigned st[4];
} mt32_t;

void mt32_next(mt32_t *r) {
    unsigned y = r->st[3];
    unsigned x = (r->st[0] & 0x7fffffffu) ^ r->st[1] ^ r->st[2];
    x ^= (x << 1);
    y ^= (y >> 1) ^ x;
    r->st[0] = r->st[1];
    r->st[1] = r->st[2];
    r->st[2] = x ^ (y << 10);
    r->st[3] = y;
    if (y & 1) {
        r->st[1] ^= 0x8f7011eeu;
        r->st[2] ^= 0xfc78ff1fu;
    }
}

void mt32_init(mt32_t *r, unsigned seed) {
    r->st[0] = seed;
    r->st[1] = 0x8f7011eeu;
    r->st[2] = 0xfc78ff1fu;
    r->st[3] = 0x3793fdffu;
    for (int i = 1; i < 8; i++) {
        r->st[i & 3] ^= i + 1812433253u * (r->st[(i - 1) & 3] ^ (r->st[(i - 1) & 3] >> 30));
    }
    for (int i = 0; i < 8; i++) {
        mt32_next(r);
    }
}

unsigned mt32_unsigned(mt32_t *r) {
    mt32_next(r);
    unsigned t0 = r->st[3];
    unsigned t1 = r->st[0] + (r->st[2] >> 8);
    t0 ^= t1;
    if (t1 & 1) {
        t0 ^= 0x3793fdffu;
    }
    return t0;
}

double mt32_double(mt32_t *r) {
    return mt32_unsigned(r) * (1.0 / 4294967296.0);
}

//
// Differential Evolution (DE) algorithm implementation
//

__kernel void init(
    __global mt32_t *restrict rng,
    __constant unsigned *restrict seeds,
    __global double *restrict pop,
    unsigned num_pop,
    unsigned num_attr,
    double mu,
    double sigma
) {
    const unsigned id = get_global_id(0);

    mt32_t r;
    mt32_init(&r, seeds[id]);

    for(unsigned a = 0; a < num_attr; a++) {
        // Box-Muller method to generate a Normal(mu, sigma^2) distributed number.
        const double x = mt32_double(&r);
        const double y = mt32_double(&r);
        const double z = mu + sigma * sqrt(-2.0 * log(x)) * cos(2.0 * M_PI * y);
        pop[id * num_attr + a] = z;
    }

    rng[id] = r;
}

__kernel void mutate(
    __global mt32_t *restrict rng,
    __constant double *restrict in_pop,
    __global double *restrict out_pop,
    unsigned num_pop,
    unsigned num_attr,
    double shrink,
    double crossover
) {
    const unsigned id = get_global_id(0);
    const unsigned t = id * num_attr;

    mt32_t r = rng[id];

    const unsigned u = (mt32_unsigned(&r) % num_pop) * num_attr;
    const unsigned v = (mt32_unsigned(&r) % num_pop) * num_attr;
    const unsigned w = (mt32_unsigned(&r) % num_pop) * num_attr;

    for(unsigned a = 0; a < num_attr; a++) { 
        const double p = in_pop[t + a];
        const double q = in_pop[u + a] + shrink * (in_pop[v + a] - in_pop[w + a]);
        out_pop[t + a] = mt32_double(&r) >= crossover ? p : q;
    }

    rng[id] = r;
}

__kernel void select(
    __constant double *restrict in1_pop,
    __constant double *restrict in1_cost,
    __constant double *restrict in2_pop,
    __constant double *restrict in2_cost,
    __global double *restrict out_pop,
    __global double *restrict out_cost,
    unsigned num_pop,
    unsigned num_attr
) {
    const unsigned id = get_global_id(0);
    const unsigned t = id * num_attr;

    const bool better_1 = in1_cost[id] < in2_cost[id];

    for(unsigned a = 0; a < num_attr; a++) {
        const unsigned ta = t + a;
        out_pop[ta] = better_1 ? in1_pop[ta] : in2_pop[ta];
    }

    out_cost[id] = better_1 ? in1_cost[id] : in2_cost[id];
}

)
