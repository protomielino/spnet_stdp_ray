#ifndef SIM_H_
#define SIM_H_

#include "neuron.h"

#define DT 1.0         /* ms per sim step */

#define MAX_DELAY 20 /* ms */

/* STDP parameters (pair-based) */
#define A_plus 0.1f
#define A_minus 0.12f
#define TAU_PLUS 20.0f
#define TAU_MINUS 20.0f
#define W_MIN 0.0f
#define W_MAX 10.0f

/* Delay queues: for each delay (1..MAX_DELAY) maintain list of targets arriving after that many ms */
typedef struct
{
    int   *neuron; /* target neuron ids */
    float *weight; /* corresponding weights */
    int    count;
    int    cap;
} DelayBucket;

/* Raster storage */
#define FIRING_BUF 2000000 /* pair (time, neuron) capacity */

typedef struct
{
    int   neuron;
    float time_ms;
} FiringTime;

typedef struct
{
    int num_exc; // number of exc neurons
    int num_inh; // number of inh neurons
    float exc_to_inh_ratio;
    IzkNeuron *neurons;
    DelayBucket *delaybuckets;    /* index 1..MAX_DELAY used; 0 unused */
    int current_delay_index;         /* rotates every ms */
    FiringTime *firing_times;            /* circular buffer of pairs (time, neuron) */
    int firing_count;
    int firing_cap;
    float *v_next;
    float *u_next;
    int *order;

    /* per-neuron v/u history index for selected trace (circular) */
    int vhist_idx;
    int uhist_idx;

    float input_prob;   // default synaptic input noise probability
    float input_val;     // default synaptic input current

    /* Simulation time */
    float t_ms;
} sim;

//static void init_delay_buckets(sim *s);
//static void ensure_bucket_cap(sim *s, DelayBucket *db, int need);
//static void bucket_push(sim *s, DelayBucket *db, int neuron, float weight);
//static void bucket_clear(sim *s, DelayBucket *db);
//static void add_firing_record(sim *s, float time_ms, int neuron);
//static void apply_stdp_on_pre(sim *s, int pre, float t_pre);
//static void apply_stdp_on_post(sim *s, Grid *grid, int post, float t_post);
//static void schedule_spike_delivery(sim *s, int pre, int conn_index);
void sim_init_network(sim *s, Grid *grid);
void sim_free_network(sim *s, Grid *grid);
void sim_step(sim *s, Grid *grid);

#endif /* SIM_H_ */
