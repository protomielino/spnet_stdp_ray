#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#include <raylib.h>
#define RAYMATH_IMPLEMENTATION
#include <raymath.h>

#define STB_DS_IMPLEMENTATION
#include "stb_ds.h"

#include "math_utils.h"
#include "sim.h"
#include "grid.h"
#include "palette.h"
#include "neuron.h"
#include "neuron_classification.h"

ColourEntry *palette = NULL;

/* Window */
#define WIDTH 1360
#define HEIGHT 740

#define DT 1.0         /* ms per sim step */

/* Visualization layout */
#define RASTER_H 350
#define VTRACE_H 110
#define PANEL_H 50
#define SELECT_TRACE_H 100

/* Delay queues: for each delay (1..MAX_DELAY) maintain list of targets arriving after that many ms */
typedef struct
{
    int *neuron;   /* target neuron ids */
    float *weight; /* corresponding weights */
    int count;
    int cap;
} DelayBucket;

/* Raster storage */
#define FIRING_BUF 2000000 /* pair (time, neuron) capacity */

/* STDP parameters (pair-based) */
#define A_plus 0.1f
#define A_minus 0.12f
#define TAU_PLUS 20.0f
#define TAU_MINUS 20.0f
#define W_MIN 0.0f
#define W_MAX 10.0f

typedef struct
{
    int   neuron;
    float time_ms;
} FiringTime;

/* Globals */
int num_exc = 0.0; // number of exc neurons
int num_inh = 0.0; // number of inh neurons
static IzkNeuron *neurons = NULL;
static DelayBucket *delaybuckets = NULL;    /* index 1..MAX_DELAY used; 0 unused */
static int current_delay_index = 0;         /* rotates every ms */
static FiringTime *firing_times = NULL;            /* circular buffer of pairs (time, neuron) */
static int firing_count = 0;
static int firing_cap = FIRING_BUF;
static float *v_next = NULL;
static float *u_next = NULL;
static int *order = NULL;

/* per-neuron v/u history index for selected trace (circular) */
static int vhist_idx = 0;
static int uhist_idx = 0;

/* Simulation time */
static float t_ms = 0.0;

/* Initialize delay buckets */
static void init_delay_buckets()
{
    if (arrlen(delaybuckets) != 0) {
        arrfree(delaybuckets);
    }
    delaybuckets = NULL;
    arrsetlen(delaybuckets, MAX_DELAY+1);
    for (int d = 0; d <= MAX_DELAY; d++) {
        delaybuckets[d].neuron = NULL;
        delaybuckets[d].weight = NULL;
        delaybuckets[d].count = 0;
        delaybuckets[d].cap = 0;
    }
    current_delay_index = 0;
}

/* Ensure capacity for bucket */
static void ensure_bucket_cap(DelayBucket *db, int need)
{
    if (db->cap >= need)
        return;
    int newcap = db->cap>0 ? db->cap*2 : 64;
    while (newcap < need)
        newcap *= 2;
    arrsetlen(db->neuron, newcap);
    arrsetlen(db->weight, newcap);
    db->cap = newcap;
}

/* Push to bucket (used when scheduling spikes) */
static void bucket_push(DelayBucket *db, int neuron, float weight)
{
    ensure_bucket_cap(db, db->count + 1);
    db->neuron[db->count] = neuron;
    db->weight[db->count] = weight;
    db->count++;
}

/* Pop all from bucket (used when delivering) */
static void bucket_clear(DelayBucket *db)
{
    db->count = 0;
}

int* array_permute(int *arr, int N)
{
    if (arr == NULL || arrlen(arr) == 0) {
        arrsetlen(arr, N);
    }
    for (int i = 0; i < N; ++i)
        arr[i] = i;
    for (int i = N-1; i > 0; --i) {
        int j = rand() % (i+1);
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
    return arr;
}

/* Initialize network (connections, weights, delays, v/u, buffers) */
static void init_network(Grid *grid)
{
    /* per-neuron v,u history index for selected trace (circular) */
    vhist_idx = 0;
    uhist_idx = 0;

    // exc to inh ratio (from paper)
    float exc_to_inh_ratio = 1.0/(4.0 + 1.0);
    num_exc = grid->numCells * exc_to_inh_ratio * 4.0; // number of exc neurons
    num_inh = grid->numCells * exc_to_inh_ratio * 1.0; // number of inh neurons

    /* flags: 1 = eccitatorio, 0 = inibitorio */
    uint8_t *flags = (uint8_t*)calloc(grid->numCells, sizeof(uint8_t));
    int *pool = (int*)malloc(grid->numCells * sizeof(int));
    for (int i = 0; i < grid->numCells; ++i)
        pool[i] = i;
    for (int k = 0; k < num_exc; ++k) {
        int r = rand() % (grid->numCells - k);
        flags[pool[r]] = 1;
        pool[r] = pool[grid->numCells - k - 1]; /* rimosso dallo pool */
    }
    free(pool);

    /* allocate */
    arrsetlen(neurons, grid->numCells);
    arrsetlen(firing_times, firing_cap);

    for (int i = 0; i < grid->numCells; i++) {
        neurons[i].cell_activity = 0.0f;
    }

    /* parametri del paper in mm */
    const float local_exc_span_mm = 1.5f;
    const float inh_span_mm = 0.5f;
    const float long_range_axon_length_mm = 12.0f;
    const float collateral_radius_mm = 0.5f;
    const float vel_myelinated = 1.0f;
    const float vel_unmyelinated = 0.15f;
    const int exc_local_targets = 75.0;      // assuming
    const int exc_distant_targets = 25.0;    // 10000
    const int inh_local_targets = 25.0;      // neurons

    /* init neurons usando flags[i] */
    for (int i = 0; i < grid->numCells; i++) {
        float ra = 0.0f;
        float ra2 = 0.0f;

        neurons[i].is_exc = (uint8_t)flags[i];

        ra = rand01(); ra2 = ra * ra;
        neurons[i].a = neurons[i].is_exc ? 0.02f : 0.02f + 0.08f * ra;
        neurons[i].b = neurons[i].is_exc ? 0.2f : 0.25f - 0.05f * ra;
        neurons[i].c = neurons[i].is_exc ? -65.0f + 15.0f * ra2 : -65.0f;
        neurons[i].d = neurons[i].is_exc ? 8.0f - 6.0f * ra2 : 2.0f;

        neurons[i].v = neurons[i].c;
        neurons[i].u = neurons[i].b * neurons[i].v;
        neurons[i].last_spike_time = -1000000;
        for (int k = 0; k < VUBUF_LEN_MS; k++) {
            neurons[i].v_hist[k] = neurons[i].v;
            neurons[i].u_hist[k] = neurons[i].u;
        }
    }

    /* principale loop sui neuroni */
    for (int i = 0; i < grid->numCells; ++i) {
        int is_exc = neurons[i].is_exc;

        neurons[i].outconn.targets = NULL;
        neurons[i].outconn.weights = NULL;
        neurons[i].outconn.delay = NULL;
        neurons[i].num_targets = 0;
        neurons[i].num_near_conn = 0;
        neurons[i].num_far_conn = 0;
        neurons[i].target_center = (CellPos){0};

        CellPos this_cell = {0};
        grid_index_to_cellpos(grid, i, &this_cell);

        /* 1) local targets: usa annulus con rmin=0, rmax = radius_in_cells */
        float local_mm = is_exc ? local_exc_span_mm : inh_span_mm;
        int cells_rmin_local = 0;
        int cells_rmax_local = (int)ceilf(mm_to_cells_float(grid, local_mm));
        CellPos *local_post = NULL;
        int need = is_exc ? exc_local_targets : inh_local_targets;
        arrsetlen(local_post, need);
        int found_local = grid_pick_random_cells_in_annulus(grid,
                this_cell,
                cells_rmin_local, cells_rmax_local,
                need,
                local_post);
        arrsetlen(local_post, found_local);
        for (int j = 0; j < found_local; ++j) {
            int targ = grid_cellpos_to_index(grid, local_post[j]);
            if (targ != i) {
                arrput(neurons[i].outconn.targets, targ);
                if (is_exc) {
                    arrput(neurons[i].outconn.weights, 6.0f * frandf());
                } else {
                    arrput(neurons[i].outconn.weights, -5.0f * frandf());
                }
                int dsq = grid_toroidal_dist_sq(grid, this_cell, local_post[j]);
                int cell_dist = (int)floorf(sqrtf((float)dsq) + 0.5f);
                arrput(neurons[i].outconn.delay, compute_delay_from_cells(grid, cell_dist, vel_unmyelinated));
                neurons[i].num_near_conn ++;
            }
        }
        arrfree(local_post);

        /* 2) excitatory distant targets: scegli punto a distanza ~12mm e prendi annulus radius = 0.5mm attorno ad esso */
        if (is_exc) {
            /* convert lengths to cells */
            int target_cell_dist = (int)roundf(mm_to_cells_float(grid, long_range_axon_length_mm));
            int collateral_rcells = (int)ceilf(mm_to_cells_float(grid, collateral_radius_mm));

            /* troviamo candidate center cells a distanza target_cell_dist (annulus rmin=r-1,rmax=r+1) */
            CellPos *distant_targets = NULL;
            int need_distant_targets = 1;
            arrsetlen(distant_targets, need_distant_targets);
            /* usa annulus intorno al centro della sorgente: rmin=rmax=target_cell_dist per cercare celle a quella distanza */
            int found_distant_targets = grid_pick_random_cells_in_annulus(grid, this_cell, target_cell_dist - 1, target_cell_dist + 1, need_distant_targets, distant_targets);
            arrsetcap(distant_targets, found_distant_targets);
            arrsetlen(distant_targets, found_distant_targets);
            /* se non troviamo abbastanza centri, accettiamo quelli trovati; da ciascuno prendiamo fino a exc_distant_targets */
            if (found_distant_targets > 0) {
                int need = exc_distant_targets;
                for (int dist_targ_idx = 0; dist_targ_idx < found_distant_targets && need > 0; ++dist_targ_idx) {
                    CellPos center = distant_targets[dist_targ_idx];
                    neurons[i].target_center = center;

                    int dsq = grid_toroidal_dist_sq(grid, this_cell, distant_targets[dist_targ_idx]);
                    int targ_dist = (int)floorf(sqrtf((float)dsq) + 0.5f);
                    unsigned char axon_delay = compute_delay_from_cells(grid, targ_dist, vel_myelinated);

                    CellPos *far_post = NULL;
                    arrsetlen(far_post, need);
                    int got = grid_pick_random_cells_in_annulus(grid, center, 0, collateral_rcells, need, far_post);
                    arrsetcap(far_post, got);
                    arrsetlen(far_post, got);
                    for (int jj = 0; jj < got; ++jj) {
                        int targ = grid_cellpos_to_index(grid, far_post[jj]);
                        arrput(neurons[i].outconn.targets, targ);
                        arrput(neurons[i].outconn.weights, 6.0f * frandf());
                        int dsq = grid_toroidal_dist_sq(grid, center, far_post[jj]);
                        int cell_dist = (int)floorf(sqrtf((float)dsq) + 0.5f);
                        arrput(neurons[i].outconn.delay, axon_delay + compute_delay_from_cells(grid, cell_dist, vel_unmyelinated));
                        neurons[i].num_far_conn ++;
                    }
                    arrfree(far_post); far_post = NULL;

                    neurons[i].num_targets ++;
                }
            }
            arrfree(distant_targets); distant_targets = NULL;
        }

//        /* 3) riempi con bersagli casuali se necessario */
//        while (filled < K) {
//            int targ = rand() % N;
//            neurons[i].outconn.targets[filled] = targ;
//            if (is_exc) {
//                neurons[i].outconn.weights[filled] = 6.0f * frandf();
//                int tr, tc;
//                index_to_cellpos(targ, &tr, &tc);
//                int dsq = toroidal_dist_sq(cr, cc, tr, tc);
//                int cell_dist = (int)floorf(sqrtf((float)dsq) + 0.5f);
//                neurons[i].outconn.delay[filled] = compute_delay_from_cells(cell_dist, vel_unmyelinated);
//            } else {
//                neurons[i].outconn.weights[filled] = -5.0f * frandf();
//                neurons[i].outconn.delay[filled] = 1;
//            }
//            filled++;
//        }
//    }
//    /* init connections usando neurons[i].is_exc */
//    for (int i = 0; i < grid->numCells; i++) {
//        int K = neurons[i].is_exc ? CE : CI;
//        CellPos *selected = NULL;
//        arrsetlen(selected, K);
//
//        int rmax = neurons[i].is_exc ? 6 : 3;
//        CellPos this_cell = grid_index_to_cellpos(grid, i, &this_cell);
//        // primo picking iniziale
//        grid_pick_random_cells_in_annulus(grid, this_cell, 0, rmax, K, selected);
//
//        neurons[i].outconn.targets = NULL;
//        neurons[i].outconn.weights = NULL;
//        neurons[i].outconn.delay = NULL;
//        for (int j = 0; j < K; j++) {
//            CellPos c = {selected[j].r, selected[j].c};
//            arrput(neurons[i].outconn.targets, grid_cellpos_to_index(grid, c));
//            if (neurons[i].is_exc) {
//                /* excitatory initial weight random around 6.0 +- */
//                arrput(neurons[i].outconn.weights, 6.0f * frandf());
//                /* excitatory delay 1..MAX_DELAY */
//                arrput(neurons[i].outconn.delay, (unsigned char)(1 + (rand() % MAX_DELAY)));
//            } else {
//                /* inhibitory negative weight */
//                arrput(neurons[i].outconn.weights, -5.0f * frandf());
//                arrput(neurons[i].outconn.delay, 1); /* inhibitory delay 1 ms */
//            }
//        }
//
//        arrfree(selected);
    }

    init_delay_buckets();
    firing_count = 0;
    t_ms = 0.0;
    vhist_idx = 0;
    grid->selected_cell = -1;

    free(flags);
}

/* Free network memory */
static void free_network(Grid *grid)
{
    if (!neurons)
        return;
    for (int i = 0; i < grid->numCells; i++) {
        arrfree(neurons[i].outconn.targets);
        arrfree(neurons[i].outconn.weights);
        arrfree(neurons[i].outconn.delay);
    }
    arrfree(firing_times);
    for (int d = 0; d <= MAX_DELAY; d++) {
        arrfree(delaybuckets[d].neuron);
        arrfree(delaybuckets[d].weight);
    }
    arrfree(delaybuckets);
    arrfree(neurons);
    arrfree(order);
    arrfree(v_next);
    arrfree(u_next);
}

/* Add firing to circular buffer of pairs (time, neuron) */
static void add_firing_record(float time_ms, int neuron)
{
    if (firing_count >= firing_cap) {
        /* simple downsample: shift keep half */
        int keep = firing_cap / 2;
        int start = (firing_count - keep);
        memmove(firing_times, firing_times + start, keep * sizeof(int));
        firing_count = keep;
    }
    firing_times[firing_count] = (FiringTime){ neuron, time_ms };
    firing_count++;
}

/* STDP weight update on spike: apply pair-based approximation
   When neuron 'pre' spikes at time t_pre, potentiate outgoing synapses to posts that spiked recently.
   When neuron 'post' spikes at time t_post, depress incoming excitatory synapses from pres that spiked recently.
   We'll implement updates at pre spike time on outgoing weights using last_spike_time[post].
*/
static void apply_stdp_on_pre(int pre, float t_pre)
{
    int K = arrlen(neurons[pre].outconn.targets); /* only excitatory neurons have CE */
    if (!neurons[pre].is_exc) /* only excitatory synapses are plastic */
        return;
    for (int i = 0; i < K; i++) {
        int post = neurons[pre].outconn.targets[i];
        float t_post = neurons[post].last_spike_time;
        if (t_post <= -100000)
            continue;
        float dt = t_pre - t_post; /* positive if pre after post => depression */
        if (dt > 0 && dt < 1000) {
            /* pre after post -> LTD (A_minus), dt positive */
            float dw = -A_minus * expf(- dt / TAU_MINUS);
            neurons[pre].outconn.weights[i] += dw;
        } else {
            /* pre before post => potentiation handled when post spikes, to keep symmetry we handle both sides:
               We'll also handle LTP when pre precedes post by applying when post spikes (below). */
        }
        /* clamp */
        neurons[pre].outconn.weights[i] =
                clampf(neurons[pre].outconn.weights[i], W_MIN, W_MAX);
    }
}

/* Called when a neuron spikes (post), apply LTP for incoming excitatory synapses.
   We don't store incoming lists for memory reasons, so iterate all excitatory neurons and check if they connect to 'post' - costly but acceptable for moderate CE/NE.
   Optimization: only check outgoing from excitatory population.
*/
static void apply_stdp_on_post(Grid *grid, int post, float t_post)
{
    /* For each excitatory neuron pre, check its outgoing connections for post */
    for (int pre = 0; pre < grid->numCells; pre++) {
        if (neurons[pre].is_exc) {
            int K = arrlen(neurons[pre].outconn.targets);
            for (int j = 0; j < K; j++) {
                if (neurons[pre].outconn.targets[j] != post)
                    continue;
                float t_pre = neurons[pre].last_spike_time;
                if (t_pre <= -100000)
                    continue;
                float dt = t_post - t_pre; /* positive if post after pre => potentiation */
                if (dt > 0 && dt < 1000) {
                    float dw = A_plus * expf(- (float)dt / TAU_PLUS);
                    neurons[pre].outconn.weights[j] += dw;
                    /* clamp */
                    neurons[pre].outconn.weights[j] = clampf(neurons[pre].outconn.weights[j], W_MIN, W_MAX);
                }
            }
        }
    }
}

/* Schedule a delivered event: place target in delay bucket for appropriate arrival time */
static void schedule_spike_delivery(int pre, int conn_index)
{
    int post = neurons[pre].outconn.targets[conn_index];
    float w = neurons[pre].outconn.weights[conn_index];
    int delay = neurons[pre].outconn.delay[conn_index];
    if (delay < 1)
        delay = 1;
    if (delay > MAX_DELAY)
        delay = MAX_DELAY;
    int bucket_idx = (current_delay_index + delay) % (MAX_DELAY+1);
    /* note: using 0..MAX_DELAY buckets, but we never place into index 0 unless delay==0; safe since bucket array sized */
    bucket_push(&delaybuckets[bucket_idx], post, w);
}

static float input_prob = 0.056f;   // default synaptic input noise probability
static float input_val = 24.0f;     // default synaptic input current

/* Simulation single step (1 ms) */
static void sim_step(Grid *grid)
{
    /* 1) Deliver all events in the current bucket (arrivals scheduled for this ms) */
    DelayBucket *db = &delaybuckets[current_delay_index];
    /* produce an input array I for this ms */
    for (int i = 0; i < grid->numCells; i++)
        neurons[i].I = 0.0f;

    for (int k = 0; k < db->count; k++) {
        int neuron = db->neuron[k];
        float w = db->weight[k];
        neurons[neuron].I += w;
    }
    /* clear bucket for reuse (it will be filled for future times) */
    bucket_clear(db);

    /* 2) External noisy input: Poisson-like drive to excitatory neurons */
    for (int i = 0; i < grid->numCells; i++) {
        if (neurons[i].is_exc)
            if (frandf() < input_prob)
                neurons[i].I += input_val * frandf();
    }

    int num_sub_steps = 4;

#define INTEGRATOR_BIAS_CORRECTION 2
#if (INTEGRATOR_BIAS_CORRECTION == 0)
    /* 3) Integrate neuron dynamics (Izhikevich) — versione naive (drifting bias) */
    for (int step = 0; step < num_sub_steps; ++step) {
        for (int i = 0; i < grid->numCells; i++) {
            float dv = 0.04f * neurons[i].v * neurons[i].v + 5.0f * neurons[i].v + 140.0f - neurons[i].u + neurons[i].I;
            neurons[i].v += dv * (DT / (float)num_sub_steps);
            neurons[i].u += neurons[i].a * (neurons[i].b * neurons[i].v - neurons[i].u) * (DT / (float)num_sub_steps);
        }
    }
#elif (INTEGRATOR_BIAS_CORRECTION == 1)
    /* 3) Integrate: permuta deterministica dell'ordine */
    if (!order)
        arrsetlen(order, grid->numCells);
    /* inizializza ordine una volta (0..N-1) */
    for (int i = 0; i < grid->numCells; i++)
        order[i] = i;
    /* shuffle deterministico, ad es. xorshift con seed basato su t_ms */
    uint32_t seed = (uint32_t)(t_ms + 123456);
    for (int i = grid->numCells - 1; i > 0; --i) {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        uint32_t r = seed % (i + 1);
        int tmp = order[i];
        order[i] = order[r];
        order[r] = tmp;
    }

    /* due sottopassi: per ogni sottopasso aggiorna secondo order[] */
    for (int step = 0; step < num_sub_steps; ++step) {
        for (int idx = 0; idx < grid->numCells; ++idx) {
            int i = order[idx];
            float dv = 0.04f * neurons[i].v * neurons[i].v + 5.0f * neurons[i].v + 140.0f - neurons[i].u + neurons[i].I;
            neurons[i].v += dv * (DT / (float)num_sub_steps);
            neurons[i].u += neurons[i].a * (neurons[i].b * neurons[i].v - neurons[i].u) * (DT / (float)num_sub_steps);
        }
    }
#elif (INTEGRATOR_BIAS_CORRECTION == 2)
    /* 3) Integrate neuron dynamics (Izhikevich) — versione buffer */
    if (!v_next) {
        arrsetlen(v_next, grid->numCells);
        arrsetlen(u_next, grid->numCells);
    }
    /* inizializza con valori correnti (necessario per sottopassi) */
    for (int i = 0; i < grid->numCells; i++) {
        v_next[i] = neurons[i].v;
        u_next[i] = neurons[i].u;
    }

    for (int step = 0; step < num_sub_steps; ++step) {
        /* calcola nuovi v/u su base dei valori *correnti* in neurons[] */
        for (int i = 0; i < grid->numCells; i++) {
            float v_cur = neurons[i].v;
            float u_cur = neurons[i].u;
            float I = neurons[i].I;
            float dv = 0.04f * v_cur * v_cur + 5.0f * v_cur + 140.0f - u_cur + I;
            float v_new = v_cur + dv * (DT / (float)num_sub_steps);
            float u_new = u_cur + neurons[i].a * (neurons[i].b * v_cur - u_cur) * (DT / (float)num_sub_steps);
            v_next[i] = v_new;
            u_next[i] = u_new;
        }
        /* dopo aver calcolato tutti i nuovi valori, copia indietro (swap) */
        for (int i = 0; i < grid->numCells; i++) {
            neurons[i].v = v_next[i];
            neurons[i].u = u_next[i];
        }
    }
#endif

    /* 4) Check for spikes (v >= 30) */
    for (int i = 0; i < grid->numCells; i++) {
        if (neurons[i].v >= 30.0f) {
            /* record spike */
            add_firing_record(t_ms, i);
            /* STDP: apply pre-spike rule (depression for pre after recent post) */
            apply_stdp_on_pre(i, t_ms);
            /* reset */
            neurons[i].v = neurons[i].c;
            neurons[i].u += neurons[i].d;
            /* schedule deliveries to targets according to their delays */
            int K = arrlen(neurons[i].outconn.targets);
            for (int j = 0; j < K; j++) {
                schedule_spike_delivery(i, j);
            }
            /* STDP: handle post-spike LTP for incoming excitatory synapses */
            apply_stdp_on_post(grid, i, t_ms);
            /* update last spike time */
            neurons[i].last_spike_time = t_ms;
        }
    }

    /* 5) advance delay index and time, update v history buffer index */
    current_delay_index = (current_delay_index + 1) % (MAX_DELAY+1);
    t_ms += DT;
    vhist_idx = (vhist_idx + 1) % VUBUF_LEN_MS;
    uhist_idx = (uhist_idx + 1) % VUBUF_LEN_MS;
    for (int i = 0; i < grid->numCells; i++) {
        neurons[i].v_hist[vhist_idx] = neurons[i].v;
        neurons[i].u_hist[uhist_idx] = neurons[i].u;
    }
}

/* compute mean excitatory weight */
static float mean_exc_weight(Grid *grid)
{
    double s = 0.0;
    long cnt = 0;
    for (int i = 0; i < grid->numCells; i++) {
        if (neurons[i].is_exc) {
            int CE = arrlen(neurons[i].outconn.weights);
            for (int j = 0; j < CE; j++) {
                s += neurons[i].outconn.weights[j];
                cnt++;
            }
        }
    }
    if (cnt == 0)
        return 0.0f;
    return (float)(s / cnt);
}

/* Find neuron by clicking raster: map x,y to time and neuron id */
static int neuron_from_raster_click(Grid *grid, int click_x, int click_y, int rx, int ry, int rw, int rh)
{
    /* If click outside raster area return -1 */
    if (click_x < rx || click_x > rx+rw || click_y < ry || click_y > ry+rh)
        return -1;
    /* x -> neuron id: left excitatory, right inhibitory */
    int relx = click_x - rx;
    int nid = map(relx, 0.0, rw, 0.0, grid->numCells);
    /* excitatory map */
    if (nid < 0)
        nid = 0;
    if (nid >= grid->numCells)
        nid = grid->numCells-1;
    return nid;
}

bool graphics_raster = true;
bool graphics_grid = false;

/* Draw selected neuron v/u(t) trace */
static void draw_selected_trace(Grid *grid, int sx, int sy, int sw, int sh)
{
    if (grid->selected_cell < 0) {
        DrawText(TextFormat("No neuron selected. Click %s to select.", (graphics_raster==true)?"raster":"grid"), sx+10, sy+10, 14, GRAY);
        return;
    }

    char buf[1024];
    snprintf(buf, sizeof(buf),
                "Neuron %d  v,u(t) last %d ms\n"
                "Neuron params: a=%.4f b=%.4f c=%.2f d=%.2f\n"
                "Type: %s  Score: %.3f\n"
                "%s\n",
                grid->selected_cell, VUBUF_LEN_MS,
                neurons[grid->selected_cell].a,
                neurons[grid->selected_cell].b,
                neurons[grid->selected_cell].c,
                neurons[grid->selected_cell].d,
                neurons[grid->selected_cell].class_result.type,
                neurons[grid->selected_cell].class_result.score,
                neurons[grid->selected_cell].class_result.reason);
    DrawText(buf, sx+10, sy+10, 10, LIGHTGRAY);

    /* draw outline */
    DrawRectangleLines(sx, sy, sw, sh, LIGHTGRAY);

    /* find time window: show last VUBUF_LEN_MS ms */
    int idx = vhist_idx;
    float vv;
    float uu;
    int vpx_prev = -1, vpy_prev = -1;
    int upx_prev = -1, upy_prev = -1;
    for (int k = 0; k < VUBUF_LEN_MS; ++k) {
        int pos = (idx - (VUBUF_LEN_MS-1) + k);
        while (pos < 0)
            pos += VUBUF_LEN_MS;
        pos %= VUBUF_LEN_MS;
        vv = neurons[grid->selected_cell].v_hist[pos];
        uu = neurons[grid->selected_cell].u_hist[pos];
        /* map vv (-100..40) to y */
        float vnorm = (vv + 80.0f) / 120.0f;
        float unorm = (uu + 20.0f) / 20.0f;
        vnorm = vnorm < 0 ? 0 : vnorm;
        vnorm = vnorm > 1 ? 1 : vnorm;
        unorm = unorm < 0 ? 0 : unorm;
        unorm = unorm > 1 ? 1 : unorm;
        float x = sx + 1 + ((float)k / (float)(VUBUF_LEN_MS-1) * (float)(sw-2));
        float vy = sy + 1 + 10 + ((1.0f - vnorm) * (sh - 20));
        float uy = sy + 1 + 10 + ((1.0f - unorm) * (sh - 20));
        Color vC = neurons[grid->selected_cell].is_exc ? GREEN : DARKGREEN;
        Color uC = neurons[grid->selected_cell].is_exc ? SKYBLUE : BLUE;
        if (k > 0) {
            DrawLine(vpx_prev, vpy_prev, x, vy, vC);
            DrawLine(upx_prev, upy_prev, x, uy, uC);
        }
        vpx_prev = x; vpy_prev = vy;
        upx_prev = x; upy_prev = uy;
    }
}

// compute cell activities (using interleaved mapping)
void compute_cell_activity(Grid *grid)
{
    // zero
    for (int k = 0; k < grid->numCells; k++)
        neurons[k].cell_activity = 0.0f;

    // accumulate contributions per neuron into its cell
    for (int cell = 0; cell < grid->numCells; cell++) {
        // metric: weighted sum of receptor conductances and depolarization
        float metric = 0.0f;
//        if (disp_ampa)
//            metric += fabsf(syn[i].g_ampa);
//        if (disp_nmda)
//            metric += 0.5f * fabsf(syn[i].g_nmda);
//        if (disp_gabaa)
//            metric += fabsf(syn[i].g_gabaa);
//        if (disp_gabab)
//            metric += 0.5f * fabsf(syn[i].g_gabab);
        float vdep = neurons[cell].v + 65.0f;
        if (vdep > 0.0f)
            metric += 0.02f * vdep;
        neurons[cell].cell_activity += metric;
    }
    // normalize
    for (int k = 0; k < grid->numCells; k++) {
        // normalization scale empirical
        float val = neurons[k].cell_activity; // TESTING / 20.0f;
        if (val > 1.0f)
            val = 1.0f;
        neurons[k].cell_activity = val;
    }
}

void grid_show(Grid *grid)
{
    for (int r = 0; r < grid->numRows; r++) {
        for (int c = 0; c < grid->numCols; c++) {
            int idx = grid_cellpos_to_index(grid, (CellPos){r, c});
            float val = neurons[idx].cell_activity;
            Color col = Palette_Sample(&palette, val);
            int x = grid->margin + (float)c * grid->cellWidth;
            int y = grid->margin + (float)r * grid->cellHeight;
            DrawRectangle(x+1, y+1, grid->cellWidth - 1, grid->cellHeight - 1, col);
            if (idx == grid->selected_cell) {
                DrawRectangleLines(x, y, grid->cellWidth, grid->cellHeight, MAGENTA);
            }
        }
    }

    DrawRectangleLines(grid->margin, grid->margin, grid->width, grid->height, WHITE);

    // overlay UI text
//    DrawText("Controls: P Pause | R Reset | +/- Speed | 1 AMPA 2 NMDA 3 GABAA 4 GABAB", 10, 8, 18, WHITE);
//    DrawText(TextFormat("Sim steps/frame: %d   Paused: %s", steps_per_frame, paused ? "YES" : "NO"), 10, 30, 16, WHITE);
//    DrawText(TextFormat("Display chan: AMPA[%c] NMDA[%c] GABAA[%c] GABAB[%c]",
//            disp_ampa ? 'X' : ' ', disp_nmda ? 'X' : ' ', disp_gabaa ? 'X' : ' ', disp_gabab ? 'X' : ' '),
//            10, 50, 16, WHITE);
//    DrawText(TextFormat("Sim time: %.0f ms", (sim_step_counter * sim_dt)), 10, 70, 16, WHITE);
//    DrawText(TextFormat("Firing recorded: %d", firing_count), 10, 90, 16, WHITE);
//    DrawText(TextFormat("g_exc_gain: %f  g_inh_gain: %f", g_exc_gain, g_inh_gain), 10, 110, 16, WHITE);
}

// Custom logging function
void CustomLog(int msgType, const char *text, va_list args)
{
#if 0
    char timeStr[64] = { 0 };
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);

    strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", tm_info);
    printf("[%s] ", timeStr);

    switch (msgType)
    {
        case LOG_INFO: printf("[INFO] : "); break;
        case LOG_ERROR: printf("[ERROR]: "); break;
        case LOG_WARNING: printf("[WARN] : "); break;
        case LOG_DEBUG: printf("[DEBUG]: "); break;
        default: break;
    }

    vprintf(text, args);
    printf("\n");
#endif
}

int main(int argc, char **argv)
{
    srand((unsigned)time(NULL));

    Grid grid = {0};
    grid.margin = 20;

    int grid_width = (WIDTH - 2*grid.margin);
    int grid_height = (HEIGHT - 2*grid.margin);
    int num_cells = 5000;
    if (argc > 1)
        num_cells = atoi(argv[1]);

    GridChoice gc = grid_choose(num_cells, grid_width, grid_height, false);
    grid = (Grid){
            .numCols = gc.cols,
            .numRows = gc.rows,
            .numCells = (gc.cols*gc.rows),
            .width = grid_width,
            .height = grid_height,
            .cellWidth = ((float)grid_width / (float)gc.cols),
            .cellHeight = ((float)grid_height / (float)gc.rows),
            .margin = 20,
            .selected_cell = -1
    };

    Palette_init(&palette, STOCK_COLDHOT3);

    SetTraceLogCallback(CustomLog); // Set custom logger
    InitWindow(WIDTH, HEIGHT, "spnet_ray_stdp - Izhikevich + STDP (C + raylib)");
    SetTargetFPS(30);

    init_network(&grid);

    int paused = 0;
    int show_graphics = 1;
    int show_fps = 0;
    int steps_per_frame = 1;

    bool graphics_raster = true;
    bool graphics_grid = false;

    while (!WindowShouldClose())
    {
        int mx = GetMouseX();
        int my = GetMouseY();

        if (IsKeyPressed(KEY_F1)) {
            input_prob -= 0.001;
        }
        if (IsKeyPressed(KEY_F2)) {
            input_prob += 0.001;
        }
        if (IsKeyPressed(KEY_F3)) {
            input_val -= 0.1;
        }
        if (IsKeyPressed(KEY_F4)) {
            input_val += 0.1;
        }

        /* input */
        if (IsKeyPressed(KEY_G)) {
            graphics_raster = !graphics_raster;
            graphics_grid = !graphics_grid;
        }
        if (IsKeyPressed(KEY_F))
            show_fps = !show_fps;
        if (IsKeyPressed(KEY_D))
            show_graphics = !show_graphics;
        if (IsKeyPressed(KEY_SPACE) && !IsKeyDown(KEY_LEFT_SHIFT)) {
            paused = !paused;
        }
        if (IsKeyPressed(KEY_UP))
            steps_per_frame = Clamp(steps_per_frame+1, 1, 5000);
        if (IsKeyPressed(KEY_DOWN))
            steps_per_frame = Clamp(steps_per_frame-1, 1, 5000);
        if (IsKeyPressed(KEY_R)) {
            free_network(&grid);
            init_network(&grid);
        }
        if (paused)
            if (IsKeyPressed(KEY_RIGHT))
                sim_step(&grid);

        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
            if (graphics_raster) {
                /* raster area coords */
                int rx = grid.margin;
                int ry = grid.margin;
                int rw = WIDTH - 2*grid.margin;
                int rh = RASTER_H;

                grid.selected_cell = neuron_from_raster_click(&grid, mx, my, rx, ry, rw, rh);
            } else if (graphics_grid) {
                /* grid area coords */
                int rx = grid.margin;
                int ry = grid.margin;
                int rw = grid.width;
                int rh = grid.height;

                grid.selected_cell = cell_index_from_grid_click(&grid, mx, my, rx, ry, rw, rh);
            }

            if (grid.selected_cell >= 0) {
                float a = neurons[grid.selected_cell].a;
                float b = neurons[grid.selected_cell].b;
                float c = neurons[grid.selected_cell].c;
                float d = neurons[grid.selected_cell].d;

                ClassResult r = classify_neuron(a,b,c,d);
                neurons[grid.selected_cell].class_result = r;
            }
        }

        /* simulate */
        if (!paused)
            for (int s = 0; s < steps_per_frame; s++)
                sim_step(&grid);

        /* draw */
        BeginDrawing(); {
            ClearBackground(BLACK);
            if (show_graphics) {
                if (graphics_grid) {
                    // compute activities for visualization
                    compute_cell_activity(&grid);

                    grid_show(&grid);

//                    // evidenzia celle disponibili nell'anello (trasparente)
//                    for (int ni = 0; ni < grid.numCells; ++ni) {
//                        CellPos cell_rc = {0};
//                        cell_rc = index_to_cellpos(ni, &cell_rc.r, &cell_rc.c);
//                        if (in_annulus(cell_rc.r, cell_rc.c, center, rmin, rmax)) {
//                            Vector2 cell_xy = cellpos_to_vec2(cell_rc);
//                            DrawRectangleLines(cell_xy.x+1, cell_xy.y+1, grid.cellWidth - 2, grid.cellHeight - 2, (Color){200, 230, 255, 255});
//                        }
//                    }
//
//                    for (int i = 0; i < selected_count; i++) {
//                        // selected cell
//                        CellPos cp = {0};
//                        cp.r = selected[i].r;
//                        cp.c = selected[i].c;
//                        Vector2 xy = cellpos_to_vec2(cp);
//                        DrawRectangleLines(xy.x+2, xy.y+2, grid.cellWidth-4, grid.cellHeight-4, (Color){ 253, 249, 0, 255 });
//                    }

                    if (grid.selected_cell >= 0) {
                        /* selected neuron trace */
                        int sx;
                        int sy = my + grid.margin/2;
                        int sw;
                        int sh = SELECT_TRACE_H-10;
                        if (mx < WIDTH/2) {
                            sx = mx + grid.margin;
                            sw = WIDTH - 2*grid.margin - mx;
                        } else {
                            sx = grid.margin;
                            sw = mx - 2*grid.margin;
                        }

                        draw_selected_trace(&grid, sx, sy, sw, sh);
                    }

                    if (grid.selected_cell >= 0) {
                        /* only excitatory neurons have CE */
                        int K_near = neurons[grid.selected_cell].num_near_conn;
                        for (int i = 0; i < K_near; ++i) {
                            int post = neurons[grid.selected_cell].outconn.targets[i];
                            CellPos target_pos = {0};
                            CellPos post_pos = {0};
                            target_pos = grid_index_to_cellpos(&grid, grid.selected_cell, &target_pos);
                            post_pos = grid_index_to_cellpos(&grid, post, &post_pos);
                            Vector2 target_pos_xy = grid_cellpos_to_vec2(&grid, target_pos);
                            Vector2 post_pos_xy = grid_cellpos_to_vec2(&grid, post_pos);
                            target_pos_xy = Vector2Add(target_pos_xy, (Vector2){ grid.cellWidth/2, grid.cellHeight/2 });
                            post_pos_xy = Vector2Add(post_pos_xy, (Vector2){ grid.cellWidth/2, grid.cellHeight/2 });
                            DrawLineV(target_pos_xy, post_pos_xy, (Color){255, 255, 255, 64});
                        }
                        if (neurons[grid.selected_cell].num_targets > 0) {
                            int post = grid_cellpos_to_index(&grid, neurons[grid.selected_cell].target_center);;
                            CellPos target_pos = {0};
                            CellPos post_pos = {0};
                            target_pos = grid_index_to_cellpos(&grid, grid.selected_cell, &target_pos);
                            post_pos = grid_index_to_cellpos(&grid, post, &post_pos);
                            Vector2 target_pos_xy = grid_cellpos_to_vec2(&grid, target_pos);
                            Vector2 post_pos_xy = grid_cellpos_to_vec2(&grid, post_pos);
                            target_pos_xy = Vector2Add(target_pos_xy, (Vector2){ grid.cellWidth/2, grid.cellHeight/2 });
                            post_pos_xy = Vector2Add(post_pos_xy, (Vector2){ grid.cellWidth/2, grid.cellHeight/2 });
                            DrawLineV(target_pos_xy, post_pos_xy, (Color){128, 255, 128, 64});
                        }
                        int K_far = neurons[grid.selected_cell].num_far_conn;
                        for (int i = K_near; i < K_far+K_near; ++i) {
                            int post = neurons[grid.selected_cell].outconn.targets[i];
                            CellPos target_pos = neurons[grid.selected_cell].target_center;
                            CellPos post_pos = {0};
                            post_pos = grid_index_to_cellpos(&grid, post, &post_pos);
                            Vector2 target_pos_xy = grid_cellpos_to_vec2(&grid, target_pos);
                            Vector2 post_pos_xy = grid_cellpos_to_vec2(&grid, post_pos);
                            target_pos_xy = Vector2Add(target_pos_xy, (Vector2){ grid.cellWidth/2, grid.cellHeight/2 });
                            post_pos_xy = Vector2Add(post_pos_xy, (Vector2){ grid.cellWidth/2, grid.cellHeight/2 });
                            DrawLineV(target_pos_xy, post_pos_xy, (Color){255, 255, 255, 128});
                        }

//                        for (int i = 0; i < K; ++i) {
//                            int post = neurons[grid.selected_cell].outconn.targets[i];
//                            CellPos target_pos = {0};
//                            CellPos post_pos = {0};
//                            target_pos = grid_index_to_cellpos(&grid, grid.selected_cell, &target_pos);
//                            post_pos = grid_index_to_cellpos(&grid, post, &post_pos);
//                            Vector2 this_cell_pos_xy = grid_cellpos_to_vec2(&grid, target_pos);
//                            Vector2 post_cell_pos_xy = grid_cellpos_to_vec2(&grid, post_pos);
//                            this_cell_pos_xy = Vector2Add(this_cell_pos_xy, (Vector2){ grid.cellWidth/2, grid.cellHeight/2 });
//                            post_cell_pos_xy = Vector2Add(post_cell_pos_xy, (Vector2){ grid.cellWidth/2, grid.cellHeight/2 });
//                            DrawLineV(this_cell_pos_xy, post_cell_pos_xy, (Color){255, 255, 255, 32});
//                        }
                    }
                }
                if (graphics_raster) {
                    /* Raster */
                    int rx = grid.margin;
                    int ry = grid.margin;
                    int rw = WIDTH - 2*grid.margin;
                    int rh = RASTER_H;
                    DrawRectangleLines(rx-1, ry-1, rw+2, rh+2, LIGHTGRAY);
#if 0
                    for (float cl = 0; cl < grid.width; cl += grid.cellWidth) { 
                        DrawLine(cl+grid.margin, grid.margin, cl+grid.margin, rh+grid.margin, (Color){255,255,255,64});
                    }
#endif
                    DrawText("Raster (last 1000 ms)", rx+6, ry+6, 14, LIGHTGRAY);

                    /* draw spikes in last 1000 ms */
                    float window_ms = 1000.0;
                    float display_start = t_ms - window_ms;
                    if (display_start < 0)
                        display_start = 0;
                    for (int i = 0; i < firing_count; i++) {
                        float ft = firing_times[i].time_ms;   // firing time
                        int nid = firing_times[i].neuron;   // neuron id
                        if (ft < display_start)
                            continue;
                        float x = rx + map((float)nid, 0.0, (float)grid.numCells, 0.0, rw);
                        float y = ry + ((float)(ft - display_start) / window_ms) * rh;
                        Color pc = neurons[nid].is_exc ? GREEN : DARKGREEN;
                        pc.a = 150;

                        float dot_radius = 0.0;
                        if (grid.numCells > grid.width) {
                            DrawPixel(x, y, pc);
                            dot_radius = 1.5;
                        } else {
                            DrawRectangle(x - 1, y - 1, 2, 2, pc);
                            dot_radius = 3.0;
                        }


                        if (nid == grid.selected_cell)
                            DrawCircle(x, y, 3, YELLOW);
                        if ((ft < t_ms) && (ft > t_ms - 10)) {
                            if (neurons[nid].is_exc ) {
                                DrawCircle(x, y, dot_radius, WHITE);
                            } else {
                                DrawCircle(x, y, dot_radius, RED);
                            }
                        }
                    }

                    /* v snapshot */
                    int vx = grid.margin;
                    int vy = ry + rh + grid.margin;
                    int vw = WIDTH - 2*grid.margin;
                    int vh = VTRACE_H;
                    DrawRectangleLines(vx-1, vy-1, vw+2, vh+2, LIGHTGRAY);
                    DrawText("v snapshot (sampled neurons)", vx+6, vy+6, 14, LIGHTGRAY);

                    /* sample M neurons across population */
                    int M = grid.numCells; //300;
                    if (M > grid.numCells)
                        M = grid.numCells;
                    int step = grid.numCells / M;
                    if (step < 1)
                        step = 1;
                    int idx = 0;
                    for (int i = 0; i < grid.numCells && idx < M; i += step, idx++) {
                        float vv = neurons[i].v;
                        float norm = (vv + 100.0f) / 140.0f;
                        if (norm < 0)
                            norm = 0;
                        if (norm > 1)
                            norm = 1;
                        int x = vx + (int)((float)idx / (float)M * vw);
                        int y = vy + 20 + (int)((1.0f - norm) * (vh - 40));
                        Color col = neurons[i].is_exc ? GREEN: DARKGREEN;

                        float dot_radius = 0.0;
                        if (grid.numCells > grid.width) {
                            DrawPixel(x, y, col);
                            dot_radius = 2;
                        } else {
                            DrawRectangle(x-1, y-1, 2, 2, col);
                            dot_radius = 3.0;
                        }
                        if (i == grid.selected_cell)
                            DrawCircle(x, y, dot_radius, YELLOW);
                    }

                    /* mean weight panel */
                    int px = grid.margin, py = vy + vh + grid.margin;
                    int pw = WIDTH - 2*grid.margin;
                    int ph = PANEL_H;
                    DrawRectangleLines(px-1, py-1, pw+2, ph+2, LIGHTGRAY);
                    float mw = mean_exc_weight(&grid);
                    char info[256];
                    snprintf(info, sizeof(info), "mean_exc_weight=%.3f  steps/frame=%d  %s  ||  I=%.1f  I_prob=%.3f  ||  t=%.1f ms", mw, steps_per_frame, paused ? "PAUSED" : "RUN", input_val, input_prob, t_ms);
                    DrawText(info, px+6, py+6, 14, WHITE);
                    /* draw bar */
                    float barw = (mw / 10.0f) * (pw - 40);
                    float bary = (float)py + (float)PANEL_H * 0.1 + 20.0;
                    float barh = (float)PANEL_H - (float)PANEL_H * 0.33 * 2.0 + 0.1;
                    DrawRectangle(
                            px+20,
                            bary,
                            (int)Clamp(barw, 0, pw-40),
                            barh,
                            GREEN);

                    /* selected neuron trace */
                    int sx = grid.margin;
                    int sy = py + ph + grid.margin;
                    int sw = WIDTH - 2*grid.margin;
                    int sh = SELECT_TRACE_H-10;
                    DrawRectangleLines(sx-1, sy-1, sw+2, sh+2, LIGHTGRAY);
                    draw_selected_trace(&grid, sx, sy, sw, sh);
                }
            }

            /* infos */
            int ry = grid.margin;
            int rh = RASTER_H;
            int vy = ry + rh + grid.margin;
            int vh = VTRACE_H;

            int px = grid.margin;
            int py = vy + vh + grid.margin;
            float mw = mean_exc_weight(&grid);
            char info[256];
            snprintf(info, sizeof(info), "mean_exc_weight=%.3f  steps/frame=%d  %s  ||  I=%.1f  I_prob=%.3f  ||  t=%.1f ms", mw, steps_per_frame, paused ? "PAUSED" : "RUN", input_val, input_prob, t_ms);
            DrawText(info, px+6, py+6, 14, WHITE);

            /* footer */
            DrawText(TextFormat("SPACE: pause/run   UP/DOWN: +/- speed   D: display   G: raster<>grid   R: reset   Click %s to select neuron", graphics_raster?"raster":"grid"), grid.margin+5, HEIGHT-15, 14, GRAY);

            if (show_fps)
                DrawFPS(10, 10);
        } EndDrawing();
    }

    free_network(&grid);
    arrfree(palette);

    CloseWindow();

    return 0;
}
