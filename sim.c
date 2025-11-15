#include <stdlib.h>
#include <math.h>

#include "stb_ds.h"

#include "math_utils.h"
#include "sim.h"

/* Initialize delay buckets */
static void init_delay_buckets(sim *s)
{
    if (arrlen(s->delaybuckets) != 0) {
        arrfree(s->delaybuckets);
    }
    s->delaybuckets = NULL;
    arrsetlen(s->delaybuckets, MAX_DELAY+1);
    for (int d = 0; d <= MAX_DELAY; d++) {
        s->delaybuckets[d].neuron = NULL;
        s->delaybuckets[d].weight = NULL;
        s->delaybuckets[d].count = 0;
        s->delaybuckets[d].cap = 0;
    }
    s->current_delay_index = 0;
}

/* Ensure capacity for bucket */
static void ensure_bucket_cap(sim *s, DelayBucket *db, int need)
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
static void bucket_push(sim *s, DelayBucket *db, int neuron, float weight)
{
    ensure_bucket_cap(s, db, db->count + 1);
    db->neuron[db->count] = neuron;
    db->weight[db->count] = weight;
    db->count++;
}

/* Pop all from bucket (used when delivering) */
static void bucket_clear(sim *s, DelayBucket *db)
{
    db->count = 0;
}

/* Initialize network (connections, weights, delays, v/u, buffers) */
void init_network(sim *s, Grid *grid)
{
    /* per-neuron v,u history index for selected trace (circular) */
    s->firing_cap = FIRING_BUF;

    s->input_prob = 0.056f;   // default synaptic input noise probability
    s->input_val = 24.0f;     // default synaptic input current

    // exc to inh ratio (from paper)
    s->exc_to_inh_ratio = 1.0/(4.0 + 1.0);
    s->num_exc = grid->numCells * s->exc_to_inh_ratio * 4.0; // number of exc neurons
    s->num_inh = grid->numCells * s->exc_to_inh_ratio * 1.0; // number of inh neurons

    /* flags: 1 = eccitatorio, 0 = inibitorio */
    uint8_t *flags = (uint8_t*)calloc(grid->numCells, sizeof(uint8_t));
    int *pool = (int*)malloc(grid->numCells * sizeof(int));
    for (int i = 0; i < grid->numCells; ++i)
        pool[i] = i;
    for (int k = 0; k < s->num_exc; ++k) {
        int r = rand() % (grid->numCells - k);
        flags[pool[r]] = 1;
        pool[r] = pool[grid->numCells - k - 1]; /* rimosso dallo pool */
    }
    free(pool);

    /* allocate */
    arrsetlen(s->neurons, grid->numCells);
    arrsetlen(s->firing_times, s->firing_cap);

    for (int i = 0; i < grid->numCells; i++) {
        s->neurons[i].cell_activity = 0.0f;
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

        s->neurons[i].is_exc = (uint8_t)flags[i];

        ra = rand01(); ra2 = ra * ra;
        s->neurons[i].a = s->neurons[i].is_exc ? 0.02f : 0.02f + 0.08f * ra;
        s->neurons[i].b = s->neurons[i].is_exc ? 0.2f : 0.25f - 0.05f * ra;
        s->neurons[i].c = s->neurons[i].is_exc ? -65.0f + 15.0f * ra2 : -65.0f;
        s->neurons[i].d = s->neurons[i].is_exc ? 8.0f - 6.0f * ra2 : 2.0f;

        s->neurons[i].v = s->neurons[i].c;
        s->neurons[i].u = s->neurons[i].b * s->neurons[i].v;
        s->neurons[i].last_spike_time = -1000000;
        for (int k = 0; k < VUBUF_LEN_MS; k++) {
            s->neurons[i].v_hist[k] = s->neurons[i].v;
            s->neurons[i].u_hist[k] = s->neurons[i].u;
        }
    }

    /* principale loop sui neuroni */
    for (int i = 0; i < grid->numCells; ++i) {
        int is_exc = s->neurons[i].is_exc;

        s->neurons[i].outconn.targets = NULL;
        s->neurons[i].outconn.weights = NULL;
        s->neurons[i].outconn.delay = NULL;
        s->neurons[i].num_targets = 0;
        s->neurons[i].num_near_conn = 0;
        s->neurons[i].num_far_conn = 0;
        s->neurons[i].target_center = (CellPos){0};

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
                arrput(s->neurons[i].outconn.targets, targ);
                if (is_exc) {
                    arrput(s->neurons[i].outconn.weights, 6.0f * frandf());
                } else {
                    arrput(s->neurons[i].outconn.weights, -5.0f * frandf());
                }
                int dsq = grid_toroidal_dist_sq(grid, this_cell, local_post[j]);
                int cell_dist = (int)floorf(sqrtf((float)dsq) + 0.5f);
                arrput(s->neurons[i].outconn.delay, compute_delay_from_cells(grid, cell_dist, vel_unmyelinated));
                s->neurons[i].num_near_conn ++;
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
                    s->neurons[i].target_center = center;

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
                        arrput(s->neurons[i].outconn.targets, targ);
                        arrput(s->neurons[i].outconn.weights, 6.0f * frandf());
                        int dsq = grid_toroidal_dist_sq(grid, center, far_post[jj]);
                        int cell_dist = (int)floorf(sqrtf((float)dsq) + 0.5f);
                        arrput(s->neurons[i].outconn.delay, axon_delay + compute_delay_from_cells(grid, cell_dist, vel_unmyelinated));
                        s->neurons[i].num_far_conn ++;
                    }
                    arrfree(far_post); far_post = NULL;

                    s->neurons[i].num_targets ++;
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

    init_delay_buckets(s);
    grid->selected_cell = -1;

    free(flags);
}

/* Free network memory */
void free_network(sim *s, Grid *grid)
{
    if (!s->neurons)
        return;
    for (int i = 0; i < grid->numCells; i++) {
        arrfree(s->neurons[i].outconn.targets);
        arrfree(s->neurons[i].outconn.weights);
        arrfree(s->neurons[i].outconn.delay);
    }
    arrfree(s->firing_times);
    for (int d = 0; d <= MAX_DELAY; d++) {
        arrfree(s->delaybuckets[d].neuron);
        arrfree(s->delaybuckets[d].weight);
    }
    arrfree(s->delaybuckets);
    arrfree(s->neurons);
    arrfree(s->order);
    arrfree(s->v_next);
    arrfree(s->u_next);
}

/* Add firing to circular buffer of pairs (time, neuron) */
static void add_firing_record(sim *s, float time_ms, int neuron)
{
    if (s->firing_count >= s->firing_cap) {
        /* simple downsample: shift keep half */
        int keep = s->firing_cap / 2;
        int start = (s->firing_count - keep);
        memmove(s->firing_times, s->firing_times + start, keep * sizeof(int));
        s->firing_count = keep;
    }
    s->firing_times[s->firing_count] = (FiringTime){ neuron, time_ms };
    s->firing_count++;
}

/* STDP weight update on spike: apply pair-based approximation
   When neuron 'pre' spikes at time t_pre, potentiate outgoing synapses to posts that spiked recently.
   When neuron 'post' spikes at time t_post, depress incoming excitatory synapses from pres that spiked recently.
   We'll implement updates at pre spike time on outgoing weights using last_spike_time[post].
*/
static void apply_stdp_on_pre(sim *s, int pre, float t_pre)
{
    int K = arrlen(s->neurons[pre].outconn.targets); /* only excitatory neurons have CE */
    if (!s->neurons[pre].is_exc) /* only excitatory synapses are plastic */
        return;
    for (int i = 0; i < K; i++) {
        int post = s->neurons[pre].outconn.targets[i];
        float t_post = s->neurons[post].last_spike_time;
        if (t_post <= -100000)
            continue;
        float dt = t_pre - t_post; /* positive if pre after post => depression */
        if (dt > 0 && dt < 1000) {
            /* pre after post -> LTD (A_minus), dt positive */
            float dw = -A_minus * expf(- dt / TAU_MINUS);
            s->neurons[pre].outconn.weights[i] += dw;
        } else {
            /* pre before post => potentiation handled when post spikes, to keep symmetry we handle both sides:
               We'll also handle LTP when pre precedes post by applying when post spikes (below). */
        }
        /* clamp */
        s->neurons[pre].outconn.weights[i] =
                clampf(s->neurons[pre].outconn.weights[i], W_MIN, W_MAX);
    }
}

/* Called when a neuron spikes (post), apply LTP for incoming excitatory synapses.
   We don't store incoming lists for memory reasons, so iterate all excitatory neurons and check if they connect to 'post' - costly but acceptable for moderate CE/NE.
   Optimization: only check outgoing from excitatory population.
*/
static void apply_stdp_on_post(sim *s, Grid *grid, int post, float t_post)
{
    /* For each excitatory neuron pre, check its outgoing connections for post */
    for (int pre = 0; pre < grid->numCells; pre++) {
        if (s->neurons[pre].is_exc) {
            int K = arrlen(s->neurons[pre].outconn.targets);
            for (int j = 0; j < K; j++) {
                if (s->neurons[pre].outconn.targets[j] != post)
                    continue;
                float t_pre = s->neurons[pre].last_spike_time;
                if (t_pre <= -100000)
                    continue;
                float dt = t_post - t_pre; /* positive if post after pre => potentiation */
                if (dt > 0 && dt < 1000) {
                    float dw = A_plus * expf(- (float)dt / TAU_PLUS);
                    s->neurons[pre].outconn.weights[j] += dw;
                    /* clamp */
                    s->neurons[pre].outconn.weights[j] = clampf(s->neurons[pre].outconn.weights[j], W_MIN, W_MAX);
                }
            }
        }
    }
}

/* Schedule a delivered event: place target in delay bucket for appropriate arrival time */
static void schedule_spike_delivery(sim *s, int pre, int conn_index)
{
    int post = s->neurons[pre].outconn.targets[conn_index];
    float w = s->neurons[pre].outconn.weights[conn_index];
    int delay = s->neurons[pre].outconn.delay[conn_index];
    if (delay < 1)
        delay = 1;
    if (delay > MAX_DELAY)
        delay = MAX_DELAY;
    int bucket_idx = (s->current_delay_index + delay) % (MAX_DELAY+1);
    /* note: using 0..MAX_DELAY buckets, but we never place into index 0 unless delay==0; safe since bucket array sized */
    bucket_push(s, &s->delaybuckets[bucket_idx], post, w);
}

/* Simulation single step (1 ms) */
void sim_step(sim *s, Grid *grid)
{
    /* 1) Deliver all events in the current bucket (arrivals scheduled for this ms) */
    DelayBucket *db = &s->delaybuckets[s->current_delay_index];
    /* produce an input array I for this ms */
    for (int i = 0; i < grid->numCells; i++)
        s->neurons[i].I = 0.0f;

    for (int k = 0; k < db->count; k++) {
        int neuron = db->neuron[k];
        float w = db->weight[k];
        s->neurons[neuron].I += w;
    }
    /* clear bucket for reuse (it will be filled for future times) */
    bucket_clear(s, db);

    /* 2) External noisy input: Poisson-like drive to excitatory neurons */
    for (int i = 0; i < grid->numCells; i++) {
        if (s->neurons[i].is_exc)
            if (frandf() < s->input_prob)
                s->neurons[i].I += s->input_val * frandf();
    }

    int num_sub_steps = 4;

#define INTEGRATOR_BIAS_CORRECTION 2
#if (INTEGRATOR_BIAS_CORRECTION == 0)
    /* 3) Integrate neuron dynamics (Izhikevich) — versione naive (drifting bias) */
    for (int step = 0; step < num_sub_steps; ++step) {
        for (int i = 0; i < grid->numCells; i++) {
            float dv = 0.04f * s->neurons[i].v * s->neurons[i].v + 5.0f * s->neurons[i].v + 140.0f - s->neurons[i].u + s->neurons[i].I;
            s->neurons[i].v += dv * (DT / (float)num_sub_steps);
            s->neurons[i].u += s->neurons[i].a * (s->neurons[i].b * s->neurons[i].v - s->neurons[i].u) * (DT / (float)num_sub_steps);
        }
    }
#elif (INTEGRATOR_BIAS_CORRECTION == 1)
    /* 3) Integrate: permuta deterministica dell'ordine */
    if (!s->order)
        arrsetlen(s->order, grid->numCells);
    /* inizializza ordine una volta (0..N-1) */
    for (int i = 0; i < grid->numCells; i++)
        s->order[i] = i;
    /* shuffle deterministico, ad es. xorshift con seed basato su t_ms */
    uint32_t seed = (uint32_t)(s->t_ms + 123456);
    for (int i = grid->numCells - 1; i > 0; --i) {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        uint32_t r = seed % (i + 1);
        int tmp = s->order[i];
        s->order[i] = s->order[r];
        s->order[r] = tmp;
    }

    /* due sottopassi: per ogni sottopasso aggiorna secondo order[] */
    for (int step = 0; step < num_sub_steps; ++step) {
        for (int idx = 0; idx < grid->numCells; ++idx) {
            int i = s->order[idx];
            float dv = 0.04f * s->neurons[i].v * s->neurons[i].v + 5.0f * s->neurons[i].v + 140.0f - s->neurons[i].u + s->neurons[i].I;
            s->neurons[i].v += dv * (DT / (float)num_sub_steps);
            s->neurons[i].u += s->neurons[i].a * (s->neurons[i].b * s->neurons[i].v - s->neurons[i].u) * (DT / (float)num_sub_steps);
        }
    }
#elif (INTEGRATOR_BIAS_CORRECTION == 2)
    /* 3) Integrate neuron dynamics (Izhikevich) — versione buffer */
    if (!s->v_next) {
        arrsetlen(s->v_next, grid->numCells);
        arrsetlen(s->u_next, grid->numCells);
    }
    /* inizializza con valori correnti (necessario per sottopassi) */
    for (int i = 0; i < grid->numCells; i++) {
        s->v_next[i] = s->neurons[i].v;
        s->u_next[i] = s->neurons[i].u;
    }

    for (int step = 0; step < num_sub_steps; ++step) {
        /* calcola nuovi v/u su base dei valori *correnti* in neurons[] */
        for (int i = 0; i < grid->numCells; i++) {
            float v_cur = s->neurons[i].v;
            float u_cur = s->neurons[i].u;
            float I = s->neurons[i].I;
            float dv = 0.04f * v_cur * v_cur + 5.0f * v_cur + 140.0f - u_cur + I;
            float v_new = v_cur + dv * (DT / (float)num_sub_steps);
            float u_new = u_cur + s->neurons[i].a * (s->neurons[i].b * v_cur - u_cur) * (DT / (float)num_sub_steps);
            s->v_next[i] = v_new;
            s->u_next[i] = u_new;
        }
        /* dopo aver calcolato tutti i nuovi valori, copia indietro (swap) */
        for (int i = 0; i < grid->numCells; i++) {
            s->neurons[i].v = s->v_next[i];
            s->neurons[i].u = s->u_next[i];
        }
    }
#endif

    /* 4) Check for spikes (v >= 30) */
    for (int i = 0; i < grid->numCells; i++) {
        if (s->neurons[i].v >= 30.0f) {
            /* record spike */
            add_firing_record(s, s->t_ms, i);
            /* STDP: apply pre-spike rule (depression for pre after recent post) */
            apply_stdp_on_pre(s, i, s->t_ms);
            /* reset */
            s->neurons[i].v = s->neurons[i].c;
            s->neurons[i].u += s->neurons[i].d;
            /* schedule deliveries to targets according to their delays */
            int K = arrlen(s->neurons[i].outconn.targets);
            for (int j = 0; j < K; j++) {
                schedule_spike_delivery(s, i, j);
            }
            /* STDP: handle post-spike LTP for incoming excitatory synapses */
            apply_stdp_on_post(s, grid, i, s->t_ms);
            /* update last spike time */
            s->neurons[i].last_spike_time = s->t_ms;
        }
    }

    /* 5) advance delay index and time, update v history buffer index */
    s->current_delay_index = (s->current_delay_index + 1) % (MAX_DELAY+1);
    s->t_ms += DT;
    s->vhist_idx = (s->vhist_idx + 1) % VUBUF_LEN_MS;
    s->uhist_idx = (s->uhist_idx + 1) % VUBUF_LEN_MS;
    for (int i = 0; i < grid->numCells; i++) {
        s->neurons[i].v_hist[s->vhist_idx] = s->neurons[i].v;
        s->neurons[i].u_hist[s->uhist_idx] = s->neurons[i].u;
    }
}
