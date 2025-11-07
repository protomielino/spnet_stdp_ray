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

#include "grid.h"
#include "palette.h"
#include "neuron.h"

ColourEntry *palette = NULL;

/* Window */
#define WIDTH 1360
#define HEIGHT 768

//#define CE 75  /* excitatory outgoing per neuron */
#define CI 25   /* inhibitory outgoing per neuron */

#define MAX_DELAY 20 /* ms */
#define DT 1         /* ms per sim step */

/* Visualization layout */
#define MARGIN 20

#define RASTER_H 350
#define VTRACE_H 110
#define PANEL_H 50
#define SELECT_TRACE_H 100

typedef struct
{
    int numCols;
    int numRows;
    int numCells; // this number needs to be == N == NE+NI
    int width; // pixel
    int height; // pixel
    float cellWidth; // pixel
    float cellHeight; // pixel
} Grid;

float rand01()
{
    return (float)rand() / (float)RAND_MAX;
}

uint f_randi(uint32_t index)
{
    index = (index << 13) ^ index;
    return ((index * (index * index * 15731 + 789221) + 1376312589) & 0x7fffffff);
}

/* Delay queues: for each delay (1..MAX_DELAY) maintain list of targets arriving after that many ms */
typedef struct
{
    int *neuron;   /* target neuron ids */
    float *weight; /* corresponding weights */
    int count;
    int cap;
} DelayBucket;

int grid_toroidal_dist_sq(Grid *grid, CellPos cell1, CellPos cell2)
{
    // distanza al quadrato considerando wrap-around (toroide) con metriche di griglia euclidea
    int dr = abs(cell1.r - cell2.r);
    int dc = abs(cell1.c - cell2.c);
    if (dr > grid->numRows/2) dr = grid->numRows - dr;
    if (dc > grid->numCols/2) dc = grid->numCols - dc;
    return dr*dr + dc*dc;
}

// Restituisce 1 se la cella (r,c) è in [rmin,rmax] (inclusi) rispetto a centro (rc,cc)
int grid_in_annulus(Grid *grid, CellPos cell, CellPos center_cell, int rmin, int rmax)
{
    int dsq = grid_toroidal_dist_sq(grid, cell, center_cell);
    return (dsq >= rmin*rmin && dsq <= rmax*rmax);
}

// Seleziona k celle casuali nell'anello [rmin,rmax] attorno al centro.
// Restituisce il numero effettivo di celle selezionate (<= k)
// out array deve avere capacità almeno k.
int grid_pick_random_cells_in_annulus(Grid *grid, CellPos center_cell, int rmin, int rmax, int k, CellPos *out)
{
    // Raccogli tutte le celle ammissibili
    CellPos *candidates = NULL;
    int cnt = 0;
    for (int r = 0; r < grid->numRows; r++) {
        for (int c = 0; c < grid->numCols; c++) {
            if (grid_in_annulus(grid, (CellPos){r, c}, center_cell, rmin, rmax)) {
                CellPos cand = {r, c};
                arrput(candidates, cand);
            }
        }
    }
    cnt = arrlen(candidates);
    if (cnt == 0) {
        arrfree(candidates);
        candidates = NULL;
        return 0;
    }

    // Se k >= cnt prendiamo tutti
    if (k >= cnt) {
        for (int i = 0; i < cnt; i++)
            out[i] = candidates[i];
        arrfree(candidates);
        candidates = NULL;
        return cnt;
    }

    // altrimenti fisher-yates partial shuffle per estrarre k elementi casuali senza ripetizioni
    for (int i = 0; i < k; i++) {
        int j = i + rand() % (cnt - i);
        // scambia i,j
        CellPos tmp = candidates[i];
        candidates[i] = candidates[j];
        candidates[j] = tmp;
        out[i] = candidates[i];
    }

    arrfree(candidates);
    candidates = NULL;

    return k;
}

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
    int neuron;
    int time_ms;
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

/* Selected neuron */
static int selected_neuron = -1;

/* Simulation time */
static int t_ms = 0;

float map(float input, float input_start, float input_end, float output_start, float output_end)
{
    float slope = 1.0 * (output_end - output_start) / (input_end - input_start);
    float output = output_start + slope * (input - input_start);
    return output;
}

/* Utility random */
static float frandf()
{
    return (float)rand() / (float)RAND_MAX;
}

/* Clamp helper */
static float clampf(float x, float a, float b)
{
    if (x < a)
        return a;
    if (x > b)
        return b;
    return x;
}

/* helper: convert index -> row,col and viceversa */
static inline CellPos grid_index_to_cellpos(Grid *grid, int idx, CellPos *cell)
{
    *cell = (CellPos){idx / grid->numCols, idx % grid->numCols};
    return *cell;
}
static inline int grid_cellpos_to_index(Grid *grid, CellPos cell)
{
    if (cell.r < 0)
        cell.r = (cell.r % grid->numRows + grid->numRows) % grid->numRows;
    if (cell.c < 0)
        cell.c = (cell.c % grid->numCols + grid->numCols) % grid->numCols;
    return cell.r * grid->numCols + cell.c;
}
/* helper: convert row,col -> x,y (cell's top left+MARGIN) */
static inline Vector2 grid_cellpos_to_vec2(Grid *grid, CellPos cp)
{
    Vector2 ret = {0};
    ret.x = MARGIN + cp.c * grid->cellWidth;
    ret.y = MARGIN + cp.r * grid->cellHeight;
    return ret;
}

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

typedef struct
{
    long long rows;
    long long cols;
} gridFactors;

gridFactors factors(long long num, double R)
{
    gridFactors ret = {0};

    if (num <= 0) {
        fprintf(stderr, "N deve essere positivo\n");
        return ret;
    }

    double best_diff = INFINITY;
    // Primo pass: trovare la differenza minima assoluta |(double)a/b - R|
    for (long long col = 1; col * col <= num; ++col) {
        if (num % col == 0) {
            long long row = num / col;
            double ratio1 = (double)col / (double)row;
            double diff1 = fabs(ratio1 - R);
            if (diff1 < best_diff)
                best_diff = diff1;

            // considerare anche la coppia invertita se diversa
            if (col != row) {
                double ratio2 = (double)row / (double)col;
                double diff2 = fabs(ratio2 - R);
                if (diff2 < best_diff)
                    best_diff = diff2;
            }
        }
    }

    // Seconda pass: stampare tutte le coppie che raggiungono best_diff (tolleranza per fp)
    const double eps = 1e-12;
    printf("Fattori di %lld con rapporto vicino a %.12g (diff minima = %.12g):\n", num, R, best_diff);
    for (long long col = 1; col * col <= num; ++col) {
        if (num % col == 0) {
            long long row = num / col;
            double ratio1 = (double)col / (double)row;
            double diff1 = fabs(ratio1 - R);
            if (fabs(diff1 - best_diff) <= eps) {
                printf("%lld x %lld  -> rapporto = %.12g\n", col, row, ratio1);
            }
            if (col != row) {
                double ratio2 = (double)row / (double)col;
                double diff2 = fabs(ratio2 - R);
                if (fabs(diff2 - best_diff) <= eps) {
                    printf("%lld x %lld  -> rapporto = %.12g\n", row, col, ratio2);
                }
            }
            ret.cols = row;
            ret.rows = col;
        }
    }

    return ret;
}

static float mm_per_cell(Grid *grid)
{
    const float R_mm = 8.0f;
    const float area_mm2 = 4.0f * 3.14159265358979323846f * R_mm * R_mm;
    const float mm2_per_cell = area_mm2 / (float)grid->numCells;
    return sqrtf(mm2_per_cell);
}

static float mm_to_cells_float(Grid *grid, float mm)
{
    return mm / mm_per_cell(grid);
}

static unsigned char compute_delay_from_cells(Grid *grid, int cell_dist, float velocity_m_per_s)
{
    float dist_mm = cell_dist * mm_per_cell(grid);
    float dist_m = dist_mm / 1000.0f;
    float d_ms = (dist_m / velocity_m_per_s) * 1000.0f;
    int di = (int)ceilf(d_ms);
    if (di < 1) di = 1;
    if (di > MAX_DELAY) di = MAX_DELAY;
    return (unsigned char)di;
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
        int cells_rmax_local = (int)ceilf(mm_to_cells_float(grid, local_mm)) + 1; // NOTE: this +1 should be removed for num neurons >= ~20.000
        int cells_rmin_local = 0;
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
            int collateral_rcells = (int)ceilf(mm_to_cells_float(grid, collateral_radius_mm)) + 1; // NOTE: this +1 should be removed for num neurons >= ~20.000

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
    t_ms = 0;
    vhist_idx = 0;
    selected_neuron = -1;

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
static void add_firing_record(int time_ms, int neuron)
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
static void apply_stdp_on_pre(int pre, int t_pre)
{
    int K = arrlen(neurons[pre].outconn.targets); /* only excitatory neurons have CE */
    if (!neurons[pre].is_exc) /* only excitatory synapses are plastic */
        return;
    for (int i = 0; i < K; i++) {
        int post = neurons[pre].outconn.targets[i];
        int t_post = neurons[post].last_spike_time;
        if (t_post <= -100000)
            continue;
        int dt = t_pre - t_post; /* positive if pre after post => depression */
        if (dt > 0 && dt < 1000) {
            /* pre after post -> LTD (A_minus), dt positive */
            float dw = -A_minus * expf(- (float)dt / TAU_MINUS);
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
static void apply_stdp_on_post(Grid *grid, int post, int t_post)
{
    /* For each excitatory neuron pre, check its outgoing connections for post */
    for (int pre = 0; pre < grid->numCells; pre++) {
        if (neurons[pre].is_exc) {
            int K = arrlen(neurons[pre].outconn.targets);
            for (int j = 0; j < K; j++) {
                if (neurons[pre].outconn.targets[j] != post)
                    continue;
                int t_pre = neurons[pre].last_spike_time;
                if (t_pre <= -100000)
                    continue;
                int dt = t_post - t_pre; /* positive if post after pre => potentiation */
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

static float input_prob = 0.01f;    // default synaptic input noise probability
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

    int num_sub_steps = 2;

#if 0
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
#endif
#if 1
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
#if 0
    /* 3) Integrate neuron dynamics (Izhikevich) — versione naive (drifting bias) */
    for (int step = 0; step < num_sub_steps; ++step) {
        for (int i = 0; i < grid->numCells; i++) {
            float dv = 0.04f * neurons[i].v * neurons[i].v + 5.0f * neurons[i].v + 140.0f - neurons[i].u + neurons[i].I;
            neurons[i].v += dv * (DT / (float)num_sub_steps);
            neurons[i].u += neurons[i].a * (neurons[i].b * neurons[i].v - neurons[i].u) * (DT / (float)num_sub_steps);
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

/* Find neuron by clicking raster: map x,y to time and neuron id */
static int cell_index_from_grid_click(Grid *grid, int click_x, int click_y, int rx, int ry, int rw, int rh)
{
    int rel_x = click_x - rx;
    int rel_y = click_y - ry;

    /* If click outside grid area return -1 */
    if (click_x <= rx || click_x >= rx+rw || click_y <= ry || click_y >= ry+rh)
        return -1;

    CellPos cell = {
            map(rel_y, 0, grid->height, 0, grid->numRows),
            map(rel_x, 0, grid->width,  0, grid->numCols)
    };

    int cell_idx = cell.r * grid->numCols + cell.c;

    if (cell_idx < 0)
        cell_idx = 0;
    if (cell_idx >= grid->numCells)
        cell_idx = grid->numCells-1;

    return cell_idx;
}

bool graphics_raster = true;
bool graphics_grid = false;

/* Draw selected neuron v(t) trace */
static void draw_selected_trace(int sx, int sy, int sw, int sh)
{
    if (selected_neuron < 0) {
        DrawText(TextFormat("No neuron selected. Click %s to select.", (graphics_raster==true)?"raster":"grid"), sx+10, sy+10, 14, GRAY);
        return;
    }
    char buf[128];
    snprintf(buf, sizeof(buf), "Neuron %d  v,u(t) last %d ms", selected_neuron, VUBUF_LEN_MS);
    DrawText(buf, sx+10, sy+10, 14, LIGHTGRAY);

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
        vv = neurons[selected_neuron].v_hist[pos];
        uu = neurons[selected_neuron].u_hist[pos];
        /* map vv (-100..40) to y */
        float vnorm = (vv + 100.0f) / 140.0f;
        float unorm = (uu + 100.0f) / 140.0f;
        vnorm = vnorm < 0 ? 0 : vnorm;
        vnorm = vnorm > 1 ? 1 : vnorm;
        unorm = unorm < 0 ? 0 : unorm;
        unorm = unorm > 1 ? 1 : unorm;
        float x = sx + 1 + ((float)k / (float)(VUBUF_LEN_MS-1) * (float)(sw-2));
        float vy = sy + 1 + 10 + ((1.0f - vnorm) * (sh - 20));
        float uy = sy + 1 + 10 + ((1.0f - unorm) * (sh - 20));
        Color vC = neurons[selected_neuron].is_exc ? GREEN : DARKGREEN;
        Color uC = neurons[selected_neuron].is_exc ? SKYBLUE : BLUE;
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
            int x = MARGIN + (float)c * grid->cellWidth;
            int y = MARGIN + (float)r * grid->cellHeight;
            DrawRectangle(x+1, y+1, grid->cellWidth - 1, grid->cellHeight - 1, col);
            if (idx == selected_neuron) {
                DrawRectangleLines(x, y, grid->cellWidth, grid->cellHeight, MAGENTA);
            }
        }
    }

    DrawRectangleLines(MARGIN, MARGIN, grid->width, grid->height, WHITE);

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

    int grid_width = (WIDTH - 2*MARGIN);
    int grid_height = (HEIGHT - 2*MARGIN);
    int num_cells = 1000;
    if (argc > 1)
        num_cells = atoi(argv[1]);

    gridFactors gf = factors(num_cells, (double)grid_width/(double)grid_height);;
    Grid grid = {
            .numCols = gf.cols,
            .numRows = gf.rows,
            .numCells = (gf.cols*gf.rows),
            .width = grid_width,
            .height = grid_height,
            .cellWidth = ((float)grid_width / (float)gf.cols),
            .cellHeight = ((float)grid_height / (float)gf.rows)
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
            input_prob -= 0.01;
        }
        if (IsKeyPressed(KEY_F2)) {
            input_prob += 0.01;
        }
        if (IsKeyPressed(KEY_F3)) {
            input_val -= 1.0;
        }
        if (IsKeyPressed(KEY_F4)) {
            input_val += 1.0;
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

        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
            if (graphics_raster) {
                /* raster area coords */
                int rx = MARGIN;
                int ry = MARGIN;
                int rw = WIDTH - 2*MARGIN;
                int rh = RASTER_H;

                int nid = neuron_from_raster_click(&grid, mx, my, rx, ry, rw, rh);
                selected_neuron = (nid >= 0) ? nid : -1;
            }
            if (graphics_grid) {
                /* grid area coords */
                int rx = MARGIN;
                int ry = MARGIN;
                int rw = grid.width;
                int rh = grid.height;

                int nid = cell_index_from_grid_click(&grid, mx, my, rx, ry, rw, rh);
                selected_neuron = (nid >= 0) ? nid : -1;
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
//                    for (int ni = 0; ni < grid->numCells; ++ni) {
//                        CellPos cell_rc = {0};
//                        cell_rc = index_to_cellpos(ni, &cell_rc.r, &cell_rc.c);
//                        if (in_annulus(cell_rc.r, cell_rc.c, center, rmin, rmax)) {
//                            Vector2 cell_xy = cellpos_to_vec2(cell_rc);
//                            DrawRectangleLines(cell_xy.x+1, cell_xy.y+1, grid->cellWidth - 2, grid->cellHeight - 2, (Color){200, 230, 255, 255});
//                        }
//                    }
//
//                    for (int i = 0; i < selected_count; i++) {
//                        // selected cell
//                        CellPos cp = {0};
//                        cp.r = selected[i].r;
//                        cp.c = selected[i].c;
//                        Vector2 xy = cellpos_to_vec2(cp);
//                        DrawRectangleLines(xy.x+2, xy.y+2, grid->cellWidth-4, grid->cellHeight-4, (Color){ 253, 249, 0, 255 });
//                    }

                    if (selected_neuron >= 0) {
                        /* selected neuron trace */
                        int sx;
                        int sy = my + MARGIN/2;
                        int sw;
                        int sh = SELECT_TRACE_H-10;
                        if (mx < WIDTH/2) {
                            sx = mx + MARGIN;
                            sw = WIDTH - 2*MARGIN - mx;
                        } else {
                            sx = MARGIN;
                            sw = mx - 2*MARGIN;
                        }

                        draw_selected_trace(sx, sy, sw, sh);
                    }

                    if (selected_neuron >= 0) {
                        /* only excitatory neurons have CE */
                        int K_near = neurons[selected_neuron].num_near_conn;
                        for (int i = 0; i < K_near; ++i) {
                            int post = neurons[selected_neuron].outconn.targets[i];
                            CellPos target_pos = {0};
                            CellPos post_pos = {0};
                            target_pos = grid_index_to_cellpos(&grid, selected_neuron, &target_pos);
                            post_pos = grid_index_to_cellpos(&grid, post, &post_pos);
                            Vector2 target_pos_xy = grid_cellpos_to_vec2(&grid, target_pos);
                            Vector2 post_pos_xy = grid_cellpos_to_vec2(&grid, post_pos);
                            target_pos_xy = Vector2Add(target_pos_xy, (Vector2){ grid.cellWidth/2, grid.cellHeight/2 });
                            post_pos_xy = Vector2Add(post_pos_xy, (Vector2){ grid.cellWidth/2, grid.cellHeight/2 });
                            DrawLineV(target_pos_xy, post_pos_xy, (Color){255, 255, 255, 64});
                        }
                        if (neurons[selected_neuron].num_targets > 0) {
                            int post = grid_cellpos_to_index(&grid, neurons[selected_neuron].target_center);;
                            CellPos target_pos = {0};
                            CellPos post_pos = {0};
                            target_pos = grid_index_to_cellpos(&grid, selected_neuron, &target_pos);
                            post_pos = grid_index_to_cellpos(&grid, post, &post_pos);
                            Vector2 target_pos_xy = grid_cellpos_to_vec2(&grid, target_pos);
                            Vector2 post_pos_xy = grid_cellpos_to_vec2(&grid, post_pos);
                            target_pos_xy = Vector2Add(target_pos_xy, (Vector2){ grid.cellWidth/2, grid.cellHeight/2 });
                            post_pos_xy = Vector2Add(post_pos_xy, (Vector2){ grid.cellWidth/2, grid.cellHeight/2 });
                            DrawLineV(target_pos_xy, post_pos_xy, (Color){128, 255, 128, 64});
                        }
                        int K_far = neurons[selected_neuron].num_far_conn;
                        for (int i = K_near; i < K_far+K_near; ++i) {
                            int post = neurons[selected_neuron].outconn.targets[i];
                            CellPos target_pos = neurons[selected_neuron].target_center;
                            CellPos post_pos = {0};
                            post_pos = grid_index_to_cellpos(&grid, post, &post_pos);
                            Vector2 target_pos_xy = grid_cellpos_to_vec2(&grid, target_pos);
                            Vector2 post_pos_xy = grid_cellpos_to_vec2(&grid, post_pos);
                            target_pos_xy = Vector2Add(target_pos_xy, (Vector2){ grid.cellWidth/2, grid.cellHeight/2 });
                            post_pos_xy = Vector2Add(post_pos_xy, (Vector2){ grid.cellWidth/2, grid.cellHeight/2 });
                            DrawLineV(target_pos_xy, post_pos_xy, (Color){255, 255, 255, 128});
                        }

//                        for (int i = 0; i < K; ++i) {
//                            int post = neurons[selected_neuron].outconn.targets[i];
//                            CellPos target_pos = {0};
//                            CellPos post_pos = {0};
//                            target_pos = grid_index_to_cellpos(&grid, selected_neuron, &target_pos);
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
                    int rx = MARGIN;
                    int ry = MARGIN;
                    int rw = WIDTH - 2*MARGIN;
                    int rh = RASTER_H;
                    DrawRectangleLines(rx-1, ry-1, rw+2, rh+2, LIGHTGRAY);
                    DrawText("Raster (last 2000 ms)", rx+6, ry+6, 14, LIGHTGRAY);

                    /* draw spikes in last 2000 ms */
                    int window_ms = 2000;
                    int display_start = t_ms - window_ms;
                    if (display_start < 0)
                        display_start = 0;
                    for (int i = 0; i < firing_count; i++) {
                        int ft = firing_times[i].time_ms;   // firing time
                        int nid = firing_times[i].neuron;   // neuron id
                        if (ft < display_start)
                            continue;
                        float x = rx + map((float)nid, 0.0, (float)grid.numCells, 0.0, rw);
                        float y = ry + ((float)(ft - display_start) / window_ms) * rh;
                        Color pc = neurons[nid].is_exc ? GREEN : DARKGREEN;
                        pc.a = 150;
                        DrawRectangle((int)x-1, (int)y-1, 2, 2, pc);
                        if (nid == selected_neuron)
                            DrawCircle((int)x, (int)y, 3, YELLOW);
                        if ((ft < t_ms) && (ft > t_ms - 10)) {
                            if (neurons[nid].is_exc ) {
                                DrawCircle((int)x, (int)y, 3, WHITE);
                            } else {
                                DrawCircle((int)x, (int)y, 3, RED);
                            }
                        }
                    }

                    /* v snapshot */
                    int vx = MARGIN;
                    int vy = ry + rh + MARGIN;
                    int vw = WIDTH - 2*MARGIN;
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
                        DrawRectangle(x-1, y-1, 2, 2, col);
                        if (i == selected_neuron)
                            DrawCircle(x, y, 3, YELLOW);
                    }

                    /* mean weight panel */
                    int px = MARGIN, py = vy + vh + MARGIN;
                    int pw = WIDTH - 2*MARGIN;
                    int ph = PANEL_H;
                    DrawRectangleLines(px-1, py-1, pw+2, ph+2, LIGHTGRAY);
                    float mw = mean_exc_weight(&grid);
                    char info[256];
                    snprintf(info, sizeof(info), "t=%d ms  mean_exc_weight=%.3f  steps/frame=%d  %s", t_ms, mw, steps_per_frame, paused ? "PAUSED" : "RUN");
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
                    int sx = MARGIN;
                    int sy = py + ph + MARGIN;
                    int sw = WIDTH - 2*MARGIN;
                    int sh = SELECT_TRACE_H-10;
                    DrawRectangleLines(sx-1, sy-1, sw+2, sh+2, LIGHTGRAY);
                    draw_selected_trace(sx, sy, sw, sh);
                }
            }

            /* infos */
            int ry = MARGIN;
            int rh = RASTER_H;
            int vy = ry + rh + MARGIN;
            int vh = VTRACE_H;

            int px = MARGIN;
            int py = vy + vh + MARGIN;
            float mw = mean_exc_weight(&grid);
            char info[256];
            snprintf(info, sizeof(info), "t=%d ms  mean_exc_weight=%.3f  steps/frame=%d  %s  ||  I=%.1f  I_prob=%.2f", t_ms, mw, steps_per_frame, paused ? "PAUSED" : "RUN", input_val, input_prob);
            DrawText(info, px+6, py+6, 14, WHITE);

            /* footer */
            DrawText(TextFormat("SPACE: pause/run   UP/DOWN: +/- speed   D: display   G: raster<>grid   R: reset   Click %s to select neuron", graphics_raster?"raster":"grid"), MARGIN+5, HEIGHT-15, 14, GRAY);

            if (show_fps)
                DrawFPS(10, 10);
        } EndDrawing();
    }

    free_network(&grid);
    arrfree(palette);

    CloseWindow();

    return 0;
}
