/*
 spnet_ray_stdp.c
 C single-threaded simulation of an Izhikevich spiking network with delays and STDP,
 plus realtime visualization using raylib.

 Features:
 - NE=800 excitatory, NI=200 inhibitory (default from user).
 - Delays for excitatory synapses (1..20 ms). Inhibitory delay = 1 ms.
 - STDP on excitatory synapses (pair-based approximation), weight bounds.
 - Spike delivery via delay queues.
 - Interactive selection of a neuron by clicking the raster; shows v(t) trace for that neuron.
 - Single-threaded: simulation steps performed inside render loop. Controls:
     SPACE: pause/run
     UP/DOWN: increase/decrease steps/frame
     R: reset network
     Left click raster area: select neuron
 - Visualization:
     Raster (last 1000 ms), v snapshot, mean excitatory weight bar, v(t) trace for selected neuron.
 - Reasonable memory/layout for realtime interactivity.

 Build:
  - With CMake (assumes raylibConfig.cmake available):
      cmake -S . -B build
      cmake --build build
  - Or with gcc + pkg-config:
      gcc spnet_ray_stdp.c -o spnet_ray_stdp `pkg-config --cflags --libs raylib` -lm

 Notes:
  - Simplifications compared to full Izhikevich reference: STDP uses pair-based exponential windows,
    no homeostatic scaling, weight updates applied at spike times to outgoing excitatory synapses.
  - Performance tuned for moderate network size; reduce NE/CI/CE if too slow.

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <raylib.h>
#define RAYMATH_IMPLEMENTATION
#include <raymath.h>

/* Window */
#define WIDTH 1050
#define HEIGHT 700

/* Network params */
#define NE 800
#define NI 200
#define N (NE+NI)

#define CE 100  /* excitatory outgoing per neuron */
#define CI 25   /* inhibitory outgoing per neuron */

#define MAX_DELAY 20 /* ms */
#define DT 1         /* ms per sim step */

/* Visualization layout */
#define MARGIN 20

#define RASTER_H 350
#define VTRACE_H 110
#define PANEL_H 50
#define SELECT_TRACE_H 100

// visualization
static float *cell_activity; // length CELLS

#define GRID_COLS 40
#define GRID_ROWS 25
#define CELLS (GRID_COLS*GRID_ROWS)

#define GRID_W (WIDTH - 2*MARGIN)
#define GRID_H (HEIGHT - 2*MARGIN)
const float CELL_W = ((float)GRID_W / (float)GRID_COLS);
const float CELL_H = ((float)GRID_H / (float)GRID_ROWS);

/* Raster storage */
#define FIRING_BUF 2000000 /* pair (time, neuron) capacity */

/* Delay queues: for each delay (1..MAX_DELAY) maintain list of targets arriving after that many ms */
typedef struct
{
    int *neuron;   /* target neuron ids */
    float *weight; /* corresponding weights */
    int count;
    int cap;
} DelayBucket;

/* Outgoing connections structure */
typedef struct
{
    int *targets;            /* CE or CI targets */
    float *weights;          /* corresponding weights */
    unsigned char *delay;    /* delays in ms (for excitatory) */
} OutConn;

/* STDP parameters (pair-based) */
#define A_plus 0.1f
#define A_minus 0.12f
#define TAU_PLUS 20.0f
#define TAU_MINUS 20.0f
#define W_MIN 0.0f
#define W_MAX 10.0f

/* Globals */
static float *v;
static float *u;
static OutConn *outconn;                        /* size N */
static DelayBucket delaybuckets[MAX_DELAY+1];   /* index 1..MAX_DELAY used; 0 unused */
static int current_delay_index = 0;             /* rotates every ms */
static int *firing_times;                         /* circular buffer of pairs (time, neuron) */
static int firing_count = 0;
static int firing_cap = FIRING_BUF;

/* per-neuron last spike time for STDP (ms), initialize to very negative */
static int *last_spike_time;

/* per-neuron v history buffer for selected trace (circular) */
#define VBUF_MS 2000
static float (*v_hist)[VBUF_MS]; /* v_hist[neuron][idx] */
static int vhist_idx = 0;

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
static float frandf(void)
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

/* Initialize delay buckets */
static void init_delay_buckets(void)
{
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
    db->neuron = (int*)realloc(db->neuron, newcap * sizeof(int));
    db->weight = (float*)realloc(db->weight, newcap * sizeof(float));
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

/* Initialize network (connections, weights, delays, v/u, buffers) */
static void init_network(void)
{
    /* allocate */
    v = (float*)malloc(N * sizeof(float));
    u = (float*)malloc(N * sizeof(float));
    outconn = (OutConn*)malloc(N * sizeof(OutConn));
    firing_times = (int*)malloc(firing_cap * 2 * sizeof(int));
    last_spike_time = (int*)malloc(N * sizeof(int));
    v_hist = (float(*)[VBUF_MS])malloc(N * VBUF_MS * sizeof(float));

    cell_activity = (float*)calloc(CELLS, sizeof(float));

    /* init neurons */
    for (int i = 0; i < N;i++) {
        float r = frandf()*frandf();
        v[i] = -65.0f + 15.0f * r;
        if (i < NE)
            u[i] = 0.2f * v[i];
        else
            u[i] = 0.2f * v[i];
        last_spike_time[i] = -1000000;
        for (int k = 0; k < VBUF_MS; k++)
            v_hist[i][k] = v[i];
    }

    /* init connections */
    for (int i = 0; i < N;i++) {
        int K = (i < NE) ? CE : CI;
        outconn[i].targets = (int*)malloc(K * sizeof(int));
        outconn[i].weights = (float*)malloc(K * sizeof(float));
        outconn[i].delay = (unsigned char*)malloc(K * sizeof(unsigned char));
        for (int j = 0; j < K; j++) {
            int t = rand() % N;
            outconn[i].targets[j] = t;
            if (i < NE) {
                /* excitatory initial weight random around 6.0 +- */
                outconn[i].weights[j] = 6.0f * frandf();
                /* excitatory delay 1..MAX_DELAY */
                outconn[i].delay[j] = (unsigned char)(1 + (rand() % MAX_DELAY));
            } else {
                /* inhibitory negative weight */
                outconn[i].weights[j] = -5.0f * frandf();
                outconn[i].delay[j] = 1; /* inhibitory delay 1 ms */
            }
        }
    }

    init_delay_buckets();
    firing_count = 0;
    t_ms = 0;
    vhist_idx = 0;
    selected_neuron = -1;
}

/* Free network memory */
static void free_network(void)
{
    if (!v)
        return;
    for (int i = 0; i < N; i++) {
        free(outconn[i].targets);
        free(outconn[i].weights);
        free(outconn[i].delay);
    }
    free(outconn);
    free(v);
    free(u);
    free(firing_times);
    free(last_spike_time);
    free(v_hist);
    for (int d = 0; d <= MAX_DELAY; d++) {
        free(delaybuckets[d].neuron);
        free(delaybuckets[d].weight);
    }

    free(cell_activity);
}

/* Add firing to circular buffer of pairs (time, neuron) */
static void add_firing_record(int time_ms, int neuron)
{
    if (firing_count >= firing_cap) {
        /* simple downsample: shift keep half */
        int keep = firing_cap / 2;
        int start = (firing_count - keep) * 2;
        memmove(firing_times, firing_times + start, keep * 2 * sizeof(int));
        firing_count = keep;
    }
    firing_times[firing_count*2 + 0] = time_ms;
    firing_times[firing_count*2 + 1] = neuron;
    firing_count++;
}

/* STDP weight update on spike: apply pair-based approximation
   When neuron 'pre' spikes at time t_pre, potentiate outgoing synapses to posts that spiked recently.
   When neuron 'post' spikes at time t_post, depress incoming excitatory synapses from pres that spiked recently.
   We'll implement updates at pre spike time on outgoing weights using last_spike_time[post].
*/
static void apply_stdp_on_pre(int pre, int t_pre)
{
    int K = CE; /* only excitatory neurons have CE */
    if (pre >= NE) /* only excitatory synapses are plastic */
        return;
    for (int j = 0; j < K; j++) {
        int post = outconn[pre].targets[j];
        int t_post = last_spike_time[post];
        if (t_post <= -100000)
            continue;
        int dt = t_pre - t_post; /* positive if pre after post => depression */
        if (dt > 0 && dt < 1000) {
            /* pre after post -> LTD (A_minus), dt positive */
            float dw = -A_minus * expf(- (float)dt / TAU_MINUS);
            outconn[pre].weights[j] += dw;
        } else {
            /* pre before post => potentiation handled when post spikes, to keep symmetry we handle both sides:
               We'll also handle LTP when pre precedes post by applying when post spikes (below). */
        }
        /* clamp */
        outconn[pre].weights[j] = clampf(outconn[pre].weights[j], W_MIN, W_MAX);
    }
}

/* Called when a neuron spikes (post), apply LTP for incoming excitatory synapses.
   We don't store incoming lists for memory reasons, so iterate all excitatory neurons and check if they connect to 'post' - costly but acceptable for moderate CE/NE.
   Optimization: only check outgoing from excitatory population.
*/
static void apply_stdp_on_post(int post, int t_post)
{
    /* For each excitatory neuron pre, check its outgoing connections for post */
    for (int pre = 0; pre < NE; pre++) {
        int K = CE;
        for (int j = 0; j < K; j++) {
            if (outconn[pre].targets[j] != post)
                continue;
            int t_pre = last_spike_time[pre];
            if (t_pre <= -100000)
                continue;
            int dt = t_post - t_pre; /* positive if post after pre => potentiation */
            if (dt > 0 && dt < 1000) {
                float dw = A_plus * expf(- (float)dt / TAU_PLUS);
                outconn[pre].weights[j] += dw;
                /* clamp */
                outconn[pre].weights[j] = clampf(outconn[pre].weights[j], W_MIN, W_MAX);
            }
        }
    }
}

/* Schedule a delivered event: place target in delay bucket for appropriate arrival time */
static void schedule_spike_delivery(int pre, int conn_index)
{
    int post = outconn[pre].targets[conn_index];
    float w = outconn[pre].weights[conn_index];
    int delay = outconn[pre].delay[conn_index];
    if (delay < 1)
        delay = 1;
    if (delay > MAX_DELAY)
        delay = MAX_DELAY;
    int bucket_idx = (current_delay_index + delay) % (MAX_DELAY+1);
    /* note: using 0..MAX_DELAY buckets, but we never place into index 0 unless delay==0; safe since bucket array sized */
    bucket_push(&delaybuckets[bucket_idx], post, w);
}

/* Simulation single step (1 ms) */
static void sim_step(void)
{
    /* 1) Deliver all events in the current bucket (arrivals scheduled for this ms) */
    DelayBucket *db = &delaybuckets[current_delay_index];
    /* produce an input array I for this ms */
    static float I[N];
    for (int i = 0; i < N; i++)
        I[i] = 0.0f;

    for (int k = 0; k < db->count; k++) {
        int neuron = db->neuron[k];
        float w = db->weight[k];
        I[neuron] += w;
    }
    /* clear bucket for reuse (it will be filled for future times) */
    bucket_clear(db);

    /* 2) External noisy input: Poisson-like drive to excitatory neurons */
    for (int i = 0; i < NE; i++) {
        if (frandf() < 0.01f)
            I[i] += 20.0f * frandf();
    }

    /* 3) Integrate neuron dynamics (Izhikevich) */
    for (int i = 0; i < N; i++) {
        float a = (i < NE) ? 0.02f : 0.1f;
        float b = 0.2f;
//        float c = (i < NE) ? -65.0f : -65.0f;
//        float d = (i < NE) ? 8.0f : 2.0f;
        float dv = 0.04f * v[i] * v[i] + 5.0f * v[i] + 140.0f - u[i] + I[i];
        v[i] += dv * (DT / 1.0f);
        u[i] += a * (b * v[i] - u[i]) * (DT / 1.0f);
    }

    /* 4) Check for spikes (v >= 30) */
    for (int i = 0; i < N; i++) {
        if (v[i] >= 30.0f) {
            /* record spike */
            add_firing_record(t_ms, i);
            /* STDP: apply pre-spike rule (depression for pre after recent post) */
            apply_stdp_on_pre(i, t_ms);
            /* reset */
            float d = (i < NE) ? 8.0f : 2.0f;
            float c = (i < NE) ? -65.0f : -65.0f;
            v[i] = c;
            u[i] += d;
            /* schedule deliveries to targets according to their delays */
            int K = (i < NE) ? CE : CI;
            for (int j = 0; j < K; j++) {
                schedule_spike_delivery(i, j);
            }
            /* STDP: handle post-spike LTP for incoming excitatory synapses */
            apply_stdp_on_post(i, t_ms);
            /* update last spike time */
            last_spike_time[i] = t_ms;
        }
    }

    /* 5) advance delay index and time, update v history buffer index */
    current_delay_index = (current_delay_index + 1) % (MAX_DELAY+1);
    t_ms += DT;
    vhist_idx = (vhist_idx + 1) % VBUF_MS;
    for (int i = 0; i < N; i++)
        v_hist[i][vhist_idx] = v[i];
}

/* compute mean excitatory weight */
static float mean_exc_weight(void) {
    double s = 0.0;
    long cnt = 0;
    for (int i = 0; i < NE; i++) {
        for (int j = 0; j < CE; j++) {
            s += outconn[i].weights[j];
            cnt++;
        }
    }
    if (cnt == 0)
        return 0.0f;
    return (float)(s / cnt);
}

/* Find neuron by clicking raster: map x,y to time and neuron id */
static int neuron_from_raster_click(int click_x, int click_y, int rx, int ry, int rw, int rh)
{
    /* If click outside raster area return -1 */
    if (click_x < rx || click_x > rx+rw || click_y < ry || click_y > ry+rh)
        return -1;
    /* x -> neuron id: left excitatory, right inhibitory */
    int rely = click_y - ry;
    if (rely < 24) /* header area */
        return -1;
    int nid = -rx + map(click_x, 0.0, rw, 0.0, (float)N);
    /* excitatory map */
    if (nid < 0)
        nid = 0;
    if (nid >= N)
        nid = N-1;
    return nid;
}

/* Find neuron by clicking raster: map x,y to time and neuron id */
static int neuron_from_grid_click(int click_x, int click_y, int rx, int ry, int rw, int rh)
{
    int rel_x = click_x - MARGIN;
    int rel_y = click_y - MARGIN;

    /* If click outside grid area return -1 */
    if (click_x < rx || click_x > rx+rw || click_y < ry || click_y > ry+rh)
        return -1;

    int nidx = map(rel_x, 0, GRID_W, 0, GRID_COLS);
    int nidy = map(rel_y, 0, GRID_H, 0, GRID_ROWS);;

    int nid = nidy * GRID_COLS + nidx;

    if (nid < 0) nid = 0;
    if (nid >= N) nid = N-1;

    return nid;
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
    snprintf(buf, sizeof(buf), "Neuron %d  v(t) last %d ms", selected_neuron, VBUF_MS);
    DrawText(buf, sx+10, sy+10, 14, LIGHTGRAY);

    /* draw axes */
    DrawRectangleLines(sx, sy, sw, sh, LIGHTGRAY);

    /* find time window: show last VBUF_MS ms */
    int idx = vhist_idx;
    int px_prev = -1, py_prev = -1;
    for (int k = 0; k < VBUF_MS; ++k) {
        int pos = (idx - (VBUF_MS-1) + k);
        while (pos < 0)
            pos += VBUF_MS;
        pos %= VBUF_MS;
        float vv = v_hist[selected_neuron][pos];
        /* map vv (-100..40) to y */
        float norm = (vv + 100.0f) / 140.0f;
        if (norm < 0)
            norm = 0;
        if (norm > 1)
            norm = 1;
        int x = sx + 1 + (int)((float)k / (float)(VBUF_MS-1) * (sw-2));
        int y = sy + 1 + 10 + (int)((1.0f - norm) * (sh - 20));
//        float ci = map(x, sx, sx + sw, 0.0, 255.0);
        Color C = selected_neuron < NE ? WHITE : BLUE;
//        C.a = (int)ci;
        if (k > 0) {
            DrawLine(px_prev, py_prev, x, y, C);
        }
        px_prev = x; py_prev = y;
    }
}

// compute cell activities (using interleaved mapping)
void compute_cell_activity(void)
{
    // zero
    for (int k = 0; k < CELLS; k++)
        cell_activity[k] = 0.0f;

    // accumulate contributions per neuron into its cell
    for (int i = 0; i < N; i++) {
        int cell = i;
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
        float vdep = v[i] + 65.0f;
        if (vdep > 0.0f)
            metric += 0.02f * vdep;
        cell_activity[cell] += metric;
    }
    // normalize
    for (int k = 0; k < CELLS; k++) {
        // normalization scale empirical
        float val = cell_activity[k]; // TESTING / 20.0f;
        if (val > 1.0f)
            val = 1.0f;
        cell_activity[k] = val;
    }
}

void show_grid()
{
    for (int r = 0; r < GRID_ROWS; r++) {
        for (int c = 0; c < GRID_COLS; c++) {
            int idx = r*GRID_COLS + c;
            float val = cell_activity[idx]; // 0..1
            // color map: blue -> cyan -> green -> yellow -> red
            Color col;
            if (val <= 0.2f) {
                float t = val / 0.2f;
                col.r = (unsigned char)(0 + t * 0);
                col.g = (unsigned char)(0 + t * 180);
                col.b = (unsigned char)(32 + t * 135);
                col.a = 255;
                if (idx >= NE) {
                    unsigned char tmp = col.r;
                    col.r = col.b;
                    col.b = tmp;
                }
            } else if (val <= 0.4f) {
                float t = (val - 0.2f) / 0.2f;
                col.r = (unsigned char)(0 + t * 0);
                col.g = (unsigned char)(180 + t * 75);
                col.b = (unsigned char)(255 - t * 255);
                col.a = 255;
            } else if (val <= 0.6f) {
                float t = (val - 0.4f) / 0.2f;
                col.r = (unsigned char)(0 + t * 200);
                col.g = (unsigned char)(255 - t * 55);
                col.b = (unsigned char)(0 + t * 0);
                col.a = 255;
            } else if (val <= 0.8f) {
                float t = (val - 0.6f) / 0.2f;
                col.r = (unsigned char)(200 + t * 55);
                col.g = (unsigned char)(200 - t * 150);
                col.b = 0;
                col.a = 255;
            } else {
                float t = (val - 0.8f) / 0.2f;
                col.r = (unsigned char)(255);
                col.g = (unsigned char)(50 + t * 0);
                col.b = 0;
                col.a = 255;
            }
            int x = MARGIN + c * CELL_W;
            int y = MARGIN + r * CELL_H;
            DrawRectangle(x+1, y+1, CELL_W - 1, CELL_H - 1, col);
            if (idx == selected_neuron) {
                DrawRectangleLines(x, y, CELL_W, CELL_H, MAGENTA);
            }
        }
    }

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
int main(void)
{
    srand((unsigned)time(NULL));

    InitWindow(WIDTH, HEIGHT, "spnet_ray_stdp - Izhikevich + STDP (C + raylib)");
    SetTargetFPS(30);

    init_network();

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

        /* input */
        if (IsKeyPressed(KEY_G)) {
            graphics_raster = !graphics_raster;
            graphics_grid = !graphics_grid;
        }
        if (IsKeyPressed(KEY_F))
            show_fps = !show_fps;
        if (IsKeyPressed(KEY_D))
            show_graphics = !show_graphics;
        if (IsKeyPressed(KEY_SPACE))
            paused = !paused;
        if (IsKeyPressed(KEY_UP))
            steps_per_frame = Clamp(steps_per_frame+1, 1, 5000);
        if (IsKeyPressed(KEY_DOWN))
            steps_per_frame = Clamp(steps_per_frame-1, 1, 5000);
        if (IsKeyPressed(KEY_R)) {
            free_network();
            init_network();
        }
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
            if (graphics_raster) {
                /* raster area coords */
                int rx = MARGIN;
                int ry = MARGIN;
                int rw = WIDTH - 2*MARGIN;
                int rh = RASTER_H;

                int nid = neuron_from_raster_click(mx, my, rx, ry, rw, rh);
                selected_neuron = (nid >= 0) ? nid : -1;
            }
            if (graphics_grid) {
                /* grid area coords */
                int rx = MARGIN;
                int ry = MARGIN;
                int rw = GRID_W;
                int rh = GRID_H;

                int nid = neuron_from_grid_click(mx, my, rx, ry, rw, rh);
                selected_neuron = (nid >= 0) ? nid : -1;
            }
        }

        /* simulate */
        if (!paused)
            for (int s = 0; s < steps_per_frame; s++)
                sim_step();

        /* draw */
        BeginDrawing(); {
            ClearBackground(BLACK);
            if (show_graphics) {
                if (graphics_grid) {
                    // compute activities for visualization
                    compute_cell_activity();

                    show_grid();

                    if (selected_neuron >= 0) {
                        /* selected neuron trace */
                        int sx;
                        int sy = my + MARGIN/2;
                        int sw;
                        int sh = SELECT_TRACE_H-10;
                        if (mx < WIDTH/2) {
                            sx = mx + MARGIN/2;
                            sw = WIDTH - mx - 2*MARGIN;
                        } else {
                            sx = MARGIN;
                            sw = mx - MARGIN - MARGIN/2;
                        }

                        DrawRectangleLines(sx-1, sy-1, sw+2, sh+2, LIGHTGRAY);
                        draw_selected_trace(sx, sy, sw, sh);
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

                    /* draw spikes in last 1000 ms */
                    int window_ms = 2000;
                    int display_start = t_ms - window_ms;
                    if (display_start < 0)
                        display_start = 0;
                    for (int i = 0; i < firing_count; i++) {
                        int ft = firing_times[i*2 + 0];        // firing time
                        int nid = firing_times[i*2 + 1];    // neuron id
                        if (ft < display_start)
                            continue;
                        float x = rx + map((float)nid, 0.0, (float)N, 0.0, rw);
                        float y = ry + ((float)(ft - display_start) / window_ms) * rh;
                        Color pc = (nid < NE) ? WHITE : BLUE;
                        pc.a = 128;
                        DrawRectangle((int)x-1, (int)y-1, 2, 2, pc);
                        if (nid == selected_neuron)
                            DrawCircle((int)x, (int)y, 3, YELLOW);
                        if ((ft < t_ms) && (ft > t_ms - 10)) {
                            if (nid < NE) {
                                DrawCircle((int)x, (int)y, 3, WHITE);
                            } else {
                                DrawCircle((int)x, (int)y, 3, BLUE);
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
                    int M = N; //300;
                    if (M > N)
                        M = N;
                    int step = N / M;
                    if (step < 1)
                        step = 1;
                    int idx = 0;
                    for (int i = 0; i < N && idx < M; i += step, idx++) {
                        float vv = v[i];
                        float norm = (vv + 100.0f) / 140.0f;
                        if (norm < 0)
                            norm = 0;
                        if (norm > 1)
                            norm = 1;
                        int x = vx + (int)((float)idx / (float)M * vw);
                        int y = vy + 20 + (int)((1.0f - norm) * (vh - 40));
                        Color col = (i < NE) ? WHITE: BLUE;
                        DrawRectangle(x-1, y-1, 2, 2, col);
                        if (i == selected_neuron)
                            DrawCircle(x, y, 3, YELLOW);
                    }

                    /* mean weight panel */
                    int px = MARGIN, py = vy + vh + MARGIN;
                    int pw = WIDTH - 2*MARGIN;
                    int ph = PANEL_H;
                    DrawRectangleLines(px-1, py-1, pw+2, ph+2, LIGHTGRAY);
                    float mw = mean_exc_weight();
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
            float mw = mean_exc_weight();
            char info[256];
            snprintf(info, sizeof(info), "t=%d ms  mean_exc_weight=%.3f  steps/frame=%d  %s", t_ms, mw, steps_per_frame, paused ? "PAUSED" : "RUN");
            DrawText(info, px+6, py+6, 14, WHITE);

            /* footer */
            DrawText(TextFormat("SPACE: pause/run   UP/DOWN: +/- speed   D: display   G: raster<>grid   R: reset   Click %s to select neuron", graphics_raster?"raster":"grid"), MARGIN+5, HEIGHT-15, 14, GRAY);

            if (show_fps)
                DrawFPS(10, 10);
        } EndDrawing();
    }

    free_network();
    CloseWindow();
    return 0;
}
