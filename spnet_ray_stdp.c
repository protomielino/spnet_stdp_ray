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
#include "neuron.h"
#include "sim.h"
#include "grid.h"
#include "palette.h"

ColourEntry *palette = NULL;

/* Window */
#define WIDTH 740
#define HEIGHT 740

/* Visualization layout */
#define RASTER_H 350
#define VTRACE_H 110
#define PANEL_H 50
#define SELECT_TRACE_H 100

/* compute mean excitatory weight */
static float mean_exc_weight(Grid *grid, sim *s)
{
    double w = 0.0;
    long cnt = 0;
    for (int i = 0; i < grid->numCells; i++) {
        if (s->neurons[i].is_exc) {
            int CE = arrlen(s->neurons[i].outconn.weights);
            for (int j = 0; j < CE; j++) {
                w += s->neurons[i].outconn.weights[j];
                cnt++;
            }
        }
    }
    if (cnt == 0)
        return 0.0f;
    return (float)(w / cnt);
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
static void draw_selected_trace(Grid *grid, sim *s, int sx, int sy, int sw, int sh)
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
                s->neurons[grid->selected_cell].a,
                s->neurons[grid->selected_cell].b,
                s->neurons[grid->selected_cell].c,
                s->neurons[grid->selected_cell].d,
                s->neurons[grid->selected_cell].class_result.type,
                s->neurons[grid->selected_cell].class_result.score,
                s->neurons[grid->selected_cell].class_result.reason);
    DrawText(buf, sx+10, sy+10, 10, LIGHTGRAY);

    /* draw outline */
    DrawRectangleLines(sx, sy, sw, sh, LIGHTGRAY);

    /* find time window: show last VUBUF_LEN_MS ms */
    int idx = s->vhist_idx;
    float vv;
    float uu;
    int vpx_prev = -1, vpy_prev = -1;
    int upx_prev = -1, upy_prev = -1;
    for (int k = 0; k < VUBUF_LEN_MS; ++k) {
        int pos = (idx - (VUBUF_LEN_MS-1) + k);
        while (pos < 0)
            pos += VUBUF_LEN_MS;
        pos %= VUBUF_LEN_MS;
        vv = s->neurons[grid->selected_cell].v_hist[pos];
        uu = s->neurons[grid->selected_cell].u_hist[pos];
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
        Color vC = s->neurons[grid->selected_cell].is_exc ? GREEN : DARKGREEN;
        Color uC = s->neurons[grid->selected_cell].is_exc ? SKYBLUE : BLUE;
        if (k > 0) {
            DrawLine(vpx_prev, vpy_prev, x, vy, vC);
            DrawLine(upx_prev, upy_prev, x, uy, uC);
        }
        vpx_prev = x; vpy_prev = vy;
        upx_prev = x; upy_prev = uy;
    }
}

// compute cell activities (using interleaved mapping)
void compute_cell_activity(Grid *grid, sim *s)
{
    // zero
    for (int k = 0; k < grid->numCells; k++)
        s->neurons[k].cell_activity = 0.0f;

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
        float vdep = s->neurons[cell].v + 65.0f;
        if (vdep > 0.0f)
            metric += 0.02f * vdep;
        s->neurons[cell].cell_activity += metric;
    }
    // normalize
    for (int k = 0; k < grid->numCells; k++) {
        // normalization scale empirical
        float val = s->neurons[k].cell_activity; // TESTING / 20.0f;
        if (val > 1.0f)
            val = 1.0f;
        s->neurons[k].cell_activity = val;
    }
}

void grid_show(Grid *grid, sim *s)
{
    for (int r = 0; r < grid->numRows; r++) {
        for (int c = 0; c < grid->numCols; c++) {
            int idx = grid_cellpos_to_index(grid, (CellPos){r, c});
            float val = s->neurons[idx].cell_activity;
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

    sim s = {0};
    sim_init_network(&s, &grid);

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
            s.input_prob -= 0.001;
        }
        if (IsKeyPressed(KEY_F2)) {
            s.input_prob += 0.001;
        }
        if (IsKeyPressed(KEY_F3)) {
            s.input_val -= 0.1;
        }
        if (IsKeyPressed(KEY_F4)) {
            s.input_val += 0.1;
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
            sim_free_network(&s, &grid);
            sim_init_network(&s, &grid);
        }

        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
            if (graphics_raster) {
                /* raster area coords */
                int rx = grid.margin;
                int ry = grid.margin;
                int rw = WIDTH - 2*grid.margin;
                int rh = RASTER_H;

                grid.selected_cell =
                        neuron_from_raster_click(&grid, mx, my, rx, ry, rw, rh);
            } else if (graphics_grid) {
                /* grid area coords */
                int rx = grid.margin;
                int ry = grid.margin;
                int rw = grid.width;
                int rh = grid.height;

                grid.selected_cell =
                        cell_index_from_grid_click(&grid, mx, my, rx, ry, rw, rh);
            }

            if (grid.selected_cell >= 0) {
                ClassResult r = neuron_classify(&s.neurons[grid.selected_cell]);
                s.neurons[grid.selected_cell].class_result = r;
            }
        }

        /* simulate */
        if (paused)
            if (IsKeyPressed(KEY_RIGHT))
                sim_step(&s, &grid);
        if (!paused)
            for (int spf = 0; spf < steps_per_frame; spf++)
                sim_step(&s, &grid);

        /* draw */
        BeginDrawing(); {
            ClearBackground(BLACK);
            if (show_graphics) {
                if (graphics_grid) {
                    // compute activities for visualization
                    compute_cell_activity(&grid, &s);

                    grid_show(&grid, &s);

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

                        draw_selected_trace(&grid, &s, sx, sy, sw, sh);
                    }

                    if (grid.selected_cell >= 0) {
                        /* only excitatory neurons have CE */
                        int K_near = s.neurons[grid.selected_cell].num_near_conn;
                        for (int i = 0; i < K_near; ++i) {
                            int post = s.neurons[grid.selected_cell].outconn.targets[i];
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
                        if (s.neurons[grid.selected_cell].num_targets > 0) {
                            int post = grid_cellpos_to_index(&grid, s.neurons[grid.selected_cell].target_center);;
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
                        int K_far = s.neurons[grid.selected_cell].num_far_conn;
                        for (int i = K_near; i < K_far+K_near; ++i) {
                            int post = s.neurons[grid.selected_cell].outconn.targets[i];
                            CellPos target_pos = s.neurons[grid.selected_cell].target_center;
                            CellPos post_pos = {0};
                            post_pos = grid_index_to_cellpos(&grid, post, &post_pos);
                            Vector2 target_pos_xy = grid_cellpos_to_vec2(&grid, target_pos);
                            Vector2 post_pos_xy = grid_cellpos_to_vec2(&grid, post_pos);
                            target_pos_xy = Vector2Add(target_pos_xy, (Vector2){ grid.cellWidth/2, grid.cellHeight/2 });
                            post_pos_xy = Vector2Add(post_pos_xy, (Vector2){ grid.cellWidth/2, grid.cellHeight/2 });
                            DrawLineV(target_pos_xy, post_pos_xy, (Color){255, 255, 255, 128});
                        }
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
                        DrawLine(cl+grid.margin, grid.margin, cl+grid.margin, rh+grid.margin, (Color){255,255,255,32});
                    }
#endif
                    DrawText("Raster (last 1000 ms)", rx+6, ry+6, 14, LIGHTGRAY);

                    /* draw spikes in last 1000 ms */
                    float window_ms = 1000.0;
                    float display_start = s.t_ms - window_ms;
                    if (display_start < 0)
                        display_start = 0;
                    for (int i = 0; i < s.firing_count; i++) {
                        float ft = s.firing_times[i].time_ms;   // firing time
                        int nid = s.firing_times[i].neuron;   // neuron id
                        if (ft < display_start)
                            continue;
                        float x = rx + map((float)nid, 0.0, (float)grid.numCells, 0.0, rw);
                        float y = ry + ((float)(ft - display_start) / window_ms) * rh;
                        Color pc = s.neurons[nid].is_exc ? GREEN : DARKGREEN;
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
                            DrawCircle(x, y, dot_radius, YELLOW);
                        if ((ft < s.t_ms) && (ft > s.t_ms - 10)) {
                            if (s.neurons[nid].is_exc ) {
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
                        float vv = s.neurons[i].v;
                        float norm = (vv + 100.0f) / 140.0f;
                        if (norm < 0)
                            norm = 0;
                        if (norm > 1)
                            norm = 1;
                        int x = vx + (int)((float)idx / (float)M * vw);
                        int y = vy + 20 + (int)((1.0f - norm) * (vh - 40));
                        Color col = s.neurons[i].is_exc ? GREEN: DARKGREEN;

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
                    float mw = mean_exc_weight(&grid, &s);
                    char info[256];
                    snprintf(info, sizeof(info), "mean_exc_weight=%.3f  steps/frame=%d  %s  ||  I=%.1f  I_prob=%.3f  ||  t=%.1f ms", mw, steps_per_frame, paused ? "PAUSED" : "RUN", s.input_val, s.input_prob, s.t_ms);
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
                    draw_selected_trace(&grid, &s, sx, sy, sw, sh);
                }
            }

            /* infos */
            int ry = grid.margin;
            int rh = RASTER_H;
            int vy = ry + rh + grid.margin;
            int vh = VTRACE_H;

            int px = grid.margin;
            int py = vy + vh + grid.margin;
            float mw = mean_exc_weight(&grid, &s);
            char info[256];
            snprintf(info, sizeof(info), "mean_exc_weight=%.3f  steps/frame=%d  %s  ||  I=%.1f  I_prob=%.3f  ||  t=%.1f ms", mw, steps_per_frame, paused ? "PAUSED" : "RUN", s.input_val, s.input_prob, s.t_ms);
            DrawText(info, px+6, py+6, 14, WHITE);

            /* footer */
            DrawText(TextFormat("SPACE: pause/run   UP/DOWN: +/- speed   D: display   G: raster<>grid   R: reset   Click %s to select neuron", graphics_raster?"raster":"grid"), grid.margin+5, HEIGHT-15, 14, GRAY);

            if (show_fps)
                DrawFPS(10, 10);
        } EndDrawing();
    }

    sim_free_network(&s, &grid);
    arrfree(palette);

    CloseWindow();

    return 0;
}
