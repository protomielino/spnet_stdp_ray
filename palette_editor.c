// palette_spread.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "raylib.h"
#include "raymath.h"
#define STB_DS_IMPLEMENTATION
#include "stb_ds.h"

#define CYAN  CLITERAL(Color){ 0, 255, 255, 255 }

/* Tipi estesi: ogni entry ha anche uno "spread" (raggio d'influenza 0..1) */
typedef enum
{
    STOCK_EMPTY,
    STOCK_GREYSCALE,
    STOCK_COLDHOT,
    STOCK_SPECTRUM,
} Stock;

typedef struct
{
    double first;   // posizione 0..1
    Color  second;  // colore
    double spread;  // semilarghezza d'influenza (0..1)
} ColourEntry;

typedef enum
{
    WEIGHT_LINEAR,
    WEIGHT_QUADRATIC,
    WEIGHT_GAUSSIAN,
    WEIGHT_COSINE
} WeightMode;

/* Imposta modalità globale (default) */
static WeightMode g_weight_mode = WEIGHT_LINEAR;
static bool g_weight_wrap_around = false;

/* Se gaussian, regola la "sigma" relativa */
static double g_weight_gaussian_sigma = 0.35;

/* Peso: linear falloff (esistente)
   w = 1 - (dist / spread)  for dist <= spread, else 0 */
static double weight_linear(double dist, double spread)
{
    if (spread <= 0.0 || dist > spread)
        return 0.0;
    return 1.0 - (dist / spread);
}

/* Peso: quadratic falloff (più morbido)
   w = (1 - (dist/spread))^2 */
static double weight_quadratic(double dist, double spread)
{
    if (spread <= 0.0 || dist > spread)
        return 0.0;
    double x = 1.0 - (dist / spread);
    return x * x;
}

/* Peso: gaussian
   w = exp(-0.5 * (dist / (sigma*spread))^2)
   sigma controls larghezza relativa; usare sigma ~0.3..0.6
   Normalizziamo a 1 in dist=0. */
static double weight_gaussian(double dist, double spread, double sigma)
{
    if (spread <= 0.0)
        return 0.0;
    /* convertiamo spread in sigma-equivalente: spread è il raggio utile (cutoff),
       scegliamo sigma tale che w(dist=spread) ~ small (es. ~0.01) */
    double s = fmax(sigma, 1e-6);
    double x = dist / (spread * s);
    if (dist > spread)
        return 0.0;
    return exp(-0.5 * x * x);
}

/* Peso: cosine smoothstep (very smooth, compact support)
   w = 0.5 * (1 + cos(pi * dist / spread)) for dist <= spread, else 0 */
static double weight_cosine(double dist, double spread)
{
    if (spread <= 0.0 || dist > spread)
        return 0.0;
    double x = dist / spread;
    return 0.5 * (1.0 + cos(M_PI * x)); /* cos from 0..pi */
}

/* Funzioni API */
void Palette_init(ColourEntry **palette, Stock stock);
Color Palette_Sample(ColourEntry **palette, double t);
void Palette_SetColour(ColourEntry **palette, double d, Color col, double spread);
void Palette_SetColourSimple(ColourEntry **palette, double d, Color col);
/* Setter opzionali */
void Palette_SetWeightMode(WeightMode m) { g_weight_mode = m; }
void Palette_SetGaussianSigma(double s) { g_weight_gaussian_sigma = s; }

/* Implementazioni */

void Palette_init(ColourEntry **palette, Stock stock)
{
    arrfree(*palette);
    switch (stock)
    {
    case STOCK_EMPTY:
        break;
    case STOCK_GREYSCALE:
        arrput(*palette, ((ColourEntry){0.0, BLACK, 1.0}));
        arrput(*palette, ((ColourEntry){1.0, WHITE, 1.0}));
        break;
    case STOCK_COLDHOT:
        arrput(*palette, ((ColourEntry){0.0, CYAN, 0.5}));
        arrput(*palette, ((ColourEntry){0.5, BLACK, 0.5}));
        arrput(*palette, ((ColourEntry){1.0, YELLOW, 0.5}));
        break;
    case STOCK_SPECTRUM:
        arrput(*palette, ((ColourEntry){0.0 / 6.0, RED, 1.0 / 6.0}));
        arrput(*palette, ((ColourEntry){1.0 / 6.0, YELLOW, 1.0 / 6.0}));
        arrput(*palette, ((ColourEntry){2.0 / 6.0, GREEN, 1.0 / 6.0}));
        arrput(*palette, ((ColourEntry){3.0 / 6.0, CYAN, 1.0 / 6.0}));
        arrput(*palette, ((ColourEntry){4.0 / 6.0, BLUE, 1.0 / 6.0}));
        arrput(*palette, ((ColourEntry){5.0 / 6.0, MAGENTA, 1.0 / 6.0}));
        arrput(*palette, ((ColourEntry){6.0 / 6.0, RED, 1.0 / 6.0}));
        break;
    }
    /* Assicuriamoci ordinamento crescente per posizione */
    for (size_t i = 0; i + 1 < arrlen(*palette); i++) {
        for (size_t j = i + 1; j < arrlen(*palette); j++) {
            if ((*palette)[i].first > (*palette)[j].first) {
                ColourEntry tmp = (*palette)[i];
                (*palette)[i] = (*palette)[j];
                (*palette)[j] = tmp;
            }
        }
    }
}

/* Helper che imposta colore con spread di default 0.0 (compatibilità) */
void Palette_SetColourSimple(ColourEntry **palette, double d, Color col)
{
    Palette_SetColour(palette, d, col, 0.0);
}

void Palette_SetColour(ColourEntry **palette, double d, Color col, double spread)
{
    d = fmin(fmax(d, 0.0), 1.0);
    if (spread < 0.0)
        spread = 0.0;
    if (spread > 1.0)
        spread = 1.0;

    /* Se esiste entry esatta alla stessa posizione, sovrascrivi */
    for (size_t i = 0; i < arrlen(*palette); i++) {
        if ((*palette)[i].first == d) {
            (*palette)[i].second = col;
            (*palette)[i].spread = spread;
            return;
        }
    }

    arrput(*palette, ((ColourEntry){d, col, spread}));

    /* Ordina crescente per posizione */
    for (size_t i = 0; i + 1 < arrlen(*palette); i++) {
        for (size_t j = i + 1; j < arrlen(*palette); j++) {
            if ((*palette)[i].first > (*palette)[j].first) {
                ColourEntry tmp = (*palette)[i];
                (*palette)[i] = (*palette)[j];
                (*palette)[j] = tmp;
            }
        }
    }
}

/* Sampling: somma pesata di tutte le entry la cui distanza (wrap) <= spread.
   Peso lineare (1.0 in centro -> 0.0 al limite). Normalizza RGB(A). */
Color Palette_Sample(ColourEntry **palette, double t)
{
    if (arrlen(*palette) == 0)
        return BLACK;
    if (arrlen(*palette) == 1)
        return (*palette)[0].second;

    double pos = fmod(t, 1.0);
    if (pos < 0.0) pos += 1.0;

    double r_acc = 0.0, g_acc = 0.0, b_acc = 0.0, a_acc = 0.0;
    double weight_sum = 0.0;

    for (size_t i = 0; i < arrlen(*palette); i++) {
        double center = (*palette)[i].first;
        double spread = (*palette)[i].spread;
        /* distanza circolare minima */
        double dist = fabs(pos - center);
        if (g_weight_wrap_around)
            if (dist > 0.5) dist = 1.0 - dist;

        if (spread <= 0.0) {
            if (dist == 0.0) {
                return (*palette)[i].second;
            } else continue;
        }

        if (dist <= spread) {
            /* calcola peso in base a g_weight_mode */
            double w = 0.0;
            switch (g_weight_mode) {
            case WEIGHT_LINEAR:
                w = weight_linear(dist, spread);
                break;
            case WEIGHT_QUADRATIC:
                w = weight_quadratic(dist, spread);
                break;
            case WEIGHT_GAUSSIAN:
                w = weight_gaussian(dist, spread, g_weight_gaussian_sigma);
                break;
            case WEIGHT_COSINE:
                w = weight_cosine(dist, spread);
                break;
            }
            r_acc += w * (double)(*palette)[i].second.r;
            g_acc += w * (double)(*palette)[i].second.g;
            b_acc += w * (double)(*palette)[i].second.b;
            a_acc += w * (double)(*palette)[i].second.a;
            weight_sum += w;
        }
    }

    if (weight_sum <= 0.0) return BLACK;

    Color out;
    out.r = (unsigned char)fmin(fmax((int)round(r_acc / weight_sum), 0), 255);
    out.g = (unsigned char)fmin(fmax((int)round(g_acc / weight_sum), 0), 255);
    out.b = (unsigned char)fmin(fmax((int)round(b_acc / weight_sum), 0), 255);
    out.a = (unsigned char)fmin(fmax((int)round(a_acc / weight_sum), 0), 255);
    return out;
}

/* ----------------- driver di esempio con UI minimal per modificare spread ----------------- */

int main(void)
{
    const int screenW = 1000;
    const int screenH = 300;
    InitWindow(screenW, screenH, "Palette spread example");
    SetTargetFPS(60);

    ColourEntry *palette = NULL;
    Palette_init(&palette, STOCK_SPECTRUM);

    /* aggiungiamo un paio di entry con spread diversi */
    Palette_SetColour(&palette, 0.25, ORANGE, 0.06);
    Palette_SetColour(&palette, 0.5, SKYBLUE, 0.18); // maggiore influenza
    Palette_SetColour(&palette, 0.75, PURPLE, 0.03);

    double play_t = 0.0;
    int selected = -1; /* index entry selezionata per modifica */
    bool dragging = false;

    while (!WindowShouldClose())
    {
        float dt = GetFrameTime();
        play_t += dt * 0.08f;

        /* Input: click per selezionare entry (vicino al marcatore), drag per spostare,
           wheel per cambiare spread dell'entry selezionata, keys per preset */
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            double mx = (double)GetMouseX() / (screenW - 1);
            selected = -1;
            for (size_t i = 0; i < arrlen(palette); i++) {
                double center = palette[i].first;
                double dist = fabs(mx - center);
                if (dist > 0.5)
                    dist = 1.0 - dist;
                if (dist < 0.02) {
                    selected = (int)i;
                    break;
                }
            }
            dragging = (selected != -1);
        }
        if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) dragging = false;

        if (dragging && selected >= 0) {
            double mx = (double)GetMouseX() / (screenW - 1);
            palette[selected].first = fmin(fmax(mx, 0.0), 1.0);
            /* keep sorted after drag: simple bubble to keep order stable */
            for (size_t i = 0; i + 1 < arrlen(palette); i++) {
                for (size_t j = i + 1; j < arrlen(palette); j++) {
                    if (palette[i].first > palette[j].first) {
                        ColourEntry tmp = palette[i];
                        palette[i] = palette[j];
                        palette[j] = tmp;
                        /* update selected index if swapped */
                        if ((int)j == selected) selected = (int)i;
                        else if ((int)i == selected) selected = (int)j;
                    }
                }
            }
        }

        int wheel = GetMouseWheelMove();
        if (wheel != 0 && selected >= 0) {
            double s = palette[selected].spread;
            s += wheel * 0.01;
            if (s < 0.0) s = 0.0;
            if (s > 0.5) s = 0.5; /* limite pratico */
            palette[selected].spread = s;
        }

        if (IsKeyPressed(KEY_G)) { arrfree(palette); Palette_init(&palette, STOCK_GREYSCALE); }
        if (IsKeyPressed(KEY_C)) { arrfree(palette); Palette_init(&palette, STOCK_COLDHOT); }
        if (IsKeyPressed(KEY_S)) { arrfree(palette); Palette_init(&palette, STOCK_SPECTRUM); }
        if (IsKeyPressed(KEY_E)) { arrfree(palette); Palette_init(&palette, STOCK_EMPTY); }
        if (IsKeyPressed(KEY_A)) {
            double pos = (double)rand() / RAND_MAX;
            Color rnd = (Color){ rand() % 256, rand() % 256, rand() % 256, 255 };
            double spr = ((double)(rand() % 100)) / 500.0; /* 0..0.2 */
            Palette_SetColour(&palette, pos, rnd, spr);
        }

        if (IsKeyPressed(KEY_ONE)) Palette_SetWeightMode(WEIGHT_LINEAR);
        if (IsKeyPressed(KEY_TWO)) Palette_SetWeightMode(WEIGHT_QUADRATIC);
        if (IsKeyPressed(KEY_THREE)) Palette_SetWeightMode(WEIGHT_GAUSSIAN);
        if (IsKeyPressed(KEY_FOUR)) Palette_SetWeightMode(WEIGHT_COSINE);
        if (IsKeyDown(KEY_UP))    g_weight_gaussian_sigma = fmin(g_weight_gaussian_sigma + 0.01, 1.0);
        if (IsKeyDown(KEY_DOWN))  g_weight_gaussian_sigma = fmax(g_weight_gaussian_sigma - 0.01, 0.05);

        BeginDrawing();
        ClearBackground(BLACK);

        /* barra della palette */
        for (int x = 0; x < screenW; x++) {
            double samplePos = (double)x / (screenW - 1);
            Color c = Palette_Sample(&palette, samplePos);
            DrawLine(x, 20, x, 140, c);
        }

        /* samplePos definito (es. double samplePos = (double)GetMouseX()/(screenW-1) oppure play_pos) */
        double samplePos = (double)GetMouseX()/(screenW-1);

        int bar_x = 10;
        int bar_w = 100;
        int bar_h = 80;
        int base_y = 250; /* posizione verticale del grafico */
        char *weight_mode_buf = NULL;
        for (size_t i = 0; i < arrlen(palette); i++) {
            double center = palette[i].first;
            double dist = fabs(samplePos - center);
            if (g_weight_wrap_around)
                if (dist > 0.5) dist = 1.0 - dist;
            double spread = palette[i].spread;

            /* calcola peso con modalità corrente */
            double w = 0.0;
            switch (g_weight_mode) {
            case WEIGHT_LINEAR:
                w = weight_linear(dist, spread);
                weight_mode_buf = "linear";
                break;
            case WEIGHT_QUADRATIC:
                w = weight_quadratic(dist, spread);
                weight_mode_buf = "quadratic";
                break;
            case WEIGHT_GAUSSIAN:
                w = weight_gaussian(dist, spread, g_weight_gaussian_sigma);
                weight_mode_buf = "gaussian";
                break;
            case WEIGHT_COSINE:
                w = weight_cosine(dist, spread);
                weight_mode_buf = "cosine";
                break;
            }

            /* disegno: barra orizzontale per entry */
            int bx = bar_x + (int)i * (bar_w + 6);
            int bh = (int)round(w * bar_h);
            DrawRectangle(bx, base_y - bh, bar_w, bh, palette[i].second);
            DrawRectangleLines(bx, base_y - bar_h, bar_w, bar_h, DARKGRAY);
            DrawText(TextFormat("%.2f", w), bx + 2, base_y + 2, 10, GRAY);
        }
        DrawText(TextFormat("weight mode: %s", weight_mode_buf), bar_x + 2, base_y - bar_h - 15, 10, GRAY);

        /* draw entries as markers with spread visualized */
        for (size_t i = 0; i < arrlen(palette); i++) {
            int cx = (int)round(palette[i].first * (screenW - 1));
            int spr_px = (int)round(palette[i].spread * (screenW - 1));
            DrawRectangleLines(cx - spr_px, 18, spr_px * 2, 124, (Color){245, 245, 245, 128});
            DrawCircle(cx, 18, 7, DARKGRAY);
            DrawCircle(cx, 18, 6, palette[i].second);
            if ((int)i == selected) {
                DrawCircleLines(cx, 18, 10, WHITE);
            }
            DrawText(TextFormat("%.2f s=%.3f", (float)palette[i].first, (float)palette[i].spread), cx - 40, 145, 10, LIGHTGRAY);
        }

        /* moving marker showing sample at play_t */
        double play_pos = play_t - floor(play_t);
        int px = (int)round(play_pos * (screenW - 1));
        Color marker = Palette_Sample(&palette, play_pos);
        DrawRectangle(px - 5, 150, 10, 10, marker);

        DrawText("Click marker near top to select+drag. Mouse wheel adjusts spread.", 10, screenH - 31, 12, LIGHTGRAY);
        DrawText("G=C,B=Spectrum,C=ColdHot,E=Empty,A=Add random", 10, screenH - 15, 12, LIGHTGRAY);

        EndDrawing();
    }

    arrfree(palette);
    CloseWindow();
    return 0;
}
