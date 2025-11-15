//#define STB_DS_IMPLEMENTATION
#include "stb_ds.h"

#include "palette.h"

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

void Palette_init(ColourEntry **palette, Stock stock)
{
    arrfree(*palette);
    
    float nc = 0;
    float ci = 0;
    float val = 1.0;
    float spread = 0;
    switch (stock)
    {
    case STOCK_EMPTY:
        break;
    case STOCK_GREYSCALE:
        nc = 2 - 1;
        spread = val/nc;
        arrput(*palette, ((ColourEntry){ci++/nc, BLACK, spread}));
        arrput(*palette, ((ColourEntry){ci++/nc, WHITE, spread}));
        break;
    case STOCK_COLDHOT1:
        nc = 4 - 1;
        spread = val/nc;
        arrput(*palette, ((ColourEntry){ci++/nc, BLACK,  spread}));
        arrput(*palette, ((ColourEntry){ci++/nc, CYAN,   spread}));
        arrput(*palette, ((ColourEntry){ci++/nc, RED,    spread}));
        arrput(*palette, ((ColourEntry){ci++/nc, YELLOW, spread}));
        break;
    case STOCK_COLDHOT2:
        nc = 6 - 1;
        spread = 1.0/nc;
        arrput(*palette, ((ColourEntry){ci++/nc, BLACK,  spread}));
        arrput(*palette, ((ColourEntry){ci++/nc, BLUE,   spread}));
        arrput(*palette, ((ColourEntry){ci++/nc, CYAN,   spread}));
        arrput(*palette, ((ColourEntry){ci++/nc, GREEN,  spread}));
        arrput(*palette, ((ColourEntry){ci++/nc, YELLOW, spread}));
        arrput(*palette, ((ColourEntry){ci++/nc, RED,    spread}));
        break;
    case STOCK_COLDHOT3:
        nc = 6 - 1;
        spread = val/nc;
        arrput(*palette, ((ColourEntry){ci++/nc, BLACK,  spread}));
        arrput(*palette, ((ColourEntry){ci++/nc, (Color){ (               0), (      spread*180), ( 32 + spread*135), 255 }, spread}));
        arrput(*palette, ((ColourEntry){ci++/nc, (Color){ (               0), (180 + spread* 75), (255 - spread*255), 255 }, spread}));
        arrput(*palette, ((ColourEntry){ci++/nc, (Color){ (      spread*200), (255 - spread* 55), (               0), 255 }, spread}));
        arrput(*palette, ((ColourEntry){ci++/nc, (Color){ (200 + spread* 55), (200 - spread*150), (               0), 255 }, spread}));
        arrput(*palette, ((ColourEntry){ci++/nc, (Color){ (             255), (              50), (               0), 255 }, spread}));
        break;
    case STOCK_SPECTRUM:
        nc = 7 - 1;
        spread = val/nc;
        arrput(*palette, ((ColourEntry){ci++/nc, BLACK,   spread}));
        arrput(*palette, ((ColourEntry){ci++/nc, RED,     spread}));
        arrput(*palette, ((ColourEntry){ci++/nc, YELLOW,  spread}));
        arrput(*palette, ((ColourEntry){ci++/nc, GREEN,   spread}));
        arrput(*palette, ((ColourEntry){ci++/nc, CYAN,    spread}));
        arrput(*palette, ((ColourEntry){ci++/nc, BLUE,    spread}));
        arrput(*palette, ((ColourEntry){ci++/nc, MAGENTA, spread}));
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
        if ((*palette)[i].first == d) { // TODO: sistema questa boiata
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

/* Setter opzionali */
void Palette_SetWeightMode(WeightMode m) { g_weight_mode = m; }
void Palette_SetGaussianSigma(double s) { g_weight_gaussian_sigma = s; }
