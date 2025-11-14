#include "neuron.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

static const ParsEntry pars_table[20] = {
    {  0.02f,  0.2f, -65.0f,   6.0f, "A tonic spiking" },
    {  0.02f, 0.25f, -65.0f,   6.0f, "B phasic spiking" },
    {  0.02f,  0.2f, -50.0f,   2.0f, "C tonic bursting" },
    {  0.02f, 0.25f, -55.0f,  0.05f, "D phasic bursting" },
    {  0.02f,  0.2f, -55.0f,   4.0f, "E mixed mode" },
    {  0.01f,  0.2f, -65.0f,   8.0f, "F spike freq adaptation" },
    {  0.02f, -0.1f, -55.0f,   6.0f, "G Class 1" },
    {   0.2f, 0.26f, -65.0f,   0.0f, "H Class 2" },
    {  0.02f,  0.2f, -65.0f,   6.0f, "I spike latency" },
    {  0.05f, 0.26f, -60.0f,   0.0f, "J subthreshold oscillations" },
    {   0.1f, 0.26f, -60.0f,  -1.0f, "K resonator" },
    {  0.02f, -0.1f, -55.0f,   6.0f, "L integrator" },
    {  0.03f, 0.25f, -60.0f,   4.0f, "M rebound spike" },
    {  0.03f, 0.25f, -52.0f,   0.0f, "N rebound burst" },
    {  0.03f, 0.25f, -60.0f,   4.0f, "O threshold variability" },
    {   1.0f,  1.5f, -60.0f,   0.0f, "P bistability" },
    {   1.0f,  0.2f, -60.0f, -21.0f, "Q DAP" },
    {  0.02f,  1.0f, -55.0f,   4.0f, "R accomodation" },
    { -0.02f, -1.0f, -60.0f,   8.0f, "S inhibition-induced spiking" },
    {-0.026f, -1.0f, -45.0f,   0.0f, "T inhibition-induced bursting"}
};

/* Configurabili */
const float TOL_REL = 0.10f;    /* tolleranza relativa per a,b */
const float TOL_C = 3.0f;       /* tolleranza assoluta per c (mV) */
const float TOL_D = 1.5f;       /* tolleranza assoluta per d */
const float SCORE_THRESHOLD = 0.85f;

/* Pesi per a,b,c,d nel calcolo dello score (sommano a 1) */
const float W_A = 0.25f;
const float W_B = 0.25f;
const float W_C = 0.30f;
const float W_D = 0.20f;

static float sim_component(float x, float p, float scale_tol)
{
    float diff = fabsf(x - p);
    float s = 1.0f - fminf(diff / scale_tol, 1.0f);
    if (s < 0.0f)
        s = 0.0f;
    return s;
}

static float score_against_entry(float a, float b, float c, float d, const ParsEntry *e)
{
    float tol_a = fmaxf(fabsf(e->a) * TOL_REL, 1e-4f);
    float tol_b = fmaxf(fabsf(e->b) * TOL_REL, 1e-4f);
    float sa = sim_component(a, e->a, tol_a);
    float sb = sim_component(b, e->b, tol_b);
    float sc = sim_component(c, e->c, TOL_C);
    float sd = sim_component(d, e->d, TOL_D);
    return W_A*sa + W_B*sb + W_C*sc + W_D*sd;
}

/* Helper: near-template check with absolute tolerances per field */
static int near_template(float a, float b, float c, float d,
                         float ta, float tb, float tc, float td)
{
    if (fabsf(a - ta) > fmaxf(fabsf(ta)*TOL_REL, 0.01f))
        return 0;
    if (fabsf(b - tb) > fmaxf(fabsf(tb)*TOL_REL, 0.01f))
        return 0;
    if (fabsf(c - tc) > TOL_C)
        return 0;
    if (fabsf(d - td) > TOL_D)
        return 0;
    return 1;
}

/* differenzia CH (chattering) da IB (tonic bursting)
   CH: tende ad avere c più alto (es. -50) e d piccolo; IB: c ~ -55, d più grande.
   Questa funzione restituisce >0.5 se probabilmente CH, <-0.5 se IB, altrimenti 0.*/
static float chirp_vs_burst_score(float c, float d) {
    /* distanza normalizzata a circa intervalli tipici */
    float sc_ch = 1.0f - fminf(fabsf(c + 50.0f) / 5.0f, 1.0f); /* preferisce c ~ -50 */
    float sd_ch = 1.0f - fminf(fabsf(d - 2.0f) / 3.0f, 1.0f);  /* preferisce d ~ 2 */
    float sc_ib = 1.0f - fminf(fabsf(c + 55.0f) / 5.0f, 1.0f); /* preferisce c ~ -55 */
    float sd_ib = 1.0f - fminf(fabsf(d - 4.0f) / 4.0f, 1.0f);  /* preferisce d ~ 4 */
    float score_ch = 0.6f * sc_ch + 0.4f * sd_ch;
    float score_ib = 0.6f * sc_ib + 0.4f * sd_ib;
    return score_ch - score_ib; /* >0 -> CH-like, <0 -> IB-like */
}

ClassResult classify_neuron(float a, float b, float c, float d)
{
    ClassResult res;
    res.type[0] = '\0';
    res.score = 0.0f;
    res.reason[0] = '\0';

    /* 1) confronto con pars_table */
    int best_idx = -1;
    float best_score = -1.0f;
    for (int i = 0; i < 20; ++i) {
        float s = score_against_entry(a,b,c,d, &pars_table[i]);
        if (s > best_score) { best_score = s; best_idx = i; }
    }
//    if (best_score >= SCORE_THRESHOLD && best_idx >= 0) {
//        snprintf(res.type, sizeof(res.type), "%s", pars_table[best_idx].name);
//        res.score = best_score;
//        snprintf(res.reason, sizeof(res.reason), "Matched table row %d (score %.3f)", best_idx, best_score);
//        return res;
//    }

    /* 2) template espliciti coerenti con la tabella del paper */
    if ( near_template(a,b,c,d, 0.02f, 0.25f, -65.0f, 2.0f) ) {
        snprintf(res.type, sizeof(res.type), "LTS (Low-threshold spiking)");
        res.score = best_score;
        snprintf(res.reason, sizeof(res.reason), "Matches LTS template (0.02,0.25,-65,2)");
        return res;
    }
    if ( near_template(a,b,c,d, 0.02f, 0.20f, -65.0f, 8.0f) ) {
        snprintf(res.type, sizeof(res.type), "RS (Regular spiking)");
        res.score = best_score;
        snprintf(res.reason, sizeof(res.reason), "Matches RS template (0.02,0.2,-65,8)");
        return res;
    }
    if ( near_template(a,b,c,d, 0.10f, 0.20f, -65.0f, 2.0f) ) {
        snprintf(res.type, sizeof(res.type), "FS (Fast spiking)");
        res.score = best_score;
        snprintf(res.reason, sizeof(res.reason), "Matches FS template (0.1,0.2,-65,2)");
        return res;
    }
    /* disambiguazione CH vs IB per neuroni con a~0.02,b~0.2 */
    if (fabsf(a - 0.02f) <= 0.01f && fabsf(b - 0.20f) <= 0.04f) {
        float diff = chirp_vs_burst_score(c,d);
        if (diff > 0.2f) {
            snprintf(res.type, sizeof(res.type), "CH (Chattering)");
            res.score = best_score;
            snprintf(res.reason, sizeof(res.reason), "a~0.02,b~0.2 and CH-like by c,d (score diff %.3f)", diff);
            return res;
        } else if (diff < -0.2f) {
            snprintf(res.type, sizeof(res.type), "IB (Intrinsic bursting / tonic bursting)");
            res.score = best_score;
            snprintf(res.reason, sizeof(res.reason), "a~0.02,b~0.2 and IB-like by c,d (score diff %.3f)", diff);
            return res;
        } else {
            /* borderline: return generic bursting with score */
            snprintf(res.type, sizeof(res.type), "Bursting (uncertain CH vs IB)");
            res.score = best_score;
            snprintf(res.reason, sizeof(res.reason), "a~0.02,b~0.2 but ambiguous CH/IB (diff %.3f)", diff);
            return res;
        }
    }

    /* 3) altre regole euristiche generali (fallback) */
    if (a < 0.0f && b < 0.0f) {
        snprintf(res.type, sizeof(res.type), "Inhibition-induced (S/T)");
        res.score = best_score;
        snprintf(res.reason, sizeof(res.reason), "a<0 (%.3f) and b<0 (%.3f)", a, b);
        return res;
    }
    if (a >= 0.8f || b >= 1.0f) {
        snprintf(res.type, sizeof(res.type), "Bistable / DAP-like");
        res.score = best_score;
        snprintf(res.reason, sizeof(res.reason), "large a(%.3f) or b(%.3f)", a, b);
        return res;
    }
    if (fabsf(a - 0.02f) <= 0.01f && fabsf(b - 0.2f) <= 0.05f) {
        if (c >= -55.0f && d >= 10.0f) {
            snprintf(res.type, sizeof(res.type), "Bursting (CH/IB-like)");
            res.score = best_score;
            snprintf(res.reason, sizeof(res.reason), "a~0.02,b~0.2 and c(%.1f)>=-55,d(%.1f)>=10", c, d);
            return res;
        }
        snprintf(res.type, sizeof(res.type), "RS (Regular Spiking)");
        res.score = best_score;
        snprintf(res.reason, sizeof(res.reason), "a~0.02,b~0.2 typical RS");
        return res;
    }
    if (b >= 0.25f && a >= 0.05f) {
        snprintf(res.type, sizeof(res.type), "Fast-spiking / Class-2-like");
        res.score = best_score;
        snprintf(res.reason, sizeof(res.reason), "b>=0.25 (%.3f) and a>=0.05 (%.3f)", b, a);
        return res;
    }

    /* fallback: nearest table row as hint */
    if (best_idx >= 0) {
        snprintf(res.type, sizeof(res.type), "Uncertain (closest: %s)", pars_table[best_idx].name);
        res.score = best_score;
        snprintf(res.reason, sizeof(res.reason), "No strong rule match; closest table row %d (score %.3f)", best_idx, best_score);
    } else {
        snprintf(res.type, sizeof(res.type), "Uncertain");
        res.score = best_score;
        snprintf(res.reason, sizeof(res.reason), "No match and no heuristic triggered");
    }
    return res;
}
