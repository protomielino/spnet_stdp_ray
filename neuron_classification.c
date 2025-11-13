#include <stdio.h>
#include <string.h>
#include <math.h>

typedef struct
{
    float a,b,c,d,I;
    const char *name;
} ParsEntry;

static const ParsEntry pars_table[20] = {
    {  0.02f,  0.2f, -65.0f,   6.0f,  14.0f, "A tonic spiking"},
    {  0.02f, 0.25f, -65.0f,   6.0f,   0.5f, "B phasic spiking"},
    {  0.02f,  0.2f, -50.0f,   2.0f,  15.0f, "C tonic bursting"},
    {  0.02f, 0.25f, -55.0f,  0.05f,   0.6f, "D phasic bursting"},
    {  0.02f,  0.2f, -55.0f,   4.0f,  10.0f, "E mixed mode"},
    {  0.01f,  0.2f, -65.0f,   8.0f,  30.0f, "F spike freq adaptation"},
    {  0.02f, -0.1f, -55.0f,   6.0f,   0.0f, "G Class 1"},
    {   0.2f, 0.26f, -65.0f,   0.0f,   0.0f, "H Class 2"},
    {  0.02f,  0.2f, -65.0f,   6.0f,   7.0f, "I spike latency"},
    {  0.05f, 0.26f, -60.0f,   0.0f,   0.0f, "J subthreshold oscillations"},
    {   0.1f, 0.26f, -60.0f,  -1.0f,   0.0f, "K resonator"},
    {  0.02f, -0.1f, -55.0f,   6.0f,   0.0f, "L integrator"},
    {  0.03f, 0.25f, -60.0f,   4.0f,   0.0f, "M rebound spike"},
    {  0.03f, 0.25f, -52.0f,   0.0f,   0.0f, "N rebound burst"},
    {  0.03f, 0.25f, -60.0f,   4.0f,   0.0f, "O threshold variability"},
    {   1.0f,  1.5f, -60.0f,   0.0f, -65.0f, "P bistability"},
    {   1.0f,  0.2f, -60.0f, -21.0f,   0.0f, "Q DAP"},
    {  0.02f,  1.0f, -55.0f,   4.0f,   0.0f, "R accomodation"},
    { -0.02f, -1.0f, -60.0f,   8.0f,  80.0f, "S inhibition-induced spiking"},
    {-0.026f, -1.0f, -45.0f,   0.0f,  80.0f, "T inhibition-induced bursting"}
};

/* Configurabili */
const float TOL_REL = 0.15f;    /* tolleranza relativa per a,b */
const float TOL_C = 5.0f;       /* tolleranza assoluta per c (mV) */
const float TOL_D = 3.0f;       /* tolleranza assoluta per d */
const float SCORE_THRESHOLD = 0.8f;

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

typedef struct
{
    char type[128];
    float score;
    char reason[256];
} ClassResult;

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
    if ( near_template(a,b,c,d, 0.02f, 0.20f, -55.0f, 4.0f) ) {
        snprintf(res.type, sizeof(res.type), "IB (Intrinsic bursting)");
        res.score = best_score;
        snprintf(res.reason, sizeof(res.reason), "Matches IB template (0.02,0.2,-55,4)");
        return res;
    }
    if ( near_template(a,b,c,d, 0.02f, 0.20f, -50.0f, 2.0f) ) {
        snprintf(res.type, sizeof(res.type), "CH (Chattering)");
        res.score = best_score;
        snprintf(res.reason, sizeof(res.reason), "Matches CH template (0.02,0.2,-50,2)");
        return res;
    }
    if ( near_template(a,b,c,d, 0.10f, 0.20f, -65.0f, 2.0f) ) {
        snprintf(res.type, sizeof(res.type), "FS (Fast spiking)");
        res.score = best_score;
        snprintf(res.reason, sizeof(res.reason), "Matches FS template (0.1,0.2,-65,2)");
        return res;
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

//int main1()
//{
//    /* Esempi: modifica questi valori per testare */
//    float a = 0.02f, b = 0.25f, c = -65.0f, d = 2.0f, I = 0.0f; /* LTS example */
//
//    /* altro esempio */
//    float a2 = 0.02f, b2 = 0.20f, c2 = -65.0f, d2 = 8.0f, I2 = 14.0f; /* RS */
//    ClassResult r2 = classify_neuron(a2,b2,c2,d2,I2);
//    printf("\nNeuron params: a=%.4f b=%.4f c=%.2f d=%.2f I=%.2f\n", a2,b2,c2,d2,I2);
//    printf("Type: %s\nScore: %.3f\nReason: %s\n", r2.type, r2.score, r2.reason);
//
//    return 0;
//}

//#include <stdio.h>
//#include <string.h>
//#include <math.h>
//
//#include "neuron_classification.h"
//
//static const ParsEntry pars_table[20] =
//{
//    {  0.02f,  0.2f, -65.0f,   6.0f, "A tonic spiking"},
//    {  0.02f, 0.25f, -65.0f,   6.0f, "B phasic spiking"},
//    {  0.02f,  0.2f, -50.0f,   2.0f, "C tonic bursting"},
//    {  0.02f, 0.25f, -55.0f,  0.05f, "D phasic bursting"},
//    {  0.02f,  0.2f, -55.0f,   4.0f, "E mixed mode"},
//    {  0.01f,  0.2f, -65.0f,   8.0f, "F spike freq adaptation"},
//    {  0.02f, -0.1f, -55.0f,   6.0f, "G Class 1"},
//    {   0.2f, 0.26f, -65.0f,   0.0f, "H Class 2"},
//    {  0.02f,  0.2f, -65.0f,   6.0f, "I spike latency"},
//    {  0.05f, 0.26f, -60.0f,   0.0f, "J subthreshold oscillations"},
//    {   0.1f, 0.26f, -60.0f,  -1.0f, "K resonator"},
//    {  0.02f, -0.1f, -55.0f,   6.0f, "L integrator"},
//    {  0.03f, 0.25f, -60.0f,   4.0f, "M rebound spike"},
//    {  0.03f, 0.25f, -52.0f,   0.0f, "N rebound burst"},
//    {  0.03f, 0.25f, -60.0f,   4.0f, "O threshold variability"},
//    {   1.0f,  1.5f, -60.0f,   0.0f, "P bistability"},
//    {   1.0f,  0.2f, -60.0f, -21.0f, "Q DAP"},
//    {  0.02f,  1.0f, -55.0f,   4.0f, "R accomodation"},
//    { -0.02f, -1.0f, -60.0f,   8.0f, "S inhibition-induced spiking"},
//    {-0.026f, -1.0f, -45.0f,   0.0f, "T inhibition-induced bursting"}
//};
//
///* Configurabili */
//const float TOL_REL = 0.15f;    /* tolleranza relativa per a,b */
//const float TOL_C   = 5.0f;     /* tolleranza assoluta per c (mV) */
//const float TOL_D   = 3.0f;     /* tolleranza assoluta per d */
//const float SCORE_THRESHOLD = 0.8f;
//
///* Pesi per a,b,c,d nel calcolo dello score (sommano a 1) */
//const float W_A = 0.25f;
//const float W_B = 0.25f;
//const float W_C = 0.30f;
//const float W_D = 0.20f;
//
//static float sim_component(float x, float p, float scale_tol)
//{
//    float diff = fabsf(x - p);
//    float s = 1.0f - fminf(diff / scale_tol, 1.0f);
//    if (s < 0.0f)
//        s = 0.0f;
//
//    return s;
//}
//
//static float score_against_entry(float a, float b, float c, float d, const ParsEntry *e)
//{
//    float tol_a = fmaxf(fabsf(e->a) * TOL_REL, 1e-4f);
//    float tol_b = fmaxf(fabsf(e->b) * TOL_REL, 1e-4f);
//    float sa = sim_component(a, e->a, tol_a);
//    float sb = sim_component(b, e->b, tol_b);
//    float sc = sim_component(c, e->c, TOL_C);
//    float sd = sim_component(d, e->d, TOL_D);
//
//    return W_A*sa + W_B*sb + W_C*sc + W_D*sd;
//}
//
//ClassResult classify_neuron(float a, float b, float c, float d)
//{
//    ClassResult res;
//    res.type[0] = '\0';
//    res.score = 0.0f;
//    res.reason[0] = '\0';
//
//    /* 1) confronto con pars_table */
//    int best_idx = -1;
//    float best_score = -1.0f;
//    for (int i = 0; i < 20; ++i) {
//        float s = score_against_entry(a,b,c,d, &pars_table[i]);
//        if (s > best_score) {
//            best_score = s;
//            best_idx = i;
//        }
//    }
//    if (best_score >= SCORE_THRESHOLD && best_idx >= 0) {
//        snprintf(res.type, sizeof(res.type), "%s", pars_table[best_idx].name);
//        res.score = best_score;
//        snprintf(res.reason, sizeof(res.reason), "Matched table row %d (score %.3f)", best_idx, best_score);
//        return res;
//    }
//
//    /* 2) regole euristiche (prioritarie) */
//    if (a < 0.0f && b < 0.0f) {
//        snprintf(res.type, sizeof(res.type), "Inhibition-induced (S/T)");
//        res.score = best_score;
//        snprintf(res.reason, sizeof(res.reason), "a<0 (%.3f) and b<0 (%.3f)", a, b);
//        return res;
//    }
//    if (a >= 0.8f || b >= 1.0f) {
//        snprintf(res.type, sizeof(res.type), "Bistable / DAP-like");
//        res.score = best_score;
//        snprintf(res.reason, sizeof(res.reason), "large a(%.3f) or b(%.3f)", a, b);
//        return res;
//    }
//    /* RS / CH / bursting heuristics */
//    if (fabsf(a - 0.02f) <= 0.01f && fabsf(b - 0.2f) <= 0.05f) {
//        if (c >= -55.0f && d >= 10.0f) {
//            snprintf(res.type, sizeof(res.type), "Bursting (CH/IB-like)");
//            res.score = best_score;
//            snprintf(res.reason, sizeof(res.reason), "a~0.02,b~0.2 and c(%.1f)>=-55,d(%.1f)>=10", c, d);
//            return res;
//        }
//        snprintf(res.type, sizeof(res.type), "RS (Regular Spiking)");
//        res.score = best_score;
//        snprintf(res.reason, sizeof(res.reason), "a~0.02,b~0.2 typical RS");
//        return res;
//    }
//    /* Fast-spiking / Class 2 */
//    if (b >= 0.25f && a >= 0.05f) {
//        snprintf(res.type, sizeof(res.type), "Fast-spiking / Class-2-like");
//        res.score = best_score;
//        snprintf(res.reason, sizeof(res.reason), "b>=0.25 (%.3f) and a>=0.05 (%.3f)", b, a);
//        return res;
//    }
//    /* LTS / integrator signs */
//    if (b < 0.0f && a > 0.0f) {
//        snprintf(res.type, sizeof(res.type), "LTS / Integrator-like");
//        res.score = best_score;
//        snprintf(res.reason, sizeof(res.reason), "b<0 (%.3f) suggests integrator/low-threshold", b);
//        return res;
//    }
//
//    /* fallback: nearest table row as hint */
//    if (best_idx >= 0) {
//        snprintf(res.type, sizeof(res.type), "Uncertain (closest: %s)", pars_table[best_idx].name);
//        res.score = best_score;
//        snprintf(res.reason, sizeof(res.reason), "No strong rule match; closest table row %d (score %.3f)", best_idx, best_score);
//    } else {
//        snprintf(res.type, sizeof(res.type), "Uncertain");
//        res.score = best_score;
//        snprintf(res.reason, sizeof(res.reason), "No match and no heuristic triggered");
//    }
//
//    return res;
//}
