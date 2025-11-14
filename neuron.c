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

static float score_against_entry(IzkNeuron *neuron, const ParsEntry *e)
{
    float tol_a = fmaxf(fabsf(e->a) * TOL_REL, 1e-4f);
    float tol_b = fmaxf(fabsf(e->b) * TOL_REL, 1e-4f);
    float sa = sim_component(neuron->a, e->a, tol_a);
    float sb = sim_component(neuron->b, e->b, tol_b);
    float sc = sim_component(neuron->c, e->c, TOL_C);
    float sd = sim_component(neuron->d, e->d, TOL_D);
    return W_A*sa + W_B*sb + W_C*sc + W_D*sd;
}

/* Helper: near-template check with absolute tolerances per field */
static int near_template(IzkNeuron *neuron,
                         float ta, float tb, float tc, float td)
{
    if (fabsf(neuron->a - ta) > fmaxf(fabsf(ta)*TOL_REL, 0.01f))
        return 0;
    if (fabsf(neuron->b - tb) > fmaxf(fabsf(tb)*TOL_REL, 0.01f))
        return 0;
    if (fabsf(neuron->c - tc) > TOL_C)
        return 0;
    if (fabsf(neuron->d - td) > TOL_D)
        return 0;
    return 1;
}

/* differenzia CH (chattering) da IB (tonic bursting)
   CH: tende ad avere c più alto (es. -50) e d piccolo; IB: c ~ -55, d più grande.
   Questa funzione restituisce >0.5 se probabilmente CH, <-0.5 se IB, altrimenti 0.*/
static float chirp_vs_burst_score(float c, float d)
{
    /* distanza normalizzata a circa intervalli tipici */
    float sc_ch = 1.0f - fminf(fabsf(c + 50.0f) / 5.0f, 1.0f); /* preferisce c ~ -50 */
    float sd_ch = 1.0f - fminf(fabsf(d - 2.0f) / 3.0f, 1.0f);  /* preferisce d ~ 2 */
    float sc_ib = 1.0f - fminf(fabsf(c + 55.0f) / 5.0f, 1.0f); /* preferisce c ~ -55 */
    float sd_ib = 1.0f - fminf(fabsf(d - 4.0f) / 4.0f, 1.0f);  /* preferisce d ~ 4 */
    float score_ch = 0.6f * sc_ch + 0.4f * sd_ch;
    float score_ib = 0.6f * sc_ib + 0.4f * sd_ib;
    return score_ch - score_ib; /* >0 -> CH-like, <0 -> IB-like */
}

static void print_detailed_scores(IzkNeuron *neuron, int best_idx)
{
    printf("\nDetailed scores vs pars_table:\n");
    printf(" idx | name                            |   a_d  |   b_d  |   c_d  |   d_d  | score\n");
    printf("-----+---------------------------------+--------+--------+--------+--------+-------\n");
    for (int i = 0; i < 20; ++i) {
        const ParsEntry *e = &pars_table[i];
        float tol_a = fmaxf(fabsf(e->a) * TOL_REL, 1e-4f);
        float tol_b = fmaxf(fabsf(e->b) * TOL_REL, 1e-4f);
        float da = fabsf(neuron->a - e->a) / (tol_a>0 ? tol_a : 1.0f);
        float db = fabsf(neuron->b - e->b) / (tol_b>0 ? tol_b : 1.0f);
        float dc = fabsf(neuron->c - e->c) / TOL_C;
        float dd = fabsf(neuron->d - e->d) / TOL_D;
        float s = score_against_entry(neuron,e);
        printf("%4d | %-31s | %6.3f | %6.3f | %6.3f | %6.3f | %.3f",
               i, e->name, da, db, dc, dd, s);
        if (best_idx == i)
            printf(" <<--best match--\n");
        else
            printf("\n");
    }
}

#define CLASSIFICATION_DEBUG

ClassResult neuron_classify(IzkNeuron *neuron)
{
    ClassResult res;
    res.type[0] = '\0';
    res.score = 0.0f;
    res.reason[0] = '\0';

    /* 1) confronto con pars_table */
    int best_idx = -1;
    float best_score = -1.0f;
    for (int i = 0; i < 20; ++i) {
        float s = score_against_entry(neuron, &pars_table[i]);
        if (s > best_score) {
            best_score = s;
            best_idx = i;
        }
    }
//    if (best_score >= SCORE_THRESHOLD && best_idx >= 0) {
//        snprintf(res.type, sizeof(res.type), "%s", pars_table[best_idx].name);
//        res.score = best_score;
//        snprintf(res.reason, sizeof(res.reason), "Matched table row %d (score %.3f)", best_idx, best_score);
//        goto done;
//    }

    /* 2) template espliciti coerenti con la tabella del paper */
    if ( near_template(neuron, 0.02f, 0.25f, -65.0f, 2.0f) ) {
        snprintf(res.type, sizeof(res.type), "LTS (Low-threshold spiking)");
        res.score = best_score;
        snprintf(res.reason, sizeof(res.reason), "Matches LTS template (0.02,0.25,-65,2)");
        goto done;
    }
    if ( near_template(neuron, 0.02f, 0.20f, -65.0f, 8.0f) )
    {
        snprintf(res.type, sizeof(res.type), "RS (Regular spiking)");
        res.score = best_score;
        snprintf(res.reason, sizeof(res.reason), "Matches RS template (0.02,0.2,-65,8)");
        goto done;
    }
    if ( near_template(neuron, 0.10f, 0.20f, -65.0f, 2.0f) )
    {
        snprintf(res.type, sizeof(res.type), "FS (Fast spiking)");
        res.score = best_score;
        snprintf(res.reason, sizeof(res.reason), "Matches FS template (0.1,0.2,-65,2)");
        goto done;
    }
    /* disambiguazione CH vs IB per neuroni con a~0.02,b~0.2 */
    if (fabsf(neuron->a - 0.02f) <= 0.01f && fabsf(neuron->b - 0.20f) <= 0.04f) {
        float diff = chirp_vs_burst_score(neuron->c, neuron->d);
        if (diff > 0.2f) {
            snprintf(res.type, sizeof(res.type), "CH (Chattering)");
            res.score = best_score;
            snprintf(res.reason, sizeof(res.reason), "a~0.02,b~0.2 and CH-like by c,d (score diff %.3f)", diff);
            goto done;
        } else if (diff < -0.2f) {
            snprintf(res.type, sizeof(res.type), "IB (Intrinsic bursting / tonic bursting)");
            res.score = best_score;
            snprintf(res.reason, sizeof(res.reason), "a~0.02,b~0.2 and IB-like by c,d (score diff %.3f)", diff);
            goto done;
        } else {
            /* borderline: return generic bursting with score */
            snprintf(res.type, sizeof(res.type), "Bursting (uncertain CH vs IB)");
            res.score = best_score;
            snprintf(res.reason, sizeof(res.reason), "a~0.02,b~0.2 but ambiguous CH/IB (diff %.3f)", diff);
            goto done;
        }
    }

    /* 3) altre regole euristiche generali (fallback) */
    if (neuron->a < 0.0f && neuron->b < 0.0f) {
        snprintf(res.type, sizeof(res.type), "Inhibition-induced (S/T)");
        res.score = best_score;
        snprintf(res.reason, sizeof(res.reason), "a<0 (%.3f) and b<0 (%.3f)", neuron->a, neuron->b);
        goto done;
    }
    if (neuron->a >= 0.8f || neuron->b >= 1.0f) {
        snprintf(res.type, sizeof(res.type), "Bistable / DAP-like");
        res.score = best_score;
        snprintf(res.reason, sizeof(res.reason), "large a(%.3f) or b(%.3f)", neuron->a, neuron->b);
        goto done;
    }
    if (fabsf(neuron->a - 0.02f) <= 0.01f && fabsf(neuron->b - 0.2f) <= 0.05f) {
        if (neuron->c >= -55.0f && neuron->d >= 10.0f) {
            snprintf(res.type, sizeof(res.type), "Bursting (CH/IB-like)");
            res.score = best_score;
            snprintf(res.reason, sizeof(res.reason), "a~0.02,b~0.2 and c(%.1f)>=-55,d(%.1f)>=10", neuron->c, neuron->d);
            goto done;
        }
        snprintf(res.type, sizeof(res.type), "RS (Regular Spiking)");
        res.score = best_score;
        snprintf(res.reason, sizeof(res.reason), "a~0.02,b~0.2 typical RS");
        goto done;
    }
    if (neuron->b >= 0.25f && neuron->a >= 0.05f) {
        snprintf(res.type, sizeof(res.type), "Fast-spiking / Class-2-like");
        res.score = best_score;
        snprintf(res.reason, sizeof(res.reason), "b>=0.25 (%.3f) and a>=0.05 (%.3f)", neuron->b, neuron->a);
        goto done;
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

done:

#ifdef CLASSIFICATION_DEBUG
    /* debug: stampa dettagliata dei confronti */
    print_detailed_scores(neuron, best_idx);
#endif

    return res;
}
