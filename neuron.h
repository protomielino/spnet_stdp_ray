#ifndef NEURON_H_
#define NEURON_H_

#include <stdint.h>

#include "grid.h"
#include "neuron_classification.h"

#define VUBUF_LEN_MS 1000



typedef struct CellPos_s CellPos;

/* Outgoing connections structure */
typedef struct
{
    int *targets;            /* CE or CI targets */
    float *weights;          /* corresponding weights */
    unsigned char *delay;    /* delays in ms (for excitatory) */
} OutConn;

typedef struct
{
    double a;
    double b;
    double c;
    double d;

    double v;
    double u;

    double I;

    float g_ampa;
    float g_nmda;
    float g_gabaa;
    float g_gabab;


    /* per-neuron last spike time for STDP (ms), initialize to very negative */
    float last_spike_time;
    /* per-neuron v,u history buffer for selected trace (circular) */
    float v_hist[VUBUF_LEN_MS]; /* v_hist[idx] */
    float u_hist[VUBUF_LEN_MS]; /* u_hist[idx] */
    float ampa_hist[VUBUF_LEN_MS];
    float nmda_hist[VUBUF_LEN_MS];
    float gabaa_hist[VUBUF_LEN_MS];
    float gabab_hist[VUBUF_LEN_MS];
    float I_hist[VUBUF_LEN_MS];
    // Scale connectivity per synapse type: we'll map excitatory connections to AMPA+NMDA,
    // inhibitory to GABAA+GABAB.
    // Define fractions and gains
    float w_AMPA_frac;
    float w_NMDA_frac;
    float w_GABAA_frac;
    float w_GABAB_frac;

    OutConn outconn;
    CellPos target_center;
    int num_targets;
    int num_near_conn;
    int num_far_conn;

    uint8_t is_exc;

    float cell_activity;        /* for grid visualization purpose */

    // For visualization: buffers storing instantaneous measure for each neuron
    float instant;

    ClassResult class_result;
} IzkNeuron;

#endif /* NEURON_H_ */
