#ifndef NEURON_H_
#define NEURON_H_

#define VUBUF_LEN_MS 2000

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

    /* per-neuron last spike time for STDP (ms), initialize to very negative */
    int last_spike_time;
    /* per-neuron v,u history buffer for selected trace (circular) */
    float v_hist[VUBUF_LEN_MS]; /* v_hist[idx] */
    float u_hist[VUBUF_LEN_MS]; /* u_hist[idx] */
    float I;

    OutConn outconn;            /* size N */

    float cell_activity;        /* for grid visualization purpose */
} IzkNeuron;

#endif /* NEURON_H_ */
