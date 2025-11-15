#ifndef SIM_H_
#define SIM_H_

#include "neuron.h"

#define DT 1.0         /* ms per sim step */

#define MAX_DELAY 20 /* ms */

/* STDP parameters (pair-based) */
#define A_plus 0.1f
#define A_minus 0.12f
#define TAU_PLUS 20.0f
#define TAU_MINUS 20.0f
#define W_MIN 0.0f
#define W_MAX 10.0f

/* Delay queues: for each delay (1..MAX_DELAY) maintain list of targets arriving after that many ms */
typedef struct
{
    int *neuron;   /* target neuron ids */
    float *weight; /* corresponding weights */
    int count;
    int cap;
} DelayBucket;

typedef struct
{
    int   neuron;
    float time_ms;
} FiringTime;

typedef struct
{
    int foo;
} sim;

#endif /* SIM_H_ */
