#ifndef NEURON_CLASSIFICATION_H_
#define NEURON_CLASSIFICATION_H_

#include <stdio.h>
#include <string.h>
#include <math.h>

typedef struct
{
    float a,b,c,d;
    const char *name;
} ParsEntry;

typedef struct
{
    char type[128];
    float score;
    char reason[256];
} ClassResult;

ClassResult classify_neuron(float a, float b, float c, float d);

#endif /* NEURON_CLASSIFICATION_H_ */
