#include <stdlib.h>
#include <math.h>

#include "stb_ds.h"

#include "math_utils.h"

float map(float input, float input_start, float input_end, float output_start, float output_end)
{
    float slope = 1.0 * (output_end - output_start) / (input_end - input_start);
    float output = output_start + slope * (input - input_start);
    return output;
}

/* Utility random */
float frandf()
{
    return (float)rand() / (float)RAND_MAX;
}

/* Clamp helper */
float clampf(float x, float a, float b)
{
    if (x < a)
        return a;
    if (x > b)
        return b;
    return x;
}

float rand01()
{
    return (float)rand() / (float)RAND_MAX;
}

uint f_randi(uint32_t index)
{
    index = (index << 13) ^ index;
    return ((index * (index * index * 15731 + 789221) + 1376312589) & 0x7fffffff);
}

int* array_permute(int *arr, int N)
{
    if (arr == NULL || arrlen(arr) == 0) {
        arrsetlen(arr, N);
    }
    for (int i = 0; i < N; ++i)
        arr[i] = i;
    for (int i = N-1; i > 0; --i) {
        int j = rand() % (i+1);
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
    return arr;
}
