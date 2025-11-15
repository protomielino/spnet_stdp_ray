#ifndef MATH_UTILS_H_
#define MATH_UTILS_H_

#include <stdlib.h>
#include <stdint.h>

float    map(float input, float input_start, float input_end, float output_start, float output_end);
float    frandf();
float    clampf(float x, float a, float b);
float    rand01();
uint32_t f_randi(uint32_t index);
int* array_permute(int *arr, int N);

#endif /* MATH_UTILS_H_ */
