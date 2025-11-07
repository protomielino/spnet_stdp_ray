#ifndef GRID_H_
#define GRID_H_

#include <stdbool.h>

typedef struct CellPos_s
{
    int r;
    int c;
} CellPos;

typedef struct
{
    int cols;
    int rows;
    double cellW;
    double cellH;
    double aspectDiff; /* |cellW/cellH - 1| */
    int totalCells;
} GridChoice;

GridChoice choose_grid(int N, int width, int height, bool strict);

#endif /* GRID_H_ */
