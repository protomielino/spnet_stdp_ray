#ifndef GRID_H_
#define GRID_H_

#include <stdbool.h>

#include <raylib.h>

typedef struct
{
    int numCols;
    int numRows;
    int numCells; // this number needs to be == N == NE+NI
    int width; // pixel
    int height; // pixel
    float cellWidth; // pixel
    float cellHeight; // pixel
    int margin;
    int selected_cell;
} Grid;

typedef struct
{
    int cols;
    int rows;
    double cellW;
    double cellH;
    double aspectDiff; /* |cellW/cellH - 1| */
    int totalCells;
} GridChoice;

typedef struct CellPos_s
{
    int r;
    int c;
} CellPos;

int grid_toroidal_dist_sq(Grid *grid, CellPos cell1, CellPos cell2);
int grid_pick_random_cells_in_annulus(Grid *grid, CellPos center_cell, int rmin, int rmax, int k, CellPos *out);
int grid_in_annulus(Grid *grid, CellPos cell, CellPos center_cell, int rmin, int rmax);

GridChoice grid_choose_aroundN(int N, int width, int height);
GridChoice grid_choose_exactN(int N, int width, int height);
GridChoice grid_choose(int N, int width, int height, bool strict);

CellPos grid_index_to_cellpos(Grid *grid, int idx, CellPos *cell);
int grid_cellpos_to_index(Grid *grid, CellPos cell);
Vector2 grid_cellpos_to_vec2(Grid *grid, CellPos cp);
int cell_index_from_grid_click(Grid *grid, int click_x, int click_y, int rx, int ry, int rw, int rh);


#endif /* GRID_H_ */
