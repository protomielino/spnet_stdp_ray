#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdbool.h>
#include <stdlib.h>

#include <raylib.h>

#include "stb_ds.h"

#include "grid.h"
#include "math_utils.h"

/* Trova la migliore coppia cols,rows per N celle in un'area width x height */
GridChoice grid_choose_aroundN(int N, int width, int height)
{
    GridChoice best;
    best.cols = best.rows = 0;
    best.cellW = best.cellH = 0.0;
    best.aspectDiff = 1e9;
    best.totalCells = INT_MAX;

    /* Limiti pratici: cols non deve eccedere N (una colonna per cella). */
    int maxCols = N;
    for (int cols = 1; cols <= maxCols; ++cols) {
        int rows = (N + cols - 1) / cols; /* ceil(N / cols) */
        int total = cols * rows;
        double cellW = (double)width / cols;
        double cellH = (double)height / rows;
        if (cellW <= 0 || cellH <= 0)
            continue;
        double ratio = cellW / cellH;
        double aspectDiff = fabs(ratio - 1.0); /* quanto ci si discosta da quadrato */

        /* Preferiamo minore aspectDiff; in caso di pareggio, minore totalCells; poi celle più grandi */
        int better = 0;
        if (aspectDiff < best.aspectDiff - 1e-12)
            better = 1;
        else if (fabs(aspectDiff - best.aspectDiff) < 1e-12) {
            if (total < best.totalCells)
                better = 1;
            else if (total == best.totalCells) {
                /* Preferire celle con area maggiore (più grandi visivamente) */
                double area = cellW * cellH;
                double bestArea = best.cellW * best.cellH;
                if (area > bestArea + 1e-9)
                    better = 1;
            }
        }

        if (better) {
            best.cols = cols;
            best.rows = rows;
            best.cellW = cellW;
            best.cellH = cellH;
            best.aspectDiff = aspectDiff;
            best.totalCells = total;
        }
    }

    return best;
}

/* Cerca tutti i divisori d di N e valuta coppie (d, N/d) e (N/d, d). */
GridChoice grid_choose_exactN(int N, int width, int height)
{
    GridChoice best;
    best.cols = best.rows = 0;
    best.cellW = best.cellH = 0.0;
    best.aspectDiff = 1e9;

    for (int d = 1; d * d <= N; ++d) {
        if (N % d != 0)
            continue;
        int a = d;
        int b = N / d;

        /* Prova entrambe le orientazioni: cols=a, rows=b e cols=b, rows=a */
        int combos[2][2] = {{a, b}, {b, a}};
        for (int k = 0; k < 2; ++k) {
            int cols = combos[k][0];
            int rows = combos[k][1];

            double cellW = (double)width / cols;
            double cellH = (double)height / rows;
            if (cellW <= 0 || cellH <= 0)
                continue;
            double ratio = cellW / cellH;
            double aspectDiff = fabs(ratio - 1.0);

            int better = 0;
            if (aspectDiff < best.aspectDiff - 1e-12)
                better = 1;
            else if (fabs(aspectDiff - best.aspectDiff) < 1e-12) {
                /* tie-breaker: preferire celle con area maggiore */
                double area = cellW * cellH;
                double bestArea = best.cellW * best.cellH;
                if (area > bestArea + 1e-9)
                    better = 1;
            }

            if (better) {
                best.cols = cols;
                best.rows = rows;
                best.cellW = cellW;
                best.cellH = cellH;
                best.aspectDiff = aspectDiff;
            }
        }
    }

    best.totalCells = N;

    return best;
}

GridChoice grid_choose(int N, int width, int height, bool strict)
{
    GridChoice ret = {0};

    if (N <= 0 || width <= 0 || height <= 0) {
        ret.totalCells = -1;
        return ret;
    }

    printf("Target N (num cells): %s %d \n", strict ? "==" : "~=", N);
    printf("Grid width x height (px): %d x %d\n", width, height);

    if (strict) {
        ret = grid_choose_exactN(N, width, height);
    } else {
        ret = grid_choose_aroundN(N, width, height);
    }

    if (ret.cols == 0) {
        printf("Nessuna configurazione trovata.\n");
        ret.totalCells = -1;
        return ret;
    }

    printf("\nChosen: cols=%d, rows=%d, num cells=%d\n",
            ret.cols, ret.rows, ret.totalCells);
    printf("cell dimension: %.3f x %.3f px (cell W/H ratio = %.3f)\n",
            ret.cellW, ret.cellH, ret.cellW / ret.cellH);

    return ret;
}

int grid_toroidal_dist_sq(Grid *grid, CellPos cell1, CellPos cell2)
{
    // distanza al quadrato considerando wrap-around (toroide) con metriche di griglia euclidea
    int dr = abs(cell1.r - cell2.r);
    int dc = abs(cell1.c - cell2.c);
    if (dr > grid->numRows/2) dr = grid->numRows - dr;
    if (dc > grid->numCols/2) dc = grid->numCols - dc;
    return dr*dr + dc*dc;
}

// Seleziona k celle casuali nell'anello [rmin,rmax] attorno al centro.
// Restituisce il numero effettivo di celle selezionate (<= k)
// out array deve avere capacità almeno k.
int grid_pick_random_cells_in_annulus(Grid *grid, CellPos center_cell, int rmin, int rmax, int k, CellPos *out)
{
    // Raccogli tutte le celle ammissibili
    CellPos *candidates = NULL;
    int cnt = 0;
    for (int r = 0; r < grid->numRows; r++) {
        for (int c = 0; c < grid->numCols; c++) {
            if (grid_in_annulus(grid, (CellPos){r, c}, center_cell, rmin, rmax)) {
                CellPos cand = {r, c};
                arrput(candidates, cand);
            }
        }
    }
    cnt = arrlen(candidates);
    if (cnt == 0) {
        arrfree(candidates);
        candidates = NULL;
        return 0;
    }

    // Se k >= cnt prendiamo tutti
    if (k >= cnt) {
        for (int i = 0; i < cnt; i++)
            out[i] = candidates[i];
        arrfree(candidates);
        candidates = NULL;
        return cnt;
    }

    // altrimenti fisher-yates partial shuffle per estrarre k elementi casuali senza ripetizioni
    for (int i = 0; i < k; i++) {
        int j = i + rand() % (cnt - i);
        // scambia i,j
        CellPos tmp = candidates[i];
        candidates[i] = candidates[j];
        candidates[j] = tmp;
        out[i] = candidates[i];
    }

    arrfree(candidates);
    candidates = NULL;

    return k;
}

/* helper: convert index -> row,col and viceversa */
inline CellPos grid_index_to_cellpos(Grid *grid, int idx, CellPos *cell)
{
    *cell = (CellPos){idx / grid->numCols, idx % grid->numCols};
    return *cell;
}

inline int grid_cellpos_to_index(Grid *grid, CellPos cell)
{
    if (cell.r < 0)
        cell.r = (cell.r % grid->numRows + grid->numRows) % grid->numRows;
    if (cell.c < 0)
        cell.c = (cell.c % grid->numCols + grid->numCols) % grid->numCols;
    return cell.r * grid->numCols + cell.c;
}

/* helper: convert row,col -> x,y (cell's top left+grid->margin) */
inline Vector2 grid_cellpos_to_vec2(Grid *grid, CellPos cp)
{
    Vector2 ret = {0};
    ret.x = grid->margin + cp.c * grid->cellWidth;
    ret.y = grid->margin + cp.r * grid->cellHeight;
    return ret;
}

/* Find neuron by clicking raster: map x,y to time and neuron id */
int cell_index_from_grid_click(Grid *grid, int click_x, int click_y, int rx, int ry, int rw, int rh)
{
    int rel_x = click_x - rx;
    int rel_y = click_y - ry;

    /* If click outside grid area return -1 */
    if (click_x <= rx || click_x >= rx+rw || click_y <= ry || click_y >= ry+rh)
        return -1;

    CellPos cell = {
            map(rel_y, 0, grid->height, 0, grid->numRows),
            map(rel_x, 0, grid->width,  0, grid->numCols)
    };

    int cell_idx = cell.r * grid->numCols + cell.c;

    if (cell_idx < 0)
        cell_idx = 0;
    if (cell_idx >= grid->numCells)
        cell_idx = grid->numCells-1;

    return cell_idx;
}

// Restituisce 1 se la cella (r,c) è in [rmin,rmax] (inclusi) rispetto a centro (rc,cc)
int grid_in_annulus(Grid *grid, CellPos cell, CellPos center_cell, int rmin, int rmax)
{
    int dsq = grid_toroidal_dist_sq(grid, cell, center_cell);
    return (dsq >= rmin*rmin && dsq <= rmax*rmax);
}

