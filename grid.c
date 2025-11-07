#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdbool.h>

#include "grid.h"

/* Trova la migliore coppia cols,rows per N celle in un'area width x height */
static GridChoice choose_grid_aroundN(int N, int width, int height)
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
static GridChoice choose_grid_exactN(int N, int width, int height)
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

GridChoice choose_grid(int N, int width, int height, bool strict)
{
    GridChoice ret = {0};

    if (N <= 0 || width <= 0 || height <= 0) {
        ret.totalCells = -1;
        return ret;
    }

    printf("Target N (numero di celle): %d\n", N);
    printf("Grid width x height (px): %d x %d\n", width, height);

    if (strict) {
        ret = choose_grid_exactN(N, width, height);
    } else {
        ret = choose_grid_aroundN(N, width, height);
    }

    if (ret.cols == 0) {
        printf("Nessuna configurazione trovata.\n");
        ret.totalCells = -1;
        return ret;
    }

    printf("(%s choosing %d cells)\n", strict ? "strictly" : "loosely", N);
    printf("\nChosen: cols=%d, rows=%d, num cells=%d\n", ret.cols, ret.rows, ret.totalCells);
    printf("Dimensione cella: %.3f x %.3f px (rapporto W/H = %.3f)\n",
            ret.cellW, ret.cellH, ret.cellW / ret.cellH);

    return ret;
}
