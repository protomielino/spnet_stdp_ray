#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STB_DS_IMPLEMENTATION
#include "stb_ds.h"

typedef struct
{
    long long rows;
    long long cols;
} gridFactors;

gridFactors factors(long long num, double R)
{
    gridFactors ret = {0};

    if (num <= 0) {
        fprintf(stderr, "N deve essere positivo\n");
        return ret;
    }

    double best_diff = INFINITY;
    // Primo pass: trovare la differenza minima assoluta |(double)a/b - R|
    for (long long a = 1; a * a <= num; ++a) {
        if (num % a == 0) {
            long long b = num / a;
            double ratio1 = (double)a / (double)b;
            double diff1 = fabs(ratio1 - R);
            if (diff1 < best_diff)
                best_diff = diff1;

            // considerare anche la coppia invertita se diversa
            if (a != b) {
                double ratio2 = (double)b / (double)a;
                double diff2 = fabs(ratio2 - R);
                if (diff2 < best_diff)
                    best_diff = diff2;
            }
        }
    }

    // Seconda pass: stampare tutte le coppie che raggiungono best_diff (tolleranza per fp)
    const double eps = 1e-12;
    printf("Fattori di %lld con rapporto vicino a %.12g (diff minima = %.12g):\n", num, R, best_diff);
    for (long long a = 1; a * a <= num; ++a) {
        if (num % a == 0) {
            long long b = num / a;
            double ratio1 = (double)a / (double)b;
            double diff1 = fabs(ratio1 - R);
            if (fabs(diff1 - best_diff) <= eps) {
                printf("%lld x %lld  -> rapporto = %.12g\n", a, b, ratio1);
                ret.cols = a;
                ret.rows = b;
            }
            if (a != b) {
                double ratio2 = (double)b / (double)a;
                double diff2 = fabs(ratio2 - R);
                if (fabs(diff2 - best_diff) <= eps) {
                    printf("%lld x %lld  -> rapporto = %.12g\n", b, a, ratio2);
                    ret.cols = b;
                    ret.rows = a;
                }
            }
        }
    }

    return ret;
}

int main(void)
{
    gridFactors f = factors(10000, 16.0/9.0);

    printf("%lld rows x %lld cols\n", f.rows, f.cols);

    return 0;
}
