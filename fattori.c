#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void)
{
    long long N;
    double R;
    if (scanf("%lld %lf", &N, &R) != 2) {
        fprintf(stderr, "Uso: inserire N R\n");
        return 1;
    }
    if (N <= 0) {
        fprintf(stderr, "N deve essere positivo\n");
        return 1;
    }

    double best_diff = INFINITY;
    // Primo pass: trovare la differenza minima assoluta |(double)a/b - R|
    for (long long a = 1; a * a <= N; ++a) {
        if (N % a == 0) {
            long long b = N / a;
            double ratio1 = (double)a / (double)b;
            double diff1 = fabs(ratio1 - R);
            if (diff1 < best_diff) best_diff = diff1;

            // considerare anche la coppia invertita se diversa
            if (a != b) {
                double ratio2 = (double)b / (double)a;
                double diff2 = fabs(ratio2 - R);
                if (diff2 < best_diff) best_diff = diff2;
            }
        }
    }

    // Seconda pass: stampare tutte le coppie che raggiungono best_diff (tolleranza per fp)
    const double eps = 1e-12;
    printf("Fattori di %lld con rapporto vicino a %.12g (diff minima = %.12g):\n", N, R, best_diff);
    for (long long a = 1; a * a <= N; ++a) {
        if (N % a == 0) {
            long long b = N / a;
            double ratio1 = (double)a / (double)b;
            double diff1 = fabs(ratio1 - R);
            if (fabs(diff1 - best_diff) <= eps) {
                printf("%lld x %lld  -> rapporto = %.12g\n", a, b, ratio1);
            }
            if (a != b) {
                double ratio2 = (double)b / (double)a;
                double diff2 = fabs(ratio2 - R);
                if (fabs(diff2 - best_diff) <= eps) {
                    printf("%lld x %lld  -> rapporto = %.12g\n", b, a, ratio2);
                }
            }
        }
    }

    return 0;
}
