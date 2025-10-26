#ifndef PALETTE_H_
#define PALETTE_H_

#include <stdlib.h>
#include <math.h>

#include <raylib.h>

#define CYAN  CLITERAL(Color){ 0, 255, 255, 255 }

/* Tipi estesi: ogni entry ha anche uno "spread" (raggio d'influenza 0..1) */
typedef enum
{
    STOCK_EMPTY,
    STOCK_GREYSCALE,
    STOCK_COLDHOT1,
    STOCK_COLDHOT2,
    STOCK_COLDHOT3,
    STOCK_SPECTRUM,
} Stock;

typedef struct
{
    double first;   // posizione 0..1
    Color  second;  // colore
    double spread;  // semilarghezza d'influenza (0..1)
} ColourEntry;

typedef enum { WEIGHT_LINEAR, WEIGHT_QUADRATIC, WEIGHT_GAUSSIAN, WEIGHT_COSINE } WeightMode;

void Palette_init(ColourEntry **palette, Stock stock);
Color Palette_Sample(ColourEntry **palette, double t);
void Palette_SetColour(ColourEntry **palette, double d, Color col, double spread);
void Palette_SetColourSimple(ColourEntry **palette, double d, Color col);
/* Setter opzionali */
void Palette_SetWeightMode(WeightMode m);
void Palette_SetGaussianSigma(double s);

#endif /* PALETTE_H_ */
