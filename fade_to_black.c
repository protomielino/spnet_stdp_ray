// compile with: gcc fade_to_black.c -o fade_to_black -lm -lraylib
#include <stdlib.h> // srand, rand
#include <time.h>   // time
#include <math.h>

#include <raylib.h>
#define RAYMATH_IMPLEMENTATION
#include <raymath.h>

#define STB_DS_IMPLEMENTATION
#include "stb_ds.h"

#include "math_utils.h"

#define N 100
#define R_SPHERE 20.0f

typedef struct
{
    Vector3 pos;
} CubeItem;

Vector3 random_point_on_sphere(float radius)
{
    // Direzione uniforme sulla sfera usando metodo di randomizzazione
    float u = rand01()*2.0f - 1.0f;            // cos(theta) tra -1 e 1
    float phi = rand01() * 2.0f * PI;
    float theta = acosf(u);
    float x = radius * sinf(theta) * cosf(phi);
    float y = radius * sinf(theta) * sinf(phi);
    float z = radius * u;
    Vector3 v = { x, y, z };
    return v;
}

Color lerpColor(Color a, Color b, float t)
{
    if (t < 0)
        t = 0;
    if (t > 1)
        t = 1;
    Color r;
    r.r = (unsigned char)((1.0f - t) * a.r + t * b.r);
    r.g = (unsigned char)((1.0f - t) * a.g + t * b.g);
    r.b = (unsigned char)((1.0f - t) * a.b + t * b.b);
    r.a = (unsigned char)((1.0f - t) * a.a + t * b.a);
    return r;
}

int main(void)
{
    srand(clock());

    // Inizializzazione
    const int screenW = 1280;
    const int screenH = 700;

    InitWindow(screenW, screenH, "Cubi su sfera - fade by camera distance");
    SetTargetFPS(60);

    // Genero N posizioni sulla sfera
    CubeItem items[N];
    for (int i = 0; i < N; i++) {
        items[i].pos = random_point_on_sphere(R_SPHERE);
        // opzionale: leggero offset radiale casuale per variare distanza reale dalla camera
        // float extra = (rand01() - 0.5f) * 2.0f; // tra -1 e 1
        // items[i].pos.x *= 1.0f + 0.05f * extra;
        // items[i].pos.y *= 1.0f + 0.05f * extra;
        // items[i].pos.z *= 1.0f + 0.05f * extra;
    }

    // colore di base dei cubi (es. blu)
    Color baseColor = (Color){ 0, 150, 255, 255 };
    Color targetBlack = (Color){ 0, 0, 0, 255 };

    Camera3D camera = {
            position:   (Vector3){ 30.0f, 15.0f, 30.0f },
            target:     (Vector3){  0.0f,  0.0f,  0.0f },
            up:         (Vector3){  0.0f,  1.0f,  0.0f },
            fovy:       45.0f,
            projection: CAMERA_PERSPECTIVE
    };

    // parametri per mappare distanza -> fattore colore
    // distanza_min -> fattore 1 (colore pieno)
    // distanza_max -> fattore 0 (nero)
    float distance_min = 0.0f;
    float distance_max = 60.0f; // dipende dalla scena

//    SetCameraMode(camera, CAMERA_FREE);

    DisableCursor();

    while (!WindowShouldClose()) {
        UpdateCamera(&camera, CAMERA_FREE);

        BeginDrawing(); {
            ClearBackground(BLACK);

            BeginMode3D(camera); {
                // asse/ground per riferimento
                DrawGrid(20, 1.0f);

                // per ogni cubo, calcolo distanza dalla camera e derivo il colore
                for (int i = 0; i < N; i++) {
                    Vector3 p = items[i].pos;
                    float dist = Vector3Distance(camera.position, p);

                    // mappa la distanza in fattore t tra 0..1 -- 1 = colore, 0 = nero
                    float t;
                    if (dist <= distance_min)
                        t = 1.0f;
                    else if (dist >= distance_max)
                        t = 0.0f;
                    else
                        t = 1.0f - (dist - distance_min) / (distance_max - distance_min);

                    // (opzionale): easing (quadratico) -> sfumatura più morbida
                    // t = t * t;

                    Color c = lerpColor(targetBlack, baseColor, t);

                    DrawCube(p, 1.0f, 1.0f, 1.0f, c);
                    DrawCubeWires(p, 1.0f, 1.0f, 1.0f, Fade(BLACK, 0.2f));
                }

            } EndMode3D();

            // UI minima
            DrawText("WASD + mouse per muovere camera. Scroll per zoom. Esc per uscire.", 10, 10, 20, DARKGRAY);

        } EndDrawing();
    }

    CloseWindow();
    return 0;
}
