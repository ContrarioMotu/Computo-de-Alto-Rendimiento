#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define Nx 100
#define Ny 100
#define Mx 100
#define My 100

/**
 * Función que obtiene la hora en segundos.
 *
 * @returns El tiempo actual del cpu en segundos.
 */
double cpuTime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-06);
}

/**
 * Función que inicializa una matriz con números aleatorios entre 1 y 100.
 *
 * @param x: Cantidad de filas de la matriz.
 * @param y: Cantidad de columnas de la matriz.
 * @param c: Si c == 0, se le asigna 0 a cada elemento de la
 * matriz, si c != 0 se asigna un número aleatorio.
 */
float** initialize(int x, int y, short c)
{
    float **A = (float **)malloc(x * sizeof(float*));
    
    for (int i = 0; i < x; i++)
    {
        A[i] = (float *)malloc(y * sizeof(float));
        for (int j = 0; j < y; j++){
            if (c == 0)
            {
                A[i][j] = 0;
                break;
            }

            A[i][j] = rand() % 100 + 1;
        }
            
    }
    return A;
}

/**
 * Función que calcula el producto de dos matrices, en CPU.
 * 
 * @param A: Matriz A.
 * @param B: Matriz B.
 * @param C: Matriz donde se almacenará el resultado de A x B.
 */
void MatrixProd(float **A, float **B, float **C, int ax, int ay, int bx, int by){

    if (ay != bx)
    {
     printf("ERROR: La cantidad de columnas de A debe ser igual a la cantidad de filas de B\n");
     return;
    }

    for (int i = 0; i < ax; i++)
    {
        for (int j = 0; j < by; j++)
        {
            for (int k = 0; k < ay; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }        
    }
}

void printMatrix(float **A, int ax, int ay){

    if (ax > 5)
    {
        for (int i = 0; i < 5; i++)
        {
            if (ay > 5)
            {
                printf("[");
                for (int j = 0; j < 5; j++)
                {
                    printf("%f, ", A[i][j]);
                }
                printf("...]\n");
            }
        }
        printf("...\n");
        return;
    }
    
    for (int i = 0; i < ax; i++)
    {
        printf("[");
        for (int j = 0; j < ay; j++)
        {
            printf("%f ", A[i][j]);
        }
        printf("]\n");
    }
}

int main(){

    // 1. Asignación e inicialización de memoria dinámica.
    // -------------- CPU ---------------
    float **h_A, **h_B, **h_C;

    h_A = initialize(Nx, Ny, 1);
    h_B = initialize(Mx, My, 1);
    h_C = initialize(Nx, My, 0);
    // ----------------------------------

    double tic, toc, timeCPU;

    // Multiplicación de matrices.
    // ---------------------------
    tic = cpuTime();
    MatrixProd(h_A, h_B, h_C, Nx, Ny, Mx, My);
    toc = cpuTime();
    timeCPU = toc - tic;
    printf("CPU time: %f segs.\n", timeCPU);
    // ---------------------------

    printf("A =\n");
    printMatrix(h_A, Nx, Ny);
    printf("B =\n");
    printMatrix(h_B, Mx, My);
    printf("C =\n");
    printMatrix(h_C, Nx, My);

    // Liberación de memoria.
    // ----------------------
    free(h_A);
    free(h_B);
    free(h_C);
    // ----------------------

    return 0;
}
