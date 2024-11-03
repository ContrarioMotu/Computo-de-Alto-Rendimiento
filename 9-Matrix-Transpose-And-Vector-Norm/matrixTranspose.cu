#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>
#include <cuda_runtime.h>

/*
 * Implementación de un programa en CUDA que calcula la transpuesta de una matriz, de forma
 * paralela, comparando los resultados obtenidos en la CPU y la GPU, y obteniendo el SpeedUp.
 * 
 * @author: Ayala Morales Mauricio
 */

#define ROWS 10000
#define COLS 20000

/**
 * Inicializa una matriz con números aleatorios entre 1 y 100.
 *
 * @param x: Cantidad de filas de la matriz.
 * @param y: Cantidad de columnas de la matriz.
 */
float **initialize(int x, int y, int c = 0)
{
    float **A = (float **)malloc(x * sizeof(float *));

    for (int i = 0; i < x; i++)
    {
        A[i] = (float *)malloc(y * sizeof(float));
        for (int j = 0; j < y; j++)
        {
            if (!c)
            {
                A[i][j] = 0.0f;
                continue;
            }
            A[i][j] = rand() % 100 + 1;
        }
    }
    return A;
}

/**
 * Realiza la conversión de una matriz a un arreglo, es decir, de un arreglo de 2 dimensiones a un
 * arreglo de 1 dimensión.
 *
 * @param A: Matriz a convertir.
 * @param V: Vector donde se almacenará la matriz.
 * @param x: Cantidad de filas de la matriz.
 * @param y: Cantidad de columnas de la matriz.
 */
void matrixToVector(float **A, float *V, int x, int y)
{
    for (int i = 0; i < x; i++)
    {
        for (int j = 0; j < y; j++)
        {
            V[(y * i) + j] = A[i][j];
        }
    }
}

/**
 * (Kernel) Calcula la matriz transpuesta, en su representación vectorial, de una matriz de números
 * aleatorios utilizando la GPU.
 *
 * @param A: Representación vectorial de la matriz a transponer.
 * @param B: Representación vectorial donde se almacenará la matriz transpuesta.
 * @param rows: Cantidad de filas de la matriz transpuesta.
 * @param cols: Cantidad de columnas de la matriz transpuesta.
 */
__global__ void transposeOnGPU(float *A, float *B, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows || col >= cols) return;

    B[(row * cols) + col] = A[(col * rows) + row];
    __syncthreads();
}

/**
 * Calcula la transpuesta de una matriz, en CPU.
 * 
 * Dada una matriz 'A', su transpuesta 'A^T' es tal que para i, j,
 * A^T[i][j] = A[j][i].
 *
 * @param A: Matriz a transponer.
 * @param B: Matriz donde se almacenará la transpuesta.
 */
void transposeOnCPU(float **A, float **B)
{
    int bx = COLS; //sizeof(B) / sizeof(B[0]);
    int by = ROWS; //sizeof(B[0]) / sizeof(B[0][0]);

    for (int i = 0; i < bx; i++)
    {
        for (int j = 0; j < by; j++)
        {
            B[i][j] = A[j][i];
        }
    }
}

/**
 * Utilidad para obtener la hora en segundos.
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
 * Valida los resultados de la operación, obtenidos en la GPU y CPU.
 *
 * @param res_GPU: Resultado obtenido por la GPU, en forma vectorial.
 * @param res_CPU: Resultado obtenido por la CPU, en forma matricial.
 * @param x: Cantidad de filas de la matriz.
 * @param y: Cantidad de columnas de la matriz.
 */
void validate(float *res_GPU, float **res_CPU, int x, int y)
{
    double epsilon = 1.0E-8;
    for (int i = 0; i < x; i++)
    {
        for (int j = 0; j < y; j++)
        {
            if (abs(res_GPU[(y * i) + j] - res_CPU[i][j]) > epsilon)
            {
                printf("[%d]: %2.f != [%d][%d]: %2.f\n", (y * i) + j, i, j, res_GPU[(y * i) + j], res_CPU[i][j]);
                printf("ERROR: RESULTADO DISTINTO\n");
                return;
            }
        }
    }

    printf("RESULTADO EXITOSO\n");
}

/**
 * Calcula el SpeedUp.
 *
 * @param h_time: Tiempo utilizado por el Host (CPU) para realizar la operación.
 * @param d_time: Tiempo utilizado por el Host (GPU) para realizar la operación.
 *
 * @returns El SpeedUp obtenido del algoritmo ejecutado en paralelo en comparación con su ejecución
 * en serie en una sola unidad de procesamiento.
 */
double speedUp(double h_time, double d_time)
{
    return h_time / d_time;
}

void printMatrix(float **A, int x, int y){
    for(int i=0; i<x; i++){
        printf("[");
        for(int j=0; j<y; j++){
            printf("%.2f ", A[i][j]);
        }
        printf("]\n");
    }
}

void printVector(float *A, int size){
    printf("[");
    for(int i=0; i < size; i++){
        printf("%.2f ", A[i]);
    }
    printf("]\n");
}

int main()
{

    // 1. Inicialización de datos
    // -------------- CPU --------------
    dim3 block(16, 16);
    dim3 grid((ROWS + block.x - 1 ) / block.x, (COLS + block.y - 1 ) / block.y);
    printf("Grid: %d x %d\n", grid.x, grid.y);
    printf("Block: %d x %d\n", block.x, block.y);
    size_t nBytes = ROWS * COLS * sizeof(float);
    float **h_A, **h_B, *h_VA, *h_res;

    float *d_A, *d_B;
    // ---------------------------------


    /// 2. Asignación de memoria dinámica
    // ----------------------------------
    h_A = initialize(ROWS, COLS, 1);
    h_B = initialize(COLS, ROWS, 0);
    h_VA = new float[ROWS * COLS];
    h_res = new float[ROWS * COLS];

    matrixToVector(h_A, h_VA, ROWS, COLS);
    matrixToVector(h_B, h_res, COLS, ROWS);
    
    //--------------- GPU ---------------
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    // ----------------------------------

    // 3. Transferencia de datos, del Host al Device (CPU a GPU)
    cudaMemcpy(d_A, h_VA, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_res, nBytes, cudaMemcpyHostToDevice);
    //----------------------------------------------------------

    double tic, toc, timeCPU, timeGPU;

    // 5. Configuración de la ejecución en la GPU.
    // -------------------------------------------
    tic = cpuTime();
    transposeOnGPU<<<grid, block>>>(d_A, d_B, COLS, ROWS);
    cudaDeviceSynchronize();
    toc = cpuTime();
    timeGPU = toc - tic;
    printf("GPU time: %lf segs.\n", timeGPU);
    // -------------------------------------------

    // 6. Transferencia de datos, del Device al Host (GPU a CPU)
    cudaMemcpy(h_res, d_B, nBytes, cudaMemcpyDeviceToHost);
    //----------------------------------------------------------

    // Transposición de matriz en CPU.
    // -------------------------------
    tic = cpuTime();
    transposeOnCPU(h_A, h_B);
    toc = cpuTime();
    timeCPU = toc - tic;
    printf("CPU time: %lf segs.\n", timeCPU);
    // -------------------------------

    // 7. Validación de resultados.
    // ------------------------------
    validate(h_res, h_B, COLS, ROWS);
    // ------------------------------

    // 8. Liberación de memoria.
    // ----------------------
    free(h_A); free(h_B); free(h_VA); free(h_res);

    cudaFree(d_A); cudaFree(d_B);
    // ----------------------

    // 9. SpeedUp.
    // -----------
    printf("SpeedUp : %lf\n", speedUp(timeCPU, timeGPU));
    // -----------

    return 0;
}