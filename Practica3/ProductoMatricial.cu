#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define Nrows 1000
#define Ncols 2000
#define Mrows 2000
#define Mcols 3000

/**
 * Inicializa una matriz con números aleatorios entre 1 y 100.
 *
 * @param x: Cantidad de filas de la matriz.
 * @param y: Cantidad de columnas de la matriz.
 * @param c: Si c == 0, se le asigna 0 a cada elemento de la matriz, si c != 0 se asigna un número
 * aleatorio.
 */
float **initialize(int x, int y, short c)
{
    float **A = (float **)malloc(x * sizeof(float *));

    for (int i = 0; i < x; i++)
    {
        A[i] = (float *)malloc(y * sizeof(float));
        for (int j = 0; j < y; j++)
        {
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
 * (Kernel) Realiza el producto de dos matrices en su representación vectorial, de números
 * aleatorios utilizando la GPU.
 *
 * @param A: Primer matriz a multiplicar, en forma de arreglo.
 * @param B: Segunda matriz a multiplicar, en forma de arreglo.
 * @param C: Arreglo donde se almacenará el resultado del pructo.
 * @param Acols: Cantidad de filas de la matriz A.
 * @param k: Cantidad de columnas de la matriz A y filas de la matriz B.
 * @param Brows: Cantidad de columnas de la matriz B.
 */
__global__ void prodOnGPU(float *A, float *B, float *C, int Arows, int k, int Bcols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < Bcols && row < Arows)
    {
        for (int i = 0; i < k; i++)
        {
            C[(row * Bcols) + col] += A[(row * k) + i] * B[(i * Bcols) + col];
        }
    }
}

/**
 * Calcula el producto de dos matrices, en CPU.
 *
 * @param A: Matriz A.
 * @param B: Matriz B.
 * @param C: Matriz donde se almacenará el resultado de A x B.
 */
void prodOnCPU(float **A, float **B, float **C, int ax, int ay, int bx, int by)
{
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
 * Valida los resultados del producto, obtenidos en la GPU y CPU.
 *
 * @param res_GPU: Resultado del producto obtenido por la GPU, en forma de vector.
 * @param res_CPU: Resultado del producto obtenido por la CPU, en forma matricial.
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
                printf("%2.f != %2.f\n", res_GPU[(y * i) + j], res_CPU[i][j]);
                printf("ERROR: producto distinto\n");
                return;
            }
        }
    }

    printf("PRODUCTO CORRECTO\n");
}

void validateMatrices(int ax, int ay, int bx, int by){
    if(ay != bx){
        printf("ERROR: La cantidad de columnas de A debe ser igual a la cantidad de filas de B\n");
        exit(1);
    }
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
    float **h_A, **h_B, **h_C, *h_VA, *h_VB, *h_res;

    float *d_A, *d_B, *d_C;
    // ---------------------------------

    validateMatrices(Nrows, Ncols, Mrows, Mcols);

    /// 2. Asignación de memoria dinámica
    // ----------------------------------
    h_A = initialize(Nrows, Ncols, 1);
    h_B = initialize(Mrows, Mcols, 1);
    h_C = initialize(Nrows, Mcols, 0);
    h_VA = new float[Nrows * Ncols];
    h_VB = new float[Mrows * Mcols];
    h_res = new float[Nrows * Mcols];

    matrixToVector(h_A, h_VA, Nrows, Ncols);
    matrixToVector(h_B, h_VB, Mrows, Mcols);
    matrixToVector(h_C, h_res, Nrows, Mcols);
    
    //--------------- GPU ---------------
    cudaMalloc((float **)&d_A, Nrows * Ncols * sizeof(float));
    cudaMalloc((float **)&d_B, Mrows * Mcols * sizeof(float));
    cudaMalloc((float **)&d_C, Nrows * Mcols * sizeof(float));
    // ----------------------------------

    // 3. Transferencia de datos, del Host al Device (CPU a GPU)
    cudaMemcpy(d_A, h_VA, Nrows * Ncols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_VB, Mrows * Mcols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_res, Nrows * Mcols * sizeof(float), cudaMemcpyHostToDevice);
    //----------------------------------------------------------

    double tic, toc, timeCPU, timeGPU;

    // 5. Configuración de la ejecución en la GPU.
    // -------------------------------------------

    int brows = (int)((Nrows + 31 )/ 32);
    int bcols = (int)((Mcols + 31 )/ 32);
    dim3 grid(bcols, brows);

    int trows = (Nrows <= 32)?  Nrows : 32;
    int tcols = (Mcols <= 32)? Mcols : 32;
    dim3 block(tcols, trows);
    printf("Block: %d x %d\n", bcols, brows);
    printf("Thread: %d x %d\n", tcols, trows);

    tic = cpuTime();
    prodOnGPU<<<grid, block>>>(d_A, d_B, d_C, Nrows, Ncols, Mcols);
    cudaDeviceSynchronize();
    toc = cpuTime();
    timeGPU = toc - tic;
    printf("GPU time: %lf segs.\n", timeGPU);
    // -------------------------------------------

    // 6. Transferencia de datos, del Device al Host (GPU a CPU)
    cudaMemcpy(h_res, d_C, Nrows * Mcols * sizeof(float), cudaMemcpyDeviceToHost);
    printf("cuda Memcpy ready.\n");
    //----------------------------------------------------------

    // Producto de matrices en CPU.
    // -------------------------------
    tic = cpuTime();
    prodOnCPU(h_A, h_B, h_C, Nrows, Ncols, Mrows, Mcols);
    toc = cpuTime();
    timeCPU = toc - tic;
    printf("CPU time: %lf segs.\n", timeCPU);
    // -------------------------------

    // 7. Validación de resultados.
    // ------------------------------
    validate(h_res, h_C, Nrows, Mcols);
    // ------------------------------

    // 8. Liberación de memoria.
    // ----------------------
    free(h_A); free(h_B); free(h_C); free(h_VA); free(h_VB); free(h_res);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    // ----------------------

    // 9. SpeedUp.
    // -----------
    printf("SpeedUp : %lf\n", speedUp(timeCPU, timeGPU));
    // -----------

    return 0;
}