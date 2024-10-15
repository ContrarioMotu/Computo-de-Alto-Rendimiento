#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define Nx 1000
#define Ny 1000

/**
 * Función que inicializa un arreglo con números aleatorios entre 1 y 100.
 */
void initialize(float **A, int x, int y){
    srand(0);
    for (int i = 0; i < x; i++){
        for (int j = 0; j < y; j++){
            A[i][j] = rand() % 100 + 1;
        }
    }
}

/** 
 * Función que realiza la suma de dos matrices de números aleatorios, de manera
 * tradicional (Row-Major-Order), utilizando la CPU.
 *
 * @param A: Primer matriz a sumar.
 * @param B: Segunda matriz a sumar.
 * @param x: Cantidad de filas de la matriz.
 * @param y: Cantidad de columnas de la matriz.
 */
void rowSumOnCPU(float **A, float **B, float **C, int x, int y){
    if (x != y){
        printf("ERROR: No se puede realizar la suma de matrices de diferentes dimensiones");
        return;
    }

    for (int i = 0; i < x; i++){
        for (int j = 0; j < y; j++){
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}


/** 
 * Función que realiza la suma de dos matrices de números aleatorios, por
 * columnas (Row-Major-Order), utilizando la CPU.
 *
 * @param A: Primer matriz a sumar.
 * @param B: Segunda matriz a sumar.
 * @param x: Cantidad de filas de la matriz.
 * @param y: Cantidad de columnas de la matriz.
 */
void columnSumOnCPU(float **A, float **B, float **C, int x, int y){
    if (x != y){
        printf("ERROR: No se puede realizar la suma de matrices de diferentes dimensiones");
        return;
    }

    for (int i = 0; i < x; i++){
        for (int j = 0; j < y; j++){
            C[j][i] = A[j][i] + B[j][i];
        }
    }
}


/**
 * Función que realiza la conversión de una matriz a un arreglo, es decir,
 * de un arreglo de 2 dimensiones a un arreglo de 1 dimensión.
 * 
 * @param A: Matriz a convertir.
 * @param V: Vector donde se almacenará la matriz.
 * @param x: Cantidad de filas de la matriz.
 * @param y: Cantidad de columnas de la matriz.
 */
void matrixToVector(float **A, float *V, int x, int y){
    for (int i = 0; i < x; i++){
        for (int j = 0; j < y; j++){
            V[(x * i) + j] = A[i][j];
        }
    }
}

/**
 * 4. Kernel en CUDA que realiza la suma de dos arreglos de números aleatorios utilizando la GPU.
 *
 * @param A: Primer arreglo a sumar.
 * @param B: Segundo arreglo a sumar.
 * @param C: Arreglo donde se almacenará el resultado de la suma.
 * @param size: Tamaño de los arreglos a sumar.
 */
__global__ void sumOnGPU(float *A, float *B, float *C, int size){
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx < size){
        C[idx] = A[idx] + B[idx];
    }
    
}

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
 * Función que valida los resultados de la suma, obtenidos en la GPU y CPU.
 *
 * @param res_GPU: Resultado de la suma obtenida por la GPU, en forma de vector.
 * @param res_CPU: Resultado de la suma obtenida por la CPU, en forma matricial.
 * @param x: Cantidad de filas de la matriz.
 * @param y: Cantidad de columnas de la matriz.
 */
void validate(float *res_GPU, float **res_CPU, int x, int y){
    double epsilon = 1.0E-8;
    for (int i = 0; i < x; i++){
        for (int j = 0; j < y; j++){
            if (abs(res_GPU[(x*i) + j] - res_CPU[i][j]) > epsilon){
                printf("ERROR: suma distinta\n");
                return;
            }
        }
        
    }
    
    printf("SUMA CORRECTA\n");
}

/**
 * Función que calcula el SpeedUp.
 *
 * @param h_time: Tiempo utilizado por el Host (CPU) para realizar la operación.
 * @param d_time: Tiempo utilizado por el Host (GPU) para realizar la operación.
 *
 * @returns El SpeedUp obtenido del algoritmo ejecutado en paralelo en
 * comparación con su ejecución en serie en una sola unidad de procesamiento.
 */
double speedUp(double h_time, double d_time){
    return h_time / d_time;
}

int main(){

    // 1. Asignación de memoria dinámica
    // -------------- CPU --------------
    float **h_A, **h_B, **h_C, **h_D, *h_VA, *h_VB, *h_res;

    h_A = new float*[Nx];
    h_B = new float*[Nx];
    h_C = new float*[Nx];
    h_D = new float*[Nx];
    for (int i = 0; i < Nx; i++)
    {
        h_A[i] = new float[Ny];
        h_B[i] = new float[Ny];
        h_C[i] = new float[Ny];
        h_D[i] = new float[Ny];
    }
    
    h_VA = new float[Nx * Ny];
    h_VB = new float[Nx * Ny];
    h_res = new float[Nx * Ny];

    float *d_A, *d_B, *d_C;

    //--------------- GPU --------------
    cudaMalloc((float **)&d_A, Nx * Ny * sizeof(float));
    cudaMalloc((float **)&d_B, Nx * Ny * sizeof(float));
    cudaMalloc((float **)&d_C, Nx * Ny * sizeof(float));
    //----------------------------------

    // 1. Inicialización de datos
    initialize(h_A, Nx, Ny);
    initialize(h_B, Nx, Ny);
    initialize(h_C, Nx, Ny);
    initialize(h_D, Nx, Ny);
    
    memset(h_VA, 0, Nx * Ny * sizeof(float));
    memset(h_VB, 0, Nx * Ny * sizeof(float));
    memset(h_res, 0, Nx * Ny * sizeof(float));
    // --------------------------
    
    double tic, toc, timeCPUCol, timeCPURow, timeGPU;

    // Suma de matrices por columnas (Column-Major-Order) en CPU.
    // 5.1. Medición del tiempo de la CPU.
    // ----------------------------------------------------------
    tic = cpuTime();
    columnSumOnCPU(h_A, h_B, h_C, Nx, Ny);
    toc = cpuTime();
    timeCPUCol = toc - tic;
    printf("CPU time (Column-Major-Order): %lf segs.\n", timeCPUCol);
    // ----------------------------------------------------------

    // 2. Suma tradicional de matrices en CPU.
    // 5.2. Medición del tiempo de la CPU.
    // ---------------------------------------
    tic = cpuTime();
    rowSumOnCPU(h_A, h_B, h_D, Nx, Ny);
    toc = cpuTime();
    timeCPURow = toc - tic;
    printf("CPU time (Row-Major-Order): %lf segs.\n", timeCPURow);
    // ---------------------------------------

    // 3. Vectorización y suma vectorial en GPU.
    // 5.3. Medición del tiempo de la GPU.
    // -----------------------------------------
    matrixToVector(h_A, h_VA, Nx, Ny);
    matrixToVector(h_B, h_VB, Nx, Ny);

    cudaMemcpy(d_A, h_VA, Nx * Ny * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_VB, Nx * Ny * sizeof(float), cudaMemcpyHostToDevice);

    int b = ((Nx * Ny) / 1024) + 1;
    dim3 block(b);
    dim3 thread(1024);
    tic = cpuTime();
    sumOnGPU<<<block, thread>>>(d_A, d_B, d_C, Nx * Ny);
    cudaDeviceSynchronize();
    toc = cpuTime();
    timeGPU = toc - tic;
    printf("GPU time (normal): %lf segs.\n", timeGPU);

    cudaMemcpy(h_res, d_C, Nx * Ny * sizeof(float), cudaMemcpyDeviceToHost);
    // ------------------------------------------

    // 4. Verificación de resultados.
    // ------------------------------
    validate(h_res, h_C, Nx, Ny);
    // ------------------------------

    // 6. SpeedUp.
    // -----------
    printf("SpeedUp (Column-Major-Order): %lf\n", speedUp(timeCPUCol, timeGPU));
    printf("SpeedUp (Row-Major-Order): %lf\n", speedUp(timeCPURow, timeGPU));
    //------------

    // Liberación de memoria.
    // ----------------------
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    free(h_VA);
    free(h_VB);
    free(h_res);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    // ----------------------

    return 0;
}