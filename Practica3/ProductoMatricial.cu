#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define Nx 1000
#define Ny 1000
#define Mx 1000
#define My 1000

/**
 * Función que inicializa una matriz con números aleatorios entre 1 y 100.
 *
 * @param x: Cantidad de filas de la matriz.
 * @param y: Cantidad de columnas de la matriz.
 * @param c: Si c == 0, se le asigna 0 a cada elemento de la
 * matriz, si c != 0 se asigna un número aleatorio.
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
 * Función que realiza la conversión de una matriz a un arreglo, es decir,
 * de un arreglo de 2 dimensiones a un arreglo de 1 dimensión.
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
__global__ void prodOnGPU(float *A, float *B, float *C, int ax, int ay, int bx, int by)
{
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    int idy = threadIdx.y + (blockIdx.y * blockDim.y);

    if (idx < ax*by)
    {
        for (int i = 0; i < count; i++)
        {
            /* code */
        }
        
        C[in] += A[idx] * B[idy];
    }
}

/**
 * Función que calcula el producto de dos matrices, en CPU.
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
void validate(float *res_GPU, float **res_CPU, int x, int y)
{
    double epsilon = 1.0E-8;
    for (int i = 0; i < x; i++)
    {
        for (int j = 0; j < y; j++)
        {
            if (abs(res_GPU[(x * i) + j] - res_CPU[i][j]) > epsilon)
            {
                printf("ERROR: producto distinto\n");
                return;
            }
        }
    }

    printf("PRODUCTO CORRECTO\n");
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
double speedUp(double h_time, double d_time)
{
    return h_time / d_time;
}

int main()
{

    // 1. Asignación de memoria dinámica
    // -------------- CPU --------------
    float **h_A, **h_B, **h_C, *h_VA, *h_VB, *h_res;

    float *d_A, *d_B, *d_C;

    //--------------- GPU --------------
    cudaMalloc((float **)&d_A, Nx * Ny * sizeof(float));
    cudaMalloc((float **)&d_B, Mx * My * sizeof(float));
    cudaMalloc((float **)&d_C, Nx * My * sizeof(float));
    //----------------------------------

    // 1. Inicialización de datos
    h_A = initialize(Nx, Ny, 1);
    h_B = initialize(Mx, My, 1);
    h_C = initialize(Nx, My, 0);
    h_VA = new float[Nx * Ny];
    h_VB = new float[Nx * Ny];
    h_res = new float[Nx * Ny];

    matrixToVector(h_A, h_VA, Nx, Ny);
    matrixToVector(h_B, h_VB, Nx, Ny);
    matrixToVector(h_C, h_res, Nx, My);

    //memset(h_VA, 0, Nx * Ny * sizeof(float));
    //memset(h_VB, 0, Mx * My * sizeof(float));
    //memset(h_res, 0, Nx * My * sizeof(float));
    // --------------------------

    // 3. Transferencia de datos, del Host al Device (CPU a GPU)
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_res, nBytes, cudaMemcpyHostToDevice);
    //----------------------------------------------------------

    double tic, toc, timeCPU, timeGPU;

    // Producto de matrices en GPU.
    // ----------------------------------------------------------
    dim3 block(1024);
    dim3 thread(1024);
    tic = cpuTime();
    prodOnGPU<<block, thread>>(h_A, h_B, h_C, Nx, Ny, Mx, My);
    toc = cpuTime();
    timeGPU = toc - tic;
    printf("GPU time: %lf segs.\n", timeGPU);
    // ----------------------------------------------------------

    // 3. Transferencia de datos, del Device al Host (GPU a CPU)
    cudaMemcpy(h_res1, d_C, nBytes, cudaMemcpyDeviceToHost);
    //----------------------------------------------------------

    // 2. Suma tradicional de matrices en CPU.
    // ---------------------------------------
    tic = cpuTime();
    prodOnCPU(h_A, h_B, h_C, Nx, Ny, Mx, My);
    toc = cpuTime();
    timeCPU = toc - tic;
    printf("CPU time: %lf segs.\n", timeCPU);
    // ---------------------------------------

    // 4. Verificación de resultados.
    // ------------------------------
    validate(h_res, h_C, Nx, Ny);
    // ------------------------------

    // 6. SpeedUp.
    // -----------
    printf("SpeedUp : %lf\n", speedUp(timeCPU, timeGPU));
    //------------

    // Liberación de memoria.
    // ----------------------
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_VA);
    free(h_VB);
    free(h_res);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    // ----------------------

    return 0;
}