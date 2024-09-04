#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

/*
 * Implementación de un programa en CUDA que realiza la suma de dos vectores de forma
 * paralela, comparando los resultados obtenidos en la CPU y la GPU, y obteniendo el SpeedUp.
 * 
 * @author: Ayala Morales Mauricio
 */

#define N 10000000

/* 
 * Función que inicializa un arreglo con números aleatorios entre 1 y 100.
 *
 * @param A: Arreglo a inicializar.
 * @param size: Tamaño del arreglo.
 */
void initialize(float *A, int size){
    srand(0);

    for (int i = 0; i < size; i++)
    {
        A[i] = rand() % 100 + 1;
    }
}

/* 
 * Función que realiza la suma de dos arreglos de números aleatorios utilizando la CPU.
 *
 * @param A: Primer arreglo a sumar.
 * @param B: Segundo arreglo a sumar.
 * @param size: Tamaño de los arreglos a sumar.
 */
void sumOnCPU(float *A, float *B, float *C, int size){
    for (int i = 0; i < size; i++)
    {
        C[i] = A[i] + B[i];
    }
}

/* 
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

/* 
 * 4. Kernel en CUDA que realiza la suma de dos arreglos de números aleatorios utilizando la GPU,
 * donde cada hilo realiza la suma de dos elementos de cada vector.
 *
 * @param A: Primer arreglo a sumar.
 * @param B: Segundo arreglo a sumar.
 * @param C: Arreglo donde se almacenará el resultado de la suma.
 * @param size: Tamaño de los arreglos a sumar.
 */
__global__ void doubSumOnGPU(float *A, float *B, float *C, int size){
    int idx = (threadIdx.x + (blockIdx.x * blockDim.x)) * 2;
    if (idx < size){
        C[idx] = A[idx] + B[idx];
        C[idx + 1] = A[idx + 1] + B[idx + 1];
    }
    
}

/*
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

/*
 * Función que valida los resultados de la suma, obtenidos en la GPU y CPU.
 *
 * @param res_GPU: Resultado de la suma obtenida por la GPU.
 * @param res_CPU: Resultado de la suma obtenida por la CPU.
 * @param size: Tamaño de los arreglos con los resultados.
 */
void validate(float *res_GPU, float *res_CPU, int size){
    double epsilon = 1.0E-8;
    for (int i = 0; i < size; i++){
        if (abs(res_GPU[i] - res_CPU[i]) > epsilon){
            printf("ERROR: suma distinta\n");
            return;
        }
    }
    
    printf("SUMA CORRECTA\n");
}

/*
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

    // 2. Asignación de memoria dinámica
    // -------------- CPU --------------

    size_t nBytes = N * sizeof(float);
    float *h_A, *h_B, *h_C, *h_res1, *h_res2;

    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);
    h_res1 = (float *)malloc(nBytes);
    h_res2 = (float *)malloc(nBytes);

    float *d_A, *d_B, *d_C;

    //--------------- GPU --------------
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_C, nBytes);
    //----------------------------------

    // 1. Inicialización de datos
    initialize(h_A, N);
    initialize(h_B, N);

    memset(h_C, 0, nBytes);
    memset(h_res1, 0, nBytes);
    memset(h_res2, 0, nBytes);
    // --------------------------

    // 3. Transferencia de datos, del Host al Device (CPU a GPU)
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, nBytes, cudaMemcpyHostToDevice);
    //----------------------------------------------------------

    double tic, toc, timeCPU, timeGPU, timeGPUDoubleSum;

    // 5. Configuración de la ejecución en la GPU
    int b = (N / (1024 * 2)) + 1;
    dim3 block(b);
    dim3 thread(1024);
    tic = cpuTime();
    doubSumOnGPU<<<block, thread>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    toc = cpuTime();
    timeGPUDoubleSum = toc - tic;
    printf("GPU time: %lf segs.\n", timeGPUDoubleSum);
    //-------------------------------------------

    // 3. Transferencia de datos, del Device al Host (GPU a CPU)
    cudaMemcpy(h_res1, d_C, nBytes, cudaMemcpyDeviceToHost);
    //----------------------------------------------------------

    tic = cpuTime();
    sumOnCPU(h_A, h_B, h_C, N);
    toc = cpuTime();
    timeCPU = toc - tic;
    printf("CPU time: %lf segs.\n", timeCPU);

    //7. Validación de los resultados
    validate(h_res1, h_C, N);
    //-------------------------------

    // 5. Configuración de la ejecución en la GPU
    b = (N / 1024) + 1;
    dim3 block2(b);
    dim3 thread2(1024);
    tic = cpuTime();
    sumOnGPU<<<block2, thread2>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    toc = cpuTime();
    timeGPU = toc - tic;
    printf("GPU time (normal): %lf segs.\n", timeGPU);
    //-------------------------------------------

    // 3. Transferencia de datos, del Device al Host (GPU a CPU)
    cudaMemcpy(h_res2, d_C, nBytes, cudaMemcpyDeviceToHost);
    //----------------------------------------------------------

    //7. Validación de los resultados
    validate(h_res1, h_res2, N);
    //-------------------------------

    // 8. Liberación de memoria
    // --------- CPU ----------
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_res1);
    free(h_res2);

    // --------- GPU ----------
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    //-------------------------

    // 9. SpeedUp
    printf("SpeedUp: %lf\n", speedUp(timeCPU, timeGPUDoubleSum));

    return 0;
}