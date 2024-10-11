#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <iostream>

/*
 * Implementación de un programa en CUDA que realiza el producto punto de dos vectores de forma
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
void initialize(long *A, int size){
    //srand(0);

    for (int i = 0; i < size; i++)
    {
        A[i] = rand() % 10 + 1;
    }
}

/* 
 * Realiza el producto punto de dos arreglos de números aleatorios utilizando la CPU.
 *
 * @param A: Primer arreglo.
 * @param B: Segundo arreglo.
 * @param size: Tamaño de los arreglos.
 */
long dotPointOnCPU(long *A, long *B, int size){
    long C = 0;
    for (int i = 0; i < size; i++)
    {
        C += A[i] * B[i];
    }
    return C;
}

/* 
 * 4. (Kernel) Realiza le producto punto de dos arreglos de números aleatorios
 * utilizando la GPU.
 *
 * @param A: Primer arreglo.
 * @param B: Segundo arreglo
 * @param C: Apuntador a la variable que almacenará el resultado del producto punto.
 * @param size: Tamaño de los arreglos.
 */
__global__ void dotPointOnGPU(long *A, long *B, long *C, int size){
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    int tid = threadIdx.x;
    if (idx < size){
        A[idx] = A[idx] * B[idx];
        __syncthreads();
    }
    int index;
    int stride = blockDim.x / 2;
    while(stride > 0){
        index = idx + stride;
        if (index < size){
            A[idx] += A[index];
        }
        stride = stride / 2 ;
        __syncthreads();
    }
    
    if(tid == 0){
        C[blockIdx.x] = A[blockDim.x * blockIdx.x];
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
 * Función que valida los resultados obtenidos en la GPU y CPU.
 *
 * @param res_GPU: Resultado obtenido por la GPU.
 * @param res_CPU: Resultado obtenido por la CPU.
 * @param size: Tamaño de los arreglos con los resultados.
 */
void validate(long res_GPU, long res_CPU, int size){
    double epsilon = 1.0E-6;
    if (abs(res_GPU - res_CPU) > epsilon){
        std::cout << "ERROR: producto distinto\n";
        std::cout << res_GPU << " != " << res_CPU << "\n";
        return;
    }
    
    printf("PRODUCTO CORRECTO\n");
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
    int threads = (N <= 1024)? N : 1024;
    int blocks = (N + threads - 1)/ threads;
    size_t nBytes = N * sizeof(long);
    long *h_A, *h_B, *h_C;

    h_A = (long *)malloc(nBytes);
    h_B = (long *)malloc(nBytes);
    h_C = (long *)malloc(blocks * sizeof(long));

    long *d_A, *d_B, *d_C;

    //--------------- GPU --------------
    cudaMalloc((long **)&d_A, nBytes);
    cudaMalloc((long **)&d_B, nBytes);
    cudaMalloc((long **)&d_C, blocks * sizeof(long));
    //----------------------------------

    // 1. Inicialización de datos
    initialize(h_A, N);
    initialize(h_B, N);

    //memset(h_C, 0, nBytes);
    // --------------------------

    // 3. Transferencia de datos, del Host al Device (CPU a GPU)
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, blocks * sizeof(long), cudaMemcpyHostToDevice);
    //----------------------------------------------------------

    double tic, toc, timeCPU, timeGPU;

    // 5. Configuración de la ejecución en la GPU
    std::cout << "Grid: " << blocks << "\nThreads: " << threads << "\n";
    dim3 block(blocks);
    dim3 thread(threads);
    tic = cpuTime();
    dotPointOnGPU<<<block, thread>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    toc = cpuTime();
    timeGPU = toc - tic;
    std::cout << "GPU time:" << timeGPU << " segs.\n";
    //-------------------------------------------

    // 3. Transferencia de datos, del Device al Host (GPU a CPU)
    cudaMemcpy(h_C, d_C, blocks * sizeof(long), cudaMemcpyDeviceToHost);
    //----------------------------------------------------------

    long res_GPU = 0;
    for(int i = 0; i < blocks; i++){
        res_GPU += h_C[i];
    }

    tic = cpuTime();
    long res_CPU = dotPointOnCPU(h_A, h_B, N);
    toc = cpuTime();
    timeCPU = toc - tic;
    std::cout << "CPU time:" << timeCPU << " segs.\n";

    //7. Validación de los resultados
    validate(res_GPU, res_CPU, N);
    //-------------------------------

    // 8. Liberación de memoria
    // --------- CPU ----------
    free(h_A);
    free(h_B);
    free(h_C);

    // --------- GPU ----------
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    //-------------------------

    // 9. SpeedUp
    std::cout << "SpeedUp: " << speedUp(timeCPU, timeGPU) << "\n" << std::endl;

    return 0;
}