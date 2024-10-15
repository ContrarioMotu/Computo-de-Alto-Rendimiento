#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <iostream>

/*
 * Implementación de un programa en CUDA que realiza la suma de Riemann para hacer una aproximación
 * de pi de forma paralela, utilizando la GPU.
 * 
 * @author: Ayala Morales Mauricio
 */

#define N 1000000

/* 
 * (Kernel) Calcula la suma de Riemann de la función sqrt(1-x^2) de forma paralela, utilizando
 * la GPU.
 * 
 * Cada hilo de la GPU realiza el cálculo del área de cada rectángulo, luego se realiza una
 * reducción (suma de cada rectángulo obtenido por cada hilo) por bloque, utilizando el 
 * método de Par Entrelazado. Por último, el resultado de cada bloque se almacena en un arreglo
 * dado.
 *
 * @param A: Arreglo donde se almacenará el área de cada rectángulo.
 * @param C: Arreglo donde se almacenará el resultado de la suma de Riemann obtenida por
 * cada bloque.
 * @param start: Inicio del intervalo.
 * @param end: Fin del intervalo.
 * @param size: Cantidad de segmentos en los que se dividirá el intervalo.
 */
__global__ void piOnGPU(double *A, double *C, float start, float end, int size){
    unsigned int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    unsigned int tid = threadIdx.x;
    if (idx < size){
        double dx = ((double)(end - start)) / (double)size;
        double x = (double)(idx * dx);
        A[idx] = sqrt(1.0 - (x * x)) * dx;

        __syncthreads();
    }
    unsigned int index;
    unsigned int stride = blockDim.x / 2;
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

int main(){

    // 2. Asignación de memoria dinámica
    // -------------- CPU --------------
    //int warps = (N <= 32)? 1 : ceil((double)N/32.0f);
    //std::cout << "Warps: " << warps << "\n";
    int threads = 1024;//(N <= 1024)? warps*32 : 1024;
    int blocks = (N + threads - 1)/ threads;
    size_t nBytes = N * sizeof(double);
    float a = 0.0f, b = 1.0f;
    double *h_C, *d_A, *d_C;

    h_C = (double *)malloc(blocks * sizeof(double));

    //--------------- GPU --------------
    cudaMalloc((double **)&d_A, nBytes);
    cudaMalloc((double **)&d_C, blocks * sizeof(double));
    //----------------------------------

    // 1. Inicialización de datos
    // --------------------------
    // No es necesario inicializar los arreglos con datos aleatorios.
    // --------------------------

    // 3. Transferencia de datos, del Host al Device (CPU a GPU)
    //----------------------------------------------------------
    // No es necesario transferir algún dato al host, todos los
    // datos se calculan en él.
    //----------------------------------------------------------

    double tic, toc, timeGPU;

    // 5. Configuración de la ejecución en la GPU
    std::cout << "Grid: " << blocks << "\nThreads: " << threads << "\n";
    dim3 block(blocks);
    dim3 thread(threads);
    tic = cpuTime();
    piOnGPU<<<block, thread>>>(d_A, d_C, a, b, N);
    cudaDeviceSynchronize();
    toc = cpuTime();
    timeGPU = toc - tic;
    std::cout << "GPU time:" << timeGPU << " segs.\n";
    //-------------------------------------------

    // 3. Transferencia de datos, del Device al Host (GPU a CPU)
    cudaMemcpy(h_C, d_C, blocks * sizeof(double), cudaMemcpyDeviceToHost);
    //----------------------------------------------------------

    // Suma del cálculo de cada bloque obtenido por el Device..
    double res_GPU = 0;
    for(int i = 0; i < blocks; i++){
        res_GPU += h_C[i];

    }

    // El resultado es el área de un solo cuadrante del círculo,
    // por lo que se debe multiplicar por los 4 cuadrantes totales.
    res_GPU = res_GPU * 4.0;
    std::cout << "Result: " << res_GPU << "\n" << std::endl;

    // 8. Liberación de memoria
    // --------- CPU ----------
    free(h_C);

    // --------- GPU ----------
    cudaFree(d_A);
    cudaFree(d_C);
    //-------------------------

    return 0;
}