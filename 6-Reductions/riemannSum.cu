#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <iostream>

/*
 * Implementación de un programa en CUDA que realiza la suma de Riemann de forma
 * paralela, comparando los resultados obtenidos en la CPU y la GPU, y obteniendo el SpeedUp.
 * 
 * @author: Ayala Morales Mauricio
 */

#define N 100000
#define a 5
#define b 100000

/* 
 * Realiza la suma de Riemann utilizando la CPU.
 *
 * @param start: Inicio del intervalo.
 * @param end: Fin del intervalo.
 * @param size: Cantidad de segmentos en los que se dividirá el intervalo.
 */
double riemannSumOnCPU(double start, double end, int size){
    double sum = 0;
    double dx = (double)(end - start) / (double)size;
    double x;
    for (unsigned int i = 0; i < size; i++)
    {
        x = start + i * dx;
        sum += x * x * dx;
    }
    return sum;
    
}

/* 
 * 4. (Kernel) Realiza la suma de Riemann de forma paralela utilizando la GPU.
 * utilizando la GPU.
 *
 * @param A: Arreglo donde se almacenará el área de cada rectángulo.
 * @param C: Arreglo donde se almacenará el resultado de la suma de Riemann obtenida por
 * cada bloque.
 * @param start: Inicio del intervalo.
 * @param end: Fin del intervalo.
 * @param size: Cantidad de segmentos en los que se dividirá el intervalo.
 */
__global__ void riemannSumOnGPU(double *A, double *C, double start, double end, int size){
    unsigned int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    unsigned int tid = threadIdx.x;
    if (idx < size){
        double dx = (end - start) / (double)size;
        double x = start + (double)(idx * dx);
        A[idx] = x * x * dx;

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

/*
 * Función que valida los resultados obtenidos en la GPU y CPU.
 *
 * @param res_GPU: Resultado obtenido por la GPU.
 * @param res_CPU: Resultado obtenido por la CPU.
 * @param size: Tamaño de los arreglos con los resultados.
 */
void validate(double res_GPU, double res_CPU){
    double epsilon = 1.0e-6;
    if (abs(res_GPU - res_CPU) > epsilon){
        std::cout << "ERROR: producto distinto\n";
        std::cout << abs(res_GPU - res_CPU) << "\n";
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
    int warps = (N <= 32)? 1 : ceil((double)N/32.0f);
    std::cout << "Warps: " << warps << "\n";
    int threads = (N <= 1024)? warps*32 : 1024;
    int blocks = (N + threads - 1)/ threads;
    size_t nBytes = N * sizeof(double);
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

    double tic, toc, timeCPU, timeGPU;

    // 5. Configuración de la ejecución en la GPU
    std::cout << "Grid: " << blocks << "\nThreads: " << threads << "\n";
    dim3 block(blocks);
    dim3 thread(threads);
    tic = cpuTime();
    riemannSumOnGPU<<<block, thread>>>(d_A, d_C, a, b, N);
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
    std::cout << "\n";

    tic = cpuTime();
    double res_CPU = riemannSumOnCPU(a, b, N);
    toc = cpuTime();
    timeCPU = toc - tic;
    std::cout << "CPU time:" << timeCPU << " segs.\n";

    //7. Validación de los resultados
    validate(res_GPU, res_CPU);
    //-------------------------------

    // 8. Liberación de memoria
    // --------- CPU ----------
    free(h_C);

    // --------- GPU ----------
    cudaFree(d_A);
    cudaFree(d_C);
    //-------------------------

    // 9. SpeedUp
    std::cout << "SpeedUp: " << speedUp(timeCPU, timeGPU) << "\n" << std::endl;

    return 0;
}