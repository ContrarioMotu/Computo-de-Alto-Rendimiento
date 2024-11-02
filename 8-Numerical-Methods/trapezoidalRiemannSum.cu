#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>

/*
 * Implementación de un programa en CUDA que realiza la suma de Riemann de forma
 * paralela, comparando los resultados obtenidos en la CPU y la GPU, y obteniendo el SpeedUp.
 * 
 * @author: Ayala Morales Mauricio
 */

#define N 1000000
#define a 0.0f
#define b 2 * M_PI

/* 
 * Realiza la suma de Riemann utilizando la CPU.
 *
 * @param start: Inicio del intervalo.
 * @param end: Fin del intervalo.
 * @param size: Cantidad de segmentos en los que se dividirá el intervalo.
 */
float riemannSumOnCPU(float start, float end, int size){
    float sum = 0;
    float dx = (end - start) / (float)size;
    float xa, xb;
    for (unsigned int i = 0; i < size; i++)
    {
        xa = start + (i * dx);
        xb = xa + dx;
        sum += (abs(sin(xa)) + abs(sin(xb))) * 0.5 * dx;
    }
    return sum;
    
}

/* 
 * Función que se utilizará para realizar la suma de Riemann en la GPU.
 * En este caso se realiza la función 'sin(x)'.
 * @param x: Variable a la que se le aplicará la función.
 * @returns La función 'f' aplicada a 'x'.
 */
__device__ float f(float x){
    return sin(x);
}

/* 
 * (Kernel) Calcula la suma de Riemann de la función 'f(x)' de forma paralela, utilizando
 * la GPU.
 * 
 * Cada hilo de la GPU realiza el cálculo del área de cada trapezoide, luego se realiza una
 * reducción (suma de cada trapezoide obtenido por cada hilo) por bloque, utilizando el 
 * método de Par Entrelazado. Por último, el resultado de cada bloque se almacena en un arreglo
 * dado.
 *
 * @param A: Arreglo donde se almacenará el área de cada trapezoide.
 * @param C: Arreglo donde se almacenará el resultado de la suma de Riemann obtenida por
 * cada bloque.
 * @param start: Inicio del intervalo.
 * @param end: Fin del intervalo.
 * @param size: Cantidad de segmentos en los que se dividirá el intervalo.
 */
__global__ void riemannSumOnGPU(float *A, float *C, float start, float end, int size){
    unsigned int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    unsigned int tid = threadIdx.x;
    // Referencia al inicio de cada bloque.
    float *idata = A + blockIdx.x * blockDim.x;

    if(idx < size){

    float dx = (end - start) / (float)size;
    float xa = start + (idx * dx);
    float xb = xa + dx;
    A[idx] = (abs(f(xa)) + abs(f(xb))) * 0.5f * dx;
    }
    __syncthreads();
    for(unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if (tid < stride){
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    
    if(tid == 0){
        C[blockIdx.x] = idata[0];
    }
    __syncthreads();
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
void validate(float res_GPU, float res_CPU){
    double epsilon = 1.0E-8;
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

    if (a > b){
        std::cout << "ERROR: El inicio del intervalo debe ser menor que el final del intervalo\n";
        std::cout << "ERROR: (a > b)\n" << std::endl;
        return 1;
    }

    // 2. Asignación de memoria dinámica
    // -------------- CPU --------------
    dim3 threads(1024);
    dim3 blocks((N + threads.x - 1)/ threads.x);
    size_t nBytes = N * sizeof(float);
    float *h_C, *d_A, *d_C;

    h_C = (float *)malloc(blocks.x * sizeof(float));

    //--------------- GPU --------------
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_C, blocks.x * sizeof(float));
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
    std::cout << "Grid: " << blocks.x << "\nThreads: " << threads.x << "\n";
    tic = cpuTime();
    riemannSumOnGPU<<<blocks, threads>>>(d_A, d_C, a, (float)b, N);
    cudaDeviceSynchronize();
    toc = cpuTime();
    timeGPU = toc - tic;
    std::cout << "GPU time:" << timeGPU << " segs.\n";
    //-------------------------------------------

    // 3. Transferencia de datos, del Device al Host (GPU a CPU)
    cudaMemcpy(h_C, d_C, blocks.x * sizeof(float), cudaMemcpyDeviceToHost);
    //----------------------------------------------------------

    // Suma del cálculo de cada bloque obtenido por el Device..
    float res_GPU = 0;
    for(int i = 0; i < blocks.x; i++){
        res_GPU += h_C[i];
    }
    std::cout << "\n";

    tic = cpuTime();
    float res_CPU = riemannSumOnCPU(a, b, N);
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
    std::cout << "SpeedUp: " << speedUp(timeCPU, timeGPU) << std::endl;

    return 0;
}