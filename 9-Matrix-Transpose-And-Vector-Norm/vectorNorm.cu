#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>

/*
 * Implementación de un programa en CUDA que calcula la norma de un vector, de forma
 * paralela, comparando los resultados obtenidos en la CPU y la GPU, y obteniendo el SpeedUp.
 * 
 * @author: Ayala Morales Mauricio
 */

#define N 100000000

/* 
 * Función que inicializa un arreglo con números aleatorios entre 1 y 100.
 *
 * @param A: Arreglo a inicializar.
 * @param size: Tamaño del arreglo.
 */
void initialize(double *A, int size){
    srand(0);

    for (int i = 0; i < size; i++)
    {
        A[i] = rand() % 100 + 1;
    }
}

/* 
 * Calcula la norma de un vector, utilizando la CPU.
 *
 * @param A: Arreglo con los elementos del vector.
 * @returns La norma del vector.
 */
double normOnCPU(double *A){
    double norm = 0;
    for(int i = 0; i < N; i++){
        norm += A[i] * A[i];
    }
    return sqrt(norm);
    
}

/* 
 * (Kernel) Calcula la norma de un vector, de forma paralela, utilizando
 * la GPU.
 * 
 * Cada hilo de la GPU 'idx' calcula el cuadrado del elemento 'A[idx]' del vector, luego se
 * realiza una reducción (suma de cada elemento obtenido por cada hilo) por bloque, utilizando
 * el método de 'Par Entrelazado'. Por último, el resultado de cada bloque se almacena en un arreglo
 * dado. En CPU se realiza la suma de tos los resultados obtenidos en cada bloque y se calcula
 * la raíz cuadrada.
 *
 * @param A: Arreglo con los elemenmtos del vetor.
 * @param C: Arreglo donde se almacenará el cuadrado de cada elmento del vector, obtenido por
 * cada bloque.
 * @param size: Cantidad de elementos del vector.
 */
__global__ void riemannSumOnGPU(double *A, double *C, int size){
    unsigned int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    unsigned int tid = threadIdx.x;
    // Referencia al inicio de cada bloque.
    double *idata = A + blockIdx.x * blockDim.x;

    if(idx >= size){
        return;
    }

    A[idx] = A[idx] * A[idx];
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
void validate(double res_GPU, double res_CPU){
    double epsilon = 1.0E-8;
    if (abs(res_GPU - res_CPU) > epsilon){
        std::cout << "ERROR: RESULTADO DISTINTO\n";
        std::cout << abs(res_GPU - res_CPU) << "\n";
        std::cout << res_GPU << " != " << res_CPU << "\n";
        return;
    }
    
    printf("RESULTADO CORRECTO\n");
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
    dim3 threads(1024);
    dim3 blocks((N + threads.x - 1)/ threads.x);
    size_t nBytes = N * sizeof(double);
    double *h_A, *h_C, *d_A, *d_C;

    h_A = (double *)malloc(nBytes);
    h_C = (double *)malloc(blocks.x * sizeof(double));

    //--------------- GPU --------------
    cudaMalloc((double **)&d_A, nBytes);
    cudaMalloc((double **)&d_C, blocks.x * sizeof(double));
    //----------------------------------

    // 1. Inicialización de datos
    // --------------------------
    initialize(h_A, N);
    // --------------------------

    // 3. Transferencia de datos, del Host al Device (CPU a GPU)
    //----------------------------------------------------------
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    //----------------------------------------------------------

    double tic, toc, timeCPU, timeGPU;

    // 5. Configuración de la ejecución en la GPU
    std::cout << "Grid: " << blocks.x << "\nThreads: " << threads.x << "\n";
    tic = cpuTime();
    riemannSumOnGPU<<<blocks, threads>>>(d_A, d_C, N);
    cudaDeviceSynchronize();
    toc = cpuTime();
    timeGPU = toc - tic;
    std::cout << "GPU time:" << timeGPU << " segs.\n";
    //-------------------------------------------

    // 3. Transferencia de datos, del Device al Host (GPU a CPU)
    cudaMemcpy(h_C, d_C, blocks.x * sizeof(double), cudaMemcpyDeviceToHost);
    //----------------------------------------------------------

    // Suma del cálculo de cada bloque obtenido por el Device..
    double res_GPU = 0;
    for(int i = 0; i < blocks.x; i++){
        res_GPU += h_C[i];
    }
    res_GPU = sqrt(res_GPU);

    tic = cpuTime();
    double res_CPU = normOnCPU(h_A);
    toc = cpuTime();
    timeCPU = toc - tic;
    std::cout << "CPU time:" << timeCPU << " segs.\n";

    //7. Validación de los resultados
    validate(res_GPU, res_CPU);
    //-------------------------------

    // 8. Liberación de memoria
    // --------- CPU ----------
    free(h_A);
    free(h_C);

    // --------- GPU ----------
    cudaFree(d_A);
    cudaFree(d_C);
    //-------------------------

    // 9. SpeedUp
    std::cout << "SpeedUp: " << speedUp(timeCPU, timeGPU) << std::endl;

    return 0;
}