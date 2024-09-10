#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

/*
 * Implementación de un programa en CUDA que realiza la resta de dos vectores de forma
 * paralela, comparando los resultados obtenidos en la CPU y la GPU, y obteniendo el SpeedUp.
 * 
 * @author: Ayala Morales Mauricio
 */

#define N 9000000

/* 
* Función que inicializa un arreglo con números aleatorios entre 1 y 100.
*/
void initialize(float *A, int size){
    srand(0);

    for (int i = 0; i < size; i++)
    {
        A[i] = rand() % 100 + 1;
    }
}

/* 
* Función que realiza la resta de dos arreglos de números aleatorios utilizando la CPU.
*/
void subOnCPU(float *A, float *B, float *C, int size){
    for (int i = 0; i < size; i++)
    {
        C[i] = A[i] - B[i];
    }
}

/* 
* 4. Kernel en CUDA que realiza la resta de dos arreglos de números aleatorios utilizando la GPU.
*/
__global__ void subOnGPU(float *A, float *B, float *C, int size){
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx < size){
        C[idx] = A[idx] - B[idx];
    }
    
}

/*
* Función que obtiene la hora en segundos.
*/
double cpuTime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-06);
}

/*
* Función que valida los resultados de la resta, obtenidos en la GPU y CPU.
*/
void validate(float *res_GPU, float *res_CPU, int size){
    double epsilon = 1.0E-8;
    for (int i = 0; i < size; i++){
        if (abs(res_GPU[i] - res_CPU[i]) > epsilon){
            printf("ERROR: resta distinta\n");
            return;
        }
    }
    
    printf("RESTA CORRECTA\n");
}

/*
* Función que calcula el SpeedUp.
*/
double speedUp(double h_time, double d_time){
    return h_time / d_time;
}

int main(){

    // 2. Asignación de memoria dinámica
    // -------------- CPU --------------

    size_t nBytes = N * sizeof(float);
    float *h_A, *h_B, *h_C, *h_res;

    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);
    h_res = (float *)malloc(nBytes);

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
    memset(h_res, 0, nBytes);
    // --------------------------

    // 3. Transferencia de datos, del Host al Device (CPU a GPU)
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, nBytes, cudaMemcpyHostToDevice);
    //----------------------------------------------------------

    // 5. Configuración de la ejecución en la GPU
    int b = (N / 1024) + 1;
    dim3 block(b);
    dim3 thread(1024);
    double tic = cpuTime();
    subOnGPU<<<block, thread>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    double toc = cpuTime();
    double tictocGPU = toc - tic;
    printf("GPU time: %lf segs.\n", tictocGPU);
    //-------------------------------------------

    // 3. Transferencia de datos, del Device al Host (GPU a CPU)
    cudaMemcpy(h_res, d_C, nBytes, cudaMemcpyDeviceToHost);
    //----------------------------------------------------------

    tic = cpuTime();
    subOnCPU(h_A, h_B, h_C, N);
    toc = cpuTime();
    double tictocCPU = toc - tic;
    printf("CPU time: %lf segs.\n", tictocCPU);

    //7. Validación de los resultados
    validate(h_res, h_C, N);
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
    printf("SpeedUp: %lf\n", speedUp(tictocCPU, tictocGPU));

    return 0;
}