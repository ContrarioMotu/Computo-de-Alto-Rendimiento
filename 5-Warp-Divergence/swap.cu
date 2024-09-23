#include <cuda_runtime.h>
#include <iostream>
#include <sys/time.h>

#define SIZE 100000000

/* 
 * Inicializa un arreglo con números aleatorios entre 1 y 100.
 *
 * @param A: Arreglo a inicializar.
 * @param size: Tamaño del arreglo.
 */
float* initialize(int size){
    float *A = (float *)malloc(size * sizeof(float));

    for (int i = 0; i < size; i++)
    {
        A[i] = rand() % 100 + 1;
    }

    return A;
}

__global__ void swapOnGPU(float *A, float *B, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        float temp = A[idx];
        A[idx] = B[idx];
        B[idx] = temp;
    }
}

void swapOnCPU(float *A, float *B, int size){
    for(int i = 0; i < size; i++){
        float temp = A[i];
        A[i] = B[i];
        B[i] = temp;
    }
}

/**
 * Valida que los resultados obtenidos en la GPU y en la CPU sean iguales.
 *
 * @param GPUres: Vector obtenido en la GPU.
 * @param CPUres: Vector  obtenido en la CPU.
 * @param rows: Número de filas de la imagen.
 * @param cols: Número de columnas de la imagen.
 */
void validate(float *GPUres, float *CPUres, int size){
    double epsilon = 1.0E-8;
    for(int i = 0; i < size; i++){
        if(abs(GPUres[i] - CPUres[i]) > epsilon){
            std::cout << "[" << i << "] : " << (int)GPUres[i]
                        << " != " << (int)CPUres[i] << "\n"
                        << "ERROR: Resultado distinto\n";
            return;
        }
    }
    std::cout << "RESULTADOS CORRECTOS\n";
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
 * Calcula el SpeedUp.
 *
 * @param h_time: Tiempo utilizado por el Host (CPU) para realizar la operación.
 * @param d_time: Tiempo utilizado por el Host (GPU) para realizar la operación.
 *
 * @returns El SpeedUp obtenido del algoritmo ejecutado en paralelo en comparación con su ejecución
 * en serie en una sola unidad de procesamiento.
 */
double speedUp(float h_time, float d_time)
{
    return h_time / d_time;
}

int main(){

    // 1. Inicialización de datos
    // --------------------------

    float *h_A, *h_B, *h_AGPU, *h_BGPU, *d_A, *d_B;
    // --------------------------

    /// 2. Asignación de memoria dinámica
    // ----------------------------------

    h_A = initialize(SIZE);
    h_B = initialize(SIZE);
    h_AGPU = (float *)malloc(SIZE * sizeof(float));
    h_BGPU = (float *)malloc(SIZE * sizeof(float));
    cudaMalloc((void**)&d_A, SIZE * sizeof(float));
    cudaMalloc((void**)&d_B, SIZE * sizeof(float));
    // ----------------------------------

    // 3. Transferencia de datos, del Host al Device (CPU a GPU)
    //----------------------------------------------------------

    cudaMemcpy(d_A, h_A, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    //----------------------------------------------------------

    double tic, toc, timeGPU, timeCPU;

    // 5. Configuración de la ejecución en la GPU.
    // -------------------------------------------

    int threads = (SIZE <= 1024)? SIZE : 1024;
    int blocks = (SIZE + threads - 1)/ threads;
    printf("Grid: %d\n", blocks);
    printf("Block: %d\n", threads);
    tic = cpuTime();
    swapOnGPU<<<blocks, threads>>>(d_A, d_B, SIZE);
    cudaDeviceSynchronize();
    toc = cpuTime();
    timeGPU = toc - tic;
    std::cout << "GPU time: " << timeGPU << " segs.\n";
    // -------------------------------------------

    // 6. Transferencia de datos, del Device al Host (GPU a CPU)
    //----------------------------------------------------------

    cudaMemcpy(h_AGPU, d_A, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_BGPU, d_B, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    //----------------------------------------------------------

    tic = cpuTime();
    swapOnCPU(h_A, h_B, SIZE);
    toc = cpuTime();
    timeCPU = toc - tic;
    std::cout << "CPU time: " << timeCPU << " segs.\n";

    // 7. Validación de resultados.
    // ------------------------------

    std::cout << "A: ";
    validate(h_AGPU, h_A, SIZE);
    std::cout << "B: ";
    validate(h_BGPU, h_B, SIZE);
    // ------------------------------

    // 8. Liberación de memoria.
    // ---------------------------

    free(h_A);
    free(h_B);
    free(h_AGPU);
    free(h_BGPU);
    cudaFree(d_A);
    cudaFree(d_B);
    // ---------------------------

    // 9. SpeedUp.
    // -----------

    std::cout << "SpeedUp: " << speedUp(timeGPU, timeCPU) << "\n" << std::endl;
    // -----------
    return  0;
}