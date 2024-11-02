#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>

/*
 * Implementación de un programa en CUDA que realiza el método de bisección en múltiples
 * intervalos de forma paralela, comparando los resultados obtenidos en la CPU y la GPU, y
 * obteniendo el SpeedUp.
 * 
 * @author: Ayala Morales Mauricio
 */

#define a -100000.0f
#define b 100000.0f
#define ITERS 10000
#define EPSILON 1.0E-7

/* 
 * Función que se utilizará para busacar sus raices con la CPU.
 * En este caso se realiza la función 'x² - 2'.
 * @param x: Variable a la que se le aplicará la función.
 * @returns La función 'g' aplicada a 'x'.
 */
float g(float x){
    return (x * x) - 2.0f;
} 

/* 
 * Busca las raices de la función 'x² - 2', utilizando el método de bisección
 * con la CPU.
 *
 * @param start: Inicio del intervalo.
 * @param end: Fin del intervalo.
 * @param iters: Cantidad de iteraciones a realizar.
 * @param epsilon: Tolerancia para la convergencia.
 * @returns La raiz encontrada.
 */
float bisectionOnCPU(float start, float end, int iters, float epsilon){
    float c;
    float root = 0;
    for(int i = 0; i < iters; i++){
        c = (start + end) / 2;
        if(abs(g(c)) == 0 || (end - start) / 2 < epsilon){
            root = c;
            break;
        }
        if(c*c - 2 < 0){
            start = c;
        }else{
            end = c;
        }
        c = (start + end) / 2;
    }
    return root;
}

/* 
 * Función que se utilizará para busacar sus raices con la GPU.
 * En este caso se realiza la función 'x² - 2'.
 * @param x: Variable a la que se le aplicará la función.
 * @returns La función 'f' aplicada a 'x'.
 */
__device__ float f(float x){
    return (x * x) - 2.0f;
}

/* 
 * (Kernel) Busca las raices de la función definida en 'f(x)' de forma paralela, utilizando
 * el método de bisección, en la GPU
 * 
 * Cada hilo de la GPU realiza el método en su intervalo [A[idx], B[idx]] y almacena su resultado
 * en el arreglo 'R[idx]'. Solo se necesita el resultado del hilo que encuentre la raiz más rápido, 
 * por lo que los hilos no se sincronizan.
 *
 * @param A: Arreglo donde se almacena el inicio de cada intervalo.
 * @param C: Arreglo donde se almacenará el final de cada intervalo.
 * @param R: Arreglo donde se almacenará el resultado de cada hilo.
 * @param iters: Iteraciones a realizar en cada intervalo.
 * @param epsilon: Tolerancia para la convergencia.
 */
__global__ void bisectionOnGPU(float *A, float *B, float *R, int iters, float epsilon){
    unsigned int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx < iters){
        float izq = A[idx];
        float der = B[idx];
        float mid;
        for(unsigned int i = 0; i < iters; i++){
            mid = (der + izq) / 2;

            if(abs(f(mid)) == 0 || (der - izq) / 2 < epsilon){
                R[idx] = mid;
                return;
            }

            // Evitando divergencia de warps:
            // if (izq * f(mid) < 0 && der * f(mid) < 0) then izq = mid else der = mid
            izq = (mid * (f(mid) * f(der) < 0)) + (izq * !(f(mid) * f(der) < 0));
            der = (mid * !(f(mid) * f(der) < 0)) + (der * (f(mid) * f(der) < 0));
        }
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

    if(!(a < b)){
        std::cout << "ERROR: El inicio del intervalo debe ser menor al final del intervalo\n";
        std::cout << a << "<" << b << " !" << std::endl;
        return 1;
    }

    if(!(g(a) < 0 * g(b) > 0) || !(g(a) > 0 * g(b) < 0)){
        std::cout << "ERROR: g(a) y g(b) deben tener signos distintos.\n";
        std::cout << "g(a) = " << g(a) << ", g(b) = " << g(b) << " !" << std::endl;
        return 1;
    }


    // 2. Asignación de memoria dinámica
    // -------------- CPU --------------
    dim3 threads(1024);
    dim3 blocks((ITERS + threads.x - 1)/ threads.x);
    size_t nBytes = ITERS * sizeof(float);
    float *h_R, *h_A, *h_B;

    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_R = (float *)malloc(nBytes);

    //--------------- GPU --------------
    float *d_A, *d_B, *d_R;

    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_R, nBytes);
    //----------------------------------

    // 1. Inicialización de datos
    // --------------------------
    for(unsigned int i = 0; i < ITERS; i++){
        h_A[i] = 0.0f + (i * 0.01f);
        h_B[i] = 2.0f + (i * 0.01f);
        h_R[i] = nanf("?");
    }


    // --------------------------

    // 3. Transferencia de datos, del Host al Device (CPU a GPU)
    //----------------------------------------------------------
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_R, h_R, nBytes, cudaMemcpyHostToDevice);

    //----------------------------------------------------------

    double tic, toc, timeCPU, timeGPU;

    // 5. Configuración de la ejecución en la GPU
    std::cout << "Grid: " << blocks.x << "\nThreads: " << threads.x << "\n";
    tic = cpuTime();
    bisectionOnGPU<<<blocks, threads>>>(d_A, d_B, d_R, ITERS, EPSILON);
    cudaDeviceSynchronize();
    toc = cpuTime();
    timeGPU = toc - tic;
    std::cout << "GPU time:" << timeGPU << " segs.\n";
    //-------------------------------------------

    // 3. Transferencia de datos, del Device al Host (GPU a CPU)
    cudaMemcpy(h_R, d_R, nBytes, cudaMemcpyDeviceToHost);
    //----------------------------------------------------------

    // Suma del cálculo de cada bloque obtenido por el Device..
    float res_GPU ;
    for(int i = 0; i < nBytes; i++){
        if(h_R[i]!= nanf("?")){
            res_GPU = h_R[i];
            std::cout << "h_R[" << i << "]: " << h_R[i] << "\n";
            break;
        }
    }

    tic = cpuTime();
    float res_CPU = bisectionOnCPU(a, b, ITERS, EPSILON);
    toc = cpuTime();
    timeCPU = toc - tic;
    std::cout << "CPU time:" << timeCPU << " segs.\n";

    //7. Validación de los resultados
    validate(res_GPU, res_CPU);
    //-------------------------------

    // 8. Liberación de memoria
    // --------- CPU ----------
    free(h_A);
    free(h_B);
    free(h_R);
    // --------- GPU ----------
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_R);
    //-------------------------

    // 9. SpeedUp
    std::cout << "SpeedUp: " << speedUp(timeCPU, timeGPU) << std::endl;

    return 0;
}