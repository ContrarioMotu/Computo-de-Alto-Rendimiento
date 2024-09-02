#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime.h>

/*
 * Implementación de un programa en CUDA que realiza el cifrado d3e César, así como su descifrado,
 * de forma paralela, comparando los resultados obtenidos en la CPU y la GPU, y obteniendo el SpeedUp.
 *
 * @author: Ayala Morales Mauricio
 */

#define NL 1024

    /*
     * Función que realiza el cifrado de César a un mensaje utilizando la CPU.
     */
    void
    cesarOnCPU(char *msg, int key, int size){

    for (int i = 0; i < size; i++){
        if(msg[i] >= 'A' && msg[i] <= 'Z'){
            msg[i] = ((msg[i] - 'A') + key) % 26 + 'A';
        }

        if(msg[i] >= 'a' && msg[i] <= 'z'){
            msg[i] = ((msg[i] - 'a') + key) % 26 + 'a';
        }
    }
}

/*
 * Función que realiza el descifrado de César a un mensaje utilizando la CPU.
 */
void desCesarOnCPU(char *msg, int key, int size)
{

    for (int i = 0; i < size; i++)
    {
        if (msg[i] >= 'A' && msg[i] <= 'Z')
        {
            msg[i] = ((msg[i] - 'A') - key) % 26 + 'A';
        }

        if (msg[i] >= 'a' && msg[i] <= 'z')
        {
            msg[i] = ((msg[i] - 'a') - key) % 26 + 'a';
        }
    }
}

/*
 * Kernel que realiza el cifrado de César a un mensaje utilizando la GPU.
 */
__global__ void cesarOnGPU(char *msg, int key, int size){
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);

    if(idx < size){
        if (msg[idx] >= 'A' && msg[idx] <= 'Z'){
            msg[idx] = ((msg[idx] - 'A') + key) % 26 + 'A';
        }

        if (msg[idx] >= 'a' && msg[idx] <= 'z'){
            msg[idx] = ((msg[idx] - 'a') + key) % 26 + 'a';
        }
    }
}

/*
 * Kernel que realiza el descifrado de César a un mensaje utilizando la GPU.
 */
__global__ void desCesarOnGPU(char *msg, int key, int size)
{
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);

    if (idx < size)
    {
        if (msg[idx] >= 'A' && msg[idx] <= 'Z')
        {
            msg[idx] = ((msg[idx] - 'A') - key) % 26 + 'A';
        }

        if (msg[idx] >= 'a' && msg[idx] <= 'z')
        {
            msg[idx] = ((msg[idx] - 'a') - key) % 26 + 'a';
        }
    }
}

/*
 * Función que valida los resultados de los cifrados, obtenidos en la GPU y CPU.
 */
void validateEncrypt(char *h_msg, char *d_msg, int size){
    for(int i = 0; i < size; i++){
        if(h_msg[i]!= d_msg[i]){
            printf("ERROR: Cifrado distinto.\n");
            exit(0);
        }
    }

    printf("CIFRADO CORRECTO.\n");
}

/*
 * Función que valida los resultados de los descifrados, obtenidos en la GPU y CPU.
 */
void validateDecrypt(char *h_msg, char *d_msg, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (h_msg[i] != d_msg[i])
        {
            printf("ERROR: Descifrado distinto.\n");
            exit(0);
        }
    }

    printf("DESCIFRADO CORRECTO.\n");
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
 * Función que calcula el SpeedUp.
 */
double speedUp(double h_time, double d_time){
    return h_time / d_time;
}

int main(){

    int key = 4;
    int size = NL;

    // 2. Asignación de memoria dinámica.
    size_t nBytes = size * sizeof(char);

    char *h_msg, *d_msg, *h_res;
    // -------------- Host --------------
    h_msg = (char *)malloc(nBytes);
    h_res = (char *)malloc(nBytes);

    // ------------- Device -------------
    cudaMalloc((char **)&d_msg, nBytes);
    // ----------------------------------

    // 1. Inicialización de datos.
    strcpy(h_msg, "Hola mundo");
    // ---------------------------

    // 3. Transferencia de datos, del Host al Device (CPU a GPU).
    cudaMemcpy(d_msg, h_msg, nBytes, cudaMemcpyHostToDevice);
    // ----------------------------------------------------------

    printf("Mensaje: %s\n", h_msg);
    double tic = cpuTime();
    cesarOnCPU(h_msg, key, size);
    double toc = cpuTime();
    double tictocCPU = toc - tic;
    printf("Tiempo (CPU): %f segs.\n", tictocCPU);

    // 5.1 Configuración de la ejecución en la GPU.
    dim3 block(1);
    dim3 thread(1024);
    tic = cpuTime();
    cesarOnGPU<<<block, thread>>>(d_msg, key, size);
    cudaDeviceSynchronize();
    double toc = cpuTime();
    double tictocGPU = toc - tic;
    printf("Tiempo (GPU): %f segs.\n", tictocGPU);
    // -------------------------------------------

    // 6.1 Transferencia de datos, del Device al Host (GPU a CPU).
    cudaMemcpy(h_res, d_msg, nBytes, cudaMemcpyDeviceToHost);
    //-----------------------------------------------------------

    printf("Mensaje cifrado: %s\n", h_msg);

    // 7.1 Validación de los resultados.
    validateEncrypt(h_msg, h_res, size);
    //-------------------------------

    desCesarOnCPU(h_msg, key, size);

    // 5.2 Configuración de la ejecución en la GPU.
    desCesarOnGPU<<<block, thread>>>(d_msg, key, size);
    cudaDeviceSynchronize();
    // -------------------------------------------

    // 6.2 Transferencia de datos, del Device al Host (GPU a CPU).
    cudaMemcpy(h_res, d_msg, nBytes, cudaMemcpyDeviceToHost);
    //-----------------------------------------------------------

    // 7.2 Validación de los resultados.
    validateDecrypt(h_msg, h_res, size);
    //-------------------------------

    // 8. Liberación de memoria.
    // --------- CPU -----------
    free(h_msg);
    free(h_res);

    // --------- GPU -----------
    cudaFree(d_msg);
    //--------------------------

    // 9. SpeedUp.
    printf("SpeedUp: %lf\n", speedUp(tictocCPU, tictocGPU));

    return 0;
}