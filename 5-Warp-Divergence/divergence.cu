#include <iostream>
#include <cuda_runtime.h>

// Kernel con divergencia
__global__ void divergentKernel(int *arr, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        if (arr[idx] % 2 == 0) {
            arr[idx] *= 2;  // Si el número es par, se multiplica por 2
        } else {
            arr[idx] += 1;  // Si es impar, se incrementa en 1
        }
    }
}

// Kernel sin divergencia (ramificación balanceada)
__global__ void nonDivergentKernel(int *arr, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        int isEven = !(arr[idx] % 2);
        arr[idx] = isEven * (arr[idx] * 2) + (!isEven) * (arr[idx] + 1);
    }
}

__global__ void divergentKernel2(int *arr, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        int part = n / 3;
        arr[idx] *= (idx < part)? arr[idx] * 4 :
                    (idx >= part && idx < (part * 2))? arr[idx] * 3 : arr[idx] * 2;
    }
}

__global__ void divergentKernel3(int *arr, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        if (idx % 3 == 0) {
            arr[idx] *= idx + n;
        } else if (idx % 3 == 1) {
            arr[idx] *= -idx + n;
        } else {
            arr[idx] /= idx;
        }
    }
}

// Función para inicializar el arreglo
void initializeArray(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100;  // Asignar valores aleatorios
    }
}

int main() {
    int n = 10000000;
    int *h_arr = new int[n];
    int *d_arr;

    // Inicializar el arreglo en el host
    initializeArray(h_arr, n);

    // Reservar memoria en la GPU
    cudaMalloc((void**)&d_arr, n * sizeof(int));

    // Copiar datos del host a la GPU
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

    // Configuración de los bloques y hilos
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Variables para medir el tiempo
    cudaEvent_t start, stop;
    float elapsedTime;

    // Crear eventos para medir el tiempo
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 1. Medir el tiempo del kernel con divergencia
    cudaEventRecord(start, 0);
    divergentKernel<<<gridSize, blockSize>>>(d_arr, n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Tiempo con divergencia: " << elapsedTime << " ms" << std::endl;

    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);  // Reiniciar el arreglo

    cudaEventRecord(start, 0);
    divergentKernel2<<<gridSize, blockSize>>>(d_arr, n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Tiempo con divergencia (operador ternario): " << elapsedTime << " ms" << std::endl;

    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);  // Reiniciar el arreglo

    cudaEventRecord(start, 0);
    divergentKernel3<<<gridSize, blockSize>>>(d_arr, n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Tiempo con divergencia (3 condicionales): " << elapsedTime << " ms" << std::endl;

    // 2. Medir el tiempo del kernel sin divergencia
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);  // Reiniciar el arreglo

    cudaEventRecord(start, 0);
    nonDivergentKernel<<<gridSize, blockSize>>>(d_arr, n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Tiempo sin divergencia: " << elapsedTime << " ms" << std::endl;

    // Limpiar
    cudaFree(d_arr);
    delete[] h_arr;

    return 0;
}
