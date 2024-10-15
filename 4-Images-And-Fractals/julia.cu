#include <sys/time.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace std;

#define HEIGHT 2048
#define WIDTH 2048
#define MAX_ITER 100


/**
 * (Kernel) Calcula el conjunto de Julia, utilizando la GPU.
 * 
 * @param img: Vector de la imagen a procesar.
 * @param width: Ancho de la imagen.
 * @param height: Alto de la imagen.
 * @param zoom: Factor de zoom.
 * @param moveX: Movimiento en el eje de los reales.
 * @param moveY: Movimiento en el eje de los imaginarios.
 * @param cRe: Parte real de la constante C.
 * @param cIm: Parte imaginaria de la constante C.
 */
__global__ void juliaOnGPU(uchar *img, int width, int height, float zoom, float moveX, float moveY,
                        float cRe, float cIm){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if(idx < width && idy < height){
        float zx = (idx - (width / 2))/(0.5f * zoom * width) + moveX;
        float zy = (idy - (height / 2))/(0.5f * zoom * height) + moveY;

        int iter = 0;
        while(zx*zx + zy*zy < 4.0f && iter < MAX_ITER){
            float temp = (zx*zx - zy*zy);
            zy = (2.0f * zx * zy) + cIm;
            zx = temp + cRe;
            iter++;
        }

        int r = (iter == MAX_ITER)? 0 : (iter * 136 / MAX_ITER);
        int g = (iter == MAX_ITER)? 0 : (iter * 112 / MAX_ITER);
        int b = (iter == MAX_ITER)? 0 : (iter * 255 / MAX_ITER);

        img[(idx + (idy * width)) * 3 + 2] = r;
        img[(idx + (idy * width)) * 3 + 1] = g;
        img[(idx + (idy * width)) * 3 + 0] = b;
    }
}

/**
 * Calcula el conjunto de Julia, utilizando la CPU.
 * 
 * @param img: Vector de la imagen a procesar.
 * @param width: Ancho de la imagen.
 * @param height: Alto de la imagen.
 * @param zoom: Factor de zoom.
 * @param moveX: Movimiento en el eje de los reales.
 * @param moveY: Movimiento en el eje de los imaginarios.
 * @param cRe: Parte real de la constante C.
 * @param cIm: Parte imaginaria de la constante C.
 */
void juliaOnCPU(uchar *img, int width, int height, float zoom, float moveX, float moveY,
                        float cRe, float cIm){
    
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            float zx = (col - (width / 2))/(0.5f * zoom * width) + moveX;
            float zy = (row - (height / 2))/(0.5f * zoom * height) + moveY;

            int iter = 0;
            while(zx*zx + zy*zy < 4.0f && iter < MAX_ITER){
                float temp = (zx*zx - zy*zy);
                zy = (2.0f * zx * zy) + cIm;
                zx = temp + cRe;
                iter++;
            }

            int r = (iter == MAX_ITER)? 0 : (iter * 136 / MAX_ITER);
            int g = (iter == MAX_ITER)? 0 : (iter * 112 / MAX_ITER);
            int b = (iter == MAX_ITER)? 0 : (iter * 255 / MAX_ITER);

            img[(col + (row * width)) * 3 + 2] = r;
            img[(col + (row * width)) * 3 + 1] = g;
            img[(col + (row * width)) * 3 + 0] = b;
        }
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
void validate(uchar *GPUres, uchar *CPUres, int rows, int cols){
    for(int i = 0; i < rows * cols; i++){
        if(abs(GPUres[i] - CPUres[i]) > 4){
            cout << "[" << i << "] : " << (int)GPUres[i]
                        << " != " << (int)CPUres[i] << "\n"
                        << "ERROR: Resultado distinto\n";
            return;
        }
    }
    cout << "RESULTADOS CORRECTOS\n";
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
    cv::Mat img(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
    float moveX = 0.0f, moveY = 0.0f, zoom = 1.0f, cRe = -0.835f, cIm = -0.2321f;
    size_t imgBytes = WIDTH * HEIGHT * 3 * sizeof(uchar);
    uchar *h_imgCPU, *h_imgGPU, *d_img;
    // --------------------------

    /// 2. Asignación de memoria dinámica
    // ----------------------------------
    h_imgCPU = new uchar[WIDTH * HEIGHT * 3];
    h_imgGPU = new uchar[WIDTH * HEIGHT * 3];
    cudaMalloc((void **)&d_img, imgBytes);
    // ----------------------------------

    double tic, toc, timeCPU, timeGPU;

    // 5. Configuración de la ejecución en la GPU.
    // -------------------------------------------
    int bCols = (WIDTH <= 32)? WIDTH : 32, bRows = (HEIGHT <= 32)? HEIGHT : 32;
    dim3 block(bCols, bRows);
    int gCols = (WIDTH + block.x - 1)/ block.x, gRows = (HEIGHT + block.y - 1)/ block.y;
    dim3 grid(gCols, gRows);
    printf("Grid: %d x %d\n", gCols, gRows);
    printf("Block: %d x %d\n", bCols, bRows);

    
    tic = cpuTime();
    juliaOnGPU<<<grid, block>>>(d_img, WIDTH, HEIGHT, zoom, moveX, moveY, cRe, cIm);
    toc = cpuTime();
    timeGPU = toc - tic;
    cout << "GPU time: " << timeGPU << " segs." << endl;
    // -------------------------------------------

    // // 6. Transferencia de datos, del Device al Host (GPU a CPU)
    // ------------------------------------------------------------
    cudaMemcpy(h_imgGPU, d_img, imgBytes, cudaMemcpyDeviceToHost);
    // ------------------------------------------------------------

    // Conjunto de Julia en CPU.
    // -------------------------
    tic = cpuTime();
    juliaOnCPU(h_imgCPU, WIDTH, HEIGHT, zoom, moveX, moveY, cRe, cIm);
    toc = cpuTime();
    timeCPU = toc - tic;
    cout << "CPU time: " << timeCPU << " segs." << endl;
    // -------------------------

    // 7. Validación de resultados.
    // ** No logré averiguar por qué los resultados salen distintos
    // a pesar de permitir que se tenga una diferencia de 4 en el
    // valor del pixel :( , pero las imagenes resultantes son casi idénticas.
    // ------------------------------
    validate(h_imgGPU, h_imgCPU, HEIGHT, WIDTH);
    // ------------------------------

    memcpy(img.ptr(), h_imgGPU, imgBytes);
    imwrite("Images/imgJuliaGPU.png", img);

    memcpy(img.ptr(), h_imgCPU, imgBytes);
    imwrite("Images/imgJuliaCPU.png", img);

    // 8. Liberación de memoria.
    // -------------------------
    free(h_imgCPU);
    free(h_imgGPU);
    cudaFree(d_img);
    // -------------------------

    // 9. SpeedUp.
    // -----------
    cout << "SpeedUp : " << speedUp(timeCPU, timeGPU) << "\n" << endl;
    // -----------

}