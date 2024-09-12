#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

int main(){
    // Cargar la imagen en color
    cv::Mat image = cv::imread("Imagen/BananaCat.jpg", cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cerr << "No se pudo abrir o encontrar la imagen" << std::endl;
        return -1;
    }

    // Coordenadas del píxel que deseas acceder
    int fila    = 100;
    int columna = 100;


    // Usar ptr para acceder a un píxel en la imagen
    cv::Vec3b* pixelPtr = image.ptr<cv::Vec3b>(fila); // Obtener el puntero al inicio de la fila y
    cv::Vec3b pixelValue = pixelPtr[columna];            // Acceder al valor del píxel en la columna x

    // Mostrar los valores de los canales B, G, R
    std::cout << "El valor del píxel en (" << columna << ", " << fila << ") es: "
              << "B: " << static_cast<int>(pixelValue[0]) << ", "
              << "G: " << static_cast<int>(pixelValue[1]) << ", "
              << "R: " << static_cast<int>(pixelValue[2]) << std::endl;

    return 0;
}