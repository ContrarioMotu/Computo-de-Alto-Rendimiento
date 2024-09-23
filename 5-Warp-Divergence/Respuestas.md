* ¿Por qué el kernel con divergencia es más lento?

    Se debe a la instrucción condicional 'if', ya que cada hilo del warp la misma cantidad de instrucciones, entonces los hilos que no cumplan con la condición se pondrán en espera (idle) hasta que los hilos que sí la cumplieron terminen de ejecutar el código dentro de la condicional.

* ¿Qué impacto tiene la divergencia en el rendimiento de un programa CUDA?

    Cada condicional en el código provoca una ramificación de la ejecución, donde los hilos que no cumplen con alguna condición se ponen en espera activa, lo que sigue consumiendo recursos, esto hasta que los hilos que cumplen con la condición terminen de ejecutar el código.
    
    Debido a lo anterior, todos los hilos terminan ejecutando la misma cantidad de instrucciones, por lo que el tiempo total que le toma al kernel en terminar es la suma de todas las instrucciones en cada ramificación.

    Por lo tanto, la divergencia eleva la cantidad de instrucciones y el tiempo de ejecución del kernel.

* ¿Cómo puedes evitar la divergencia en otros casos comunes de programación paralela?

    Minimizando el uso de instrucciones condicionales o ciclos (`if-else`, `for`, `while`), esto se puede lograr utilizando operaciones aritméticas, que por ejemplo, modifiquen los valores de la operación a realizar dependiendo del valor de su identificador (código del kernel sin divergencia).