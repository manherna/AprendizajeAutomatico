# Apuntes Aprendizaje Automático.

## Regresion Lineal Simple
Para este tipo de aprendizaje tenemos una variable de entrada **x**, y otra de salida **y**. A partir de una serie de datos, normalmente aportados en un csv, el programa genera una hipótesis lineal que recibe un **x** y estima para ese un **y**.

Para obtener la recta de hipótesis más precisa, deberemos calcular la función de error cuadrático (o función de coste, J(theta)) y minimizarlo.


Hay veces que la función de coste es una función en 3 dimensiones.
Para poder representarla se hacen parejas de valores dentro de la función.
Se puede querer representarla mediante curvas de nivel para hallar el minimo de la función.

Para minimizar la función, se utiliza la bajada de Gradiente.

### Bajada de Gradiente
Es un método que consiste en: desde un coste malo para nuestra función de coste, buscar la pendiente mayor hacia el siguiente punto hasta que encontremos un mínimo.

Esta función no es perfecta, porque puede ser que la función tenga varias zonas con costes bajos y acabemos en uno de ellos, pero no en el mínimo. Aunque esto puede ser bueno ya que si a veces no es bueno llegar mucho al minimo ya que puede implicar sobreajustarse a los ejemplos de aprendizaje.

Para la practica 0, el algoritmo de descenso de gradiente se deberá repetir 2000 veces.




## Clasificacion
Esta clase de problemas son aquellos en las que tenemos que identificar una serie de datos dentro de un grupo.
El caso más simple es el de discernir entre 2 casos. *Por ejemplo, distinguir si un
correo es spam o no es spam*

La aproximación mediante regresión lineal no es válida para este caso.

La función utilizada para aproximar los valores a uno u otro de los casos es la **función Sigmoide**:

g(*z*) = 1/(1+*e*^(-z)) 

Esta función sigmoide nos devolverá un valor entre 0 y 1, que significará la probabilidad de que cierto caso x, sea 1.

Pero nosotros no queremos trabajar con probabilidades, así que para g(*z*) >= 0.5 la daremos como 1 y g(*z*)< 0.5 entonces la daremos como 0.

De esta manera obtenemos una recta que nos divide los casos en 0 o 1, de la forma x1 + x2 - N = 0.








