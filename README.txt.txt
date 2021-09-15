----------------------------------------------------
|Selección de características en modelos predictivos|
----------------------------------------------------

Abrir NotebookFinal en cualquier lector de notebooks de python (Jupiter,Sage...)

Ejecutar la primera celda. Esta carga todos los métodos necesarios que serán utilizados por la segunda celda de código.

Ejecutar la segunda celda para obtener la tabla de resultados, esta se compone de una serie de propiedades a especificar por el usuario :

	- datos_csv = pandas.read_csv("fichero.csv")  <- Conjunto de datos en formato csv del cual seleccionar sus características

	- names = []                                  <- Lista de nombres de las variables a evaluar (Todas las variables por defecto)

	- umbral = 10                                 <- Umbral para SFFS y SFS (10 por defecto)

	- n_exp = 1                                   <- Número de repeticiones del experimento por validación cruzada. (10 por defecto)

	- cv = 3                                      <-  Número de folds a considerar en la validación cruzada (10 por defecto).

	- metrica = None                              <- Métrica de evaluación a usar ( ‘balanced_accuracy’ por defecto )

	- x = 10                                      <- Indicar los x mejores conjuntos de características a mostrar

A continuación debemos descomentar (borrar '#') la linea de código que llama al algoritmo que deseamos usar (SFS o SFFS) 

El programa nos mostrará una tabla con los x mejores conjuntos de características del conjunto de datos además de dos gráficas: 

	- La primera gráfica nos muestra la evolución del rendimiento medio de los conjuntos de características según el tamaño de este.
	- La segunda gráfica nos muestra la evolución del rendimiento de los conjuntos de características según el tamaño indicado 
