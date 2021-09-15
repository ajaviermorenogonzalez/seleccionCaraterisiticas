import Proyecto
import pandas
import numpy
from sklearn import tree
from sklearn import model_selection

datos_csv = pandas.read_csv("titanic.csv")

atributos = [ 'SibSp','Deck','Title','Is_Married','Sex']

lista_atributos = []

for columna in atributos:
    lista_atributos.append(datos_csv[columna])

atributos_escogidos = pandas.DataFrame(lista_atributos)
atributos_escogidos = atributos_escogidos.transpose()

objetivo = datos_csv.iloc[:, -1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(atributos_escogidos, objetivo)

SibSp = 5 #Entero, numero de esposas e hijas en el barco
Deck = 7 #Entero, cubierta en la que te encuentras
Title = 0#Entero, número de títulos de los que dispones
Is_Married = 1 # Binario, 0 = no casado , 1 = casado
Sex = 1 # Binario, 0 = mujer , 1 = hombre

prediccion = clf.predict([[SibSp,Deck,Title,Is_Married,Sex]])















if(prediccion == 0):
    print("Tengo una buena y una mala noticia:\n "
          "La mala es que te has muerto... \n"
          "pero la buena, es que va a quedar una peli de puta madre")
else:
    print("¡Felicidades, saltaste a tiempo y continuas con vida! \n",
          "(Siempre y cuando eso que te roza el pie sea una sardina...)")