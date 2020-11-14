# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 11:50:23 2020

@author: adrien
"""
from Classe_Reg import Algo_Var_Num

#A modifier : forme des tableaux : essayer d'en faire des jolies comme une 
#image comme les plots pour une plus belle application

#importation du fichier
#On recup les données ensuite faut récupérer le nom de la variable cible, puis 
#vérif si elle est à la fin et sinon changer !!

#Donnée pour l'Arbre de décision    
#Variable X quanti ou quali discrete mais de type numériques:
import pandas as pd
df = pd.read_csv("C:/Users/adrien/Downloads/AirQualityUCI.csv", sep=";" , decimal =",")



dfqual = pd.read_csv("C:/Users/adrien/Downloads/cars2.csv", sep=";")
dfqual.head(5)


test=Algo_Var_Num(dfqual)

test.Anova_Desequilibre()

test1=Algo_Var_Num(df)
test1.Regression_line_multiple()

#test.K_Proches_Voisins_Reg()

#test.Reseau_Neurone()

