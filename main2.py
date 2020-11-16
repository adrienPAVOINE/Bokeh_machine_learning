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



dfq = pd.read_csv("C:/Users/adrien/Downloads/house.csv", sep=";")

dfq = dfq.iloc[:,[5,6,7,8,12,69]]
for col in dfq.columns:
    dfq[col] = dfq[col].astype('category')
dfq['price'] = dfq['price'].astype('float64')
dfq.head(20)


type(dfq.info())

test=Algo_Var_Num(dfq)

test.Anova_Desequilibre()

test1=Algo_Var_Num(df)
#test1.Regression_line_multiple()

#test1.K_Proches_Voisins_Reg()

#test1.Reseau_Neurone()

