# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 12:14:51 2020

@author: ameli
"""
from Classe import Algo_Var_Cat

#A modifier : forme des tableaux : essayer d'en faire des jolies comme une 
#image comme les plots pour une plus belle application

#importation du fichier
#On recup les données ensuite faut récupérer le nom de la variable cible, puis 
#vérif si elle est à la fin et sinon changer !!

#Donnée pour l'Arbre de décision    
#Variable X quanti ou quali discrete mais de type numériques:
import pandas
df = pandas.read_csv("opt_digits.txt",sep="\t",header=0)
#df.info()
df=df.iloc[:,0:65]
df['classe']=df['chiffre']
df=df.iloc[:,1:66]

#Donnée pour Analyse discriminante linéaire:
dfTrain = pandas.read_excel("seeds_dataset_python.xlsx",0)
#chargement de l'échantillon test
dfTest = pandas.read_excel("seeds_dataset_python.xlsx",1)
df2=pandas.concat([dfTrain,dfTest])

#Donnée pour la Regression logistique
#Variable cible à 2 facteurs (0 et 1 et de type (int)) : 
DTrain = pandas.read_excel("infidelites_python.xlsx", sheet_name = "train")
#chargement des données test
DTest = pandas.read_excel("infidelites_python.xlsx", sheet_name = "test")
df3=pandas.concat([DTrain,DTest])

#Données pour reg test 2
df4=pandas.read_excel("diabete_reg_logistique.xlsx")
for i in range(0,len(df4['diabete'])):
    if (df4['diabete'][i]=='positive') : 
        df4['diabete'][i]=0
    if (df4['diabete'][i]=='negative') : 
        df4['diabete'][i]=1
df4['diabete'] = df4['diabete'].astype('int')  

#test=Algo_Var_Cat(df)
#test.Arbre_decision()

#test2=Algo_Var_Cat(df2)
#test2.Analyse_Discriminante()

test3=Algo_Var_Cat(df4)
test3.Regression_log()


import Classe_Reg
from Classe_Reg import Algo_Var_Num


df = pandas.read_csv("AirQualityUCI.csv", sep="," , decimal =".")

df.info()

test=Algo_Var_Num(df)
test.Regression_line_multiple()

test.K_Proches_Voisins_Reg()

test.Reseau_Neurone()



