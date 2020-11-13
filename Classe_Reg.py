# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 11:31:03 2020

@author: adrien
"""

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
import matplotlib.pyplot as plt
import statsmodels as sm
from statsmodels.tools import add_constant
from statsmodels.api import Logit
import scipy
import numpy as np
import pandas
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression

class Algo_Var_Num():
 
    #-------------------------------------------------------------------------
    #Initialisation des données 
    #-------------------------------------------------------------------------
    def __init__(self,df,size=-1):
        #dernière colonne est celle des Y
        self.df=df
        self.y=df.iloc[:,-1]
        self.X=df.iloc[:,0:(len(self.df.columns)-1)]
        if (size==-1) : 
            size=round(len(self.df.values[:,-1])*0.3)
        #subdiviser les données en échantillons d'apprentissage et de test
        #Choix à l'utilisateur, taille de l'échantillon test : sinon le choix par défaut 70% 30%
        dfTrain, dfTest = train_test_split(self.df,test_size=size,random_state=9)
        #print(dfTrain, dfTest)
        
        #Séparer X et Y :
        self.yTrain=dfTrain.iloc[:,-1]
        self.XTrain=dfTrain.iloc[:,0:(len(self.df.columns)-1)]
        self.yTest=dfTest.iloc[:,-1]
        self.XTest=dfTest.iloc[:,0:(len(self.df.columns)-1)]
        #print(self.df)
        
        #distribution des classes
        print(self.yTrain.value_counts(normalize=True))
        print(self.yTest.value_counts(normalize=True))
        print(self.yTrain,self.XTrain)
    
    
    #-------------------------------------------------------------------------
    #Création de de la régression linéaire multiple et de la prédiction
    #-------------------------------------------------------------------------
    def Regression_line_multiple(self):

        #instanciation - objet arbre de décision
        #max_depth = nombre de feuille de l'arbre possible de demander à l'utilisateur
        lin_reg_mod = LinearRegression()
        lin_reg_mod.fit(self.XTrain,self.yTrain)
    
        # The coefficients
        print('Coefficients: \n', lin_reg_mod.coef_)

        #-------------------------------------------------------------------------
        #Prédiction : 
        #-------------------------------------------------------------------------

        #prédiction en test

        yPred = lin_reg_mod.predict(self.XTest)
        
         # The mean squared error
        print("Mean squared error")
        print(mean_squared_error(self.yTest, yPred))
        print("R2 score")
        print(r2_score(self.yTest, yPred))
        
        plt.scatter(range(len(self.yTest)), yPred[np.argsort(self.yTest)], color = "green") #Prédictions
        plt.plot(range(len(self.yTest)), np.sort(self.yTest), color = "red") #Données réelles
        plt.title("Y_pred en vert, y_test en rouge")
        plt.show()
        #validation croisée
        from sklearn.model_selection import cross_val_score
        print(cross_val_score(lin_reg_mod, self.X, self.y, cv=5))
        

        
    #-------------------------------------------------------------------------
    #Création de l'analyse discrinimante linéaire
    #-------------------------------------------------------------------------  
    def K_Proches_Voisins_Reg(self):
    
        #instanciation - objet arbre de décision
        #max_depth = nombre de feuille de l'arbre possible de demander à l'utilisateur
        knnRegressor = KNeighborsRegressor()
        knnRegressor.fit(self.XTrain,self.yTrain)


 
        #-------------------------------------------------------------------------
        #Prédiction : 
        #-------------------------------------------------------------------------

        #prédiction en test

        yPred = knnRegressor.predict(self.XTest)

        
        plt.scatter(range(len(self.yTest)), yPred[np.argsort(self.yTest)], color = "orange") #Prédictions
        plt.plot(range(len(self.yTest)), np.sort(self.yTest), color = "red") #Données réelles
        plt.title("Y_pred en orange, y_test en rouge")
        plt.show()

    #-------------------------------------------------------------------------
    #Création de la regressionn logistiques
    #-------------------------------------------------------------------------  
    def Reseau_Neurone(self):
        regr = MLPRegressor(random_state=1, max_iter=500).fit(self.XTrain, self.yTrain)
        yPred = regr.predict(self.XTest)
        
        regr.score(self.XTest, self.yTest)
        
        
        plt.scatter(range(len(self.yTest)), yPred[np.argsort(self.yTest)], color = "blue") #Prédictions
        plt.plot(range(len(self.yTest)), np.sort(self.yTest), color = "red") #Données réelles
        plt.title("Y_pred en bleue, y_test en rouge")
        plt.show()
        
         
        

         