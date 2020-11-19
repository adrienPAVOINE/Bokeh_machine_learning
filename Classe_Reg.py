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
from sklearn.metrics import r2_score,make_scorer
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.model_selection import cross_val_score
from statistics import mean

from bokeh.io import curdoc, show
import io
from bokeh import plotting
from bokeh.layouts import row, column, gridplot, layout
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, FileInput
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral10
from bokeh.models import HoverTool, Div,Panel,Tabs
from bokeh.models.widgets import MultiSelect, Select, RangeSlider, Button, DataTable, DateFormatter,RadioGroup, TableColumn, Dropdown




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
        self.dfTrain, self.dfTest = train_test_split(self.df,test_size=size,random_state=9)
        #print(dfTrain, dfTest)
        
        #Séparer X et Y :
        self.yTrain=self.dfTrain.iloc[:,-1]
        self.XTrain=self.dfTrain.iloc[:,0:(len(self.df.columns)-1)]
        self.yTest=self.dfTest.iloc[:,-1]
        self.XTest=self.dfTest.iloc[:,0:(len(self.df.columns)-1)]
        #print(self.df)
        
        #distribution des classes
        print(self.yTrain.value_counts(normalize=True))
        print(self.yTest.value_counts(normalize=True))
        print(self.yTrain,self.XTrain)
    
    
    #-------------------------------------------------------------------------
    #Création de de la régression linéaire multiple et de la prédiction
    #-------------------------------------------------------------------------
    def Regression_line_multiple(self,nb_cv=5):

        #instanciation - objet arbre de décision
        #max_depth = nombre de feuille de l'arbre possible de demander à l'utilisateur
        lin_reg_mod = LinearRegression()
        lin_reg_mod.fit(self.XTrain,self.yTrain)
    
        # The coefficients
        print('Coefficients: \n', lin_reg_mod.coef_)
        
        #update section for bokeh-------------
        coeff_lin_reg=lin_reg_mod.coef_
        xt=self.XTrain
        temp=pandas.DataFrame({"var":xt.columns,"coef":coeff_lin_reg})
        columns=[TableColumn(field=Ci, title=Ci) for Ci in temp.columns] 
        self.title_for_coeff=Div(text="Coefficients de la régression pour les variables sélectionnées : ")
        self.coef=DataTable(source=ColumnDataSource(temp),columns=columns)
        print(self.coef)
        #end section for bokeh-------------
        
        #-------------------------------------------------------------------------
        #Prédiction : 
        #-------------------------------------------------------------------------

        #prédiction en test

        yPred = lin_reg_mod.predict(self.XTest)
        
         # The mean squared error
        print("Mean squared error")
        print(mean_squared_error(self.yTest, yPred))
        self.mse=Div(text= "Mean squared error :"+str(mean_squared_error(self.yTest, yPred)))
        print("R2 score")
        print(r2_score(self.yTest, yPred))
        self.r2=Div(text=" R2 score : "+str(r2_score(self.yTest, yPred)))
        
        self.fig= figure(title="Y_pred en vert VS y_test en rouge")
        self.fig.circle(range(len(self.yTest)), yPred[np.argsort(self.yTest)], color="green", size=8)
        self.fig.circle(range(len(self.yTest)), np.sort(self.yTest), color = "red", line_width=2)
        
        #plt.scatter(range(len(self.yTest)), yPred[np.argsort(self.yTest)], color = "green") #Prédictions
        #plt.plot(range(len(self.yTest)), np.sort(self.yTest), color = "red") #Données réelles
        #plt.title("Y_pred en vert, y_test en rouge")
        #plt.show()
        #validation croisée
        val_cro = cross_val_score(lin_reg_mod, self.X, self.y, cv=nb_cv,scoring=make_scorer(mean_squared_error))
        lst_cv=[]
        for i in range(1,nb_cv+1):
            lst_cv.append((str("essai : ")+str(i)))
            
        temp=pandas.DataFrame({"num de validation croisé":lst_cv,"res":val_cro})
        columns=[TableColumn(field=Ci, title=Ci) for Ci in temp.columns] 
        self.val_cro=DataTable(source=ColumnDataSource(temp),columns=columns)

        #self.val_cro=Div(text=" Cross Validation : "+str(val_cro))
        self.title_for_vc=Div(text="Résultats de la validation croisée : ")

        self.mean_val_cro=Div(text="MEAN Cross Validation :"+str(mean(val_cro)))
        print(val_cro)
        print(mean(val_cro))
        return self

        
    #-------------------------------------------------------------------------
    #Création de l'analyse discrinimante linéaire
    #-------------------------------------------------------------------------  
    def K_Proches_Voisins_Reg(self,kv = 5, nb_cv=5):
    

        knnRegressor = KNeighborsRegressor(kv)
        knnRegressor.fit(self.XTrain,self.yTrain)
        #-------------------------------------------------------------------------
        #Prédiction : 
        #-------------------------------------------------------------------------




        #prédiction en test

        yPred = knnRegressor.predict(self.XTest)
      
        
         # The mean squared error
        print("Mean squared error")
        print(mean_squared_error(self.yTest, yPred))
        self.mse=Div(text= "Mean squared error :"+str(mean_squared_error(self.yTest, yPred)))
        print("R2 score")
        print(r2_score(self.yTest, yPred))
        self.r2=Div(text=" R2 score : "+str(r2_score(self.yTest, yPred)))
        
        self.fig= figure(title="Y_pred en vert VS y_test en rouge")
        self.fig.circle(range(len(self.yTest)), yPred[np.argsort(self.yTest)], color="green", size=8)
        self.fig.line(range(len(self.yTest)), np.sort(self.yTest), color = "red", line_width=2)
        
        #plt.scatter(range(len(self.yTest)), yPred[np.argsort(self.yTest)], color = "green") #Prédictions
        #plt.plot(range(len(self.yTest)), np.sort(self.yTest), color = "red") #Données réelles
        #plt.title("Y_pred en vert, y_test en rouge")
        #plt.show()
        #validation croisée
        val_cro = cross_val_score(knnRegressor, self.X, self.y, cv=nb_cv)
        lst_cv=[]
        for i in range(1,nb_cv+1):
            lst_cv.append((str("essai : ")+str(i)))
            
        temp=pandas.DataFrame({"num de validation croisé":lst_cv,"res":val_cro})
        columns=[TableColumn(field=Ci, title=Ci) for Ci in temp.columns] 
        self.val_cro=DataTable(source=ColumnDataSource(temp),columns=columns)

        #self.val_cro=Div(text=" Cross Validation : "+str(val_cro))
        self.title_for_vc=Div(text="Résultats de la validation croisée : ")

        self.mean_val_cro=Div(text="MEAN Cross Validation :"+str(mean(val_cro)))
        print(val_cro)
        print(mean(val_cro))
        return self


    #-------------------------------------------------------------------------
    #Création de la regressionn logistiques
    #-------------------------------------------------------------------------  
    def Reseau_Neurone(self,max_iter,nb_cv=5):
        regr = MLPRegressor(random_state=1, max_iter = max_iter).fit(self.XTrain, self.yTrain)
        yPred = regr.predict(self.XTest)
        
        #regr.score(self.XTest, self.yTest)
        #print(regr.score(self.XTest, self.yTest))
        

               # The mean squared error
        print("Mean squared error")
        print(mean_squared_error(self.yTest, yPred))
        self.mse=Div(text= "Mean squared error :"+str(mean_squared_error(self.yTest, yPred)))
        print("R2 score")
        print(r2_score(self.yTest, yPred))
        self.r2=Div(text=" R2 score : "+str(r2_score(self.yTest, yPred)))
        
        self.fig= figure(title="Y_pred en vert VS y_test en rouge")
        self.fig.circle(range(len(self.yTest)), yPred[np.argsort(self.yTest)], color="green", size=8)
        self.fig.line(range(len(self.yTest)), np.sort(self.yTest), color = "red", line_width=2)
        
        #plt.scatter(range(len(self.yTest)), yPred[np.argsort(self.yTest)], color = "green") #Prédictions
        #plt.plot(range(len(self.yTest)), np.sort(self.yTest), color = "red") #Données réelles
        #plt.title("Y_pred en vert, y_test en rouge")
        #plt.show()
        #validation croisée
        val_cro = cross_val_score(regr, self.X, self.y, cv=nb_cv)
        lst_cv=[]
        for i in range(1,nb_cv+1):
            lst_cv.append((str("essai : ")+str(i)))
            
        temp=pandas.DataFrame({"num de validation croisé":lst_cv,"res":val_cro})
        columns=[TableColumn(field=Ci, title=Ci) for Ci in temp.columns] 
        self.val_cro=DataTable(source=ColumnDataSource(temp),columns=columns)

        #self.val_cro=Div(text=" Cross Validation : "+str(val_cro))
        self.title_for_vc=Div(text="Résultats de la validation croisée : ")

        self.mean_val_cro=Div(text="MEAN Cross Validation :"+str(mean(val_cro)))
        print(val_cro)
        print(mean(val_cro))
        return self

    def Anova_Desequilibre(self):
        ystr = str(self.yTrain.name)
        var_qual = ''
        for col in self.XTrain.columns: 
            var_qual += str(col)
            var_qual += '+'
        var_qual = var_qual[:-1]
        var = str(ystr + "~" + var_qual)
        anova = ols(var, data = self.dfTrain)
        lm = anova.fit()
        table= sm.stats.anova_lm(lm)
        print(table)
        aov = sm.stats.anova_lm(lm, typ=2)
        res = lm.resid 
        fig = sm.qqplot(res, line='s')
        plt.show()
        y_test = self.dfTest[str(ystr)]
        x_test = self.dfTest
 
        
        x_test = x_test.drop([str(ystr)],axis=1)
        pred = lm.predict(x_test)
        print(pred)
        
        

        pred.iloc[np.argsort(y_test)]
         # The mean squared error
        print("Mean squared error")
        print(mean_squared_error(y_test, pred))
        print("R2 score")
        print(r2_score(y_test, pred))   


        plt.plot(range(len(y_test)), pred.iloc[np.argsort(y_test)], color = "green") #Prédictions
        plt.scatter(range(len(y_test)), np.sort(y_test), color = "red") #Données réelles
        plt.title("Y_pred en vert, y_test en rouge")
        plt.show()
        #validation croisée
        #val_cro = cross_val_score(aov, self.X, self.y, cv=5)
        #print(val_cro)
        #print(mean(val_cro))
        

         