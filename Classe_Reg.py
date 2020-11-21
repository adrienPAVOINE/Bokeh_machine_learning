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
from bokeh.models.widgets import Paragraph,MultiSelect, Select, RangeSlider, Button, DataTable, DateFormatter,RadioGroup, TableColumn, Dropdown
import warnings
warnings.filterwarnings("ignore")



class Algo_Var_Num():
 
    #-------------------------------------------------------------------------
    #Initialisation des données 
    #-------------------------------------------------------------------------
    def __init__(self,df,size=-1):
        #dernière colonne est celle des Y
        self.df=df
                #################
        self.df = pandas.get_dummies(self.df)
        #############################
        self.y=df.iloc[:,-1]
        self.X=df.iloc[:,0:(len(self.df.columns)-1)]
        self.X = pandas.get_dummies(data=self.X, drop_first=True)
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
        #print(self.yTrain.value_counts(normalize=True))
        #print(self.yTest.value_counts(normalize=True))
        #print(self.yTrain,self.XTrain)
    
    
    #-------------------------------------------------------------------------
    #Création de de la régression linéaire multiple et de la prédiction
    #-------------------------------------------------------------------------
    def Regression_line_multiple(self,nb_cv=5):

        self.msg=Paragraph(text="""Vous êtes dans la partie réservé à la prédiction d'une variable quantitative par régression linéaire multiple. Dans cet onglet vous trouverez tout d'abord une pré-visualisation des données sur lesquelles va s'appliquer l'algorithme. Puis vous pourrez observer la liste des coefficients correspondant à chaque variables explicative. Afin de savoir si votre modèle est bon ou non, vous retrouverez deux indicateurs qui sont le R2 score ainsi que le MSE. Enfin sous la visualisation de vos données 'prédites vs test', vous pourrez vous même faire de la cross validation selon deux critères le R2 ou le MSE, utilisez juste le slider pour confirmer que votre modèle est bon ou non !""",width=1200, height=100)

        #instanciation - objet arbre de décision
        #max_depth = nombre de feuille de l'arbre possible de demander à l'utilisateur
        lin_reg_mod = LinearRegression()
        lin_reg_mod.fit(self.XTrain,self.yTrain)
    
        # The coefficients
       # print('Coefficients: \n', lin_reg_mod.coef_)
        
        coeff_lin_reg=lin_reg_mod.coef_
        xt=self.XTrain
        temp=pandas.DataFrame({"var":xt.columns,"coef":coeff_lin_reg})
        columns=[TableColumn(field=Ci, title=Ci) for Ci in temp.columns] 
        self.title_for_coeff=Div(text="<h2>Coefficients de la régression pour les variables sélectionnées </h2>")
        self.coef=DataTable(source=ColumnDataSource(temp),columns=columns)
       # print(self.coef)

        
        #-------------------------------------------------------------------------
        #Prédiction : 
        #-------------------------------------------------------------------------

        #prédiction en test

        yPred = lin_reg_mod.predict(self.XTest)
        
         # The mean squared error
        #print("Mean squared error")
        #print(mean_squared_error(self.yTest, yPred))
        self.title_indicators=Div(text= "<h2>Indicateurs de qualité</h2>")
        self.mse=Div(text= "Mean squared error :"+str(mean_squared_error(self.yTest, yPred)))
        #print("R2 score")
        #print(r2_score(self.yTest, yPred))
        self.r2=Div(text=" R2 score : "+str(r2_score(self.yTest, yPred)))
        self.title_fig=Div(text= "<h2>Visualisation des résultats</h2>")
        self.fig= figure(title="Y_pred en vert VS y_test en rouge")
        self.fig.circle(range(len(self.yTest)), yPred[np.argsort(self.yTest)], color="green", size=8)
        self.fig.line(range(len(self.yTest)), np.sort(self.yTest), color = "red", line_width=2)
        

        #validation croisée
        #,scoring=make_scorer(mean_squared_error)
        val_cro = cross_val_score(lin_reg_mod, self.X, self.y, cv=nb_cv)
        lst_cv=[]
        for i in range(1,nb_cv+1):
            lst_cv.append((str("essai : ")+str(i)))
            
        temp=pandas.DataFrame({"num de validation croisé":lst_cv,"res":val_cro})
        columns=[TableColumn(field=Ci, title=Ci) for Ci in temp.columns] 
        self.val_cro=DataTable(source=ColumnDataSource(temp),columns=columns)

        #self.val_cro=Div(text=" Cross Validation : "+str(val_cro))
        self.title_for_vc=Div(text="<h2>Résultats de la validation croisée</h2>")

        self.mean_val_cro=Div(text="Moyenne des Cross Validation :"+str(mean(val_cro)))
        print(val_cro)
        print(mean(val_cro))

        return self

        
    #-------------------------------------------------------------------------
    #Création de l'analyse discrinimante linéaire
    #-------------------------------------------------------------------------  
    def K_Proches_Voisins_Reg(self,kv = 5, nb_cv=5):

        self.msg=Paragraph(text="""Vous êtes dans la partie réservé à la prédiction d'une variable quantitative par méthode des k plus proches voisins. Dans cet onglet vous trouverez tout d'abord une pré-visualisation des données sur lesquelles va s'appliquer l'algorithme. Puis vous pourrez définir le nombre plus proche voisins observer la liste des coefficients correspondant à chaque variables explicative. Afin de savoir si votre modèle est bon ou non, vous retrouverez deux indicateurs qui sont le R2 score ainsi que le MSE. Enfin sous la visualisation de vos données 'prédites vs test', vous pourrez vous même faire de la cross validation selon deux critères le R2 ou le MSE, utilisez juste le slider pour confirmer que votre modèle est bon ou non !""",width=1200, height=100)
        knnRegressor = KNeighborsRegressor(kv)
        knnRegressor.fit(self.XTrain,self.yTrain)
        #-------------------------------------------------------------------------
        #Prédiction : 
        #-------------------------------------------------------------------------

        #prédiction en test

        yPred = knnRegressor.predict(self.XTest)
        self.title_indicators=Div(text= "<h2>Indicateurs de qualité</h2>")
         # The mean squared error
        #print("Mean squared error")
        #print(mean_squared_error(self.yTest, yPred))
        self.mse=Div(text= "Mean squared error :"+str(mean_squared_error(self.yTest, yPred)))
        #print("R2 score")
        #print(r2_score(self.yTest, yPred))
        self.r2=Div(text=" R2 score : "+str(r2_score(self.yTest, yPred)))
        self.title_fig=Div(text= "<h2>Visualisation des résultats</h2>")          
        self.fig= figure(title="Y_pred en bleu VS y_test en rouge")
        self.fig.circle(range(len(self.yTest)), yPred[np.argsort(self.yTest)], color="blue", size=8)
        self.fig.line(range(len(self.yTest)), np.sort(self.yTest), color = "red", line_width=2)
        
        #validation croisée
        val_cro = cross_val_score(knnRegressor, self.X, self.y, cv=nb_cv)
        lst_cv=[]
        for i in range(1,nb_cv+1):
            lst_cv.append((str("essai : ")+str(i)))
            
        temp=pandas.DataFrame({"num de validation croisé":lst_cv,"res":val_cro})
        columns=[TableColumn(field=Ci, title=Ci) for Ci in temp.columns] 
        self.val_cro=DataTable(source=ColumnDataSource(temp),columns=columns)

        self.title_for_vc=Div(text="<h2>Résultats de la validation croisée</h2>")

        self.mean_val_cro=Div(text="MEAN Cross Validation :"+str(mean(val_cro)))
        #print(val_cro)
        #print(mean(val_cro))
        return self


    #-------------------------------------------------------------------------
    #Création de la regressionn logistiques
    #-------------------------------------------------------------------------  
    def Reseau_Neurone(self,nb_cv=5):
        max_iter=50
        self.msg=Paragraph(text="""Vous êtes dans la partie réservé à la prédiction d'une variable quantitative par méthode des réseaux de neurones. Dans cet onglet vous trouverez tout d'abord une pré-visualisation des données sur lesquelles va s'appliquer l'algorithme. Puis vous pourrez observer la liste des coefficients correspondant à chaque variables explicative. Afin de savoir si votre modèle est bon ou non, vous retrouverez deux indicateurs qui sont le R2 score ainsi que le MSE. Enfin sous la visualisation de vos données 'prédites vs test', vous pourrez vous même faire de la cross validation selon deux critères le R2 ou le MSE, utilisez juste le slider pour confirmer que votre modèle est bon ou non !""",width=1200, height=100)
        regr = MLPRegressor(random_state=1, max_iter = max_iter).fit(self.XTrain, self.yTrain)
        yPred = regr.predict(self.XTest)
        
        self.title_indicators=Div(text= "<h2>Indicateurs de qualité</h2>")

        # The mean squared error
        #print("Mean squared error")
        #print(mean_squared_error(self.yTest, yPred))
        self.mse=Div(text= "Mean squared error :"+str(mean_squared_error(self.yTest, yPred)))
        #print("R2 score")
        #print(r2_score(self.yTest, yPred))
        self.r2=Div(text=" R2 score : "+str(r2_score(self.yTest, yPred)))
        self.title_fig=Div(text= "<h2>Visualisation des résultats</h2>")          
       
        self.fig= figure(title="Y_pred en noir VS y_test en rouge")
        self.fig.circle(range(len(self.yTest)), yPred[np.argsort(self.yTest)], color="black", size=8)
        self.fig.line(range(len(self.yTest)), np.sort(self.yTest), color = "red", line_width=2)

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
        #print(val_cro)
        #print(mean(val_cro))
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
        

   