# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 12:18:22 2020

@author: ameli
"""
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.tree import export_text
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection
import statsmodels as sm
from statsmodels.tools import add_constant
import matplotlib.pyplot as plt
import numpy as np
import pandas

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


class Algo_Var_Cat():
 
    #-------------------------------------------------------------------------
    #Initialisation des données 
    #-------------------------------------------------------------------------
    def __init__(self,df,size=-1):
        #dernière colonne est celle des Y
        self.df=df
        if (size==-1) : 
            size=round(len(self.df.values[:,-1])*0.3)
        #subdiviser les données en échantillons d'apprentissage et de test
        #Choix à l'utilisateur, taille de l'échantillon test : sinon le choix par défaut 70% 30%
        dfTrain, dfTest = train_test_split(self.df,test_size=size,random_state=1,stratify=self.df.iloc[:,-1])
        #print(dfTrain, dfTest)
        
        #Séparer X et Y :
        self.y=df.iloc[:,-1]
        self.X=df.iloc[:,0:(len(self.df.columns)-1)]
        self.yTrain=dfTrain.iloc[:,-1]
        self.XTrain=dfTrain.iloc[:,0:(len(self.df.columns)-1)]
        self.yTest=dfTest.iloc[:,-1]
        self.XTest=dfTest.iloc[:,0:(len(self.df.columns)-1)]
        
        #print(self.df)
        
        #distribution des classes
        self.distrib=Div(text="Distribution des classes :"+str(self.yTrain.value_counts(normalize=True))+"</br>"+str(self.yTest.value_counts(normalize=True)))
            
    
    #-------------------------------------------------------------------------
    #Création de l'arbre de décision et de la prédiction
    #-------------------------------------------------------------------------
    def Arbre_decision(self,nb_feuille=2,nb_cross_val=10):
        #classe arbre de décision
        #instanciation - objet arbre de décision
        #max_depth = nombre de feuille de l'arbre possible de demander à l'utilisateur
        dtree = DecisionTreeClassifier(max_depth = nb_feuille)
        print(dtree)
        #appliquer l'algo sur les données d'apprentissage
        dtree.fit(self.XTrain,self.yTrain)
        
        #-------------------------------------------------------------------------
        #Afficher l'arbre : 
        #-------------------------------------------------------------------------
    
        #Graph avec toute les information
        #plot_tree(dtree,filled=True)
        
        #Uniquement le graph possible de changer la taille (figsize) :
        plt.figure(figsize=(30,30))
        plot_tree(dtree,feature_names = list(self.df.columns[:-1]),filled=True)
        plt.show()
        
        #affichage sous forme de règles
        #plus facile à appréhender quand l'arbre est très grand
        tree_rules = export_text(dtree,feature_names = list(self.df.columns[:-1]),show_weights=True)
        print(tree_rules)
        
        #-------------------------------------------------------------------------
        #Prédiction : 
        #-------------------------------------------------------------------------
        
        #importance des variables
        imp = {"VarName":self.df.columns[1:65],"Importance":dtree.feature_importances_}
        print(pandas.DataFrame(imp).sort_values(by="Importance",ascending=False))
        
        #prédiction en test
        yPred = dtree.predict(self.XTest)
        
        #distribution des classes prédictes
        #Intéressant d'afficher cette information
        print(np.unique(yPred,return_counts=True))
        
        
        #matrice de confusion
        #Afficher la matrice de confusion !
        mc = metrics.confusion_matrix(self.yTest,yPred)
        print("Matrice de confusion :", mc)
        
        #taux de reconnaissance
        acc = metrics.accuracy_score(self.yTest,yPred)
        print("Taux de reconnaissance : %.4f" % acc)
        
        #calcul du taux d'erreur
        print("Taux d'erreur :%.4f" % 1.0-metrics.accuracy_score(self.yTest,yPred))
        
        #rappel par classe
        print(metrics.recall_score(self.yTest,yPred,average=None))
        
        #precision par classe
        print(metrics.precision_score(self.yTest,yPred,average=None))
        
        #rapport général
        print("Rapport sur la qualité de prédiction : ")
        print(metrics.classification_report(self.yTest,yPred))
        
        #-------------------------------------------------------------------------
        #Validation croisée : 
        #-------------------------------------------------------------------------
        #évaluation en validation croisée : 
        # paramètre par défaut : nb_cross_val=10
        succes = model_selection.cross_val_score(dtree,self.X,self.y,cv=nb_cross_val,scoring='accuracy')
        #détail des itérations
        print("Succès de la validation croisée :", succes)
        #moyenne des taux de succès = estimation du taux de succès en CV
        print("Moyenne des succès : %.4f " % succes.mean()) 

    #-------------------------------------------------------------------------
    #Création de l'analyse discrinimante linéaire
    #-------------------------------------------------------------------------  
    def Analyse_Discriminante(self,nb_cross_val=10):
        
        #classe LDA
        #instanciation 
        lda = LinearDiscriminantAnalysis()
        #apprentissage
        lda.fit(self.XTrain,self.yTrain)
        
        #structure pour affichage des coefficients et des intercepts
        tmp= pandas.DataFrame(lda.coef_.transpose(),columns=lda.classes_,index=self.XTrain.columns)
        tmp2={tmp.columns[0] : [lda.intercept_[0]]}
        for i in range(1,len(tmp.columns)): 
            tmp2.update({tmp.columns[i] : [lda.intercept_[i]]})
            
        tmp2=pandas.DataFrame(tmp2)
        tmp2=tmp2.rename(index={0 : "Constante"})
        final=pandas.concat([tmp2,tmp])
        print("Table des coefficients et des intercepts : ",final)
                
        #-------------------------------------------------------------------------
        #Prédiction : 
        #-------------------------------------------------------------------------
        ypred = lda.predict(self.XTest)
        #matrice de confusion
        mc=pandas.crosstab(self.yTest,ypred)
        print("Matrice de confusion : ",mc)
        #transformer en matrice Numpy
        mcSmNumpy = mc.values
        #taux de reconnaissance
        accSm = np.sum(np.diagonal(mcSmNumpy))/np.sum(mcSmNumpy)
        print("Taux de reconnaissance : %.4f" % (accSm))
                
        #calcul du taux d'erreur
        print("Taux d'erreur :" , 1.0-metrics.accuracy_score(self.yTest,ypred))
        
        #calcul des sensibilité (rappel) et précision par classe
        print("Rapport sur la qualité de prédiction : ")
        print(metrics.classification_report(self.yTest,ypred))
        
        #-------------------------------------------------------------------------
        #Validation croisée : 
        #-------------------------------------------------------------------------
        #évaluation en validation croisée : 
        # paramètre par défaut : nb_cross_val=10
        succes = model_selection.cross_val_score(lda,self.X,self.y,cv=nb_cross_val,scoring='accuracy')
        #détail des itérations
        print("Succès de la validation croisée :", succes)
        #moyenne des taux de succès = estimation du taux de succès en CV
        print("Moyenne des succès : %.4f " % succes.mean()) 


    #-------------------------------------------------------------------------
    #Création de la regressionn logistiques binaire
    #-------------------------------------------------------------------------  
    def Regression_log(self,multi,alpha=0.1,nb_cross_val=10):
        
        #instanciation
        stds = preprocessing.StandardScaler()
        #transformation centrer et reduire
        ZTrain = sm.tools.add_constant(self.XTrain)
        ZTrainBis = stds.fit_transform(ZTrain)
        
        if (multi==True) :
            #instanciation
            lrSkStd = LogisticRegression(penalty='none', multi_class='multinomial')
        elif (multi==False):
            #instanciation
            lrSkStd = LogisticRegression(penalty='none')
        
        #lancement des calculs -- pas nécessaire de rajouter la constante
        lrSkStd.fit(ZTrainBis,self.yTrain)
        
        #correction des coefficients - dé-standardisation
        #par les écarts-type utilisés lors de la standardisation des variables
        coefUnstd = lrSkStd.coef_[0] / stds.scale_
        temp=pandas.DataFrame({"var":ZTrain.columns,"coef":coefUnstd})
        columns=[TableColumn(field=Ci, title=Ci) for Ci in temp.columns] 
        self.coef=DataTable(source=ColumnDataSource(temp),columns=columns)
        interceptUnStd = lrSkStd.intercept_ + np.sum(lrSkStd.coef_[0]*(-stds.mean_/stds.scale_))
        self.const=Div(text= "Intercepts :"+str(interceptUnStd))
        
        if (multi==False):
            #Le log-vraisemblance
            #probabilités d'affectation
            proba01 = lrSkStd.predict_proba(ZTrain)
            #récupération de la colonne n°1
            proba1 = proba01[:,1]
            #log-vraisemblance
            log_likelihood = np.sum(self.yTrain*np.log(proba1)+(1.0-self.yTrain)*np.log(1.0-proba1))
            self.log_vraisemblance=Div(text="La log-vraisemblance vaut :"+str(log_likelihood))
            
        
        #-------------------------------------------------------------------------
        #Prédiction : 
        #-------------------------------------------------------------------------
        #ajout d'une constante :
        ZTest = add_constant(self.XTest)
        #transformation de l'échantillon test (centrer-réduire)
        ZTest_Bis = stds.transform(ZTest)
        
        #calcul de la prédiction sur l'échantillon test
        predProbaSk = lrSkStd.predict_proba(ZTest_Bis)
        
        #convertir en prédiction brute
        predSk = np.where(predProbaSk[:,1] > 0.5, 1, 0)
        #matrice de confusion
        mcSm=pandas.crosstab(self.yTest,predSk)
        self.matrice_confusion=Div(text="</br>Matrice de confusion : </br>"+str(np.array(mcSm)))
        #transformer en matrice Numpy
        mcSmNumpy = mcSm.values
        #taux de reconnaissance
        accSm = np.sum(np.diagonal(mcSmNumpy))/np.sum(mcSmNumpy)
        self.Tx_reconnaissance=Div(text="Taux de reconnaissance : " + str(accSm))
        #taux d'erreur
        errSm = 1.0 - accSm
        self.Tx_erreur=Div(text="Taux d'erreur : "+str(errSm))
        #rapport sur la qualité de prédiction
        self.rapport=Div(text="Rapport sur la qualité de prédiction : " + str(metrics.classification_report(self.yTest,predSk)))
        
        # Construire la courbe ROC et calculer la valeur de l'AUC sur l'échantillon test
        if (multi==False):
            
            #colonnes pour les courbes ROC
            fprSm, tprSm, _ = metrics.roc_curve(self.yTest,predProbaSk[:,1],pos_label=1)
            
            #graphique
             #xs:[val X courbe 1,val X courbe 2]/ ys:[val y courbe 1,val y courbe 2]
            self.fig2= figure(title="Courbe ROC")
            self.fig2.multi_line(xs=[fprSm,np.arange(0,1.1,0.1)],ys=[tprSm,np.arange(0,1.1,0.1)], color=['green','blue'])
            
            #valeur de l'AUC
            aucSm = metrics.roc_auc_score(self.yTest,predSk)
            self.aucSm2=Div(text="<h4 > AUC : " +str(round(aucSm,4))+"</h4 >")
            
            
            
        #-------------------------------------------------------------------------
        #Validation croisée : 
        #-------------------------------------------------------------------------
        #évaluation en validation croisée : 10 cross-validation
        # paramètre par défaut : nb_cross_val=10
        succes = model_selection.cross_val_score(lrSkStd,self.X,self.y,cv=nb_cross_val,scoring='accuracy')
        #détail des itérations
        self.int_succes=Div(text="Succès de la validation croisée :"+ str(succes))
        #moyenne des taux de succès = estimation du taux de succès en CV
        self.moy_succes=Div(text="Moyenne des succès :" + str(round(succes.mean(),4))) 
