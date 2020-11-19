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


from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, FileInput
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral10
from bokeh.models import HoverTool, Div,Panel,Tabs
from bokeh.models.widgets import MultiSelect, Select, RangeSlider, Button, DataTable, DateFormatter,RadioGroup, TableColumn, Dropdown,StringFormatter,SumAggregator, DataCube,GroupingInfo

class Algo_Var_Cat():
 
    #-------------------------------------------------------------------------
    #Initialisation des données 
    #-------------------------------------------------------------------------
    def __init__(self,df,var_cible,size=-1):
        self.df=df
        self.var_cible=var_cible
        #dernière colonne est celle des Y
        if (self.var_cible!=self.df.iloc[:,-1].name) :
            self.df['classe']=df[self.var_cible]
            self.df=self.df.drop(self.var_cible,axis='columns')
            self.var_cible='classe'
        #initialisation de la taille de l'ech train :
        #print(self.df)
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
        
        #print(self.XTrain,self.yTrain)
        
        test = self.yTrain.value_counts(normalize=True)
        
        #distribution des classes
        self.distrib1=Div(text="<h4>Distribution des classes :</h4>")
        self.distrib2=Div(text="Classe d'entrainement : <br/>")
        temp=pandas.DataFrame({"var":test.index,"distribution":test.values})
        columns=[TableColumn(field=Ci, title=Ci) for Ci in temp.columns] 
        self.distrib3=DataTable(source=ColumnDataSource(temp),columns=columns)
        self.distrib4=Div(text="Classe de test : <br/>")
        temp=pandas.DataFrame({"var":test.index,"distribution":test.values})
        columns=[TableColumn(field=Ci, title=Ci) for Ci in temp.columns] 
        self.distrib5=DataTable(source=ColumnDataSource(temp),columns=columns)
    
    #-------------------------------------------------------------------------
    #Création de l'arbre de décision et de la prédiction
    #-------------------------------------------------------------------------
    def Arbre_decision(self,nb_feuille=2,nb_cross_val=10):
        #classe arbre de décision
        #instanciation - objet arbre de décision
        #max_depth = nombre de feuille de l'arbre possible de demander à l'utilisateur
        dtree = DecisionTreeClassifier(max_depth = nb_feuille)
        #appliquer l'algo sur les données d'apprentissage
        dtree.fit(self.XTrain,self.yTrain)
        
        #-------------------------------------------------------------------------
        #Afficher l'arbre : 
        #-------------------------------------------------------------------------
    
        #Graph avec toutes les informations
        
        #affichage sous forme de règles
        #plus facile à appréhender quand l'arbre est très grand
        tree_rules = export_text(dtree,feature_names = list(self.df.columns[:-1]),show_weights=True)
        self.regles=Div(text=str(tree_rules))
        
        #-------------------------------------------------------------------------
        #Prédiction : 
        #-------------------------------------------------------------------------
        
        #importance des variables
        imp = {"VarName":self.df.columns[1:65],"Importance":dtree.feature_importances_}
        self.ceof1=Div(text="<h4>Importance des variables :</h4>")
        self.coef=Div(text=str(pandas.DataFrame(imp).sort_values(by="Importance",ascending=False)))
        
        #prédiction en test
        yPred = dtree.predict(self.XTest)
        
        #distribution des classes prédictes
        #Intéressant d'afficher cette information
        self.distribpred1=Div(text="Distribution des classes prédictes : </h4>" +str(np.unique(yPred,return_counts=True)))
        #columns=[TableColumn(field=Ci, title=Ci) for Ci in temp.columns] 
        #self.distribpred2=DataTable(source=temp[1],columns=temp[0])
                
        
        #matrice de confusion
        #Afficher la matrice de confusion !
        mc = metrics.confusion_matrix(self.yTest,yPred)
        self.matrice_confusion=Div(text="</br><h4>Matrice de confusion :</h4></br>")
            
        d = dict()
        d["affichage"]= []
        
        d["var"]=self.df[str(self.var_cible)].unique()
        for i in range(len(d["var"])):
            d[d["var"][i]]=list(mc[i])
            d["affichage"].append("")
            
        source = ColumnDataSource(data=d)
        target = ColumnDataSource(data=dict(row_indices=[], labels=[]))
        formatter = StringFormatter(font_style='bold')
        columns=[TableColumn(field='var', title=str(self.var_cible), width=40, sortable=False, formatter=formatter)]
        columns[1:(len(self.df[str(self.var_cible)].unique()))]=[TableColumn(field=str(NomMod), title=str(NomMod), width=40, sortable=False) for NomMod in self.df[str(self.var_cible)].unique()]
        grouping = [GroupingInfo(getter='affichage'),]
        self.cube = DataCube(source=source, columns=columns, grouping=grouping, target=target)
        
        #taux de reconnaissance
        acc = metrics.accuracy_score(self.yTest,yPred)
        self.Tx_reconnaissance=Div(text="<h4>Taux de reconnaissance :</h4>" + str(round(acc,4)))
        
        #calcul du taux d'erreur
        self.Tx_erreur=Div(text="<h4>Taux d'erreur :</h4>" + str(round(1.0-metrics.accuracy_score(self.yTest,yPred),4)))
        
        #rappel par classe
        self.rapclass=Div(text="<h4>Rappel par classe :</h4>" +str(metrics.recall_score(self.yTest,yPred,average=None)))
        
        #precision par classe
        self.precclasse=Div(text="<h4>Précision par classe : </h4>" + str(metrics.precision_score(self.yTest,yPred,average=None)))
        
        #rapport général
        self.rapport=Div(text="<h4>Rapport sur la qualité de prédiction :</h4> "+str(metrics.classification_report(self.yTest,yPred)))
        
        #-------------------------------------------------------------------------
        #Validation croisée : 
        #-------------------------------------------------------------------------
        #évaluation en validation croisée : 
        # paramètre par défaut : nb_cross_val=10
        succes = model_selection.cross_val_score(dtree,self.X,self.y,cv=nb_cross_val,scoring='accuracy')
        #détail des itérations
        self.int_succes=Div(text="<h4>Succès de la validation croisée :</h4>"+ str(succes))
        #moyenne des taux de succès = estimation du taux de succès en CV
        self.moy_succes=Div(text="<h4>Moyenne des succès :</h4>" + str(round(succes.mean(),4))) 

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
        self.coef=Div(text="<h4>Table des coefficients et des intercepts : </h4>"+str(final))
                
        #-------------------------------------------------------------------------
        #Prédiction : 
        #-------------------------------------------------------------------------
        ypred = lda.predict(self.XTest)
        #matrice de confusion
        mc=pandas.crosstab(self.yTest,ypred)
        mcSmNumpy = mc.values
        self.matrice_confusion=Div(text="</br><h4>Matrice de confusion :</h4>")
            
        d = dict()
        d["affichage"]= []
        
        d["var"]=self.df[str(self.var_cible)].unique()
        for i in range(len(d["var"])):
            d[d["var"][i]]=mcSmNumpy[0]
            d["affichage"].append("")
            
        source = ColumnDataSource(data=d)
        target = ColumnDataSource(data=dict(row_indices=[], labels=[]))
        formatter = StringFormatter(font_style='bold')
        columns=[TableColumn(field='var', title=str(self.var_cible), width=40, sortable=False, formatter=formatter)]
        columns[1:(len(self.df[str(self.var_cible)].unique()))]=[TableColumn(field=str(NomMod), title=str(NomMod), width=40, sortable=False) for NomMod in self.df[str(self.var_cible)].unique()]
        grouping = [GroupingInfo(getter='affichage'),]
        self.cube = DataCube(source=source, columns=columns, grouping=grouping, target=target)
        
        #taux de reconnaissance
        accSm = np.sum(np.diagonal(mcSmNumpy))/np.sum(mcSmNumpy)
        self.Tx_reconnaissance=Div(text="<h4>Taux de reconnaissance :</h4> " + str(round(accSm),4))
                
        #calcul du taux d'erreur
        self.Tx_erreur=Div(text="<h4>Taux d'erreur :</h4><br/>" + str(1.0-metrics.accuracy_score(self.yTest,ypred)))
        
        #calcul des sensibilité (rappel) et précision par classe
        self.rapport=Div(text="<h4>Rapport sur la qualité de prédiction : </h4>"+str(metrics.classification_report(self.yTest,ypred)))
        
        #-------------------------------------------------------------------------
        #Validation croisée : 
        #-------------------------------------------------------------------------
        #évaluation en validation croisée : 
        # paramètre par défaut : nb_cross_val=10
        succes = model_selection.cross_val_score(lda,self.X,self.y,cv=nb_cross_val,scoring='accuracy')
        #détail des itérations
        self.int_succes=Div(text="<h4>Succès de la validation croisée :</h4>"+ str(succes))
        #moyenne des taux de succès = estimation du taux de succès en CV
        self.moy_succes=Div(text="<h4>Moyenne des succès :</h4>" + str(round(succes.mean(),4))) 


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
            self.log_vraisemblance=Div(text="<h4>La log-vraisemblance vaut : </h4>"+str(round(log_likelihood,4)))
            
        
        #-------------------------------------------------------------------------
        #Prédiction : 
        #-------------------------------------------------------------------------
        #ajout d'une constante :
        ZTest = add_constant(self.XTest)
        #transformation de l'échantillon test (centrer-réduire)
        ZTest_Bis = stds.transform(ZTest)
        
        if (multi==False):
            #calcul de la prédiction sur l'échantillon test
            predProbaSk = lrSkStd.predict_proba(ZTest_Bis)
            
            #convertir en prédiction brute
            predSk = np.where(predProbaSk[:,1] > 0.5, 1, 0)
            
            #matrice de confusion
            #if len(DTrain[str(self.var_cible)].unique())==2 :
            mcSm=pandas.crosstab(self.yTest,predSk)
            print(mcSm[0][0],mcSm[0][1])
            self.matrice_confusion=Div(text="</br><h4>Matrice de confusion :</h4>")
            source = ColumnDataSource(data=dict(
                affichage=["",""],
                var=['positif', 'negatif'],
                positif=[mcSm[0]],
                negatif=[mcSm[1]]
            ))
            target = ColumnDataSource(data=dict(row_indices=[], labels=[]))
            formatter = StringFormatter(font_style='bold')
            columns = [
                TableColumn(field='var', title=str(self.var_cible), width=40, sortable=False, formatter=formatter),
                TableColumn(field='positif', title='positif', width=40, sortable=False),
                TableColumn(field='negatif', title='negatif', width=40, sortable=False),
            ]
            grouping = [
                GroupingInfo(getter='affichage'),
            ]
            self.cube = DataCube(source=source, columns=columns, grouping=grouping, target=target)
            

            #transformer en matrice Numpy
            mcSmNumpy = mcSm.values
            #taux de reconnaissance
            accSm = np.sum(np.diagonal(mcSmNumpy))/np.sum(mcSmNumpy)
            self.Tx_reconnaissance=Div(text="<h4>Taux de reconnaissance : </h4>" + str(accSm))
            #taux d'erreur
            errSm = 1.0 - accSm
            self.Tx_erreur=Div(text="<h4>Taux d'erreur : </h4><br/>"+str(round(errSm,4)))
            #rapport sur la qualité de prédiction
            self.rapport=Div(text="<h4>Rapport sur la qualité de prédiction : </h4>" + str(metrics.classification_report(self.yTest,predSk)))
            
            # Construire la courbe ROC et calculer la valeur de l'AUC sur l'échantillon test
            #colonnes pour les courbes ROC
            fprSm, tprSm, _ = metrics.roc_curve(self.yTest,predProbaSk[:,1],pos_label=1)
            
            #graphique
             #xs:[val X courbe 1,val X courbe 2]/ ys:[val y courbe 1,val y courbe 2]
            self.fig2= figure(title="Courbe ROC")
            self.fig2.multi_line(xs=[fprSm,np.arange(0,1.1,0.1)],ys=[tprSm,np.arange(0,1.1,0.1)], color=['green','blue'])
            
            #valeur de l'AUC
            aucSm = metrics.roc_auc_score(self.yTest,predSk)
            self.aucSm2=Div(text="<h4 > AUC : </h4>" +str(round(aucSm,4)))
            
        else :
            #calcul de la prédiction sur l'échantillon test
            predSk = lrSkStd.predict(ZTest_Bis)
            
            #matrice de confusion
            mcSm=pandas.crosstab(self.yTest,predSk)
            mcSm=mcSm.values
            self.matrice_confusion=Div(text="</br><h4>Matrice de confusion :</h4>")
            
            d = dict()
            d["affichage"]= []
            
            d["var"]=self.df[str(self.var_cible)].unique()
            for i in range(len(d["var"])):
                d[d["var"][i]]=mcSm[0]
                d["affichage"].append("")
                
            source = ColumnDataSource(data=d)
            target = ColumnDataSource(data=dict(row_indices=[], labels=[]))
            formatter = StringFormatter(font_style='bold')
            columns=[TableColumn(field='var', title=str(self.var_cible), width=40, sortable=False, formatter=formatter)]
            columns[1:(len(self.df[str(self.var_cible)].unique()))]=[TableColumn(field=str(NomMod), title=str(NomMod), width=40, sortable=False) for NomMod in self.df[str(self.var_cible)].unique()]
            grouping = [GroupingInfo(getter='affichage'),]
            self.cube = DataCube(source=source, columns=columns, grouping=grouping, target=target)
            
            
            #taux de reconnaissance
            accSm = np.sum(np.diagonal(mcSm))/np.sum(mcSm)
            self.Tx_reconnaissance=Div(text="<h4>Taux de reconnaissance :</h4> " + str(round(accSm,4)))
            #taux d'erreur
            errSm = 1.0 - accSm
            self.Tx_erreur=Div(text="<h4>Taux d'erreur : </h4>"+str(round(errSm,4)))
            #rapport sur la qualité de prédiction
            self.rapport=Div(text="<h4>Rapport sur la qualité de prédiction : </h4>" + str(metrics.classification_report(self.yTest,predSk)))
            
        #-------------------------------------------------------------------------
        #Validation croisée : 
        #-------------------------------------------------------------------------
        #évaluation en validation croisée : 10 cross-validation
        # paramètre par défaut : nb_cross_val=10
        succes = model_selection.cross_val_score(lrSkStd,self.X,self.y,cv=nb_cross_val,scoring='accuracy')
        #détail des itérations
        self.int_succes=Div(text="<h4>Succès de la validation croisée :</h4>"+ str(succes))
        #moyenne des taux de succès = estimation du taux de succès en CV
        self.moy_succes=Div(text="<h4>Moyenne des succès :</h4><br/>" + str(round(succes.mean(),4))) 
