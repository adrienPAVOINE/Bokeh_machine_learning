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
import matplotlib.pyplot as plt
import numpy
import pandas

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
        print("Distribution des classes :")
        print(self.yTrain.value_counts(normalize=True))
        print(self.yTest.value_counts(normalize=True))
        print(self.yTrain,self.XTrain)
    
    
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
        print(numpy.unique(yPred,return_counts=True))
        
        
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
        accSm = numpy.sum(numpy.diagonal(mcSmNumpy))/numpy.sum(mcSmNumpy)
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
        ZTrain = stds.fit_transform(self.XTrain)
        
        if (multi==True) :
            #instanciation
            lrSkStd = LogisticRegression(penalty='none', multi_class='multinomial')
        elif (multi==False):
            #instanciation
            lrSkStd = LogisticRegression(penalty='none')
        
        #lancement des calculs -- pas nécessaire de rajouter la constante
        lrSkStd.fit(ZTrain,self.yTrain)
        
        #correction des coefficients - dé-standardisation
        #par les écarts-type utilisés lors de la standardisation des variables
        coefUnstd = lrSkStd.coef_[0] / stds.scale_
        #affichage des coefficients corrigés
        print(pandas.DataFrame({"var":self.XTrain.columns,"coef":coefUnstd}))
        #pour la constante, l'opération est plus complexe
        interceptUnStd = lrSkStd.intercept_ + numpy.sum(lrSkStd.coef_[0]*(-stds.mean_/stds.scale_))
        print("Constante(s) :",interceptUnStd)
        
        if (multi==False):
            #Le log-vraisemblance
            #probabilités d'affectation
            proba01 = lrSkStd.predict_proba(ZTrain)
            #récupération de la colonne n°1
            proba1 = proba01[:,1]
            #log-vraisemblance
            log_likelihood = numpy.sum(self.yTrain*numpy.log(proba1)+(1.0-self.yTrain)*numpy.log(1.0-proba1))
            print("La log-vraisemblance vaut :",log_likelihood)
            
        
        #-------------------------------------------------------------------------
        #Prédiction : 
        #-------------------------------------------------------------------------
        #transformation de l'échantillon test (centrer-réduire)
        ZTest = stds.transform(self.XTest)
        #appliquer la prédiction
        predSk = lrSkStd.predict(ZTest)
        #matrice de confusion
        mcSm=pandas.crosstab(self.yTest,predSk)
        print("Matrice de confusion :",mcSm)
        #transformer en matrice Numpy
        mcSmNumpy = mcSm.values
        #taux de reconnaissance
        accSm = numpy.sum(numpy.diagonal(mcSmNumpy))/numpy.sum(mcSmNumpy)
        print("Taux de reconnaissance : %.4f" % (accSm))
        #taux d'erreur
        errSm = 1.0 - accSm
        print("Taux d'erreur' : %.4f" % (errSm))
        #rapport sur la qualité de prédiction
        print("Rapport sur la qualité de prédiction : ")
        print(metrics.classification_report(self.yTest,predSk))
        
        # Construire la courbe ROC et calculer la valeur de l'AUC sur l'échantillon test
        if (multi==False):
            #colonnes pour les courbes ROC
            fprSm, tprSm, _ = metrics.roc_curve(self.yTest,predSk,pos_label=1)
            
            #graphique
            #construire la diagonale
            plt.plot(numpy.arange(0,1.1,0.1),numpy.arange(0,1.1,0.1),'b')
            #rajouter notre diagramme
            plt.plot(fprSm,tprSm,"g")
            #titre
            plt.title("Courbe ROC")
            #faire apparaître
            plt.show()
            
            #valeur de l'AUC
            aucSm = metrics.roc_auc_score(self.yTest,predSk)
            print("AUC : %.4f" % (aucSm))
        
        #-------------------------------------------------------------------------
        #Validation croisée : 
        #-------------------------------------------------------------------------
        #évaluation en validation croisée : 10 cross-validation
        # paramètre par défaut : nb_cross_val=10
        succes = model_selection.cross_val_score(lrSkStd,self.X,self.y,cv=nb_cross_val,scoring='accuracy')
        #détail des itérations
        print("Succès de la validation croisée :", succes)
        #moyenne des taux de succès = estimation du taux de succès en CV
        print("Moyenne des succès : %.4f " % succes.mean()) 

         
