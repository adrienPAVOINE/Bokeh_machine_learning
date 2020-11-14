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
from sklearn import model_selection
import matplotlib.pyplot as plt
import statsmodels as sm
from statsmodels.tools import add_constant
from statsmodels.api import Logit
import scipy
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
        print(mc)
        
        #taux de reconnaissance
        acc = metrics.accuracy_score(self.yTest,yPred)
        print(acc)
        
        #taux d'erreur
        err = 1.0 - acc
        print(err)
        
        #rappel par classe
        print(metrics.recall_score(self.yTest,yPred,average=None))
        
        #precision par classe
        print(metrics.precision_score(self.yTest,yPred,average=None))
        
        #rapport général
        print(metrics.classification_report(self.yTest,yPred))
        
        #-------------------------------------------------------------------------
        #Validation croisée : 
        #-------------------------------------------------------------------------
        #évaluation en validation croisée : 
        # paramètre par défaut : nb_cross_val=10
        succes = model_selection.cross_val_score(dtree,self.X,self.y,cv=nb_cross_val,scoring='accuracy')
        #détail des itérations
        print(succes)
        #moyenne des taux de succès = estimation du taux de succès en CV
        print(succes.mean())
        
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
        print(final)
                
        #-------------------------------------------------------------------------
        #Prédiction : 
        #-------------------------------------------------------------------------
        ypred = lda.predict(self.XTest)
        #matrice de confusion
        mc = metrics.confusion_matrix(self.yTest,ypred)
        print(mc)
        
        #calcul du taux d'erreur
        print(1.0-metrics.accuracy_score(self.yTest,ypred))
        
        #calcul des sensibilité (rappel) et précision par classe
        print(metrics.classification_report(self.yTest,ypred))
        
        #-------------------------------------------------------------------------
        #Validation croisée : 
        #-------------------------------------------------------------------------
        #évaluation en validation croisée : 
        # paramètre par défaut : nb_cross_val=10
        succes = model_selection.cross_val_score(lda,self.X,self.y,cv=nb_cross_val,scoring='accuracy')
        #détail des itérations
        print(succes)
        #moyenne des taux de succès = estimation du taux de succès en CV
        print(succes.mean())


    #-------------------------------------------------------------------------
    #Création de la regressionn logistiques binaire
    #-------------------------------------------------------------------------  
    def Regression_log(self,alpha=0.1,nb_cross_val=10):
        #rajout d'une constante de valeur 1 dans la première colonne
        XTrainBis = sm.tools.add_constant(self.XTrain)
        #régression logistique - on passe la cible et les explicatives
        lr = Logit(endog=self.yTrain,exog=XTrainBis)
        #algorithme de Newton-Raphson utilisé par défaut
        res = lr.fit() 
        
        print("Résumé des résultats")
        print(res.summary()) 
        
        print("Intervalle de confiance des coefficients à %.i pourcent" % ((1-alpha)*100))
        print(res.conf_int(alpha=alpha))
         
        print("Voici les coefficients estimés")
        print(res.params)
         
        print("Matrice de confusion")
        print(res.pred_table()) 
        
        print("#Pseudo-R2 :")
        print("Log-vraisemblance du modèle : %.4f" % (res.llf))
        
        #log-vraisemblance du null modèle
        print("Log-vraisemblance du null modèle : %.4f" % (res.llnull))
        #R2 de McFadden
        print("R2 de McFadden : %.4f" % (res.prsquared))
        #exponenielle de LL_null
        L0 = numpy.exp(res.llnull)
        #exponentielle de LL_modèle
        La = numpy.exp(res.llf)
        #taille de l'échantillon
        n = self.XTrain.shape[0]
        #R2 de Cox et Snell
        R2CS = 1.0 - (L0 / La)**(2.0/n)
        print("R2 de Cox - Snell : %.4f" % (R2CS))
        #max du R2 de COx-Snell
        maxR2CS = 1.0 - (L0)**(2.0/n)
        #R2 de Nagelkerke
        R2N = R2CS / maxR2CS
        print("R2 de Nagelkerke : %.4f" % (R2N))
        
        print("#Evaluation basés sur les scores")
        # On se base sur les scores - c.-à-d. probabilité d'affectation à la modalité cible - pour évaluer la qualité d'approximation du modèle. Il faut déjà les calculer.
        #scores fournis par la régression pour chaque individus
        scores = lr.cdf(res.fittedvalues)        
        
        ### Diagramme de fiabilité
        #data frame temporaire avec y et les scores
        df = pandas.DataFrame({"y":self.yTrain,"score":scores})
        #5 intervalles de largeur égales
        intv = pandas.cut(df.score,bins=5,include_lowest=True)
        #intégrées dans le df
        df['intv'] = intv
           
        print("Moyenne des scores par groupe :")
        m_score = df.pivot_table(index="intv",values="score",aggfunc="mean")
        print(m_score)
        print("Moyenne des y - qui équivaut à une proportion puisque 0/1")
        m_y = df.pivot_table(index="intv",values="y",aggfunc="mean")
        print(m_y)
        
        #construire la diagonale
        plt.plot(numpy.arange(0,1,0.1),numpy.arange(0,1,0.1),'b')
        #rajouter notre diagramme
        plt.plot(m_score,m_y,"go-")
        #titre
        plt.title("Diagramme de fiabilité")
        #faire apparaître
        plt.show()
        
        print("#Test de Hosmer-Lemeshow : ")
        #effectifs par groupe
        n_tot = df.pivot_table(index="intv",values="y",aggfunc="count").values[:,0]
        #somme des scores par groupes
        s_scores = df.pivot_table(index='intv',values="score",aggfunc="sum").values[:,0]
        #nombre de positifs par groupe
        n_pos = df.pivot_table(index="intv",values="y",aggfunc="sum").values[:,0]
        #nombre de négatifs par groupe
        n_neg = n_tot - n_pos
        #statistique de Hosmer-Lemeshow
        C1 = numpy.sum((n_pos - s_scores)**2/s_scores)
        C2 = numpy.sum((n_neg - (n_tot - s_scores))**2/((n_tot - s_scores)))
        HL = C1 + C2
        print("Statistique de Hosmer-Lemeshow : %.4f  " % (HL))
        #probabilité critique
        pvalue = 1.0 - scipy.stats.chi2.cdf(HL,8)
        print("p-value : %.4f " % (pvalue))
        
        print("Tests de significativité des coefficients : ")
        #AIC du modèle
        print("AIC du modèle : %.4f" % (res.aic))
        #AIC du modèle trivial - 1 seul param. estimé, la constante
        aic_null = (-2) * res.llnull + 2 * (1)
        print("AIC du modèle trivial : %.4f" % (aic_null))
        #BIC du modèle
        print("BIC du modèle : %.4f" % (res.bic))
        #BIC du modèle trivial - 1 seul param. estimé, la constante
        
        bic_null = (-2) * res.llnull + numpy.log(n) * (1)
        print("BIC du modèle trivial : %.4f" % (bic_null))
        
        print("# Prédiction et matrice de confusion")
        #préparation de l'échantillon test
        #par adjonction de la constante
        XTest_Bis = add_constant(self.XTest)
        #calcul de la prédiction sur l'échantillon test
        predProbaSm = res.predict(XTest_Bis)
        
        #convertir en prédiction brute
        predSm = numpy.where(predProbaSm > 0.5, 1, 0)
        print(numpy.unique(predSm,return_counts=True))
        print("Nombre d'individus associé à la classe 0 : %.i" % numpy.unique(predSm,return_counts=True)[1][0])
        print("Nombre d'individus associé à la classe 1 : %.i" %numpy.unique(predSm,return_counts=True)[1][1])
        
        print("Matrice de confusion")
        mcSm = pandas.crosstab(self.yTest,predSm)
        print(mcSm)
        
        #transformer en matrice Numpy
        mcSmNumpy = mcSm.values
        
        #Indicateur de performance :
        print("Indicateur de performance :")
        #taux de reconnaissance
        accSm = numpy.sum(numpy.diagonal(mcSmNumpy))/numpy.sum(mcSmNumpy)
        print("Taux de reconnaissance : %.4f" % (accSm))
        #taux d'erreur
        errSm = 1.0 - accSm
        print("Taux d'erreur : %.4f" % (errSm))
        
        # ### Courbe ROC en test
        # Construire la courbe ROC et calculer la valeur de l'AUC sur l'échantillon test. On utilise "scikit-learn" pour aller au plus court.
        #colonnes pour les courbes ROC
        fprSm, tprSm, _ = metrics.roc_curve(self.yTest,predProbaSm,pos_label=1)
        
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
        aucSm = metrics.roc_auc_score(self.yTest,predProbaSm)
        print("AUC : %.4f" % (aucSm))
        
        #-------------------------------------------------------------------------
        #Validation croisée : 
        #-------------------------------------------------------------------------
        lr2 = LogisticRegression()
        #évaluation en validation croisée : 10 cross-validation
        # paramètre par défaut : nb_cross_val=10
        succes = model_selection.cross_val_score(lr2,self.X,self.y,cv=nb_cross_val,scoring='accuracy')
        #détail des itérations
        print(succes)
        #moyenne des taux de succès = estimation du taux de succès en CV
        print(succes.mean()) 

         
