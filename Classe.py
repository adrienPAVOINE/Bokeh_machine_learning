# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 12:18:22 2020

@author: ameli
"""
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
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
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, FileInput
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral10
from bokeh.models import HoverTool, Div,Panel,Tabs
from bokeh.models.widgets import Paragraph,MultiSelect, Select, RangeSlider, Button, DataTable, DateFormatter,RadioGroup, TableColumn, Dropdown,StringFormatter,SumAggregator, DataCube,GroupingInfo

class Algo_Var_Cat():
 
    #-------------------------------------------------------------------------
    #Initialisation des données 
    #-------------------------------------------------------------------------
    def __init__(self,df,size=-1):
        self.df=df
        self.var_cible=self.df.columns[-1]
        self.y=self.df.iloc[:,-1]
        
        #initialisation de la taille de l'ech train :
        
        if (size==-1) : 
            size=round(len(self.df.values[:,-1])*0.3)
        #subdiviser les données en échantillons d'apprentissage et de test
        #Choix à l'utilisateur, taille de l'échantillon test : sinon le choix par défaut 70% 30%
        dfTrain, dfTest = train_test_split(self.df,test_size=size,random_state=1,stratify=self.df.iloc[:,-1])
        
        #Séparer X et Y :
        #self.y=self.df.iloc[:,-1]
        self.X=self.df.iloc[:,0:(len(self.df.columns)-1)]
        self.yTrain=dfTrain.iloc[:,-1]
        self.XTrain=dfTrain.iloc[:,0:(len(self.df.columns)-1)]
        self.yTest=dfTest.iloc[:,-1]
        self.XTest=dfTest.iloc[:,0:(len(self.df.columns)-1)]
        
        train = self.yTrain.value_counts(normalize=True)
        test = self.yTest.value_counts(normalize=True)
        
        #distribution des classes
        self.distrib1=Div(text="<h4>Distribution des classes de la variables cible :</h4>")
        self.distrib2=Div(text="Classe d'entrainement : <br/>")
        temp=pandas.DataFrame({"var":train.index,"distribution":train.values})
        columns=[TableColumn(field=Ci, title=Ci) for Ci in temp.columns] 
        self.distrib3=DataTable(source=ColumnDataSource(temp),columns=columns)
        self.distrib4=Div(text="Classe de test : <br/>")
        temp=pandas.DataFrame({"var":test.index,"distribution":test.values})
        columns=[TableColumn(field=Ci, title=Ci) for Ci in temp.columns] 
        self.distrib5=DataTable(source=ColumnDataSource(temp),columns=columns)
    
    #-------------------------------------------------------------------------
    #Création de l'arbre de décision et de la prédiction
    #-------------------------------------------------------------------------
    def Arbre_decision(self,nb_feuille=2,nb_cross_val=3):
        
        self.msg=Paragraph(text="""Vous êtes dans la partie réservé à la prédiction d'une variable 
                           qualitative à l'aide d'un arbre de décicion. Dans cet onglet vous trouverez 
                           tout d'abord une pré-visualisation des données sur lesquelles va 
                           s'appliquer l'algorithme. Puis vous pourrez observer la distribution 
                           de la variables cible pour l'échantillon d'apprentissage et l'échantillon test.
                           Aprés la création de l'arbre de décision, nous l'affichons à l'aide d'une image 
                           (si elle ne s'affiche pas elle se trouve dans le même répetroire que le projet 
                            sous le nom 'tree.jpg'), ainsi que la liste de l'importance des variables. 
                           Il est possible grâce à un slider de choisir le nombre de niveau de l'arbre 
                           de décision.
                           Afin de savoir si votre modèle est bon ou non, vous retrouverez la matrice de 
                           confusion, le taux de reconnaissance et le taux d'erreur, le rappel et la 
                           précision par classe. Enfin vous pourrez vous même 
                           faire de la cross validation, utilisez 
                           juste le slider pour réaliser le nombre de cross validation choisi, et
                           obtenir le pourcentage de succès pour chacun d'eux.""",width=1200, height=100)

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
        #tree_rules = export_text(dtree,feature_names = list(self.df.columns[:-1]),show_weights=True)
        tree_rules = export_graphviz(dtree,feature_names = list(self.df.columns[:-1]))
        self.reglesT=Div(text="<h4>Régles de décision : </h4>")
        self.regles=Div(text=str(tree_rules))
        
        self.treeT=Div(text="<h4>Arbre de décision : </h4>")
        
        plt.figure()
        plot_tree(dtree,feature_names = list(self.df.columns[:-1]),filled=True)
        plt.savefig('tree.jpg', dpi=95)
        self.tree= Div(text="""<img src="tree.jpg", alt="L'image 'tree.jpg' est enregistrée dans le répertoire courant">""", width=150, height=150)
        
        
        #-------------------------------------------------------------------------
        #Prédiction : 
        #-------------------------------------------------------------------------
        
        #importance des variables
        imp = {"VarName":self.X.columns,"Importance":dtree.feature_importances_}
        self.coef1=Div(text="<h4>Importance des variables :</h4>")
        temp=pandas.DataFrame(imp).sort_values(by="Importance",ascending=False)
        columns=[TableColumn(field=Ci, title=Ci) for Ci in temp.columns] 
        self.coef=DataTable(source=ColumnDataSource(temp),columns=columns)

        #prédiction en test
        yPred = dtree.predict(self.XTest)
        
        #distribution des classes prédictes
       #Intéressant d'afficher cette information
        self.distribpred1=Div(text="Classe de prédiction : <br/>")
        temp=pandas.DataFrame({"var":np.unique(yPred,return_counts=True)[0],"distribution":np.unique(yPred,return_counts=True)[1]})
        columns=[TableColumn(field=Ci, title=Ci) for Ci in temp.columns] 
        self.distribpred2=DataTable(source=ColumnDataSource(temp),columns=columns)
               
        
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
        rap=metrics.classification_report(self.yTest,yPred)
        #,output_dict=True
        self.rap1=Div(text="<h4> Rapport sur la qualité de prédiction :</h4>")
        #temp=pandas.DataFrame({"var":rap['C0'].keys(),"valeur":rap['C0'].values() })
        #columns=[TableColumn(field=Ci, title=Ci) for Ci in temp.columns] 
        #self.rapport=DataTable(source=ColumnDataSource(temp),columns=columns)
        self.rapport=Div(text=str(rap))
        
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
        
        self.msg=Paragraph(text="""Vous êtes dans la partie réservé à la prédiction d'une variable 
                           qualitative à l'aide d'une analyse discriminante. Dans cet onglet vous trouverez 
                           tout d'abord une pré-visualisation des données sur lesquelles va 
                           s'appliquer l'algorithme. Puis vous pourrez observer la distribution 
                           de la variables cible pour l'échantillon d'apprentissage et l'échantillon test.
                           Aprés l'analyse discriminante, nous affichons les résultats à l'aide d'une 
                           table les coefficients et les intercepts (pour une variable cible à plus 
                           de deux modalités) et la table des coefficients pour la modalité positive (
                           pour une variable à deux modalités).                     
                           Afin de savoir si votre modèle est bon ou non, vous retrouverez la matrice de 
                           confusion, le taux de reconnaissance et le taux d'erreur. 
                           Enfin vous pourrez vous même faire de la cross validation, utilisez 
                           juste le slider pour réaliser le nombre de cross validation choisi, et
                           obtenir le pourcentage de succès pour chacun d'eux.""",width=1200, height=100)

        #classe LDA
        #instanciation 
        lda = LinearDiscriminantAnalysis()
        #apprentissage
        lda.fit(self.XTrain,self.yTrain)
        
        if len(self.y.unique())==2 :
            self.coefT=Div(text="<h4>Table des coefficients de chaque variable : </h4>")
            temp=pandas.DataFrame({"var":self.XTrain.columns,"positif":lda.coef_[0]})
            columns=[TableColumn(field=Ci, title=Ci) for Ci in temp.columns] 
            self.coef=DataTable(source=ColumnDataSource(temp),columns=columns)
        else :
            #structure pour affichage des coefficients et des intercepts
            tmp= pandas.DataFrame(lda.coef_.transpose(),columns=lda.classes_,index=self.XTrain.columns)
            tmp2={tmp.columns[0] : [lda.intercept_[0]]}
            for i in range(1,len(tmp.columns)): 
                tmp2.update({tmp.columns[i] : [lda.intercept_[i]]})
            tmp2=pandas.DataFrame(tmp2)
            tmp2=tmp2.rename(index={0 : "Constante"})
            final=pandas.concat([tmp2,tmp])
            self.coefT=Div(text="<h4>Table des coefficients et des intercepts : </h4>")
            
            d = dict()
            d["affichage"]= ['','','','','']
            d["var"]=['Constante', 'area','parameter','compactness',"len_kernel"]
            for i in (range(len(final.columns))):
                d[final.columns[i]]=final.iloc[:,i]
                
            source = ColumnDataSource(data=d)
            target = ColumnDataSource(data=dict(row_indices=[], labels=[]))
            formatter = StringFormatter(font_style='bold')
            columns=[TableColumn(field='var', title="", width=40, sortable=False, formatter=formatter)]
            columns[1:(len(final.columns))]=[TableColumn(field=str(NomMod), title=str(NomMod), width=40, sortable=False) for NomMod in final.columns]
            grouping = [GroupingInfo(getter='affichage'),]
            self.coef = DataCube(source=source, columns=columns, grouping=grouping, target=target)
        
        
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
            d[d["var"][i]]=mcSmNumpy[i]
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
        self.Tx_reconnaissance=Div(text="<h4>Taux de reconnaissance :</h4> " + str(round(accSm,4)))
                
        #calcul du taux d'erreur
        self.Tx_erreur=Div(text="<h4>Taux d'erreur :</h4><br/>" + str(round(1.0-metrics.accuracy_score(self.yTest,ypred),4)))
        
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
    def Regression_log(self,nb_cross_val=10):
        
        if (len(self.y.unique())==2) :
            mod1=self.y.unique()[0]
            mod2=self.y.unique()[1]
            for i in range(0,len(self.df[str(self.var_cible)])):
                if (self.df[str(self.var_cible)][i]==mod1) : 
                    self.df[str(self.var_cible)][i]=0
                if (self.df[str(self.var_cible)][i]==mod2) : 
                    self.df[str(self.var_cible)][i]=1
            self.df[str(self.var_cible)] = self.df[str(self.var_cible)].astype('int')
            size=round(len(self.df.values[:,-1])*0.3)
            dfTrain, dfTest = train_test_split(self.df,test_size=size,random_state=1,stratify=self.df.iloc[:,-1])
            self.y=self.df.iloc[:,-1]
            self.X=self.df.iloc[:,0:(len(self.df.columns)-1)]
            self.yTrain=dfTrain.iloc[:,-1]
            self.XTrain=dfTrain.iloc[:,0:(len(self.df.columns)-1)]
            self.yTest=dfTest.iloc[:,-1]
            self.XTest=dfTest.iloc[:,0:(len(self.df.columns)-1)]
        
        self.msg=Paragraph(text="""Vous êtes dans la partie réservé à la prédiction d'une variable 
                           qualitative à l'aide d'une régression logistique. Dans cet onglet vous trouverez 
                           tout d'abord une pré-visualisation des données sur lesquelles va 
                           s'appliquer l'algorithme. Puis vous pourrez observer la distribution 
                           de la variables cible pour l'échantillon d'apprentissage et l'échantillon test.
                           Aprés la régression logistique, nous affichons les résultats à l'aide d'une 
                           table des coeficients des variables et les intercepts.
                           Afin de savoir si votre modèle est bon ou non, vous retrouverez la matrice de 
                           confusion, le taux de reconnaissance et le taux d'erreur. Si la variable cible 
                           à deux madalités, alors nous afficherons le log de vraissemblance, la courbe 
                           ROC ainsi que l'AUC correspondant.
                           Enfin vous pourrez vous même faire de la cross validation, utilisez 
                           juste le slider pour réaliser le nombre de cross validation choisi, et
                           obtenir le pourcentage de succès pour chacun d'eux.""",width=1200, height=100)

        #Test si la variable cible est multiclasse (+ de 2 niveaux de modalités)
        if (len(self.df[str(self.var_cible)].unique())==2) :
            multi=False
        else : 
            multi=True
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
        self.coefT=Div(text="<h4>Coefficients des variables : </h4>")
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
        else :
            self.log_vraisemblance=Div(text="")
            
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
            
            self.matrice_confusion=Div(text="</br><h4>Matrice de confusion :</h4>")
            source = ColumnDataSource(data=dict(
                affichage=["",""],
                var=['positif', 'negatif'],
                positif=[mcSm[0][0],mcSm[0][1]],
                negatif=[mcSm[1][0],mcSm[1][1]]
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
            self.Tx_reconnaissance=Div(text="<h4>Taux de reconnaissance : </h4>" + str(round(accSm,4)))
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
            self.aucSm2=Div(text=" AUC : " +str(round(aucSm,4)))
            
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
                d[d["var"][i]]=mcSm[i]
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
            
            self.fig2= Div(text="")
            self.aucSm2=Div(text="")
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
