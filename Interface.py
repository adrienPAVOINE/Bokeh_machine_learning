# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 18:11:58 2020

@author: clementlepadellec
"""
'''
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve Interface.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/Interface
in your browser.
'''

####################################################################
#                    PACKAGES IMPORT                               #######################
####################################################################

import numpy as np
import pandas as pd
import base64
from bokeh.io import curdoc, show
import io
from bokeh.layouts import row, column, gridplot, layout
from bokeh.models import ColumnDataSource,OpenURL
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, FileInput
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral10
from bokeh.models import HoverTool, Div,Panel,Tabs
from bokeh.models.widgets import Paragraph,MultiSelect, Select, RangeSlider, Button, DataTable, DateFormatter,RadioGroup, TableColumn, Dropdown
import Classe
import Classe_Reg
from Classe import Algo_Var_Cat
from Classe_Reg import Algo_Var_Num
#
from bokeh.models import CustomJS, MultiChoice
from bokeh.plotting import figure, show, output_file
import warnings
warnings.filterwarnings("ignore")

####################################################################
#                    ONGLET 1 - SETTINGS                           #######################
####################################################################

#-------------------------------------------------------------------------
#Initialisation du TextInput : 
#-------------------------------------------------------------------------

file_input = TextInput(value="Youtubers.csv",title="Import de votre fichier")

#-------------------------------------------------------------------------
#Fonction pour définir le type de la variable : 
#-------------------------------------------------------------------------
def test_var_cible(df,var_cible):
    if (np.issubdtype(df[var_cible].dtype, np.number)==True):
        var_type='Numeric'
    else:
        var_type='String'
    return var_type

#-------------------------------------------------------------------------
#Initialisation du menu pour sélection de variables cible/explicatives : 
#-------------------------------------------------------------------------

#selectionner la var cible
menu = Select(options=[],value='', title='Variable cible')

#selectionner var expli
multi_select_var = MultiSelect(value=[], options=[])

#bouton pour lancer l'analyse
bt = Button(label="Lancer l'analyse")

#-----------------------------------------------------------------------------------------------
#Fonction Update qui met à jour tous les widgets/données quand l'utilisateur change de valeurs :
#-----------------------------------------------------------------------------------------------

#si changement de jeux de données on reset les valeurs des widgets
def load_data():
    #pas de variable sélectionnée au debut
    menu.value=''
    #pas de variables explicatives sélectionnées au début
    multi_select_var.value=[]
    #on a changé les données on appelle donc la fonction d'update des widgets
    df=pd.read_csv(str(file_input.value), sep=",")
    Columns = [TableColumn(field=Ci, title=Ci) for Ci in df.columns] # bokeh columns
    data_table = DataTable(columns=Columns, source=ColumnDataSource(df[:10])) # bokeh table
    #on affecte la data_table (prévisualisation des données) à l'élément du layout qui est attribué à la pré-visaulisation des données
    child_onglet1.children[6] =data_table
    update_controls()

def update_controls():
    #on recharge le jeu de données
    df=pd.read_csv(str(file_input.value), sep=",")
    #on met en place une data_table (car impossible de print un df avec bokeh)

    #on créer une liste des potentielles vars explicatives
    lst_expl=list(df.columns)
    
    #on attribue au "menu" la liste des variables du df, toutes peuvent être choisies comme var cible
    menu.options=list(df.columns)
    #si la valeur du widget menu != '', cela signifie que l'utilisateur à choisit une var cible
    if (menu.value != ''):
        #on va donc tester le type de la variable
        var_type=test_var_cible(df,menu.value)
        #on supprime la var cible de la liste des var explicatives
        del lst_expl[lst_expl.index(str(menu.value))]
        
        #si c'est une var textuelle alors les 3 algos possibles sont :
        if(var_type=='String'):
            your_alg1.text='<center><h1 >Arbre de décision</h1></center>'
            your_alg2.text='<center><h1 >Analyse Discriminante</h1></center>'
            your_alg3.text='<center><h1 >Regression Logistique</h1></center>'
        # sinon :
        else:
            your_alg1.text='<center><h1 >Regression Linéaire Multiple</h1></center>'
            your_alg2.text='<center><h1 >K plus proches voisins</h1></center>'
            your_alg3.text='<center><h1 >Réseaux de Neurones</h1></center>'  
    # si la valeur de menu=='' l'utilisateur n'a pas encore définit de var cible
    else:
        var_type='Pas de colonnes sélectionnées'
    #update du widget qui permet une multi-selection de variables (vars explicatives)
    multi_select_var.options=lst_expl
    
    #on prépare des string qui seront affichées pour indiquer les vars choisies par l'utilisateur
    text_for_target=str("<center><h4 > Votre variable cible est : "+str(menu.value)+"("+str(var_type)+")"+"</h4></center>")
    text_for_explain=str("<center><h4 > Vos variables explicatives sont : "+str(multi_select_var.value)+"</h4></center>")
    
    #mise à jour des indications aves les strings ci-dessus
    your_target.text=text_for_target
    your_explain.text=text_for_explain    
    
#fonction qui est appelée dès qu'un widget est modifié par l'utilisateur
def update_algos():
    #on recharge le jeu de données
    df=pd.read_csv(str(file_input.value), sep=",")
    if menu.value != '':
        var_type=test_var_cible(df,menu.value)
        #si l'utilisateur à choisit une var cible et au moins une var explicative
        if ((menu.value != '')&(multi_select_var.value !=[])):
            
            #on créer un nouveau df qui contient les vars explicatives et la var cible en dernière colonne
            new_df= pd.concat([df[multi_select_var.value],df[str(menu.value)]],axis=1)
            
            #on créer une visualisation des données qui vont être l'input de l'algo
            Columns_new_df = [TableColumn(field=Ci, title=Ci) for Ci in new_df.columns] # bokeh columns
            data_table_new_df = DataTable(columns=Columns_new_df, source=ColumnDataSource(new_df[:10])) # bokeh table
            #affectation de la prévisaulisation sur les 3 onglets des algos
            child_alg1.children[6]=data_table_new_df
            child_alg2.children[6]=data_table_new_df
            child_alg3.children[6]=data_table_new_df
            
            #si la var cible est textuelle alors on execute des fonctions qui vont lancer les algos appropriés
            if(var_type=='String'):
                nb_cv=Slider_vc.value
                nb_f=Slider_f.value
                decision_tree_maker(new_df,nb_f,nb_cv)
                analyse_disc_maker(new_df,nb_cv)
                reg_log_maker(new_df,nb_cv)
                
            else:
                nb_cv=Slider_vc.value
                nb_cv_knn=Slider_vc_knn.value
                nb_cv_rn=Slider_vc_rn.value
                nb_kv = Slider_kv.value
                reg_mult_maker(new_df,nb_cv) 
                knn_maker(new_df,nb_kv,nb_cv_knn)
                r_neur_maker(new_df,nb_cv_rn)



#-------------------------------------------------------------------------
#Si une valeur d'un widget change alors on update tout : 
#-------------------------------------------------------------------------   

#les controls correspondent aux widgets présent sur l'onglet 1
controls = [menu,multi_select_var]

#si l'utilisateur utilise un widget alors on appelle la fonction update()
for control in controls:
    control.on_change('value', lambda attr, old, new: update_controls())

#si l'utilisateur change de données alors on exécute la fonction load_data()
file_input.on_change('value', lambda attr, old, new: load_data())
bt.on_click(update_algos)
#-------------------------------------------------------------------------
#Organisation du 1er onglet : 
#-------------------------------------------------------------------------
p = Paragraph(text="""Bienvenue dans notre outil de prédiction de données par apprentassage non supervisé. Pour commencer, vous pouvez importer vos données en mettant le chemin complet de votre fichier, attention votre fichier doit être au format .csv et avoir comme séparateur des virgules. Attention il ne doit pas contenir de NA ! Une fois votre fichier importé, vous pourrez facilement pré-visualiser vos données. Pour finir, choisissez votre variables cible et vos variables explicatives, puis lancez l'analyse ! Laissez vous ensuite guider dans les onglets dédiés aux algorithmes... :)""",width=1200, height=100)
line=Div(text="__________________________________________________________________________")
title = Div(text="<center><h1 >Interface d'analyse de données</h1></center>")
sdl=Div(text="<br/>")
authors = Div(text="<h3 >Clément Le Padellec - Adrien Pavoine - Amélie Picard</h3>")
Previsualisation_data=Div(text="<center><h2 >Prévisualisation des données</h2></center>")
Target_choice=Div(text="<center><h2 >Choix de la variable cible</h2></center>")
text_for_target=""
Explain_choice=Div(text="<center><h2 >Choix des variables explicatives</h2></center>")
text_for_explain=""
your_target=Div(text=text_for_target)
your_explain=Div(text=text_for_explain)
header=column(title,authors,p, width=1500)
contents=row(file_input, width=800)

#mise en place de l'onglet un, les éléments à l'intérieur sont les "childrens"
child_onglet1 = layout([header],[line],[Previsualisation_data],[line],[sdl],[file_input],[],[line],[Target_choice],[line],[sdl],[menu],[your_target],[line],[Explain_choice],[line],[sdl],[multi_select_var],[your_explain],[bt])

#création de l'onglet
onglet1= Panel(child=child_onglet1,title="Welcome !")

####################################################################
#                    ONGLET 2 - ALGO N°1                           #######################
####################################################################

#Slider pour choisir le nombre de niveaux de l'arbre :
Slider_f=Slider(start=1, end=15, value=2, step=1, title="Nombre de niveau de l'arbre de décision ")

#Slider commun pour validation croisée :
Slider_vc=Slider(start=0, end=10, value=5, step=1, title="Cross Validation")
def update_vc(new_df):
    #si le nb de vc a changé alors on relance l'algo et on change les valeurs des childrens dans le layout
    nb_cv=Slider_vc.value
    obj=Algo_Var_Num(new_df)
    obj.Regression_line_multiple(nb_cv)
    child_alg1.children[21]=obj.val_cro
    child_alg1.children[22]=obj.mean_val_cro
    
def update_vc2(new_df):
    #si le nb de vc a changé alors on relance l'algo et on change les valeurs des childrens dans le layout
    nb_f=Slider_f.value
    nb_cv=Slider_vc.value
    obj=Algo_Var_Cat(new_df)
    obj.Arbre_decision(nb_feuille=nb_f,nb_cross_val=nb_cv)
    child_alg1.children[18]=obj.treeT
    child_alg1.children[19]=obj.tree
    child_alg1.children[32]=obj.int_succes
    child_alg1.children[33]=obj.moy_succes
    

#Arbre de décision (cas target QL)--------------------------------------------

#fonction qui execute le decision_tree sur le df bien formaté comme il faut
def decision_tree_maker(new_df,nb_f,nb_cv):
    
    obj=Algo_Var_Cat(new_df)
    obj.Arbre_decision(nb_feuille=nb_f,nb_cross_val=nb_cv)
    
    child_alg1.children[1]=obj.msg
    
    child_alg1.children[8]=obj.distrib1
    child_alg1.children[9]=obj.distrib2
    child_alg1.children[10]=obj.distrib3
    child_alg1.children[11]=obj.distrib4
    child_alg1.children[12]=obj.distrib5
    child_alg1.children[13]=sdl
    child_alg1.children[14]=obj.distribpred1
    child_alg1.children[15]=obj.distribpred2
    child_alg1.children[16]=sdl
    child_alg1.children[17]=Slider_f
    child_alg1.children[18]=obj.treeT
    child_alg1.children[19]=obj.tree
    child_alg1.children[20]=sdl
    child_alg1.children[21]=obj.coef1
    child_alg1.children[22]=obj.coef
    child_alg1.children[23]=sdl
    child_alg1.children[24]=obj.matrice_confusion
    child_alg1.children[25]=obj.cube
    child_alg1.children[26]=sdl
    child_alg1.children[27]=obj.Tx_reconnaissance
    child_alg1.children[28]=obj.Tx_erreur
    child_alg1.children[29]=obj.rapclass
    child_alg1.children[30]=obj.precclasse
    child_alg1.children[31]=Slider_vc
    child_alg1.children[32]=obj.int_succes
    child_alg1.children[33]=obj.moy_succes
    Slider_vc.on_change('value', lambda attr, old, new: update_vc2(new_df))
    Slider_f.on_change('value', lambda attr, old, new: update_vc2(new_df))

#regression linéaire multiple (cas target QT)---------------------------------
    
def reg_mult_maker(new_df,nb_cv):
    
    #instanciation de l'objet
    obj=Algo_Var_Num(new_df)
    #on récupère tout ce qu'on souhaite afficher dans l'onglet (appel de la méthode de la classe Classe_Reg)
    #coeff,mse,r2s,cross_val,msg=
    obj.Regression_line_multiple(nb_cv)
    #affectation aux childrends du layout
    child_alg1.children[1]=obj.msg

    child_alg1.children[8]=obj.title_for_coeff
    child_alg1.children[9]=sdl
    child_alg1.children[10]=obj.coef
    child_alg1.children[11]=obj.title_indicators
    child_alg1.children[12]=sdl
    child_alg1.children[13]=obj.mse
    child_alg1.children[14]=obj.r2
    child_alg1.children[15]=obj.title_fig
    child_alg1.children[16]=sdl
    child_alg1.children[17]=obj.fig
    child_alg1.children[18]=obj.title_for_vc
    child_alg1.children[19]=sdl
    child_alg1.children[20]=Slider_vc
    child_alg1.children[21]=obj.val_cro
    child_alg1.children[22]=obj.mean_val_cro
    Slider_vc.on_change('value', lambda attr, old, new: update_vc(new_df))
    

#ici on instancie car c'est modifier dans la fonction update (donc il faut pouvoir y avoir accès en global)
text_for_alg1=""
your_alg1=Div(text=text_for_alg1)

#creation du layout correspondant
child_alg1=layout([your_alg1],[],[line],[Previsualisation_data],[line],[sdl],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[])

#creation de l'algo
onglet2 = Panel(child=child_alg1, title="ALGO 1")
####################################################################
#                    ONGLET 3 - ALGO N°2                           #######################
####################################################################

#K plus proche voisin (cas target QT)
#slider de selection du nombre de voisin
Slider_kv=Slider(start=1, end=15, value=5, step=1, title="Nombre de K plus proches voisins")

Slider_vc_knn=Slider(start=1, end=10, value=5, step=1, title="Cross Validation")

def update_kv(new_df):
    #si le nb de k plus proches voisins a changé alors on relance l'algo et on change les valeurs des childrens dans le layout
    nb_kv=Slider_kv.value
    nb_cv_k=Slider_vc_knn.value
    obj=Algo_Var_Num(new_df)
    obj.K_Proches_Voisins_Reg(nb_kv,nb_cv_k)
    
    child_alg2.children[11]=obj.mse
    child_alg2.children[12]=obj.r2
    child_alg2.children[15]=obj.fig
    child_alg2.children[19]=obj.val_cro
    child_alg2.children[20]=obj.mean_val_cro    

def update_kv2(new_df):
    nb_cv_k=Slider_vc_knn.value
    obj=Algo_Var_Cat(new_df)
    obj.Analyse_Discriminante(nb_cross_val=nb_cv_k)
    
    child_alg2.children[21]=obj.int_succes
    child_alg2.children[22]=obj.moy_succes    


#Analyse Discriminante (cas target QL)
#fonction qui execute le decision_tree sur le df bien formaté comme il faut
def analyse_disc_maker(new_df,nb_cv_k):
    obj=Algo_Var_Cat(new_df)
    obj.Analyse_Discriminante(nb_cv_k)
    
    #affectation aux childrends du layout
    child_alg2.children[1]=obj.msg
    child_alg2.children[8]=obj.distrib1
    child_alg2.children[9]=obj.distrib2
    child_alg2.children[10]=obj.distrib3
    child_alg2.children[11]=obj.distrib4
    child_alg2.children[12]=obj.distrib5
    child_alg2.children[13]=sdl
    child_alg2.children[14]=obj.coefT
    child_alg2.children[15]=obj.coef
    child_alg2.children[16]=obj.matrice_confusion
    child_alg2.children[17]=obj.cube
    child_alg2.children[18]=obj.Tx_reconnaissance
    child_alg2.children[19]=obj.Tx_erreur
    child_alg2.children[20]=Slider_vc_knn
    child_alg2.children[21]=obj.int_succes
    child_alg2.children[22]=obj.moy_succes
    Slider_vc_knn.on_change('value', lambda attr, old, new: update_kv2(new_df))
    



def knn_maker(new_df,kv, nb_cv_knn):
    
    #instanciation de l'objet
    obj=Algo_Var_Num(new_df)
    #on récupère tout ce qu'on souhaite afficher dans l'onglet (appel de la méthode de la classe Classe_Reg)
    #coeff,mse,r2s,cross_val,msg=
    obj.K_Proches_Voisins_Reg(nb_cv_knn)
    
    #affectation aux childrends du layout
    child_alg2.children[1]=obj.msg
    child_alg2.children[8]=Slider_kv
    child_alg2.children[9]=obj.title_indicators
    child_alg2.children[10]=sdl
    child_alg2.children[11]=obj.mse
    child_alg2.children[12]=obj.r2
    child_alg2.children[13]=obj.title_fig
    child_alg2.children[14]=sdl
    child_alg2.children[15]=obj.fig
    child_alg2.children[16]=obj.title_for_vc
    child_alg2.children[17]=sdl
    child_alg2.children[18]=Slider_vc_knn
    child_alg2.children[19]=obj.val_cro
    child_alg2.children[20]=obj.mean_val_cro
    Slider_vc_knn.on_change('value', lambda attr, old, new: update_kv(new_df))
    Slider_kv.on_change('value', lambda attr, old, new: update_kv(new_df))

#ici on instancie car c'est modifier dans la fonction update (donc il faut pouvoir y avoir accès en global)
text_for_alg2=""
your_alg2=Div(text=text_for_alg2)

#creation du layout correspondant
child_alg2=layout([your_alg2],[],[line],[Previsualisation_data],[line],[sdl],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[])

#creation de l'algo
onglet3 = Panel(child=child_alg2, title="ALGO 2")


####################################################################
#                    ONGLET 4 - ALGO N°3                           #######################
####################################################################

#slider de selection du nombre de voisin

Slider_vc_rn=Slider(start=1, end=15, value=5, step=1, title="Cross Validation")

#fonction qui execute la reg multiple sur le df bien formaté comme il faut
    
    
def update_vc_rn(new_df):
    #si le nb de k plus proches voisins a changé alors on relance l'algo et on change les valeurs des childrens dans le layout
    nb_cv_rn=Slider_vc_rn.value
    obj=Algo_Var_Num(new_df)
    obj.Reseau_Neurone(nb_cv_rn)
    child_alg3.children[19]=obj.val_cro
    child_alg3.children[20]=obj.mean_val_cro     
    
def update_vc_rn2(new_df):
    #si le nb de k plus proches voisins a changé alors on relance l'algo et on change les valeurs des childrens dans le layout
    nb_cv_rn=Slider_vc_rn.value
    obj=Algo_Var_Cat(new_df) 
    obj.Regression_log(nb_cv_rn)
    child_alg3.children[27]=obj.int_succes
    child_alg3.children[28]=obj.moy_succes  
    

#Reg Log (cas target QL)
def reg_log_maker(new_df,nb_cv_rn):
    
    obj=Algo_Var_Cat(new_df) 
    obj.Regression_log(nb_cv_rn)
    
    child_alg3.children[1]=obj.msg

    child_alg3.children[8]=obj.distrib1
    child_alg3.children[9]=obj.distrib2
    child_alg3.children[10]=obj.distrib3
    child_alg3.children[11]=obj.distrib4
    child_alg3.children[12]=obj.distrib5
    child_alg3.children[13]=sdl
    child_alg3.children[14]=obj.coefT
    child_alg3.children[15]=obj.coef
    child_alg3.children[16]=obj.const
    child_alg3.children[17]=sdl
    child_alg3.children[18]=obj.log_vraisemblance
    child_alg3.children[19]=obj.matrice_confusion
    child_alg3.children[20]=obj.cube
    child_alg3.children[21]=sdl
    child_alg3.children[22]=obj.Tx_reconnaissance
    child_alg3.children[23]=obj.Tx_erreur
    child_alg3.children[24]=obj.fig2
    child_alg3.children[25]=obj.aucSm2
    child_alg3.children[26]=Slider_vc_rn
    child_alg3.children[27]=obj.int_succes
    child_alg3.children[28]=obj.moy_succes
    Slider_vc_rn.on_change('value', lambda attr, old, new: update_vc_rn2(new_df))
 


def r_neur_maker(new_df,nb_cv_rn):
    #instanciation de l'objet
    obj=Algo_Var_Num(new_df)
    #on récupère tout ce qu'on souhaite afficher dans l'onglet (appel de la méthode de la classe Classe_Reg)
    #coeff,mse,r2s,cross_val,msg=
    obj.Reseau_Neurone(nb_cv_rn)
    #affectation aux childrends du layout
    
    child_alg3.children[1]=obj.msg

    child_alg3.children[9]=obj.title_indicators
    child_alg3.children[10]=sdl
    child_alg3.children[11]=obj.mse
    child_alg3.children[12]=obj.r2
    child_alg3.children[13]=obj.title_fig
    child_alg3.children[14]=sdl
    child_alg3.children[15]=obj.fig
    child_alg3.children[16]=obj.title_for_vc
    child_alg3.children[17]=sdl
    child_alg3.children[18]=Slider_vc_rn
    child_alg3.children[19]=obj.val_cro
    child_alg3.children[20]=obj.mean_val_cro
    Slider_vc_rn.on_change('value', lambda attr, old, new: update_vc_rn(new_df))
        

#ici on instancie car c'est modifier dans la fonction update (donc il faut pouvoir y avoir accès en global)
text_for_alg3=""
your_alg3=Div(text=text_for_alg3)

#creation du layout correspondant
child_alg3=layout([your_alg3],[],[line],[Previsualisation_data],[line],[sdl],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[])

#creation de l'algo
onglet4 = Panel(child=child_alg3, title="ALGO 3")


####################################################################
#                    MISE EN PLACE DU PANEL                        #######################
####################################################################

#on met en place l'interface globale avec nos 4 onglets
panel = Tabs(tabs=[onglet1,onglet2,onglet3,onglet4])
doc=curdoc()
curdoc().add_root(panel) 
