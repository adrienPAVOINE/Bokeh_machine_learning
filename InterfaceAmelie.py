# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 18:11:58 2020
@author: clementlepadellec
"""


####################################################################
#                    PACKAGES IMPORT                               #######################
####################################################################

import numpy as np
import pandas as pd
import base64
from bokeh.io import curdoc, show
import io
from bokeh.layouts import row, column, gridplot, layout
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, FileInput
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral10
from bokeh.models import HoverTool, Div,Panel,Tabs
from bokeh.models.widgets import MultiSelect, Select, RangeSlider, Button, DataTable, DateFormatter,RadioGroup, TableColumn, Dropdown
#


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


menu = Select(options=[],value='', title='Variable cible')
multi_select_var = MultiSelect(value=[], options=[])


#-----------------------------------------------------------------------------------------------
#Fonction Update qui met à jour tous les widgets/données quand l'utilisateur change de valeurs :
#-----------------------------------------------------------------------------------------------

def update():
    df=pd.read_csv(str(file_input.value), sep=",")
    Columns = [TableColumn(field=Ci, title=Ci) for Ci in df.columns] # bokeh columns
    data_table = DataTable(columns=Columns, source=ColumnDataSource(df[:10])) # bokeh table
    child_onglet1.children[4] =data_table
    #menu=Select(options=list(df.columns),value=list(df.columns)[0], title='bo')
    lst_expl=list(df.columns)
    menu.options=list(df.columns)
    if (menu.value != ''):
        var_type=test_var_cible(df,menu.value)
        
        del lst_expl[lst_expl.index(str(menu.value))]
        if(var_type=='String'):
            your_alg1.text='Arbre de décision'
            your_alg2.text='Analyse_Discriminante'
            your_alg3.text='Regression_log'
        else:
            your_alg1.text='Regression_line_multiple'
            your_alg2.text='K_Proches_Voisins_Reg'
            your_alg3.text='Reseau_Neurone'            
    else:
        var_type='Pas de colonnes sélectionnées'
    
    
    if ((menu.value != '')&(multi_select_var.value !=[])):
        new_df= pd.concat([df[multi_select_var.value],df[str(menu.value)]],axis=1)
        Columns_new_df = [TableColumn(field=Ci, title=Ci) for Ci in new_df.columns] # bokeh columns
        data_table_new_df = DataTable(columns=Columns_new_df, source=ColumnDataSource(new_df[:10])) # bokeh table
        child_alg1.children[1]=data_table_new_df
        child_alg2.children[1]=data_table_new_df
        child_alg3.children[1]=data_table_new_df
        
    multi_select_var.options=lst_expl
    text_for_target=str("<center><h4 > Votre variable cible est : "+str(menu.value)+"("+str(var_type)+")"+"</h4></center>")
    text_for_explain=str("<center><h4 > Vos variables explicatives sont : "+str(multi_select_var.value)+"</h4></center>")
    
    your_target.text=text_for_target
    your_explain.text=text_for_explain
    return menu,your_target,your_explain,


#-------------------------------------------------------------------------
#Si une valeur d'un widget change alors on update tout : 
#-------------------------------------------------------------------------   
controls = [file_input,menu,multi_select_var]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())


#-------------------------------------------------------------------------
#Organisation du 1er onglet : 
#-------------------------------------------------------------------------

title = Div(text="<center><h1 >Interface d'analyse de données</h1></center>")
sdl=column(Div(text="<br/>"))
authors = Div(text="<h3 >Clément Le Padellec - Adrien Pavoine - Amélie Picard</h3>")
Previsualisation_data=Div(text="<center><h2 >Prévisualisation des données</h2></center>")
Target_choice=Div(text="<center><h2 >Choix de la variable cible</h2></center>")
text_for_target=""
Explain_choice=Div(text="<center><h2 >Choix des variables explicatives</h2></center>")
text_for_explain=""
your_target=Div(text=text_for_target)
your_explain=Div(text=text_for_explain)
header=column(title,authors, width=1500)
contents=row(file_input, width=800)

child_onglet1 = layout([header],[sdl],[Previsualisation_data],[file_input],[],[Target_choice],[menu],[your_target],[Explain_choice],[multi_select_var],[your_explain])
onglet1= Panel(child=child_onglet1,title="Welcome !")

####################################################################
#                    ONGLET 2 - ALGO N°1                           #######################
####################################################################
    
from Classe import Algo_Var_Cat
#Données pour reg test 2
df4=pd.read_excel("diabete_reg_logistique.xlsx")
#Vérif si la var cible a pour modalité O et 1
#if ((df4['diabete'].unique()==[0,1])[0] & (df4['diabete'].unique()==[0,1])[1]) :
   # mod1=df4['diabete'].unique()[0]
    #mod2=df4['diabete'].unique()[1]
for i in range(0,len(df4['diabete'])):
    if (df4['diabete'][i]=='positive') : 
        df4['diabete'][i]=0
    
    if (df4['diabete'][i]=="negative") : 
        df4['diabete'][i]=1
df4['diabete'] = df4['diabete'].astype('int')  

test3=Algo_Var_Cat(df4)
#multi = T ou F
multi_classe=False
test3.Regression_log(multi=multi_classe)

text_for_alg1=""
your_alg1=Div(text=text_for_alg1)
#ordre d'affichage des variables de classe de l'Algo
if multi_classe==False :
    child_alg1=layout([your_alg1],[test3.distrib],[test3.coef],[test3.const],[test3.log_vraisemblance],
                      [test3.matrice_confusion],[test3.Tx_reconnaissance],
                      [test3.rapport],[test3.fig2],[test3.aucSm2],[test3.int_succes],
                      [test3.moy_succes])
else :
    child_alg1=layout([your_alg1],[test3.distrib],[test3.coef],[test3.const],
                      [test3.matrice_confusion],[test3.Tx_reconnaissance],
                      [test3.rapport],[test3.int_succes],[test3.moy_succes])
onglet2 = Panel(child=child_alg1, title="ALGO 1")



####################################################################
#                    ONGLET 3 - ALGO N°2                           #######################
####################################################################

x = np.arange(start=1, stop=6)
x_exp = np.exp(x)

x_log = np.log(x)
fig1= figure(title='Fonction logarithme', x_axis_label='Ascisses', y_axis_label='Ordonnées')
fig1.circle(x, x_log, legend_label="log x", line_width=2, line_color="green", color='green', size=5)

text_for_alg2=""
your_alg2=Div(text=text_for_alg2)
child_alg2=layout([your_alg2],[],[fig1])
onglet3 = Panel(child=child_alg2, title="ALGO 2")


####################################################################
#                    ONGLET 4 - ALGO N°3                           #######################
####################################################################

x_log = np.log(x)
fig1= figure(title='Fonction logarithme', x_axis_label='Ascisses', y_axis_label='Ordonnées')
fig1.circle(x, x_log, legend_label="log x", line_width=2, line_color="green", color='green', size=5)

text_for_alg3=""
your_alg3=Div(text=text_for_alg3)
child_alg3=layout([your_alg3],[],[fig1])
onglet4 = Panel(child=child_alg3, title="ALGO 3")


####################################################################
#                    MISE EN PLACE DU PANEL                        #######################
####################################################################

panel = Tabs(tabs=[onglet1,onglet2,onglet3,onglet4])
doc=curdoc()
curdoc().add_root(panel) 
