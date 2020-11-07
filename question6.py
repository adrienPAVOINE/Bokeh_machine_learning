''' Present an interactive function explorer with slider widgets.
Scrub the sliders to change the properties of the ``sin`` curve, or
type into the title text box to update the title of the plot.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve sliders.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/sliders
in your browser.
'''
import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral10
from bokeh.models import HoverTool
from bokeh.models.widgets import MultiSelect, Select, RangeSlider, Button
#
import pandas as pd

prenoms = pd.read_csv("C:/Users/clementlepadellec/Downloads/prenoms_france.csv", sep=";")
prenoms = prenoms[(prenoms["annais"].isin(["X", "XX", "XXX", "XXXX"]))==False] #Suupression de valeurs abérrantes
prenoms["annais"] =  prenoms["annais"].astype("int64")
liste_annees = pd.DataFrame(prenoms["annais"].value_counts().index)
liste_annees.columns = ["annais"]
#Fonction qui cree et retourne la figure
def create_figure(): 
	#Préparation des données
	annee = slider1.value
	selectedPrenom = text_prenom.value
	
	df = prenoms.loc[(prenoms['annais']>=annee[0])&(prenoms['annais']<=annee[1]) &(prenoms['preusuel'] == selectedPrenom.upper()),:]
	grouped = df.groupby('annais').aggregate({'nombre':'sum'}).reset_index().sort_index(ascending=False)
	
	grouped = pd.merge(liste_annees, grouped, on="annais", how="left").fillna(0).sort_values("annais")
	grouped['annais'] = grouped['annais'].astype('str')#Change type of annais
	data_source2 = ColumnDataSource(grouped)
	
	fig = figure( title="Evolution de la popularité d'un prenoms en France ( PRENOM:"+selectedPrenom.upper()+", ANNEES: ["+str(annee[0])+'-'+str(annee[1])+']')

	
	fig.line('annais', 'nombre', line_width=5, source=data_source2)
	#fig.circle('annee', 'nombre', legend="x puissance", line_width=2, fill_color="white", line_color="blue", size=5, source=data_source)
	
	fig.plot_width = 900
	# Effet du survol
	tooltips = [
			
			('annee', '@annais'),
			('prénom', selectedPrenom),
			('occurrences', '@nombre'),
		   ]
	fig.add_tools(HoverTool(tooltips=tooltips))

	
	# we will specify just "blue" for the color
	#wordcloud = WordCloud2(source=test1,wordCol="names",sizeCol="weights",colors="blue")
	#show(wordcloud)
	return fig

def update_data():
    panel.children[1] = create_figure()

#Définition des composants et association d'écouteurs
slider1 = RangeSlider(title="annee", value=(2000,2005), start=1920, end=2015)
#slider1.on_change('value', update_data)

text_prenom = TextInput(title="PRENOM", value="nicolas")

button_update = Button(label="Curve", button_type="success")
button_update.on_click(update_data)

#Affichage
panel = row(column(slider1, text_prenom, button_update), create_figure(), width=800) #On rassemble le slider, le muti-select et la figure
curdoc().add_root(panel) 
