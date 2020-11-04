# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 11:06:46 2020

@author: clementlepadellec
"""

####################################################################
#                    PACKAGES IMPORT                               #########
####################################################################

#pip install dash==1.17.0 pour installer dash
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
#Surement d'autres packages plotly à importer
import pandas as pd
import numpy as np


####################################################################
#                          FONCTIONS                               #########
####################################################################
#Brouillon, à organiser en modules.py plus tard

#1#fonction d'import CSV, avec header et ',' comme séparateur
def import_csv(file_path):
    df = pd.read_csv('file_path',sep=',',header=True)
    return df

#2#fonction de test de la variable cible
def test_var_cible(df,var_cible,var_type):
    if (np.issubdtype(df[var_cible].dtype, np.number)==True):
        var_type='Numeric'
    else:
        var_type='String'
    return var_type

#3#fonction de pré-visualisation du df
def print_df(df, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in df.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(df.iloc[i][col]) for col in df.columns
            ]) for i in range(min(len(df), max_rows))
        ])
    ])

#fonction de description des données (rajouter graph ?)
def resume_df(df):
    print(df.dtypes)
    print(df.describe())
    
    
####################################################################
#                          BUILD APP                               #########
####################################################################

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# Create app layout

app.layout = html.Div(children=[
    html.Div(
        [
            html.H3(
                "Interface d’analyse de données",
                style={"margin-bottom": "0px", "text-align":"center"}
                ),
            html.H5(
                "Clément Le Padellec - Adrien Pavoine - Amélie Picard", style={"margin-top": "0px", "text-align":"center"}
                )
            ]
        ),
    html.A(
        html.Button("Learn More About Our Project", id="learn-more-button",style={"margin-bottom": "0px", "text-align":"center"}),
        href="https://drive.google.com/drive/folders/1qPSh1zW8bdjgdiC5Bz5O2bpEjUCdQ_Ig",
        
        ),
    html.H4(children='US Agriculture Exports (2011)',style={
        'textAlign': 'center',}),
    html.Label('Chemin du fichier'),
    dcc.Input(id='path',value='', type='text'),
    path=Input('path', 'value'),
    df=import_csv(path),
    print_df(df)
    ]
)


####################################################################
#                          RUN APP                                 #########
####################################################################

if __name__ == '__main__':
    app.run_server(debug=True)
    