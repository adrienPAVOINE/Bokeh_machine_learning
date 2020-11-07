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
from dash.dependencies import Input, Output, State
import base64
import dash_table
import io

####################################################################
#                          FONCTIONS                               #########
####################################################################
#Brouillon, à organiser en modules.py plus tard

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


#app initialization
app = dash.Dash(__name__)
# Create app layout

app.layout = html.Div(children=[
    html.Div(
        [
            # Banner display
            html.Div(
                className="header-title",
                children=[
                    html.H2(
                        id="title",
                        children="Interface d’analyse de données",
                    ),
                    html.Div(
                        id="learn_more",
                        children=[
                            html.Img(className="logo", src=app.get_asset_url("logo.png"))
                        ],
                    ),
                    html.Br(),                    
                    html.H6(
                        id="authors",
                        children="Clément Le Padellec - Adrien Pavoine - Amélie Picard"
                    ),
                    html.Br(),
                    #button link to DRIVE with instructions
                    html.A(
                        html.Button("Learn More About Our Project", id="learn-more-button",style={'width': '100%', "text-align":"center"}),
                        href="https://drive.google.com/drive/folders/1qPSh1zW8bdjgdiC5Bz5O2bpEjUCdQ_Ig",
                    ),

                ],
            ),
  
            ]
        ),
    html.Br(),
    html.Hr(),
    html.Hr(),
    html.Br(),
    html.Div(
        [
            
            html.H3(
                "Import des données",
                style={"margin-bottom": "0px", "text-align":"center"}
                ),            


            #upload csv file with comma as separator
            dcc.Upload(
                #define id (for update)
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                },
                # Allow multiple files to be uploaded (remove ?)
                multiple=True
            ),
            html.Div(id='output-data-upload'),

            ]
        )
    ]
)

####################################################################
#                         UPDATE FONCTIONS                         #########
####################################################################

#1# Callback and Update for import and vizualisation

# import & print data
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    # Assume that the user uploaded a CSV file with comma as sep
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')),sep=',')
    return html.Div([
        html.H6("Prévisualisation des données",style={"text-align":"center"}),
        dash_table.DataTable(
            data=df[0:10].to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns]
        ),
        html.Hr(),  # horizontal line
    ])

#callback for update
@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])

#update function
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n) for c, n in
            zip(list_of_contents, list_of_names)]
        return children


####################################################################
#                          RUN APP                                 #########
####################################################################

if __name__ == '__main__':
    app.run_server(debug=True)
    