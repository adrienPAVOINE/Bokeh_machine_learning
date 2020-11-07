# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 15:36:34 2020

@author: clementlepadellec
"""

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
#update function
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n) for c, n in
            zip(list_of_contents, list_of_names)]
        return children
    
    
    
html.Div(
    className="analyse",
    children=
    [
        
        html.H3(
            "Choix de la variable cible",
            style={"margin-bottom": "0px", "text-align":"center"}
            ),
        dcc.Dropdown(
            id="input-target",
            options=[df.columns]
            ),
        html.Div(id='output-target'),

        ]
    ),

def _update_legend_gene(t):
    return 'You have selected "{}" target'.format(t)