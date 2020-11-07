# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 18:11:58 2020

@author: clementlepadellec
"""
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
from bokeh.models import HoverTool, Div,Panel,Tabs
from bokeh.models.widgets import MultiSelect, Select, RangeSlider, Button
#
import pandas as pd
from bokeh.io import output_file, output_notebook

title = Div(text='<h1 style="text-align: center">Example Header</h1>')
p1 = figure(plot_width=300, plot_height=300)
p1.circle([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], size=20, color="navy", alpha=0.5)
tab1 = Panel(child=p1, title="circle")

p2 = figure(plot_width=300, plot_height=300)
p2.line([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], line_width=3, color="navy", alpha=0.5)
tab2 = Panel(child=p2, title="line")

tabs = Tabs(tabs=[tab1, tab2])

layout = column(title, tabs, sizing_mode='scale_width')

curdoc().add_root(layout)