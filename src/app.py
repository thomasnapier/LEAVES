"""
Program Name: LEAVES (Large-scale Ecoacoustics Annotation and Visualisation with Efficient Segmentation)
Developer: Thomas Napier
Organisation: James Cook University
GitHub: https://github.com/thomasnapier/LEAVES
Description: This is a program designed to improve the audio labelling efficiency of
samples derived from the Australian Acoustic Observatory (A2O)
"""

import dash
import dash_bootstrap_components as dbc
import dash_uploader as du
from dash import html
import pygame

from flask import Flask
from components.layout import create_layout
from components.callbacks import register_callbacks

# Initialize the app and server
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Initialise pygame mixer
pygame.mixer.init()

# Configure dash-uploader
UPLOAD_FOLDER_ROOT = "uploads"
du.configure_upload(app, UPLOAD_FOLDER_ROOT)

# Set the layout
app.layout = create_layout(app)

# Register callbacks
register_callbacks(app)

if __name__ == '__main__':
    app.run_server(debug=True)
