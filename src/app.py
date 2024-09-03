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
import os

from flask import Flask

# Initialize the app and server
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Load the classes from the text file
def load_classes_from_file(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

# Path to the classes file - adjust this to point to the correct location
base_dir = os.path.dirname(os.path.dirname(__file__))  # This will give you the directory above `src`
classes_file_path = os.path.join(base_dir, 'classes.txt')
label_options = load_classes_from_file(classes_file_path)

# Initialise pygame mixer
pygame.mixer.init()

# Import callbacks and layout
from components.layout import create_layout
from components.callbacks import register_callbacks

# Configure dash-uploader
UPLOAD_FOLDER_ROOT = "uploads"
du.configure_upload(app, UPLOAD_FOLDER_ROOT)

# Set the layout
app.layout = create_layout(app, label_options)

# Register callbacks
register_callbacks(app, label_options)

if __name__ == '__main__':
    app.run_server(debug=True)
