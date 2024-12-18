"""
Program Name: LEAVES (Large-scale Ecoacoustics Annotation and Visualisation with Efficient Segmentation)
Developer: Thomas Napier
Organisation: James Cook University
GitHub: https://github.com/thomasnapier/LEAVES
Description: This is a program designed to improve the audio labelling efficiency of
samples derived from the Australian Acoustic Observatory (A2O)
"""

#Imports
from re import T
#from tkinter.tix import Tree
import click
import pandas as pd
from py import process
import plotly.graph_objs as go
import dash
from dash import dcc, html, State
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash import no_update
import plotly
import pygame
import random
import json
from itertools import cycle
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import glob
import matplotlib
matplotlib.use('Agg')
from dash.exceptions import PreventUpdate
from dash import dcc
import os
from pydub import AudioSegment
import zipfile
from flask import Flask, send_from_directory
import shutil
import tempfile
import librosa
import charset_normalizer
from sklearn.preprocessing import MinMaxScaler
import umap
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
import numpy as np
import tempfile
import dash_uploader as du
import sys


# Determine the base path of the application
def get_base_path():
    if getattr(sys, 'frozen', False):  # If bundled as an executable
        return sys._MEIPASS
    else:  # If running in development mode
        return os.path.dirname(os.path.abspath(__file__))
    
BASE_DIR = get_base_path()
UPLOADS_DIRECTORY = os.path.join(BASE_DIR, "uploads")
DATA_DIRECTORY = os.path.join(BASE_DIR, "data")
ASSETS_DIRECTORY = os.path.join(BASE_DIR, "assets")
LOGOS_DIRECTORY = os.path.join(ASSETS_DIRECTORY, "logos")

# Initialise pygame mixer
pygame.mixer.init()

# Load data
df = pd.read_csv(os.path.join(DATA_DIRECTORY, 'Undara-DryB.csv'))
label_options = ["Background Silence", "Birds", "Frogs", "Human Speech", "Insects", "Mammals",
                 "Misc/Uncertain", "Rain (Heavy)", "Rain (Light)", "Vehicles (Aircraft/Cars)",
                 "Wind (Strong)", "Wind (Light)"]
file_options = [{'label': os.path.basename(file), 'value': file} for file in glob.glob(os.path.join(DATA_DIRECTORY, "*.csv"))]
current_cluster_index = 0
current_csv_file = os.path.join(DATA_DIRECTORY, 'Undara-DryB.csv')
sampled_point_index = 0
samples_json = None
uploaded_files = []

# Temporary folder for server-side storage
TEMP_FOLDER = tempfile.mkdtemp()

for label_option in label_options:
    if label_option.lower().replace(' ', '_') not in df.columns:
        df[label_option.lower().replace(' ', '_')] = 0

# Initialize the DataFrame
columns = ['embedding_x', 'embedding_y', 'embedding_z', 'sound_path']
columns.extend([label.lower().replace(' ', '_') for label in label_options])
annotations_df = pd.DataFrame(columns=columns)

def save_annotations_to_csv(df, filepath):
    filepath = os.path.join(DATA_DIRECTORY, filepath)  # Ensure file is saved in the data directory
    df.to_csv(filepath, index=False)

def create_figure(df, current_cluster_index):
    classes = df['class'].unique()
    colors = plt.cm.get_cmap('tab10', len(classes))
    fig = go.Figure()
    for i, c in enumerate(classes):
        dfi = df[df['class'] == c]
        marker_size = 10 if i == current_cluster_index else 5  # Highlight current cluster
        fig.add_trace(go.Scatter3d(x=dfi['x'], y=dfi['y'], z=dfi['z'],
                                mode='markers',
                                name="Cluster " + str(c),
                                customdata=dfi['sound_path'],
                                marker=dict(size=marker_size, line=dict(width=0.1, color='black')),
                                marker_color=[f'rgba{colors(i)}' for _ in range(len(dfi))]))
        
    fig.update_layout(
        paper_bgcolor="var(--background-color)",
        legend_orientation='v',
        autosize=True,
            scene=dict(
                xaxis=dict(title='X', titlefont=dict(size=12), gridcolor='gray', gridwidth=1),
                yaxis=dict(title='Y', titlefont=dict(size=12), gridcolor='gray', gridwidth=1),
                zaxis=dict(title='Z', titlefont=dict(size=12), gridcolor='gray', gridwidth=1)
            ),
            margin=dict(l=0, r=0, b=0, t=40),
                legend=dict(
                x=0.7,
                y=0.5,
                font=dict(size=10),
                bgcolor='rgba(255,255,255,0.5)'
            )
        )
    
    return fig


# Define app and layout
fig = create_figure(df, current_cluster_index)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Load the logo image
logo_path = os.path.join(LOGOS_DIRECTORY, "logo.png")
encoded_logo = base64.b64encode(open(logo_path, 'rb').read()).decode('ascii')

# Initialize dash-uploader
UPLOAD_FOLDER_ROOT = "uploads"
du.configure_upload(app, UPLOAD_FOLDER_ROOT)

# Set default values
default_values = {
    'preprocessing-sds': True,
    'preprocessing-stw': True,
    'data-sampling': 20,
    'windowing': 4.5,
    'denoising-method': 'none',
    'feature-extraction-mmn': True,
    'feature-extraction-imd': False,
    'complexity-reduction': 'UMAP',
    'n-neighbours': 15,
    'min-dist': 0.1,
    'clustering-algorithm': 'DBScan',
    'eps': 0.5,
    'min-samples': 5,
    'tsne-perplexity': 30,
    'tsne-learning-rate': 200,
    'kmeans-k': 5,
    'agglomerative-distance': 10
}

app.layout = html.Div([
    dcc.Loading(
        id="loading-indicator",
        type="default",  # Or "circle", "dot", etc.
        children=[html.Div([
    dcc.Store(id='config-store', data=default_values),
    html.Div([
        html.Img(src=f'data:image/png;base64,{encoded_logo}', id='logo'),
        html.Span('LEAVES', id='top-banner-title')
    ], id='top-banner'),
    html.Button('⚙️', id='open-settings', style={'position': 'absolute', 'top': '10px', 'right': '10px'}),
    html.Button('🌙/☀️', id='theme-toggle', style={'position': 'absolute', 'top': '10px', 'right': '150px'}),
    html.Div([
        html.Div([
            html.H2("Software Configuration", style={'textAlign': 'center', 'color': 'var(--text-color)'}),
            html.Hr(style={'background-color': 'var(--text-color)'}),
            html.Div([
                html.H4("Preprocessing", className='section-header'),
                html.Div([
                    html.Label('Systematic Data Sampling', className='option-label'),
                    dcc.Checklist(
                        options=[{'label': '', 'value': 'preprocessing-sds'}],
                        value=['preprocessing-sds'] if default_values['preprocessing-sds'] else [],
                        id='preprocessing-sds',
                        className='toggle-switch'
                    )
                ], className='option-row'),
                html.Div([
                    html.Label('Short Term Windowing', className='option-label'),
                    dcc.Checklist(
                        options=[{'label': '', 'value': 'preprocessing-stw'}],
                        value=['preprocessing-stw'] if default_values['preprocessing-stw'] else [],
                        id='preprocessing-stw',
                        className='toggle-switch'
                    )
                ], className='option-row'),
                html.Div([
                    html.Label('Windowing Input (sec)', className='option-label', style={'marginLeft': '40px'}),
                    dcc.Slider(id='windowing-slider', min=1, max=10, step=0.5, value=default_values['windowing'], marks={i: str(i) for i in range(1, 11)}, className='option-component'),
                    dcc.Input(id='windowing-input', type='number', value=default_values['windowing'], step=0.5, min=1, max=10, className='option-component', style={'width': '50px'})
                ], className='option-row', style={'marginLeft': '20px'}),
                html.Div([
                    html.Label('Data Sampling (min)', className='option-label'),
                    dcc.Slider(id='data-sampling-slider', min=0, max=40, value=default_values['data-sampling'], step=1, marks={i: f'{i} min' for i in range(0, 41, 5)}, className='option-component'),
                    dcc.Input(id='data-sampling-input', type='number', value=default_values['data-sampling'], step=1, min=0, max=40, className='option-component', style={'width': '50px'})
                ], className='option-row'),
                html.Div([
                    html.Label('Denoising Method', className='option-label'),
                    dbc.ButtonGroup(
                        [
                            dbc.Button('None', id='denoising-none', color='secondary', className='toggle-button', active=default_values['denoising-method'] == 'none'),
                            dbc.Button('Wavelet-based', id='denoising-wavelet', color='secondary', className='toggle-button', active=default_values['denoising-method'] == 'wavelet'),
                            dbc.Button('Low-pass', id='denoising-lowpass', color='secondary', className='toggle-button', active=default_values['denoising-method'] == 'low-pass'),
                            dbc.Button('High-pass', id='denoising-highpass', color='secondary', className='toggle-button', active=default_values['denoising-method'] == 'high-pass'),
                            dbc.Button('Band-pass', id='denoising-bandpass', color='secondary', className='toggle-button', active=default_values['denoising-method'] == 'band-pass')
                        ], className='option-component', id='denoising-method'
                    )
                ], className='option-row'),
            ], className='section'),

            html.Hr(),
            html.Div([
                html.H4("Feature Extraction", className='section-header'),
                html.Div([
                    html.Label('Min-Max Normalisation', className='option-label'),
                    dcc.Checklist(
                        options=[{'label': '', 'value': 'feature-extraction-mmn'}],
                        value=['feature-extraction-mmn'] if default_values['feature-extraction-mmn'] else [],
                        id='feature-extraction-mmn',
                        className='toggle-switch'
                    )
                ], className='option-row'),
                html.Div([
                    html.Label('Include MFCC Derivatives', className='option-label'),
                    dcc.Checklist(
                        options=[{'label': '', 'value': 'feature-extraction-imd'}],
                        value=['feature-extraction-imd'] if default_values['feature-extraction-imd'] else [],
                        id='feature-extraction-imd',
                        className='toggle-switch'
                    )
                ], className='option-row'),
            ], className='section'),

            html.Hr(),
            html.Div([
                html.H4("Complexity Reduction", className='section-header'),
                html.Div([
                    html.Label('Reduction Method', className='option-label'),
                    dbc.ButtonGroup(
                        [
                            dbc.Button('UMAP', id='complexity-umap', color='secondary', className='toggle-button', active=default_values['complexity-reduction'] == 'UMAP'),
                            dbc.Button('t-SNE', id='complexity-tsne', color='secondary', className='toggle-button', active=default_values['complexity-reduction'] == 't-SNE'),
                            dbc.Button('PCA', id='complexity-pca', color='secondary', className='toggle-button', active=default_values['complexity-reduction'] == 'PCA')
                        ], className='option-component', id='complexity-reduction'
                    )
                ], className='option-row'),
                html.Div([
                    html.Div([
                        html.Label('N Neighbours', className='option-label'),
                        dcc.Slider(
                            id='umap-neighbors-slider',
                            min=5,
                            max=50,
                            value=default_values.get('n-neighbours', 15),
                            step=1,
                            marks={i: str(i) for i in range(5, 51, 5)},
                            className='option-component'
                        ),
                        dcc.Input(
                            id='umap-neighbors-input',
                            type='number',
                            value=default_values.get('n-neighbours', 15),
                            step=1,
                            min=5,
                            max=50,
                            className='option-component',
                            style={'width': '50px'}
                        )
                    ], id='umap-options-neighbors'),

                    html.Div([
                        html.Label('Min Distance', className='option-label'),
                        dcc.Slider(
                            id='umap-min-dist-slider',
                            min=0.0,
                            max=1.0,
                            value=default_values.get('min-dist', 0.1),
                            step=0.01,
                            marks={i / 10: str(i / 10) for i in range(0, 11)},
                            className='option-component'
                        ),
                        dcc.Input(
                            id='umap-min-dist-input',
                            type='number',
                            value=default_values.get('min-dist', 0.1),
                            step=0.01,
                            min=0.0,
                            max=1.0,
                            className='option-component',
                            style={'width': '50px'}
                        )
                    ], id='umap-options-min-dist')
                ]),
                html.Div([
                html.Label('Perplexity', className='option-label'),
                dcc.Slider(
                    id='tsne-perplexity-slider',
                    min=5,
                    max=50,
                    value=default_values.get('tsne-perplexity', 30),
                    step=1,
                    marks={i: str(i) for i in range(5, 51, 5)},
                    className='option-component'
                ),
                dcc.Input(
                    id='tsne-perplexity-input',
                    type='number',
                    value=default_values.get('tsne-perplexity', 30),
                    step=1,
                    min=5,
                    max=50,
                    className='option-component',
                    style={'width': '50px'}
                )
            ], className='option-row', id='tsne-options-perplexity', style={'display': 'none'}),
            html.Div([
                html.Label('Learning Rate', className='option-label'),
                dcc.Slider(
                    id='tsne-learning-rate-slider',
                    min=10,
                    max=1000,
                    value=default_values.get('tsne-learning-rate', 200),
                    step=10,
                    marks={i: str(i) for i in range(10, 1001, 100)},
                    className='option-component'
                ),
                dcc.Input(
                    id='tsne-learning-rate-input',
                    type='number',
                    value=default_values.get('tsne-learning-rate', 200),
                    step=10,
                    min=10,
                    max=1000,
                    className='option-component',
                    style={'width': '50px'}
                )
            ], className='option-row', id='tsne-options-learning-rate', style={'display': 'none'}),
            html.Div([
            html.Label('Number of Components', className='option-label'),
            dcc.Slider(
                id='pca-components-slider',
                min=1,
                max=100,
                value=default_values.get('pca-components', 2),
                step=1,
                marks={i: str(i) for i in range(1, 101, 10)},
                className='option-component'
            ),
            dcc.Input(
                id='pca-components-input',
                type='number',
                value=default_values.get('pca-components', 2),
                step=1,
                min=1,
                max=100,
                className='option-component',
                style={'width': '50px'}
            )
        ], className='option-row', id='pca-options', style={'display': 'none'}),
            ], className='section'),

            html.Hr(),
            html.Div([
                html.H4("Clustering", className='section-header'),
                html.Div([
                    html.Label('Clustering Algorithm', className='option-label'),
                    dbc.ButtonGroup(
                        [
                            dbc.Button('DBScan', id='clustering-dbscan', color='secondary', className='toggle-button', active=default_values['clustering-algorithm'] == 'DBScan'),
                            dbc.Button('k-means', id='clustering-kmeans', color='secondary', className='toggle-button', active=default_values['clustering-algorithm'] == 'k-means'),
                            dbc.Button('Agglomerative', id='clustering-agglomerative', color='secondary', className='toggle-button', active=default_values['clustering-algorithm'] == 'Agglomerative')
                        ], className='option-component', id='clustering-algorithm'
                    )
                ], className='option-row'),
                html.Div([
                    html.Div([
                        html.Label('Eps', className='option-label'),
                        dcc.Slider(
                            id='dbscan-eps-slider',
                            min=0.0,
                            max=5.0,
                            value=default_values.get('eps', 0.5),
                            step=0.1,
                            marks={i / 10: str(i / 10) for i in range(0, 51, 5)},
                            className='option-component'
                        ),
                        dcc.Input(
                            id='dbscan-eps-input',
                            type='number',
                            value=default_values.get('eps', 0.5),
                            step=0.1,
                            min=0.0,
                            max=5.0,
                            className='option-component',
                            style={'width': '50px'}
                        )
                    ], id='dbscan-options-eps'),

                    html.Div([
                        html.Label('Min Samples', className='option-label'),
                        dcc.Slider(
                            id='dbscan-min-samples-slider',
                            min=1,
                            max=100,
                            value=default_values.get('min-samples', 5),
                            step=1,
                            marks={i: str(i) for i in range(1, 101, 10)},
                            className='option-component'
                        ),
                        dcc.Input(
                            id='dbscan-min-samples-input',
                            type='number',
                            value=default_values.get('min-samples', 5),
                            step=1,
                            min=1,
                            max=100,
                            className='option-component',
                            style={'width': '50px'}
                        )
                    ], id='dbscan-options-min-samples')
                ]),
                html.Div([
                html.Label('Number of Clusters (k)', className='option-label'),
                dcc.Slider(
                    id='kmeans-k-slider',
                    min=1,
                    max=100,
                    value=default_values.get('kmeans-k', 3),
                    step=1,
                    marks={i: str(i) for i in range(1, 101, 10)},
                    className='option-component'
                ),
                dcc.Input(
                    id='kmeans-k-input',
                    type='number',
                    value=default_values.get('kmeans-k', 3),
                    step=1,
                    min=1,
                    max=100,
                    className='option-component',
                    style={'width': '50px'}
                )
            ], className='option-row', id='kmeans-options', style={'display': 'none'}),
            html.Div([
            html.Label('Linkage Type', className='option-label'),
            dcc.Dropdown(
                id='agglomerative-linkage-dropdown',
                options=[
                    {'label': 'Ward', 'value': 'ward'},
                    {'label': 'Complete', 'value': 'complete'},
                    {'label': 'Average', 'value': 'average'},
                    {'label': 'Single', 'value': 'single'}
                ],
                value='ward',
                className='option-component'
            )
        ], className='option-row', id='agglomerative-options-linkage', style={'display': 'none', 'color':'black'}),
        html.Div([
            html.Label('Distance Threshold', className='option-label'),
            dcc.Slider(
                id='agglomerative-distance-slider',
                min=0,
                max=100,
                value=default_values.get('agglomerative-distance', 10),
                step=1,
                marks={i: str(i) for i in range(0, 101, 10)},
                className='option-component'
            ),
            dcc.Input(
                id='agglomerative-distance-input',
                type='number',
                value=default_values.get('agglomerative-distance', 10),
                step=1,
                min=0,
                max=100,
                className='option-component',
                style={'width': '50px'}
            )
        ], className='option-row', id='agglomerative-options-distance', style={'display': 'none', 'color': 'black'}),
            ], className='section'),

            html.Button('Done', id='close-settings', style={'margin': '20px'}),
        ], style={'padding': '20px', 'color': 'var(--text-color)', 'background-color': 'var(--background-color)', 'max-height': '80vh', 'overflow-y': 'auto'}),
    ], id='settings-modal', style={'display': 'none', 'position': 'fixed', 'z-index': '1000', 'left': '25%', 'top': '5%', 'width': '50%', 'background-color': 'var(--background-color)', 'border': '2px solid var(--border-color)', 'border-radius': '5px', 'color': 'var(--text-color)', 'max-height': '80vh', 'overflow-y': 'auto'}),
    html.Div([
        html.Div([
            # Dash-Uploader component
            du.Upload(
            id='uploader',
            text='Drag and Drop or Select Files',
            text_completed='Upload Complete: ',
            pause_button=True,
            cancel_button=True,
            max_file_size=1800,  # 1.8 GB
            max_files = 100
        ),
        html.Div(id='upload-status', style={'color': 'var(--text-color)', 'display': 'none'}),
            html.Button('Process Uploaded Data', id='process-data-button', style={'margin-top': '10px', 'margin-bottom': '10px'}),
            dcc.Dropdown(id='file-dropdown', options=file_options, value=file_options[0]['value'], style={'color': 'black'}),
            html.Div(dcc.Graph(id='scatter-plot', figure=fig, style={'height': '100%', 'width': '100%'}), id='scatterplot-container'),
            html.Div([
                html.Div(id='checklist-title', children='Classes:'),
                html.Div([
                    dcc.Checklist(id='class-labels-checklist',
                                  options=[{'label': label, 'value': label} for label in label_options],
                                  value=[],
                                  labelStyle={'display': 'inline-block', 'margin-right': '10px', 'color': 'var(--text-color)'}
                    )
                ], id='annotation-tags-container')
            ], id='checklist-container'),
            html.Div([
                html.Button('Save Annotations', id='control-button'),
                html.A('Download CSV', id='csv-download-link', href='', download='annotations.csv')
            ]),
        ], id='left-column'),
        html.Div([
            html.Img(id='mel-spectrogram', src=''),
            html.Img(id='waveform-plot', src=''),
        ], id='main-window'),
    ], id='main-horizontal-layout'),
    html.Div([
        html.Div(id='audio-status', children='No audio being played currently.', style={'color': 'var(--text-color)'}),
        html.Div([
            html.Button('|◁◁', id='previous-cluster'),  # New button for previous cluster
            html.Button('◁', id='previous-point'),
            html.Button('||', id='play-audio'),
            html.Button('▷', id='next-point'),
            html.Button('▷▷|', id='next-cluster'),  # New button for next cluster
        ], id='button-group'),
    ], id='bottom-timeline'),
    html.Div(id='hidden-sample-data'),
    html.Div(id='csv-dummy-output'),
    html.Div(id='csv-test'),
    html.Div(id='temporary-storage', style={'display': 'none'}),
    dcc.Store(id='processed-data-store')
], id='main-container'),
html.Div(id="app-content"),
        ],
        fullscreen=False  # Keep this false to allow positioning
    ),
    dbc.Progress(id="progress-bar", value=0, striped=True, animated=True,
                 style={"position": "fixed", "bottom": "0", "width": "100%"})
])


# Callbacks
@app.callback(
    Output('settings-modal', 'style'),
    [Input('open-settings', 'n_clicks'), Input('close-settings', 'n_clicks')],
    [State('settings-modal', 'style')]
)
def toggle_modal(open_clicks, close_clicks, style):
    ctx = dash.callback_context

    if not ctx.triggered:
        raise PreventUpdate

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'open-settings' and open_clicks:
        return {'display': 'block', 'position': 'fixed', 'z-index': '1000', 'left': '25%', 'top': '10%', 'width': '50%', 'background-color': 'var(--background-color)', 'border': '2px solid var(--border-color)', 'border-radius': '5px', 'color': 'var(--text-color)', 'max-height': '80vh'}

    elif button_id == 'close-settings' and close_clicks:
        return {'display': 'none'}

    return style

# Callback for toggling active button in button group
@app.callback(
    [Output('denoising-none', 'active'),
     Output('denoising-wavelet', 'active'),
     Output('denoising-lowpass', 'active'),
     Output('denoising-highpass', 'active'),
     Output('denoising-bandpass', 'active')],
    [Input('denoising-none', 'n_clicks'),
     Input('denoising-wavelet', 'n_clicks'),
     Input('denoising-lowpass', 'n_clicks'),
     Input('denoising-highpass', 'n_clicks'),
     Input('denoising-bandpass', 'n_clicks')]
)
def toggle_denoising_method(n_none, n_wavelet, n_lowpass, n_highpass, n_bandpass):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    return [
        button_id == 'denoising-none',
        button_id == 'denoising-wavelet',
        button_id == 'denoising-lowpass',
        button_id == 'denoising-highpass',
        button_id == 'denoising-bandpass'
    ]

# Callback for toggling active button in complexity reduction
@app.callback(
    [Output('complexity-umap', 'active'),
     Output('complexity-tsne', 'active'),
     Output('complexity-pca', 'active')],
    [Input('complexity-umap', 'n_clicks'),
     Input('complexity-tsne', 'n_clicks'),
     Input('complexity-pca', 'n_clicks')]
)
def toggle_complexity_method(n_umap, n_tsne, n_pca):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    return [
        button_id == 'complexity-umap',
        button_id == 'complexity-tsne',
        button_id == 'complexity-pca'
    ]

# Callback for toggling active button in clustering algorithm
@app.callback(
    [Output('clustering-dbscan', 'active'),
     Output('clustering-kmeans', 'active'),
     Output('clustering-agglomerative', 'active')],
    [Input('clustering-dbscan', 'n_clicks'),
     Input('clustering-kmeans', 'n_clicks'),
     Input('clustering-agglomerative', 'n_clicks')]
)
def toggle_clustering_algorithm(n_dbscan, n_kmeans, n_agglomerative):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    return [
        button_id == 'clustering-dbscan',
        button_id == 'clustering-kmeans',
        button_id == 'clustering-agglomerative'
    ]

@app.callback(
    [
        Output('umap-options-neighbors', 'style'),
        Output('umap-options-min-dist', 'style'),
        Output('dbscan-options-eps', 'style'),
        Output('dbscan-options-min-samples', 'style'),
        Output('tsne-options-perplexity', 'style'),
        Output('tsne-options-learning-rate', 'style'),
        Output('pca-options', 'style'),
        Output('kmeans-options', 'style'),
        Output('agglomerative-options-linkage', 'style'),
        Output('agglomerative-options-distance', 'style')
    ],
    [
        Input('complexity-umap', 'active'),
        Input('clustering-dbscan', 'active'),
        Input('complexity-tsne', 'active'),
        Input('complexity-pca', 'active'),
        Input('clustering-kmeans', 'active'),
        Input('clustering-agglomerative', 'active')
    ]
)
def toggle_hyperparameters(
    umap_active, dbscan_active, tsne_active, pca_active, kmeans_active, agglomerative_active
):
    hidden_style = {'display': 'none'}
    visible_style = {'display': 'block'}

    # UMAP
    umap_neighbors_style = visible_style if umap_active else hidden_style
    umap_min_dist_style = visible_style if umap_active else hidden_style

    # DBSCAN
    dbscan_eps_style = visible_style if dbscan_active else hidden_style
    dbscan_min_samples_style = visible_style if dbscan_active else hidden_style

    # Other Algorithms
    tsne_perplexity_style = visible_style if tsne_active else hidden_style
    tsne_learning_rate_style = visible_style if tsne_active else hidden_style
    pca_style = visible_style if pca_active else hidden_style
    kmeans_style = visible_style if kmeans_active else hidden_style
    agglomerative_linkage_style = visible_style if agglomerative_active else hidden_style
    agglomerative_distance_style = visible_style if agglomerative_active else hidden_style

    return (
        umap_neighbors_style,
        umap_min_dist_style,
        dbscan_eps_style,
        dbscan_min_samples_style,
        tsne_perplexity_style,
        tsne_learning_rate_style,
        pca_style,
        kmeans_style,
        agglomerative_linkage_style,
        agglomerative_distance_style
    )


@app.callback(
    [
        Output('windowing-slider', 'value'),
        Output('windowing-input', 'value'),
        Output('data-sampling-slider', 'value'),
        Output('data-sampling-input', 'value'),
        Output('umap-neighbors-slider', 'value'),
        Output('umap-neighbors-input', 'value'),
        Output('umap-min-dist-slider', 'value'),
        Output('umap-min-dist-input', 'value'),
        Output('tsne-perplexity-slider', 'value'),
        Output('tsne-perplexity-input', 'value'),
        Output('tsne-learning-rate-slider', 'value'),
        Output('tsne-learning-rate-input', 'value'),
        Output('dbscan-eps-slider', 'value'),
        Output('dbscan-eps-input', 'value'),
        Output('dbscan-min-samples-slider', 'value'),
        Output('dbscan-min-samples-input', 'value'),
        Output('kmeans-k-slider', 'value'),
        Output('kmeans-k-input', 'value'),
        Output('agglomerative-distance-slider', 'value'),
        Output('agglomerative-distance-input', 'value'),
        Output('config-store', 'data')
    ],
    [
        Input('windowing-slider', 'value'),
        Input('windowing-input', 'value'),
        Input('data-sampling-slider', 'value'),
        Input('data-sampling-input', 'value'),
        Input('umap-neighbors-slider', 'value'),
        Input('umap-neighbors-input', 'value'),
        Input('umap-min-dist-slider', 'value'),
        Input('umap-min-dist-input', 'value'),
        Input('tsne-perplexity-slider', 'value'),
        Input('tsne-perplexity-input', 'value'),
        Input('tsne-learning-rate-slider', 'value'),
        Input('tsne-learning-rate-input', 'value'),
        Input('dbscan-eps-slider', 'value'),
        Input('dbscan-eps-input', 'value'),
        Input('dbscan-min-samples-slider', 'value'),
        Input('dbscan-min-samples-input', 'value'),
        Input('kmeans-k-slider', 'value'),
        Input('kmeans-k-input', 'value'),
        Input('agglomerative-distance-slider', 'value'),
        Input('agglomerative-distance-input', 'value'),
        Input('config-store', 'data')
    ]
)
def handle_settings(windowing_slider, windowing_input,
                    data_sampling_slider, data_sampling_input,
                    umap_neighbors_slider, umap_neighbors_input,
                    umap_min_dist_slider, umap_min_dist_input,
                    tsne_perplexity_slider, tsne_perplexity_input,
                    tsne_learning_rate_slider, tsne_learning_rate_input,
                    dbscan_eps_slider, dbscan_eps_input,
                    dbscan_min_samples_slider, dbscan_min_samples_input,
                    kmeans_k_slider, kmeans_k_input,
                    agglomerative_distance_slider, agglomerative_distance_input,
                    config_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        # On load: Populate all sliders/inputs from config_store
        return (
            config_data.get('windowing', 1.0), config_data.get('windowing', 1.0),
            config_data.get('data-sampling', 10.0), config_data.get('data-sampling', 10.0),
            config_data.get('n-neighbours', 15), config_data.get('n-neighbours', 15),
            config_data.get('min-dist', 0.1), config_data.get('min-dist', 0.1),
            config_data.get('tsne-perplexity', 30), config_data.get('tsne-perplexity', 30),
            config_data.get('tsne-learning-rate', 200), config_data.get('tsne-learning-rate', 200),
            config_data.get('eps', 0.5), config_data.get('eps', 0.5),
            config_data.get('min-samples', 5), config_data.get('min-samples', 5),
            config_data.get('kmeans-k', 3), config_data.get('kmeans-k', 3),
            config_data.get('agglomerative-distance', 10), config_data.get('agglomerative-distance', 10),
            config_data
        )

    # Sync triggered input with config_store
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    input_to_config_key = {
        'windowing-slider': ('windowing', windowing_slider),
        'windowing-input': ('windowing', windowing_input),
        'data-sampling-slider': ('data-sampling', data_sampling_slider),
        'data-sampling-input': ('data-sampling', data_sampling_input),
        'umap-neighbors-slider': ('n-neighbours', umap_neighbors_slider),
        'umap-neighbors-input': ('n-neighbours', umap_neighbors_input),
        'umap-min-dist-slider': ('min-dist', umap_min_dist_slider),
        'umap-min-dist-input': ('min-dist', umap_min_dist_input),
        'tsne-perplexity-slider': ('tsne-perplexity', tsne_perplexity_slider),
        'tsne-perplexity-input': ('tsne-perplexity', tsne_perplexity_input),
        'tsne-learning-rate-slider': ('tsne-learning-rate', tsne_learning_rate_slider),
        'tsne-learning-rate-input': ('tsne-learning-rate', tsne_learning_rate_input),
        'dbscan-eps-slider': ('eps', dbscan_eps_slider),
        'dbscan-eps-input': ('eps', dbscan_eps_input),
        'dbscan-min-samples-slider': ('min-samples', dbscan_min_samples_slider),
        'dbscan-min-samples-input': ('min-samples', dbscan_min_samples_input),
        'kmeans-k-slider': ('kmeans-k', kmeans_k_slider),
        'kmeans-k-input': ('kmeans-k', kmeans_k_input),
        'agglomerative-distance-slider': ('agglomerative-distance', agglomerative_distance_slider),
        'agglomerative-distance-input': ('agglomerative-distance', agglomerative_distance_input)
    }

    if triggered_id in input_to_config_key:
        config_key, value = input_to_config_key[triggered_id]
        config_data[config_key] = value

    # Update values for all sliders/inputs
    return (
        config_data['windowing'], config_data['windowing'],
        config_data['data-sampling'], config_data['data-sampling'],
        config_data['n-neighbours'], config_data['n-neighbours'],
        config_data['min-dist'], config_data['min-dist'],
        config_data['tsne-perplexity'], config_data['tsne-perplexity'],
        config_data['tsne-learning-rate'], config_data['tsne-learning-rate'],
        config_data['eps'], config_data['eps'],
        config_data['min-samples'], config_data['min-samples'],
        config_data['kmeans-k'], config_data['kmeans-k'],
        config_data['agglomerative-distance'], config_data['agglomerative-distance'],
        config_data
    )


# Callback for toggle switches
# @app.callback(
#     [Output('preprocessing-sds', 'value'),
#      Output('preprocessing-stw', 'value'),
#      Output('feature-extraction-mmn', 'value'),
#      Output('feature-extraction-imd', 'value')],
#     [Input('preprocessing-sds', 'value'),
#      Input('preprocessing-stw', 'value'),
#      Input('feature-extraction-mmn', 'value'),
#      Input('feature-extraction-imd', 'value')]
# )
# def sync_toggles(sds, stw, mmn, imd):
#     return sds, stw, mmn, imd

@app.callback(
    Output('class-labels-checklist', 'value'),
    [Input('play-audio', 'n_clicks'),
     Input('next-point', 'n_clicks'),
     Input('previous-point', 'n_clicks')],
    [State('hidden-sample-data', 'children')])
def update_checklist(play_clicks, next_clicks, prev_clicks, samples_json):
    if samples_json is None:
        return []
    
    samples = json.loads(samples_json)
    current_sample = samples["data"][samples["current_index"]]
    sound_file = current_sample['sound_path']
    
    # Extract the row corresponding to the current sound file
    row = df[df['sound_path'] == sound_file].iloc[0]
    
    # Determine which checkboxes should be checked based on the DataFrame
    checked_boxes = [col for col in label_options if row[col.lower().replace(' ', '_')] == 1]
    return checked_boxes

@app.callback(
    Output('csv-dummy-output', 'children'),
    [Input('control-button', 'n_clicks')],
    [State('class-labels-checklist', 'value'),
     State('hidden-sample-data', 'children')]
)
def update_annotations(n_clicks, selected_labels, samples_json):
    if n_clicks is None:
        raise PreventUpdate

    if samples_json is None:
        return dash.no_update
    
    samples = json.loads(samples_json)
    current_sample = samples["data"][samples["current_index"]]
    sound_path = current_sample['sound_path']

    row_index = annotations_df.index[annotations_df['sound_path'] == sound_path].tolist()
    if not row_index:
        row_index = len(annotations_df)
        annotations_df.loc[row_index] = [current_sample['x'], current_sample['y'], current_sample['z'], sound_path] + [0] * len(label_options)

    for label in label_options:
        column_name = label.lower().replace(' ', '_')
        annotations_df.loc[row_index, column_name] = 1 if label in selected_labels else 0

    return dash.no_update

@app.callback(
    Output('csv-download-link', 'href'),
    Input('control-button', 'n_clicks'),
    prevent_initial_call=True
)
def save_and_download_csv(n_clicks):
    if n_clicks is None:
        raise PreventUpdate

    filepath = 'annotations.csv'
    save_annotations_to_csv(annotations_df, filepath)
    return f'/download/{filepath}'

@app.server.route('/download/<path:filename>')
def serve_file(filename):
    file_path = os.path.join(DATA_DIRECTORY, filename)
    if not os.path.exists(file_path):
        return f"File not found: {filename}", 404
    return send_from_directory(DATA_DIRECTORY, filename, as_attachment=True)

@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('audio-status', 'children'),
     Output('hidden-sample-data', 'children')],
    [Input('play-audio', 'n_clicks'),
     Input('next-point', 'n_clicks'),
     Input('previous-point', 'n_clicks'),
     Input('next-cluster', 'n_clicks'),  # New input
     Input('previous-cluster', 'n_clicks'),  # New input
     Input('file-dropdown', 'value'),
     Input('scatter-plot', 'clickData'),
     Input('class-labels-checklist', 'value'),
     Input('csv-test', 'children')],
    State('hidden-sample-data', 'children'))
def process_audio(play_clicks, next_clicks, prev_clicks, next_cluster_clicks, prev_cluster_clicks, selected_file, clickData, selected_labels, csv_path, samples_json):
    global current_cluster_index, sampled_point_index, current_csv_file
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Initialize samples if necessary
    if samples_json is None or current_cluster_index >= len(df['class'].unique()):
        current_class = df['class'].unique()[current_cluster_index]
        sampled_points = df[df['class'] == current_class].sample(10).to_dict('records')
        samples_json = json.dumps({"data": sampled_points, "current_index": 0})

    samples = json.loads(samples_json)

    if button_id in ['next-cluster', 'previous-cluster']:
        if button_id == 'next-cluster':
            current_cluster_index += 1
        elif button_id == 'previous-cluster':
            current_cluster_index -= 1
        
        if current_cluster_index >= len(df['class'].unique()):
            current_cluster_index = 0  # Reset to the first cluster
        elif current_cluster_index < 0:
            current_cluster_index = len(df['class'].unique()) - 1  # Reset to the last cluster
            
        current_class = df['class'].unique()[current_cluster_index]
        sampled_points = df[df['class'] == current_class].sample(10).to_dict('records')
        samples_json = json.dumps({"data": sampled_points, "current_index": 0})
        samples = json.loads(samples_json)
        sampled_point_index = 0

    if button_id == 'csv-test' and csv_path:
        new_df = pd.read_csv(csv_path)
        new_fig = create_figure(new_df, current_cluster_index)
        new_samples_json = json.dumps({"data": new_df.to_dict('records'), "current_index": 0})
        return new_fig, "Scatter plot updated from new CSV file.", new_samples_json

    elif button_id == 'next-point':
        sampled_point_index = samples.get('current_index', 0) + 1
        if sampled_point_index >= 10:
            current_cluster_index += 1
            if current_cluster_index >= len(df['class'].unique()):
                current_cluster_index = 0
            current_class = df['class'].unique()[current_cluster_index]
            sampled_points = df[df['class'] == current_class].sample(10).to_dict('records')
            samples_json = json.dumps({"data": sampled_points, "current_index": 0})
            samples = json.loads(samples_json)
            sampled_point_index = 0

    elif button_id == 'previous-point':
        sampled_point_index = samples.get('current_index', 0) - 1
        if sampled_point_index < 0:
            current_cluster_index -= 1
            if current_cluster_index < 0:
                current_cluster_index = len(df['class'].unique()) - 1
            current_class = df['class'].unique()[current_cluster_index]
            sampled_points = df[df['class'] == current_class].sample(10).to_dict('records')
            samples_json = json.dumps({"data": sampled_points, "current_index": 9})
            samples = json.loads(samples_json)
            sampled_point_index = 9

    elif button_id == 'play-audio':
        if pygame.mixer.music.get_busy():
            pause_sound()
            status = "Audio paused."
        else:
            if samples_json is None:
                return dash.no_update, "No audio to play.", dash.no_update

            samples = json.loads(samples_json)
            current_sample = samples["data"][samples["current_index"]]
            play_sound(current_sample['sound_path'])
            current_class = df[df['sound_path'] == current_sample['sound_path']]['class'].iloc[0]
            status = f"Playing sample {samples['current_index'] + 1} from cluster: {current_class}"
        return dash.no_update, status, samples_json

    elif button_id == 'file-dropdown':
        new_fig = update_figure(selected_file)
        current_csv_file = selected_file
        current_class = df['class'].unique()[current_cluster_index]
        sampled_points = df[df['class'] == current_class].sample(10).to_dict('records')
        new_samples_json = json.dumps({"data": sampled_points, "current_index": 0})
        return new_fig, "File changed. Playing from the first point in the new cluster.", new_samples_json
    
    elif button_id == 'control-button': 
        df.to_csv(current_csv_file, index=False)

    elif button_id == 'class-labels-checklist':
        if samples_json is None:
            return dash.no_update, "No sample data to update.", dash.no_update

        samples = json.loads(samples_json)
        current_sample = samples["data"][samples["current_index"]]
        
        sound_path = current_sample.get('sound_path')
        
        df_current = pd.read_csv(current_csv_file)
        
        idx = df_current[df_current['sound_path'] == sound_path].index[0]
        
        for label in label_options:
            column_name = label.lower().replace(" ", "_")
            
            if column_name in selected_labels:
                df_current.loc[idx, column_name] = 1
            else:
                if column_name not in df_current.columns:
                    df_current[column_name] = 0
                df_current.loc[idx, column_name] = 0
        
        df_current.to_csv(current_csv_file, index=False)
        
        return dash.no_update, "Labels saved.", dash.no_update
    
    elif clickData:
        clicked_point = clickData['points'][0]['customdata']
        play_sound(clicked_point)
        current_class = df[df['sound_path'] == clicked_point]['class'].iloc[0]
        status = f"Playing sample from clicked point in cluster: {current_class}"
        return dash.no_update, status, samples_json

    current_sample = samples["data"][sampled_point_index]
    play_sound(current_sample['sound_path'])
    current_class = df[df['sound_path'] == current_sample['sound_path']]['class'].iloc[0]
    status = f"Playing sample {sampled_point_index + 1} from cluster: {current_class}"
    
    samples['current_index'] = sampled_point_index
    samples_json = json.dumps(samples)

    new_fig = create_figure(df, current_cluster_index)

    return new_fig, status, samples_json

def get_sampled_points(df, current_cluster_index, sample_size=10):
    """
    Helper function to sample points from a DataFrame based on a specified class.

    Parameters:
    - df: The DataFrame containing the data.
    - current_cluster_index: Index of the current class to sample from.
    - sample_size: Desired sample size, defaults to 10.

    Returns:
    - A list of sampled points in dictionary format.
    """
    current_class = df['class'].unique()[current_cluster_index]
    class_subset = df[df['class'] == current_class]

    # Determine the sample size, limiting it to the available number of records if fewer than `sample_size`
    actual_sample_size = min(sample_size, len(class_subset))
    return class_subset.sample(actual_sample_size, replace=False).to_dict('records')

@app.callback(
    [Output('mel-spectrogram', 'src'),
     Output('waveform-plot', 'src')],
    Input('hidden-sample-data', 'children'),
    prevent_initial_call=True
)
def update_plots(samples_json):
    # Clear previous figure
    plt.clf()
    matplotlib.pyplot.close()

    if samples_json is None:
        return dash.no_update, dash.no_update

    samples = json.loads(samples_json)
    current_sample = samples["data"][samples["current_index"]]
    sound_file = current_sample['sound_path']

    # Generate Mel spectrogram using matplotlib and soundfile
    data, samplerate = sf.read(sound_file)
    plt.figure(figsize=(12, 3))
    plt.specgram(data, NFFT=1024, Fs=samplerate, noverlap=512)
    plt.tight_layout()

    # Convert the matplotlib figure to a PNG image
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    mel_spectrogram_base64 = base64.b64encode(buf.read()).decode()

    plt.close()

    # Generate waveform plot using matplotlib and soundfile
    plt.figure(figsize=(12, 3))
    plt.plot(data)
    plt.tight_layout()

    # Convert the matplotlib figure to a PNG image
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    waveform_base64 = base64.b64encode(buf.read()).decode()

    plt.close()

    # Return the base64 encoded images as the src of the HTML image tags
    return f'data:image/png;base64,{mel_spectrogram_base64}', f'data:image/png;base64,{waveform_base64}'


# @app.callback(
#     Output('temporary-storage', 'children'),
#     Input('upload-audio', 'contents'),
#     State('upload-audio', 'filename'),
#     State('temporary-storage', 'children')
# )
# def store_uploaded_files(contents, filenames, stored_files):
#     if contents is not None:
#         # Use a persistent file to store the paths
#         if stored_files is None:
#             temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.txt')
#             temp_file_path = temp_file.name
#             temp_file.close()
#         else:
#             temp_file_path = stored_files

#         with open(temp_file_path, 'a') as temp_file:
#             for content, filename in zip(contents, filenames):
#                 data = content.split(',')[1]
#                 decoded = base64.b64decode(data)
#                 file_path = os.path.join(TEMP_FOLDER, filename)
#                 with open(file_path, 'wb') as fp:
#                     fp.write(decoded)
#                 temp_file.write(file_path + '\n')

#         return temp_file_path

#     return stored_files

@app.callback(
    Output('upload-status', 'children'),
    [Input('uploader', 'isCompleted')],
    [State('uploader', 'fileNames'),
     State('uploader', 'upload_id')]
)
def handle_file_upload(isCompleted, filenames, upload_id):
    if not isCompleted:
        return "No files uploaded."

    if filenames is None or len(filenames) == 0:
        return "No files uploaded."

    # The folder path where files are uploaded
    folder_path = os.path.join(UPLOAD_FOLDER_ROOT, upload_id)

    # Save full file paths in a temporary storage for later processing
    for filename in filenames:
        file_path = os.path.join(folder_path, filename)
        uploaded_files.append(file_path)
    
    # Store the paths in a hidden div or another suitable storage element
    return json.dumps(uploaded_files)


@app.callback(
    [Output('file-dropdown', 'options'), 
     Output('file-dropdown', 'value'), 
     Output('csv-test', 'children')],
    Input('process-data-button', 'n_clicks'),
    State('upload-status', 'children'),
    prevent_initial_call=True
)
def process_uploaded_files_and_update_dropdown(n_clicks, stored_file_paths_json):
    if n_clicks is None or stored_file_paths_json is None:
        raise PreventUpdate

    stored_file_paths = json.loads(stored_file_paths_json)

    combined_feature_vectors = []
    combined_sound_paths = []

    # Process each uploaded file
    for file_path in stored_file_paths:
        # Ensure the file path is normalized
        file_path = os.path.normpath(file_path)
        print(f"Processing file: {file_path}")  # Debugging log

        audio = AudioSegment.from_file(file_path, format=file_path.rsplit('.', 1)[1].lower())
        output_folder = os.path.dirname(file_path)  # Extract the correct directory
        print(f"Output folder: {output_folder}")  # Debugging log

        feature_vectors, sound_paths = process_audio_chunk(
            audio, output_folder, file_path.rsplit('.', 1)[1].lower(), 
            os.path.splitext(os.path.basename(file_path))[0], duration=20
        )

        # Stack feature vectors and extend sound paths
        combined_feature_vectors.append(feature_vectors)
        combined_sound_paths.extend(sound_paths)

        # Clean up the temporary source file after processing
        os.remove(file_path)

    # Combine feature vectors
    combined_feature_vectors = np.vstack(combined_feature_vectors)

    # Apply UMAP for dimensionality reduction
    reducer = umap.UMAP(n_components=3, random_state=0, n_neighbors=6, min_dist=0)
    embedding = reducer.fit_transform(combined_feature_vectors)

    # Perform DBSCAN clustering and calculate silhouette score
    best_eps, best_min_samples, best_score, labels = calculate_silhouette_score(embedding)

    # Prepare a DataFrame with results
    #sound_paths_no_ext = [os.path.splitext(os.path.basename(path))[0] for path in combined_sound_paths]
    results_df = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'z': embedding[:, 2],
        'class': labels,
        'sound_path': combined_sound_paths
    })

    # Save processed data as a new CSV file
    processed_csv_filename = 'data/processed_file.csv'
    results_df.to_csv(processed_csv_filename, index=False)

    # Update dropdown options with the new file
    new_file_options = [{'label': os.path.basename(file), 'value': file} for file in glob.glob("data/*.csv")]

    # Clear stored file paths after processing
    stored_file_paths.clear()

    # Return updated options, set new file as the selected CSV, and path for debugging
    return new_file_options, processed_csv_filename, processed_csv_filename





# # Flask route for serving the results CSV file
# @app.server.route('/download/<path:filename>')
# def serve_file(filename):
#     # Ensure the file exists
#     file_path = os.path.join(TEMP_FOLDER, filename)
#     if not os.path.exists(file_path):
#         return f"File not found: {filename}", 404  # Return a 404 if the file doesn't exist

#     return send_from_directory(TEMP_FOLDER, filename, as_attachment=True)

# @app.callback(
#     Output('upload-file-info', 'children'),
#     [Input('upload-audio', 'filename')],
#     [State('temporary-storage', 'children')]
# )
# def update_file_info(filenames, stored_files):
#     if not filenames:
#         return 'No new files uploaded.'

#     # Read the stored file paths
#     all_files = []
#     if stored_files:
#         with open(stored_files, 'r') as file:
#             all_files = file.read().splitlines()

#     # Combine the new filenames with the existing ones
#     new_files = [os.path.basename(f) for f in filenames]  # Extract just the file names
#     all_files += new_files

#     # Format the display string
#     display_str = "" #f"Total Files Uploaded: {len(all_files)}".join(all_files)

#     return display_str

# Functions
def play_sound(sound_file):
    for subfolder in os.listdir(UPLOADS_DIRECTORY):
        subfolder_path = os.path.join(UPLOADS_DIRECTORY, subfolder)
        if os.path.isdir(subfolder_path):
            full_path = os.path.join(subfolder_path, sound_file)
            print(f"Checking path: {full_path}")  # Debug log
            if os.path.exists(full_path):
                pygame.mixer.music.load(full_path)
                pygame.mixer.music.play()
                return
    raise FileNotFoundError(f"Sound file not found in any subdirectory: {sound_file}")

# def play_sound(sound_file):
#     # Define the expected audio file extensions
#     valid_extensions = ['.wav', '.mp3', '.flac']  # Add any other formats supported by pygame

#     # Iterate over subdirectories in the uploads directory
#     for subfolder in os.listdir(UPLOADS_DIRECTORY):
#         subfolder_path = os.path.join(UPLOADS_DIRECTORY, subfolder)

#         # Ensure it's a directory
#         if os.path.isdir(subfolder_path):
#             for ext in valid_extensions:
#                 # Add the extension to the sound file name
#                 full_path = os.path.join(subfolder_path, sound_file + ext)

#                 # Check if the file exists
#                 if os.path.exists(full_path):
#                     pygame.mixer.music.load(full_path)
#                     pygame.mixer.music.play()
#                     return  # Exit the function after playing

#     # Raise an error if the file was not found with any valid extension
#     raise FileNotFoundError(f"Sound file not found with any of the valid extensions: {sound_file}")
    
def pause_sound():
    pygame.mixer.music.pause()

def update_figure(selected_file):
    global df, current_cluster_index, sampled_point_index
    # Read the new CSV file and update dataframe
    file_path = os.path.join(DATA_DIRECTORY, os.path.basename(selected_file))
    df = pd.read_csv(file_path)

    # Reset global variables
    current_cluster_index = 0
    sampled_point_index = 0

    # # Create a new figure based on the new data
    # fig = go.Figure()
    # fig.update_layout(paper_bgcolor="#171b26")
    # colors = cycle(plotly.colors.sequential.Rainbow)
    # for c in df['class'].unique():
    #     dfi = df[df['class'] == c]
    #     fig.add_trace(go.Scatter3d(x=dfi['x'], y=dfi['y'], z=dfi['z'],
    #                                mode='markers',
    #                                name=str(c),
    #                                customdata=dfi['sound_path'],
    #                                marker=dict(size=2),
    #                                marker_color=next(colors)))

    fig = create_figure(df, current_cluster_index)

    return fig

def get_features(y, sr):
    y = y[0:sr]
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc, mode='nearest')
    delta2_mfcc = librosa.feature.delta(mfcc, order=2, mode='nearest')
    feature_vector = np.concatenate((np.mean(mfcc,1), np.mean(delta_mfcc,1), np.mean(delta2_mfcc,1)))
    feature_vector = (feature_vector-np.mean(feature_vector)) / np.std(feature_vector)
    return feature_vector

def process_audio_chunk(audio, output_folder, file_format, original_filename, duration):
    # Normalize the output folder to prevent duplication
    output_folder = os.path.normpath(output_folder)

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    duration_seconds = duration * 60  # Convert duration from minutes to seconds
    chunk_length_ms = 4500  # Length of each audio chunk in milliseconds
    feature_vectors = []
    sound_paths = []

    total_length_ms = len(audio)
    total_chunks = total_length_ms // chunk_length_ms
    total_time_ms = 0

    for chunk_index in range(total_chunks):
        if total_time_ms > duration_seconds * 1000:
            break

        start_time = chunk_index * chunk_length_ms
        end_time = start_time + chunk_length_ms
        chunk_data = audio[start_time:end_time]

        # Construct the file path
        chunk_file_name = f"{original_filename}_{chunk_index}.{file_format}"
        chunk_file_path = os.path.join(output_folder, chunk_file_name)
        print(f"Saving chunk to: {chunk_file_path}")  # Debugging log

        # Export the chunk
        chunk_data.export(chunk_file_path, format=file_format)
        total_time_ms += chunk_length_ms

        # Append the correct file path
        sound_paths.append(chunk_file_path)

        # Load chunk and extract features
        y, sr = librosa.load(chunk_file_path, sr=None)
        feat = get_features(y, sr)
        feature_vectors.append(feat)

    # Normalize the feature vectors
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(feature_vectors)
    feature_vectors = pd.DataFrame(x_scaled)

    # Take only the first 13 MFCCs
    feature_vectors = feature_vectors.iloc[:, :13]

    return feature_vectors, sound_paths

def calculate_silhouette_score(embedding):
    X = embedding  # Assuming 'embedding' is your UMAP output

    # Define the range for DBSCAN parameters
    eps_values = np.arange(0.3, 1, 0.1)
    min_samples_values = range(5, 20, 2)

    best_eps = None
    best_min_samples = None
    best_score = -1

    for eps in eps_values:
        for min_samples in min_samples_values:
            # Apply DBSCAN clustering
            dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
            labels = dbscan.labels_

            # Calculate the silhouette score, excluding noise points
            if len(set(labels)) - (1 if -1 in labels else 0) > 1:
                score = silhouette_score(X[labels != -1], labels[labels != -1])
                print(f"EPS: {eps}, Min Samples: {min_samples}, Score: {score}, Clusters: {len(set(labels)) - 1}")

                # Check if this score is the best so far
                if score > best_score:
                    best_score = score
                    best_eps = eps
                    best_min_samples = min_samples

    print(f"Best EPS: {best_eps}, Best Min Samples: {best_min_samples}, Best Silhouette Score: {best_score}")

    # Apply DBSCAN with the best parameters
    dbscan = DBSCAN(eps=0.6, min_samples=6).fit(X)
    labels = dbscan.labels_

    return best_eps, best_min_samples, best_score, labels

def create_zip_file(output_folder, original_filename, csv_file_path):
    zip_path = os.path.join(TEMP_FOLDER, f'{original_filename}.zip')
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        # Add files in the output folder to the ZIP
        for root, dirs, files in os.walk(output_folder):
            for file in files:
                file_path = os.path.join(root, file)
                # Avoid adding the CSV file again if it's already in the output folder
                if file_path != csv_file_path:
                    zipf.write(file_path, os.path.relpath(file_path, output_folder))

        # Add the CSV file to the ZIP if it's not in the output folder
        if not os.path.exists(os.path.join(output_folder, os.path.basename(csv_file_path))):
            zipf.write(csv_file_path, os.path.basename(csv_file_path))
            
    return zip_path

if __name__ == '__main__':
    app.run_server(debug=True)


