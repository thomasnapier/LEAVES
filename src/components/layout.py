"""
layout.py - Defines the layout of the LEAVES application.

Author: Thomas Napier
"""

import dash_uploader as du
from dash import dcc, html
import dash_bootstrap_components as dbc
import base64
import os

def create_layout(app, label_options):
    """Creates the layout for the Dash app."""
    logo_path = 'assets/logos/logo.png'
    encoded_logo = base64.b64encode(open(logo_path, 'rb').read()).decode('ascii')

    return html.Div([
        html.Div([
            html.Img(src=f'data:image/png;base64,{encoded_logo}', id='logo'),
            html.Span('LEAVES', id='top-banner-title')
        ], id='top-banner'),

        dcc.Store(id='current-cluster-index', data=0),
        dcc.Store(id='sampled-point-index', data=0),
        dcc.Store(id='current-csv-file', data='data/Undara-DryB.csv'),
        dcc.Store(id='annotations-store', data=[]),  # Store for annotations
        dcc.Store(id='uploaded-files-store', data=[]),

        
        html.Button('⚙️', id='open-settings', style={'position': 'absolute', 'top': '10px', 'right': '10px'}),
        html.Button('🌙/☀️', id='theme-toggle', style={'position': 'absolute', 'top': '10px', 'right': '150px'}),
        
        html.Div([
            html.H2("Settings"),
            html.Div([
                html.H4("Denoising Method"),
                dbc.ButtonGroup(
                    [
                        dbc.Button("None", id="denoising-none", active=True),
                        dbc.Button("Wavelet", id="denoising-wavelet"),
                        dbc.Button("Low-pass", id="denoising-lowpass"),
                        dbc.Button("High-pass", id="denoising-highpass"),
                        dbc.Button("Band-pass", id="denoising-bandpass"),
                    ],
                    vertical=True
                ),
            ]),
            html.Div([
                html.H4("Complexity Reduction"),
                dbc.ButtonGroup(
                    [
                        dbc.Button("UMAP", id="complexity-umap", active=True),
                        dbc.Button("t-SNE", id="complexity-tsne"),
                        dbc.Button("PCA", id="complexity-pca"),
                    ],
                    vertical=True
                ),
            ]),
            html.Div([
                html.H4("Clustering Algorithm"),
                dbc.ButtonGroup(
                    [
                        dbc.Button("DBSCAN", id="clustering-dbscan", active=True),
                        dbc.Button("K-Means", id="clustering-kmeans"),
                        dbc.Button("Agglomerative", id="clustering-agglomerative"),
                    ],
                    vertical=True
                ),
            ]),
            html.Button("Close", id="close-settings", n_clicks=0),
        ], id="settings-modal", style={"display": "none"}),

        html.Div([
            html.Div([
                du.Upload(
                    id='uploader',
                    text='Drag and Drop or Select Files',
                    text_completed='Upload Complete: ',
                    pause_button=True,
                    cancel_button=True,
                    max_file_size=1800,  # 1.8 GB
                    max_files=100
                ),
                html.Div(id='upload-status', style={'color': 'var(--text-color)'}),
                html.Button('Process Uploaded Data', id='process-data-button'),
                dcc.Dropdown(id='file-dropdown', options=[], value=None),
                html.Div(dcc.Graph(id='scatter-plot', style={'height': '400px', 'width': '100%', 'overflow': 'hidden'}), id='scatterplot-container'),
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
                html.Button('|◁◁', id='previous-cluster'),
                html.Button('◁', id='previous-point'),
                html.Button('||', id='play-audio'),
                html.Button('▷', id='next-point'),
                html.Button('▷▷|', id='next-cluster'),
            ], id='button-group'),
        ], id='bottom-timeline'),

        html.Div(id='hidden-sample-data'),
        html.Div(id='csv-dummy-output'),
        html.Div(id='csv-test'),
        html.Div(id='temporary-storage', style={'display': 'none'})
    ], id='main-container')
