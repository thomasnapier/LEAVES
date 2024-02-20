"""
Program Name: A2OAudioLabeller
Developer: Thomas Napier
Organisation: James Cook University
GitHub: https://github.com/thomasnapier/A2OAudioLabeller
Description: This is a program designed to improve the audio labelling efficiency of
samples derived from the Australian Acoustic Observatory (A2O)
"""

#Imports
import pandas as pd
from py import process
import plotly.graph_objs as go
import dash
from dash import dcc, html, State
from dash.dependencies import Input, Output
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


# Initialise
pygame.mixer.init()
df = pd.read_csv('data/umap-Duval-DryA-20min-full-day.csv')
label_options = ["Background Silence", "Birds", "Frogs", "Human Speech", "Insects", "Mammals",
                 "Misc/Uncertain", "Rain (Heavy)", "Rain (Light)", "Vehicles (Aircraft/Cars)",
                 "Wind (Strong)", "Wind (Light)"]
file_options = [{'label': file.split("\\")[-1], 'value': file} for file in glob.glob("data/*.csv")]
current_cluster_index = 0
current_csv_file = 'data/umap-Duval-DryA-20min-full-day.csv'
sampled_point_index = 0
samples_json = None
uploaded_files = []

# Assume temporary storage on the server-side
TEMP_FOLDER = tempfile.mkdtemp()

colors = cycle(plotly.colors.sequential.Rainbow)
fig = go.Figure()
for c in df['class'].unique():
    dfi = df[df['class'] == c]
    fig.add_trace(go.Scatter3d(x=dfi['x'], y=dfi['y'], z=dfi['z'],
                               mode='markers',
                               name=str(c),
                               customdata=dfi['sound_path'],
                               marker=dict(size=2),
                               marker_color=next(colors)))
    
for label_option in label_options:
    if label_option.lower().replace(' ', '_') not in df.columns:
        df[label_option.lower().replace(' ', '_')] = 0

fig.update_layout(
    paper_bgcolor="#171b26",
    legend_orientation='h',  # Horizontal orientation
    legend=dict(
        x=0,  # Adjust legend X position
        y=-0.5,  # Adjust legend Y position to be below the plot
        xanchor='left',
        yanchor='top'
    )
)

# Define app and layout
app = dash.Dash(__name__, external_stylesheets=["assets\\styles.css"])

app.layout = html.Div([
    html.Div('A20AudioLabeller', id='top-banner'),  # Top banner with the app title
    html.Button('⚙️', id='open-settings', style={'position': 'absolute', 'top': '10px', 'right': '10px'}),
    # Settings Modal Structure
    html.Div(
        [
            html.Div(
                [
                    html.H2("Software Configuration", style={'textAlign': 'center', 'color': '#FFFFFF'}),
                    html.Hr(style={'background-color': '#FFFFFF'}),

                    html.H4("Preprocessing"),
                    dcc.Checklist(
                        options=[
                            {'label': ' Systematic Data Sampling', 'value': 'SDS'},
                            {'label': ' Short Term Windowing', 'value': 'STW'}
                        ],
                        id='preprocessing-options'
                    ),
                    dcc.Slider(id='data-sampling-slider', min=0, max=40, value=20, step=1, marks={i: f'{i} min' for i in range(0, 41, 5)}),
                    dcc.Input(id='windowing-input', type='number', value=4.5, step=0.5, min=3.5, max=5.5),
                    dcc.Tabs(
                        id="denoising-tabs",
                        value='none',
                        children=[
                            dcc.Tab(label='None', value='none'),
                            dcc.Tab(label='Wavelet-based', value='wavelet'),
                            dcc.Tab(label='Low-pass', value='low-pass'),
                            dcc.Tab(label='High-pass', value='high-pass'),
                            dcc.Tab(label='Band-pass', value='band-pass'),
                        ]
                    ),
                    html.Hr(),

                    html.H4("Feature Extraction"),
                    dcc.Checklist(
                        options=[
                            {'label': ' Min-Max Normalisation', 'value': 'MMN'},
                            {'label': ' Include MFCC Derivatives', 'value': 'IMD'}
                        ],
                        id='feature-extraction-options'
                    ),
                    html.Hr(),

                    html.H4("Complexity Reduction"),
                    dcc.Tabs(
                        id="complexity-tabs",
                        value='UMAP',
                        children=[
                            dcc.Tab(label='UMAP', value='UMAP'),
                            dcc.Tab(label='t-SNE', value='t-SNE'),
                            dcc.Tab(label='PCA', value='PCA'),
                        ]
                    ),
                    dcc.Slider(id='n-neighbours-slider', min=0, max=100, value=15, step=1, marks={i: str(i) for i in range(0, 101, 10)}, tooltip={"placement": "bottom", "always_visible": True}),
                    dcc.Slider(id='min-dist-slider', min=0, max=1, value=0.1, step=0.1, marks={i/10: str(i/10) for i in range(0, 11, 1)}, tooltip={"placement": "bottom", "always_visible": True}),
                    html.Hr(),

                    html.H4("Clustering"),
                    dcc.Dropdown(
                        id='clustering-algorithm-dropdown',
                        options=[
                            {'label': 'DBScan', 'value': 'DBScan'},
                            {'label': 'k-means', 'value': 'k-means'},
                            {'label': 'Agglomerative', 'value': 'Agglomerative'},
                        ],
                        value='DBScan'
                    ),
                    dcc.Slider(id='eps-slider', min=0, max=5, value=0.5, step=0.1, marks={i/10: str(i/10) for i in range(0, 51, 5)}, tooltip={"placement": "bottom", "always_visible": True}),
                    dcc.Slider(id='min-samples-slider', min=0, max=100, value=5, step=1, marks={i: str(i) for i in range(0, 101, 10)}, tooltip={"placement": "bottom", "always_visible": True}),

                    html.Button('Close', id='close-settings', style={'margin': '20px'}),
                ],
                style={'padding': '20px', 'color': '#FFFFFF', 'background-color': '#171b26'}
            ),
        ],
        id='settings-modal',
        style={'display': 'none', 'position': 'fixed', 'z-index': '1000', 'left': '25%', 'top': '10%', 'width': '50%', 'background-color': '#171b26', 'border': '2px solid #ddd', 'border-radius': '5px', 'color': '#FFFFFF'}
    ),
    html.Div([  # Main content area
        html.Div([  # Left column container
        dcc.Loading(  # Add the Loading component
            id="loading-upload",
            children=[
                dcc.Upload(
                    id='upload-audio',
                    children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin-bottom': '10px',
                    },
                    # Allow multiple files to be uploaded
                    multiple=True
                ),
                html.Div(id='upload-file-info', style={'white-space': 'pre-line'}),
            ],
            type="circle", 
        ),
            html.Button('Process Uploaded Data', id='process-data-button'),
            dcc.Dropdown(id='file-dropdown', options=file_options, value=file_options[0]['value'], style={'display': 'none'}),
            html.Div(dcc.Graph(id='scatter-plot', figure=fig), 
                     id='scatterplot-container'),
        ], id='left-column'),  # Closing left column
        html.Div([  # Right column container
            html.Div(id='project-info', children='This is a program designed to improve the audio labelling efficiency of samples derived from the Australian Acoustic Observatory (A2O)'),
            html.Img(id='mel-spectrogram', src=''),
            html.Div([  # Control buttons
                html.Button('◁', id='previous-point'),
                html.Button('||', id='play-audio'),
                html.Button('▷', id='next-point'),
            ], id='button-group'),
            html.Div(id='audio-status', children='No audio being played currently.'),
            html.Div([
            html.Div(id='checklist-title', children='Classes:'),
            dcc.Checklist(id='class-labels-checklist',
                  options=[
                      {'label': 'Background Silence', 'value': 'Background Silence'},
                      {'label': 'Birds', 'value': 'Birds'},
                      {'label': 'Frogs', 'value': 'Frogs'},
                      {'label': 'Human Speech', 'value': 'Human Speech'},
                      {'label': 'Insects', 'value': 'Insects'},
                      {'label': 'Mammals', 'value': 'Mammals'},
                      {'label': 'Misc/Uncertain', 'value': 'Misc/Uncertain'},
                      {'label': 'Rain (Heavy)', 'value': 'Rain (Heavy)'},
                      {'label': 'Rain (Light)', 'value': 'Rain (Light)'},
                      {'label': 'Vehicles (Aircraft/Cars)', 'value': 'Vehicles (Aircraft/Cars)'},
                      {'label': 'Wind (Strong)', 'value': 'Wind (Strong)'},
                      {'label': 'Wind (Light)', 'value': 'Wind (Light)'},
                      #TODO: Make this list user-defined and/or enable users to make changes
                        ],value=[]),
                html.Div([
            html.Button('Save Data File', id='control-button')
        ])], id='checklist-container'),
        ], id='right-column')
    ], id='main-horizontal-layout'),
    html.Div(id='hidden-sample-data'),
    html.Div(id='csv-dummy-output'),
    html.Div(id='csv-test'),
    html.Div(id='temporary-storage', style={'display': 'none'})
], id='main-container')

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
        return {'display': 'block', 'position': 'fixed', 'z-index': '1000', 'left': '25%', 'top': '10%', 'width': '50%', 'background-color': '#171b26', 'border': '2px solid #ddd', 'border-radius': '5px', 'color': '#FFFFFF'}

    elif button_id == 'close-settings' and close_clicks:
        return {'display': 'none'}

    return style


# Callbacks
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
    Output('csv-dummy-output', 'children'),  # Create a new dummy output to save CSV
    [Input('control-button', 'n_clicks'),
     Input('class-labels-checklist', 'value')],
    [State('hidden-sample-data', 'children')])
def save_to_csv(n_clicks, checklist_values, samples_json):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'control-button':
        # Save the DataFrame to the same CSV file
        df.to_csv(current_csv_file, index=False)
    else:
        # Update the DataFrame based on which checkboxes are checked
        if samples_json:
            samples = json.loads(samples_json)
            current_sample = samples["data"][samples["current_index"]]
            sound_file = current_sample['sound_path']

            # Reset all label columns to 0 for the current sound file
            for col in label_options:
                df.loc[df['sound_path'] == sound_file, col.lower().replace(' ', '_')] = 0
            
            # Set selected label columns to 1 for the current sound file
            for value in checklist_values:
                df.loc[df['sound_path'] == sound_file, value] = 1

    return dash.no_update  #TODO: Display some text here to confirm save

@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('audio-status', 'children'),
     Output('hidden-sample-data', 'children')],
    [Input('play-audio', 'n_clicks'),
     Input('next-point', 'n_clicks'),
     Input('previous-point', 'n_clicks'),
     Input('file-dropdown', 'value'),
     Input('scatter-plot', 'clickData'),
     Input('class-labels-checklist', 'value'),
     Input('csv-test', 'children')],
    State('hidden-sample-data', 'children'))
def process_audio(play_clicks, next_clicks, prev_clicks, selected_file, clickData, selected_labels, csv_path, samples_json):
    global current_cluster_index, sampled_point_index, current_csv_file
    # Trigger context
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


    if button_id == 'csv-test' and csv_path:
        # Read the new CSV file
        new_df = pd.read_csv(csv_path)

        # Create a new figure based on the new data
        new_fig = go.Figure()
        new_fig.update_layout(paper_bgcolor="#171b26")
        colors = cycle(plotly.colors.sequential.Rainbow)
        for c in new_df['class'].unique():
            dfi = new_df[new_df['class'] == c]
            new_fig.add_trace(go.Scatter3d(x=dfi['x'], y=dfi['y'], z=dfi['z'],
                                           mode='markers',
                                           name=str(c),
                                           customdata=dfi['sound_path'],
                                           marker=dict(size=2),
                                           marker_color=next(colors)))

        # Update the sample data state
        new_samples_json = json.dumps({"data": new_df.to_dict('records'), "current_index": 0})

        return new_fig, "Scatter plot updated from new CSV file.", new_samples_json

    elif button_id == 'next-point':
        sampled_point_index = samples.get('current_index', 0) + 1
        if sampled_point_index >= 10:
            # Move to the next cluster
            current_cluster_index += 1
            if current_cluster_index >= len(df['class'].unique()):
                current_cluster_index = 0  # Reset to the first cluster
            current_class = df['class'].unique()[current_cluster_index]
            sampled_points = df[df['class'] == current_class].sample(10).to_dict('records')
            samples_json = json.dumps({"data": sampled_points, "current_index": 0})
            samples = json.loads(samples_json)
            sampled_point_index = 0

    elif button_id == 'previous-point':
        sampled_point_index = samples.get('current_index', 0) - 1
        if sampled_point_index < 0:
            # Move to the previous cluster
            current_cluster_index -= 1
            if current_cluster_index < 0:
                current_cluster_index = len(df['class'].unique()) - 1  # Reset to the last cluster
            current_class = df['class'].unique()[current_cluster_index]
            sampled_points = df[df['class'] == current_class].sample(10).to_dict('records')
            samples_json = json.dumps({"data": sampled_points, "current_index": 9})
            samples = json.loads(samples_json)
            sampled_point_index = 9

    elif button_id == 'play-audio':
        current_sample = samples["data"][samples["current_index"]]
        play_sound(current_sample['sound_path'])
        current_class = df[df['sound_path'] == current_sample['sound_path']]['class'].iloc[0]
        status = f"Playing sample {samples['current_index'] + 1} from cluster: {current_class}"

    elif button_id == 'file-dropdown':
        new_fig = update_figure(selected_file)
        current_csv_file = selected_file
        # Initialize samples_json when file changes
        current_class = df['class'].unique()[current_cluster_index]
        sampled_points = df[df['class'] == current_class].sample(10).to_dict('records')
        new_samples_json = json.dumps({"data": sampled_points, "current_index": 0})
        return new_fig, "File changed. Playing from the first point in the new cluster.", new_samples_json
    
    elif button_id == 'control-button':  # Assuming this is the ID for your save button
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
        return dash.no_update, status, samples_json  # Use dash.no_update for figure so that it doesn't change.

    # Update the sample index and audio status
    current_sample = samples["data"][sampled_point_index]
    play_sound(current_sample['sound_path'])
    current_class = df[df['sound_path'] == current_sample['sound_path']]['class'].iloc[0]
    status = f"Playing sample {sampled_point_index + 1} from cluster: {current_class}"
    
    samples['current_index'] = sampled_point_index
    samples_json = json.dumps(samples)

    # Highlight the current cluster in the scatter plot
    for trace in fig.data:
        if trace.name == str(current_class):
            trace.marker.size = 10
        else:
            trace.marker.size = 2

    return fig, status, samples_json

@app.callback(Output('mel-spectrogram', 'src'),
              [Input('play-audio', 'n_clicks'),
               Input('next-point', 'n_clicks')],
              [State('hidden-sample-data', 'children')])
def update_spectrogram(play_clicks, next_clicks, samples_json):
    # Clear previous figure
    plt.clf()
    matplotlib.pyplot.close()

    plt.figure(figsize=(12, 3))
    if samples_json is None:
        return dash.no_update

    samples = json.loads(samples_json)
    current_sample = samples["data"][samples["current_index"]]
    sound_file = current_sample['sound_path']

    # Generate Mel spectrogram using matplotlib and soundfile
    data, samplerate = sf.read(sound_file)
    Pxx, freqs, bins, im = plt.specgram(data, NFFT=1024, Fs=samplerate, noverlap=512)
    plt.title('Spectrogram')

    # Convert the matplotlib figure to a PNG image
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode the image as base64
    image_base64 = base64.b64encode(buf.read()).decode()

    # Return the base64 encoded image as the src of the HTML image tag
    return f'data:image/png;base64,{image_base64}'

@app.callback(
    Output('temporary-storage', 'children'),
    Input('upload-audio', 'contents'),
    State('upload-audio', 'filename'),
    State('temporary-storage', 'children')
)
def store_uploaded_files(contents, filenames, stored_files):
    if contents is not None:
        # Use a persistent file to store the paths
        if stored_files is None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.txt')
            temp_file_path = temp_file.name
            temp_file.close()
        else:
            temp_file_path = stored_files

        with open(temp_file_path, 'a') as temp_file:
            for content, filename in zip(contents, filenames):
                data = content.split(',')[1]
                decoded = base64.b64decode(data)
                file_path = os.path.join(TEMP_FOLDER, filename)
                with open(file_path, 'wb') as fp:
                    fp.write(decoded)
                temp_file.write(file_path + '\n')

        return temp_file_path

    return stored_files

@app.callback(
    Output('csv-test', 'children'),
    Input('process-data-button', 'n_clicks'),
    State('temporary-storage', 'children'),
    prevent_initial_call=True
)
def process_all_uploaded_files(n_clicks, stored_files):
    if n_clicks is None or stored_files is None:
        raise PreventUpdate

    # Read file paths from the temporary file
    with open(stored_files, 'r') as file:
        file_paths = file.read().splitlines()

    combined_feature_vectors = []
    combined_sound_paths = []

    for file_path in file_paths:
        audio = AudioSegment.from_file(file_path, format=file_path.rsplit('.', 1)[1].lower())
        feature_vectors, sound_paths = process_audio_chunk(audio, os.path.dirname(file_path), file_path.rsplit('.', 1)[1].lower(), os.path.splitext(file_path)[0], duration=20)
        
        # Stack feature_vectors and extend sound_paths
        combined_feature_vectors.append(feature_vectors)
        combined_sound_paths.extend(sound_paths)

        # Clean up the temporary source file
        os.remove(file_path)

    # Convert list of arrays to a single 2D array for feature_vectors
    combined_feature_vectors = np.vstack(combined_feature_vectors)

    # Apply UMAP Dimension Reduction
    reducer = umap.UMAP(n_components=3, random_state=0, n_neighbors=6, min_dist=0)
    embedding = reducer.fit_transform(combined_feature_vectors)

    # Calculate silhouette score and apply DBSCAN
    best_eps, best_min_samples, best_score, labels = calculate_silhouette_score(embedding)

    # Create a DataFrame with the results
    if len(embedding) != len(combined_sound_paths):
        return "Error: Mismatch in lengths of embedding and sound paths."

    results_df = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'z': embedding[:, 2],
        'class': labels,
        'sound_path': combined_sound_paths
    })

    # Save the results to a CSV file
    results_csv_filename = 'combined_umap_dbscan.csv'
    results_csv_path = os.path.join(TEMP_FOLDER, results_csv_filename)
    results_df.to_csv(results_csv_path, index=False)

    # Clear the global list after processing
    uploaded_files.clear()
    return results_csv_path

# Flask route for serving the results CSV file
@app.server.route('/download/<path:filename>')
def serve_file(filename):
    # Ensure the file exists
    file_path = os.path.join(TEMP_FOLDER, filename)
    if not os.path.exists(file_path):
        return f"File not found: {filename}", 404  # Return a 404 if the file doesn't exist

    return send_from_directory(TEMP_FOLDER, filename, as_attachment=True)

@app.callback(
    Output('upload-file-info', 'children'),
    [Input('upload-audio', 'filename')],
    [State('temporary-storage', 'children')]
)
def update_file_info(filenames, stored_files):
    if not filenames:
        return 'No new files uploaded.'

    # Read the stored file paths
    all_files = []
    if stored_files:
        with open(stored_files, 'r') as file:
            all_files = file.read().splitlines()

    # Combine the new filenames with the existing ones
    new_files = [os.path.basename(f) for f in filenames]  # Extract just the file names
    all_files += new_files

    # Format the display string
    display_str = "" #f"Total Files Uploaded: {len(all_files)}".join(all_files)

    return display_str

# Functions
def play_sound(sound_file):
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()
    
def pause_sound():
    pygame.mixer.music.pause()

def update_figure(selected_file):
    global df, current_cluster_index, sampled_point_index
    # Read the new CSV file and update dataframe
    df = pd.read_csv(selected_file)

    # Reset global variables
    current_cluster_index = 0
    sampled_point_index = 0

    # Create a new figure based on the new data
    fig = go.Figure()
    fig.update_layout(paper_bgcolor="#171b26")
    colors = cycle(plotly.colors.sequential.Rainbow)
    for c in df['class'].unique():
        dfi = df[df['class'] == c]
        fig.add_trace(go.Scatter3d(x=dfi['x'], y=dfi['y'], z=dfi['z'],
                                   mode='markers',
                                   name=str(c),
                                   customdata=dfi['sound_path'],
                                   marker=dict(size=2),
                                   marker_color=next(colors)))

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

        chunk_file_name = f'{original_filename}_{chunk_index}.{file_format}'
        chunk_file_path = os.path.join(output_folder, chunk_file_name)

        chunk_data.export(chunk_file_path, format=file_format)
        total_time_ms += chunk_length_ms

        # Load chunk and extract features
        y, sr = librosa.load(chunk_file_path, sr=None)
        feat = get_features(y, sr)
        feature_vectors.append(feat)
        sound_paths.append(chunk_file_path)

    # Normalize the feature vectors
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(feature_vectors)
    feature_vectors = pd.DataFrame(x_scaled)
    
    # Take only the first 13 MFCCs
    feature_vectors = feature_vectors.iloc[:, :13]

    # Save feature vectors to CSV
    paths = pd.DataFrame(sound_paths)
    df = pd.concat([feature_vectors, paths], ignore_index=True, sort=False, axis=1)
    features_csv_path = os.path.join(output_folder, f'{original_filename}_features.csv')
    df.to_csv(features_csv_path, index=False)

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

