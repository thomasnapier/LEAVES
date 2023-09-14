"""
Program Name: A2OAudioLabeller
Developer: Thomas Napier
Organisation: James Cook University
GitHub: https://github.com/thomasnapier/A2OAudioLabeller
Description: This is a program designed to improve the audio labelling efficiency of
samples derived from the Australian Acoustic Observatory (A2O)
Date: 14/09/23
"""

#Imports
import pandas as pd
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

# Initialise
pygame.mixer.init()
df = pd.read_csv('data/umap-Duval-DryA-20min-full-day.csv')
label_options = ["Background Silence", "Birds", "Frogs", "Human Speech", "Insects", "Mammals",
                 "Misc/Uncertain", "Rain (Heavy)", "Rain (Light)", "Vehicles (Aircraft/Cars)",
                 "Wind (Strong)", "Wind (Light)"]
file_options = [{'label': file.split("\\")[-1], 'value': file} for file in glob.glob("data/*.csv")]

# Global Variables
current_cluster_index = 0
current_csv_file = 'data/umap-Duval-DryA-20min-full-day.csv'
sampled_point_index = 0
samples_json = None

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

# Define app and layout
app = dash.Dash(__name__, external_stylesheets=["assets\\styles.css"])

# Layout
app.layout = html.Div([
    dcc.Dropdown(id='file-dropdown', options=file_options, value=file_options[0]['value']),
    html.Div(id='audio-status', children='No audio being played currently.'),
    html.Div([
        html.Img(id='mel-spectrogram', src=''),
    ], id='spectrogram-area'),
    html.Div([
        html.Div([
            html.Div(id='checklist-title', children='Classes:'),
            dcc.Checklist(id='class-labels-checklist',
                  options=[
                      {'label': 'Background Silence', 'value': 'background_silence'},
                      {'label': 'Birds', 'value': 'birds'},
                      {'label': 'Frogs', 'value': 'frogs'},
                      {'label': 'Human Speech', 'value': 'human_speech'},
                      {'label': 'Insects', 'value': 'insects'},
                      {'label': 'Mammals', 'value': 'mammals'},
                      {'label': 'Misc/Uncertain', 'value': 'misc'},
                      {'label': 'Rain (Heavy)', 'value': 'rain_heavy'},
                      {'label': 'Rain (Light)', 'value': 'rain_light'},
                      {'label': 'Vehicles (Aircraft/Cars)', 'value': 'vehicles'},
                      {'label': 'Wind (Strong)', 'value': 'wind_strong'},
                      {'label': 'Wind (Light)', 'value': 'wind_light'},
                      #TODO: Make this list user-defined and/or enable users to make changes
                        ],value=[]),
                html.Div([
            html.Button('Save Data File', id='control-button')
        ])], id='checklist-container'),
        html.Div(dcc.Graph(id='scatter-plot', figure=fig, style={'height': '50vh'}), 
                 id='scatterplot-container')
    ], id='main-horizontal-layout'),
    html.Div([
            html.Button('◁', id='previous-point'),
            html.Button('⏯', id='play-audio'),
            html.Button('▷', id='next-point'),
                ], id='button-group'),
    html.Div(id='hidden-sample-data'),
    html.Div(id='csv-dummy-output')
], id='main-container')

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
     Input('class-labels-checklist', 'value')],
    State('hidden-sample-data', 'children'))
def process_audio(play_clicks, next_clicks, prev_clicks, selected_file, clickData, selected_labels, samples_json):
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

    if button_id == 'next-point':
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

if __name__ == '__main__':
    app.run_server(debug=True)

