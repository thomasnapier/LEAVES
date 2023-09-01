import pandas as pd
import plotly.graph_objs as go
import dash
from dash import dcc, html, State
from dash.dependencies import Input, Output
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

# Existing code
df = pd.read_csv('data/umap-Wambiana-WetB-20min-full-day.csv')
pygame.mixer.init()

current_cluster_index = 0
sampled_point_index = 0
samples_json = None
label_options = ["Background Silence", "Birds", "Frogs", "Human Speech", "Insects", "Mammals",
                 "Misc/Uncertain", "Rain (Heavy)", "Rain (Light)", "Vehicles (Aircraft/Cars)",
                 "Wind (Strong)", "Wind (Light)"]
file_options = [{'label': file.split("\\")[-1], 'value': file} for file in glob.glob("data/*.csv")]


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

# Define app and layout
app = dash.Dash(__name__, external_stylesheets=["assets\\styles.css"])

# Layout
app.layout = html.Div([
    dcc.Dropdown(id='file-dropdown', options=file_options, value=file_options[0]['value']),
    dcc.Graph(id='scatter-plot', figure=fig, style={'height': '60vh'}),
    html.Div([
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
                  ],
                  value=[]),
        html.Img(id='mel-spectrogram', src='')
    ], id='feature-area'),
    html.Div([
        html.Button('◁', id='previous-point'),
        html.Button('⏯', id='play-audio'),
        html.Button('▷', id='next-point'),
    ], id='button-group'),
    html.Div(id='audio-status', children='No audio being played currently.'),
    html.Div(id='hidden-sample-data')
])

@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('audio-status', 'children'),
     Output('hidden-sample-data', 'children')],
    [Input('play-audio', 'n_clicks'),
     Input('next-point', 'n_clicks'),
     Input('previous-point', 'n_clicks'),
     Input('file-dropdown', 'value')],
    State('hidden-sample-data', 'children'))
def process_audio(play_clicks, next_clicks, prev_clicks, selected_file, samples_json):
    global current_cluster_index, sampled_point_index
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
        # Initialize samples_json when file changes
        current_class = df['class'].unique()[current_cluster_index]
        sampled_points = df[df['class'] == current_class].sample(10).to_dict('records')
        new_samples_json = json.dumps({"data": sampled_points, "current_index": 0})
        return new_fig, "File changed. Playing from the first point in the new cluster.", new_samples_json

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

    plt.figure(figsize=(5, 2))
    if samples_json is None:
        return dash.no_update

    samples = json.loads(samples_json)
    current_sample = samples["data"][samples["current_index"]]
    sound_file = current_sample['sound_path']

    # Generate Mel spectrogram using matplotlib and soundfile
    data, samplerate = sf.read(sound_file)
    plt.specgram(data, NFFT=1024, Fs=samplerate, noverlap=512)
    plt.title('Mel Spectrogram')

    # Convert the matplotlib figure to a PNG image
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode the image as base64
    image_base64 = base64.b64encode(buf.read()).decode()

    # Return the base64 encoded image as the src of the HTML image tag
    return f'data:image/png;base64,{image_base64}'


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
