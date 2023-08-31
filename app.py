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

# Existing code
df = pd.read_csv('data/umap-Wambiana-WetB-20min-full-day.csv')
pygame.mixer.init()

# Global variables
current_cluster_index = 0
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

# Define app and layout
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    dcc.Graph(id='scatter-plot', figure=fig, style={'height': '80vh'}),
    html.Div([
        html.Button('◁', id='previous-point'),
        html.Button('⏯', id='play-audio'),
        html.Button('▷', id='next-point'),
    ]),
    html.Div(id='audio-status', children='No audio being played currently.'),
    html.Div(id='hidden-sample-data')
], style={'textAlign': 'center'})

@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('audio-status', 'children'),
     Output('hidden-sample-data', 'children')],
    [Input('play-audio', 'n_clicks'),
     Input('next-point', 'n_clicks'),
     Input('previous-point', 'n_clicks')],
    State('hidden-sample-data', 'children'))
def process_audio(play_clicks, next_clicks, prev_clicks, samples_json):
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



        return fig, status, samples_json

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


def play_sound(sound_file):
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()
    
def pause_sound():
    pygame.mixer.music.pause()

if __name__ == '__main__':
    app.run_server(debug=True)
