
import pandas as pd
import plotly.graph_objs as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly
import pygame
import csv
import random
import json
from itertools import cycle

# Existing code
df = pd.read_csv('data\\umap-Wambiana-WetB-20min-full-day.csv')
pygame.mixer.init()

# Global variables
current_cluster_index = 0
sampled_point_index = 0


# Define your app and layout
app = dash.Dash(__name__)

colors = cycle(plotly.colors.sequential.Rainbow)
fig = go.Figure()
for c in df['class'].unique():
    dfi = df[df['class'] == c]
    fig.add_trace(go.Scatter3d(x=dfi['x'], y = dfi['y'], z = dfi['z'],
                               mode = 'markers',
                               name = str(c),
                               customdata = dfi['sound_path'],
                               marker=dict(
                                        size=2),
                               marker_color = next(colors)))

# # keep legend readable and edit graph so it doesn't reset upon user interaction
# fig.update_layout(legend= {'itemsizing': 'constant'}, uirevision="some value") 

# Layout
app.layout = html.Div([
    dcc.Graph(id='scatter-plot', figure=fig),
    html.Button('Play', id='play-audio'),
    html.Div(id='audio-status', children='No audio being played currently.'),
    dcc.Checklist(id='class-label',
                  options=[
                      {'label': 'Anthrophony', 'value': 'anthrophony'},
                      {'label': 'Biophony', 'value': 'biophony'},
                      {'label': 'Geophony', 'value': 'geophony'},
                      {'label': 'Other', 'value': 'other'}
                  ],
                  inline=True),
    html.Button('Next Point', id='next-point'),
    html.Div(id='hidden-sample-data', style={'display': 'none'})
])

@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('audio-status', 'children'),
     Output('hidden-sample-data', 'children')],
    [Input('play-audio', 'n_clicks'),
     Input('next-point', 'n_clicks')],
    State('hidden-sample-data', 'children'))
def process_audio(play_clicks, next_clicks, samples_json):
    global current_cluster_index

    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if samples_json is None:
        current_class = df['class'].unique()[current_cluster_index]
        sampled_points = df[df['class'] == current_class].sample(10).to_dict('records')
        samples_json = json.dumps({"data": sampled_points, "current_index": 0})

    samples = json.loads(samples_json)

    if button_id == 'play-audio':
        current_sample = samples["data"][samples["current_index"]]
        play_sound(current_sample['sound_path'])
        current_class = df[df['sound_path'] == current_sample['sound_path']]['class'].iloc[0]
        status = f"Playing sample {samples['current_index'] + 1} from cluster: {current_class}"

        # Highlight the current cluster in the scatter plot
        for trace in fig.data:
            if trace.name == str(current_class):
                trace.marker.size = 10
            else:
                trace.marker.size = 2

        return fig, status, samples_json

    elif button_id == 'next-point':
        sampled_point_index = samples.get('current_index', 0) + 1
        if sampled_point_index < 10:
            current_sample = samples["data"][sampled_point_index]
            play_sound(current_sample['sound_path'])
            current_class = df[df['sound_path'] == current_sample['sound_path']]['class'].iloc[0]
            status = f"Playing sample {sampled_point_index + 1} from cluster: {current_class}"

            # Save the current index into the hidden data
            samples['current_index'] = sampled_point_index
            samples_json = json.dumps(samples)

        else:
            # Move to the next cluster
            current_cluster_index += 1
            for trace in fig.data:
                trace.marker.size = 2
            status = "All samples played from the current cluster."

        return fig, status, samples_json

    return dash.no_update, dash.no_update, samples_json


# @app.callback(Output('hidden-sample-data', 'children'),
#               [Input('class-label', 'value')])
# def save_labels_and_provide_next_sample(labels):
#     samples = json.loads(samples_json)
#     current_index = samples.get('current_index', 0)
#     samples[current_index]['labels'] = labels

#     # Save the labels to the CSV
#     for i, label in enumerate(labels):
#         col_name = f"UserChoice{i+1}"
#         if col_name not in df.columns:
#             df[col_name] = ""
#         df.loc[df['sound_path'] == samples[current_index]['sound_path'], col_name] = label

#     df.to_csv('umap-Wambiana-WetB-20min-full-day.csv', index=False)

#     return json.dumps(samples)


def play_sound(sound_file):
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()

if __name__ == '__main__':
    app.run_server(debug=True)
