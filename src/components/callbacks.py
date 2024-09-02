"""
callbacks.py - Registers callbacks for the LEAVES application.

Author: Thomas Napier
"""

# Imports
import pandas as pd
import plotly.graph_objs as go
import dash
from dash import dcc, html, State
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash import no_update
import json
import pygame
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import os
from pydub import AudioSegment
import tempfile
import librosa
from sklearn.preprocessing import MinMaxScaler
import umap
from sklearn.metrics import silhouette_score
from dash.exceptions import PreventUpdate
from sklearn.cluster import DBSCAN
import dash_uploader as du
from components.utils import (
    create_figure,
    save_annotations_to_csv,
    play_sound,
    pause_sound,
    update_figure,
    process_audio_chunk,
    calculate_silhouette_score,
    generate_plots
)

UPLOAD_FOLDER_ROOT = "uploads"
df = pd.read_csv('data/Undara-DryB.csv')
label_options = ["Background Silence", "Birds", "Frogs", "Human Speech", "Insects", "Mammals",
                 "Misc/Uncertain", "Rain (Heavy)", "Rain (Light)", "Vehicles (Aircraft/Cars)",
                 "Wind (Strong)", "Wind (Light)"]

# Registering the callbacks
def register_callbacks(app):
    """Register all the callbacks for the Dash app."""

    @app.callback(
        Output('settings-modal', 'style'),
        [Input('open-settings', 'n_clicks'), Input('close-settings', 'n_clicks')],
        [State('settings-modal', 'style')]
    )
    def toggle_modal(open_clicks, close_clicks, style):
        """Toggles the visibility of the settings modal."""
        ctx = dash.callback_context

        if not ctx.triggered:
            raise PreventUpdate

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'open-settings' and open_clicks:
            return {'display': 'block'}

        elif button_id == 'close-settings' and close_clicks:
            return {'display': 'none'}

        return style

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
        """Toggles the active denoising method."""
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

    @app.callback(
        [Output('complexity-umap', 'active'),
         Output('complexity-tsne', 'active'),
         Output('complexity-pca', 'active')],
        [Input('complexity-umap', 'n_clicks'),
         Input('complexity-tsne', 'n_clicks'),
         Input('complexity-pca', 'n_clicks')]
    )
    def toggle_complexity_method(n_umap, n_tsne, n_pca):
        """Toggles the active complexity reduction method."""
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        return [
            button_id == 'complexity-umap',
            button_id == 'complexity-tsne',
            button_id == 'complexity-pca'
        ]

    @app.callback(
        [Output('clustering-dbscan', 'active'),
         Output('clustering-kmeans', 'active'),
         Output('clustering-agglomerative', 'active')],
        [Input('clustering-dbscan', 'n_clicks'),
         Input('clustering-kmeans', 'n_clicks'),
         Input('clustering-agglomerative', 'n_clicks')]
    )
    def toggle_clustering_algorithm(n_dbscan, n_kmeans, n_agglomerative):
        """Toggles the active clustering algorithm."""
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
        Output('class-labels-checklist', 'value'),
        [Input('play-audio', 'n_clicks'),
         Input('next-point', 'n_clicks'),
         Input('previous-point', 'n_clicks')],
        [State('hidden-sample-data', 'children'),
         State('current-csv-file', 'data')]
    )
    def update_checklist(play_clicks, next_clicks, prev_clicks, samples_json, current_csv_file):
        """Updates the checklist based on the current audio sample."""
        if samples_json is None:
            return []

        samples = json.loads(samples_json)
        current_sample = samples["data"][samples["current_index"]]
        sound_file = current_sample['sound_path']

        # Load the current CSV file into a DataFrame
        df = pd.read_csv(current_csv_file)

        # Extract the row corresponding to the current sound file
        row = df[df['sound_path'] == sound_file].iloc[0]

        # Determine which checkboxes should be checked based on the DataFrame
        checked_boxes = [col for col in label_options if row[col.lower().replace(' ', '_')] == 1]
        return checked_boxes

    @app.callback(
        [Output('csv-dummy-output', 'children'),
         Output('annotations-store', 'data')],
        [Input('control-button', 'n_clicks')],
        [State('class-labels-checklist', 'value'),
         State('hidden-sample-data', 'children'),
         State('annotations-store', 'data'),
         State('current-csv-file', 'data')]
    )
    def update_annotations(n_clicks, selected_labels, samples_json, annotations, current_csv_file):
        """Updates annotations in the DataFrame and the annotations store."""
        if n_clicks is None:
            raise PreventUpdate

        if samples_json is None:
            return no_update, annotations

        samples = json.loads(samples_json)
        current_sample = samples["data"][samples["current_index"]]
        sound_path = current_sample['sound_path']

        df_current = pd.read_csv(current_csv_file)
        idx = df_current[df_current['sound_path'] == sound_path].index[0]

        for label in label_options:
            column_name = label.lower().replace(' ', '_')
            df_current.at[idx, column_name] = 1 if label in selected_labels else 0

        df_current.to_csv(current_csv_file, index=False)

        # Update the annotations store with the new annotations
        annotations.append({'sound_path': sound_path, 'labels': selected_labels})

        return no_update, annotations

    @app.callback(
        Output('csv-download-link', 'href'),
        Input('control-button', 'n_clicks'),
        [State('current-csv-file', 'data')],
        prevent_initial_call=True
    )
    def save_and_download_csv(n_clicks, current_csv_file):
        """Saves the annotations to CSV and returns the download link."""
        if n_clicks is None:
            raise PreventUpdate

        # Save annotations to the current CSV file
        save_annotations_to_csv(pd.read_csv(current_csv_file), current_csv_file)
        return f'/download/{current_csv_file}'

    @app.server.route('/download/<path:filename>')
    def serve_file(filename):
        """Serves the requested file for download."""
        file_path = os.path.join(os.getcwd(), filename)
        return send_from_directory(os.getcwd(), filename, as_attachment=True)

    @app.callback(
        [Output('scatter-plot', 'figure'),
         Output('audio-status', 'children'),
         Output('hidden-sample-data', 'children'),
         Output('current-cluster-index', 'data'),
         Output('sampled-point-index', 'data')],
        [Input('play-audio', 'n_clicks'),
         Input('next-point', 'n_clicks'),
         Input('previous-point', 'n_clicks'),
         Input('next-cluster', 'n_clicks'),
         Input('previous-cluster', 'n_clicks'),
         Input('file-dropdown', 'value'),
         Input('scatter-plot', 'clickData'),
         Input('class-labels-checklist', 'value')],
        [State('hidden-sample-data', 'children'),
         State('current-cluster-index', 'data'),
         State('sampled-point-index', 'data'),
         State('current-csv-file', 'data')]
    )
    def process_audio(play_clicks, next_clicks, prev_clicks, next_cluster_clicks, prev_cluster_clicks, selected_file, clickData, selected_labels, samples_json, current_cluster_index, sampled_point_index, current_csv_file):
        """Processes the audio data based on user interactions."""
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update, no_update, no_update

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

        if button_id == 'next-point':
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
                    return no_update, "No audio to play.", no_update, no_update, no_update

                samples = json.loads(samples_json)
                current_sample = samples["data"][samples["current_index"]]
                play_sound(current_sample['sound_path'])
                current_class = df[df['sound_path'] == current_sample['sound_path']]['class'].iloc[0]
                status = f"Playing sample {samples['current_index'] + 1} from cluster: {current_class}"
            return no_update, status, samples_json, current_cluster_index, sampled_point_index

        elif button_id == 'file-dropdown':
            new_fig = update_figure(selected_file)
            current_csv_file = selected_file
            current_class = df['class'].unique()[current_cluster_index]
            sampled_points = df[df['class'] == current_class].sample(10).to_dict('records')
            new_samples_json = json.dumps({"data": sampled_points, "current_index": 0})
            return new_fig, "File changed. Playing from the first point in the new cluster.", new_samples_json, current_cluster_index, sampled_point_index

        current_sample = samples["data"][sampled_point_index]
        play_sound(current_sample['sound_path'])
        current_class = df[df['sound_path'] == current_sample['sound_path']]['class'].iloc[0]
        status = f"Playing sample {sampled_point_index + 1} from cluster: {current_class}"

        samples['current_index'] = sampled_point_index
        samples_json = json.dumps(samples)

        new_fig = create_figure(df, current_cluster_index)

        return new_fig, status, samples_json, current_cluster_index, sampled_point_index

    @app.callback(
        [Output('mel-spectrogram', 'src'),
         Output('waveform-plot', 'src')],
        [Input('play-audio', 'n_clicks'),
         Input('next-point', 'n_clicks'),
         Input('previous-point', 'n_clicks'),
         Input('previous-cluster', 'n_clicks'),
         Input('next-cluster', 'n_clicks')],
        [State('hidden-sample-data', 'children')]
    )
    def update_plots(play_clicks, next_clicks, previous_clicks, previous_cluster_clicks, next_cluster_clicks, samples_json):
        """Updates the plots for the Mel spectrogram and waveform."""
        if samples_json is None:
            return no_update, no_update

        samples = json.loads(samples_json)
        current_sample = samples["data"][samples["current_index"]]
        sound_file = current_sample['sound_path']

        return generate_plots(sound_file)

    @app.callback(
        [Output('upload-status', 'children'),
        Output('uploaded-files-store', 'data')],
        [Input('uploader', 'isCompleted')],
        [State('uploader', 'fileNames'),
        State('uploader', 'upload_id'),
        State('uploaded-files-store', 'data')]
    )
    def handle_file_upload(isCompleted, filenames, upload_id, uploaded_files):
        """Handles the upload of audio files."""
        if not isCompleted:
            return "No files uploaded.", uploaded_files

        if filenames is None or len(filenames) == 0:
            return "No files uploaded.", uploaded_files

        folder_path = os.path.join(UPLOAD_FOLDER_ROOT, upload_id)

        for filename in filenames:
            file_path = os.path.join(folder_path, filename)
            uploaded_files.append(file_path)

        return json.dumps(uploaded_files), uploaded_files

    @app.callback(
        Output('csv-test', 'children'),
        Input('process-data-button', 'n_clicks'),
        State('upload-status', 'children'),
        prevent_initial_call=True
    )
    def process_uploaded_files(n_clicks, stored_file_paths_json):
        """Processes the uploaded files and updates the CSV."""
        if n_clicks is None or stored_file_paths_json is None:
            raise PreventUpdate

        stored_file_paths = json.loads(stored_file_paths_json)
        combined_feature_vectors = []
        combined_sound_paths = []

        for file_path in stored_file_paths:
            audio = AudioSegment.from_file(file_path, format=file_path.rsplit('.', 1)[1].lower())
            feature_vectors, sound_paths = process_audio_chunk(audio, os.path.dirname(file_path), file_path.rsplit('.', 1)[1].lower(), os.path.splitext(file_path)[0], duration=20)

            combined_feature_vectors.append(feature_vectors)
            combined_sound_paths.extend(sound_paths)

            os.remove(file_path)

        combined_feature_vectors = np.vstack(combined_feature_vectors)
        reducer = umap.UMAP(n_components=3, random_state=0, n_neighbors=6, min_dist=0)
        embedding = reducer.fit_transform(combined_feature_vectors)

        best_eps, best_min_samples, best_score, labels = calculate_silhouette_score(embedding)

        if len(embedding) != len(combined_sound_paths):
            return "Error: Mismatch in lengths of embedding and sound paths."

        results_df = pd.DataFrame({
            'x': embedding[:, 0],
            'y': embedding[:, 1],
            'z': embedding[:, 2],
            'class': labels,
            'sound_path': combined_sound_paths
        })

        results_csv_filename = 'combined_umap_dbscan.csv'
        results_csv_path = os.path.join(UPLOAD_FOLDER_ROOT, results_csv_filename)
        results_df.to_csv(results_csv_path, index=False)

        stored_file_paths.clear()

        return results_csv_path
