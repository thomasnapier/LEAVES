"""
utils.py - Utility functions for the LEAVES application.

Author: Thomas Napier
"""

import pandas as pd
import plotly.graph_objs as go
import pygame
import librosa
import zipfile
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from io import BytesIO
import base64
import os

def save_annotations_to_csv(df, filepath):
    """Saves the DataFrame annotations to a CSV file."""
    df.to_csv(filepath, index=False)

def create_figure(df, current_cluster_index):
    """Creates a 3D scatter plot for the provided DataFrame."""
    classes = df['class'].unique()
    colors = plt.cm.get_cmap('tab10', len(classes))
    fig = go.Figure()

    for i, c in enumerate(classes):
        dfi = df[df['class'] == c]
        marker_size = 10 if i == current_cluster_index else 5  # Highlight current cluster
        fig.add_trace(go.Scatter3d(
            x=dfi['x'], y=dfi['y'], z=dfi['z'],
            mode='markers',
            name=f"Cluster {c}",
            customdata=dfi['sound_path'],
            marker=dict(size=marker_size, line=dict(width=0.1, color='black')),
            marker_color=[f'rgba{colors(i)}' for _ in range(len(dfi))]
        ))

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
        legend=dict(x=0.7, y=0.5, font=dict(size=10), bgcolor='rgba(255,255,255,0.5)')
    )

    return fig

def play_sound(sound_file):
    """Plays the sound file using Pygame."""
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()

def pause_sound():
    """Pauses the currently playing sound."""
    pygame.mixer.music.pause()

def update_figure(selected_file):
    """Updates the scatter plot figure when a new file is selected."""
    df = pd.read_csv(selected_file)
    current_cluster_index = 0  # Reset index
    fig = create_figure(df, current_cluster_index)
    return fig

def get_features(y, sr):
    """Extracts features from the audio data."""
    y = y[0:sr]
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc, mode='nearest')
    delta2_mfcc = librosa.feature.delta(mfcc, order=2, mode='nearest')
    feature_vector = np.concatenate((np.mean(mfcc, 1), np.mean(delta_mfcc, 1), np.mean(delta2_mfcc, 1)))
    feature_vector = (feature_vector - np.mean(feature_vector)) / np.std(feature_vector)
    return feature_vector

def process_audio_chunk(audio, output_folder, file_format, original_filename, duration):
    """Processes the audio into chunks and extracts features."""
    duration_seconds = duration * 60  # Convert duration from minutes to seconds
    chunk_length_ms = 4500  # Length of each audio chunk in milliseconds
    feature_vectors = []
    sound_paths = []

    # Ensure the directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    total_length_ms = len(audio)
    total_chunks = total_length_ms // chunk_length_ms
    total_time_ms = 0

    # Remove any directory components from the original filename
    clean_filename = os.path.basename(original_filename)

    for chunk_index in range(total_chunks):
        if total_time_ms > duration_seconds * 1000:
            break

        start_time = chunk_index * chunk_length_ms
        end_time = start_time + chunk_length_ms
        chunk_data = audio[start_time:end_time]

        chunk_file_name = f'{clean_filename}_chunk_{chunk_index}.{file_format}'
        chunk_file_path = os.path.join(output_folder, chunk_file_name)

        # Print statements to debug the path construction
        print(f"Exporting chunk to: {chunk_file_path}")
        
        try:
            chunk_data.export(chunk_file_path, format=file_format)
        except FileNotFoundError as e:
            print(f"Error exporting chunk: {e}")
            raise

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
    features_csv_path =  f'{clean_filename}_features.csv'
    df.to_csv(features_csv_path, index=False)

    return feature_vectors, sound_paths

def calculate_silhouette_score(embedding):
    """Calculates the silhouette score for the given embedding."""
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

                # Check if this score is the best so far
                if score > best_score:
                    best_score = score
                    best_eps = eps
                    best_min_samples = min_samples

    # Apply DBSCAN with the best parameters
    dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples).fit(X)
    labels = dbscan.labels_

    return best_eps, best_min_samples, best_score, labels

def create_zip_file(output_folder, original_filename, csv_file_path):
    """Creates a ZIP file containing the processed audio chunks and feature vectors."""
    zip_path = os.path.join(output_folder, f'{original_filename}.zip')
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

def generate_plots(sound_file):
    """Generates Mel spectrogram and waveform plots from the sound file."""
    # Read the sound file
    data, samplerate = sf.read(sound_file)

    # Generate Mel spectrogram using matplotlib and soundfile
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

    return f'data:image/png;base64,{mel_spectrogram_base64}', f'data:image/png;base64,{waveform_base64}'
