<p align="center"><img width=30% src="https://github.com/thomasnapier/LEAVES/blob/main/src/assets/logos/logo.png"></p>

# LEAVES: Large-scale Ecoacoustics Annotation and Visualisation with Efficient Segmentation

![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Basic Overview
LEAVES is an advanced tool designed to streamline the process of annotating and visualizing large-scale natural soundscape datasets. It leverages cutting-edge machine learning and efficient data processing techniques to make ecoacoustic analysis faster, more accurate, and accessible to researchers and citizen scientists alike.

![](https://github.com/thomasnapier/LEAVES/blob/main/src/assets/images/interface_1.png)
![](https://github.com/thomasnapier/LEAVES/blob/main/src/assets/images/interface_2.png)
![](https://github.com/thomasnapier/LEAVES/blob/main/src/assets/images/interface_3.png)

## Key Features
- Efficiency-Optimized Labelling: Significantly reduces the time required to annotate large datasets.
- Modular Design: Adaptable to a range of ecoacoustic research needs with customizable settings.
- User-Friendly Interface: Intuitive controls and visualizations make it accessible to users of all technical levels.
- Support for Various Audio Formats: Ensures broad compatibility with different recording setups.
- Real-Time Spectrogram Analysis: Facilitates in-depth exploration of soundscapes.

## Getting Started
To start using LEAVES, follow these steps:

1. Clone the Repository: 
```bash
git clone https://github.com/thomasnapier/LEAVES.git
```
2. Install Dependencies: Navigate to the project directory and run the following to install necessary Python libraries.
```bash
pip install -r requirements.txt 
```
3. Run the Application: Execute python app.py to start the web-based interface.
4. Upload Your Data: Use the upload feature to import your audio recordings.
5. Explore and Annotate: Utilize the provided tools to process, visualize, and annotate your data

## How it Works

1. Data Ingestion: Upload audio recordings in common formats like .WAV, .MP3, and .FLAC.
2. Signal Processing: The tool preprocesses the audio data, including denoising and feature extraction, to prepare it for analysis.
3. Dimensionality Reduction: Techniques like UMAP and t-SNE are applied to reduce data complexity, making it easier to visualize and analyze.
4. Clustering: LEAVES employs algorithms such as DBSCAN and k-means to group similar sounds, enhancing the efficiency of the annotation process.
5. Annotation: Users can annotate clusters of similar sounds, significantly speeding up the labelling process compared to manual methods.
6. Visualization: Interactive 3D plots and spectrograms allow users to explore and understand the data in detail.

<p align="center"><img width=50% src="https://github.com/thomasnapier/LEAVES/blob/main/src/assets/images/Scheme.png"></p>

## Notes
  - **Technologies/Libraries**: Python3, Python Dash, Plotly, Librosa, Scikit-Learn
  - **Status**:  Alpha
  - **Links to production or demo instances**
  - Describe what sets this apart from related-projects. Linking to another doc or page is OK if this can't be expressed in a sentence or two.

## Dependencies

Describe any dependencies that must be installed for this software to work.
This includes programming languages, databases or other storage mechanisms, build tools, frameworks, and so forth.
If specific versions of other software are required, or known not to work, call that out.

## Installation

Detailed instructions on how to install, configure, and get the project running.
This should be frequently tested to ensure reliability. Alternatively, link to
a separate [INSTALL](INSTALL.md) document.

## Configuration

If the software is configurable, describe it in detail, either here or in other documentation to which you link.

## Usage

Show users how to use the software.
Be specific.
Use appropriate formatting when showing code snippets.

## How to test the software

If the software includes automated tests, detail how to run those tests.

## Known issues

Document any known significant shortcomings with the software.

## Getting help

Instruct users how to get help with this software; this might include links to an issue tracker, wiki, mailing list, etc.

**Example**

If you have questions, concerns, bug reports, etc, please file an issue in this repository's Issue Tracker.

## Getting involved

This section should detail why people should get involved and describe key areas you are
currently focusing on; e.g., trying to get feedback on features, fixing certain bugs, building
important pieces, etc.

General instructions on _how_ to contribute should be stated with a link to [CONTRIBUTING](CONTRIBUTING.md).


----

## Open source licensing info
1. [TERMS](TERMS.md)
2. [LICENSE](LICENSE)
3. [CFPB Source Code Policy](https://github.com/cfpb/source-code-policy/)


----

## Credits and references

1. Projects that inspired you
2. Related projects
3. Books, papers, talks, or other sources that have meaningful impact or influence on this project