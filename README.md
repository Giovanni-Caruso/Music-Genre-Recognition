# Music Genre Classification
This project aims to determine the best **Machine Learning** or **Deep Learning model** for classifying a music track into one of the following eight genres:
Folk, Rock, International, Experimental, Hip-Hop, Pop, Electronic, Instrumental.
Each track is assigned to only one genre, ensuring mutual exclusivity.

The data used for this project comes from the **Free Music Archive (FMA)**, a platform that offers a large collection of free and legal music.
The **FMA** provides tracks spanning multiple genres, making it an excellent resource for building a genre classification model. It is widely used in research and creative projects due to its diverse music catalog and availability under legal and open licenses, such as Creative Commons.

## Project Approach
To tackle the classification problem, two distinct strategies were explored:
### 1. Classic Audio Feature-Based Approach
In this approach, traditional features were extracted from the raw audio signal, commonly used in harmonic wave analysis, such as:
* Spectral centroid
* Zero-crossing rate
* Mel-frequency cepstral coefficients (MFCCs)
* Chroma features

These features were used as input to various Machine Learning models, including:
* Random Forests
* Support Vector Machines (SVMs)
* Gradient Boosting Decision Trees

### 2. Neural Network Approach
In this approach, Deep Learning techniques were employed by working directly with spectrograms of the audio tracks. Spectrograms, which visually represent the frequency spectrum of the audio signal over time, were treated as image data.

To classify the genres, a hybrid Neural Network was implemented, combining:
* **Convolutional Neural Networks (CNNs)**: For feature extraction from the spectrogram images.
* **Recurrent Neural Networks (RNNs)**: For capturing temporal patterns in the audio.

## Data
All metadata and features for all tracks are distributed in **[`fma_metadata.zip`]** (342 MiB), which includes:
* `tracks.csv`: per track metadata such as ID, title, artist, genres, tags and play counts, for all 106,574 tracks.
* `genres.csv`: all 163 genres with name and parent (used to infer the genre hierarchy and top-level genres).
* `features.csv`: common features extracted with [librosa].
* `echonest.csv`: audio features provided by [Echonest] (now [Spotify]) for a subset of 13,129 tracks.

[pandas]:   https://pandas.pydata.org/
[librosa]:  https://librosa.org/
[spotify]:  https://www.spotify.com/
[echonest]: https://web.archive.org/web/20170519050040/http://the.echonest.com/

Then, audio data is available in various sizes of MP3-encoded datasets:

1. **[`fma_small.zip`]**: 8,000 tracks of 30s, 8 balanced genres (GTZAN-like) (7.2 GiB)
2. **[`fma_medium.zip`]**: 25,000 tracks of 30s, 16 unbalanced genres (22 GiB)
3. **[`fma_large.zip`]**: 106,574 tracks of 30s, 161 unbalanced genres (93 GiB)
4. **[`fma_full.zip`]**: 106,574 untrimmed tracks, 161 unbalanced genres (879 GiB)

For this project, the **[`fma_small.zip`]** dataset was used.

[`fma_metadata.zip`]: https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
[`fma_small.zip`]:    https://os.unil.cloud.switch.ch/fma/fma_small.zip
[`fma_medium.zip`]:   https://os.unil.cloud.switch.ch/fma/fma_medium.zip
[`fma_large.zip`]:    https://os.unil.cloud.switch.ch/fma/fma_large.zip
[`fma_full.zip`]:     https://os.unil.cloud.switch.ch/fma/fma_full.zip

## Code
This repository contains a series of Jupyter notebooks that guide you through the different steps of the project. 
Below is a summary of the purpose and content of each notebook:

* `1_Feature_analysis.ipynb`: Provides an in-depth analysis of traditional audio features extracted from the raw audio signal, such as spectral and harmonic properties.
* `2_ML_models.ipynb`: Covers the creation of the dataset and the data preprocessing pipeline (including feature selection, extraction, and dimensionality reduction). Demonstrates the application of various Machine Learning models to classify music genres.
* `3_Spectrograms_overview.ipynb`: Focuses on the generation and visualization of spectrograms, creating a dataset of spectrogram images for Deep Learning experiments.
* `4_RNN_CNN.ipynb`: Implements a Deep Learning approach using a combination of Recurrent Neural Networks (RNNs) and Convolutional Neural Networks (CNNs) to classify music genres based on spectrogram data.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/Giovanni-Caruso/Music-Genre-Recognition  
    cd Music-Genre-Recognition
    ```

2. Create a Python 3.9.19 environment:
    ```bash
    # with Anaconda
    conda create -n music_genre python=3.9.19
    conda activate music_genre
    ```
  
3. Install dependencies:
    ```bash
    pip install -r requirements.txt  
    ```

4. Download and Prepare Data:
    * Download the following datasets:
        * **[`fma_small.zip`]**
        * **[`fma_metadata.zip`]**
    * Move the downloaded files to the <code>**data**</code> folder
    * Extract the contents of the <code>**.zip**</code> files to create the following folders:
        * <code>**fma_small**</code>
        * <code>**fma_metadata**</code>
    
    **Note:** Ensure the folder structure is consistent for the scripts to locate the data correctly.
