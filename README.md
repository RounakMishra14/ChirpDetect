# ChirpDetect

## About the Project

**ChirpDetect** is a cool project that uses machine learning to help identify bird species from audio recordings. The main idea is to take bird sounds, process them, and classify them into specific bird species. This project is designed for birdwatchers, researchers, and even casual nature enthusiasts who want to know what birds are singing around them. It combines several key features:

- **Audio Augmentation**: We didn’t just rely on the original bird recordings. By using methods like pitch shifting, time stretching, and adding noise, we made the dataset more diverse and robust for training.
  
- **Spectrogram Preprocessing**: Instead of working directly with raw audio, we convert it into spectrograms (visual representations of sound frequencies over time). These act as images for our CNN model to analyze.

- **CNN Model for Classification**: We built a Convolutional Neural Network (CNN) to classify bird sounds. The model learns from the spectrograms and predicts which bird species is calling. 

- **Streamlit Web App**: The project includes a simple web app built with Streamlit, where users can upload their own audio recordings. The app then processes the audio and gives back the predicted bird species.

- **Data Visualization**: To make results easy to interpret, the app shows the classification results in tables and pie charts (using Plotly), so you can see the density and types of birds in the audio clip at a glance.



## Table of Contents

1. [About the Project](#about-the-project)
2. [Tech Stack](#tech-stack)
3. [Key Features](#key-features)
4. [Installation](#installation)
5. [Usage](#usage)
    - [Running the Streamlit App](#running-the-streamlit-app)
    - [Uploading an Audio File](#uploading-an-audio-file)
    - [Interpreting the Results](#interpreting-the-results)
6. [Model Training](#model-training)
    - [Dataset Augmentation](#dataset-augmentation)
    - [Spectrogram Preprocessing](#spectrogram-preprocessing)
    - [CNN Model Architecture](#cnn-model-architecture)
7. [Visualization](#visualization)
8. [Contributing](#contributing)
9. [License](#license)
10. [Contact](#contact)


## Tech Stack

### Languages and Frameworks:
- **Python**: Core programming language for model training and data preprocessing.
- **TensorFlow**: Used for building and training the Convolutional Neural Network (CNN) model.
- **Streamlit**: Framework to create the interactive web app for bird sound classification.
- **Librosa**: Library for audio processing and feature extraction.
- **Plotly**: For creating interactive visualizations, including pie charts for bird classification results.

### Libraries and Tools:
- **NumPy**: Efficient handling of numerical data and arrays.
- **Pandas**: For managing and presenting classification data in a structured table format.
- **Matplotlib**: Used for plotting training and validation curves during model development.
- **SciPy**: Utility functions for audio processing.
- **Soundfile**: To handle reading and writing audio files in different formats.
  
### Environment:
- **Jupyter Notebooks**: For model training and experimentation.
- **Streamlit Cloud**: To deploy and run the bird sound classifier app.
  
### Miscellaneous:
- **Git**: For version control and collaboration.
- **GitHub**: Repository hosting for the project and collaboration.


## Key Features

- **Real-time Bird Classification**: Classifies and identifies bird species from uploaded audio clips, providing instant results.

- **Multi-Class Support**: Capable of recognizing various bird classes, including collared dove, Indian myna, kingfisher, nightingale, owl, sparrow, along with noise and unknown sounds.

- **Interactive Web Application**: Built with Streamlit, offering a user-friendly interface to upload audio files and visualize results.

- **Audio Segmentation**: Automatically segments long audio files into manageable clips for classification, ensuring efficient processing.

- **Visual Analytics**: Displays classification results in an intuitive table format, highlighting the counts of each detected bird class.

- **Dynamic Pie Charts**: Utilizes Plotly to create interactive pie charts that illustrate the density of detected bird classes in the audio clip.

- **Robust Model Training**: Employs a Convolutional Neural Network (CNN) trained on augmented audio data for improved accuracy in bird sound classification.

- **Cross-Platform Compatibility**: Accessible from various devices, including desktops and mobile devices, for seamless user experience.

**Installation** 

```markdown
## Installation

To set up the **ChirpDetect** project on your local machine, follow these steps:

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.6 or higher
- pip (Python package installer)

### Step 1: Clone the Repository

First, clone the repository to your local machine using the following command:

```bash
git clone https://github.com/yourusername/ChirpDetect.git
```

### Step 2: Navigate to the Project Directory

Change to the project directory:

```bash
cd ChirpDetect
```

### Step 3: Create a Virtual Environment (Optional)

It is recommended to create a virtual environment to manage your project dependencies:

```bash
python -m venv venv
```

Activate the virtual environment:

- On Windows:

  ```bash
  venv\Scripts\activate
  ```

- On macOS/Linux:

  ```bash
  source venv/bin/activate
  ```

### Step 4: Install Required Packages

Install the necessary packages using pip:

```bash
pip install -r requirements.txt
```

### Step 5: Run the Application

Finally, run the Streamlit application:

```bash
streamlit run app.py
```

Your default web browser should open with the **ChirpDetect** application running.
```



