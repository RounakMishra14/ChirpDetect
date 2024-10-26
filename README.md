# ChirpDetect 🐦
**ChirpDetect** is an interactive bird sound classification app built using Streamlit, leveraging Convolutional Neural Networks (CNNs) to provide users with insights into identifying various bird species based on their calls. The app allows users to visualize results, analyze audio input, and classify bird sounds in real-time.

## 📋 Table of Contents
1. [Overview](#overview)
2. [Demo](#demo)
3. [Data Collection Process](#data-collection-process)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Building](#model-building)
6. [App Features](#app-features)

## 📘 Overview
**ChirpDetect** is an interactive bird sound classification app built with Streamlit and powered by Convolutional Neural Networks (CNNs) to help users identify bird species from their calls. The app allows users to upload recordings and then classifies the bird sounds in real time. By leveraging TensorFlow for the machine learning model, Librosa for audio processing, and Plotly for visualization, the app provides a seamless and informative user experience.

The project is built using locally recorded bird call samples from the surrounding area [located here](https://maps.app.goo.gl/45fe6eRNAD3a4nrz6). Due to the limited number of available recordings, **data augmentation** techniques were applied to enhance the dataset. Methods such as time-stretching, pitch-shifting, and adding background noise helped simulate various real-world scenarios. This allowed the model to generalize better and perform effectively, even with the small dataset, by creating a more diverse and robust set of training samples.

## 🌐 Demo
Try the live demo here: [ChirpDetect - Bird Sound Classification App](https://yourdemo.link)

##Here's the **Data Collection** section based on your requirements, including the Wikipedia links and the augmentation details:

---

## 📈 Data Collection Process

To build the bird sound classification model, local bird species were recorded to gather as many samples as possible. The following species, commonly found in the area, were captured:

- [Indian Myna](https://en.wikipedia.org/wiki/Common_myna)
- [Collared Dove](https://en.wikipedia.org/wiki/Eurasian_collared_dove)
- [Kingfisher](https://en.wikipedia.org/wiki/Kingfisher)
- [Owl](https://en.wikipedia.org/wiki/Owl)
- [Sparrow](https://en.wikipedia.org/wiki/House_sparrow)
- [Nightingale](https://en.wikipedia.org/wiki/Common_nightingale)

These recordings were captured from the region [located here](https://maps.app.goo.gl/45fe6eRNAD3a4nrz6). However, due to the limited number of recordings, **data augmentation** techniques were used to increase the size and diversity of the dataset. Augmentation was performed by applying transformations like pitch shifting, time-stretching, and adding background noise, simulating different environments and variations in bird calls.

Here’s the augmentation function used to enhance the dataset:

```python
import os
import librosa
import soundfile as sf
import numpy as np

# Define paths
input_dir = 'processed_data'
output_dir = 'augmented_data'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to augment audio
def augment_audio(audio):
    # Augmentation techniques
    # 1. Pitch shift
    n_steps = np.random.randint(-2, 3)  # Random pitch shift steps
    pitch_shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

    # 2. Time stretch
    stretch_factor = np.random.uniform(0.8, 1.2)  # Time stretch between 80% and 120%
    stretched = librosa.effects.time_stretch(audio, rate=stretch_factor)

    # 3. Add noise
    noise = np.random.randn(len(audio)) * 0.005  # Adjust noise level as needed
    noisy_audio = audio + noise

    return [pitch_shifted, stretched, noisy_audio]

# Iterate through each class folder and augment audio files
for class_folder in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_folder)
    if os.path.isdir(class_path):
        # Create a folder for the class in the output directory
        class_output_path = os.path.join(output_dir, class_folder)
        os.makedirs(class_output_path, exist_ok=True)

        # Initialize a counter for samples
        sample_count = 0

        # Process each audio file in the class folder
        for audio_file in os.listdir(class_path):
            if audio_file.endswith('.wav'):
                input_file_path = os.path.join(class_path, audio_file)
                audio, sr = librosa.load(input_file_path, sr=None)

                # Generate augmented samples until we reach 150 samples
                while sample_count < 150:
                    augmented_samples = augment_audio(audio)

                    # Save each augmented sample
                    for idx, sample in enumerate(augmented_samples):
                        output_file_name = f"{class_folder}_{audio_file.split('.')[0]}_sample{sample_count + 1}_aug{idx + 1}.wav"
                        output_file_path = os.path.join(class_output_path, output_file_name)
                        sf.write(output_file_path, sample, sr)
                        
                    sample_count += len(augmented_samples)

                # Reset sample count for the next file
                sample_count = 0

print("Audio augmentation completed with 150 samples per class.")

```
By augmenting the data, the model is better equipped to handle real-world variations in bird calls, despite having a limited number of original samples.

## 🧹 Data Preprocessing

The preprocessing steps in this project involve converting raw audio data into spectrograms, a visual representation of the audio signals in the frequency domain. Two key functions are used for this purpose:

1. **`load_wav_16k_mono(filename)`**: This function loads the audio file and converts it into a 16 kHz mono-channel format using the `librosa` library. It ensures consistency in the data format by down-sampling and making all audio files single-channel.

2. **`preprocess(file_path)`**: This function takes the audio file path as input, converts it into a spectrogram using Short-Time Fourier Transform (STFT), and ensures it has a fixed size by either padding or cropping.

Here’s the code for these functions:

```python
import librosa
import soundfile as sf

def load_wav_16k_mono(filename):
    # Load the audio file using librosa
    wav, sample_rate = librosa.load(filename, sr=16000, mono=True)
    
    # Convert the wav to a mono channel if it’s not already
    if wav.ndim > 1:
        wav = librosa.to_mono(wav)
        
    return wav

def preprocess(file_path):
    # Load the audio file and convert to a mono channel
    wav = load_wav_16k_mono(file_path)
    
    # Compute the STFT of the audio signal
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    
    # Take the absolute value of the STFT to get the magnitude spectrogram
    spectrogram = tf.abs(spectrogram)
    
    # Add the channel dimension to make the spectrogram 3D (height, width, channels)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)  # Shape becomes (time, frequency, 1)
    
    # Ensure fixed size by padding or truncating (150 time steps, 257 frequency bins, 1 channel)
    desired_shape = (150, 257, 1)
    spectrogram = tf.image.resize_with_crop_or_pad(spectrogram, desired_shape[0], desired_shape[1])
    
    return spectrogram
```

### Example Spectrograms

Below are a few sample spectrograms generated from the bird call audio data:

![Spectrogram 1](example_spectrogram_1.png)
![Spectrogram 2](example_spectrogram_2.png)

The spectrograms represent the intensity of frequencies over time, which are crucial for training the machine learning model in detecting bird species based on their sounds.



