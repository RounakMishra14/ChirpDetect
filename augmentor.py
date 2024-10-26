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
