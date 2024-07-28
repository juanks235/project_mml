import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
from scipy.io.wavfile import write

def pad_or_truncate_waveform(waveform, desired_length):
    current_length = len(waveform)
    if current_length < desired_length:
        padding = desired_length - current_length
        waveform = np.pad(waveform, (0, padding), 'constant')
    else:
        waveform = waveform[:desired_length]
    return waveform

def generate_spectrogram(waveform, desired_length, frame_length=256, frame_step=128, fft_length=256):
    waveform = pad_or_truncate_waveform(waveform, desired_length)
    f, t, Zxx = stft(waveform, nperseg=frame_length, noverlap=frame_length - frame_step, nfft=fft_length)
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)
    return magnitude, phase

def find_max_length(directory):
    max_length = 0
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            file_path = os.path.join(directory, filename)
            example = np.load(file_path, allow_pickle=True).tolist()
            waveform = np.array(example["time_series"], dtype=np.float32)
            if len(waveform) > max_length:
                max_length = len(waveform)
    return max_length

def process_files(directory, output_dir, max_length, frame_length=256, frame_step=128, fft_length=256):
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            file_path = os.path.join(directory, filename)
            example = np.load(file_path, allow_pickle=True).tolist()
            waveform = np.array(example["time_series"], dtype=np.float32)
            
            # Generate spectrogram
            magnitude, phase = generate_spectrogram(waveform, max_length, frame_length, frame_step, fft_length)
            
            # Save spectrogram magnitude and phase as .npy
            output_path_magnitude = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_magnitude.npy')
            output_path_phase = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_phase.npy')
            np.save(output_path_magnitude, magnitude)
            np.save(output_path_phase, phase)
            print(f'Saved magnitude and phase for {filename} at {output_path_magnitude} and {output_path_phase}')

def spectrogram_to_waveform(magnitude, phase, frame_length=256, frame_step=128, fft_length=256):
    # Combine magnitude and phase to create the complex spectrogram
    Zxx = magnitude * np.exp(1j * phase)
    # Perform inverse STFT to get the waveform back
    _, waveform = istft(Zxx, nperseg=frame_length, noverlap=frame_length - frame_step, nfft=fft_length)
    return waveform

def save_waveform(waveform, save_path, sample_rate=44100):
    # Normalize waveform to the range of int16
    waveform = waveform / np.max(np.abs(waveform))
    waveform = (waveform * 32767).astype(np.int16)
    write(save_path, sample_rate, waveform)

def convert_npy_to_audio(magnitude_path, phase_path, output_wav_path, frame_length=256, frame_step=128, fft_length=256, sample_rate=44100):
    # Load the spectrogram magnitude and phase .npy files
    magnitude = np.load(magnitude_path)
    phase = np.load(phase_path)
    
    # Convert spectrogram back to waveform
    waveform = spectrogram_to_waveform(magnitude, phase, frame_length, frame_step, fft_length)
    
    # Save the waveform as a wav file
    save_waveform(waveform, output_wav_path, sample_rate)
    print(f'Saved waveform to {output_wav_path}')

def plot_spectrogram(magnitude, phase, output_image_prefix):
    # Plot magnitude
    plt.figure(figsize=(12, 6))
    plt.title('Magnitude Spectrogram')
    plt.imshow(np.log(magnitude + 1e-10).T, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')
    plt.tight_layout()
    plt.savefig(f'{output_image_prefix}_magnitude.png')
    plt.close()
    
    # Plot phase
    plt.figure(figsize=(12, 6))
    plt.title('Phase Spectrogram')
    plt.imshow(phase.T, aspect='auto', origin='lower', cmap='hsv')
    plt.colorbar()
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')
    plt.tight_layout()
    plt.savefig(f'{output_image_prefix}_phase.png')
    plt.close()

if __name__ == "__main__":
    # Parameters
    frame_length = 256
    frame_step = 128
    fft_length = 256

    # Directories
    clean_data_dir = '../cleaned_data'
    output_dir = '../spectrogram_npy'
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Find the longest audio length in the dataset
    max_length = find_max_length(clean_data_dir)
    print(f'Max length found: {max_length}')

    # Step 2: Process each .npy file and pad/truncate to max_length
    process_files(clean_data_dir, output_dir, max_length, frame_length, frame_step, fft_length)

    # Example usage of converting a spectrogram .npy back to audio
    ex = '254'
    magnitude_path = f'../spectrogram_npy/{ex}_magnitude.npy'
    phase_path = f'../spectrogram_npy/{ex}_phase.npy'
    output_wav_path = f'reconstructed_{ex}.wav'
    convert_npy_to_audio(magnitude_path, phase_path, output_wav_path)

    # Load the magnitude and phase
    magnitude = np.load(magnitude_path)
    phase = np.load(phase_path)

    # Plot and save the spectrogram images
    plot_spectrogram(magnitude, phase, ex)
