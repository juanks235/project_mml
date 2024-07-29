import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
from scipy.io.wavfile import write

def pad_or_truncate_waveform(waveform, desired_length):
    """
    Ajusta la longitud de la señal de audio a la longitud deseada.
    
    Args:
        waveform (numpy.ndarray): Señal de audio.
        desired_length (int): Longitud deseada de la señal.

    Returns:
        numpy.ndarray: Señal de audio ajustada.
    """
    current_length = len(waveform)
    if current_length < desired_length:
        padding = desired_length - current_length
        waveform = np.pad(waveform, (0, padding), 'constant')
    else:
        waveform = waveform[:desired_length]
    return waveform

def generate_spectrogram(waveform, desired_length, frame_length=256, frame_step=128, fft_length=256):
    """
    Genera el espectrograma a partir de una señal de audio.

    Args:
        waveform (numpy.ndarray): Señal de audio.
        desired_length (int): Longitud deseada para la señal de audio.
        frame_length (int): Longitud del marco para STFT.
        frame_step (int): Paso del marco para STFT.
        fft_length (int): Longitud del FFT.

    Returns:
        tuple: Magnitud y fase del espectrograma.
    """
    waveform = pad_or_truncate_waveform(waveform, desired_length)
    f, t, Zxx = stft(waveform, nperseg=frame_length, noverlap=frame_length - frame_step, nfft=fft_length)
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)
    return magnitude, phase

def find_max_length(directory):
    """
    Encuentra la longitud máxima de las señales de audio en un directorio.

    Args:
        directory (str): Ruta al directorio que contiene los archivos .npy.

    Returns:
        int: Longitud máxima encontrada.
    """
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
    """
    Procesa archivos .npy en un directorio, generando espectrogramas y guardándolos como .npy.

    Args:
        directory (str): Ruta al directorio con archivos .npy.
        output_dir (str): Ruta al directorio donde se guardarán los espectrogramas.
        max_length (int): Longitud máxima para truncar o rellenar la señal de audio.
        frame_length (int): Longitud del marco para STFT.
        frame_step (int): Paso del marco para STFT.
        fft_length (int): Longitud del FFT.
    """
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            file_path = os.path.join(directory, filename)
            example = np.load(file_path, allow_pickle=True).tolist()
            waveform = np.array(example["time_series"], dtype=np.float32)
            
            # Generar espectrograma
            magnitude, phase = generate_spectrogram(waveform, max_length, frame_length, frame_step, fft_length)
            
            # Guardar magnitud y fase del espectrograma como archivos .npy
            output_path_magnitude = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_magnitude.npy')
            output_path_phase = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_phase.npy')
            np.save(output_path_magnitude, magnitude)
            np.save(output_path_phase, phase)
            print(f'Se guardaron la magnitud y la fase para {filename} en {output_path_magnitude} y {output_path_phase}')

def spectrogram_to_waveform(magnitude, phase, frame_length=256, frame_step=128, fft_length=256):
    """
    Convierte un espectrograma (magnitud y fase) de vuelta a una señal de audio.

    Args:
        magnitude (numpy.ndarray): Magnitud del espectrograma.
        phase (numpy.ndarray): Fase del espectrograma.
        frame_length (int): Longitud del marco para STFT.
        frame_step (int): Paso del marco para STFT.
        fft_length (int): Longitud del FFT.

    Returns:
        numpy.ndarray: Señal de audio reconstruida.
    """
    # Combinar magnitud y fase para crear el espectrograma complejo
    Zxx = magnitude * np.exp(1j * phase)
    # Realizar STFT inverso para obtener la señal de audio
    _, waveform = istft(Zxx, nperseg=frame_length, noverlap=frame_length - frame_step, nfft=fft_length)
    return waveform

def save_waveform(waveform, save_path, sample_rate=44100):
    """
    Guarda una señal de audio en un archivo .wav.

    Args:
        waveform (numpy.ndarray): Señal de audio.
        save_path (str): Ruta donde se guardará el archivo .wav.
        sample_rate (int): Tasa de muestreo para el archivo .wav.
    """
    # Normalizar la señal de audio al rango de int16
    waveform = waveform / np.max(np.abs(waveform))
    waveform = (waveform * 32767).astype(np.int16)
    write(save_path, sample_rate, waveform)

def convert_npy_to_audio(magnitude_path, phase_path, output_wav_path, frame_length=256, frame_step=128, fft_length=256, sample_rate=44100):
    """
    Convierte archivos .npy de magnitud y fase a un archivo de audio .wav.

    Args:
        magnitude_path (str): Ruta al archivo .npy de magnitud.
        phase_path (str): Ruta al archivo .npy de fase.
        output_wav_path (str): Ruta donde se guardará el archivo .wav.
        frame_length (int): Longitud del marco para STFT.
        frame_step (int): Paso del marco para STFT.
        fft_length (int): Longitud del FFT.
        sample_rate (int): Tasa de muestreo para el archivo .wav.
    """
    # Cargar los archivos .npy de magnitud y fase
    magnitude = np.load(magnitude_path)
    phase = np.load(phase_path)
    
    # Convertir el espectrograma de vuelta a la señal de audio
    waveform = spectrogram_to_waveform(magnitude, phase, frame_length, frame_step, fft_length)
    
    # Guardar la señal de audio en un archivo .wav
    save_waveform(waveform, output_wav_path, sample_rate)
    print(f'Se guardó la señal de audio en {output_wav_path}')

def plot_spectrogram(magnitude, phase, output_image_prefix):
    """
    Grafica y guarda las imágenes del espectrograma de magnitud y fase.

    Args:
        magnitude (numpy.ndarray): Magnitud del espectrograma.
        phase (numpy.ndarray): Fase del espectrograma.
        output_image_prefix (str): Prefijo para el nombre de los archivos de imagen.
    """
    # Graficar la magnitud
    plt.figure(figsize=(12, 6))
    plt.title('Espectrograma de Magnitud')
    plt.imshow(np.log(magnitude + 1e-10).T, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Frames de Tiempo')
    plt.ylabel('Bins de Frecuencia')
    plt.tight_layout()
    plt.savefig(f'{output_image_prefix}_magnitude.png')
    plt.close()
    
    # Graficar la fase
    plt.figure(figsize=(12, 6))
    plt.title('Espectrograma de Fase')
    plt.imshow(phase.T, aspect='auto', origin='lower', cmap='hsv')
    plt.colorbar()
    plt.xlabel('Frames de Tiempo')
    plt.ylabel('Bins de Frecuencia')
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
