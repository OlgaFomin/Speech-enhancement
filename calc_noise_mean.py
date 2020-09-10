from args import parser
import os
from data_tools import audio_files_to_numpy
import numpy as np

def main():

    args = parser.parse_args()
    noise_dir = args.noise_dir
    #folder containing clean voices
    voice_dir = args.voice_dir
    #path to save time series
    path_save_time_serie = args.path_save_time_serie
    #path to save sounds
    path_save_sound = args.path_save_sound
    #path to save spectrograms
    path_save_spectrogram = args.path_save_spectrogram
    # Sample rate to read audio
    sample_rate = args.sample_rate
    # Minimum duration of audio files to consider
    min_duration = args.min_duration
    #Frame length for training data
    frame_length = args.frame_length
    # hop length for clean voice files
    hop_length_frame = args.hop_length_frame
    # hop length for noise files
    hop_length_frame_noise = args.hop_length_frame_noise
    # How much frame to create for training
    nb_samples = args.nb_samples
    #nb of points for fft(for spectrogram computation)
    n_fft = args.n_fft
    #hop length for fft
    hop_length_fft = args.hop_length_fft

    check_noise_mean(noise_dir, voice_dir, path_save_time_serie, path_save_sound, path_save_spectrogram, sample_rate, min_duration, frame_length, hop_length_frame, hop_length_frame_noise, nb_samples, n_fft, hop_length_fft)

def check_noise_mean(noise_dir, voice_dir, path_save_time_serie, path_save_sound, path_save_spectrogram, sample_rate,min_duration, frame_length, hop_length_frame, hop_length_frame_noise, nb_samples, n_fft, hop_length_fft):
    """This function will randomly blend some clean voices from voice_dir with some noises from noise_dir
    and save the spectrograms of noisy voice, noise and clean voices to disk as well as complex phase,
    time series and sounds. This aims at preparing datasets for denoising training. It takes as inputs
    parameters defined in args module"""
    print(noise_dir)
    list_noise_files = os.listdir(noise_dir)

    nb_noise_files = len(list_noise_files)


    # Extracting noise and voice from folder and convert to numpy
    noise = audio_files_to_numpy(noise_dir, list_noise_files, sample_rate,
          #print(sum(sum(noise))/len(noise))
    totsum = 0
    for i in range(nb_samples):
        id_noise = np.random.randint(0, noise.shape[0])
        subsum = sum(noise[id_noise, :])/len(noise[id_noise, :])
        print('subsum: ',subsum)
        totsum = totsum + subsum

    print('totsum: ' , totsum)


if __name__ == "__main__":
    main()


                                                                                  
