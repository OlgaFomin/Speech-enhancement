from scipy.io.wavfile import read
import numpy as np
from math import log10, sqrt



def main():
    noisy_voice         = read("/home/fomin@ef.technion.ac.il/Speech-enhancement/data/Test/sound/noisy_voice_long.wav")
    voice               = read("/home/fomin@ef.technion.ac.il/Speech-enhancement/data/Test/sound/voice_long.wav")
    denoised_HUBER_N2C  = read("/home/fomin@ef.technion.ac.il/Speech-enhancement/data/save_predictions/denoised_HUBER_N2C.wav")
    denoised_L2_N2C     = read("/home/fomin@ef.technion.ac.il/Speech-enhancement/data/save_predictions/denoised_L2_N2C.wav")
    denoised_HUBER_N2N  = read("/home/fomin@ef.technion.ac.il/Speech-enhancement/data/save_predictions/denoised_HUBER_N2N.wav")
    denoised_L2_N2N     = read("/home/fomin@ef.technion.ac.il/Speech-enhancement/data/save_predictions/denoised_L2_N2N.wav")


    np_noisy_voice          = np.array(noisy_voice[1],dtype=float)
    np_voice                = np.array(voice[1],dtype=float)
    np_denoised_HUBER_N2C   = np.array(denoised_HUBER_N2C[1],dtype=float)
    np_denoised_L2_N2C      = np.array(denoised_L2_N2C[1],dtype=float)
    np_denoised_HUBER_N2N   = np.array(denoised_HUBER_N2N[1],dtype=float)
    np_denoised_L2_N2N      = np.array(denoised_L2_N2N[1],dtype=float)

    PSNR_HUBER_N2C = PSNR(np_voice, np_denoised_HUBER_N2C)
    PSNR_L2_N2C = PSNR(np_voice, np_denoised_L2_N2C)
    PSNR_HUBER_N2N = PSNR(np_voice, np_denoised_HUBER_N2N)
    PSNR_L2_N2N = PSNR(np_voice, np_denoised_L2_N2N)

    print('PSNR_HUBER_N2C' , PSNR_HUBER_N2C)
    print('PSNR_L2_N2C' , PSNR_L2_N2C)
    print('PSNR_HUBER_N2N' , PSNR_HUBER_N2N)
    print('PSNR_L2_N2N' , PSNR_L2_N2N)


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = max(compressed)
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


if __name__ == "__main__":
    main()
