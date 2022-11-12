import librosa
import numpy as np
import soundfile as sf
import torch
from PIL import Image
import pyloudnorm as pyln

STANDARD_LUFS = -23.0

def extract(filename, sr=None, energy = 1.0, hop_length = 64, state = None):
    """
        Extracts spectrogram from an input audio file
        Arguments:
            filename: path of the audio file
            n_fft: length of the windowed signal after padding with zeros.
    """
    data, sr = librosa.load(filename, sr=sr)
    data *= energy

    ##Normalizing to standard -23.0 LuFS
  #  meter = pyln.Meter(sr)
   # loudness = meter.integrated_loudness(data)
    #data = pyln.normalize.loudness(data, loudness, target_loudness = STANDARD_LUFS)
    ##################################################

    comp_spec = librosa.stft(data, n_fft=256, hop_length = hop_length, window='hamming')

    mag_spec, phase = librosa.magphase(comp_spec)

    phase_in_angle = np.angle(phase)
    return mag_spec, phase_in_angle, sr

def power_to_db(mag_spec):
    return librosa.power_to_db(mag_spec)

def db_to_power(mag_spec):
    return librosa.db_to_power(mag_spec)

def denorm_and_numpy(inp_tensor):
    inp_tensor = inp_tensor[0, :, :, :] #drop batch dimension
    inp_tensor = inp_tensor.permute((1, 2, 0)) #permute the tensor from C x H x W to H x W x C (numpy equivalent)
    inp_tensor = ((inp_tensor * 0.5) + 0.5) * 255 #to get back from transformation
    inp_tensor = inp_tensor.cpu().numpy().astype(np.uint8) #generating Numpy ndarray
    return inp_tensor

def getTimeSeries(im, img_path, pow, energy = 1.0, state = None):
    mag_spec, phase, sr = extract(img_path[0], 8000, energy, state = state)
    log_spec = power_to_db(mag_spec)

    h, w = mag_spec.shape
    
    ######Ignoring padding
    fix_w = 128
    mod_fix_w = w % fix_w
    extra_cols = 0
    if(mod_fix_w != 0):
        extra_cols = fix_w - mod_fix_w
        im = im[:, :-extra_cols]
    #########################
    print("im shape (ex. padding) = ", im.shape)
    print("spec shape (original) = ", mag_spec.shape)

    _min, _max = log_spec.min(), log_spec.max()

    if(len(im.shape) > 2):
        im = np.mean(im, axis=2)
    im = np.flip(im, axis=0)

    im = unscale_minmax(im, float(_min), float(_max), 0, 255)
    spec = db_to_power(im)
    spec = np.power(spec, 1. / pow)

    return reconstruct(spec, phase)/energy, sr

def reconstruct(mag_spec, phase):
    """
        Reconstructs frames from a spectrogram and phase information.
        Arguments:
            mag_spec: Magnitude component of a spectrogram
            phase:  Phase info. of a spectrogram
    """
    temp = mag_spec * np.exp(phase * 1j)
    data_out = librosa.istft(temp)
    return data_out

# to convert the spectrogram ( an 2d-array of real numbers) to a storable form (0-255)
def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled, X.min(), X.max()


# to get the original spectrogram ( an 2d-array of real numbers) from an image form (0-255)
def unscale_minmax(X, X_min, X_max, min=0.0, max=1.0):
    X = X.astype(np.float)
    X -= min
    X /= (max - min)
    X *= (X_max - X_min)
    X += X_min
    return X

def to_rgb(im, chann):  # converting the image into 3-channel for singan
    if(chann == 1):
        return im
    w, h = im.shape
    ret = np.empty((w, h, chann), dtype=np.uint8)
    ret[:, :, 0] = im
    for i in range(1, chann):
        ret[:, :, i] = ret[:, :, 0]
    return ret
