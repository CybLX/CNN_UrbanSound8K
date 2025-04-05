#%%
'''
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Audio File Analysis ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
'''

from src.utils import AudioUtil
from pathlib import Path
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
wav1 = './UrbanSound8K/audio/fold1/101415-3-0-2.wav'
wav2 = './UrbanSound8K/audio/fold2/4201-3-3-0.wav'

audio_file = wav1
class_id = 2
aud = AudioUtil.open(audio_file)

newsr = 44100
newchannel = 2
newduration = 4000
shift_pct = 0.4

# Plot original audio
plt.figure(figsize=(14,5))
plt.ylabel('Amplitude')
librosa.display.waveshow(np.asarray(aud[0][0]), sr=aud[1])
plt.title("Original Audio")
plt.show()

#%%
# Standardize sampling rate
reaud = AudioUtil.resample(aud, newsr)
print(f'Original Sampling Rate: {aud[1]} -> Transformed Sampling Rate: {reaud[1]}')


# Plot resampled audio
plt.figure(figsize=(14,5))
plt.ylabel('Amplitude')
librosa.display.waveshow(np.asarray(reaud[0][0]), sr=reaud[1])
plt.title("Resampled Audio")
plt.show()

#%%
# Standardize number of channels
rechan = AudioUtil.rechannel(reaud, newchannel)
print(f'Original Channels: {len(aud[0])} -> Transformed Channels: {len(rechan[0])}')


# Plot re-channeled audio
plt.figure(figsize=(14,5))
plt.ylabel('Amplitude')
librosa.display.waveshow(np.asarray(rechan[0]), sr=rechan[1])
plt.title("Re-channeled Audio")
plt.show()

#%%
# Standardize sample duration
dur_aud = AudioUtil.pad_trunc(rechan, newduration)
print(f'Original Duration: {len(rechan[0][0])} samples -> Transformed Duration: {len(dur_aud[0][0])} samples')


# Plot duration adjusted audio
plt.figure(figsize=(14,5))
plt.ylabel('Amplitude')
librosa.display.waveshow(np.asarray(dur_aud[0]), sr=dur_aud[1])
plt.title("Duration Adjusted Audio")
plt.show()

#%%
# Apply time shift
shift_aud = AudioUtil.time_shift(dur_aud, shift_pct)
print(f'Time Shift Applied: {shift_pct * 100}%')


# Plot time-shifted audio
plt.figure(figsize=(14,5))
plt.ylabel('Amplitude')
librosa.display.waveshow(np.asarray(shift_aud[0]), sr=shift_aud[1])
plt.title("Time-Shifted Audio")
plt.show()

#%%
# Generate MEL spectrogram
sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)

# Plot spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(np.asarray(sgram[0]), sr=shift_aud[1], x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("MEL Spectrogram")
plt.show()

#%%
# Apply Frequency Mask and Time Mask
aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

# Plot augmented spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(np.asarray(aug_sgram[0]), sr=shift_aud[1], x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("Augmented MEL Spectrogram")
plt.show()

'''
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Audio File Analysis Completed ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
'''

# %%
