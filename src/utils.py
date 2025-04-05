import random
import torch
import torchaudio
from torchaudio import transforms
from torch.utils.data import Dataset
from typing import Tuple
import os
import json
import matplotlib.pyplot as plt
import matplotlib


# ----------------------------
# Utilities
# ----------------------------
class AudioUtil():
    """
    Utility class for audio preprocessing.
    Provides methods for loading, resampling, rechanneling,
    padding/truncating, time shifting, and augmenting audio data.
    """
    
    @staticmethod
    def open(audio_file: str) -> tuple[torch.Tensor, int]:
        """
        Load an audio file.

        Parameters
        ----------
        audio_file : str
            Path to the audio file.

        Returns
        -------
        Tuple[torch.Tensor, int]
            Audio signal and sample rate.
        """
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)

    # ------------------------------------
    # Adjust audio to the specified number of channels
    # ------------------------------------
    @staticmethod
    def resample(aud: tuple[torch.Tensor, int], newsr: int) -> tuple[torch.Tensor, int]:
        """
        Resample the audio to a new sample rate.

        Parameters
        ----------
        aud : Tuple[torch.Tensor, int]
            Audio signal and sample rate.
        newsr : int
            Desired sample rate.

        Returns
        -------
        Tuple[torch.Tensor, int]
            Resampled audio signal and new sample rate.
        """

        sig, sr = aud

        if (sr == newsr):
          # Nothing to do
          return aud

        num_channels = sig.shape[0]
        # Resample first channel
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
        if (num_channels > 1):
          # Resample the second channel and merge both channels
          retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
          resig = torch.cat([resig, retwo])

        return ((resig, newsr))

    # ----------------------------
    # Resample each channel individually, as Resample operates on a single channel
    # ----------------------------
    @staticmethod
    def rechannel(aud: tuple[torch.Tensor, int], new_channel: int) -> tuple[torch.Tensor, int]:
        """
        Convert audio to the specified number of channels.

        Parameters
        ----------
        aud : Tuple[torch.Tensor, int]
            Audio signal and sample rate.
        new_channel : int
            Desired number of channels.

        Returns
        -------
        Tuple[torch.Tensor, int]
            Audio signal with the specified number of channels.
        """

        sig, sr = aud

        if (sig.shape[0] == new_channel):
          # Nothing to do
          return aud

        if (new_channel == 1):
          # Convert from stereo to mono by selecting only the first channel
          resig = sig[:1, :]
        else:
          # Convert from mono to stereo by duplicating the first channel
          resig = torch.cat([sig, sig])

        return ((resig, sr))
    

    # ----------------------------
    # Adjust the signal length to a fixed duration 'max_ms' (milliseconds) [PAD]
    # ----------------------------
    @staticmethod
    def pad_trunc(aud: tuple[torch.Tensor, int], max_ms: int) -> tuple[torch.Tensor, int]:
        """
        Pad or truncate the audio signal to a fixed duration.

        Parameters
        ----------
        aud : Tuple[torch.Tensor, int]
            Audio signal and sample rate.
        max_ms : int
            Maximum duration in milliseconds.

        Returns
        -------
        Tuple[torch.Tensor, int]
            Processed audio signal and sample rate.
        """

        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms

        if (sig_len > max_len):
          # Truncate the signal to the given length
          sig = sig[:,:max_len]

        elif (sig_len < max_len):
          # Length of padding to add at the beginning and end of the signal
          pad_begin_len = random.randint(0, max_len - sig_len)
          pad_end_len = max_len - sig_len - pad_begin_len

          # Pad with 0s
          pad_begin = torch.zeros((num_rows, pad_begin_len))
          pad_end = torch.zeros((num_rows, pad_end_len))

          sig = torch.cat((pad_begin, sig, pad_end), 1)

        return (sig, sr)



    # ----------------------------
    # Apply time shift to the signal, moving it left or right by a percentage.
    # Values at the edges are wrapped around to maintain continuity.
    # ----------------------------    
    @staticmethod
    def time_shift(aud: tuple[torch.Tensor, int], shift_limit: float) -> tuple[torch.Tensor, int]:
        """
        Apply time shift augmentation.

        Parameters
        ----------
        aud : Tuple[torch.Tensor, int]
            Audio signal and sample rate.
        shift_limit : float
            Maximum percentage of shift.

        Returns
        -------
        Tuple[torch.Tensor, int]
            Time-shifted audio signal and sample rate.
        """

        sig,sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)

    # ----------------------------
    # Generate a spectrograms on the MEL scale
    # ----------------------------
    @staticmethod
    def spectro_gram(aud: tuple[torch.Tensor, int], n_mels: int = 64, n_fft: int = 1024, hop_len: int = None) -> torch.Tensor:
        """
        Generate a MEL spectrogram.
        
        Parameters
        ----------
        aud : Tuple[torch.Tensor, int]
            Audio signal and sample rate.
        n_mels : int
            Number of MEL filter banks.
        n_fft : int
            FFT window size.
        hop_len : int
            Hop length.
        
        Returns
        -------
        torch.Tensor
            MEL spectrogram.
        """

        sig,sr = aud
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)


    # ----------------------------
    # Apply SpecAugment by masking sections of the spectrogram in both frequency 
    # (horizontal bars) and time (vertical bars) dimensions. This prevents overfitting 
    # and improves model generalization. Masked areas are replaced with the mean value.
    # ----------------------------
    @staticmethod
    def spectro_augment(spec: torch.Tensor, max_mask_pct: float = 0.1, n_freq_masks: int = 1, n_time_masks: int = 1) -> torch.Tensor:
        """
        Apply SpecAugment (frequency and time masking) on the spectrogram.
        
        Parameters
        ----------
        spec : torch.Tensor
            Spectrogram.
        max_mask_pct : float
            Maximum percentage of masking.
        n_freq_masks : int
            Number of frequency masks.
        n_time_masks : int
            Number of time masks.
        
        Returns
        -------
        torch.Tensor
            Augmented spectrogram.
        """

        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
          aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
          aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec

# Loader dos audios para pre processamento


# ----------------------------
# Dataset
# ----------------------------
class SoundDS(Dataset):
    """
    Custom dataset for loading and processing audio files.
    """

    def __init__(self, df, data_path: str) -> None:
        """
        Initialize the dataset.

        Parameters
        ----------
        df : DataFrame
            DataFrame containing file paths and labels.
        data_path : str
            Base directory path where audio files are stored.
        """

        self.df = df
        self.data_path = str(data_path)
        self.duration = 4000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4

    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self) -> int:
        """
        Get the number of items in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return len(self.df)    

    # ----------------------------
    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get the i-th item in the dataset.

        Parameters
        ----------
        idx : int
            Index of the item.

        Returns
        -------
        Tuple[torch.Tensor, int]
            Augmented spectrogram and class ID.
        """

        # Absolute file path of the audio file - concatenate the audio directory with
        # the relative path
        audio_file = self.data_path + self.df.loc[idx, 'relative_path']
        # Get the Class ID
        class_id = self.df.loc[idx, 'classID']

        aud = AudioUtil.open(audio_file)
        # Some sounds have a higher sample rate, or fewer channels compared to the
        # majority. So make all sounds have the same number of channels and same 
        # sample rate. Unless the sample rate is the same, the pad_trunc will still
        # result in arrays of different lengths, even though the sound duration is
        # the same.
        #Padroniza a taxa de amostragem
        reaud = AudioUtil.resample(aud, self.sr)
        #Padroniza o numero de canais
        rechan = AudioUtil.rechannel(reaud, self.channel)
        #Padroniza o tamanho da amostra
        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        #Aplica o deslocamento no tempo
        shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
        #Retira o espectograma MEL
        sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        #Aplica a Frequency mask e Time mask
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        return aug_sgram, class_id
    
# ----------------------------
# Save Metrics Graphs
# ----------------------------
def save_graphic_metrics(save_dir: str, model_state_version: str) -> None:
    """
    Carrega o estado de treinamento salvo e gera gráficos de métricas.

    Parameters
    ----------
    save_dir : str
        Diretório onde o arquivo JSON está salvo e onde o gráfico será salvo.
    model_state_version : str
        Nome da versão do modelo, usado para localizar o arquivo JSON.

    Returns
    -------
    None
        Salva um gráfico com as métricas de treinamento e validação.
    """

    json_filename = f"{model_state_version}_train_state.json"
    filepath = os.path.join(save_dir, json_filename)

    # Verifica se o arquivo existe
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

    # Carrega o estado de treinamento
    with open(filepath, "r") as f:
        train_state = json.load(f)

    epochs = range(1, len(train_state["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Plot da perda (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_state["train_loss"], "bo-", label="Train Loss")
    plt.plot(epochs, train_state["val_loss"], "ro-", label="Validation Loss")
    plt.xlabel("Época")
    plt.ylabel("Perda")
    plt.title("Evolução da Perda")
    plt.legend()

    # Plot da acurácia (Accuracy) - Verifica se existem os dados
    if "train_accuracy" in train_state and "val_accuracy" in train_state:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_state["train_accuracy"], "bo-", label="Train Accuracy")
        plt.plot(epochs, train_state["val_accuracy"], "ro-", label="Validation Accuracy")
        plt.xlabel("Época")
        plt.ylabel("Acurácia")
        plt.title("Evolução da Acurácia")
        plt.legend()

    plt.tight_layout()

    # Salvar o gráfico
    plot_filename = os.path.join(save_dir, "train_and_val_metrics.png")
    plt.savefig(plot_filename)
    plt.show()
    plt.close()

    print(f"\nGráfico salvo: {plot_filename}", flush=True)
