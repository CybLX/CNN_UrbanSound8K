import os
import torch
import librosa
import numpy as np
from typing import Union
from src.model import AudioClassifier
from src.utils import AudioUtil


# ----------------------------
# Audio Classification Inference
# ----------------------------
def classify(audio_file: str, model_version: Union[int, str], save_dir: str = "./checkpoint") -> torch.Tensor:
    """
    Classifies an audio file using a pre-trained model.

    Parameters:
    -----------
    audio_file : str
        Path to the audio file.
    model_version : Union[int, str]
        Version of the saved model.
    save_dir : str, optional
        Directory where the model checkpoints are stored (default: "./checkpoint").

    Returns:
    --------
    torch.Tensor
        Index of the predicted class by the model.
    """
    class_mapping = {3: 'dog_bark',
                     2: 'children_playing',
                     1: 'car_horn',
                     0: 'air_conditioner',
                     9: 'street_music',
                     6: 'gun_shot',
                     8: 'siren',
                     5: 'engine_idling',
                     7: 'jackhammer',
                     4: 'drilling'}

    # Define the device (GPU if available, otherwise CPU)
    cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using CUDA: {}\n\n".format(torch.cuda.get_device_name()), flush = True)

    if os.path.exists(save_dir):
        weights_path = os.path.join(save_dir, f"{model_version}.pt")


        # Load the model
        model = AudioClassifier.load_checkpoint(path = weights_path, device =  cuda)
        model.eval()  # Set the model to evaluation mode

        # Audio processing
        aud = AudioUtil.open(audio_file)

        # Generate a spectrograms on the MEL scale
        sgram = AudioUtil.spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None)

        # Plot o spectograma MEL
        librosa.display.specshow(np.asarray(sgram)[0], sr=aud[1], x_axis='time', y_axis='mel')
        
        # Convert to 4D tensor and move to the same device as the model
        sgram = torch.tensor(np.asarray(sgram)).unsqueeze(0).to(cuda)
        
        # Apply the model
        outputs = model(sgram)

        # Get the class with the highest probability
        _, prediction = torch.max(outputs,1)
        class_name = class_mapping[prediction.item()]


        print(f"Prediction Class is: {prediction.item()}-{class_name}")
        
        return aud, prediction

    else:
        raise FileNotFoundError(f"The path {save_dir} does not exist.")