#%%
from src.inference import classify
from IPython.display import Audio

# ----------------------------
# Model Inference
# ----------------------------
# Run inference on trained model with the validation set
model_state_version = "4"
audio_dir = './UrbanSound8K/audio'
save_dir = "./checkpoint"



audio, predict_label = classify(
    audio_file=f'{audio_dir}/fold2/4201-3-3-0.wav', 
    model_version=model_state_version,
    save_dir=save_dir
)

Audio(data=audio[0], rate=audio[1])

# %%
