#%%
import torch
import os
import pandas as pd
from torch.utils.data import random_split
from src.utils import SoundDS, save_graphic_metrics
from src.model import AudioClassifier
from src.training import Trainer
from src.inference import classify

# ----------------------------
# Configuration Parameters
# ----------------------------
metadata_path = "UrbanSound8K/metadata/UrbanSound8K.csv"
audio_dir = './UrbanSound8K/audio'
save_dir = "./checkpoint"
os.makedirs(save_dir, exist_ok=True)
model_state_version = str(4)
batch_size = 16
num_epochs = 100
early_stopping_criteria = 20
learning_rate = 0.001

# ----------------------------
# Load and Preprocess Metadata
# ----------------------------
df = pd.read_csv(metadata_path)
df['relative_path'] = '/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)
df = df[['relative_path', 'classID']]

# ----------------------------
# Create Training and Validation Datasets
# ----------------------------
myds = SoundDS(df, audio_dir)
num_items = len(myds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(myds, [num_train, num_val])

train_dl = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size, shuffle=False)

# ----------------------------
# Initialize Model and Move to Device
# ----------------------------
myModel = AudioClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)

# ----------------------------
# Training Configuration
# ----------------------------
trainer = Trainer(
    model=myModel, 
    device=device,
    early_stopping_criteria=early_stopping_criteria, 
    model_state_version=model_state_version,
    save_dir=save_dir, 
    lr=learning_rate
)

# Start training
trainer.training_and_validation(train_dl=train_dl, val_dl=val_dl, num_epochs=num_epochs, device=device)

# ----------------------------
# Save Training Metrics
# ----------------------------
try:
    save_graphic_metrics(save_dir, model_state_version)
except FileNotFoundError as e:
    print(e)


