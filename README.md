# 🔊 Urban Sound Classifier - Deep Learning with PyTorch

This project implements a full pipeline for preprocessing, modeling, and metric visualization for **urban sound classification** using **PyTorch** and the **UrbanSound8K** dataset. The pipeline handles audio files, applies `data augmentation` techniques, and converts data into MEL spectrograms ready to feed into a CNN.

---

## 📁 Project Structure

```bash
.
├── checkpoint/                     # Checkpoints with models, metrics, scheduler and optimizer
│   └── train_and_val_metrics.png   # Plot of accuracy and loss
├── data analysis/                  # Notebook or scripts with data preprocessing analysis
├── src/                            # Source code
│   ├── inference.py                # Inference routine for trained models
│   ├── model.py                    # CNN architecture
│   ├── training.py                 # Training loop, validation, early stopping
│   └── utils.py                    # Dataset class, preprocessing functions
├── UrbanSound8K/                   # Dataset folder
├── ForPrediction.py                # Script for inference on new audio files
├── UrbanSound_Training.py          # Main training routine
└── README.md
```

---

## 📚 Objective

The goal of this project is to develop a classifier for **urban sounds** using **convolutional neural networks (CNNs)** with **PyTorch**. Sounds are extracted from the **UrbanSound8K** dataset, and the system can identify noise such as car horns, dog barks, sirens, and more, based on MEL spectrograms. The pipeline is complete: from raw audio loading to CNN training and results visualization.

---

## 🔄 Preprocessing Pipeline

- 🎵 Reading `.wav` files using `torchaudio`
- 🔁 Transformation into **MelSpectrogram** with configurable parameters
- 🧪 Application of **SpecAugment** (time and frequency masking)
- 🔢 Normalization of spectrogram data
- 🏷️ Conversion to tensors and label pairing

Each sample is standardized to a fixed input size, making CNN training consistent.

---

## 📦 Custom Dataset

Custom dataset based on `torch.utils.data.Dataset` and `UrbanSound8K`:

- Reads from `metadata/UrbanSound8K.csv`
- Uses the `fold` column to split training and validation sets
- Lazy loading of `.wav` files
- Spectrogram normalization and caching
- Conditional `data augmentation` only during training

---

## 🏋️ Training

Model training is handled by the `Trainer` class, which includes:

- ✅ Support for **early stopping** and automatic **checkpoints**
- 📉 Calculation of metrics like loss, accuracy, recall, and F1
- 📝 Logs saved in `.json` formats
- 📊 Automatic plotting of training curves (loss and accuracy)
- 🧪 Validation at the end of each epoch

CNN architecture includes:

- 🔹 4 convolutional blocks with `BatchNorm`, `ReLU`, `Dropout`
- 🔹 `MaxPooling` between blocks
- 🔹 `Flatten` + fully connected layers
- 🔹 Final `Softmax` layer for 10-class classification

---

## 📊 Metrics Visualization

Automatic visualizations after training:

- Metric logs saved per epoch as CSV files
- Visualization script in `utils/visualization.py`
- Charts for:
  - 🎯 Accuracy and loss per epoch
  - 🔄 Execution time per epoch

---

## 🎼 Dataset

Using the **UrbanSound8K** dataset:

- 🔊 **8732 audio files** (`.wav`)
- 🏷️ **10 classes of urban sounds** (e.g., siren, bark, car horn)
- 📁 **Split into 10 folders** (`fold1` to `fold10`)
- 🗂️ Metadata file `metadata/UrbanSound8K.csv` contains:
  - `slice_file_name`
  - `fold`
  - `classID`

🔗 **Download link**: [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html)

---

## 📁 Requirements

Install the requirements with:

```bash
pip install -r requirements.txt
```

Key libraries:
- torch
- torchaudio
- scikit-learn
- matplotlib
- tqdm
- pandas
- numpy

---

## 📌 References

- [UrbanSound8K Dataset](https://urbansounddataset.weebly.com/urbansound8k.html)  
- [SpecAugment: Data Augmentation for ASR](https://arxiv.org/abs/1904.08779)  
- [PyTorch](https://pytorch.org/)  
- [Torchaudio Docs](https://pytorch.org/audio/stable/index.html)  
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)  
- [Audio Deep Learning Made Simple](https://medium.com/data-science/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5) - Ketan Doshi

---

## 👨‍💻 Author

Developed by **Lucas Alves**  
📧 Email: [alves_lucasoliveira@usp.br](mailto:alves_lucasoliveira@usp.br)  
🐙 GitHub: [cyblx](https://github.com/cyblx)  
💼 LinkedIn: [cyblx](https://www.linkedin.com/in/cyblx)
