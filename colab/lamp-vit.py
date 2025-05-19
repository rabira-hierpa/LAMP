# Desert Locust Prediction using Vision Transformer in Google Colab
# Optimized for memory constraints on free tier

# Import libraries
import random
import rasterio.windows
from multiprocessing import Pool
import os
import gc
import numpy as np
import glob
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import drive
import dask.array as da

# Suppress debugger warnings
%env PYDEVD_DISABLE_FILE_VALIDATION = 1

# Mount Google Drive
drive.mount('/content/drive')

# Function to extract label from file name


def get_label_from_filename(file):
    base = os.path.basename(file)
    name = os.path.splitext(base)[0]
    parts = name.split('_')
    try:
        label_index = parts.index('label') + 1
        return int(parts[label_index])
    except (ValueError, IndexError):
        print(f"Error parsing label from {file}")
        return None

# Function to process a single GeoTIFF file with enhanced validation


def process_file(file):
    with rasterio.open(file) as src:
        # Read only the required window (up to 41x41)
        target_h, target_w = 41, 41
        window = rasterio.windows.Window(
            0, 0, min(src.width, target_w), min(src.height, target_h))
        data = src.read(window=window).astype(np.float32)  # Enforce float32

        # Replace NaN/inf with 0
        data = np.where(np.isnan(data) | np.isinf(data), 0, data)

        # Pad if dimensions are less than 41x41
        pad_h = target_h - data.shape[1]
        pad_w = target_w - data.shape[2]
        if pad_h > 0 or pad_w > 0:
            data_padded = np.pad(
                data,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode='constant',
                constant_values=0
            )
        else:
            data_padded = data

        inputs = data_padded[:59]  # Shape: [59, 41, 41]
        label = data_padded[59]    # Shape: [41, 41]

        # Ensure labels are binary (0 or 1)
        label = np.where(label > 0.5, 1, 0).astype(np.float32)

        return inputs, label

# Function to load all data in parallel


def load_all_data_parallel(file_list, num_workers=4):
    with Pool(num_workers) as p:
        results = p.map(process_file, file_list)
    inputs_list, labels_list = zip(*results)
    inputs_array = np.stack(inputs_list).astype(np.float32)  # Enforce float32
    labels_array = np.stack(labels_list).astype(np.float32)
    return inputs_array, labels_array


# Paths for preprocessed data on Google Drive
preprocessed_inputs = '/content/drive/MyDrive/inputs.npy'
preprocessed_labels = '/content/drive/MyDrive/labels.npy'

# Load preprocessed data from Google Drive
print("Loading preprocessed data from Google Drive...")
try:
    # Load with memory mapping to reduce RAM usage
    inputs_array = np.load(preprocessed_inputs,
                           mmap_mode='r').astype(np.float32)
    labels_array = np.load(preprocessed_labels,
                           mmap_mode='r').astype(np.float32)
    print(f"Successfully loaded data:")
    print(f"Inputs shape: {inputs_array.shape}")
    print(f"Labels shape: {labels_array.shape}")
except Exception as e:
    print(f"Error loading preprocessed data: {e}")
    print("Falling back to preprocessing from scratch...")

    # Preprocess data
    print("Preprocessing data...")
    file_list = glob.glob(
        '/content/drive/MyDrive/Desert_Locust_Exported_Images_Ethiopia/*.tif')
    print(f"Found {len(file_list)} GeoTIFF files")
    presence_files = []
    absence_files = []
    for file in file_list:
        label = get_label_from_filename(file)
        if label == 1:
            presence_files.append(file)
        elif label == 0:
            absence_files.append(file)

    print(
        f"Presence files: {len(presence_files)}, Absence files: {len(absence_files)}")

    # Select equal number of presence and absence files
    random.seed(42)  # For reproducibility
    num_samples = min(len(presence_files), len(absence_files))  # 4,938
    # Optionally reduce dataset size for testing
    num_samples = min(num_samples, 2000)  # Limit to 2000 samples for free tier
    balanced_file_list = random.sample(
        presence_files, num_samples) + random.sample(absence_files, num_samples)
    random.shuffle(balanced_file_list)
    print(f"Selected {len(balanced_file_list)} files for balanced dataset")

    inputs_array, labels_array = load_all_data_parallel(
        balanced_file_list, num_workers=2)  # Reduce workers to save memory

    # Validate and clean arrays
    print("Validating preprocessed data...")
    inputs_array = np.where(np.isnan(inputs_array) |
                            np.isinf(inputs_array), 0, inputs_array)
    labels_array = np.where(np.isnan(labels_array) |
                            np.isinf(labels_array), 0, labels_array)
    labels_array = np.where(labels_array > 0.5, 1, 0).astype(np.float32)

    # Save clean arrays
    np.save(preprocessed_inputs, inputs_array)
    np.save(preprocessed_labels, labels_array)
    print("Preprocessed data saved to Google Drive.")
    gc.collect()

# Convert to Dask arrays for chunked processing
print("Converting data to Dask arrays...")
inputs_dask = da.from_array(inputs_array, chunks=(
    100, 59, 41, 41))  # Chunk by samples
labels_dask = da.from_array(labels_array, chunks=(100, 41, 41))
inputs_tensor = torch.from_numpy(inputs_dask.compute()).float()
labels_tensor = torch.from_numpy(labels_dask.compute()).float()

# Normalize inputs in chunks
print("Normalizing inputs...")
inputs_tensor = (inputs_tensor - inputs_tensor.mean()) / \
    (inputs_tensor.std() + 1e-8)

# Create TensorDataset
dataset = TensorDataset(inputs_tensor, labels_tensor)
print(f"Created dataset with {len(dataset)} samples")
del inputs_array, labels_array, inputs_dask, labels_dask
gc.collect()

# Split dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders with pinned memory
batch_size = 8  # Reduced batch size
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)

# Define Vision Transformer model


class ViTForSegmentation(nn.Module):
    def __init__(self, in_channels=59, hidden_dim=128, num_layers=4, num_heads=4, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.patch_embed = nn.Linear(in_channels, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 41*41, hidden_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=mlp_dim,
                dropout=dropout, batch_first=True
            ),
            num_layers=num_layers
        )
        self.head = nn.Linear(hidden_dim, 1)

        # Initialize weights
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H*W).transpose(1, 2)
        x = self.patch_embed(x) + self.pos_embed
        x = self.transformer(x)
        x = self.head(x).view(B, H, W)
        return x


# Initialize model, loss function, and optimizer
model = ViTForSegmentation()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 5  # Reduced for testing
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(
            device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        del inputs, labels, outputs, loss
        gc.collect()
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0
    val_preds, val_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs).cpu().numpy()
            val_preds.append(preds)
            val_labels.append(labels.cpu().numpy())
            del inputs, labels, outputs, loss
            gc.collect()
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    # Compute validation metrics
    val_preds = np.concatenate(val_preds).flatten()
    val_labels = np.concatenate(val_labels).flatten()
    val_preds = np.where(np.isnan(val_preds) |
                         np.isinf(val_preds), 0, val_preds)
    val_labels = np.where(np.isnan(val_labels) |
                          np.isinf(val_labels), 0, val_labels)
    preds_binary = (val_preds > 0.5).astype(int)
    accuracy = accuracy_score(val_labels, preds_binary)
    precision = precision_score(val_labels, preds_binary, zero_division=0)
    recall = recall_score(val_labels, preds_binary, zero_division=0)
    f1 = f1_score(val_labels, preds_binary, zero_division=0)
    roc_auc = roc_auc_score(val_labels, val_preds)

    print(
        f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    print(
        f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}')

# Save the trained model
torch.save(model.state_dict(), '/content/drive/MyDrive/locust_model.pth')

# Plot loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')
plt.show()

# Plot confusion matrix
cm = confusion_matrix(val_labels, preds_binary)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plot ROC curve
fpr, tpr, _ = roc_curve(val_labels, val_preds)
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Visualize sample predictions
num_samples = 3
indices = np.random.choice(len(val_dataset), num_samples, replace=False)
for i in indices:
    inputs, labels = val_dataset[i]
    inputs = inputs.unsqueeze(0).to(device)
    with torch.no_grad():
        preds = torch.sigmoid(model(inputs)).cpu().squeeze().numpy() > 0.5
    input_band = inputs[0, 0].cpu().numpy()
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(input_band, cmap='viridis')
    plt.title('Input Band')
    plt.subplot(1, 3, 2)
    plt.imshow(labels.numpy(), cmap='gray')
    plt.title('Ground Truth')
    plt.subplot(1, 3, 3)
    plt.imshow(preds, cmap='gray')
    plt.title('Prediction')
    plt.show()
