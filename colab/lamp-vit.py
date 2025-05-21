# Desert Locust Prediction using Vision Transformer in Google Colab
# Optimized for memory constraints with enhanced logging and progress tracking

# Import libraries
import random
import rasterio.windows
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
from tqdm import tqdm
import logging
from pathlib import Path
from torch.amp import autocast, GradScaler
import sys
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Ensure plots display in Colab
%matplotlib inline

# Suppress debugger warnings
%env PYDEVD_DISABLE_FILE_VALIDATION = 1

# Mount Google Drive
print("Mounting GDrive")
drive.mount('/content/drive', force_remount=True)


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('/content/drive/MyDrive/locust_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Function to load .npy file with progress bar
def load_numpy_with_progress(file_path, mmap_mode='r'):
    file_path = Path(file_path)
    total_size = file_path.stat().st_size
    with tqdm(total=total_size, desc=f"Loading {file_path.name}", unit='B', unit_scale=True) as pbar:
        try:
            data = np.load(file_path, mmap_mode=mmap_mode)
            pbar.update(total_size)
            return data
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise

# Function to extract label from file name


def get_label_from_filename(file):
    base = os.path.basename(file)
    name = os.path.splitext(base)[0]
    parts = name.split('_')
    try:
        label_index = parts.index('label') + 1
        label = int(parts[label_index])
        return label
    except (ValueError, IndexError):
        logger.error(f"Failed to parse label from {file}")
        return None

# Function to process a single GeoTIFF file


def process_file(file):
    try:
        with rasterio.open(file) as src:
            target_h, target_w = 41, 41
            window = rasterio.windows.Window(
                0, 0, min(src.width, target_w), min(src.height, target_h))
            data = src.read(window=window).astype(np.float32)
            data = np.where(np.isnan(data) | np.isinf(data), 0, data)
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
            inputs = data_padded[:59]
            label = data_padded[59]
            label = np.where(label > 0.5, 1, 0).astype(np.float32)
            return inputs, label
    except Exception as e:
        logger.error(f"Error processing {file}: {e}")
        return None

# Function to load all data with progress


def load_all_data(file_list):
    inputs_list = []
    labels_list = []
    for file in tqdm(file_list, desc="Processing GeoTIFFs"):
        result = process_file(file)
        if result is not None:
            inputs, label = result
            inputs_list.append(inputs)
            labels_list.append(label)
    inputs_array = np.stack(inputs_list).astype(np.float32)
    labels_array = np.stack(labels_list).astype(np.float32)
    return inputs_array, labels_array


# Paths for preprocessed data
preprocessed_inputs = '/content/drive/MyDrive/inputs.npy'
preprocessed_labels = '/content/drive/MyDrive/labels.npy'

# Load preprocessed data
logger.info("Loading preprocessed data from Google Drive...")
try:
    inputs_array = load_numpy_with_progress(
        preprocessed_inputs, mmap_mode='r').astype(np.float32)
    labels_array = load_numpy_with_progress(
        preprocessed_labels, mmap_mode='r').astype(np.float32)
    logger.info(
        f"Loaded data: Inputs shape: {inputs_array.shape}, Labels shape: {labels_array.shape}")
except Exception as e:
    logger.error(f"Failed to load preprocessed data: {e}")
    logger.info("Preprocessing from scratch...")
    file_list = glob.glob(
        '/content/drive/MyDrive/Desert_Locust_Exported_Images_Ethiopia/*.tif')
    logger.info(f"Found {len(file_list)} GeoTIFF files")
    print(f"Found {len(file_list)} GeoTIFF files")
    presence_files = []
    absence_files = []
    for file in tqdm(file_list, desc="Classifying files"):
        label = get_label_from_filename(file)
        if label == 1:
            presence_files.append(file)
        elif label == 0:
            absence_files.append(file)
    logger.info(
        f"Presence files: {len(presence_files)}, Absence files: {len(absence_files)}")
    print(
        f"Presence files: {len(presence_files)}, Absence files: {len(absence_files)}")
    random.seed(42)
    num_samples = min(len(presence_files), len(
        absence_files))  # Use all available images
    balanced_file_list = random.sample(
        presence_files, num_samples) + random.sample(absence_files, num_samples)
    random.shuffle(balanced_file_list)
    logger.info(
        f"Selected {len(balanced_file_list)} files for balanced dataset")
    print(f"Selected {len(balanced_file_list)} files for balanced dataset")
    inputs_array, labels_array = load_all_data(balanced_file_list)
    logger.info("Validating preprocessed data...")
    print("Validating preprocessed data...")
    inputs_array = np.where(np.isnan(inputs_array) |
                            np.isinf(inputs_array), 0, inputs_array)
    labels_array = np.where(np.isnan(labels_array) |
                            np.isinf(labels_array), 0, labels_array)
    labels_array = np.where(labels_array > 0.5, 1, 0).astype(np.float32)
    logger.info("Saving preprocessed data...")
    print("Saving preprocessed data...")
    np.save(preprocessed_inputs, inputs_array)
    np.save(preprocessed_labels, labels_array)
    logger.info("Preprocessed data saved.")
    print("Preprocessed data saved.")
    gc.collect()

# Convert to Dask arrays with smaller chunks
logger.info("Converting to Dask arrays...")
print("Converting to Dask arrays...")
inputs_dask = da.from_array(inputs_array, chunks=(50, 59, 41, 41))
labels_dask = da.from_array(labels_array, chunks=(50, 41, 41))
logger.info("Computing Dask arrays to PyTorch tensors...")
print("Computing Dask arrays to PyTorch tensors...")
inputs_tensor = torch.from_numpy(
    inputs_dask.compute(scheduler='threads')).float()
labels_tensor = torch.from_numpy(
    labels_dask.compute(scheduler='threads')).float()
# Normalize inputs
logger.info("Normalizing inputs...")
inputs_tensor = (inputs_tensor - inputs_tensor.mean()) / \
    (inputs_tensor.std() + 1e-8)
# Create TensorDataset with augmentation


class AugmentedDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(0, 360), fill=0),
        ])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_data = self.inputs[idx]
        label = self.labels[idx]
        # Apply transformations
        input_data = self.transform(input_data)
        label = self.transform(label.unsqueeze(0)).squeeze(0)
        return input_data, label


dataset = AugmentedDataset(inputs_tensor, labels_tensor)
logger.info(f"Created dataset with {len(dataset)} samples")
del inputs_array, labels_array, inputs_dask, labels_dask
gc.collect()
# Split dataset into train, val, test
N = len(dataset)
train_size = int(0.6 * N)
val_size = int(0.2 * N)
test_size = N - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size])
logger.info(
    f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")
# Create DataLoaders
batch_size = 8
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)
# Define Vision Transformer model with increased dropout


class ViTForSegmentation(nn.Module):
    def __init__(self, in_channels=59, hidden_dim=128, num_heads=8, num_layers=4, mlp_dim=1024, dropout=0.3):
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


# Initialize model
logger.info("Initializing model...")
model = ViTForSegmentation()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)
scaler = GradScaler()
logger.info(f"Using device: {device}")
# Training loop with early stopping
num_epochs = 30
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
best_val_loss = float('inf')
patience = 10
counter = 0
for epoch in range(num_epochs):
    logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(
            device, non_blocking=True)
        optimizer.zero_grad()
        with autocast(device_type=str(device)):
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
        if torch.isnan(loss):
            logger.warning("NaN loss detected, skipping batch")
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item() * inputs.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        train_correct += (preds == labels).sum().item()
        train_total += labels.numel()
        del inputs, labels, outputs, loss, preds
        gc.collect()
    train_loss /= len(train_loader.dataset)
    train_accuracy = train_correct / train_total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    print(
        f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    logger.info(
        f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    sys.stdout.flush()
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True)
            with autocast(device_type=str(device)):
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.numel()
            del inputs, labels, outputs, loss, preds
            gc.collect()
    val_loss /= len(val_loader.dataset)
    val_accuracy = val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    print(
        f"Epoch {epoch+1}: Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    logger.info(
        f"Epoch {epoch+1}: Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    sys.stdout.flush()
    scheduler.step(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(),
                   '/content/drive/MyDrive/best_locust_model.pth')
    else:
        counter += 1
        if counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

# Load best model for test evaluation and visualizations
model.load_state_dict(torch.load(
    '/content/drive/MyDrive/best_locust_model.pth'))
model.eval()

# Function to evaluate on test set


def evaluate_test_set(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            test_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            test_correct += (preds == labels).sum().item()
            test_total += labels.numel()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    test_loss /= len(test_loader.dataset)
    test_accuracy = test_correct / test_total
    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    test_f1 = f1_score(all_labels, all_preds, zero_division=0)
    test_precision = precision_score(all_labels, all_preds, zero_division=0)
    test_recall = recall_score(all_labels, all_preds, zero_division=0)
    test_roc_auc = roc_auc_score(all_labels, all_preds)
    return test_loss, test_accuracy, test_f1, test_precision, test_recall, test_roc_auc


# Evaluate on test set after final epoch
test_loss, test_accuracy, test_f1, test_precision, test_recall, test_roc_auc = evaluate_test_set(
    model, test_loader, criterion, device)
print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f} Test F1-score: {test_f1:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test ROC-AUC: {test_roc_auc:.4f}")

# Compute average metrics for train and val
avg_train_loss = sum(train_losses) / len(train_losses)
avg_train_accuracy = sum(train_accuracies) / len(train_accuracies)
avg_val_loss = sum(val_losses) / len(val_losses)
avg_val_accuracy = sum(val_accuracies) / len(val_accuracies)
logger.info(f"Average Train Loss: {avg_train_loss:.4f}")
print(f"Average Train Loss: {avg_train_loss:.4f}")
logger.info(f"Average Train Accuracy: {avg_train_accuracy:.4f}")
print(f"Average Train Accuracy: {avg_train_accuracy:.4f}")
logger.info(f"Average Validation Loss: {avg_val_loss:.4f}")
print(f"Average Validation Loss: {avg_val_loss:.4f}")
logger.info(f"Average Validation Accuracy: {avg_val_accuracy:.4f}")
print(f"Average Validation Accuracy: {avg_val_accuracy:.4f}")

# Compute validation metrics for visualizations
val_preds, val_labels = [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = torch.sigmoid(outputs).cpu().numpy()
        val_preds.append(preds)
        val_labels.append(labels.cpu().numpy())
val_preds = np.concatenate(val_preds).flatten()
val_labels = np.concatenate(val_labels).flatten()
val_preds = np.where(np.isnan(val_preds) | np.isinf(val_preds), 0, val_preds)
val_labels = np.where(np.isnan(val_labels) |
                      np.isinf(val_labels), 0, val_labels)
preds_binary = (val_preds > 0.5).astype(int)

# Plot combined metrics
logger.info("Generating combined metrics plot...")
print("Generating combined metrics plot...")
fig, ax1 = plt.subplots(figsize=(10, 5))
color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color)
ax1.plot(train_losses, label='Train Loss', color='red')
ax1.plot(val_losses, label='Val Loss', color='orange')
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color)
ax2.plot(train_accuracies, label='Train Accuracy', color='blue')
ax2.plot(val_accuracies, label='Val Accuracy', color='green')
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
plt.title('Training and Validation Metrics')
plt.show()

# Plot accuracy
logger.info("Generating accuracy plot...")
print("Generating accuracy plot...")
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

# Plot confusion matrix (validation set)
logger.info("Generating confusion matrix...")
print("Generating confusion matrix...")
cm = confusion_matrix(val_labels, preds_binary)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Validation Set)')
plt.show()

# Plot ROC curve (validation set)
logger.info("Generating ROC curve...")
print("Generating ROC curve...")
fpr, tpr, _ = roc_curve(val_labels, val_preds)
roc_auc = roc_auc_score(val_labels, val_preds)
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Validation Set)')
plt.legend()
plt.show()

# Visualize sample predictions (validation set)
logger.info("Generating sample predictions...")
print("Generating sample predictions....")
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
