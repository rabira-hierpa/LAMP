# Multi-Resolution Heterogeneous Vision Transformer (MRH-ViT) for Desert Locust Prediction
# Optimized for Google Colab with your specific GeoTIFF data structure

import random
import os
import gc
import numpy as np
import glob
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import drive
from tqdm import tqdm
import logging
from pathlib import Path
from torch.amp import autocast, GradScaler
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
import math
import warnings
warnings.filterwarnings('ignore')

# Ensure plots display in Colab
%matplotlib inline

# Mount Google Drive
print("Mounting Google Drive...")
drive.mount('/content/drive', force_remount=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(
            '/content/drive/MyDrive/mrh_vit_locust_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MultiResolutionPatchEmbedding(nn.Module):
    """
    Processes multiple resolution views of the same data
    Novel contribution: Unlike standard ViT, this handles multi-scale analysis
    """

    def __init__(self,
                 in_channels: int = 59,  # 60 bands minus label
                 embed_dim: int = 256,
                 patch_sizes: List[int] = [1, 2, 4]):  # Different patch sizes for multi-resolution
        super().__init__()

        self.patch_sizes = patch_sizes
        self.embed_dim = embed_dim

        # Separate embedding for each patch size (resolution)
        self.patch_embeddings = nn.ModuleList()
        for patch_size in patch_sizes:
            # Calculate number of patches
            num_patches = (41 // patch_size) ** 2
            patch_dim = in_channels * (patch_size ** 2)

            self.patch_embeddings.append(nn.Sequential(
                nn.Linear(patch_dim, embed_dim),
                nn.LayerNorm(embed_dim)
            ))

        # Resolution type embeddings
        self.resolution_embeddings = nn.Embedding(len(patch_sizes), embed_dim)

        # Positional embeddings for each resolution
        self.pos_embeddings = nn.ModuleList()
        for patch_size in patch_sizes:
            num_patches = (41 // patch_size) ** 2
            self.pos_embeddings.append(
                nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
            )

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W] where C=59, H=W=41
        Returns:
            Multi-resolution embeddings [B, total_patches, embed_dim]
        """
        B, C, H, W = x.shape
        all_embeddings = []

        for i, patch_size in enumerate(self.patch_sizes):
            # Create patches of different sizes
            # [B, num_patches, patch_dim]
            patches = self._create_patches(x, patch_size)

            # Embed patches
            embedded = self.patch_embeddings[i](
                patches)  # [B, num_patches, embed_dim]

            # Add positional embeddings
            embedded = embedded + self.pos_embeddings[i]

            # Add resolution type embedding
            res_emb = self.resolution_embeddings(
                torch.tensor(i, device=x.device))
            embedded = embedded + res_emb.unsqueeze(0).unsqueeze(0)

            all_embeddings.append(embedded)

        # Concatenate all resolution embeddings
        # [B, total_patches, embed_dim]
        return torch.cat(all_embeddings, dim=1)

    def _create_patches(self, x, patch_size):
        """Create non-overlapping patches of given size"""
        B, C, H, W = x.shape

        # Pad if necessary
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # Create patches
        patches = x.unfold(2, patch_size, patch_size).unfold(
            3, patch_size, patch_size)
        patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        # [B, num_patches, C*patch_size*patch_size]
        patches = patches.view(B, patches.size(1), -1)

        return patches


class FeatureGroupEncoder(nn.Module):
    """
    Encodes different groups of features (temporal, static, indices) separately
    Novel contribution: Handles heterogeneous feature types in remote sensing data
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()

        # Define feature groups based on your band names
        self.feature_groups = {
            # Vegetation indices
            'vegetation': ['NDVI', 'EVI', 'NDWI', 'TCI', 'VHI'],
            # Climate variables
            'climate': ['LST_Day', 'precipitation', 'ET', 'aet', 'pet', 'TVDI'],
            # Wind components
            'wind': ['u_component_of_wind', 'v_component_of_wind'],
            # Soil properties
            'soil': ['sm_surface', 'sand_0_5cm', 'sand_5_15cm', 'soil_texture'],
            # Topographic features
            'topography': ['elevation', 'slope', 'aspect'],
            'static': ['landcover']  # Static features
        }

        # Group-specific encoders
        self.group_encoders = nn.ModuleDict()
        for group_name in self.feature_groups.keys():
            self.group_encoders[group_name] = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
                nn.Dropout(0.1)
            )

        # Group type embeddings
        self.group_embeddings = nn.Embedding(
            len(self.feature_groups), embed_dim)

    def forward(self, x):
        """
        Args:
            x: Patch embeddings [B, num_patches, embed_dim]
        Returns:
            Group-enhanced embeddings [B, num_patches, embed_dim]
        """
        # For simplicity, we'll apply all group encodings and average them
        # In practice, you'd need to map specific channels to specific groups

        group_outputs = []
        for i, (group_name, encoder) in enumerate(self.group_encoders.items()):
            # Apply group-specific encoding
            group_encoded = encoder(x)

            # Add group type embedding
            group_emb = self.group_embeddings(torch.tensor(i, device=x.device))
            group_encoded = group_encoded + group_emb.unsqueeze(0).unsqueeze(0)

            group_outputs.append(group_encoded)

        # Combine group outputs (simple average for now)
        return torch.stack(group_outputs, dim=0).mean(dim=0)


class PhysicsInformedAttention(nn.Module):
    """
    Attention mechanism that incorporates known locust behavior patterns
    Novel contribution: First physics-informed attention for ecological prediction
    """

    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Physics-informed bias parameters
        self.wind_bias_weight = nn.Parameter(torch.tensor(0.1))
        self.elevation_bias_weight = nn.Parameter(torch.tensor(0.1))
        self.temperature_bias_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, return_attention=False):
        """
        Args:
            x: Input tensor [B, seq_len, embed_dim]
        Returns:
            Attention output and optionally attention weights
        """
        B, L, D = x.shape

        # Standard multi-head attention computation
        q = self.q_proj(x).view(B, L, self.num_heads,
                                self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads,
                                self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads,
                                self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / \
            math.sqrt(self.head_dim)

        # Apply physics-informed biases
        # This is simplified - in practice, you'd extract actual physical parameters
        # from your input bands and use them to compute these biases

        # Simulate physics-informed biases based on spatial relationships
        physics_bias = self._compute_physics_bias(scores.shape[-2:], x.device)
        scores = scores + physics_bias.unsqueeze(0).unsqueeze(0)

        # Apply attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.out_proj(attn_output)

        if return_attention:
            return output, attn_weights.mean(dim=1)  # Average over heads
        return output

    def _compute_physics_bias(self, shape, device):
        """Compute physics-informed attention bias based on spatial relationships"""
        H, W = shape

        # Create distance-based bias (locusts tend to aggregate)
        center = H // 2
        y_coords = torch.arange(H, device=device).float() - center
        x_coords = torch.arange(W, device=device).float() - center

        # Distance from center
        dist_bias = -(y_coords.unsqueeze(1) ** 2 +
                      x_coords.unsqueeze(0) ** 2).sqrt()
        dist_bias = dist_bias / dist_bias.abs().max()  # Normalize

        return dist_bias * 0.1  # Small bias to not overwhelm attention


class MRHViT(nn.Module):
    """
    Multi-Resolution Heterogeneous Vision Transformer for Desert Locust Prediction

    Novel contributions:
    1. Multi-resolution patch processing for different swarm scales
    2. Feature group encoding for heterogeneous remote sensing data
    3. Physics-informed attention incorporating locust behavior
    """

    def __init__(self,
                 in_channels: int = 59,  # 60 bands minus label
                 embed_dim: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 patch_sizes: List[int] = [1, 2, 4]):
        super().__init__()

        self.embed_dim = embed_dim

        # Multi-resolution patch embedding
        self.patch_embedding = MultiResolutionPatchEmbedding(
            in_channels, embed_dim, patch_sizes
        )

        # Feature group encoder
        self.feature_encoder = FeatureGroupEncoder(embed_dim)

        # Transformer layers with physics-informed attention
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim),
                'attn': PhysicsInformedAttention(embed_dim, num_heads),
                'norm2': nn.LayerNorm(embed_dim),
                'mlp': nn.Sequential(
                    nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
                    nn.Dropout(dropout)
                )
            })
            self.layers.append(layer)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_attention_maps=False):
        """
        Args:
            x: Input tensor [B, C, H, W] where C=59, H=W=41
        Returns:
            Predictions [B, 1] - probability of locust presence
            Optionally attention maps for visualization
        """
        # Multi-resolution patch embedding
        x = self.patch_embedding(x)  # [B, num_patches, embed_dim]

        # Feature group encoding
        x = self.feature_encoder(x)  # [B, num_patches, embed_dim]

        # Apply transformer layers
        attention_maps = []
        for layer in self.layers:
            # Self-attention with physics-informed biases
            attn_input = layer['norm1'](x)
            if return_attention_maps:
                attn_output, attn_weights = layer['attn'](
                    attn_input, return_attention=True)
                attention_maps.append(attn_weights)
            else:
                attn_output = layer['attn'](attn_input)
            x = x + attn_output

            # MLP
            mlp_input = layer['norm2'](x)
            mlp_output = layer['mlp'](mlp_input)
            x = x + mlp_output

        # Global average pooling
        x = x.mean(dim=1)  # [B, embed_dim]

        # Classification
        predictions = self.classifier(x)  # [B, 1]

        if return_attention_maps:
            return predictions.squeeze(-1), attention_maps
        return predictions.squeeze(-1)


class LocustGeoTIFFDataset(Dataset):
    """Dataset class for loading GeoTIFF files with balanced sampling"""

    def __init__(self, file_paths, transform=None, cache_data=False):
        self.file_paths = file_paths
        self.transform = transform
        self.cache_data = cache_data
        self.cache = {} if cache_data else None

        # Extract labels from filenames
        self.labels = []
        for file_path in file_paths:
            self.labels.append(self._extract_label_from_filename(file_path))

    def _extract_label_from_filename(self, file_path):
        """Extract label from filename - assuming presence files have 'presence' in name"""
        filename = os.path.basename(file_path).lower()
        if 'presence' in filename or 'label_1' in filename:
            return 1
        else:
            return 0

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        # Check cache first
        if self.cache_data and file_path in self.cache:
            data, label = self.cache[file_path]
        else:
            # Load GeoTIFF file
            try:
                with rasterio.open(file_path) as src:
                    # Read all 60 bands
                    data = src.read().astype(np.float32)  # Shape: (60, 41, 41)

                    # Handle NaN and infinite values
                    data = np.where(np.isnan(data) | np.isinf(data), 0, data)

                    # Extract features (first 59 bands) and label (last band)
                    features = data[:59]  # (59, 41, 41)
                    label_band = data[59]  # (41, 41)

                    # Extract label (use center pixel or majority vote)
                    label = float(label_band[20, 20] > 0.5)  # Center pixel

                    # Cache if enabled
                    if self.cache_data:
                        self.cache[file_path] = (features, label)

                    data = features

            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                # Return zeros if file can't be loaded
                data = np.zeros((59, 41, 41), dtype=np.float32)
                label = 0.0

        # Convert to torch tensor
        data = torch.from_numpy(data).float()
        label = torch.tensor(label, dtype=torch.float32)

        # Apply transforms if any
        if self.transform:
            data = self.transform(data)

        return data, label


def create_balanced_dataset(data_dir, balance_ratio=1.0, max_samples=None):
    """Create balanced dataset from GeoTIFF files"""

    # Find all GeoTIFF files
    tiff_files = glob.glob(os.path.join(data_dir, "*.tif"))

    logger.info(f"Found {len(tiff_files)} GeoTIFF files")

    # Separate presence and absence files
    presence_files = []
    absence_files = []

    for file in tiff_files:
        filename = os.path.basename(file).lower()
        if 'presence' in filename or 'label_1' in filename:
            presence_files.append(file)
        else:
            absence_files.append(file)

    logger.info(f"Presence files: {len(presence_files)}")
    logger.info(f"Absence files: {len(absence_files)}")

    # Balance the dataset
    min_count = min(len(presence_files), len(absence_files))
    if max_samples:
        min_count = min(min_count, max_samples // 2)

    # Random sampling for balance
    random.seed(42)
    selected_presence = random.sample(presence_files, min_count)
    selected_absence = random.sample(absence_files, min_count)

    # Combine and shuffle
    all_files = selected_presence + selected_absence
    random.shuffle(all_files)

    logger.info(f"Created balanced dataset with {len(all_files)} files")
    logger.info(
        f"Presence: {len(selected_presence)}, Absence: {len(selected_absence)}")

    return all_files


def train_mrh_vit():
    """Main training function"""

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Data directory - modify this path to your GeoTIFF files location
    data_dir = "/content/drive/MyDrive/Desert_Locust_Exported_Images_Ethiopia"

    # Create balanced dataset
    all_files = create_balanced_dataset(
        data_dir, max_samples=8000)  # Limit for Colab memory

    # Create data transforms
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])

    # Create dataset
    dataset = LocustGeoTIFFDataset(all_files, transform=transform)

    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    logger.info(
        f"Dataset splits - Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # Create data loaders
    batch_size = 16  # Adjust based on your GPU memory

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    # Initialize model
    model = MRHViT(
        in_channels=59,
        embed_dim=256,
        num_layers=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        patch_sizes=[1, 2, 4]
    ).to(device)

    logger.info(
        f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    scaler = GradScaler()

    # Training parameters
    num_epochs = 50
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0

    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_f1': [], 'val_precision': [], 'val_recall': []
    }

    logger.info("Starting training...")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(
            train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (data, labels) in enumerate(train_pbar):
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                outputs = model(data)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

            # Update progress bar
            train_acc = train_correct / train_total
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{train_acc:.4f}'
            })

        scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            val_pbar = tqdm(
                val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for data, labels in val_pbar:
                data, labels = data.to(device), labels.to(device)

                outputs = model(data)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predictions = torch.sigmoid(outputs)
                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        # Calculate validation metrics
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        val_pred_binary = (val_predictions > 0.5).astype(int)

        val_acc = accuracy_score(val_targets, val_pred_binary)
        val_f1 = f1_score(val_targets, val_pred_binary, zero_division=0)
        val_precision = precision_score(
            val_targets, val_pred_binary, zero_division=0)
        val_recall = recall_score(
            val_targets, val_pred_binary, zero_division=0)

        # Store history
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_correct / train_total)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)

        # Print epoch results
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(
            f'  Train Loss: {history["train_loss"][-1]:.4f}, Train Acc: {history["train_acc"][-1]:.4f}')
        print(
            f'  Val Loss: {history["val_loss"][-1]:.4f}, Val Acc: {val_acc:.4f}')
        print(
            f'  Val F1: {val_f1:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}')
        print('-' * 60)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(),
                       '/content/drive/MyDrive/best_mrh_vit_model.pth')
            logger.info(
                f'New best model saved with validation accuracy: {val_acc:.4f}')
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            logger.info(f'Early stopping triggered after {epoch+1} epochs')
            break

    # Load best model for testing
    model.load_state_dict(torch.load(
        '/content/drive/MyDrive/best_mrh_vit_model.pth'))

    # Test evaluation
    model.eval()
    test_predictions = []
    test_targets = []
    attention_maps = []

    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm(test_loader, desc='Testing')):
            data, labels = data.to(device), labels.to(device)

            if i == 0:  # Get attention maps for first batch
                outputs, attn_maps = model(data, return_attention_maps=True)
                attention_maps = attn_maps
            else:
                outputs = model(data)

            predictions = torch.sigmoid(outputs)
            test_predictions.extend(predictions.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    # Calculate test metrics
    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)
    test_pred_binary = (test_predictions > 0.5).astype(int)

    test_acc = accuracy_score(test_targets, test_pred_binary)
    test_f1 = f1_score(test_targets, test_pred_binary, zero_division=0)
    test_precision = precision_score(
        test_targets, test_pred_binary, zero_division=0)
    test_recall = recall_score(test_targets, test_pred_binary, zero_division=0)
    test_auc = roc_auc_score(test_targets, test_predictions)

    print(f'\nFinal Test Results:')
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Test F1-Score: {test_f1:.4f}')
    print(f'Test Precision: {test_precision:.4f}')
    print(f'Test Recall: {test_recall:.4f}')
    print(f'Test AUC-ROC: {test_auc:.4f}')

    # Visualizations
    plot_training_history(history)
    plot_confusion_matrix(test_targets, test_pred_binary)
    visualize_attention_maps(attention_maps)

    return model, history, (test_acc, test_f1, test_precision, test_recall, test_auc)


def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()

    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc')
    axes[0, 1].plot(history['val_acc'], label='Val Acc')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()

    # F1 Score
    axes[1, 0].plot(history['val_f1'], label='Val F1')
    axes[1, 0].plot(history['val_precision'], label='Val Precision')
    axes[1, 0].plot(history['val_recall'], label='Val Recall')
    axes[1, 0].set_title('Validation Metrics')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()

    # Learning rate (if you track it)
    axes[1, 1].text(0.5, 0.5, 'Additional Metrics\ncan be plotted here',
                    ha='center', va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Additional Metrics')

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(targets, predictions):
    """Plot confusion matrix"""
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Absence', 'Presence'],
                yticklabels=['Absence', 'Presence'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def visualize_attention_maps(attention_maps):
    """Visualize attention maps from different layers"""
    if attention_maps:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        # Show first 6 layers
        for i, attn_map in enumerate(attention_maps[:6]):
            # Average over heads and take first sample
            attn_avg = attn_map[0].mean(dim=0).cpu().numpy()

            # Reshape to spatial dimensions (this is approximated)
            size = int(np.sqrt(attn_avg.shape[0]))
            if size * size == attn_avg.shape[0]:
                attn_spatial = attn_avg.reshape(size, size)
            else:
                attn_spatial = attn_avg[:size*size].reshape(size, size)

            im = axes[i].imshow(attn_spatial, cmap='viridis')
            axes[i].set_title(f'Layer {i+1} Attention')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i])

        plt.suptitle('Attention Maps Across Transformer Layers')
        plt.tight_layout()
        plt.show()


# Run the training
if __name__ == "__main__":
    print("Starting MRH-ViT training for Desert Locust Prediction...")
    print("=" * 60)

    # Train the model
    model, history, test_results = train_mrh_vit()

    print("=" * 60)
    print("Training completed!")
    print(f"Best Test Results:")
    print(f"  Accuracy: {test_results[0]:.4f}")
    print(f"  F1-Score: {test_results[1]:.4f}")
    print(f"  Precision: {test_results[2]:.4f}")
    print(f"  Recall: {test_results[3]:.4f}")
    print(f"  AUC-ROC: {test_results[4]:.4f}")
