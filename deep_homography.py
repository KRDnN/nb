#!/usr/bin/env python3
"""
Deep Image Homography Estimation

This module implements a deep learning approach for estimating homography transformations
between pairs of images. The model is based on a CNN architecture that directly regresses
the homography parameters from image patches.

Architecture:
- VGG-style CNN backbone
- Regression head for 8-parameter homography
- Data augmentation and synthetic training data generation
- Pre-trained model loading and fine-tuning capabilities

References:
- "Deep Image Homography Estimation" by DeTone et al.
- "Unsupervised Deep Homography: A Fast and Robust Homography Estimation Model"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from sklearn.metrics import mean_squared_error
import kornia
from typing import Tuple, List, Dict, Optional


class HomographyNet(nn.Module):
    """
    Deep CNN for homography estimation.
    
    Takes two image patches as input and outputs 8 homography parameters.
    The architecture is based on VGG with batch normalization and dropout.
    """
    
    def __init__(self, input_channels: int = 2, dropout_rate: float = 0.5):
        super(HomographyNet, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Adaptive pooling to handle variable input sizes
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Output 8 homography parameters
            nn.Linear(128, 8)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Homography parameters of shape (batch_size, 8)
        """
        # Extract features
        features = self.features(x)
        
        # Flatten for regression
        features = features.view(features.size(0), -1)
        
        # Regress homography parameters
        h_params = self.regressor(features)
        
        return h_params


class HomographyDataset(Dataset):
    """
    Dataset class for homography estimation training.
    
    Generates synthetic training data by applying random homographies
    to image patches and creating training pairs.
    """
    
    def __init__(self, 
                 image_dir: str,
                 patch_size: int = 128,
                 perturbation_range: int = 32,
                 num_samples: int = 10000,
                 transform=None):
        self.image_dir = image_dir
        self.patch_size = patch_size
        self.perturbation_range = perturbation_range
        self.num_samples = num_samples
        self.transform = transform
        
        # Load all images
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            import glob
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))
        
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")
    
    def __len__(self):
        return self.num_samples
    
    def generate_homography_params(self) -> np.ndarray:
        """Generate random homography parameters."""
        # Random perturbation for the four corners
        points_1 = np.array([
            [0, 0],
            [self.patch_size, 0],
            [self.patch_size, self.patch_size],
            [0, self.patch_size]
        ], dtype=np.float32)
        
        # Add random perturbations
        perturbations = np.random.uniform(
            -self.perturbation_range, 
            self.perturbation_range, 
            (4, 2)
        ).astype(np.float32)
        
        points_2 = points_1 + perturbations
        
        # Compute homography
        H = cv2.getPerspectiveTransform(points_1, points_2)
        
        # Return the 8 parameters (excluding the last element which is 1)
        return H[:2, :].flatten()  # Shape: (8,)
    
    def __getitem__(self, idx):
        # Randomly select an image
        img_path = random.choice(self.image_paths)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            # Fallback to a random different image
            img_path = random.choice(self.image_paths)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Ensure minimum image size
        h, w = img.shape
        if h < self.patch_size * 2 or w < self.patch_size * 2:
            img = cv2.resize(img, (self.patch_size * 2, self.patch_size * 2))
            h, w = img.shape
        
        # Randomly crop a patch
        start_x = random.randint(0, w - self.patch_size)
        start_y = random.randint(0, h - self.patch_size)
        patch1 = img[start_y:start_y + self.patch_size, 
                    start_x:start_x + self.patch_size]
        
        # Generate random homography
        h_params = self.generate_homography_params()
        H = np.zeros((3, 3), dtype=np.float32)
        H[:2, :] = h_params.reshape(2, 4)
        H[2, 2] = 1.0
        
        # Apply homography to create second patch
        patch2 = cv2.warpPerspective(patch1, H, (self.patch_size, self.patch_size))
        
        # Stack patches as input
        input_tensor = np.stack([patch1, patch2], axis=0).astype(np.float32) / 255.0
        
        # Apply transforms if provided
        if self.transform:
            input_tensor = self.transform(input_tensor)
        
        return torch.from_numpy(input_tensor), torch.from_numpy(h_params.astype(np.float32))


class HomographyTrainer:
    """Training class for homography estimation model."""
    
    def __init__(self, 
                 model: HomographyNet,
                 device: torch.device,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=50, 
            gamma=0.5
        )
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.6f}'
            })
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader, 
              epochs: int,
              save_path: str = 'models/homography_net.pth'):
        """Full training loop."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, save_path)
                print(f"New best model saved with val_loss: {val_loss:.6f}")
    
    def plot_training_history(self, save_path: str = 'training_history.png'):
        """Plot training history."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()


class HomographyEstimator:
    """Inference class for homography estimation."""
    
    def __init__(self, model_path: str, device: torch.device):
        self.device = device
        self.model = HomographyNet()
        
        # Load trained model
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        self.patch_size = 128
    
    def preprocess_images(self, img1: np.ndarray, img2: np.ndarray) -> torch.Tensor:
        """Preprocess image pair for inference."""
        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Resize to patch size
        img1 = cv2.resize(img1, (self.patch_size, self.patch_size))
        img2 = cv2.resize(img2, (self.patch_size, self.patch_size))
        
        # Normalize
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0
        
        # Stack and convert to tensor
        input_tensor = np.stack([img1, img2], axis=0)
        input_tensor = torch.from_numpy(input_tensor).unsqueeze(0)  # Add batch dimension
        
        return input_tensor
    
    def estimate_homography(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Estimate homography between two images.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            3x3 homography matrix
        """
        input_tensor = self.preprocess_images(img1, img2).to(self.device)
        
        with torch.no_grad():
            h_params = self.model(input_tensor)
            h_params = h_params.cpu().numpy().flatten()
        
        # Reconstruct homography matrix
        H = np.zeros((3, 3), dtype=np.float32)
        H[:2, :] = h_params.reshape(2, 4)
        H[2, 2] = 1.0
        
        return H
    
    def estimate_homography_keypoints(self, 
                                    img1: np.ndarray, 
                                    img2: np.ndarray,
                                    keypoints1: List[cv2.KeyPoint],
                                    keypoints2: List[cv2.KeyPoint],
                                    matches: List[cv2.DMatch]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate homography using keypoint matches as regions of interest.
        
        Args:
            img1: First image
            img2: Second image
            keypoints1: Keypoints in first image
            keypoints2: Keypoints in second image
            matches: Matches between keypoints
            
        Returns:
            Tuple of (homography_matrix, inlier_mask)
        """
        if len(matches) < 10:
            # Fall back to full image estimation
            return self.estimate_homography(img1, img2), np.ones(len(matches), dtype=bool)
        
        # Extract matched point coordinates
        pts1 = np.array([keypoints1[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts2 = np.array([keypoints2[m.trainIdx].pt for m in matches], dtype=np.float32)
        
        # Use multiple patch pairs around matched keypoints
        homographies = []
        
        for i in range(min(5, len(matches))):  # Use up to 5 matches
            # Extract patches around keypoints
            x1, y1 = int(pts1[i][0]), int(pts1[i][1])
            x2, y2 = int(pts2[i][0]), int(pts2[i][1])
            
            # Define patch bounds
            patch_half = self.patch_size // 2
            
            # Patch 1
            y1_start = max(0, y1 - patch_half)
            y1_end = min(img1.shape[0], y1 + patch_half)
            x1_start = max(0, x1 - patch_half)
            x1_end = min(img1.shape[1], x1 + patch_half)
            
            # Patch 2
            y2_start = max(0, y2 - patch_half)
            y2_end = min(img2.shape[0], y2 + patch_half)
            x2_start = max(0, x2 - patch_half)
            x2_end = min(img2.shape[1], x2 + patch_half)
            
            if (x1_end - x1_start >= 64 and y1_end - y1_start >= 64 and
                x2_end - x2_start >= 64 and y2_end - y2_start >= 64):
                
                patch1 = img1[y1_start:y1_end, x1_start:x1_end]
                patch2 = img2[y2_start:y2_end, x2_start:x2_end]
                
                try:
                    H_local = self.estimate_homography(patch1, patch2)
                    homographies.append(H_local)
                except:
                    continue
        
        if homographies:
            # Average the homographies (simple approach)
            H_avg = np.mean(homographies, axis=0)
            
            # Verify with traditional method for comparison
            H_traditional, mask = cv2.findHomography(
                pts1, pts2, cv2.RANSAC, 5.0
            )
            
            if H_traditional is not None:
                # Use traditional method mask but deep learning homography
                return H_avg, mask.flatten().astype(bool)
            else:
                return H_avg, np.ones(len(matches), dtype=bool)
        else:
            # Fall back to traditional method
            H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
            if H is not None:
                return H, mask.flatten().astype(bool)
            else:
                return self.estimate_homography(img1, img2), np.ones(len(matches), dtype=bool)


def create_training_data(image_dir: str, output_dir: str, num_samples: int = 50000):
    """
    Create training dataset for homography estimation.
    
    Args:
        image_dir: Directory containing training images
        output_dir: Directory to save training data
        num_samples: Number of training samples to generate
    """
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = HomographyDataset(
        image_dir=image_dir,
        num_samples=num_samples,
        patch_size=128
    )
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"Creating {num_samples} training samples...")
    
    all_data = []
    all_targets = []
    
    for batch_idx, (data, targets) in enumerate(tqdm(dataloader)):
        all_data.append(data.numpy())
        all_targets.append(targets.numpy())
    
    # Save data
    np.save(os.path.join(output_dir, 'training_data.npy'), np.concatenate(all_data))
    np.save(os.path.join(output_dir, 'training_targets.npy'), np.concatenate(all_targets))
    
    print(f"Training data saved to {output_dir}")


def train_homography_model(image_dir: str, 
                          model_save_path: str = 'models/homography_net.pth',
                          epochs: int = 100,
                          batch_size: int = 32):
    """
    Train homography estimation model.
    
    Args:
        image_dir: Directory containing training images
        model_save_path: Path to save trained model
        epochs: Number of training epochs
        batch_size: Training batch size
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = HomographyDataset(
        image_dir=image_dir,
        num_samples=40000,
        patch_size=128
    )
    
    val_dataset = HomographyDataset(
        image_dir=image_dir,
        num_samples=8000,
        patch_size=128
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model and trainer
    model = HomographyNet(input_channels=2)
    trainer = HomographyTrainer(model, device)
    
    # Train model
    trainer.train(train_loader, val_loader, epochs, model_save_path)
    
    # Plot training history
    trainer.plot_training_history('training_history.png')
    
    print(f"Training completed. Model saved to {model_save_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deep Image Homography Estimation")
    parser.add_argument("--mode", choices=["train", "inference"], required=True)
    parser.add_argument("--image_dir", type=str, help="Directory containing images")
    parser.add_argument("--model_path", type=str, default="models/homography_net.pth")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img1", type=str, help="First image for inference")
    parser.add_argument("--img2", type=str, help="Second image for inference")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        if not args.image_dir:
            raise ValueError("--image_dir is required for training")
        train_homography_model(args.image_dir, args.model_path, args.epochs, args.batch_size)
    
    elif args.mode == "inference":
        if not args.img1 or not args.img2:
            raise ValueError("--img1 and --img2 are required for inference")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        estimator = HomographyEstimator(args.model_path, device)
        
        img1 = cv2.imread(args.img1)
        img2 = cv2.imread(args.img2)
        
        H = estimator.estimate_homography(img1, img2)
        print("Estimated Homography:")
        print(H)