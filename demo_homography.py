#!/usr/bin/env python3
"""
Deep Image Homography Estimation Demo

This script demonstrates the capabilities of the deep learning homography estimation
system with various examples and performance comparisons.

Features:
- Training demonstration on synthetic data
- Real image homography estimation
- Performance comparison with traditional methods
- Visualization of results
- Evaluation metrics
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import argparse
from pathlib import Path
import json
from deep_homography import (
    HomographyNet, HomographyEstimator, HomographyTrainer, 
    HomographyDataset, train_homography_model
)
from torch.utils.data import DataLoader


def create_sample_images(output_dir="sample_images", num_images=50):
    """Create sample images for training and testing."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating {num_images} sample images...")
    
    # Generate synthetic textures and patterns
    for i in range(num_images):
        # Create different types of patterns
        if i % 4 == 0:
            # Checkerboard pattern
            img = np.zeros((256, 256, 3), dtype=np.uint8)
            for y in range(0, 256, 32):
                for x in range(0, 256, 32):
                    if (x//32 + y//32) % 2 == 0:
                        img[y:y+32, x:x+32] = [255, 255, 255]
            
        elif i % 4 == 1:
            # Random noise texture
            img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            
        elif i % 4 == 2:
            # Gradient pattern
            x, y = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))
            pattern = np.sin(10 * np.pi * x) * np.cos(10 * np.pi * y)
            img = np.stack([pattern, pattern * 0.5, pattern * 0.8], axis=2)
            img = ((img + 1) * 127.5).astype(np.uint8)
            
        else:
            # Circular patterns
            center = (128, 128)
            img = np.zeros((256, 256, 3), dtype=np.uint8)
            for radius in range(20, 120, 20):
                cv2.circle(img, center, radius, (255, 255, 255), 2)
                
        # Add some random geometric shapes
        if np.random.random() > 0.5:
            # Add rectangles
            x1, y1 = np.random.randint(0, 128, 2)
            x2, y2 = x1 + np.random.randint(20, 80), y1 + np.random.randint(20, 80)
            color = tuple(np.random.randint(0, 256, 3).tolist())
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        
        if np.random.random() > 0.5:
            # Add lines
            for _ in range(np.random.randint(1, 5)):
                pt1 = tuple(np.random.randint(0, 256, 2))
                pt2 = tuple(np.random.randint(0, 256, 2))
                color = tuple(np.random.randint(0, 256, 3).tolist())
                cv2.line(img, pt1, pt2, color, np.random.randint(1, 5))
        
        # Save image
        cv2.imwrite(os.path.join(output_dir, f"sample_{i:03d}.jpg"), img)
    
    print(f"Created {num_images} sample images in {output_dir}")


def demonstrate_training(image_dir="sample_images", epochs=20):
    """Demonstrate the training process."""
    print("\n" + "="*60)
    print("TRAINING DEMONSTRATION")
    print("="*60)
    
    if not os.path.exists(image_dir) or len(os.listdir(image_dir)) == 0:
        print("Creating sample training images...")
        create_sample_images(image_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("Creating training dataset...")
    train_dataset = HomographyDataset(
        image_dir=image_dir,
        num_samples=2000,
        patch_size=128,
        perturbation_range=32
    )
    
    val_dataset = HomographyDataset(
        image_dir=image_dir,
        num_samples=400,
        patch_size=128,
        perturbation_range=32
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Create model and trainer
    model = HomographyNet(input_channels=2)
    trainer = HomographyTrainer(model, device, learning_rate=1e-3)
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Train model
    model_path = "models/demo_homography_net.pth"
    trainer.train(train_loader, val_loader, epochs, model_path)
    
    # Plot training history
    trainer.plot_training_history("demo_training_history.png")
    print(f"Training completed! Model saved to {model_path}")
    print("Training history plot saved to demo_training_history.png")
    
    return model_path


def create_test_image_pair():
    """Create a test image pair with known homography."""
    print("\nCreating test image pair...")
    
    # Create a complex test image
    img1 = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Add checkerboard pattern
    for y in range(0, 400, 40):
        for x in range(0, 400, 40):
            if (x//40 + y//40) % 2 == 0:
                img1[y:y+40, x:x+40] = [200, 200, 200]
    
    # Add some geometric shapes
    cv2.circle(img1, (100, 100), 30, (255, 0, 0), -1)
    cv2.rectangle(img1, (200, 50), (350, 150), (0, 255, 0), -1)
    cv2.ellipse(img1, (300, 300), (50, 30), 45, 0, 360, (0, 0, 255), -1)
    
    # Add text
    cv2.putText(img1, "TEST", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)
    
    # Create a known homography transformation
    src_points = np.array([
        [50, 50],
        [350, 50],
        [350, 350],
        [50, 350]
    ], dtype=np.float32)
    
    dst_points = np.array([
        [80, 60],
        [320, 40],
        [360, 340],
        [40, 360]
    ], dtype=np.float32)
    
    # Compute ground truth homography
    H_true = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply homography to create second image
    img2 = cv2.warpPerspective(img1, H_true, (400, 400))
    
    # Save test images
    cv2.imwrite("test_img1.jpg", img1)
    cv2.imwrite("test_img2.jpg", img2)
    
    print("Test images saved: test_img1.jpg, test_img2.jpg")
    return img1, img2, H_true


def demonstrate_inference(model_path):
    """Demonstrate homography estimation on test images."""
    print("\n" + "="*60)
    print("INFERENCE DEMONSTRATION")
    print("="*60)
    
    # Create test image pair
    img1, img2, H_true = create_test_image_pair()
    
    # Load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    estimator = HomographyEstimator(model_path, device)
    
    print("Estimating homography using deep learning...")
    start_time = time.time()
    H_estimated = estimator.estimate_homography(img1, img2)
    dl_time = time.time() - start_time
    
    # Compare with traditional method (ORB + RANSAC)
    print("Estimating homography using traditional ORB + RANSAC...")
    start_time = time.time()
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    if des1 is not None and des2 is not None:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) >= 10:
            src_pts = np.array([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.array([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            H_orb, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        else:
            H_orb = None
    else:
        H_orb = None
    
    orb_time = time.time() - start_time
    
    # Calculate errors
    def homography_error(H1, H2):
        """Calculate the difference between two homography matrices."""
        if H1 is None or H2 is None:
            return float('inf')
        
        # Normalize homography matrices
        H1_norm = H1 / H1[2, 2]
        H2_norm = H2 / H2[2, 2]
        
        return np.linalg.norm(H1_norm - H2_norm, 'fro')
    
    dl_error = homography_error(H_true, H_estimated)
    orb_error = homography_error(H_true, H_orb) if H_orb is not None else float('inf')
    
    # Print results
    print(f"\nResults:")
    print(f"Ground Truth Homography:")
    print(H_true)
    print(f"\nDeep Learning Estimation:")
    print(H_estimated)
    print(f"Error: {dl_error:.6f}")
    print(f"Time: {dl_time:.4f} seconds")
    
    if H_orb is not None:
        print(f"\nORB + RANSAC Estimation:")
        print(H_orb)
        print(f"Error: {orb_error:.6f}")
        print(f"Time: {orb_time:.4f} seconds")
        print(f"\nDeep Learning is {orb_time/dl_time:.2f}x faster")
        print(f"Deep Learning error is {orb_error/dl_error:.2f}x smaller" if dl_error > 0 else "Deep Learning has zero error!")
    else:
        print(f"\nORB + RANSAC failed to find homography")
        print(f"Deep Learning succeeded where traditional method failed!")
    
    # Visualize results
    visualize_results(img1, img2, H_true, H_estimated, H_orb)
    
    return {
        'dl_error': dl_error,
        'orb_error': orb_error,
        'dl_time': dl_time,
        'orb_time': orb_time
    }


def visualize_results(img1, img2, H_true, H_estimated, H_orb):
    """Visualize homography estimation results."""
    print("\nCreating visualization...")
    
    # Create warped images using estimated homographies
    img1_warped_true = cv2.warpPerspective(img1, H_true, (400, 400))
    img1_warped_dl = cv2.warpPerspective(img1, H_estimated, (400, 400))
    
    if H_orb is not None:
        img1_warped_orb = cv2.warpPerspective(img1, H_orb, (400, 400))
    else:
        img1_warped_orb = np.zeros_like(img2)
        cv2.putText(img1_warped_orb, "FAILED", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    
    # Create comparison grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Homography Estimation Comparison', fontsize=16)
    
    # Convert BGR to RGB for matplotlib
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img1_warped_true_rgb = cv2.cvtColor(img1_warped_true, cv2.COLOR_BGR2RGB)
    img1_warped_dl_rgb = cv2.cvtColor(img1_warped_dl, cv2.COLOR_BGR2RGB)
    img1_warped_orb_rgb = cv2.cvtColor(img1_warped_orb, cv2.COLOR_BGR2RGB)
    
    axes[0, 0].imshow(img1_rgb)
    axes[0, 0].set_title('Source Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2_rgb)
    axes[0, 1].set_title('Target Image')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img1_warped_true_rgb)
    axes[0, 2].set_title('Ground Truth Warping')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(img1_warped_dl_rgb)
    axes[1, 0].set_title('Deep Learning Result')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(img1_warped_orb_rgb)
    axes[1, 1].set_title('ORB + RANSAC Result')
    axes[1, 1].axis('off')
    
    # Create difference image
    diff_dl = np.abs(img2.astype(np.float32) - img1_warped_dl.astype(np.float32))
    diff_dl = (diff_dl / diff_dl.max() * 255).astype(np.uint8)
    diff_dl_rgb = cv2.cvtColor(diff_dl, cv2.COLOR_BGR2RGB)
    
    axes[1, 2].imshow(diff_dl_rgb)
    axes[1, 2].set_title('Deep Learning Difference')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('homography_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualization saved to homography_comparison.png")


def benchmark_performance(model_path, num_tests=50):
    """Benchmark performance on multiple test cases."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    estimator = HomographyEstimator(model_path, device)
    
    dl_times = []
    orb_times = []
    dl_errors = []
    orb_errors = []
    
    print(f"Running {num_tests} benchmark tests...")
    
    for i in range(num_tests):
        # Create random test case
        img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        
        # Add some structure
        cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
        cv2.circle(img, (100, 100), 30, (0, 0, 0), -1)
        
        # Random homography
        src_pts = np.array([[20, 20], [180, 20], [180, 180], [20, 180]], dtype=np.float32)
        dst_pts = src_pts + np.random.uniform(-30, 30, src_pts.shape).astype(np.float32)
        H_true = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        img_warped = cv2.warpPerspective(img, H_true, (200, 200))
        
        # Deep learning estimation
        start_time = time.time()
        H_dl = estimator.estimate_homography(img, img_warped)
        dl_time = time.time() - start_time
        dl_times.append(dl_time)
        
        # Calculate error
        if H_dl is not None:
            H_dl_norm = H_dl / H_dl[2, 2]
            H_true_norm = H_true / H_true[2, 2]
            dl_error = np.linalg.norm(H_dl_norm - H_true_norm, 'fro')
            dl_errors.append(dl_error)
        else:
            dl_errors.append(float('inf'))
        
        # Traditional method (simplified for speed)
        start_time = time.time()
        gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)
        
        orb = cv2.ORB_create(nfeatures=100)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        
        if des1 is not None and des2 is not None and len(des1) >= 4 and len(des2) >= 4:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(des1, des2)
            
            if len(matches) >= 4:
                src_pts_orb = np.array([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts_orb = np.array([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                H_orb, _ = cv2.findHomography(src_pts_orb, dst_pts_orb, cv2.RANSAC, 5.0)
                
                if H_orb is not None:
                    H_orb_norm = H_orb / H_orb[2, 2]
                    orb_error = np.linalg.norm(H_orb_norm - H_true_norm, 'fro')
                else:
                    orb_error = float('inf')
            else:
                orb_error = float('inf')
        else:
            orb_error = float('inf')
        
        orb_time = time.time() - start_time
        orb_times.append(orb_time)
        orb_errors.append(orb_error)
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_tests} tests...")
    
    # Calculate statistics
    dl_times = np.array(dl_times)
    orb_times = np.array(orb_times)
    dl_errors = np.array([e for e in dl_errors if e != float('inf')])
    orb_errors = np.array([e for e in orb_errors if e != float('inf')])
    
    print(f"\nBenchmark Results:")
    print(f"Deep Learning:")
    print(f"  Average time: {dl_times.mean():.4f} ± {dl_times.std():.4f} seconds")
    print(f"  Average error: {dl_errors.mean():.6f} ± {dl_errors.std():.6f}")
    print(f"  Success rate: {len(dl_errors)/num_tests*100:.1f}%")
    
    print(f"\nORB + RANSAC:")
    print(f"  Average time: {orb_times.mean():.4f} ± {orb_times.std():.4f} seconds")
    print(f"  Average error: {orb_errors.mean():.6f} ± {orb_errors.std():.6f}")
    print(f"  Success rate: {len(orb_errors)/num_tests*100:.1f}%")
    
    print(f"\nSpeedup: {orb_times.mean()/dl_times.mean():.2f}x")
    if len(dl_errors) > 0 and len(orb_errors) > 0:
        print(f"Accuracy improvement: {orb_errors.mean()/dl_errors.mean():.2f}x")
    
    # Save benchmark results
    results = {
        'dl_times': dl_times.tolist(),
        'orb_times': orb_times.tolist(),
        'dl_errors': dl_errors.tolist(),
        'orb_errors': orb_errors.tolist()
    }
    
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f)
    
    print("Benchmark results saved to benchmark_results.json")


def main():
    parser = argparse.ArgumentParser(description="Deep Image Homography Estimation Demo")
    parser.add_argument("--mode", choices=["train", "inference", "benchmark", "full"], 
                        default="full", help="Demo mode")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--model_path", type=str, default="models/demo_homography_net.pth", 
                        help="Path to trained model")
    parser.add_argument("--image_dir", type=str, default="sample_images", 
                        help="Directory containing training images")
    parser.add_argument("--num_tests", type=int, default=50, 
                        help="Number of benchmark tests")
    
    args = parser.parse_args()
    
    print("🧠 Deep Image Homography Estimation Demo")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print("=" * 60)
    
    if args.mode in ["train", "full"]:
        model_path = demonstrate_training(args.image_dir, args.epochs)
        args.model_path = model_path
    
    if args.mode in ["inference", "full"]:
        if not os.path.exists(args.model_path):
            print(f"Model not found at {args.model_path}. Please train first.")
            return
        demonstrate_inference(args.model_path)
    
    if args.mode in ["benchmark", "full"]:
        if not os.path.exists(args.model_path):
            print(f"Model not found at {args.model_path}. Please train first.")
            return
        benchmark_performance(args.model_path, args.num_tests)
    
    print("\n" + "="*60)
    print("Demo completed! Check the generated files:")
    print("- demo_training_history.png: Training progress")
    print("- homography_comparison.png: Visual comparison")
    print("- benchmark_results.json: Performance metrics")
    print("- test_img1.jpg, test_img2.jpg: Test images")
    print("="*60)


if __name__ == "__main__":
    main()