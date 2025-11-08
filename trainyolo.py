import os
import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch


# ==================== CONFIGURATION ====================
DATASET_ROOT = dataset.location
OUTPUT_MODEL_NAME = "yolov8n_finetuned.pt"
PROJECT_DIR = "."
CHECKPOINT_DIR = Path(PROJECT_DIR) / "checkpoints"


# Training hyperparameters
EPOCHS = 50
BATCH_SIZE = 16
IMG_SIZE = 640
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
CHECKPOINT_INTERVAL = 30  # Save checkpoint every 30 epochs


print(f"Device: {DEVICE}")


# ==================== STEP 0: CREATE CHECKPOINT DIRECTORY ====================
def setup_checkpoint_dir():
    """Create checkpoint directory if it doesn't exist"""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Checkpoint directory ready: {CHECKPOINT_DIR}")


# ==================== CHECKPOINT SAVE FUNCTION ====================
def save_checkpoint(model, epoch, checkpoint_dir=CHECKPOINT_DIR):
    """
    Save model checkpoint at specific epoch
    Args:
        model: YOLO model to save
        epoch: Current epoch number
        checkpoint_dir: Directory to save checkpoints
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"


    try:
        # Save the model
        model.save(str(checkpoint_path))
        print(f"✓ Checkpoint saved at epoch {epoch}: {checkpoint_path}")
        print(f"  File size: {checkpoint_path.stat().st_size / (1024*1024):.2f} MB")
        return checkpoint_path
    except Exception as e:
        print(f"ERROR: Failed to save checkpoint at epoch {epoch}: {e}")
        return None


# ==================== LOAD CHECKPOINT FUNCTION ====================
def load_checkpoint(checkpoint_path):
    """
    Load model from checkpoint
    Args:
        checkpoint_path: Path to checkpoint file
    Returns:
        Loaded YOLO model
    """
    try:
        model = YOLO(str(checkpoint_path))
        print(f"✓ Loaded checkpoint from: {checkpoint_path}")
        return model
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}")
        return None


# ==================== STEP 1: VERIFY DATASET STRUCTURE ====================
print("\n" + "="*60)
print("STEP 1: Verifying Dataset Structure")
print("="*60)


dataset_path = Path(DATASET_ROOT).resolve()
print(f"Dataset path: {dataset_path}")


if not dataset_path.exists():
    print(f"ERROR: Dataset directory not found at {dataset_path}")
    exit(1)


required_dirs = ['train', 'valid']
for dir_name in required_dirs:
    dir_path = dataset_path / dir_name
    if not (dir_path / 'images').exists() or not (dir_path / 'labels').exists():
        print(f"ERROR: {dir_name}/images or {dir_name}/labels not found")
        exit(1)


print("✓ Dataset structure verified")
print(f"  - train/images: {len(list((dataset_path / 'train' / 'images').glob('*')))} images")
print(f"  - train/labels: {len(list((dataset_path / 'train' / 'labels').glob('*')))} labels")
print(f"  - valid/images: {len(list((dataset_path / 'valid' / 'images').glob('*')))} images")
print(f"  - valid/labels: {len(list((dataset_path / 'valid' / 'labels').glob('*')))} labels")


# ==================== STEP 2: VERIFY/CREATE data.yaml ====================
print("\n" + "="*60)
print("STEP 2: Verifying/Creating data.yaml")
print("="*60)


data_yaml_path = dataset_path / "data.yaml"


if data_yaml_path.exists():
    print(f"✓ Found existing data.yaml at {data_yaml_path}")
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    print(f"  Classes: {data.get('names', {})}")
else:
    print("⚠ data.yaml not found. Creating a default one...")
    data = {
        'train': str(dataset_path / 'train' / 'images'),
        'val': str(dataset_path / 'valid' / 'images'),
        'nc': 1,
        'names': ['obstacle']
    }
    with open(data_yaml_path, 'w') as f:
        yaml.safe_dump(data, f)
    print(f"✓ Created data.yaml with default configuration")


# ==================== STEP 3: SETUP CHECKPOINT DIRECTORY ====================
print("\n" + "="*60)
print("STEP 3: Setting Up Checkpoint Directory")
print("="*60)


setup_checkpoint_dir()


# ==================== STEP 4: CHECK FOR EXISTING CHECKPOINTS ====================
print("\n" + "="*60)
print("STEP 4: Checking for Existing Checkpoints")
print("="*60)


checkpoint_files = sorted(CHECKPOINT_DIR.glob("checkpoint_epoch_*.pt"))
resume_from_checkpoint = None
start_epoch = 0


if checkpoint_files:
    latest_checkpoint = checkpoint_files[-1]
    print(f"✓ Found {len(checkpoint_files)} checkpoint(s)")
    print(f"  Latest: {latest_checkpoint.name}")


    # Extract epoch number from filename
    epoch_num = int(latest_checkpoint.name.split('_')[-1].replace('.pt', ''))
    start_epoch = epoch_num
    resume_from_checkpoint = latest_checkpoint


    print(f"  Resuming training from epoch {start_epoch + 1}")
else:
    print("ℹ No checkpoints found. Starting fresh training.")


# ==================== STEP 5: LOAD MODEL ====================
print("\n" + "="*60)
print("STEP 5: Loading Model")
print("="*60)


try:
    if resume_from_checkpoint:
        print(f"Loading from checkpoint: {resume_from_checkpoint}")
        model = load_checkpoint(resume_from_checkpoint)
        if model is None:
            print("ERROR: Failed to load checkpoint. Exiting.")
            exit(1)
    else:
        print("Loading pretrained YOLOv8n model...")
        model = YOLO('yolov8n.pt')
        print("✓ YOLOv8n model loaded successfully")
except Exception as e:
    print(f"ERROR: Failed to load model: {e}")
    exit(1)


# ==================== STEP 6: TRAIN WITH CHECKPOINT INTERVALS ====================
print("\n" + "="*60)
print("STEP 6: Training with Checkpoint Intervals")
print("="*60)
print(f"Training from epoch {start_epoch + 1} to {EPOCHS}")
print(f"Saving checkpoints every {CHECKPOINT_INTERVAL} epochs")


# Calculate checkpoint epochs
checkpoint_epochs = []
current = CHECKPOINT_INTERVAL
while current <= EPOCHS:
    checkpoint_epochs.append(current)
    current += CHECKPOINT_INTERVAL

print(f"Checkpoints will be saved at epochs: {checkpoint_epochs}")

# SIMPLIFIED APPROACH: Use Ultralytics built-in resume functionality
try:
    # Calculate remaining epochs
    remaining_epochs = EPOCHS - start_epoch
    
    if remaining_epochs > 0:
        print(f"\nTraining for {remaining_epochs} more epochs...")
        
        # Train for remaining epochs - let Ultralytics handle resume internally
        results = model.train(
            data=str(data_yaml_path),
            epochs=EPOCHS,  # Total target epochs
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            workers=4,
            device=DEVICE,
            patience=10,
            save=True,
            project=PROJECT_DIR,
            name='obstacle_detection',
            verbose=True,
            close_mosaic=10,
            augment=True,
            resume=True if start_epoch > 0 else False  # Let Ultralytics handle resume
        )

        print(f"\n✓ Training completed up to epoch {EPOCHS}")

        # Save final checkpoint at the end
        save_checkpoint(model, EPOCHS)
        
        print(f"✓ Final checkpoint saved at epoch {EPOCHS}")
        
    else:
        print(f"✓ Training already completed (current: {start_epoch}, target: {EPOCHS})")

except KeyboardInterrupt:
    print(f"\n⚠ Training interrupted by user")
    
    # Find the latest training epoch from Ultralytics logs
    current_epoch = start_epoch
    try:
        # Get the latest training run
        runs_dir = Path(PROJECT_DIR) / 'obstacle_detection'
        if runs_dir.exists():
            # Find the latest run
            run_dirs = sorted(runs_dir.glob('train*'))
            if run_dirs:
                latest_run = run_dirs[-1]
                # Read the results file to find current epoch
                results_file = latest_run / 'results.csv'
                if results_file.exists():
                    import pandas as pd
                    df = pd.read_csv(results_file)
                    current_epoch = start_epoch + len(df)
                    print(f"Current training epoch: {current_epoch}")
    except Exception as e:
        print(f"Could not determine current epoch: {e}")
    
    # Save checkpoint at current epoch
    save_checkpoint(model, current_epoch)
    print(f"Checkpoint saved at epoch {current_epoch}. You can resume training later.")
    exit(0)
    
except Exception as e:
    print(f"\nERROR during training: {e}")
    # Save emergency checkpoint
    save_checkpoint(model, start_epoch)
    print(f"Emergency checkpoint saved at epoch {start_epoch}. You can resume training later.")
    exit(1)


# ==================== STEP 7: EVALUATE THE MODEL ====================
print("\n" + "="*60)
print("STEP 7: Evaluating Model")
print("="*60)


try:
    metrics = model.val()
    print("✓ Validation completed")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
except Exception as e:
    print(f"WARNING: Evaluation failed: {e}")


# ==================== STEP 8: SAVE FINAL MODEL ====================
print("\n" + "="*60)
print("STEP 8: Saving Final Fine-Tuned Model")
print("="*60)


best_model_path = Path(PROJECT_DIR) / 'obstacle_detection' / 'weights' / 'best.pt'


if best_model_path.exists():
    output_path = Path(PROJECT_DIR) / OUTPUT_MODEL_NAME
    try:
        shutil.copy(str(best_model_path), str(output_path))
        print(f"✓ Fine-tuned model saved to: {output_path}")
        print(f"  File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"ERROR: Failed to save model: {e}")
        exit(1)
else:
    print(f"ERROR: Best model not found at {best_model_path}")
    exit(1)


# ==================== STEP 9: TEST INFERENCE ====================
print("\n" + "="*60)
print("STEP 9: Testing Fine-Tuned Model Inference")
print("="*60)


try:
    fine_tuned_model = YOLO(str(output_path))
    print(f"✓ Loaded fine-tuned model from {output_path}")


    val_images = list((dataset_path / 'valid' / 'images').glob('*.jpg'))
    if val_images:
        sample_image = str(val_images[0])
        print(f"\nRunning inference on sample image: {Path(sample_image).name}")


        results = fine_tuned_model.predict(source=sample_image, conf=0.5, imgsz=IMG_SIZE)


        for result in results:
            print(f"  Detections: {len(result.boxes)}")
            if len(result.boxes) > 0:
                for box in result.boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    print(f"    - Class: {class_id}, Confidence: {confidence:.2f}")
    else:
        print("⚠ No validation images found for testing")


except Exception as e:
    print(f"ERROR during inference: {e}")


# ==================== FINAL SUMMARY ====================
print("\n" + "="*60)
print("✅ FINE-TUNING PIPELINE COMPLETE")
print("="*60)
print(f"\n📊 Summary:")
print(f"  Input Dataset: {dataset_path}")
print(f"  Total Training Epochs: {EPOCHS}")
print(f"  Checkpoint Interval: {CHECKPOINT_INTERVAL} epochs")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Image Size: {IMG_SIZE}")
print(f"  Device: {DEVICE}")
print(f"\n📁 Output:")
print(f"  Final Model: {output_path}")
print(f"  Checkpoints Directory: {CHECKPOINT_DIR}")
print(f"  Number of Checkpoints: {len(list(CHECKPOINT_DIR.glob('checkpoint_epoch_*.pt')))}")
print(f"  Training Logs: {Path(PROJECT_DIR) / 'obstacle_detection'}")
print(f"\n🔄 Checkpoint Files:")
for cp in sorted(CHECKPOINT_DIR.glob('checkpoint_epoch_*.pt')):
    print(f"  - {cp.name} ({cp.stat().st_size / (1024*1024):.2f} MB)")
print(f"\n🎯 Next Steps:")
print(f"  1. Use the model: from ultralytics import YOLO")
print(f"  2. Load it: model = YOLO('{output_path}')")
print(f"  3. Run inference: results = model.predict(source='image.jpg')")
print(f"\n📌 Resume Training:")
print(f"  Checkpoints are saved every {CHECKPOINT_INTERVAL} epochs in {CHECKPOINT_DIR}")
print(f"  Next run will automatically resume from the latest checkpoint!")
print("\n" + "="*60)