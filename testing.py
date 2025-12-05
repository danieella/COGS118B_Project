import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
import pandas as pd
from tqdm import tqdm
import open_clip

# ------------------------------------------------------------------------------
# Device
# ------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------------------------------------------------------
# Load OpenCLIP model
# ------------------------------------------------------------------------------
print("Loading OpenCLIP model...")
model_name = "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

# Load model with open_clip
model, preprocess = open_clip.create_model_from_pretrained(model_name)
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model = model.to(device).eval()  # Set to evaluation mode

print("Model loaded successfully!")
print(f"Model dtype: {next(model.parameters()).dtype}")

# ------------------------------------------------------------------------------
# Embedding Helpers
# ------------------------------------------------------------------------------
def embed_images(image_paths, batch_size=8):
    """Return (N,1024) image embeddings"""
    all_embeddings = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
        batch_paths = image_paths[i:i+batch_size]
        images = []
        
        # Load and preprocess images
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                img = preprocess(img)  # Use OpenCLIP's preprocessing
                images.append(img)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                # Add a dummy image if loading fails
                images.append(torch.zeros_like(preprocess(Image.new('RGB', (224, 224), color='black'))))
        
        # Stack images into batch
        images = torch.stack(images).to(device)
        
        # Get embeddings
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_embeddings.append(image_features.cpu())
    
    return torch.cat(all_embeddings, dim=0)

def embed_texts(texts, batch_size=32):
    """Return (N,1024) text embeddings"""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        tokenized = open_clip.tokenize(batch_texts).to(device)
        
        # Get embeddings
        with torch.no_grad():
            text_features = model.encode_text(tokenized)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            all_embeddings.append(text_features.cpu())
    
    return torch.cat(all_embeddings, dim=0)

# ------------------------------------------------------------------------------
# Load Bird Class Names
# ------------------------------------------------------------------------------
TRAIN_DIR = "/Users/milin/Downloads/COGS118B_FA25_Project-main_2/bird-species-220/Train"

# Check if directory exists
if not os.path.exists(TRAIN_DIR):
    print(f"Error: Training directory not found: {TRAIN_DIR}")
    # Try to find the directory
    possible_paths = [
        "/Users/milin/Downloads/bird-species-220/Train",
        "./bird-species-220/Train",
        "./Train"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            TRAIN_DIR = path
            print(f"Found training directory at: {TRAIN_DIR}")
            break

bird_classes = sorted([d for d in os.listdir(TRAIN_DIR) 
                      if os.path.isdir(os.path.join(TRAIN_DIR, d))])
print(f"Detected {len(bird_classes)} bird species.")

# Create text prompts for each class
class_prompts = [f"a photo of a {cls.replace('_', ' ')}" for cls in bird_classes]

# Compute CLIP text embeddings (prototypes)
print("Computing text embeddings for bird classes...")
text_embeds = embed_texts(class_prompts)  # (C, 1024)
print(f"Text embeddings shape: {text_embeds.shape}")

# ------------------------------------------------------------------------------
# Classification Function
# ------------------------------------------------------------------------------
def classify(image_path):
    """Returns the predicted bird species for a single image."""
    # Get image embedding
    img_vec = embed_images([image_path])  # (1, 1024)
    
    # Compute cosine similarity
    sims = torch.matmul(img_vec, text_embeds.T)
    idx = sims.argmax(dim=1).item()
    
    return bird_classes[idx], sims[0, idx].item()

def classify_batch(image_paths):
    """Classify multiple images at once."""
    if not image_paths:
        return [], []
    
    # Get image embeddings for all images
    img_vecs = embed_images(image_paths)  # (N, 1024)
    
    # Compute cosine similarity with all text embeddings
    sims = torch.matmul(img_vecs, text_embeds.T)  # (N, C)
    
    # Get predictions
    pred_indices = sims.argmax(dim=1)
    pred_classes = [bird_classes[idx] for idx in pred_indices]
    pred_scores = [sims[i, idx].item() for i, idx in enumerate(pred_indices)]
    
    return pred_classes, pred_scores

# ------------------------------------------------------------------------------
# Evaluate on Test Folder
# ------------------------------------------------------------------------------
TEST_DIR = "/Users/milin/Downloads/COGS118B_FA25_Project-main_2/bird-species-220/Test"

# Check if test directory exists
if not os.path.exists(TEST_DIR):
    print(f"Error: Test directory not found: {TEST_DIR}")
    # Try to find the directory
    possible_paths = [
        "/Users/milin/Downloads/bird-species-220/Test",
        "./bird-species-220/Test",
        "./Test"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            TEST_DIR = path
            print(f"Found test directory at: {TEST_DIR}")
            break

def run_evaluation():
    """Run evaluation on all test images."""
    # Collect all test images
    all_images = []
    for root, _, files in os.walk(TEST_DIR):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                all_images.append(os.path.join(root, f))
    
    print(f"Found {len(all_images)} test images.")
    
    if len(all_images) == 0:
        print("No test images found. Please check the TEST_DIR path.")
        return pd.DataFrame()
    
    # Classify in batches
    batch_size = 16
    results = []
    
    for i in tqdm(range(0, len(all_images), batch_size), desc="Classifying images"):
        batch_paths = all_images[i:i+batch_size]
        pred_classes, pred_scores = classify_batch(batch_paths)
        
        for img_path, pred, score in zip(batch_paths, pred_classes, pred_scores):
            # Get true label from directory structure
            rel_path = os.path.relpath(img_path, TEST_DIR)
            true_class = rel_path.split(os.sep)[0] if os.sep in rel_path else "unknown"
            
            # Check if prediction is correct
            is_correct = (pred == true_class)
            
            results.append({
                "image": img_path,
                "true_class": true_class,
                "prediction": pred,
                "similarity": score,
                "correct": is_correct
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate accuracy
    if len(df) > 0:
        accuracy = df['correct'].mean() * 100
        print(f"\nClassification Accuracy: {accuracy:.2f}%")
        print(f"Total images: {len(df)}")
        print(f"Correct predictions: {df['correct'].sum()}")
    
    # Save results
    df.to_csv("clip_bird_predictions.csv", index=False)
    print("\nSaved predictions to clip_bird_predictions.csv")
    
    return df

# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Simple test: classify one random training image
    if bird_classes:
        example_class = bird_classes[0]
        example_folder = os.path.join(TRAIN_DIR, example_class)
        
        if os.path.exists(example_folder):
            # Find an image in the folder
            image_files = [f for f in os.listdir(example_folder) 
                          if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            
            if image_files:
                example_img = image_files[0]
                example_path = os.path.join(example_folder, example_img)
                
                print(f"\nExample classification: {example_path}")
                pred, score = classify(example_path)
                print(f"---> Predicted: {pred} (similarity = {score:.4f})")
                print(f"---> True class: {example_class}")
                print(f"---> Correct: {pred == example_class}")
            else:
                print(f"No images found in {example_folder}")
        else:
            print(f"Example folder not found: {example_folder}")
    else:
        print("No bird classes found. Check your TRAIN_DIR path.")
    
    # Full evaluation
    print("\n" + "="*50)
    print("Starting full evaluation...")
    print("="*50)
    df = run_evaluation()
    
    if len(df) > 0:
        # Show some sample results
        print("\nSample predictions:")
        print(df[['image', 'true_class', 'prediction', 'similarity', 'correct']].head(10))