import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from tqdm import tqdm

def prepare_aptos_data(csv_path, images_dir, output_dir):
    """
    Prepare APTOS 2019 dataset for training by organizing images into folders by class
    
    Args:
        csv_path (str): Path to train.csv file
        images_dir (str): Path to directory containing images
        output_dir (str): Output directory for organized data
    """
    
    print("📊 Reading CSV file...")
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    print(f"Dataset info:")
    print(f"Total images: {len(df)}")
    print(f"Class distribution:")
    print(df['diagnosis'].value_counts().sort_index())
    
    # Create output directories for each stage
    print("\n📁 Creating directories...")
    for stage in range(5):
        stage_dir = os.path.join(output_dir, f'stage_{stage}')
        os.makedirs(stage_dir, exist_ok=True)
        print(f"Created: {stage_dir}")
    
    # Copy images to appropriate directories
    print(f"\n🖼️ Organizing {len(df)} images...")
    copied_count = 0
    missing_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        image_id = row['id_code']
        stage = row['diagnosis']
        
        # Try different extensions
        src_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_path = os.path.join(images_dir, f'{image_id}{ext}')
            if os.path.exists(potential_path):
                src_path = potential_path
                break
        
        if src_path is None:
            missing_count += 1
            print(f"❌ Missing: {image_id}")
            continue
        
        # Destination path
        dst_dir = os.path.join(output_dir, f'stage_{stage}')
        dst_path = os.path.join(dst_dir, f'{image_id}{os.path.splitext(src_path)[1]}')
        
        # Copy image
        try:
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        except Exception as e:
            print(f"❌ Error copying {image_id}: {e}")
    
    print(f"\n✅ Data preparation complete!")
    print(f"Images copied: {copied_count}")
    print(f"Images missing: {missing_count}")
    print(f"Organized data saved in: {output_dir}")
    
    # Print final distribution
    print(f"\nFinal distribution:")
    for stage in range(5):
        stage_dir = os.path.join(output_dir, f'stage_{stage}')
        count = len(os.listdir(stage_dir))
        print(f"Stage {stage}: {count} images")

def create_balanced_dataset(organized_data_dir, balanced_output_dir, samples_per_class=1000):
    """
    Create a balanced dataset by sampling equal number of images from each class
    
    Args:
        organized_data_dir (str): Directory with organized data
        balanced_output_dir (str): Output directory for balanced data
        samples_per_class (int): Number of samples per class
    """
    
    print(f"🎯 Creating balanced dataset with {samples_per_class} samples per class...")
    
    # Create output directories
    for stage in range(5):
        stage_dir = os.path.join(balanced_output_dir, f'stage_{stage}')
        os.makedirs(stage_dir, exist_ok=True)
    
    for stage in range(5):
        src_dir = os.path.join(organized_data_dir, f'stage_{stage}')
        dst_dir = os.path.join(balanced_output_dir, f'stage_{stage}')
        
        if not os.path.exists(src_dir):
            print(f"❌ Source directory not found: {src_dir}")
            continue
        
        # Get all images in the class
        images = os.listdir(src_dir)
        available_samples = len(images)
        
        # Sample images
        if available_samples >= samples_per_class:
            sampled_images = np.random.choice(images, samples_per_class, replace=False)
        else:
            # Use all available images and duplicate some if needed
            sampled_images = images.copy()
            while len(sampled_images) < samples_per_class:
                sampled_images.extend(np.random.choice(images, 
                                    min(len(images), samples_per_class - len(sampled_images)), 
                                    replace=False))
        
        # Copy sampled images
        for i, img_name in enumerate(sampled_images):
            src_path = os.path.join(src_dir, img_name)
            # Add index to filename to avoid duplicates
            name, ext = os.path.splitext(img_name)
            dst_path = os.path.join(dst_dir, f"{name}_{i}{ext}")
            shutil.copy2(src_path, dst_path)
        
        print(f"Stage {stage}: {len(sampled_images)} images (from {available_samples} available)")
    
    print(f"✅ Balanced dataset created in: {balanced_output_dir}")

def preprocess_images(input_dir, output_dir, target_size=(224, 224)):
    """
    Preprocess images by resizing and applying basic enhancements
    
    Args:
        input_dir (str): Directory containing organized images
        output_dir (str): Output directory for preprocessed images
        target_size (tuple): Target size for images (width, height)
    """
    
    print(f"🔄 Preprocessing images to size {target_size}...")
    
    # Create output directories
    for stage in range(5):
        stage_dir = os.path.join(output_dir, f'stage_{stage}')
        os.makedirs(stage_dir, exist_ok=True)
    
    total_processed = 0
    
    for stage in range(5):
        src_dir = os.path.join(input_dir, f'stage_{stage}')
        dst_dir = os.path.join(output_dir, f'stage_{stage}')
        
        if not os.path.exists(src_dir):
            continue
        
        images = os.listdir(src_dir)
        print(f"\nProcessing Stage {stage}: {len(images)} images")
        
        for img_name in tqdm(images, desc=f"Stage {stage}"):
            src_path = os.path.join(src_dir, img_name)
            dst_path = os.path.join(dst_dir, img_name)
            
            try:
                # Read image
                img = cv2.imread(src_path)
                if img is None:
                    print(f"❌ Could not read: {img_name}")
                    continue
                
                # Resize image
                img_resized = cv2.resize(img, target_size)
                
                # Apply basic enhancement (optional)
                # Convert to LAB color space for better contrast enhancement
                lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                l_enhanced = clahe.apply(l)
                
                # Merge channels and convert back to BGR
                enhanced_lab = cv2.merge([l_enhanced, a, b])
                img_final = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                
                # Save preprocessed image
                cv2.imwrite(dst_path, img_final)
                total_processed += 1
                
            except Exception as e:
                print(f"❌ Error processing {img_name}: {e}")
    
    print(f"✅ Preprocessing complete! Processed {total_processed} images")
    print(f"Preprocessed images saved in: {output_dir}")

def analyze_dataset(data_dir):
    """
    Analyze the dataset and provide statistics
    
    Args:
        data_dir (str): Directory containing organized data
    """
    
    print("📈 Analyzing dataset...")
    
    total_images = 0
    stage_counts = {}
    
    for stage in range(5):
        stage_dir = os.path.join(data_dir, f'stage_{stage}')
        if os.path.exists(stage_dir):
            count = len(os.listdir(stage_dir))
            stage_counts[stage] = count
            total_images += count
        else:
            stage_counts[stage] = 0
    
    print(f"\n📊 Dataset Statistics:")
    print(f"Total images: {total_images}")
    print(f"Distribution:")
    
    stage_names = {
        0: "No DR (Healthy)",
        1: "Mild DR",
        2: "Moderate DR",
        3: "Severe DR",
        4: "Proliferative DR"
    }
    
    for stage in range(5):
        count = stage_counts[stage]
        percentage = (count / total_images * 100) if total_images > 0 else 0
        print(f"  Stage {stage} ({stage_names[stage]}): {count} images ({percentage:.1f}%)")
    
    # Check for class imbalance
    if total_images > 0:
        max_count = max(stage_counts.values())
        min_count = min(stage_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"\n⚖️ Class Imbalance Analysis:")
        print(f"Imbalance ratio: {imbalance_ratio:.2f}")
        if imbalance_ratio > 5:
            print("⚠️  High class imbalance detected! Consider using balanced sampling.")
        elif imbalance_ratio > 2:
            print("⚠️  Moderate class imbalance detected.")
        else:
            print("✅ Dataset is relatively balanced.")

def main():
    """
    Main function to demonstrate usage
    """
    
    # Configuration
    APTOS_CSV = 'APTOS_2019_dataset/train.csv'
    APTOS_IMAGES = 'APTOS_2019_dataset/train_images/'
    ORGANIZED_DIR = 'organized_data/'
    BALANCED_DIR = 'balanced_data/'
    PREPROCESSED_DIR = 'preprocessed_data/'
    
    print("🚀 Starting APTOS 2019 dataset preparation...")
    
    # Step 1: Organize data by class
    if os.path.exists(APTOS_CSV) and os.path.exists(APTOS_IMAGES):
        prepare_aptos_data(APTOS_CSV, APTOS_IMAGES, ORGANIZED_DIR)
    else:
        print("❌ APTOS dataset files not found!")
        print(f"Please ensure these files exist:")
        print(f"  - {APTOS_CSV}")
        print(f"  - {APTOS_IMAGES}")
        return
    
    # Step 2: Analyze original dataset
    analyze_dataset(ORGANIZED_DIR)
    
    # Step 3: Create balanced dataset (optional)
    create_balanced = input("\n🤔 Create balanced dataset? (y/n): ").lower().strip() == 'y'
    if create_balanced:
        samples_per_class = int(input("Enter samples per class (default 1000): ") or 1000)
        create_balanced_dataset(ORGANIZED_DIR, BALANCED_DIR, samples_per_class)
        analyze_dataset(BALANCED_DIR)
    
    # Step 4: Preprocess images (optional)
    preprocess = input("\n🔄 Preprocess images? (y/n): ").lower().strip() == 'y'
    if preprocess:
        source_dir = BALANCED_DIR if create_balanced else ORGANIZED_DIR
        preprocess_images(source_dir, PREPROCESSED_DIR)
    
    print("\n✅ Dataset preparation complete!")
    print("You can now use the organized data for training your model.")

if __name__ == "__main__":
    main()