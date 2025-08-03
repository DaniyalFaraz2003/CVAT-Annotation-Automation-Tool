import cv2
import os
import logging
import shutil
import subprocess
from tqdm import tqdm
from ultralytics import YOLO

# --- Logging Setup ---
logging.basicConfig(
    format='[%(levelname)s] %(asctime)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

class AnnotationAutomationTool:
    def __init__(self, config):
        """
        Initialize the annotation automation tool with configuration
        
        Args:
            config (dict): Configuration dictionary containing all parameters
        """
        self.config = config
        self.setup_directories()
        self.model = YOLO(config['model_path'])
        
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            self.config['frames_dir'],
            self.config['labels_dir'],
            self.config['yolo_dataset_dir'],
            f"{self.config['yolo_dataset_dir']}/images",
            f"{self.config['yolo_dataset_dir']}/labels",
            self.config['cvat_output_dir']
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logging.info(f"Created directory: {directory}")
    
    def extract_frames(self):
        """Extract frames from video at specified FPS"""
        logging.info("=== STEP 1: Extracting frames from video ===")
        
        cap = cv2.VideoCapture(self.config['video_path'])
        if not cap.isOpened():
            logging.error("❌ Could not open input video.")
            return False

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        interval = int(round(original_fps / self.config['target_fps']))
        frame_idx = 0
        saved_idx = 0
        frame_paths = []

        logging.info(f"Original video FPS: {original_fps:.2f}, Target FPS: {self.config['target_fps']}, Extracting every {interval} frames")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval == 0:
                frame_file = os.path.join(self.config['frames_dir'], f"{saved_idx:05d}.jpg")
                cv2.imwrite(frame_file, frame)
                frame_paths.append(frame_file)
                saved_idx += 1

            frame_idx += 1

        cap.release()
        logging.info(f"✅ Extracted {saved_idx} frames at {self.config['target_fps']} FPS to: {self.config['frames_dir']}")
        return frame_paths
    
    def auto_annotate(self, frame_paths):
        """Run YOLO inference on extracted frames"""
        logging.info("=== STEP 2: Running YOLO auto-annotation ===")
        
        for img_path in tqdm(frame_paths, desc="Annotating"):
            results = self.model(img_path)[0]
            h, w = cv2.imread(img_path).shape[:2]

            label_file = os.path.join(self.config['labels_dir'], os.path.basename(img_path).replace(".jpg", ".txt"))
            with open(label_file, "w") as f:
                for box in results.boxes:
                    cls = int(box.cls[0])
                    x1, y1, x2, y2 = box.xyxy[0]
                    xc = ((x1 + x2) / 2) / w
                    yc = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        logging.info(f"✅ YOLO annotations complete. Labels saved in: {self.config['labels_dir']}")
    
    def prepare_yolo_dataset(self):
        """Organize data into YOLO dataset structure"""
        logging.info("=== STEP 3: Preparing YOLO dataset structure ===")
        
        # Save class names
        class_path = os.path.join(self.config['yolo_dataset_dir'], "obj.names")
        with open(class_path, "w") as f:
            for c in self.config['classes']:
                f.write(c + "\n")
        logging.info(f"Saved class labels to: {class_path}")

        # Copy images
        image_count = 0
        for fname in os.listdir(self.config['frames_dir']):
            if fname.endswith('.jpg'):
                src = os.path.join(self.config['frames_dir'], fname)
                dst = os.path.join(self.config['yolo_dataset_dir'], "images", fname)
                shutil.copy(src, dst)
                image_count += 1

        # Copy labels
        label_count = 0
        for fname in os.listdir(self.config['labels_dir']):
            if fname.endswith('.txt'):
                src = os.path.join(self.config['labels_dir'], fname)
                dst = os.path.join(self.config['yolo_dataset_dir'], "labels", fname)
                shutil.copy(src, dst)
                label_count += 1

        logging.info(f"✅ Copied {image_count} images and {label_count} label files to YOLO dataset structure.")
    
    def clean_dataset(self):
        """Filter annotations to keep only specified classes"""
        logging.info("=== STEP 4: Cleaning dataset (filtering classes) ===")
        
        labels_dir = f"{self.config['yolo_dataset_dir']}/labels"
        removed_count = 0
        kept_count = 0

        for filename in os.listdir(labels_dir):
            if not filename.endswith('.txt'):
                continue

            filepath = os.path.join(labels_dir, filename)
            
            with open(filepath, 'r') as file:
                lines = file.readlines()

            # Filter to keep only specified classes
            filtered_lines = []
            for line in lines:
                line = line.strip()
                if line:
                    cls = int(line.split()[0])
                    if cls in self.config['keep_classes']:
                        filtered_lines.append(line + '\n')

            # Write back filtered lines
            with open(filepath, 'w') as file:
                file.writelines(filtered_lines)

            removed = len(lines) - len(filtered_lines)
            kept = len(filtered_lines)
            removed_count += removed
            kept_count += kept

        logging.info(f"✅ Cleaned YOLO labels in '{labels_dir}'")
        logging.info(f"Kept {kept_count} objects, removed {removed_count} from all files.")
    
    def convert_to_cvat(self):
        """Convert YOLO format to CVAT format using Datumaro"""
        logging.info("=== STEP 5: Converting to CVAT format ===")
        
        def run_command(command, description):
            logging.info(description)
            try:
                subprocess.run(command, shell=True, check=True)
                logging.info("✅ Success")
            except subprocess.CalledProcessError as e:
                logging.error(f"❌ Command failed: {e}")
                return False
            return True
        
        command = f"datum convert -if yolo -i {self.config['yolo_dataset_dir']} -f cvat -o {self.config['cvat_output_dir']} --overwrite"
        success = run_command(command, "Converting YOLO to CVAT format...")
        
        if success:
            logging.info(f"🎯 Conversion complete! Find your CVAT dataset at: {self.config['cvat_output_dir']}")
        
        return success
    
    def run_full_pipeline(self):
        """Run the complete annotation automation pipeline"""
        logging.info("🚀 Starting Annotation Automation Pipeline")
        logging.info("=" * 50)
        
        # Step 1: Extract frames
        frame_paths = self.extract_frames()
        if not frame_paths:
            logging.error("❌ Frame extraction failed. Exiting.")
            return False
        
        # Step 2: Auto-annotate
        self.auto_annotate(frame_paths)
        
        # Step 3: Prepare YOLO dataset
        self.prepare_yolo_dataset()
        
        # Step 4: Clean dataset (if enabled)
        if self.config['enable_cleaning']:
            self.clean_dataset()
        
        # Step 5: Convert to CVAT
        if self.config['enable_cvat_conversion']:
            self.convert_to_cvat()
        
        logging.info("=" * 50)
        logging.info("🎉 Annotation Automation Pipeline Complete!")
        return True


def main():
    """Main function with default configuration"""
    
    # Default configuration - modify as needed
    config = {
        # Input/Output paths
        'video_path': 'video11.mp4',
        'frames_dir': 'frames',
        'labels_dir': 'yolo_labels',
        'yolo_dataset_dir': 'yolo_dataset',
        'cvat_output_dir': 'cvat_format',
        
        # Model configuration
        'model_path': 'yolov8n.pt',
        'target_fps': 10,  # Match CVAT import FPS
        
        # Class configuration
        'classes': ['person', 'car', 'bicycle'],  # YOLO class names
        'keep_classes': [0],  # Class indices to keep (0 = person)
        
        # Pipeline options
        'enable_cleaning': True,  # Enable dataset cleaning
        'enable_cvat_conversion': True,  # Enable CVAT conversion
    }
    
    # Create and run the automation tool
    tool = AnnotationAutomationTool(config)
    success = tool.run_full_pipeline()
    
    if success:
        print("\n" + "="*60)
        print("🎯 ANNOTATION AUTOMATION COMPLETE!")
        print("="*60)
        print(f"📁 Frames extracted to: {config['frames_dir']}")
        print(f"🏷️  Labels saved to: {config['labels_dir']}")
        print(f"📊 YOLO dataset prepared at: {config['yolo_dataset_dir']}")
        if config['enable_cvat_conversion']:
            print(f"📋 CVAT format ready at: {config['cvat_output_dir']}")
        print("="*60)
    else:
        print("\n❌ Pipeline failed. Check logs above for details.")


if __name__ == "__main__":
    main() 