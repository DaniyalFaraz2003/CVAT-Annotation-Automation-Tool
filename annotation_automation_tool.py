import cv2
import os
import logging
import shutil
import subprocess
import argparse
import sys
from tqdm import tqdm
from ultralytics import YOLO

# --- Logging Setup ---
logging.basicConfig(
    format='[%(levelname)s] %(asctime)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

def clear_console():
    """Clear the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    """Print the tool banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    ANNOTATION AUTOMATION TOOL                ║
║                        Powered by YOLOv8                     ║
║                    For CVAT Dataset Preparation              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def get_video_info(video_path):
    """Get video information including FPS"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    return {
        'fps': fps,
        'frame_count': frame_count,
        'duration': duration,
        'width': width,
        'height': height
    }

def interactive_config():
    """Interactive configuration setup"""
    print_banner()
    print("🎯 Welcome to the Annotation Automation Tool!")
    print("Let's configure your annotation pipeline...\n")
    
    config = {}
    
    # Video path
    print("📹 VIDEO CONFIGURATION")
    print("-" * 30)
    
    while True:
        video_path = input("Enter video file path (or press Enter for 'video11.mp4'): ").strip()
        if not video_path:
            video_path = 'video11.mp4'
        
        if os.path.exists(video_path):
            video_info = get_video_info(video_path)
            if video_info:
                print(f"✅ Video found! FPS: {video_info['fps']:.2f}, Duration: {video_info['duration']:.2f}s")
                config['video_path'] = video_path
                config['original_fps'] = video_info['fps']
                break
            else:
                print("❌ Could not read video file. Please check the file format.")
        else:
            print(f"❌ File not found: {video_path}")
    
    # Target FPS
    print(f"\n🎬 FRAME EXTRACTION")
    print("-" * 30)
    print(f"Original video FPS: {video_info['fps']:.2f}")
    
    while True:
        try:
            target_fps_input = input("Enter target FPS for frame extraction (or press Enter for 10 FPS): ").strip()
            if not target_fps_input:
                target_fps = 10
            else:
                target_fps = float(target_fps_input)
            
            if target_fps > 0 and target_fps <= video_info['fps']:
                config['target_fps'] = target_fps
                break
            else:
                print(f"❌ Target FPS must be between 0 and {video_info['fps']:.2f}")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Model selection
    print(f"\n🤖 YOLO MODEL CONFIGURATION")
    print("-" * 30)
    
    model_options = {
        '1': 'yolov8n.pt (nano - fastest)',
        '2': 'yolov8s.pt (small)',
        '3': 'yolov8m.pt (medium)',
        '4': 'yolov8l.pt (large)',
        '5': 'yolov8x.pt (xlarge - most accurate)',
        '6': 'Custom model path'
    }
    
    for key, value in model_options.items():
        print(f"{key}. {value}")
    
    while True:
        model_choice = input("\nSelect model (1-6): ").strip()
        if model_choice in model_options:
            if model_choice == '6':
                model_path = input("Enter custom model path: ").strip()
                if os.path.exists(model_path):
                    config['model_path'] = model_path
                    break
                else:
                    print("❌ Model file not found!")
            else:
                model_path = model_options[model_choice].split(' ')[0]
                config['model_path'] = model_path
                break
        else:
            print("❌ Please select a valid option (1-6)")
    
    # Output directories
    print(f"\n📁 OUTPUT CONFIGURATION")
    print("-" * 30)
    
    frames_dir = input("Frames directory (or press Enter for 'frames'): ").strip() or 'frames'
    labels_dir = input("Labels directory (or press Enter for 'yolo_labels'): ").strip() or 'yolo_labels'
    yolo_dataset_dir = input("YOLO dataset directory (or press Enter for 'yolo_dataset'): ").strip() or 'yolo_dataset'
    cvat_output_dir = input("CVAT output directory (or press Enter for 'cvat_format'): ").strip() or 'cvat_format'
    
    config.update({
        'frames_dir': frames_dir,
        'labels_dir': labels_dir,
        'yolo_dataset_dir': yolo_dataset_dir,
        'cvat_output_dir': cvat_output_dir
    })
    
    # Class configuration
    print(f"\n🏷️  CLASS CONFIGURATION")
    print("-" * 30)
    
    class_options = {
        '1': 'Person only (class 0)',
        '2': 'Person + Car (classes 0, 2)',
        '3': 'Person + Car + Bicycle (classes 0, 2, 1)',
        '4': 'All COCO classes (0-79)',
        '5': 'Custom classes'
    }
    
    for key, value in class_options.items():
        print(f"{key}. {value}")
    
    while True:
        class_choice = input("\nSelect class configuration (1-5): ").strip()
        if class_choice == '1':
            config['classes'] = ['person']
            config['keep_classes'] = [0]
            break
        elif class_choice == '2':
            config['classes'] = ['person', 'car']
            config['keep_classes'] = [0, 2]
            break
        elif class_choice == '3':
            config['classes'] = ['person', 'car', 'bicycle']
            config['keep_classes'] = [0, 2, 1]
            break
        elif class_choice == '4':
            config['classes'] = ['person'] + [f'class_{i}' for i in range(1, 80)]
            config['keep_classes'] = list(range(80))
            break
        elif class_choice == '5':
            custom_classes = input("Enter class names separated by commas: ").strip()
            config['classes'] = [c.strip() for c in custom_classes.split(',')]
            keep_input = input("Enter class indices to keep separated by commas: ").strip()
            config['keep_classes'] = [int(c.strip()) for c in keep_input.split(',')]
            break
        else:
            print("❌ Please select a valid option (1-5)")
    
    # Pipeline options
    print(f"\n⚙️  PIPELINE OPTIONS")
    print("-" * 30)
    
    enable_cleaning = input("Enable dataset cleaning/filtering? (y/n, default: y): ").strip().lower()
    config['enable_cleaning'] = enable_cleaning != 'n'
    
    enable_cvat = input("Enable CVAT format conversion? (y/n, default: y): ").strip().lower()
    config['enable_cvat_conversion'] = enable_cvat != 'n'
    
    return config

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Annotation Automation Tool - Convert videos to CVAT-ready datasets using YOLOv8',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python annotation_automation_tool.py                    # Interactive mode
  python annotation_automation_tool.py --video video.mp4  # Quick start with video
  python annotation_automation_tool.py --config config.json # Load from config file
        """
    )
    
    parser.add_argument('--video', '-v', help='Input video file path')
    parser.add_argument('--model', '-m', help='YOLO model path (default: yolov8n.pt)')
    parser.add_argument('--fps', '-f', type=float, help='Target FPS for frame extraction (default: 10)')
    parser.add_argument('--classes', '-c', help='Comma-separated class names')
    parser.add_argument('--keep-classes', '-k', help='Comma-separated class indices to keep')
    parser.add_argument('--output-dir', '-o', help='Output directory prefix')
    parser.add_argument('--no-cleaning', action='store_true', help='Disable dataset cleaning')
    parser.add_argument('--no-cvat', action='store_true', help='Disable CVAT conversion')
    parser.add_argument('--config', help='Load configuration from JSON file')
    parser.add_argument('--interactive', '-i', action='store_true', help='Force interactive mode')
    
    return parser.parse_args()

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

        original_fps = self.config.get('original_fps', cap.get(cv2.CAP_PROP_FPS))
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
        
        # Clear console before starting annotation
        
        
        for img_path in tqdm(frame_paths, desc="Annotating", unit="frame"):
            clear_console()
            print("🤖 Running YOLO auto-annotation...")
            print("=" * 50)
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
    """Main function with command line interface"""
    args = parse_arguments()
    
    # Determine configuration method
    if args.interactive or (not args.video and not args.config):
        # Interactive mode
        config = interactive_config()
    else:
        # Command line mode
        config = {
            'video_path': args.video or 'video11.mp4',
            'model_path': args.model or 'yolov8n.pt',
            'target_fps': args.fps or 10,
            'frames_dir': 'frames',
            'labels_dir': 'yolo_labels',
            'yolo_dataset_dir': 'yolo_dataset',
            'cvat_output_dir': 'cvat_format',
            'classes': ['person', 'car', 'bicycle'],
            'keep_classes': [0],
            'enable_cleaning': not args.no_cleaning,
            'enable_cvat_conversion': not args.no_cvat,
        }
        
        # Override with command line arguments
        if args.classes:
            config['classes'] = [c.strip() for c in args.classes.split(',')]
        if args.keep_classes:
            config['keep_classes'] = [int(c.strip()) for c in args.keep_classes.split(',')]
        if args.output_dir:
            config['frames_dir'] = f"{args.output_dir}_frames"
            config['labels_dir'] = f"{args.output_dir}_labels"
            config['yolo_dataset_dir'] = f"{args.output_dir}_dataset"
            config['cvat_output_dir'] = f"{args.output_dir}_cvat"
    
    # Get video info for FPS
    if 'original_fps' not in config:
        video_info = get_video_info(config['video_path'])
        if video_info:
            config['original_fps'] = video_info['fps']
    
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