import cv2
import os
import logging
import shutil
import subprocess
import argparse
import sys
import json
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from abc import ABC, abstractmethod

# --- Abstract Base Classes ---
class BaseAnnotator(ABC):
    """Abstract base class for all annotators"""
    
    def __init__(self, config):
        self.config = config
        self.labels_dir = config['labels_dir']
    
    @abstractmethod
    def annotate_frame(self, img_path):
        """Annotate a single frame"""
        pass
    
    @abstractmethod
    def convert_to_cvat(self):
        """Convert annotations to CVAT format"""
        pass
    
    @abstractmethod
    def get_output_info(self):
        """Get output information for success message"""
        pass

# --- Pose Detection Constants ---
# Body Keypoint names (excluding head keypoints)
KEYPOINT_NAMES = [
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Keypoint connections for skeleton drawing (body only)
SKELETON_CONNECTIONS = [
    # Torso
    (0, 1), (0, 6), (1, 7), (6, 7),  # shoulders, shoulders to hips, hips
    # Arms
    (0, 2), (2, 4), (1, 3), (3, 5),  # shoulders to elbows, elbows to wrists
    # Legs
    (6, 8), (8, 10), (7, 9), (9, 11)  # hips to knees, knees to ankles
]

def extract_body_keypoints(full_keypoints):
    """
    Extract only body keypoints from full COCO keypoints (excluding head)
    
    Args:
        full_keypoints: Full 17 COCO keypoints (x, y, conf)
    
    Returns:
        Body keypoints (12 keypoints: shoulders, elbows, wrists, hips, knees, ankles)
    """
    # COCO keypoint indices for body parts (excluding head: 0-4)
    body_indices = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # shoulders to ankles
    
    body_keypoints = []
    for idx in body_indices:
        if idx < len(full_keypoints):
            body_keypoints.append(full_keypoints[idx])
        else:
            # If keypoint doesn't exist, add a zero keypoint
            body_keypoints.append([0, 0, 0])
    
    return np.array(body_keypoints)

def filter_detections(detections, iou_threshold=0.5):
    """
    Filter detections to keep only one bounding box per person using Non-Maximum Suppression
    
    Args:
        detections: List of detection dictionaries
        iou_threshold: IoU threshold for NMS
    
    Returns:
        List of filtered detections
    """
    if not detections:
        return []
    
    # Sort detections by confidence (highest first)
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    filtered_detections = []
    
    for detection in detections:
        # Check if this detection overlaps significantly with any already selected detection
        should_keep = True
        
        for kept_detection in filtered_detections:
            iou = calculate_iou(detection['box'], kept_detection['box'])
            if iou > iou_threshold:
                should_keep = False
                break
        
        if should_keep:
            filtered_detections.append(detection)
    
    return filtered_detections

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

# --- Concrete Annotator Classes ---
class BoundingBoxAnnotator(BaseAnnotator):
    """Handles traditional bounding box annotations"""
    
    def __init__(self, config):
        super().__init__(config)
        self.model = YOLO(config['model_path'])
    
    def annotate_frame(self, img_path):
        """Annotate frame with bounding boxes only"""
        results = self.model(img_path, verbose=False)[0]
        h, w = cv2.imread(img_path).shape[:2]

        label_file = os.path.join(self.labels_dir, os.path.basename(img_path).replace(".jpg", ".txt"))
        with open(label_file, "w") as f:
            for box in results.boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0]
                xc = ((x1 + x2) / 2) / w
                yc = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
    
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
        else:
            logging.error("❌ CVAT conversion failed!")
        
        return success
    
    def get_output_info(self):
        """Get output information for success message"""
        return f"📋 CVAT format ready at: {self.config['cvat_output_dir']}"

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
    
    # Annotation type selection
    print(f"\n🎯 ANNOTATION TYPE CONFIGURATION")
    print("-" * 30)
    
    annotation_options = {
        '1': 'Bounding boxes only',
        '2': 'Bounding boxes with skeleton keypoints'
    }
    
    for key, value in annotation_options.items():
        print(f"{key}. {value}")
    
    while True:
        annotation_choice = input("\nSelect annotation type (1-2): ").strip()
        if annotation_choice == '1':
            config['annotation_type'] = 'bbox_only'
            config['use_pose_model'] = False
            break
        elif annotation_choice == '2':
            config['annotation_type'] = 'bbox_with_skeleton'
            config['use_pose_model'] = True
            break
        else:
            print("❌ Please select a valid option (1-2)")
    
    # Pose model selection (if needed)
    if config['use_pose_model']:
        print(f"\n🤖 POSE MODEL CONFIGURATION")
        print("-" * 30)
        
        pose_model_options = {
            '1': 'yolov8n-pose.pt (nano - fastest)',
            '2': 'yolov8s-pose.pt (small)',
            '3': 'yolov8m-pose.pt (medium - recommended)',
            '4': 'yolov8l-pose.pt (large)',
            '5': 'yolov8x-pose.pt (xlarge - most accurate)',
            '6': 'Custom pose model path'
        }
        
        for key, value in pose_model_options.items():
            print(f"{key}. {value}")
        
        while True:
            pose_model_choice = input("\nSelect pose model (1-6): ").strip()
            if pose_model_choice in pose_model_options:
                if pose_model_choice == '6':
                    pose_model_path = input("Enter custom pose model path: ").strip()
                    if os.path.exists(pose_model_path):
                        config['pose_model_path'] = pose_model_path
                        break
                    else:
                        print("❌ Pose model file not found!")
                else:
                    pose_model_path = pose_model_options[pose_model_choice].split(' ')[0]
                    config['pose_model_path'] = pose_model_path
                    break
            else:
                print("❌ Please select a valid option (1-6)")
    
    # Output directory configuration
    print(f"\n📁 OUTPUT CONFIGURATION")
    print("-" * 30)
    
    print("You can specify a main output directory where all generated files will be organized.")
    output_dir = input("Main output directory (or press Enter to use current directory): ").strip()
    
    if output_dir:
        # Create subdirectories inside the main output directory
        config.update({
            'frames_dir': os.path.join(output_dir, 'frames'),
            'labels_dir': os.path.join(output_dir, 'labels'),
            'yolo_dataset_dir': os.path.join(output_dir, 'yolo_dataset')
        })
        
        # Set output directory based on annotation type
        if config.get('annotation_type') == 'bbox_with_skeleton':
            config['coco_output_dir'] = os.path.join(output_dir, 'coco_format')
        else:
            config['cvat_output_dir'] = os.path.join(output_dir, 'cvat_format')
    else:
        # Use individual directory names in current directory
        frames_dir = input("Frames directory (or press Enter for 'frames'): ").strip() or 'frames'
        labels_dir = input("Labels directory (or press Enter for 'yolo_labels'): ").strip() or 'yolo_labels'
        yolo_dataset_dir = input("YOLO dataset directory (or press Enter for 'yolo_dataset'): ").strip() or 'yolo_dataset'
        
        config.update({
            'frames_dir': frames_dir,
            'labels_dir': labels_dir,
            'yolo_dataset_dir': yolo_dataset_dir
        })
        
        # Set output directory based on annotation type
        if config.get('annotation_type') == 'bbox_with_skeleton':
            coco_output_dir = input("COCO output directory (or press Enter for 'coco_format'): ").strip() or 'coco_format'
            config['coco_output_dir'] = coco_output_dir
        else:
            cvat_output_dir = input("CVAT output directory (or press Enter for 'cvat_format'): ").strip() or 'cvat_format'
            config['cvat_output_dir'] = cvat_output_dir
    
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
    
    if config['enable_cleaning']:
        print(f"\n📏 BOUNDING BOX FILTERING")
        print("-" * 30)
        print("Remove small bounding boxes to eliminate unwanted annotations")
        
        while True:
            try:
                min_height_input = input("Minimum bounding box height in pixels (or press Enter for 150): ").strip()
                if not min_height_input:
                    min_height = 150
                else:
                    min_height = int(min_height_input)
                
                if min_height > 0:
                    config['min_box_height'] = min_height
                    break
                else:
                    print("❌ Minimum height must be greater than 0")
            except ValueError:
                print("❌ Please enter a valid number")
    
    enable_cvat = input("\nEnable CVAT format conversion? (y/n, default: y): ").strip().lower()
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
    parser.add_argument('--output-dir', '-o', help='Output directory for all generated files (creates subdirectories inside)')
    parser.add_argument('--no-cleaning', action='store_true', help='Disable dataset cleaning')
    parser.add_argument('--no-cvat', action='store_true', help='Disable CVAT conversion')
    parser.add_argument('--config', help='Load configuration from JSON file')
    parser.add_argument('--interactive', '-i', action='store_true', help='Force interactive mode')
    parser.add_argument('--annotation-type', '-a', choices=['bbox_only', 'bbox_with_skeleton'], 
                       default='bbox_only', help='Annotation type (default: bbox_only)')
    parser.add_argument('--pose-model', '-p', help='Pose model path (default: yolov8m-pose.pt)')
    
    return parser.parse_args()

class CombinedAnnotator(BaseAnnotator):
    """Handles both bounding boxes and skeleton annotations"""
    
    def __init__(self, config):
        super().__init__(config)
        self.detection_model = YOLO(config['model_path'])
        self.pose_model = YOLO(config.get('pose_model_path', 'yolov8m-pose.pt'))
        logging.info(f"🤖 Loaded pose model: {config.get('pose_model_path', 'yolov8m-pose.pt')}")
    
    def annotate_frame(self, img_path):
        """Annotate frame with both bounding boxes and skeleton keypoints"""
        # Run detection model for bounding boxes
        bbox_results = self.detection_model(img_path, verbose=False)[0]
        
        # Run pose model for keypoints
        pose_results = self.pose_model(img_path, verbose=False)[0]
        
        h, w = cv2.imread(img_path).shape[:2]
        label_file = os.path.join(self.labels_dir, os.path.basename(img_path).replace(".jpg", ".txt"))
        
        # Collect pose detections
        pose_detections = []
        for result in pose_results:
            if result.keypoints is not None:
                keypoints = result.keypoints.data
                boxes = result.boxes
                
                for i, box in enumerate(boxes):
                    if i < len(keypoints):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        kps = keypoints[i].cpu().numpy()
                        body_kps = extract_body_keypoints(kps)
                        keypoint_coords = body_kps[:, :2]
                        keypoint_confs = body_kps[:, 2]
                        
                        pose_detections.append({
                            'box': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'keypoints': keypoint_coords,
                            'keypoint_confs': keypoint_confs
                        })
        
        # Filter pose detections
        filtered_pose_detections = filter_detections(pose_detections, iou_threshold=0.5)
        
        # Collect bounding box detections
        bbox_detections = []
        for box in bbox_results.boxes:
            cls = int(box.cls[0])
            if cls in self.config.get('keep_classes', [0]):  # Only keep specified classes
                x1, y1, x2, y2 = box.xyxy[0]
                confidence = box.conf[0]
                bbox_detections.append({
                    'box': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'class': cls
                })
        
        # Match bounding boxes with pose detections
        matched_annotations = []
        
        for bbox_det in bbox_detections:
            best_match = None
            best_iou = 0
            
            for pose_det in filtered_pose_detections:
                iou = calculate_iou(bbox_det['box'], pose_det['box'])
                if iou > best_iou and iou > 0.3:  # Minimum IoU threshold for matching
                    best_iou = iou
                    best_match = pose_det
            
            if best_match:
                # Combined annotation with both bbox and keypoints
                matched_annotations.append({
                    'bbox': bbox_det,
                    'pose': best_match,
                    'iou': best_iou
                })
            else:
                # Bbox only (no matching pose)
                matched_annotations.append({
                    'bbox': bbox_det,
                    'pose': None,
                    'iou': 0
                })
        
        # Save combined annotations
        with open(label_file, "w") as f:
            for annotation in matched_annotations:
                bbox = annotation['bbox']
                pose = annotation['pose']
                
                # Save bounding box
                x1, y1, x2, y2 = bbox['box']
                xc = ((x1 + x2) / 2) / w
                yc = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                
                if pose:
                    # Save keypoints along with bbox
                    keypoints = pose['keypoints']
                    confidences = pose['keypoint_confs']
                    
                    normalized_keypoints = []
                    for kp, conf in zip(keypoints, confidences):
                        if conf > 0.1:
                            x_norm = kp[0] / w
                            y_norm = kp[1] / h
                            normalized_keypoints.extend([x_norm, y_norm, conf])
                        else:
                            normalized_keypoints.extend([0, 0, 0])
                    
                    keypoint_str = " ".join([f"{val:.6f}" for val in normalized_keypoints])
                    f.write(f"{bbox['class']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {keypoint_str}\n")
                else:
                    # Bbox only
                    f.write(f"{bbox['class']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
    
    def convert_to_cvat(self):
        """Create COCO keypoints JSON file with both bounding boxes and keypoints"""
        logging.info("=== STEP 5: Converting to COCO keypoints format ===")
        
        coco_data = self._generate_coco_keypoints_data()
        
        json_file = os.path.join(self.config['coco_output_dir'], 'person_keypoints.json')
        os.makedirs(self.config['coco_output_dir'], exist_ok=True)
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2)
        
        # Copy images to COCO format directory
        images_dir = os.path.join(self.config['coco_output_dir'], 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        yolo_images_dir = os.path.join(self.config['yolo_dataset_dir'], 'images')
        for filename in os.listdir(yolo_images_dir):
            if filename.endswith('.jpg'):
                src = os.path.join(yolo_images_dir, filename)
                dst = os.path.join(images_dir, filename)
                shutil.copy(src, dst)
        
        logging.info(f"✅ Created COCO keypoints JSON: {json_file}")
        return True
    
    def _generate_coco_keypoints_data(self):
        """Generate COCO keypoints JSON data with body skeleton annotations"""
        labels_dir = os.path.join(self.config['yolo_dataset_dir'], 'labels')
        images_dir = os.path.join(self.config['yolo_dataset_dir'], 'images')
        
        # COCO keypoints format structure
        coco_data = {
            "licenses": [
                {
                    "name": "",
                    "id": 0,
                    "url": ""
                }
            ],
            "info": {
                "contributor": "",
                "date_created": "",
                "description": "Body skeleton annotations with bounding boxes",
                "url": "",
                "version": "1.0",
                "year": "2024"
            },
            "categories": [
                {
                    "id": 1,
                    "name": "person",
                    "supercategory": "person",
                    "keypoints": [
                        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                        "left_wrist", "right_wrist", "left_hip", "right_hip",
                        "left_knee", "right_knee", "left_ankle", "right_ankle"
                    ],
                    "skeleton": [
                        [1, 2],   # shoulders
                        [1, 3], [3, 5],   # left arm
                        [2, 4], [4, 6],   # right arm
                        [1, 7], [2, 8],   # shoulders to hips
                        [7, 8],   # hips
                        [7, 9], [9, 11],  # left leg
                        [8, 10], [10, 12] # right leg
                    ]
                }
            ],
            "images": [],
            "annotations": []
        }
        
        # Process each image
        image_id = 0
        annotation_id = 0
        
        for filename in sorted(os.listdir(labels_dir)):
            if not filename.endswith('.txt'):
                continue
            
            image_name = filename.replace('.txt', '.jpg')
            image_path = os.path.join(images_dir, image_name)
            
            if not os.path.exists(image_path):
                continue
            
            # Get image dimensions
            img = cv2.imread(image_path)
            if img is None:
                continue
            
            height, width = img.shape[:2]
            image_id += 1
            
            # Add image info
            coco_data["images"].append({
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": image_name,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0
            })
            
            # Process annotations for this image
            label_path = os.path.join(labels_dir, filename)
            annotations = self._process_coco_annotations(label_path, width, height, image_id, annotation_id)
            coco_data["annotations"].extend(annotations)
            annotation_id += len(annotations)
        
        return coco_data
    
    def _process_coco_annotations(self, label_path, img_width, img_height, image_id, start_annotation_id):
        """Process annotations for a single image and return COCO format"""
        annotations = []
        
        if not os.path.exists(label_path):
            return annotations
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        annotation_id = start_annotation_id
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 5:
                continue
            
            try:
                class_id = int(parts[0])
                if class_id != 0:  # Only process person class
                    continue
                
                # Create COCO annotation
                annotation = self._create_coco_annotation(parts, img_width, img_height, image_id, annotation_id)
                if annotation:
                    annotations.append(annotation)
                    annotation_id += 1
                
            except (ValueError, IndexError) as e:
                logging.warning(f"Error parsing annotation in {label_path}: {line} - {e}")
                continue
        
        return annotations
    
    def _create_coco_annotation(self, parts, img_width, img_height, image_id, annotation_id):
        """Create COCO annotation for a single person"""
        # Parse bounding box (YOLO format: center_x, center_y, width, height)
        center_x_norm, center_y_norm, width_norm, height_norm = map(float, parts[1:5])
        
        # Convert to absolute coordinates
        center_x = center_x_norm * img_width
        center_y = center_y_norm * img_height
        width = width_norm * img_width
        height = height_norm * img_height
        
        # Convert to COCO bbox format [x, y, width, height] (top-left corner)
        bbox_x = center_x - (width / 2)
        bbox_y = center_y - (height / 2)
        
        # Calculate area
        area = width * height
        
        # Initialize keypoints array (12 keypoints * 3 values each = 36 values)
        keypoints = [0] * 36  # [x1, y1, v1, x2, y2, v2, ..., x12, y12, v12]
        num_keypoints = 0
        
        if len(parts) > 5:
            keypoints_data = parts[5:]
            
            # Process only body keypoints (12 keypoints)
            for i in range(12):
                keypoint_idx = i * 3
                
                if keypoint_idx + 2 < len(keypoints_data):
                    try:
                        x_norm = float(keypoints_data[keypoint_idx])
                        y_norm = float(keypoints_data[keypoint_idx + 1])
                        conf = float(keypoints_data[keypoint_idx + 2])
                        
                        # Convert to absolute coordinates
                        x_abs = x_norm * img_width
                        y_abs = y_norm * img_height
                        
                        # Set visibility (0=not labeled, 1=labeled but not visible, 2=visible)
                        if conf > 0.1:
                            visibility = 2  # visible
                            num_keypoints += 1
                        elif conf > 0:
                            visibility = 1  # labeled but not visible
                        else:
                            visibility = 0  # not labeled
                        
                        # Set keypoint values
                        keypoints[i * 3] = x_abs      # x
                        keypoints[i * 3 + 1] = y_abs  # y
                        keypoints[i * 3 + 2] = visibility  # visibility
                        
                    except (ValueError, IndexError):
                        # Set default values for missing keypoints
                        keypoints[i * 3] = 0      # x
                        keypoints[i * 3 + 1] = 0  # y
                        keypoints[i * 3 + 2] = 0  # visibility
        
        # Create COCO annotation
        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,  # person category
            "segmentation": [],
            "area": area,
            "bbox": [bbox_x, bbox_y, width, height],
            "iscrowd": 0,
            "attributes": {
                "occluded": False,
                "keyframe": False
            },
            "keypoints": keypoints,
            "num_keypoints": num_keypoints
        }
        
        return annotation
    
    def get_output_info(self):
        """Get output information for success message"""
        return f"📋 COCO keypoints JSON ready at: {self.config['coco_output_dir']}/person_keypoints.json"

# --- Annotator Factory ---
class AnnotatorFactory:
    """Factory class to create appropriate annotator based on configuration"""
    
    @staticmethod
    def create_annotator(config):
        annotation_type = config.get('annotation_type', 'bbox_only')
        
        if annotation_type == 'bbox_only':
            return BoundingBoxAnnotator(config)
        elif annotation_type == 'bbox_with_skeleton':
            return CombinedAnnotator(config)
        else:
            raise ValueError(f"Unknown annotation type: {annotation_type}")

class AnnotationAutomationTool:
    def __init__(self, config):
        """
        Initialize the annotation automation tool with configuration
        
        Args:
            config (dict): Configuration dictionary containing all parameters
        """
        self.config = config
        self.setup_directories()
        self.annotator = AnnotatorFactory.create_annotator(config)
        
    def setup_directories(self):
        """Create necessary directories"""
        annotation_type = self.config.get('annotation_type', 'bbox_only')
        
        # Base directories
        directories = [
            self.config['frames_dir'],
            self.config['labels_dir'],
            self.config['yolo_dataset_dir'],
            f"{self.config['yolo_dataset_dir']}/images",
            f"{self.config['yolo_dataset_dir']}/labels",
        ]
        
        # Add output directory based on annotation type
        if annotation_type == 'bbox_only':
            directories.append(self.config['cvat_output_dir'])
        else:  # bbox_with_skeleton
            directories.append(self.config['coco_output_dir'])
        
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
        annotation_type = self.config.get('annotation_type', 'bbox_only')
        
        if annotation_type == 'bbox_only':
            logging.info("=== STEP 2: Running YOLO bounding box annotation ===")
        else:  # bbox_with_skeleton
            logging.info("=== STEP 2: Running YOLO annotation (bounding boxes + skeletons) ===")
        
        # Clear console and set up the display
        clear_console()
        print("🤖 Running YOLO auto-annotation...")
        print("=" * 50)
        print(f"📊 Processing {len(frame_paths)} frames...")
        print(f"🎯 Annotation type: {annotation_type}")
        print()
        
        # Suppress YOLO's verbose output
        import warnings
        warnings.filterwarnings('ignore')
        
        # Use tqdm with dynamic_ncols=True and proper positioning
        with tqdm(frame_paths, desc="Annotating", unit="frame", 
                 dynamic_ncols=True, position=0, leave=True, 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            
            for img_path in pbar:
                # Suppress YOLO's print output by redirecting stdout temporarily
                import sys
                from io import StringIO
                
                # Capture YOLO's output
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                
                try:
                    self.annotator.annotate_frame(img_path)
                finally:
                    # Restore stdout
                    sys.stdout = old_stdout

        print()  # Add space after progress bar
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
        """Filter annotations to keep only specified classes and remove small bounding boxes"""
        logging.info("=== STEP 4: Cleaning dataset (filtering classes and small boxes) ===")
        
        labels_dir = f"{self.config['yolo_dataset_dir']}/labels"
        images_dir = f"{self.config['yolo_dataset_dir']}/images"
        removed_count = 0
        kept_count = 0
        small_box_count = 0

        for filename in os.listdir(labels_dir):
            if not filename.endswith('.txt'):
                continue

            filepath = os.path.join(labels_dir, filename)
            image_path = os.path.join(images_dir, filename.replace('.txt', '.jpg'))
            
            # Get image dimensions for height calculation
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                img_height, img_width = img.shape[:2]
            else:
                # Fallback if image not found
                img_height, img_width = 1080, 1920  # Default dimensions
            
            with open(filepath, 'r') as file:
                lines = file.readlines()

            # Filter to keep only specified classes and remove small boxes
            filtered_lines = []
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split()
                    cls = int(parts[0])
                    
                    # Check if class is in keep_classes
                    if cls in self.config['keep_classes']:
                        # Check bounding box height (convert from normalized to pixel coordinates)
                        if len(parts) >= 5:
                            _, center_y, _, _, height_norm = map(float, parts[:5])
                            height_pixels = height_norm * img_height
                            
                            # Keep only boxes with height >= minimum threshold
                            min_height = self.config.get('min_box_height', 150)
                            if height_pixels >= min_height:
                                filtered_lines.append(line + '\n')
                                kept_count += 1
                            else:
                                small_box_count += 1
                        else:
                            # If we can't parse the line properly, keep it
                            filtered_lines.append(line + '\n')
                            kept_count += 1

            # Write back filtered lines
            with open(filepath, 'w') as file:
                file.writelines(filtered_lines)

            removed = len(lines) - len(filtered_lines)
            removed_count += removed

        min_height = self.config.get('min_box_height', 150)
        logging.info(f"✅ Cleaned YOLO labels in '{labels_dir}'")
        logging.info(f"Kept {kept_count} objects, removed {removed_count} objects total")
        logging.info(f"Removed {small_box_count} bounding boxes with height < {min_height} pixels")
    
    def convert_to_cvat(self):
        """Convert annotations to CVAT format"""
        return self.annotator.convert_to_cvat()
    
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
            'annotation_type': args.annotation_type,
            'use_pose_model': args.annotation_type == 'bbox_with_skeleton',
        }
        
        # Override with command line arguments
        if args.classes:
            config['classes'] = [c.strip() for c in args.classes.split(',')]
        if args.keep_classes:
            config['keep_classes'] = [int(c.strip()) for c in args.keep_classes.split(',')]
        if args.output_dir:
            # Create subdirectories inside the output directory
            config['frames_dir'] = os.path.join(args.output_dir, 'frames')
            config['labels_dir'] = os.path.join(args.output_dir, 'labels')
            config['yolo_dataset_dir'] = os.path.join(args.output_dir, 'yolo_dataset')
            
            # Set output directory based on annotation type
            if args.annotation_type == 'bbox_with_skeleton':
                config['coco_output_dir'] = os.path.join(args.output_dir, 'coco_format')
            else:
                config['cvat_output_dir'] = os.path.join(args.output_dir, 'cvat_format')
        
        # Handle pose model configuration
        if args.annotation_type == 'bbox_with_skeleton':
            config['pose_model_path'] = args.pose_model or 'yolov8m-pose.pt'
    
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
        
        annotation_type = config.get('annotation_type', 'bbox_only')
        if annotation_type == 'bbox_only':
            print(f"📊 YOLO dataset prepared at: {config['yolo_dataset_dir']}")
        else:
            print(f"📊 YOLO dataset prepared at: {config['yolo_dataset_dir']}")
        
        print(f"🎯 Annotation type: {annotation_type}")
        if config.get('use_pose_model', False):
            print(f"🤖 Pose model used: {config.get('pose_model_path', 'yolov8m-pose.pt')}")
        
        if config['enable_cvat_conversion']:
            print(tool.annotator.get_output_info())
        print("="*60)
    else:
        print("\n❌ Pipeline failed. Check logs above for details.")


if __name__ == "__main__":
    main() 