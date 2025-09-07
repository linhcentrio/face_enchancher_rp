#!/usr/bin/env python3
"""
RunPod Serverless Handler for Face Enhancement using GFPGAN
Tách riêng từ LatentSync service để chuyên biệt hóa việc enhance khuôn mặt
"""

import runpod
import os
import tempfile
import uuid
import requests
import time
import cv2
import numpy as np
import torch
import sys
from pathlib import Path
from minio import Minio
from urllib.parse import quote
from tqdm import tqdm
import onnxruntime
import logging
import gc
import subprocess
from datetime import datetime

# Add path for local modules
sys.path.append('/app')

# Import required modules
try:
    from utils.retinaface import RetinaFace
    from utils.face_alignment import get_cropped_head_256
    from enhancers.GFPGAN.GFPGAN import GFPGAN
    from faceID.faceID import FaceRecognition
except ImportError as e:
    logging.error(f"Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MinIO Configuration
MINIO_ENDPOINT = "media.aiclip.ai"
MINIO_ACCESS_KEY = "VtZ6MUPfyTOH3qSiohA2"
MINIO_SECRET_KEY = "8boVPVIynLEKcgXirrcePxvjSk7gReIDD9pwto3t"
MINIO_BUCKET = "video"
MINIO_SECURE = False

# Initialize MinIO client
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE
)

# Global model instances
detector = None
enhancer = None
recognition = None

# Model paths
MODEL_PATHS = {
    "face_detector": "/app/models/face_detection/scrfd_2.5g_bnkps.onnx",
    "face_enhancer": "/app/models/face_enhancement/GFPGANv1.4.onnx",
    "face_recognition": "/app/models/face_detection/recognition.onnx"
}

def create_optimized_session_options():
    """Create optimized ONNX Runtime session options"""
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
    session_options.enable_cpu_mem_arena = False
    session_options.enable_mem_pattern = False
    return session_options

def get_optimized_providers():
    """Get optimized ONNX Runtime providers"""
    if torch.cuda.is_available():
        cuda_provider_options = {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 8 * 1024 * 1024 * 1024,  # 8GB limit
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }
        providers = [
            ('CUDAExecutionProvider', cuda_provider_options),
            'CPUExecutionProvider'
        ]
        logger.info("🚀 Using optimized CUDA providers")
    else:
        providers = ['CPUExecutionProvider']
        logger.info("⚠️ CUDA not available, using CPU")
    
    return providers

def initialize_models():
    """Initialize face enhancement models with optimized ONNX Runtime settings"""
    global detector, enhancer, recognition
    
    try:
        # Create optimized session options
        session_options = create_optimized_session_options()
        providers = get_optimized_providers()
        
        # Initialize face detector với optimized settings
        detector_path = MODEL_PATHS["face_detector"]
        if not os.path.exists(detector_path):
            raise FileNotFoundError(f"Face detector model not found: {detector_path}")
            
        detector = RetinaFace(
            detector_path,
            provider=providers,
            session_options=session_options
        )
        logger.info("✅ Face detector initialized with optimized settings")
        
        # Initialize face recognition với optimized settings
        recognition_path = MODEL_PATHS["face_recognition"]
        if not os.path.exists(recognition_path):
            raise FileNotFoundError(f"Face recognition model not found: {recognition_path}")
            
        # Create optimized recognition session
        recognition_session = onnxruntime.InferenceSession(
            recognition_path,
            sess_options=session_options,
            providers=providers
        )
        recognition = FaceRecognition(session=recognition_session)
        logger.info("✅ Face recognition initialized with optimized settings")
        
        # Initialize face enhancer với optimized settings
        enhancer_path = MODEL_PATHS["face_enhancer"]
        if not os.path.exists(enhancer_path):
            raise FileNotFoundError(f"Face enhancer model not found: {enhancer_path}")
        
        # Determine device for GFPGAN
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Running on {device}")
        
        enhancer = GFPGAN(
            model_path=enhancer_path, 
            device=device,
            session_options=session_options,
            providers=providers
        )
        logger.info(f"✅ Face enhancer initialized on {device} with optimized settings")
        
        # Log providers info
        logger.info(f"🔧 Available ONNX providers: {onnxruntime.get_available_providers()}")
        if torch.cuda.is_available():
            logger.info(f"🎮 GPU: {torch.cuda.get_device_name()}")
            logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        raise e

def download_file(url: str, local_path: str) -> bool:
    """Download file from URL with progress tracking"""
    try:
        logger.info(f"📥 Downloading {url}")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        
        logger.info(f"✅ Downloaded: {local_path} ({downloaded/1024/1024:.1f} MB)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return False

def upload_to_minio(local_path: str, object_name: str) -> str:
    """Upload file to MinIO with enhanced error handling"""
    try:
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        minio_client.fput_object(MINIO_BUCKET, object_name, local_path)
        file_url = f"https://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{quote(object_name)}"
        logger.info(f"✅ Uploaded successfully: {file_url}")
        return file_url
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise e

def process_batch(frame_buffer, enhancer, face_mask, out, frame_width, frame_height):
    """Process batch of frames with face enhancement"""
    frames, aligned_faces, mats = zip(*frame_buffer)
    enhanced_faces = enhancer.enhance_batch(aligned_faces)
    
    for frame, aligned_face, mat, enhanced_face in zip(frames, aligned_faces, mats, enhanced_faces):
        enhanced_face_resized = cv2.resize(enhanced_face, (aligned_face.shape[1], aligned_face.shape[0]))
        face_mask_resized = cv2.resize(face_mask, (enhanced_face_resized.shape[1], enhanced_face_resized.shape[0]))
        blended_face = (face_mask_resized * enhanced_face_resized + (1 - face_mask_resized) * aligned_face).astype(np.uint8)
        
        mat_rev = cv2.invertAffineTransform(mat)
        dealigned_face = cv2.warpAffine(blended_face, mat_rev, (frame_width, frame_height))
        mask = cv2.warpAffine(face_mask_resized, mat_rev, (frame_width, frame_height))
        final_frame = (mask * dealigned_face + (1 - mask) * frame).astype(np.uint8)
        
        out.write(final_frame)

def enhance_video_with_gfpgan(input_video_path: str, output_path: str = None) -> tuple[bool, dict]:
    """Apply face enhancement to video using GFPGAN"""
    global detector, enhancer
    
    stats = {
        "total_frames": 0,
        "frames_with_faces": 0,
        "frames_without_faces": 0,
        "faces_enhanced": 0,
        "enhancement_applied": False
    }
    
    try:
        logger.info(f"✨ Starting face enhancement: {input_video_path}")
        
        video_stream = cv2.VideoCapture(input_video_path)
        if not video_stream.isOpened():
            raise ValueError(f"Failed to open video file: {input_video_path}")
        
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        frame_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        
        stats["total_frames"] = total_frames
        
        if output_path is None:
            output_path = os.path.splitext(input_video_path)[0] + '_enhanced_gfpgan.mp4'
        
        temp_video_path = output_path.replace('.mp4', '_temp.mp4')
        out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        
        # Create face mask for blending
        face_mask = np.zeros((256, 256), dtype=np.uint8)
        face_mask = cv2.rectangle(face_mask, (66, 69), (190, 240), (255, 255, 255), -1)
        face_mask = cv2.GaussianBlur(face_mask.astype(np.uint8), (19, 19), cv2.BORDER_DEFAULT)
        face_mask = cv2.cvtColor(face_mask, cv2.COLOR_GRAY2RGB)
        face_mask = face_mask / 255
        
        batch_size = 4  # Increased batch size for better performance
        frame_buffer = []
        
        logger.info(f"Processing {total_frames} frames with face enhancement...")
        
        detection_times = []
        enhancement_times = []
        
        for frame_idx in tqdm(range(total_frames), desc="Enhancing faces"):
            ret, frame = video_stream.read()
            if not ret:
                break
            
            # Try to detect faces với timing
            face_detection_start = time.time()
            try:
                bboxes, kpss = detector.detect(frame, input_size=(320, 320), det_thresh=0.3)
                detection_times.append(time.time() - face_detection_start)
            except Exception as e:
                logger.debug(f"Face detection failed for frame {frame_idx}: {e}")
                bboxes, kpss = [], []
                detection_times.append(time.time() - face_detection_start)
            
            if len(kpss) == 0:
                # No face → Keep original frame
                stats["frames_without_faces"] += 1
                out.write(frame)
                continue
            
            # Face detected → Enhance with GFPGAN
            stats["frames_with_faces"] += 1
            
            try:
                aligned_face, mat = get_cropped_head_256(frame, kpss[0], size=256, scale=1.0)
                frame_buffer.append((frame, aligned_face, mat))
                
                if len(frame_buffer) >= batch_size:
                    enhancement_start = time.time()
                    process_batch(frame_buffer, enhancer, face_mask, out, frame_width, frame_height)
                    enhancement_times.append(time.time() - enhancement_start)
                    stats["faces_enhanced"] += len(frame_buffer)
                    frame_buffer = []
                    
                    # Log performance every 10 batches
                    if len(enhancement_times) % 10 == 0:
                        avg_detection = np.mean(detection_times[-40:]) if detection_times else 0
                        avg_enhancement = np.mean(enhancement_times[-10:]) if enhancement_times else 0
                        logger.info(f"🔄 Batch {len(enhancement_times)}: Detection={avg_detection:.3f}s, Enhancement={avg_enhancement:.3f}s")
                        
            except Exception as e:
                logger.debug(f"Face processing failed for frame {frame_idx}: {e}")
                # If face processing fails, keep original frame
                out.write(frame)
                stats["frames_with_faces"] -= 1
                stats["frames_without_faces"] += 1
        
        # Log final performance stats
        if detection_times:
            logger.info(f"📊 Average face detection time: {np.mean(detection_times):.3f}s")
        if enhancement_times:
            logger.info(f"📊 Average enhancement batch time: {np.mean(enhancement_times):.3f}s")
        
        # Process remaining frames
        if frame_buffer:
            process_batch(frame_buffer, enhancer, face_mask, out, frame_width, frame_height)
            stats["faces_enhanced"] += len(frame_buffer)
        
        video_stream.release()
        out.release()
        
        if stats["frames_with_faces"] > 0:
            stats["enhancement_applied"] = True
            logger.info(f"✅ Face enhancement applied to {stats['frames_with_faces']}/{stats['total_frames']} frames")
        else:
            logger.info(f"ℹ️ No faces detected in any frame. Original video unchanged.")
            stats["enhancement_applied"] = False
        
        # Extract and combine audio
        audio_path = os.path.splitext(output_path)[0] + '.aac'
        
        subprocess.run([
            'ffmpeg', '-y', '-i', input_video_path, 
            '-vn', '-acodec', 'aac', '-b:a', '192k', audio_path
        ], check=True, capture_output=True)
        
        subprocess.run([
            'ffmpeg', '-y', '-i', temp_video_path, '-i', audio_path, 
            '-c:v', 'libx264', '-crf', '23', '-preset', 'medium', 
            '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', output_path
        ], check=True, capture_output=True)
        
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"✅ Face enhancement completed: {output_path} ({file_size:.1f} MB)")
        return True, stats
        
    except Exception as e:
        logger.error(f"❌ Face enhancement failed: {e}")
        return False, stats
    finally:
        if 'video_stream' in locals():
            video_stream.release()
        if 'out' in locals():
            out.release()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def handler(job):
    """Main RunPod handler for face enhancement"""
    job_id = job.get("id", "unknown")
    start_time = time.time()
    
    try:
        job_input = job.get("input", {})
        video_url = job_input.get("video_url")
        
        if not video_url:
            return {"error": "Missing video_url parameter"}
        
        # Enhancement parameters
        batch_size = job_input.get("batch_size", 4)
        detection_threshold = job_input.get("detection_threshold", 0.3)
        
        logger.info(f"🚀 Job {job_id}: Face Enhancement Service")
        logger.info(f"📺 Video: {video_url}")
        logger.info(f"⚙️ Parameters: batch_size={batch_size}, detection_threshold={detection_threshold}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_video_path = os.path.join(temp_dir, "input_video.mp4")
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_video_path = os.path.join(temp_dir, f"enhanced_{current_time}.mp4")
            
            # Step 1: Download input video
            logger.info("📥 Step 1/3: Downloading input video...")
            if not download_file(video_url, input_video_path):
                return {"error": "Failed to download video"}
            
            # Step 2: Apply face enhancement
            logger.info("✨ Step 2/3: Applying face enhancement...")
            enhancement_success, face_stats = enhance_video_with_gfpgan(input_video_path, output_video_path)
            
            if not enhancement_success:
                return {"error": "Face enhancement failed"}
            
            if not os.path.exists(output_video_path):
                return {"error": "Enhanced video not generated"}
            
            # Step 3: Upload result
            logger.info("📤 Step 3/3: Uploading enhanced video...")
            output_filename = f"enhanced_faces_{job_id}_{uuid.uuid4().hex[:8]}.mp4"
            output_url = upload_to_minio(output_video_path, output_filename)
            
            processing_time = time.time() - start_time
            
            # Prepare response
            response = {
                "output_video_url": output_url,
                "processing_time_seconds": round(processing_time, 2),
                "enhancement_stats": {
                    "total_frames": face_stats["total_frames"],
                    "frames_enhanced": face_stats["frames_with_faces"],
                    "enhancement_rate": round(face_stats["frames_with_faces"] / face_stats["total_frames"] * 100, 1) if face_stats["total_frames"] > 0 else 0,
                    "enhancement_applied": face_stats["enhancement_applied"]
                },
                "status": "completed"
            }
            
            return response
            
    except Exception as e:
        logger.error(f"❌ Handler error: {e}")
        return {
            "error": str(e),
            "status": "failed",
            "processing_time_seconds": round(time.time() - start_time, 2)
        }
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    logger.info("🚀 Starting Face Enhancement Serverless Worker...")
    logger.info(f"🗄️ Storage: {MINIO_ENDPOINT}/{MINIO_BUCKET}")
    logger.info(f"✨ Service: Face Enhancement with GFPGAN")
    
    # Verify environment
    try:
        logger.info(f"🐍 Python: {sys.version}")
        logger.info(f"🔥 PyTorch: {torch.__version__}")
        logger.info(f"⚡ CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"🎮 GPU: {torch.cuda.get_device_name()}")
            logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    except Exception as e:
        logger.warning(f"⚠️ Environment check failed: {e}")
    
    # Verify model files exist
    all_models_exist = True
    for model_name, model_path in MODEL_PATHS.items():
        if os.path.exists(model_path):
            logger.info(f"✅ {model_name} model verified: {model_path}")
        else:
            logger.error(f"❌ {model_name} model missing: {model_path}")
            all_models_exist = False
    
    if not all_models_exist:
        logger.error("❌ Some model files are missing!")
        sys.exit(1)
    
    # Initialize models
    try:
        initialize_models()
        logger.info("✅ All models initialized successfully")
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        sys.exit(1)
    
    # Start RunPod serverless worker
    logger.info("✨ Ready to process face enhancement requests...")
    runpod.serverless.start({"handler": handler})
