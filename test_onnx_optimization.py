#!/usr/bin/env python3
"""
Test script để kiểm tra ONNX Runtime optimization cho Face Enhancement Service
"""

import sys
import os
sys.path.append('/app')
sys.path.append('.')

import time
import torch
import onnxruntime
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_onnx_providers():
    """Test ONNX Runtime providers availability"""
    logger.info("🔧 Testing ONNX Runtime Providers...")
    
    available_providers = onnxruntime.get_available_providers()
    logger.info(f"Available providers: {available_providers}")
    
    if 'CUDAExecutionProvider' in available_providers:
        logger.info("✅ CUDA Provider available")
        
        # Get CUDA provider details
        cuda_provider_options = onnxruntime.get_provider_options('CUDAExecutionProvider')
        logger.info(f"CUDA Provider options: {cuda_provider_options}")
        
        return True
    else:
        logger.warning("⚠️ CUDA Provider NOT available")
        return False

def test_cuda_environment():
    """Test CUDA environment"""
    logger.info("🎮 Testing CUDA Environment...")
    
    # PyTorch CUDA
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available in PyTorch: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info(f"Current GPU: {torch.cuda.current_device()}")
        logger.info(f"GPU name: {torch.cuda.get_device_name()}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        return True
    else:
        logger.warning("⚠️ CUDA not available in PyTorch")
        return False

def create_test_session(use_optimization=True):
    """Create test ONNX session"""
    
    # Create a simple test model path - use any available model
    test_models = [
        "/app/models/face_detection/scrfd_2.5g_bnkps.onnx",
        "/app/models/face_enhancement/GFPGANv1.4.onnx",
        "/app/models/face_detection/recognition.onnx"
    ]
    
    model_path = None
    for path in test_models:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        logger.error("❌ No test model found")
        return None
    
    logger.info(f"Testing with model: {model_path}")
    
    if use_optimization:
        # Optimized session
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
        session_options.enable_cpu_mem_arena = False
        session_options.enable_mem_pattern = False
        
        if torch.cuda.is_available():
            cuda_provider_options = {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # 4GB
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }
            providers = [('CUDAExecutionProvider', cuda_provider_options), 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
    else:
        # Basic session
        session_options = None
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    
    try:
        session = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=providers
        )
        
        # Log which providers are actually being used
        logger.info(f"Session providers: {session.get_providers()}")
        
        return session
    except Exception as e:
        logger.error(f"❌ Failed to create session: {e}")
        return None

def test_session_performance():
    """Test session performance"""
    logger.info("⚡ Testing session performance...")
    
    # Test optimized session
    logger.info("Testing optimized session...")
    optimized_session = create_test_session(use_optimization=True)
    
    if optimized_session:
        logger.info("✅ Optimized session created successfully")
        logger.info(f"Optimized session providers: {optimized_session.get_providers()}")
    else:
        logger.error("❌ Failed to create optimized session")
    
    # Test basic session
    logger.info("Testing basic session...")
    basic_session = create_test_session(use_optimization=False)
    
    if basic_session:
        logger.info("✅ Basic session created successfully")
        logger.info(f"Basic session providers: {basic_session.get_providers()}")
    else:
        logger.error("❌ Failed to create basic session")

def test_face_enhancement_modules():
    """Test face enhancement modules initialization"""
    logger.info("🧪 Testing Face Enhancement modules...")
    
    try:
        # Test imports
        from utils.retinaface import RetinaFace
        from utils.face_alignment import get_cropped_head_256
        from enhancers.GFPGAN.GFPGAN import GFPGAN
        from faceID.faceID import FaceRecognition
        logger.info("✅ All modules imported successfully")
        
        # Test if model files exist
        model_paths = {
            "face_detector": "/app/models/face_detection/scrfd_2.5g_bnkps.onnx",
            "face_enhancer": "/app/models/face_enhancement/GFPGANv1.4.onnx", 
            "face_recognition": "/app/models/face_detection/recognition.onnx"
        }
        
        missing_models = []
        for name, path in model_paths.items():
            if os.path.exists(path):
                logger.info(f"✅ {name}: {path}")
            else:
                logger.warning(f"⚠️ {name} missing: {path}")
                missing_models.append(name)
        
        if missing_models:
            logger.warning(f"Missing models: {missing_models}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Module test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 ONNX Runtime Optimization Test Suite")
    logger.info("=" * 60)
    
    tests = [
        ("CUDA Environment", test_cuda_environment),
        ("ONNX Providers", test_onnx_providers),
        ("Session Performance", test_session_performance),
        ("Face Enhancement Modules", test_face_enhancement_modules),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🔬 Running {test_name} test...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.warning(f"⚠️ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: CRASHED - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Test Results Summary:")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\n🎯 {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! ONNX Runtime optimization ready.")
        return 0
    else:
        logger.warning("⚠️ Some tests failed. Check configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
