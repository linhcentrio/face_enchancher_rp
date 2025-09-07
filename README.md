# Face Enhancement Service với GFPGAN

Service RunPod Serverless chuyên biệt cho việc nâng cao chất lượng khuôn mặt trong video sử dụng mô hình GFPGAN.

**🔗 Tách riêng từ LatentSync service gốc** - Sử dụng cùng base image và phiên bản dependencies để đảm bảo tương thích 100%.

**⚡ ONNX Runtime Optimized** - Tối ưu hóa CUDA providers để tránh Fallback mode và tăng tốc xử lý.

## Tính năng

- ✨ Nâng cao chất lượng khuôn mặt trong video sử dụng GFPGANv1.4
- 🎯 Tự động phát hiện khuôn mặt và chỉ xử lý frame có khuôn mặt
- 🚀 Xử lý batch để tối ưu hiệu suất 
- ⚡ ONNX Runtime tối ưu với CUDA providers
- 📊 Thống kê chi tiết về quá trình enhancement
- 🔄 Giữ nguyên audio gốc của video
- ☁️ Tự động upload kết quả lên MinIO storage
- 📈 Real-time performance monitoring

## ONNX Runtime Optimization

Service này được tối ưu để tránh ONNX Runtime fallback mode:

### ⚡ Optimized CUDA Providers:
```python
cuda_provider_options = {
    'device_id': 0,
    'arena_extend_strategy': 'kNextPowerOfTwo',
    'gpu_mem_limit': 8 * 1024 * 1024 * 1024,  # 8GB
    'cudnn_conv_algo_search': 'EXHAUSTIVE',
    'do_copy_in_default_stream': True,
}
```

### 🔧 Session Options:
```python
session_options.graph_optimization_level = ORT_ENABLE_ALL
session_options.execution_mode = ORT_PARALLEL
session_options.enable_cpu_mem_arena = False
session_options.enable_mem_pattern = False
```

### 📊 Performance Monitoring:
- Real-time face detection timing
- Enhancement batch processing metrics
- GPU utilization tracking
- Memory usage monitoring

## Cấu trúc Project

```
face_enchancher_rp/
├── Dockerfile              # Docker image với base gốc
├── requirements.txt        # Dependencies tương thích
├── rp_handler.py           # RunPod handler optimized
├── test_onnx_optimization.py # ONNX Runtime test suite
├── README.md              # Hướng dẫn này
├── enhancers/             # Module GFPGAN optimized
│   └── GFPGAN/
│       └── GFPGAN.py
├── utils/                 # Utilities cho face processing
│   ├── retinaface.py      # Face detection optimized
│   └── face_alignment.py  # Face alignment
└── faceID/               # Face recognition optimized
    └── faceID.py
```

## Cách sử dụng

### 1. Test ONNX Runtime Optimization

Trước khi build, test xem ONNX Runtime có hoạt động đúng:

```bash
cd face_enchancher_rp
python test_onnx_optimization.py
```

### 2. Build Docker Image

**Manual build:**
```bash
docker build -t face-enhancement-service .
```

### 3. API Request Format

```json
{
  "input": {
    "video_url": "https://example.com/input_video.mp4",
    "batch_size": 4,
    "detection_threshold": 0.3
  }
}
```

#### Tham số đầu vào:

- `video_url` (bắt buộc): URL của video cần enhancement
- `batch_size` (tùy chọn): Số frame xử lý cùng lúc (default: 4)
- `detection_threshold` (tùy chọn): Ngưỡng phát hiện khuôn mặt (default: 0.3)

### 4. API Response Format

```json
{
  "output_video_url": "https://media.aiclip.ai/video/enhanced_faces_job123_abc12345.mp4",
  "processing_time_seconds": 45.67,
  "enhancement_stats": {
    "total_frames": 300,
    "frames_enhanced": 250,
    "enhancement_rate": 83.3,
    "enhancement_applied": true
  },
  "status": "completed"
}
```

## Mô hình AI sử dụng

- **GFPGAN**: Nâng cao chất lượng khuôn mặt
- **RetinaFace**: Phát hiện khuôn mặt và landmark
- **Face Recognition**: Nhận diện khuôn mặt (cho alignment)

Tất cả models đều được tối ưu với ONNX Runtime CUDA providers.

## Yêu cầu hệ thống

- CUDA-compatible GPU (khuyến nghị RTX 3080 trở lên)
- 8GB+ GPU memory cho batch processing
- 16GB+ RAM
- Docker 20+
- Base image: `spxiong/pytorch:2.7.1-py3.10.15-cuda12.8.1-ubuntu22.04`
- ONNX Runtime GPU: 1.21.0
- OpenCV: 4.9.0.80
- NumPy: 1.24.3

## Performance Notes

### 🚀 Optimization Tips:
- **Batch size**: Tăng để xử lý nhanh hơn (4-8 optimal)
- **Detection threshold**: 0.3 cho balance speed/accuracy
- **GPU Memory**: Service tự động quản lý GPU memory limits
- **CUDA optimization**: Sử dụng EXHAUSTIVE conv algo search

### 📊 Expected Performance:
- **Face Detection**: ~0.01-0.03s per frame
- **Enhancement Batch**: ~0.1-0.5s per batch (4 frames)
- **Overall Speedup**: 3-5x faster với CUDA optimization

### ⚠️ Troubleshooting Fallback Mode:
Nếu thấy warning "running in Fallback mode":

1. Check CUDA availability:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

2. Test ONNX Runtime:
```bash
python test_onnx_optimization.py
```

3. Check GPU memory:
```bash
nvidia-smi
```

## Error Handling

Service sẽ trả về lỗi trong các trường hợp:
- Video URL không hợp lệ hoặc không download được
- Video format không được hỗ trợ
- Không đủ GPU memory để xử lý
- ONNX Runtime fallback mode (với warning)
- Lỗi upload kết quả

## Logs và Monitoring

Service log chi tiết quá trình xử lý:
- ONNX Runtime provider initialization
- GPU utilization status
- Face detection performance metrics
- Enhancement batch timing
- Memory usage patterns
- Upload status
- Error details với stack trace

### Log Examples:
```
✅ Face detector initialized with optimized settings
🚀 Using optimized CUDA providers
🔄 Batch 10: Detection=0.015s, Enhancement=0.245s
📊 Average face detection time: 0.018s
📊 Average enhancement batch time: 0.231s
```

## Development

### Test locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Test ONNX optimization
python test_onnx_optimization.py

# Run handler với debug
export RUNPOD_DEBUG=1
python rp_handler.py
```

### Customize Optimization:

Có thể điều chỉnh các tham số trong `rp_handler.py`:
- CUDA provider options
- Session optimization levels
- Memory limits cho từng model
- Batch processing logic
- Performance monitoring intervals

### Monitor Performance:

Service automatically logs:
- ONNX Runtime provider status
- GPU memory usage
- Processing times per stage
- Batch efficiency metrics
- Overall throughput stats

## Deployment

### RunPod:
```bash
docker push your-registry/face-enhancement-service
# Use trong RunPod dashboard
```

### Local với GPU:
```bash
docker run --gpus all -p 8000:8000 face-enhancement-service
```

Với optimization này, service sẽ tránh được "Fallback mode" và đạt hiệu suất tối ưu trên GPU! 🚀
