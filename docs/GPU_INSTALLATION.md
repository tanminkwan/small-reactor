# GPU 설치 가이드

## 개요

이 문서는 얼굴 교체 애플리케이션에서 GPU를 사용하기 위한 설치 가이드입니다. GPU 사용 시 처리 속도가 크게 향상됩니다.

## 시스템 요구사항

### 하드웨어 요구사항
- **NVIDIA GPU**: CUDA 지원 GPU (GTX 1060 이상 권장)
- **VRAM**: 최소 4GB (8GB 이상 권장)
- **RAM**: 최소 8GB (16GB 이상 권장)

### 소프트웨어 요구사항
- **운영체제**: Windows 10/11, Ubuntu 18.04+, CentOS 7+
- **Python**: 3.8 이상 (3.10 권장)
- **CUDA**: 11.8 또는 12.1+
- **cuDNN**: 8.9+ (CUDA 버전에 맞는 버전)

## 1. CUDA 설치

### Windows

1. **NVIDIA 드라이버 업데이트**
   ```bash
   # NVIDIA 공식 사이트에서 최신 드라이버 다운로드 및 설치
   # https://www.nvidia.com/drivers/
   ```

2. **CUDA Toolkit 설치**
   - [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)에서 다운로드
   - CUDA 11.8 또는 12.1 버전 선택
   - 설치 후 환경 변수 설정 확인

3. **cuDNN 설치**
   - [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)에서 다운로드
   - CUDA 버전에 맞는 cuDNN 설치

### Ubuntu/Linux

1. **NVIDIA 드라이버 설치**
   ```bash
   # 드라이버 확인
   nvidia-smi
   
   # 드라이버 설치 (Ubuntu)
   sudo apt update
   sudo apt install nvidia-driver-525  # 버전은 GPU에 따라 다름
   ```

2. **CUDA Toolkit 설치**
   ```bash
   # CUDA 11.8 설치
   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
   sudo sh cuda_11.8.0_520.61.05_linux.run
   
   # 환경 변수 설정
   echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

3. **cuDNN 설치**
   ```bash
   # cuDNN 다운로드 및 설치
   # NVIDIA 계정 필요
   tar -xvf cudnn-linux-x86_64-8.9.0.131_cuda11-archive.tar.xz
   sudo cp -r cudnn-*-archive/include/* /usr/local/cuda/include/
   sudo cp -r cudnn-*-archive/lib/* /usr/local/cuda/lib64/
   ```

## 2. Python 환경 설정

### 가상환경 생성
```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### pip 업그레이드
```bash
pip install --upgrade pip setuptools wheel
```

## 3. GPU 지원 패키지 설치

### CUDA 12.8 버전용 설치

```bash
# PyTorch (CUDA 12.8)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# ONNX Runtime GPU
pip install onnxruntime-gpu>=1.15.0

# InsightFace
pip install insightface>=0.7.3

# 기타 의존성
pip install -r requirements.txt
```

### CUDA 12.1 버전용 설치

```bash
# PyTorch (CUDA 12.1)
pip install torch>=2.0.0+cu121 torchvision>=0.15.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# ONNX Runtime GPU (CUDA 12)
pip install onnxruntime-gpu>=1.15.0 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# InsightFace
pip install insightface>=0.7.3

# 기타 의존성
pip install -r requirements.txt
```

## 4. 설치 확인

### CUDA 설치 확인
```bash
# CUDA 버전 확인
nvcc --version

# GPU 정보 확인
nvidia-smi
```

### Python에서 GPU 확인
```python
import torch
import onnxruntime as ort

# PyTorch GPU 확인
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# ONNX Runtime GPU 확인
print(f"ONNX Runtime providers: {ort.get_available_providers()}")
```

### InsightFace GPU 확인
```python
import insightface

# InsightFace GPU 확인
app = insightface.app.FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0은 GPU 사용
print("InsightFace GPU setup successful")
```

## 5. 환경 변수 설정

### .env 파일 설정
```bash
# .env 파일 생성
cp env.example .env

# .env 파일 편집
USE_GPU=true
CUDA_VISIBLE_DEVICES=0
```

### 환경 변수 확인
```bash
# Windows
set CUDA_VISIBLE_DEVICES=0

# Linux/Mac
export CUDA_VISIBLE_DEVICES=0
```

## 6. 모델 파일 다운로드

### Buffalo_L 모델
```bash
# InsightFace가 자동으로 다운로드하지만, 수동으로도 가능
mkdir -p models/buffalo_l
# InsightFace 공식 저장소에서 다운로드
```

### Inswapper 모델
```bash
# inswapper_128.onnx 다운로드
mkdir -p models
wget https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx -O models/inswapper_128.onnx
```

### Codeformer 모델
```bash
# codeformer-v0.1.0.pth 다운로드
wget https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth -O models/codeformer-v0.1.0.pth
```

## 7. 문제 해결

### 일반적인 문제들

#### 1. CUDA 버전 불일치
```bash
# 문제: CUDA 버전이 맞지 않음
# 해결: 올바른 CUDA 버전의 PyTorch 설치
pip uninstall torch torchvision
pip install torch>=2.0.0+cu118 torchvision>=0.15.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

#### 2. cuDNN 오류
```bash
# 문제: cuDNN 라이브러리 누락
# 해결: cuDNN 재설치
# NVIDIA 공식 사이트에서 다운로드 후 설치
```

#### 3. 메모리 부족
```bash
# 문제: GPU 메모리 부족
# 해결: 배치 크기 줄이기 또는 이미지 크기 줄이기
# .env 파일에서 설정 조정
MAX_IMAGE_SIZE=512
BATCH_SIZE=1
```

#### 4. InsightFace 설치 오류
```bash
# 문제: InsightFace 설치 실패
# 해결: Visual Studio C++ 빌드 도구 설치 (Windows)
# 또는 build-essential 설치 (Linux)
sudo apt-get install build-essential
```

### 성능 최적화

#### 1. GPU 메모리 최적화
```python
# PyTorch 메모리 최적화
import torch
torch.cuda.empty_cache()

# ONNX Runtime 최적화
import onnxruntime as ort
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
```

#### 2. 배치 처리 최적화
```python
# 배치 크기 조정
BATCH_SIZE = 1  # GPU 메모리에 따라 조정
MAX_IMAGE_SIZE = 1024  # 이미지 크기 조정
```

## 8. Docker를 사용한 GPU 설치

### Dockerfile 예시
```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Python 설치
RUN apt-get update && apt-get install -y python3 python3-pip

# 의존성 설치
COPY requirements.txt .
RUN pip install -r requirements.txt

# GPU 지원 패키지 설치
RUN pip install torch>=2.0.0+cu118 torchvision>=0.15.0+cu118 --index-url https://download.pytorch.org/whl/cu118
RUN pip install onnxruntime-gpu>=1.15.0

# 애플리케이션 복사
COPY . /app
WORKDIR /app

# 실행
CMD ["python", "main.py"]
```

### Docker 실행
```bash
# GPU 지원 Docker 실행
docker run --gpus all -p 7860:7860 your-app:latest
```

## 9. 성능 벤치마크

### 예상 성능 (RTX 3080 기준)
- **CPU**: 30-60초/이미지
- **GPU**: 3-8초/이미지
- **메모리 사용량**: 4-6GB VRAM

### 성능 측정
```python
import time
import torch

# GPU 성능 측정
start_time = time.time()
# 얼굴 교체 작업 수행
end_time = time.time()
print(f"Processing time: {end_time - start_time:.2f} seconds")
```

## 결론

GPU 설정이 완료되면 얼굴 교체 애플리케이션의 처리 속도가 크게 향상됩니다. 문제가 발생하면 위의 문제 해결 섹션을 참고하거나 로그를 확인하여 디버깅하세요.
