#!/bin/bash

# GPU 설치 스크립트 (Linux/Mac)
# 사용법: ./scripts/install_gpu.sh [cuda_version]

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# CUDA 버전 확인
check_cuda_version() {
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        log_info "Detected CUDA version: $CUDA_VERSION"
        return 0
    else
        log_warning "CUDA not found. Please install CUDA first."
        return 1
    fi
}

# GPU 확인
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        log_info "GPU Information:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
        return 0
    else
        log_error "nvidia-smi not found. Please install NVIDIA drivers first."
        return 1
    fi
}

# Python 버전 확인
check_python_version() {
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    log_info "Python version: $PYTHON_VERSION"
    
    if [[ $(echo "$PYTHON_VERSION >= 3.8" | bc -l) -eq 1 ]]; then
        log_success "Python version is compatible"
        return 0
    else
        log_error "Python 3.8+ is required"
        return 1
    fi
}

# 가상환경 확인 및 생성
setup_venv() {
    if [ ! -d "venv" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    log_info "Activating virtual environment..."
    source venv/bin/activate
    
    log_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
}

# CUDA 11.8 설치
install_cuda118() {
    log_info "Installing PyTorch with CUDA 11.8 support..."
    pip install torch>=2.0.0+cu118 torchvision>=0.15.0+cu118 --index-url https://download.pytorch.org/whl/cu118
    
    log_info "Installing ONNX Runtime GPU..."
    pip install onnxruntime-gpu>=1.15.0
}

# CUDA 12.1 설치
install_cuda121() {
    log_info "Installing PyTorch with CUDA 12.1 support..."
    pip install torch>=2.0.0+cu121 torchvision>=0.15.0+cu121 --index-url https://download.pytorch.org/whl/cu121
    
    log_info "Installing ONNX Runtime GPU..."
    pip install onnxruntime-gpu>=1.15.0 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
}

# CPU 버전 설치
install_cpu() {
    log_info "Installing CPU-only versions..."
    pip install torch>=2.0.0 torchvision>=0.15.0
    pip install onnxruntime>=1.15.0
}

# InsightFace 설치
install_insightface() {
    log_info "Installing InsightFace..."
    pip install insightface>=0.7.3
}

# 기타 의존성 설치
install_dependencies() {
    log_info "Installing other dependencies..."
    pip install -r requirements.txt
}

# 설치 확인
verify_installation() {
    log_info "Verifying installation..."
    
    python3 -c "
import torch
import onnxruntime as ort
import insightface

print('=== Installation Verification ===')
print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'PyTorch CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')

print(f'ONNX Runtime providers: {ort.get_available_providers()}')

try:
    app = insightface.app.FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    print('InsightFace GPU setup: SUCCESS')
except Exception as e:
    print(f'InsightFace setup: FAILED - {e}')
"
}

# 모델 다운로드
download_models() {
    log_info "Downloading model files..."
    
    # 모델 디렉토리 생성
    mkdir -p models
    
    # Inswapper 모델 다운로드
    if [ ! -f "models/inswapper_128.onnx" ]; then
        log_info "Downloading inswapper_128.onnx..."
        wget -O models/inswapper_128.onnx https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx
    else
        log_info "inswapper_128.onnx already exists"
    fi
    
    # Codeformer 모델 다운로드
    if [ ! -f "models/codeformer-v0.1.0.pth" ]; then
        log_info "Downloading codeformer-v0.1.0.pth..."
        wget -O models/codeformer-v0.1.0.pth https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth
    else
        log_info "codeformer-v0.1.0.pth already exists"
    fi
}

# 환경 변수 설정
setup_env() {
    log_info "Setting up environment variables..."
    
    if [ ! -f ".env" ]; then
        cp env.example .env
        log_info "Created .env file from env.example"
    fi
    
    # GPU 사용 설정
    sed -i 's/USE_GPU=false/USE_GPU=true/' .env
    log_success "GPU enabled in .env file"
}

# 메인 함수
main() {
    log_info "Starting GPU installation process..."
    
    # 시스템 확인
    if ! check_gpu; then
        log_error "GPU check failed. Please install NVIDIA drivers first."
        exit 1
    fi
    
    if ! check_python_version; then
        log_error "Python version check failed."
        exit 1
    fi
    
    # 가상환경 설정
    setup_venv
    
    # CUDA 버전에 따른 설치
    CUDA_VERSION=${1:-"auto"}
    
    if [ "$CUDA_VERSION" = "auto" ]; then
        if check_cuda_version; then
            if [[ $CUDA_VERSION == 11.* ]]; then
                install_cuda118
            elif [[ $CUDA_VERSION == 12.* ]]; then
                install_cuda121
            else
                log_warning "Unsupported CUDA version: $CUDA_VERSION. Installing CPU version."
                install_cpu
            fi
        else
            log_warning "CUDA not detected. Installing CPU version."
            install_cpu
        fi
    elif [ "$CUDA_VERSION" = "11.8" ]; then
        install_cuda118
    elif [ "$CUDA_VERSION" = "12.1" ]; then
        install_cuda121
    elif [ "$CUDA_VERSION" = "cpu" ]; then
        install_cpu
    else
        log_error "Unsupported CUDA version: $CUDA_VERSION"
        exit 1
    fi
    
    # 기타 패키지 설치
    install_insightface
    install_dependencies
    
    # 모델 다운로드
    download_models
    
    # 환경 설정
    setup_env
    
    # 설치 확인
    verify_installation
    
    log_success "GPU installation completed successfully!"
    log_info "To activate the virtual environment, run: source venv/bin/activate"
    log_info "To start the application, run: python main.py"
}

# 스크립트 실행
main "$@"
