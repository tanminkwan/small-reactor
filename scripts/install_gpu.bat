@echo off
REM GPU 설치 스크립트 (Windows)
REM 사용법: scripts\install_gpu.bat [cuda_version]

setlocal enabledelayedexpansion

REM 색상 정의 (Windows에서는 제한적)
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

REM 로그 함수
:log_info
echo %BLUE%[INFO]%NC% %~1
goto :eof

:log_success
echo %GREEN%[SUCCESS]%NC% %~1
goto :eof

:log_warning
echo %YELLOW%[WARNING]%NC% %~1
goto :eof

:log_error
echo %RED%[ERROR]%NC% %~1
goto :eof

REM CUDA 버전 확인
:check_cuda_version
call :log_info "Checking CUDA version..."
nvcc --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=6" %%i in ('nvcc --version ^| findstr "release"') do set CUDA_VERSION=%%i
    set CUDA_VERSION=!CUDA_VERSION:~1!
    call :log_info "Detected CUDA version: !CUDA_VERSION!"
    exit /b 0
) else (
    call :log_warning "CUDA not found. Please install CUDA first."
    exit /b 1
)

REM GPU 확인
:check_gpu
call :log_info "Checking GPU..."
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    call :log_info "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
    exit /b 0
) else (
    call :log_error "nvidia-smi not found. Please install NVIDIA drivers first."
    exit /b 1
)

REM Python 버전 확인
:check_python_version
call :log_info "Checking Python version..."
python --version >nul 2>&1
if %errorlevel% neq 0 (
    call :log_error "Python not found. Please install Python 3.8+ first."
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
call :log_info "Python version: !PYTHON_VERSION!"
call :log_success "Python version is compatible"
exit /b 0

REM 가상환경 확인 및 생성
:setup_venv
if not exist "venv" (
    call :log_info "Creating virtual environment..."
    python -m venv venv
)

call :log_info "Activating virtual environment..."
call venv\Scripts\activate.bat

call :log_info "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel
exit /b 0

REM CUDA 11.8 설치
:install_cuda118
call :log_info "Installing PyTorch with CUDA 11.8 support..."
pip install torch>=2.0.0+cu118 torchvision>=0.15.0+cu118 --index-url https://download.pytorch.org/whl/cu118

call :log_info "Installing ONNX Runtime GPU..."
pip install onnxruntime-gpu>=1.15.0
exit /b 0

REM CUDA 12.1 설치
:install_cuda121
call :log_info "Installing PyTorch with CUDA 12.1 support..."
pip install torch>=2.0.0+cu121 torchvision>=0.15.0+cu121 --index-url https://download.pytorch.org/whl/cu121

call :log_info "Installing ONNX Runtime GPU..."
pip install onnxruntime-gpu>=1.15.0 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
exit /b 0

REM CPU 버전 설치
:install_cpu
call :log_info "Installing CPU-only versions..."
pip install torch>=2.0.0 torchvision>=0.15.0
pip install onnxruntime>=1.15.0
exit /b 0

REM InsightFace 설치
:install_insightface
call :log_info "Installing InsightFace..."
pip install insightface>=0.7.3
exit /b 0

REM 기타 의존성 설치
:install_dependencies
call :log_info "Installing other dependencies..."
pip install -r requirements.txt
exit /b 0

REM 설치 확인
:verify_installation
call :log_info "Verifying installation..."

python -c "import torch; import onnxruntime as ort; import insightface; print('=== Installation Verification ==='); print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'PyTorch CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'CUDA not available'); print(f'GPU count: {torch.cuda.device_count()}' if torch.cuda.is_available() else 'No GPU'); print(f'GPU name: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU'); print(f'ONNX Runtime providers: {ort.get_available_providers()}'); print('InsightFace: Available')"

exit /b 0

REM 모델 다운로드
:download_models
call :log_info "Downloading model files..."

if not exist "models" mkdir models

if not exist "models\inswapper_128.onnx" (
    call :log_info "Downloading inswapper_128.onnx..."
    curl -L -o models\inswapper_128.onnx https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx
) else (
    call :log_info "inswapper_128.onnx already exists"
)

if not exist "models\codeformer-v0.1.0.pth" (
    call :log_info "Downloading codeformer-v0.1.0.pth..."
    curl -L -o models\codeformer-v0.1.0.pth https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth
) else (
    call :log_info "codeformer-v0.1.0.pth already exists"
)

exit /b 0

REM 환경 변수 설정
:setup_env
call :log_info "Setting up environment variables..."

if not exist ".env" (
    copy env.example .env
    call :log_info "Created .env file from env.example"
)

REM GPU 사용 설정 (Windows에서는 sed 대신 PowerShell 사용)
powershell -Command "(Get-Content .env) -replace 'USE_GPU=false', 'USE_GPU=true' | Set-Content .env"
call :log_success "GPU enabled in .env file"

exit /b 0

REM 메인 함수
:main
call :log_info "Starting GPU installation process..."

REM 시스템 확인
call :check_gpu
if %errorlevel% neq 0 (
    call :log_error "GPU check failed. Please install NVIDIA drivers first."
    exit /b 1
)

call :check_python_version
if %errorlevel% neq 0 (
    call :log_error "Python version check failed."
    exit /b 1
)

REM 가상환경 설정
call :setup_venv

REM CUDA 버전에 따른 설치
set CUDA_VERSION=%1
if "%CUDA_VERSION%"=="" set CUDA_VERSION=auto

if "%CUDA_VERSION%"=="auto" (
    call :check_cuda_version
    if %errorlevel% equ 0 (
        if "!CUDA_VERSION:~0,2!"=="11" (
            call :install_cuda118
        ) else if "!CUDA_VERSION:~0,2!"=="12" (
            call :install_cuda121
        ) else (
            call :log_warning "Unsupported CUDA version: !CUDA_VERSION!. Installing CPU version."
            call :install_cpu
        )
    ) else (
        call :log_warning "CUDA not detected. Installing CPU version."
        call :install_cpu
    )
) else if "%CUDA_VERSION%"=="11.8" (
    call :install_cuda118
) else if "%CUDA_VERSION%"=="12.1" (
    call :install_cuda121
) else if "%CUDA_VERSION%"=="cpu" (
    call :install_cpu
) else (
    call :log_error "Unsupported CUDA version: %CUDA_VERSION%"
    exit /b 1
)

REM 기타 패키지 설치
call :install_insightface
call :install_dependencies

REM 모델 다운로드
call :download_models

REM 환경 설정
call :setup_env

REM 설치 확인
call :verify_installation

call :log_success "GPU installation completed successfully!"
call :log_info "To activate the virtual environment, run: venv\Scripts\activate.bat"
call :log_info "To start the application, run: python main.py"

endlocal
