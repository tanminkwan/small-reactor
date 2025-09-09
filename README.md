# Face Swap Application

Python Gradio를 사용한 얼굴 교체 애플리케이션입니다. SOLID 원칙을 준수하여 TDD 방식으로 개발되었습니다.

## 🚀 주요 기능

- **🎯 정확한 얼굴 탐지**: Buffalo_L 모델을 이용한 고정밀 얼굴 탐지
- **🔄 고품질 얼굴 교체**: inswapper_128.onnx 모델을 이용한 자연스러운 얼굴 교체
- **✨ 이미지 복원**: codeformer-v0.1.0.pth 모델을 이용한 이미지 품질 향상
- **🌐 웹 UI**: Gradio를 이용한 사용자 친화적인 웹 인터페이스
- **📱 드래그 앤 드롭**: 직관적인 이미지 업로드 및 교체
- **🎨 실시간 미리보기**: 업로드된 이미지 즉시 표시
- **👄 입 원본유지**: 얼굴 교체 후 입 부분을 원본 이미지로 복원하는 고급 기능
- **🗑️ 결과 관리**: 생성된 결과 이미지 파일 삭제 및 화면 초기화
- **📊 스마트 인덱싱**: 박스 안쪽에 큰 폰트로 표시되는 직관적인 얼굴 인덱스

## 🎯 얼굴 인덱스 시스템

얼굴은 **왼쪽에서 오른쪽으로, x좌표가 동일하면 위에서 아래 순서**로 인덱스가 부여됩니다:

```
이미지에서 얼굴 배치:
[얼굴1] [얼굴2] [얼굴3]
[얼굴4] [얼굴5] [얼굴6]
[얼굴7] [얼굴8] [얼굴9]

인덱스 순서: 1, 2, 3, 4, 5, 6, 7, 8, 9
```

- **사용자 입력**: 1-based (1, 2, 3, ...)
- **내부 처리**: 0-based (0, 1, 2, ...)
- **자동 변환**: UI에서 입력한 번호가 자동으로 내부 인덱스로 변환

## 🚀 빠른 시작

### 1. 저장소 클론
```bash
git clone <repository-url>
cd small-reactor
```

### 2. 가상환경 생성 및 활성화
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. 의존성 설치

#### GPU 버전 (권장)
```bash
# PyTorch GPU 버전 (사용자가 직접 설치)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu

# 나머지 의존성
pip install -r requirements.txt
```

#### CPU 버전
```bash
pip install -r requirements.txt
```

### 4. 모델 파일 준비
`./models` 디렉토리에 다음 모델 파일들을 배치하세요:
- `buffalo_l/` - Buffalo_L 얼굴 탐지 모델
- `inswapper_128.onnx` - 얼굴 교체 모델
- `codeformer-v0.1.0.pth` - 이미지 복원 모델

### 5. 환경 설정
```bash
cp env.example .env
# .env 파일을 편집하여 설정 조정
```

### 6. 애플리케이션 실행
```bash
python gradio_face_manager.py
```

웹 브라우저에서 `http://localhost:7860`으로 접속하세요.

## 🎮 사용 방법

### 웹 인터페이스 구성
애플리케이션은 3개의 주요 탭으로 구성되어 있습니다:

1. **🔄 얼굴 교체**: 메인 기능 - 얼굴 교체 및 복원
2. **📸 얼굴 추출**: 새로운 얼굴 데이터베이스 구축
3. **📋 Embedding 목록**: 저장된 얼굴 정보 갤러리 관리

### 얼굴 교체
1. **타겟 이미지 업로드**: 드래그 앤 드롭으로 이미지 업로드
2. **얼굴 인덱스 선택**: 교체할 얼굴 번호 입력 (예: "1,3,5" 또는 비워두면 모든 얼굴)
3. **소스 얼굴 선택**: 드롭다운에서 교체할 얼굴 선택 (🔄 새로고침 버튼으로 목록 업데이트)
4. **CodeFormer 복원**: 체크박스로 선택적 이미지 품질 향상 (기본값: 체크됨)
5. **입 원본유지**: 체크박스로 입 부분 원본 복원 기능 활성화
   - **확장 비율**: 입 마스크 확장 정도 (기본값: 0.2)
   - **크기 조정**: 가로/세로 스케일링 (기본값: 1.0)
   - **위치 조정**: 가로/세로 오프셋 (기본값: 0)
6. **얼굴 변경**: 버튼 클릭으로 교체 실행
7. **결과 관리**: 🗑️ 버튼으로 결과 이미지 파일 삭제 및 화면 초기화

### 얼굴 추출
1. **이미지 업로드**: 드래그 앤 드롭으로 이미지 업로드
2. **얼굴 추출**: 첫 번째 얼굴 자동 추출 및 저장
3. **Embedding 관리**: 추출된 얼굴 정보를 `./faces` 디렉토리에 저장

### Embedding 목록 관리
1. **갤러리 보기**: 저장된 모든 얼굴을 이미지 갤러리로 확인
2. **새로고침**: 목록 새로고침 버튼으로 최신 상태 유지
3. **파일 관리**: `./faces` 디렉토리에서 직접 파일 추가/삭제 가능

## 🏗️ 프로젝트 구조

```
small-reactor/
├── src/                           # 소스 코드
│   ├── interfaces/               # SOLID 원칙의 ISP 적용
│   │   ├── __init__.py
│   │   ├── face_detector.py      # 얼굴 탐지 인터페이스
│   │   ├── face_swapper.py       # 얼굴 교체 인터페이스
│   │   ├── image_enhancer.py     # 이미지 향상 인터페이스
│   │   └── image_processor.py    # 이미지 처리 인터페이스
│   ├── services/                 # SRP 원칙 적용
│   │   ├── __init__.py
│   │   ├── buffalo_detector.py   # Buffalo_L 얼굴 탐지 서비스
│   │   ├── inswapper_detector.py # inswapper 얼굴 교체 서비스
│   │   └── codeformer_enhancer.py # CodeFormer 이미지 복원 서비스
│   ├── utils/                    # 공통 유틸리티
│   │   ├── __init__.py
│   │   ├── config.py             # 설정 관리
│   │   └── mouth_mask.py         # 입 마스크 생성 유틸리티
│   └── ui/                       # UI 관련 모듈
│       └── __init__.py
├── tests/                        # 테스트 코드
│   ├── __init__.py
│   ├── conftest.py              # pytest 설정
│   ├── fixtures/                # 테스트 픽스처
│   │   ├── mock_models/         # 모의 모델 파일
│   │   └── sample_images/       # 샘플 이미지
│   ├── images/                  # 테스트용 이미지
│   │   ├── face.jpg
│   │   ├── faces.jpg
│   │   └── sosi.jpg
│   ├── test_buffalo_detector.py # BuffaloDetector 테스트
│   ├── test_inswapper_detector.py # InswapperDetector 테스트
│   ├── test_integration.py      # 통합 테스트
│   └── test_interfaces.py       # 인터페이스 테스트
├── docs/                         # 문서
│   ├── API_REFERENCE.md         # API 문서
│   ├── DEVELOPER_GUIDE.md       # 개발자 가이드
│   ├── GPU_INSTALLATION.md      # GPU 설치 가이드
│   └── USER_GUIDE.md            # 사용자 가이드
├── models/                       # AI 모델 파일
│   ├── buffalo_l/               # Buffalo_L 얼굴 탐지 모델
│   │   ├── 1k3d68.onnx
│   │   ├── 2d106det.onnx
│   │   ├── det_10g.onnx
│   │   ├── genderage.onnx
│   │   └── w600k_r50.onnx
│   ├── inswapper_128.onnx       # 얼굴 교체 모델
│   ├── codeformer-v0.1.0.pth    # 이미지 복원 모델
│   ├── RealESRGAN_x2plus.pth    # 이미지 업스케일링 모델
│   └── RealESRGAN_x4plus.pth
├── faces/                        # 추출된 얼굴 데이터
│   ├── 강성연2.jpg              # 얼굴 이미지
│   ├── 강성연2.json             # 얼굴 임베딩
│   ├── 고아성.jpg
│   ├── 고아성.json
│   └── ...                      # 기타 얼굴 데이터
├── outputs/                      # 결과 이미지 저장
├── logs/                         # 로그 파일
├── temp/                         # 임시 파일
├── scripts/                      # 설치 스크립트
│   ├── install_gpu.bat          # Windows GPU 설치
│   └── install_gpu.sh           # Linux/Mac GPU 설치
├── gradio_face_manager.py       # 메인 Gradio UI 애플리케이션
├── face_swap_demo.py            # 데모 스크립트
├── requirements.txt             # Python 의존성
├── pytest.ini                  # pytest 설정
├── env.example                  # 환경 설정 예시
├── .env                         # 환경 설정 (생성 필요)
├── .gitignore                   # Git 무시 파일
├── LICENSE                      # 라이선스
├── README.md                    # 프로젝트 설명
├── CHANGELOG.md                 # 변경 이력
├── PROJECT_PLAN.md              # 프로젝트 계획
└── TESTING_STRATEGY.md          # 테스트 전략
```

## 🧪 테스트

TDD 방식으로 개발되었으며, 포괄적인 테스트 스위트를 포함합니다.

```bash
# 모든 테스트 실행
pytest

# 커버리지와 함께 실행
pytest --cov=src --cov-report=html

# 특정 테스트 실행
pytest tests/test_buffalo_detector.py
pytest tests/test_inswapper_detector.py
pytest tests/test_integration.py
```

## ⚙️ 환경 설정

`.env` 파일에서 다음 설정을 조정할 수 있습니다:

```env
# GPU 사용 설정
USE_GPU=true

# 모델 경로 설정
MODELS_PATH=./models

# 결과 이미지 저장 경로
OUTPUT_PATH=./outputs

# 얼굴 임베딩 저장 경로
FACES_PATH=./faces

# Gradio 서버 설정
GRADIO_SERVER_PORT=7860
```

## 🎯 주요 특징

### 🎨 사용자 경험
- **드래그 앤 드롭**: 새로운 이미지 드래그 시 자동 교체
- **실시간 미리보기**: 업로드된 이미지 즉시 표시
- **한글 파일명 지원**: PIL을 통한 안전한 파일 처리
- **색상 공간 일관성**: RGB/BGR 변환 최적화
- **갤러리 인터페이스**: 저장된 얼굴을 이미지 갤러리로 직관적 관리
- **실시간 새로고침**: 새로고침 버튼으로 목록 즉시 업데이트
- **자동 기본 선택**: 첫 번째 저장된 얼굴을 기본값으로 자동 선택
- **스마트 인덱싱**: 박스 안쪽에 큰 폰트로 표시되는 직관적인 얼굴 번호
- **고급 입 복원**: 혀나 입 안 물체 등 왜곡 방지를 위한 입 부분 원본 유지
- **결과 관리**: 생성된 파일 삭제 및 화면 초기화 기능

### 🔧 기술적 특징
- **SOLID 원칙 준수**: 유지보수 가능한 코드 구조
- **TDD 개발**: 테스트 주도 개발 방식
- **GPU 가속**: CUDA를 통한 고속 처리
- **모듈화 설계**: 각 기능별 독립적인 서비스

### 📊 성능 최적화
- **배치 처리**: 여러 얼굴 동시 처리
- **메모리 효율성**: 필요한 모델만 로드
- **캐싱**: 임베딩 데이터 재사용
- **비동기 처리**: UI 응답성 향상

## 🛠️ 개발 가이드

### SOLID 원칙 준수
- **SRP**: 각 클래스는 하나의 책임만 담당
- **OCP**: 확장에는 열려있고 수정에는 닫혀있음
- **LSP**: 하위 타입은 상위 타입을 완전히 대체 가능
- **ISP**: 작고 구체적인 인터페이스 분리
- **DIP**: 추상화에 의존, 구체적 구현에 의존하지 않음

### TDD 개발 프로세스
1. **Red**: 실패하는 테스트 작성
2. **Green**: 테스트를 통과하는 최소한의 코드 작성
3. **Refactor**: 코드 개선 및 리팩토링

## 📋 시스템 요구사항

### 최소 요구사항
- **Python**: 3.8 이상 (3.10 권장)
- **RAM**: 8GB 이상
- **저장공간**: 5GB 이상

### 권장 요구사항 (GPU)
- **NVIDIA GPU**: GTX 1060 이상 (RTX 시리즈 권장)
- **VRAM**: 6GB 이상 (8GB 이상 권장)
- **CUDA**: 11.8 또는 12.1+
- **RAM**: 16GB 이상

## 🐛 문제 해결

### 자주 발생하는 문제

1. **GPU 인식 안됨**
   ```bash
   # CUDA 설치 확인
   nvidia-smi
   
   # PyTorch GPU 버전 확인
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **모델 로딩 실패**
   - `./models` 디렉토리에 필요한 모델 파일이 있는지 확인
   - 파일 권한 및 경로 확인

3. **메모리 부족**
   - GPU VRAM 부족 시 CPU 모드로 전환
   - 이미지 크기 축소

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 지원

문제가 발생하거나 질문이 있으시면 이슈를 생성해 주세요.