# 개발자 가이드

Face Swap Application의 개발 및 확장을 위한 가이드입니다.

## 📋 목차

- [개발 환경 설정](#개발-환경-설정)
- [아키텍처 개요](#아키텍처-개요)
- [SOLID 원칙 적용](#solid-원칙-적용)
- [TDD 개발 프로세스](#tdd-개발-프로세스)
- [코드 스타일 가이드](#코드-스타일-가이드)
- [테스트 작성](#테스트-작성)
- [새로운 기능 추가](#새로운-기능-추가)
- [성능 최적화](#성능-최적화)
- [디버깅](#디버깅)
- [최신 기능 구현](#최신-기능-구현)

## 🛠️ 개발 환경 설정

### 필수 도구

```bash
# Python 3.8+ (3.10 권장)
python --version

# Git
git --version

# 가상환경 도구
python -m venv venv
```

### 개발 의존성

```bash
# 개발 도구 설치
pip install pytest pytest-cov pytest-mock
pip install black flake8 mypy
pip install pre-commit

# 코드 포맷팅 설정
black --config pyproject.toml .
flake8 --config setup.cfg .
```

### IDE 설정

#### VS Code 권장 확장
- Python
- Pylance
- Black Formatter
- Flake8
- pytest

#### 설정 파일 (.vscode/settings.json)
```json
{
    "python.defaultInterpreterPath": "./venv/Scripts/python.exe",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"]
}
```

## 🏗️ 아키텍처 개요

### 전체 구조

```
src/
├── interfaces/          # 인터페이스 정의 (ISP)
│   ├── face_detector.py
│   ├── face_swapper.py
│   ├── image_enhancer.py
│   └── image_processor.py
├── services/           # 구체적 구현 (SRP)
│   ├── buffalo_detector.py
│   ├── inswapper_detector.py
│   └── codeformer_enhancer.py
├── ui/                 # 사용자 인터페이스
│   └── gradio_interface.py
└── utils/              # 공통 유틸리티
    ├── config.py
    ├── image_utils.py
    └── error_handler.py
```

### 의존성 흐름

```
UI Layer (Gradio)
    ↓
Service Layer (BuffaloDetector, InswapperDetector, CodeFormerEnhancer)
    ↓
Interface Layer (IFaceDetector, IFaceSwapper, IImageEnhancer)
    ↓
Utility Layer (Config, ImageUtils, ErrorHandler)
```

## 🎯 SOLID 원칙 적용

### 1. Single Responsibility Principle (SRP)

각 클래스는 하나의 책임만 가집니다.

```python
# ✅ 좋은 예: 각 클래스가 하나의 책임
class BuffaloDetector:
    """얼굴 탐지만 담당"""
    def detect_faces(self, image): pass

class InswapperDetector:
    """얼굴 교체만 담당"""
    def swap_faces(self, target_image, target_faces, source_face): pass

# ❌ 나쁜 예: 여러 책임을 가진 클래스
class FaceProcessor:
    """얼굴 탐지, 교체, 향상을 모두 담당 - SRP 위반"""
    def detect_faces(self): pass
    def swap_faces(self): pass
    def enhance_faces(self): pass
```

### 2. Open/Closed Principle (OCP)

확장에는 열려있고, 수정에는 닫혀있습니다.

```python
# ✅ 좋은 예: 새로운 탐지기 추가 시 기존 코드 수정 불필요
class IFaceDetector(ABC):
    @abstractmethod
    def detect_faces(self, image): pass

class BuffaloDetector(IFaceDetector):
    def detect_faces(self, image): pass

class RetinaFaceDetector(IFaceDetector):  # 새로운 구현 추가
    def detect_faces(self, image): pass
```

### 3. Liskov Substitution Principle (LSP)

하위 타입은 상위 타입을 완전히 대체할 수 있어야 합니다.

```python
# ✅ 좋은 예: 모든 구현이 동일한 인터페이스 준수
def process_faces(detector: IFaceDetector, image):
    return detector.detect_faces(image)  # 어떤 구현이든 동일하게 작동

# 사용 예
buffalo_detector = BuffaloDetector(config)
retina_detector = RetinaFaceDetector(config)

# 둘 다 동일하게 사용 가능
faces1 = process_faces(buffalo_detector, image)
faces2 = process_faces(retina_detector, image)
```

### 4. Interface Segregation Principle (ISP)

클라이언트는 사용하지 않는 인터페이스에 의존하면 안 됩니다.

```python
# ✅ 좋은 예: 작고 구체적인 인터페이스
class IFaceDetector(ABC):
    @abstractmethod
    def detect_faces(self, image): pass

class IFaceSwapper(ABC):
    @abstractmethod
    def swap_faces(self, target_image, target_faces, source_face): pass

# ❌ 나쁜 예: 큰 인터페이스
class IFaceProcessor(ABC):
    @abstractmethod
    def detect_faces(self, image): pass
    @abstractmethod
    def swap_faces(self, target_image, target_faces, source_face): pass
    @abstractmethod
    def enhance_faces(self, image, faces): pass
    @abstractmethod
    def save_faces(self, faces): pass  # 모든 클라이언트가 필요하지 않음
```

### 5. Dependency Inversion Principle (DIP)

고수준 모듈은 저수준 모듈에 의존하면 안 됩니다.

```python
# ✅ 좋은 예: 추상화에 의존
class FaceSwapOrchestrator:
    def __init__(self, detector: IFaceDetector, swapper: IFaceSwapper):
        self.detector = detector  # 구체적 구현이 아닌 인터페이스에 의존
        self.swapper = swapper

# ❌ 나쁜 예: 구체적 구현에 의존
class FaceSwapOrchestrator:
    def __init__(self):
        self.detector = BuffaloDetector()  # 구체적 구현에 의존
        self.swapper = InswapperDetector()
```

## 🧪 TDD 개발 프로세스

### Red-Green-Refactor 사이클

#### 1. Red 단계: 실패하는 테스트 작성

```python
def test_detect_faces_with_valid_image_returns_face_list():
    # Arrange
    detector = BuffaloDetector(config)
    image = create_test_image()
    
    # Act
    faces = detector.detect_faces(image)
    
    # Assert
    assert len(faces) > 0
    assert all(isinstance(face, Face) for face in faces)
```

#### 2. Green 단계: 테스트를 통과하는 최소한의 코드 작성

```python
class BuffaloDetector:
    def detect_faces(self, image):
        # 최소한의 구현으로 테스트 통과
        return [Face(bbox=[0, 0, 100, 100], embedding=np.zeros(512))]
```

#### 3. Refactor 단계: 코드 개선

```python
class BuffaloDetector:
    def detect_faces(self, image):
        # 실제 구현으로 리팩토링
        faces = self._face_analysis.get(image)
        return [self._convert_face(face) for face in faces]
```

### 테스트 작성 가이드라인

#### 테스트 명명 규칙

```python
def test_<기능>_<조건>_<예상결과>():
    pass

# 예시
def test_detect_faces_with_valid_image_returns_face_list():
    pass

def test_detect_faces_with_invalid_image_raises_exception():
    pass

def test_swap_faces_with_single_face_returns_swapped_image():
    pass
```

#### AAA 패턴 (Arrange-Act-Assert)

```python
def test_example():
    # Arrange - 테스트 데이터 준비
    detector = BuffaloDetector(config)
    image = create_test_image()
    
    # Act - 테스트 실행
    faces = detector.detect_faces(image)
    
    # Assert - 결과 검증
    assert len(faces) == 1
    assert faces[0].det_score > 0.5
```

## 📝 코드 스타일 가이드

### Python 스타일 가이드

#### 함수 및 변수 명명

```python
# ✅ 좋은 예: snake_case 사용
def detect_faces_in_image(image):
    face_list = []
    return face_list

# ❌ 나쁜 예: camelCase 사용
def detectFacesInImage(image):
    faceList = []
    return faceList
```

#### 클래스 명명

```python
# ✅ 좋은 예: PascalCase 사용
class BuffaloDetector:
    pass

class FaceSwapOrchestrator:
    pass

# ❌ 나쁜 예: snake_case 사용
class buffalo_detector:
    pass
```

#### 상수 명명

```python
# ✅ 좋은 예: UPPER_SNAKE_CASE 사용
DEFAULT_MODEL_PATH = "./models"
MAX_FACE_COUNT = 10
MIN_DETECTION_SCORE = 0.5
```

### 문서화

#### Docstring 형식

```python
def detect_faces(self, image: np.ndarray) -> List[Face]:
    """
    이미지에서 얼굴을 탐지합니다.
    
    Args:
        image: 입력 이미지 (BGR 또는 RGB)
        
    Returns:
        탐지된 얼굴 리스트 (좌에서 우, 위에서 아래 순으로 정렬)
        
    Raises:
        ValueError: 이미지가 유효하지 않은 경우
        RuntimeError: 모델 로딩 또는 추론 실패 시
        
    Example:
        >>> detector = BuffaloDetector(config)
        >>> faces = detector.detect_faces(image)
        >>> print(f"탐지된 얼굴 수: {len(faces)}")
    """
```

#### 타입 힌트

```python
from typing import List, Optional, Tuple, Union
import numpy as np

def process_image(
    image: np.ndarray,
    faces: List[Face],
    threshold: float = 0.5
) -> Tuple[bool, str, Optional[np.ndarray]]:
    pass
```

## 🧪 테스트 작성

### 단위 테스트

```python
import pytest
from unittest.mock import Mock, patch
from src.services.buffalo_detector import BuffaloDetector

class TestBuffaloDetector:
    @pytest.fixture
    def detector(self):
        config = Mock()
        config.get_model_path.return_value = "./models"
        return BuffaloDetector(config)
    
    def test_detect_faces_with_valid_image(self, detector):
        # Arrange
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Act
        faces = detector.detect_faces(image)
        
        # Assert
        assert isinstance(faces, list)
        assert all(isinstance(face, Face) for face in faces)
    
    def test_detect_faces_with_none_image_raises_error(self, detector):
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Invalid image"):
            detector.detect_faces(None)
```

### 통합 테스트

```python
class TestFaceSwapIntegration:
    def test_complete_face_swap_workflow(self):
        # Arrange
        config = Config()
        detector = BuffaloDetector(config)
        swapper = InswapperDetector(config)
        
        target_image = load_test_image("target.jpg")
        source_image = load_test_image("source.jpg")
        
        # Act
        target_faces = detector.detect_faces(target_image)
        source_faces = detector.detect_faces(source_image)
        
        result_image = swapper.swap_faces(
            target_image, target_faces, source_faces[0]
        )
        
        # Assert
        assert result_image is not None
        assert result_image.shape == target_image.shape
```

### Mock 사용

```python
@patch('src.services.buffalo_detector.FaceAnalysis')
def test_model_initialization(mock_face_analysis):
    # Arrange
    mock_instance = Mock()
    mock_face_analysis.return_value = mock_instance
    
    # Act
    detector = BuffaloDetector(config)
    
    # Assert
    mock_face_analysis.assert_called_once()
    mock_instance.prepare.assert_called_once()
```

## 🚀 새로운 기능 추가

### 1. 새로운 얼굴 탐지기 추가

#### 인터페이스 구현

```python
# src/services/retina_face_detector.py
from src.interfaces.face_detector import IFaceDetector, Face

class RetinaFaceDetector(IFaceDetector):
    def __init__(self, config: Config):
        self._config = config
        self._initialize_model()
    
    def detect_faces(self, image: np.ndarray) -> List[Face]:
        # 구현
        pass
```

#### 테스트 작성

```python
# tests/test_retina_face_detector.py
class TestRetinaFaceDetector:
    def test_detect_faces(self):
        # 테스트 구현
        pass
```

#### 설정 추가

```python
# src/utils/config.py
def get_model_path(self, model_name: str) -> str:
    model_paths = {
        "buffalo_l": os.path.join(self._models_path, "buffalo_l"),
        "retina_face": os.path.join(self._models_path, "retina_face"),  # 추가
    }
    return model_paths.get(model_name, "")
```

### 2. 새로운 UI 컴포넌트 추가

#### Gradio 컴포넌트

```python
# src/ui/components/face_selector.py
import gradio as gr

class FaceSelector:
    def __init__(self):
        self.face_dropdown = gr.Dropdown(
            label="얼굴 선택",
            choices=[],
            interactive=True
        )
    
    def update_choices(self, faces_dir: str):
        # 드롭다운 선택지 업데이트
        pass
```

### 3. 새로운 유틸리티 추가

```python
# src/utils/image_utils.py
class ImageUtils:
    @staticmethod
    def resize_image(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
        """이미지 크기 조정"""
        pass
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """이미지 정규화"""
        pass
```

## ⚡ 성능 최적화

### 1. 메모리 최적화

```python
class OptimizedDetector:
    def __init__(self):
        self._model_cache = {}
    
    def detect_faces(self, image):
        # 모델 재사용
        if "detector" not in self._model_cache:
            self._model_cache["detector"] = self._load_model()
        
        return self._model_cache["detector"].get(image)
```

### 2. 배치 처리

```python
def process_multiple_images(images: List[np.ndarray]) -> List[List[Face]]:
    """여러 이미지를 배치로 처리"""
    batch_size = 4
    results = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        batch_results = detector.detect_faces_batch(batch)
        results.extend(batch_results)
    
    return results
```

### 3. 비동기 처리

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncFaceProcessor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def process_image_async(self, image: np.ndarray):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self.detect_faces, image
        )
```

## 🐛 디버깅

### 로깅 설정

```python
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 사용 예
logger.debug(f"이미지 크기: {image.shape}")
logger.info(f"탐지된 얼굴 수: {len(faces)}")
logger.warning(f"낮은 신뢰도: {face.det_score}")
logger.error(f"모델 로딩 실패: {e}")
```

### 성능 프로파일링

```python
import time
import cProfile
import pstats

def profile_function(func):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        profiler.disable()
        
        print(f"실행 시간: {end_time - start_time:.2f}초")
        
        # 프로파일 결과 저장
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(10)
        
        return result
    return wrapper

# 사용 예
@profile_function
def detect_faces(self, image):
    return self._face_analysis.get(image)
```

### 메모리 사용량 모니터링

```python
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"메모리 사용량: {memory_info.rss / 1024 / 1024:.2f} MB")

# 사용 예
def detect_faces(self, image):
    monitor_memory()
    faces = self._face_analysis.get(image)
    monitor_memory()
    return faces
```

## 📦 배포

### Docker 설정

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 복사
COPY . .

# 포트 노출
EXPOSE 7860

# 실행
CMD ["python", "gradio_face_manager.py"]
```

### 환경 변수 설정

```bash
# .env.production
USE_GPU=true
MODELS_PATH=/app/models
OUTPUT_PATH=/app/outputs
FACES_PATH=/app/faces
GRADIO_SERVER_PORT=7860
GRADIO_SERVER_NAME=0.0.0.0
```

## 🔄 CI/CD

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.10
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## 🆕 최신 기능 구현

### 입 원본유지 기능

#### 구현 위치
- **핵심 로직**: `src/utils/mouth_mask.py`
- **UI 통합**: `gradio_face_manager.py`의 `apply_mouth_preservation` 메서드
- **처리 순서**: `swap -> CodeFormer 복구 -> 입원본복구`

#### 핵심 코드 구조
```python
def create_mouth_mask(landmarks, image_shape, expand_ratio=0.2, 
                     expand_weights={'scale_x': 1.0, 'scale_y': 1.0, 
                                   'offset_x': 0, 'offset_y': 0}):
    """
    다양한 랜드마크 포인트 수를 지원하는 입 마스크 생성
    - 106포인트: 정확한 입 영역 마스킹
    - 68포인트: 표준 얼굴 랜드마크 지원  
    - 5포인트: 기본 키포인트로 타원형 마스크 생성
    """
```

#### 랜드마크 지원
- **106포인트**: `landmarks[52:72]` - 정확한 입 영역
- **68포인트**: `landmarks[48:68]` - 표준 얼굴 랜드마크
- **5포인트**: `[왼쪽 눈, 오른쪽 눈, 코, 왼쪽 입, 오른쪽 입]` - 타원형 마스크

### 결과 이미지 삭제 기능

#### 구현 위치
- **삭제 함수**: `delete_result_image()` 함수
- **UI 통합**: `🗑️ 결과 이미지 삭제` 버튼
- **파일 선택**: 가장 최근 생성된 `final_result_*.jpg` 파일 자동 선택

#### 핵심 로직
```python
def delete_result_image():
    """
    가장 최근 생성된 final_result 파일을 찾아 삭제
    - 파일 생성 시간으로 정렬
    - 삭제 후 화면 초기화
    - 성공/실패 메시지 반환
    """
```

### 스마트 얼굴 인덱싱

#### 구현 위치
- **박스 그리기**: `draw_face_boxes` 메서드
- **적응형 폰트**: 박스 크기에 비례한 폰트 크기 (1.5~4.0)
- **가독성 향상**: 흰색 배경과 검은색 텍스트

#### 핵심 로직
```python
def draw_face_boxes(self, image: np.ndarray, faces: List) -> np.ndarray:
    """
    박스 안쪽에 큰 폰트로 인덱스 표시
    - 박스 크기에 비례한 폰트 크기 계산
    - 박스 중앙에 정확히 배치
    - 흰색 배경으로 가독성 향상
    """
```

### 처리 순서 최적화

#### 변경된 순서
1. **얼굴 교체 (Swap)**
2. **CodeFormer 복원** (선택사항)
3. **입 원본유지** (선택사항)

#### 장점
- **더 나은 품질**: CodeFormer로 먼저 복원한 후 입 부분만 선택적으로 원본으로 복구
- **자연스러운 결과**: CodeFormer 복원의 이점을 유지하면서 입 부분만 원본 유지
- **유연성**: 각 단계를 독립적으로 제어 가능

이 가이드를 따라 개발하면 유지보수 가능하고 확장 가능한 코드를 작성할 수 있습니다.



