# API Reference

Face Swap Application의 API 문서입니다.

## 📋 목차

- [Core Services](#core-services)
- [Interfaces](#interfaces)
- [Configuration](#configuration)
- [Error Handling](#error-handling)

## 🔧 Core Services

### BuffaloDetector

Buffalo_L 모델을 사용한 얼굴 탐지 서비스입니다. `IFaceDetector` 인터페이스를 구현합니다.

#### Methods

##### `__init__(config: Config)`
BuffaloDetector 인스턴스를 초기화합니다.

**Parameters:**
- `config` (Config): 설정 객체

**Raises:**
- `RuntimeError`: 모델 초기화 실패 시

**Example:**
```python
from src.services.buffalo_detector import BuffaloDetector
from src.utils.config import Config

config = Config()
detector = BuffaloDetector(config)
```

##### `detect_faces(image: np.ndarray) -> List[Face]`
이미지에서 얼굴을 탐지하고 좌에서 우, 위에서 아래 순으로 정렬합니다.

**Parameters:**
- `image` (np.ndarray): 입력 이미지 (BGR 또는 RGB)

**Returns:**
- `List[Face]`: 탐지된 얼굴 리스트 (정렬됨)

**Raises:**
- `ValueError`: 이미지가 유효하지 않은 경우
- `RuntimeError`: 모델 로딩 또는 추론 실패 시

**Example:**
```python
import cv2
import numpy as np

# 이미지 로드
image = cv2.imread("test_image.jpg")

# 얼굴 탐지
faces = detector.detect_faces(image)
print(f"탐지된 얼굴 수: {len(faces)}")

# 각 얼굴 정보 출력
for i, face in enumerate(faces):
    print(f"얼굴 {i+1}: 신뢰도 {face.det_score:.3f}")
    print(f"  중심점: {face.get_center()}")
    print(f"  크기: {face.get_size()}")
```

##### `is_initialized() -> bool`
모델이 초기화되었는지 확인합니다.

**Returns:**
- `bool`: 초기화 여부

##### `get_model_info() -> dict`
모델 정보를 반환합니다.

**Returns:**
- `dict`: 모델 정보 딕셔너리

##### `_sort_faces_by_position(faces: List[Face]) -> List[Face]`
얼굴들을 위치 기준으로 정렬합니다. (내부 메서드)

**Parameters:**
- `faces` (List[Face]): 정렬할 얼굴 리스트

**Returns:**
- `List[Face]`: 정렬된 얼굴 리스트

### InswapperDetector

inswapper_128.onnx 모델을 사용한 얼굴 교체 서비스입니다. `IFaceSwapper` 인터페이스를 구현합니다.

#### Methods

##### `__init__(config: Config)`
InswapperDetector 인스턴스를 초기화합니다.

**Parameters:**
- `config` (Config): 설정 객체

**Raises:**
- `FileNotFoundError`: 모델 파일을 찾을 수 없는 경우
- `RuntimeError`: 모델 초기화 실패 시

**Example:**
```python
from src.services.inswapper_detector import InswapperDetector
from src.utils.config import Config

config = Config()
swapper = InswapperDetector(config)
```

##### `swap_faces(target_image: np.ndarray, target_faces: List[Face], source_face: Face) -> np.ndarray`
타겟 이미지의 얼굴들을 소스 얼굴로 교체합니다.

**Parameters:**
- `target_image` (np.ndarray): 타겟 이미지 (BGR)
- `target_faces` (List[Face]): 교체할 타겟 얼굴들
- `source_face` (Face): 소스 얼굴

**Returns:**
- `np.ndarray`: 교체된 이미지 (BGR)

**Raises:**
- `ValueError`: 입력 이미지나 얼굴이 유효하지 않은 경우
- `RuntimeError`: 모델 추론 실패 시

**Example:**
```python
import cv2
import numpy as np

# 타겟 이미지와 얼굴들
target_image = cv2.imread("target.jpg")
target_faces = detector.detect_faces(target_image)

# 소스 얼굴
source_image = cv2.imread("source.jpg")
source_faces = detector.detect_faces(source_image)
source_face = source_faces[0]

# 얼굴 교체
result_image = swapper.swap_faces(target_image, target_faces, source_face)

# 결과 저장
cv2.imwrite("result.jpg", result_image)
```

##### `is_initialized() -> bool`
모델이 초기화되었는지 확인합니다.

**Returns:**
- `bool`: 초기화 여부

### CodeFormerEnhancer

codeformer-v0.1.0.pth 모델을 사용한 이미지 복원 서비스입니다. `IImageEnhancer` 인터페이스를 구현합니다.

#### Methods

##### `__init__(config: Config)`
CodeFormerEnhancer 인스턴스를 초기화합니다.

**Parameters:**
- `config` (Config): 설정 객체

**Raises:**
- `FileNotFoundError`: 모델 파일을 찾을 수 없는 경우
- `RuntimeError`: 모델 초기화 실패 시

**Example:**
```python
from src.services.codeformer_enhancer import CodeFormerEnhancer
from src.utils.config import Config

config = Config()
enhancer = CodeFormerEnhancer(config)
```

##### `enhance_faces(image: np.ndarray, faces: List[Face]) -> np.ndarray`
얼굴 영역을 복원합니다.

**Parameters:**
- `image` (np.ndarray): 입력 이미지 (BGR)
- `faces` (List[Face]): 복원할 얼굴들

**Returns:**
- `np.ndarray`: 복원된 이미지 (BGR)

**Raises:**
- `ValueError`: 입력 이미지나 얼굴이 유효하지 않은 경우
- `RuntimeError`: 모델 추론 실패 시

**Example:**
```python
import cv2
import numpy as np

# 이미지와 얼굴들
image = cv2.imread("input.jpg")
faces = detector.detect_faces(image)

# 얼굴 복원
enhanced_image = enhancer.enhance_faces(image, faces)

# 결과 저장
cv2.imwrite("enhanced_result.jpg", enhanced_image)
```

##### `is_initialized() -> bool`
모델이 초기화되었는지 확인합니다.

**Returns:**
- `bool`: 초기화 여부

##### `restore(face_crop: np.ndarray, w: float = 0.5) -> np.ndarray`
단일 얼굴 영역을 복원합니다. (내부 메서드)

**Parameters:**
- `face_crop` (np.ndarray): 얼굴 영역 이미지
- `w` (float): 복원 강도 (0.0 ~ 1.0)

**Returns:**
- `np.ndarray`: 복원된 얼굴 영역

## 🔌 Interfaces

### IFaceDetector

얼굴 탐지 인터페이스입니다.

#### Methods

##### `detect_faces(image: np.ndarray) -> List[Face]`
이미지에서 얼굴을 탐지합니다.

**Parameters:**
- `image` (np.ndarray): 입력 이미지

**Returns:**
- `List[Face]`: 탐지된 얼굴 리스트

### IFaceSwapper

얼굴 교체 인터페이스입니다.

#### Methods

##### `swap_faces(target_image: np.ndarray, target_faces: List[Face], source_face: Face) -> np.ndarray`
얼굴을 교체합니다.

**Parameters:**
- `target_image` (np.ndarray): 타겟 이미지
- `target_faces` (List[Face]): 교체할 타겟 얼굴들
- `source_face` (Face): 소스 얼굴

**Returns:**
- `np.ndarray`: 교체된 이미지

### IImageEnhancer

이미지 향상 인터페이스입니다.

#### Methods

##### `enhance_faces(image: np.ndarray, faces: List[Face]) -> np.ndarray`
얼굴 영역을 향상시킵니다.

**Parameters:**
- `image` (np.ndarray): 입력 이미지
- `faces` (List[Face]): 향상시킬 얼굴들

**Returns:**
- `np.ndarray`: 향상된 이미지

## ⚙️ Configuration

### Config

애플리케이션 설정을 관리하는 클래스입니다.

#### Methods

##### `__init__(config_path: str = ".env")`
Config 인스턴스를 초기화합니다.

**Parameters:**
- `config_path` (str): 설정 파일 경로

##### `is_gpu_available() -> bool`
GPU 사용 가능 여부를 확인합니다.

**Returns:**
- `bool`: GPU 사용 가능 여부

##### `get_model_path(model_name: str) -> str`
모델 경로를 반환합니다.

**Parameters:**
- `model_name` (str): 모델 이름

**Returns:**
- `str`: 모델 경로

##### `get(key: str, default: Any = None) -> Any`
설정 값을 반환합니다.

**Parameters:**
- `key` (str): 설정 키
- `default` (Any): 기본값

**Returns:**
- `Any`: 설정 값

## 📊 Data Models

### Face

얼굴 정보를 담는 데이터 클래스입니다.

#### Attributes

- `bbox` (List[float]): 바운딩 박스 [x1, y1, x2, y2]
- `kps` (np.ndarray): 얼굴 랜드마크 포인트
- `embedding` (np.ndarray): 얼굴 임베딩 벡터
- `det_score` (float): 탐지 신뢰도 점수
- `age` (Optional[int]): 예상 나이
- `gender` (Optional[str]): 예상 성별

#### Methods

##### `get_center() -> Tuple[float, float]`
얼굴 중심점을 반환합니다.

**Returns:**
- `Tuple[float, float]`: (center_x, center_y)

##### `get_size() -> Tuple[float, float]`
얼굴 크기를 반환합니다.

**Returns:**
- `Tuple[float, float]`: (width, height)

## 🚨 Error Handling

### Custom Exceptions

#### `FaceDetectionError`
얼굴 탐지 관련 오류입니다.

**Attributes:**
- `message` (str): 오류 메시지
- `image_shape` (Optional[Tuple]): 이미지 크기

#### `FaceSwapError`
얼굴 교체 관련 오류입니다.

**Attributes:**
- `message` (str): 오류 메시지
- `face_count` (Optional[int]): 얼굴 수

#### `ModelLoadError`
모델 로딩 관련 오류입니다.

**Attributes:**
- `message` (str): 오류 메시지
- `model_path` (str): 모델 경로

### Error Handling Best Practices

```python
try:
    faces = detector.detect_faces(image)
except FaceDetectionError as e:
    logger.error(f"얼굴 탐지 실패: {e.message}")
    return None
except Exception as e:
    logger.error(f"예상치 못한 오류: {e}")
    raise
```

## 📝 Usage Examples

### Complete Workflow Example

```python
import cv2
import numpy as np
from src.services.buffalo_detector import BuffaloDetector
from src.services.inswapper_detector import InswapperDetector
from src.services.codeformer_enhancer import CodeFormerEnhancer
from src.utils.config import Config

# 설정 로드
config = Config()

# 서비스 초기화
detector = BuffaloDetector(config)
swapper = InswapperDetector(config)
enhancer = CodeFormerEnhancer(config)

# 타겟 이미지 로드
target_image = cv2.imread("target.jpg")
target_faces = detector.detect_faces(target_image)
print(f"탐지된 얼굴 수: {len(target_faces)}")

# 소스 얼굴 로드
source_image = cv2.imread("source.jpg")
source_faces = detector.detect_faces(source_image)
source_face = source_faces[0]

# 얼굴 교체
result_image = swapper.swap_faces(target_image, target_faces, source_face)
cv2.imwrite("swapped_result.jpg", result_image)

# 얼굴 복원
enhanced_image = enhancer.enhance_faces(result_image, target_faces)
cv2.imwrite("final_result.jpg", enhanced_image)
```

### Face Detection Only

```python
from src.services.buffalo_detector import BuffaloDetector
from src.utils.config import Config
import cv2

# 설정 로드
config = Config()

# 탐지기 초기화
detector = BuffaloDetector(config)

# 이미지 로드
image = cv2.imread("test_image.jpg")

# 얼굴 탐지
faces = detector.detect_faces(image)

# 결과 출력
for i, face in enumerate(faces):
    print(f"얼굴 {i+1}: 신뢰도 {face.det_score:.3f}")
    print(f"  중심점: {face.get_center()}")
    print(f"  크기: {face.get_size()}")
    print(f"  바운딩박스: {face.bbox}")
```

### Face Swapping Only

```python
from src.services.inswapper_detector import InswapperDetector
import cv2

# 교체기 초기화
swapper = InswapperDetector(config)

# 타겟 이미지와 얼굴들
target_image = cv2.imread("target.jpg")
target_faces = detector.detect_faces(target_image)

# 소스 얼굴
source_image = cv2.imread("source.jpg")
source_faces = detector.detect_faces(source_image)
source_face = source_faces[0]

# 얼굴 교체
result_image = swapper.swap_faces(target_image, target_faces, source_face)

# 결과 저장
cv2.imwrite("result.jpg", result_image)
```

### Face Enhancement Only

```python
from src.services.codeformer_enhancer import CodeFormerEnhancer

# 향상기 초기화
enhancer = CodeFormerEnhancer(config)

# 얼굴 향상
enhanced_image = enhancer.enhance_faces(result_image, target_faces)

# 결과 저장
cv2.imwrite("enhanced_result.jpg", enhanced_image)
```

### Error Handling Example

```python
try:
    # 얼굴 탐지
    faces = detector.detect_faces(image)
    
    if not faces:
        print("얼굴을 찾을 수 없습니다.")
        return
    
    # 얼굴 교체
    result_image = swapper.swap_faces(target_image, faces, source_face)
    
    # 얼굴 복원
    enhanced_image = enhancer.enhance_faces(result_image, faces)
    
except FileNotFoundError as e:
    print(f"파일을 찾을 수 없습니다: {e}")
except ValueError as e:
    print(f"입력 값 오류: {e}")
except RuntimeError as e:
    print(f"모델 실행 오류: {e}")
except Exception as e:
    print(f"예상치 못한 오류: {e}")
```

## 🔍 Logging

애플리케이션은 구조화된 로깅을 사용합니다.

### Log Levels

- `DEBUG`: 상세한 디버그 정보
- `INFO`: 일반적인 정보
- `WARNING`: 경고 메시지
- `ERROR`: 오류 메시지
- `CRITICAL`: 치명적인 오류

### Log Format

```
2024-01-01 12:00:00,000 - INFO - src.services.buffalo_detector - Buffalo_L model initialized successfully. GPU: True
```

### Logging Configuration

```python
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
```



