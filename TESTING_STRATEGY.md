# 테스트 전략 문서

## 테스트 철학

### TDD (Test-Driven Development) 접근법
- **Red-Green-Refactor 사이클**: 테스트 실패 → 구현 → 리팩토링
- **테스트 우선 개발**: 모든 기능 구현 전에 테스트 코드 작성
- **지속적인 피드백**: 빠른 피드백을 통한 품질 보장

### SOLID 원칙과 테스트
- **단일 책임 원칙**: 각 클래스별로 명확한 테스트 범위
- **의존성 역전 원칙**: Mock 객체를 통한 의존성 격리
- **인터페이스 분리 원칙**: 인터페이스별 테스트 분리

## 테스트 피라미드

### 1. 단위 테스트 (Unit Tests) - 70%
- **목적**: 개별 클래스/함수의 동작 검증
- **범위**: 각 서비스, 유틸리티, 인터페이스
- **도구**: pytest, pytest-mock
- **목표**: 빠른 실행, 높은 커버리지

### 2. 통합 테스트 (Integration Tests) - 20%
- **목적**: 컴포넌트 간 상호작용 검증
- **범위**: 서비스 간 통합, 모델 로딩, 데이터 흐름
- **도구**: pytest, 실제 모델 파일 사용
- **목표**: 실제 환경과 유사한 테스트

### 3. E2E 테스트 (End-to-End Tests) - 10%
- **목적**: 전체 시스템의 사용자 시나리오 검증
- **범위**: UI부터 백엔드까지 전체 워크플로우
- **도구**: pytest, Gradio 테스트 유틸리티
- **목표**: 실제 사용자 경험 검증

## 테스트 구조

```
tests/
├── unit/                    # 단위 테스트
│   ├── test_interfaces.py
│   ├── test_buffalo_detector.py
│   ├── test_inswapper_service.py
│   ├── test_codeformer_enhancer.py
│   ├── test_face_swap_orchestrator.py
│   ├── test_gradio_interface.py
│   ├── test_image_utils.py
│   ├── test_model_manager.py
│   ├── test_error_handler.py
│   └── test_config.py
├── integration/             # 통합 테스트
│   ├── test_face_detection.py
│   ├── test_face_swapping.py
│   ├── test_image_enhancement.py
│   ├── test_face_swap_pipeline.py
│   ├── test_ui_integration.py
│   └── test_full_system.py
├── e2e/                     # E2E 테스트
│   ├── test_end_to_end.py
│   └── test_ui_e2e.py
├── performance/             # 성능 테스트
│   ├── test_performance.py
│   └── test_memory_usage.py
├── fixtures/                # 테스트 픽스처
│   ├── sample_images/
│   ├── mock_models/
│   └── test_data/
├── conftest.py              # pytest 공통 설정
└── test_utils.py            # 테스트 유틸리티
```

## 테스트 도구 및 라이브러리

### 핵심 테스트 프레임워크
```python
# requirements.txt에 포함될 테스트 관련 패키지
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0
pytest-asyncio>=0.21.0
pytest-xdist>=3.0.0  # 병렬 테스트 실행
pytest-html>=3.1.0   # HTML 리포트
pytest-benchmark>=4.0.0  # 성능 벤치마크
```

### Mock 및 테스트 유틸리티
```python
# 테스트에서 사용할 Mock 라이브러리
unittest.mock  # Python 내장
factory_boy>=3.2.0  # 테스트 데이터 생성
faker>=18.0.0  # 가짜 데이터 생성
responses>=0.23.0  # HTTP 요청 Mock
```

## 테스트 작성 가이드라인

### 1. 테스트 명명 규칙
```python
# 테스트 함수 명명: test_<기능>_<조건>_<예상결과>
def test_detect_faces_with_valid_image_returns_face_list():
    pass

def test_detect_faces_with_invalid_image_raises_exception():
    pass

def test_swap_face_with_missing_model_raises_error():
    pass
```

### 2. 테스트 구조 (AAA 패턴)
```python
def test_example():
    # Arrange - 테스트 데이터 준비
    input_image = load_test_image()
    detector = BuffaloDetector()
    
    # Act - 테스트 실행
    result = detector.detect_faces(input_image)
    
    # Assert - 결과 검증
    assert len(result) > 0
    assert all(isinstance(face, Face) for face in result)
```

### 3. Mock 사용 가이드라인
```python
@pytest.fixture
def mock_face_detector():
    with patch('src.services.buffalo_detector.FaceAnalysis') as mock:
        mock_instance = mock.return_value
        mock_instance.get.return_value = [create_mock_face()]
        yield mock_instance

def test_orchestrator_with_mock_detector(mock_face_detector):
    # Mock을 사용한 테스트
    pass
```

## 테스트 데이터 관리

### 1. 테스트 이미지
```python
# fixtures/sample_images/
├── single_face.jpg          # 단일 얼굴 이미지
├── multiple_faces.jpg       # 다중 얼굴 이미지
├── no_face.jpg             # 얼굴 없는 이미지
├── low_quality.jpg         # 저품질 이미지
└── large_image.jpg         # 대용량 이미지
```

### 2. Mock 모델 파일
```python
# fixtures/mock_models/
├── mock_buffalo_l.onnx     # Mock 얼굴 탐지 모델
├── mock_inswapper.onnx     # Mock 얼굴 교체 모델
└── mock_codeformer.pth     # Mock 이미지 복구 모델
```

### 3. 테스트 설정
```python
# conftest.py
@pytest.fixture(scope="session")
def test_config():
    return {
        "model_paths": {
            "buffalo_l": "fixtures/mock_models/mock_buffalo_l.onnx",
            "inswapper": "fixtures/mock_models/mock_inswapper.onnx",
            "codeformer": "fixtures/mock_models/mock_codeformer.pth"
        },
        "test_images": "fixtures/sample_images/",
        "output_dir": "tests/output/"
    }
```

## 성능 테스트

### 1. 메모리 사용량 테스트
```python
def test_memory_usage():
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # 테스트 실행
    orchestrator = FaceSwapOrchestrator(...)
    result = orchestrator.process_image(test_image)
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # 메모리 사용량이 1GB를 초과하지 않아야 함
    assert memory_increase < 1024 * 1024 * 1024
```

### 2. 처리 시간 테스트
```python
def test_processing_time():
    import time
    
    start_time = time.time()
    result = orchestrator.process_image(test_image)
    end_time = time.time()
    
    processing_time = end_time - start_time
    
    # 처리 시간이 30초를 초과하지 않아야 함
    assert processing_time < 30.0
```

## 테스트 커버리지

### 1. 커버리지 목표
- **전체 커버리지**: 90% 이상
- **핵심 비즈니스 로직**: 95% 이상
- **에러 처리 코드**: 100%

### 2. 커버리지 측정
```bash
# 커버리지 측정 실행
pytest --cov=src --cov-report=html --cov-report=term

# 커버리지 리포트 생성
pytest --cov=src --cov-report=html
```

### 3. 커버리지 설정
```ini
# pytest.ini
[tool:pytest]
addopts = --cov=src --cov-report=term-missing --cov-report=html
cov-fail-under = 90
```

## CI/CD 통합

### 1. GitHub Actions 워크플로우
```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt
      - name: Run tests
        run: pytest --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### 2. 테스트 실행 명령어
```bash
# 모든 테스트 실행
pytest

# 단위 테스트만 실행
pytest tests/unit/

# 통합 테스트만 실행
pytest tests/integration/

# E2E 테스트만 실행
pytest tests/e2e/

# 성능 테스트 실행
pytest tests/performance/

# 커버리지와 함께 실행
pytest --cov=src --cov-report=html

# 병렬 실행
pytest -n auto
```

## 테스트 품질 보장

### 1. 코드 리뷰 체크리스트
- [ ] 모든 새로운 기능에 테스트가 있는가?
- [ ] 테스트가 실제 사용 시나리오를 반영하는가?
- [ ] Mock 사용이 적절한가?
- [ ] 테스트가 독립적으로 실행 가능한가?
- [ ] 테스트 이름이 명확한가?

### 2. 테스트 유지보수
- **정기적인 테스트 리뷰**: 매주 테스트 코드 리뷰
- **테스트 리팩토링**: 중복 코드 제거 및 구조 개선
- **성능 모니터링**: 테스트 실행 시간 추적
- **커버리지 모니터링**: 커버리지 하락 방지

## 결론

이 테스트 전략은 SOLID 원칙을 준수하면서 TDD 방식으로 개발을 진행하여 높은 품질의 코드를 보장합니다. 각 단계별로 적절한 테스트를 작성하고, 지속적인 통합을 통해 품질을 유지합니다.
