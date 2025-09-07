"""
pytest 공통 설정 및 픽스처

모든 테스트에서 공통으로 사용되는 설정과 픽스처를 정의
"""

import pytest
import numpy as np
import os
from pathlib import Path
from unittest.mock import Mock, patch
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

@pytest.fixture(scope="session")
def test_config():
    """테스트용 설정"""
    return {
        "model_paths": {
            "buffalo_l": "tests/fixtures/mock_models/mock_buffalo_l",
            "inswapper": "tests/fixtures/mock_models/mock_inswapper.onnx",
            "codeformer": "tests/fixtures/mock_models/mock_codeformer.pth"
        },
        "test_images": "tests/fixtures/sample_images/",
        "output_dir": "tests/output/",
        "use_gpu": False,  # 테스트에서는 GPU 사용 안함
        "max_image_size": 512,
        "batch_size": 1
    }

@pytest.fixture
def sample_image():
    """테스트용 샘플 이미지 생성"""
    # 512x512 RGB 이미지 생성
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

@pytest.fixture
def mock_face():
    """Mock Face 객체 생성"""
    face = Mock()
    face.bbox = [100, 100, 200, 200]  # [x1, y1, x2, y2]
    face.kps = np.array([[150, 120], [180, 120], [165, 140], [150, 160], [180, 160]])  # 5개 키포인트
    face.embedding = np.random.rand(512)  # 512차원 임베딩
    face.det_score = 0.95  # 탐지 점수
    face.age = 25
    face.gender = 1
    return face

@pytest.fixture
def mock_face_analysis():
    """Mock FaceAnalysis 객체 생성"""
    with patch('insightface.app.FaceAnalysis') as mock:
        mock_instance = mock.return_value
        mock_instance.prepare.return_value = None
        mock_instance.get.return_value = []
        yield mock_instance

@pytest.fixture
def mock_onnx_session():
    """Mock ONNX Session 객체 생성"""
    with patch('onnxruntime.InferenceSession') as mock:
        mock_instance = mock.return_value
        mock_instance.get_inputs.return_value = [Mock(name='input', shape=[1, 3, 128, 128])]
        mock_instance.get_outputs.return_value = [Mock(name='output', shape=[1, 3, 128, 128])]
        mock_instance.run.return_value = [np.random.rand(1, 3, 128, 128).astype(np.float32)]
        yield mock_instance

@pytest.fixture
def mock_torch_model():
    """Mock PyTorch 모델 객체 생성"""
    with patch('torch.load') as mock:
        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_model.cuda.return_value = mock_model
        mock_model.cpu.return_value = mock_model
        mock.return_value = mock_model
        yield mock_model

@pytest.fixture(autouse=True)
def setup_test_environment():
    """각 테스트 실행 전 환경 설정"""
    # 테스트 출력 디렉토리 생성
    output_dir = Path("tests/output")
    output_dir.mkdir(exist_ok=True)
    
    # 테스트 픽스처 디렉토리 생성
    fixtures_dir = Path("tests/fixtures")
    fixtures_dir.mkdir(exist_ok=True)
    
    (fixtures_dir / "sample_images").mkdir(exist_ok=True)
    (fixtures_dir / "mock_models").mkdir(exist_ok=True)
    
    yield
    
    # 테스트 후 정리 (필요시)
    pass

@pytest.fixture
def temp_image_file(tmp_path):
    """임시 이미지 파일 생성"""
    import cv2
    
    # 테스트용 이미지 생성
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    image_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(image_path), image)
    
    return str(image_path)
