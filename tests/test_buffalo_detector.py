"""
BuffaloDetector 단위 테스트

TDD 방식으로 작성된 BuffaloDetector 클래스의 테스트
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.interfaces.face_detector import IFaceDetector, Face
from src.services.buffalo_detector import BuffaloDetector
from src.utils.config import Config


class TestBuffaloDetector:
    """BuffaloDetector 테스트 클래스"""
    
    def test_implements_interface(self):
        """BuffaloDetector가 IFaceDetector 인터페이스를 구현하는지 테스트"""
        # Arrange
        config = Config()
        
        # Act & Assert
        assert issubclass(BuffaloDetector, IFaceDetector)
    
    def test_initialization_with_config(self, test_config):
        """설정으로 초기화하는지 테스트"""
        # Arrange
        config = Config()
        config._config.update(test_config)
        
        # Act
        with patch('src.services.buffalo_detector.FaceAnalysis') as mock_face_analysis:
            detector = BuffaloDetector(config)
            
            # Assert
            assert detector is not None
            assert isinstance(detector, BuffaloDetector)
    
    def test_initialization_without_gpu(self, test_config):
        """GPU 없이 초기화하는지 테스트"""
        # Arrange
        config = Config()
        config._config.update(test_config)
        config._config["use_gpu"] = False
        
        # Act
        with patch('src.services.buffalo_detector.FaceAnalysis') as mock_face_analysis:
            detector = BuffaloDetector(config)
            
            # Assert
            assert detector is not None
            assert not detector._use_gpu
    
    def test_initialization_with_gpu(self, test_config):
        """GPU와 함께 초기화하는지 테스트"""
        # Arrange
        config = Config()
        config._config.update(test_config)
        config._config["use_gpu"] = True
        
        # Act
        with patch('src.services.buffalo_detector.FaceAnalysis') as mock_face_analysis, \
             patch('torch.cuda.is_available', return_value=True):
            detector = BuffaloDetector(config)
            
            # Assert
            assert detector is not None
            assert detector._use_gpu
    
    def test_detect_faces_with_valid_image(self, sample_image, mock_face_analysis, mock_face):
        """유효한 이미지로 얼굴 탐지 테스트"""
        # Arrange
        config = Config()
        mock_face_analysis.get.return_value = [mock_face]
        
        detector = BuffaloDetector(config)
        detector._face_analysis = mock_face_analysis
        
        # Act
        result = detector.detect_faces(sample_image)
        
        # Assert
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Face)
        mock_face_analysis.get.assert_called_once()
    
    def test_detect_faces_with_no_faces(self, sample_image, mock_face_analysis):
        """얼굴이 없는 이미지 테스트"""
        # Arrange
        config = Config()
        mock_face_analysis.get.return_value = []
        
        detector = BuffaloDetector(config)
        detector._face_analysis = mock_face_analysis
        
        # Act
        result = detector.detect_faces(sample_image)
        
        # Assert
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_detect_faces_with_multiple_faces(self, sample_image, mock_face_analysis):
        """여러 얼굴이 있는 이미지 테스트"""
        # Arrange
        config = Config()
        mock_faces = [Mock(), Mock(), Mock()]
        for face in mock_faces:
            face.bbox = [100, 100, 200, 200]
            face.kps = np.array([[150, 120], [180, 120], [165, 140], [150, 160], [180, 160]])
            face.embedding = np.random.rand(512)
            face.det_score = 0.95
            face.age = 25
            face.gender = 1
        
        mock_face_analysis.get.return_value = mock_faces
        
        detector = BuffaloDetector(config)
        detector._face_analysis = mock_face_analysis
        
        # Act
        result = detector.detect_faces(sample_image)
        
        # Assert
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(face, Face) for face in result)
    
    def test_detect_faces_with_invalid_image(self, mock_face_analysis):
        """유효하지 않은 이미지로 얼굴 탐지 테스트"""
        # Arrange
        config = Config()
        invalid_image = None
        
        detector = BuffaloDetector(config)
        detector._face_analysis = mock_face_analysis
        
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid image"):
            detector.detect_faces(invalid_image)
    
    def test_detect_faces_with_empty_image(self, mock_face_analysis):
        """빈 이미지로 얼굴 탐지 테스트"""
        # Arrange
        config = Config()
        empty_image = np.array([])
        
        detector = BuffaloDetector(config)
        detector._face_analysis = mock_face_analysis
        
        # Act & Assert
        with pytest.raises(ValueError, match="Empty image"):
            detector.detect_faces(empty_image)
    
    def test_detect_faces_with_wrong_dimensions(self, mock_face_analysis):
        """잘못된 차원의 이미지로 얼굴 탐지 테스트"""
        # Arrange
        config = Config()
        wrong_image = np.random.rand(100, 100)  # 2D 이미지 (3D여야 함)
        
        detector = BuffaloDetector(config)
        detector._face_analysis = mock_face_analysis
        
        # Act & Assert
        with pytest.raises(ValueError, match="Image must be 3D"):
            detector.detect_faces(wrong_image)
    
    def test_detect_faces_model_not_initialized(self, sample_image):
        """모델이 초기화되지 않은 상태에서 얼굴 탐지 테스트"""
        # Arrange
        config = Config()
        detector = BuffaloDetector(config)
        detector._face_analysis = None
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="Model not initialized"):
            detector.detect_faces(sample_image)
    
    def test_detect_faces_model_error(self, sample_image, mock_face_analysis):
        """모델 오류 발생 시 테스트"""
        # Arrange
        config = Config()
        mock_face_analysis.get.side_effect = Exception("Model error")
        
        detector = BuffaloDetector(config)
        detector._face_analysis = mock_face_analysis
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="Face detection failed"):
            detector.detect_faces(sample_image)
    
    def test_is_initialized_true(self, mock_face_analysis):
        """모델이 초기화된 상태 테스트"""
        # Arrange
        config = Config()
        detector = BuffaloDetector(config)
        detector._face_analysis = mock_face_analysis
        
        # Act
        result = detector.is_initialized()
        
        # Assert
        assert result is True
    
    def test_is_initialized_false(self):
        """모델이 초기화되지 않은 상태 테스트"""
        # Arrange
        config = Config()
        detector = BuffaloDetector(config)
        detector._face_analysis = None
        
        # Act
        result = detector.is_initialized()
        
        # Assert
        assert result is False
    
    def test_get_model_info(self, mock_face_analysis):
        """모델 정보 반환 테스트"""
        # Arrange
        config = Config()
        detector = BuffaloDetector(config)
        detector._face_analysis = mock_face_analysis
        
        # Act
        info = detector.get_model_info()
        
        # Assert
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "model_path" in info
        assert "use_gpu" in info
        assert "initialized" in info
        assert info["model_name"] == "buffalo_l"
        assert info["initialized"] is True
    
    def test_get_model_info_not_initialized(self):
        """초기화되지 않은 모델 정보 반환 테스트"""
        # Arrange
        config = Config()
        detector = BuffaloDetector(config)
        detector._face_analysis = None
        
        # Act
        info = detector.get_model_info()
        
        # Assert
        assert isinstance(info, dict)
        assert info["initialized"] is False
    
    def test_face_conversion(self, mock_face):
        """InsightFace Face 객체를 Face 객체로 변환 테스트"""
        # Arrange
        config = Config()
        detector = BuffaloDetector(config)
        
        # Act
        converted_face = detector._convert_face(mock_face)
        
        # Assert
        assert isinstance(converted_face, Face)
        assert converted_face.bbox == mock_face.bbox
        assert np.array_equal(converted_face.kps, mock_face.kps)
        assert np.array_equal(converted_face.embedding, mock_face.embedding)
        assert converted_face.det_score == mock_face.det_score
        assert converted_face.age == mock_face.age
        assert converted_face.gender == mock_face.gender
    
    def test_face_conversion_with_missing_attributes(self):
        """일부 속성이 없는 Face 객체 변환 테스트"""
        # Arrange
        config = Config()
        detector = BuffaloDetector(config)
        
        mock_face = Mock()
        mock_face.bbox = [100, 100, 200, 200]
        mock_face.kps = np.array([[150, 120], [180, 120], [165, 140], [150, 160], [180, 160]])
        mock_face.embedding = np.random.rand(512)
        mock_face.det_score = 0.95
        # age와 gender는 None으로 설정
        
        # Act
        converted_face = detector._convert_face(mock_face)
        
        # Assert
        assert isinstance(converted_face, Face)
        assert converted_face.age is None
        assert converted_face.gender is None
    
    @pytest.mark.parametrize("det_score", [0.1, 0.5, 0.9, 1.0])
    def test_detect_faces_with_different_scores(self, sample_image, mock_face_analysis, det_score):
        """다양한 탐지 점수로 얼굴 탐지 테스트"""
        # Arrange
        config = Config()
        mock_face = Mock()
        mock_face.bbox = [100, 100, 200, 200]
        mock_face.kps = np.array([[150, 120], [180, 120], [165, 140], [150, 160], [180, 160]])
        mock_face.embedding = np.random.rand(512)
        mock_face.det_score = det_score
        mock_face.age = 25
        mock_face.gender = 1
        
        mock_face_analysis.get.return_value = [mock_face]
        
        detector = BuffaloDetector(config)
        detector._face_analysis = mock_face_analysis
        
        # Act
        result = detector.detect_faces(sample_image)
        
        # Assert
        assert len(result) == 1
        assert result[0].det_score == det_score
    
    def test_detect_faces_with_low_score_filtering(self, sample_image, mock_face_analysis):
        """낮은 점수 얼굴 필터링 테스트"""
        # Arrange
        config = Config()
        config._config["min_det_score"] = 0.5
        
        # 높은 점수 얼굴
        high_score_face = Mock()
        high_score_face.bbox = [100, 100, 200, 200]
        high_score_face.kps = np.array([[150, 120], [180, 120], [165, 140], [150, 160], [180, 160]])
        high_score_face.embedding = np.random.rand(512)
        high_score_face.det_score = 0.8
        high_score_face.age = 25
        high_score_face.gender = 1
        
        # 낮은 점수 얼굴
        low_score_face = Mock()
        low_score_face.bbox = [300, 300, 400, 400]
        low_score_face.kps = np.array([[350, 320], [380, 320], [365, 340], [350, 360], [380, 360]])
        low_score_face.embedding = np.random.rand(512)
        low_score_face.det_score = 0.3
        low_score_face.age = 30
        low_score_face.gender = 0
        
        mock_face_analysis.get.return_value = [high_score_face, low_score_face]
        
        detector = BuffaloDetector(config)
        detector._face_analysis = mock_face_analysis
        
        # Act
        result = detector.detect_faces(sample_image)
        
        # Assert
        assert len(result) == 1  # 높은 점수 얼굴만 반환
        assert result[0].det_score == 0.8
