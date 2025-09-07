"""
InswapperDetector 단위 테스트

TDD 방식으로 작성된 InswapperDetector 클래스의 테스트
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.interfaces.face_swapper import IFaceSwapper
from src.interfaces.face_detector import Face
from src.services.inswapper_detector import InswapperDetector
from src.utils.config import Config


class TestInswapperDetector:
    """InswapperDetector 테스트 클래스"""
    
    def test_implements_interface(self):
        """InswapperDetector가 IFaceSwapper 인터페이스를 구현하는지 테스트"""
        # Arrange
        config = Config()
        
        # Act & Assert
        assert issubclass(InswapperDetector, IFaceSwapper)
    
    def test_initialization_with_config(self, test_config):
        """설정으로 초기화하는지 테스트"""
        # Arrange
        config = Config()
        config._config.update(test_config)
        
        # Act
        with patch('src.services.inswapper_detector.model_zoo.get_model') as mock_get_model:
            mock_swapper = Mock()
            mock_get_model.return_value = mock_swapper
            
            detector = InswapperDetector(config)
            
            # Assert
            assert detector is not None
            assert isinstance(detector, InswapperDetector)
            assert detector._swapper is not None
    
    def test_initialization_without_gpu(self, test_config):
        """GPU 없이 초기화하는지 테스트"""
        # Arrange
        config = Config()
        config._config.update(test_config)
        config._config["use_gpu"] = False
        
        # Act
        with patch('src.services.inswapper_detector.model_zoo.get_model') as mock_get_model:
            mock_swapper = Mock()
            mock_get_model.return_value = mock_swapper
            
            detector = InswapperDetector(config)
            
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
        with patch('src.services.inswapper_detector.model_zoo.get_model') as mock_get_model, \
             patch('torch.cuda.is_available', return_value=True):
            mock_swapper = Mock()
            mock_get_model.return_value = mock_swapper
            
            detector = InswapperDetector(config)
            
            # Assert
            assert detector is not None
            assert detector._use_gpu
    
    def test_initialization_model_not_found(self, test_config):
        """모델 파일이 없는 경우 테스트"""
        # Arrange
        config = Config()
        config._config.update(test_config)
        config._config["inswapper_model_path"] = "nonexistent_model.onnx"
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="Model initialization failed"):
            InswapperDetector(config)
    
    def test_swap_face_with_valid_inputs(self, sample_image, mock_face):
        """유효한 입력으로 얼굴 교체 테스트"""
        # Arrange
        config = Config()
        source_image = sample_image.copy()
        target_image = sample_image.copy()
        
        with patch('src.services.inswapper_detector.model_zoo.get_model') as mock_get_model:
            mock_swapper = Mock()
            mock_swapper.get.return_value = target_image.copy()
            mock_get_model.return_value = mock_swapper
            
            detector = InswapperDetector(config)
            
            # Act
            result = detector.swap_face(source_image, target_image, mock_face, mock_face)
            
            # Assert
            assert isinstance(result, np.ndarray)
            assert result.shape == target_image.shape
            assert result.dtype == target_image.dtype
            mock_swapper.get.assert_called_once_with(target_image, mock_face, mock_face)
    
    def test_swap_face_with_none_images(self, mock_face):
        """None 이미지로 얼굴 교체 테스트"""
        # Arrange
        config = Config()
        
        with patch('src.services.inswapper_detector.model_zoo.get_model') as mock_get_model:
            mock_swapper = Mock()
            mock_get_model.return_value = mock_swapper
            
            detector = InswapperDetector(config)
            
            # Act & Assert
            with pytest.raises(ValueError, match="Images cannot be None"):
                detector.swap_face(None, None, mock_face, mock_face)
    
    def test_swap_face_with_empty_images(self, mock_face):
        """빈 이미지로 얼굴 교체 테스트"""
        # Arrange
        config = Config()
        empty_image = np.array([])
        
        with patch('src.services.inswapper_detector.model_zoo.get_model') as mock_get_model:
            mock_swapper = Mock()
            mock_get_model.return_value = mock_swapper
            
            detector = InswapperDetector(config)
            
            # Act & Assert
            with pytest.raises(ValueError, match="Images cannot be empty"):
                detector.swap_face(empty_image, empty_image, mock_face, mock_face)
    
    def test_swap_face_with_wrong_dimensions(self, mock_face):
        """잘못된 차원의 이미지로 얼굴 교체 테스트"""
        # Arrange
        config = Config()
        wrong_image = np.random.rand(100, 100)  # 2D 이미지 (3D여야 함)
        
        with patch('src.services.inswapper_detector.model_zoo.get_model') as mock_get_model:
            mock_swapper = Mock()
            mock_get_model.return_value = mock_swapper
            
            detector = InswapperDetector(config)
            
            # Act & Assert
            with pytest.raises(ValueError, match="Images must be 3D"):
                detector.swap_face(wrong_image, wrong_image, mock_face, mock_face)
    
    def test_swap_face_with_none_faces(self, sample_image):
        """None 얼굴로 얼굴 교체 테스트"""
        # Arrange
        config = Config()
        
        with patch('src.services.inswapper_detector.model_zoo.get_model') as mock_get_model:
            mock_swapper = Mock()
            mock_get_model.return_value = mock_swapper
            
            detector = InswapperDetector(config)
            
            # Act & Assert
            with pytest.raises(ValueError, match="Face objects cannot be None"):
                detector.swap_face(sample_image, sample_image, None, None)
    
    def test_swap_face_with_none_bbox(self, sample_image):
        """None bbox로 얼굴 교체 테스트"""
        # Arrange
        config = Config()
        mock_face = Mock()
        mock_face.bbox = None
        
        with patch('src.services.inswapper_detector.model_zoo.get_model') as mock_get_model:
            mock_swapper = Mock()
            mock_get_model.return_value = mock_swapper
            
            detector = InswapperDetector(config)
            
            # Act & Assert
            with pytest.raises(ValueError, match="Face bbox cannot be None"):
                detector.swap_face(sample_image, sample_image, mock_face, mock_face)
    
    def test_swap_face_model_not_initialized(self, sample_image, mock_face):
        """모델이 초기화되지 않은 상태에서 얼굴 교체 테스트"""
        # Arrange
        config = Config()
        detector = InswapperDetector(config)
        detector._swapper = None
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="Model not initialized"):
            detector.swap_face(sample_image, sample_image, mock_face, mock_face)
    
    def test_swap_face_model_error(self, sample_image, mock_face):
        """모델 오류 발생 시 테스트"""
        # Arrange
        config = Config()
        
        with patch('src.services.inswapper_detector.model_zoo.get_model') as mock_get_model:
            mock_swapper = Mock()
            mock_swapper.get.side_effect = Exception("Model error")
            mock_get_model.return_value = mock_swapper
            
            detector = InswapperDetector(config)
            
            # Act & Assert
            with pytest.raises(RuntimeError, match="Face swap failed"):
                detector.swap_face(sample_image, sample_image, mock_face, mock_face)
    
    def test_swap_faces_in_image_with_valid_inputs(self, sample_image, mock_face):
        """유효한 입력으로 여러 얼굴 교체 테스트"""
        # Arrange
        config = Config()
        source_image = sample_image.copy()
        target_image = sample_image.copy()
        source_faces = [mock_face, mock_face]
        target_faces = [mock_face, mock_face]
        
        with patch('src.services.inswapper_detector.model_zoo.get_model') as mock_get_model:
            mock_swapper = Mock()
            mock_swapper.get.return_value = target_image.copy()
            mock_get_model.return_value = mock_swapper
            
            detector = InswapperDetector(config)
            
            # Act
            result = detector.swap_faces_in_image(source_image, target_image, source_faces, target_faces)
            
            # Assert
            assert isinstance(result, np.ndarray)
            assert result.shape == target_image.shape
            assert mock_swapper.get.call_count == 2
    
    def test_swap_faces_in_image_with_empty_faces(self, sample_image):
        """빈 얼굴 리스트로 여러 얼굴 교체 테스트"""
        # Arrange
        config = Config()
        source_image = sample_image.copy()
        target_image = sample_image.copy()
        
        with patch('src.services.inswapper_detector.model_zoo.get_model') as mock_get_model:
            mock_swapper = Mock()
            mock_get_model.return_value = mock_swapper
            
            detector = InswapperDetector(config)
            
            # Act
            result = detector.swap_faces_in_image(source_image, target_image, [], [])
            
            # Assert
            assert isinstance(result, np.ndarray)
            assert np.array_equal(result, target_image)
    
    def test_swap_faces_in_image_with_mismatched_count(self, sample_image, mock_face):
        """얼굴 수가 맞지 않는 경우 테스트"""
        # Arrange
        config = Config()
        source_image = sample_image.copy()
        target_image = sample_image.copy()
        source_faces = [mock_face, mock_face, mock_face]  # 3개
        target_faces = [mock_face, mock_face]  # 2개
        
        with patch('src.services.inswapper_detector.model_zoo.get_model') as mock_get_model:
            mock_swapper = Mock()
            mock_swapper.get.return_value = target_image.copy()
            mock_get_model.return_value = mock_swapper
            
            detector = InswapperDetector(config)
            
            # Act
            result = detector.swap_faces_in_image(source_image, target_image, source_faces, target_faces)
            
            # Assert
            assert isinstance(result, np.ndarray)
            assert result.shape == target_image.shape
            assert mock_swapper.get.call_count == 2  # 더 적은 수만큼만 교체
    
    def test_is_initialized_true(self):
        """모델이 초기화된 상태 테스트"""
        # Arrange
        config = Config()
        
        with patch('src.services.inswapper_detector.model_zoo.get_model') as mock_get_model:
            mock_swapper = Mock()
            mock_get_model.return_value = mock_swapper
            
            detector = InswapperDetector(config)
            
            # Act
            result = detector.is_initialized()
            
            # Assert
            assert result is True
    
    def test_is_initialized_false(self):
        """모델이 초기화되지 않은 상태 테스트"""
        # Arrange
        config = Config()
        detector = InswapperDetector(config)
        detector._swapper = None
        
        # Act
        result = detector.is_initialized()
        
        # Assert
        assert result is False
    
    def test_get_model_info(self):
        """모델 정보 반환 테스트"""
        # Arrange
        config = Config()
        
        with patch('src.services.inswapper_detector.model_zoo.get_model') as mock_get_model:
            mock_swapper = Mock()
            mock_get_model.return_value = mock_swapper
            
            detector = InswapperDetector(config)
            
            # Act
            info = detector.get_model_info()
            
            # Assert
            assert isinstance(info, dict)
            assert "model_name" in info
            assert "model_path" in info
            assert "use_gpu" in info
            assert "initialized" in info
            assert "device" in info
            assert info["model_name"] == "inswapper_128"
            assert info["initialized"] is True
    
    def test_get_model_info_not_initialized(self):
        """초기화되지 않은 모델 정보 반환 테스트"""
        # Arrange
        config = Config()
        detector = InswapperDetector(config)
        detector._swapper = None
        
        # Act
        info = detector.get_model_info()
        
        # Assert
        assert isinstance(info, dict)
        assert info["initialized"] is False