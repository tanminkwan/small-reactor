"""
통합 테스트

FaceDetector와 FaceSwapper의 통합 동작을 테스트
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.services.buffalo_detector import BuffaloDetector
from src.services.inswapper_detector import InswapperDetector
from src.interfaces.face_detector import Face
from src.utils.config import Config


class TestFaceSwapIntegration:
    """얼굴 탐지 및 교체 통합 테스트"""
    
    def test_face_detection_and_swap_integration(self, sample_image, mock_face):
        """얼굴 탐지와 교체의 통합 동작 테스트"""
        # Arrange
        config = Config()
        source_image = sample_image.copy()
        target_image = sample_image.copy()
        
        # Mock 설정
        with patch('src.services.buffalo_detector.FaceAnalysis') as mock_face_analysis, \
             patch('src.services.inswapper_detector.model_zoo.get_model') as mock_get_model:
            
            # FaceDetector Mock 설정
            mock_face_analysis.return_value.get.return_value = [mock_face]
            
            # InswapperDetector Mock 설정
            mock_swapper = Mock()
            mock_swapper.get.return_value = target_image.copy()
            mock_get_model.return_value = mock_swapper
            
            # 서비스 초기화
            detector = BuffaloDetector(config)
            swapper = InswapperDetector(config)
            
            # Act
            # 1. 얼굴 탐지
            source_faces = detector.detect_faces(source_image)
            target_faces = detector.detect_faces(target_image)
            
            # 2. 얼굴 교체
            if source_faces and target_faces:
                result = swapper.swap_face(source_image, target_image, source_faces[0], target_faces[0])
            
            # Assert
            assert len(source_faces) == 1
            assert len(target_faces) == 1
            assert isinstance(source_faces[0], Face)
            assert isinstance(target_faces[0], Face)
            assert isinstance(result, np.ndarray)
            assert result.shape == target_image.shape
    
    def test_multiple_faces_detection_and_swap(self, sample_image):
        """여러 얼굴 탐지 및 교체 테스트"""
        # Arrange
        config = Config()
        source_image = sample_image.copy()
        target_image = sample_image.copy()
        
        # 여러 얼굴 Mock 생성
        mock_faces = []
        for i in range(3):
            mock_face = Mock()
            mock_face.bbox = [100 + i*50, 100 + i*50, 200 + i*50, 200 + i*50]
            mock_face.kps = np.array([[150 + i*50, 120 + i*50], [180 + i*50, 120 + i*50], 
                                    [165 + i*50, 140 + i*50], [150 + i*50, 160 + i*50], 
                                    [180 + i*50, 160 + i*50]])
            mock_face.embedding = np.random.rand(512)
            mock_face.det_score = 0.95
            mock_face.age = 25 + i
            mock_face.gender = i % 2
            mock_faces.append(mock_face)
        
        with patch('src.services.buffalo_detector.FaceAnalysis') as mock_face_analysis, \
             patch('src.services.inswapper_detector.model_zoo.get_model') as mock_get_model:
            
            # FaceDetector Mock 설정
            mock_face_analysis.return_value.get.return_value = mock_faces
            
            # InswapperDetector Mock 설정
            mock_swapper = Mock()
            mock_swapper.get.return_value = target_image.copy()
            mock_get_model.return_value = mock_swapper
            
            # 서비스 초기화
            detector = BuffaloDetector(config)
            swapper = InswapperDetector(config)
            
            # Act
            # 1. 얼굴 탐지
            source_faces = detector.detect_faces(source_image)
            target_faces = detector.detect_faces(target_image)
            
            # 2. 여러 얼굴 교체
            result = swapper.swap_faces_in_image(source_image, target_image, source_faces, target_faces)
            
            # Assert
            assert len(source_faces) == 3
            assert len(target_faces) == 3
            assert isinstance(result, np.ndarray)
            assert result.shape == target_image.shape
    
    def test_face_detection_with_different_scores(self, sample_image):
        """다양한 탐지 점수로 얼굴 탐지 및 교체 테스트"""
        # Arrange
        config = Config()
        source_image = sample_image.copy()
        target_image = sample_image.copy()
        
        # 다양한 점수의 얼굴 Mock 생성
        mock_faces = []
        scores = [0.3, 0.6, 0.9]  # 낮은 점수, 중간 점수, 높은 점수
        
        for i, score in enumerate(scores):
            mock_face = Mock()
            mock_face.bbox = [100 + i*50, 100 + i*50, 200 + i*50, 200 + i*50]
            mock_face.kps = np.array([[150 + i*50, 120 + i*50], [180 + i*50, 120 + i*50], 
                                    [165 + i*50, 140 + i*50], [150 + i*50, 160 + i*50], 
                                    [180 + i*50, 160 + i*50]])
            mock_face.embedding = np.random.rand(512)
            mock_face.det_score = score
            mock_face.age = 25 + i
            mock_face.gender = i % 2
            mock_faces.append(mock_face)
        
        with patch('src.services.buffalo_detector.FaceAnalysis') as mock_face_analysis, \
             patch('src.services.inswapper_detector.model_zoo.get_model') as mock_get_model:
            
            # FaceDetector Mock 설정
            mock_face_analysis.return_value.get.return_value = mock_faces
            
            # InswapperDetector Mock 설정
            mock_swapper = Mock()
            mock_swapper.get.return_value = target_image.copy()
            mock_get_model.return_value = mock_swapper
            
            # 서비스 초기화
            detector = BuffaloDetector(config)
            swapper = InswapperDetector(config)
            
            # Act
            # 1. 얼굴 탐지 (낮은 점수 얼굴은 필터링됨)
            source_faces = detector.detect_faces(source_image)
            target_faces = detector.detect_faces(target_image)
            
            # 2. 얼굴 교체
            if source_faces and target_faces:
                result = swapper.swap_faces_in_image(source_image, target_image, source_faces, target_faces)
            
            # Assert
            # min_det_score가 0.5이므로 0.6과 0.9 점수 얼굴만 탐지됨
            assert len(source_faces) == 2
            assert len(target_faces) == 2
            assert all(face.det_score >= 0.5 for face in source_faces)
            assert all(face.det_score >= 0.5 for face in target_faces)
            assert isinstance(result, np.ndarray)
    
    def test_face_detection_with_no_faces(self, sample_image):
        """얼굴이 없는 이미지 테스트"""
        # Arrange
        config = Config()
        source_image = sample_image.copy()
        target_image = sample_image.copy()
        
        with patch('src.services.buffalo_detector.FaceAnalysis') as mock_face_analysis, \
             patch('src.services.inswapper_detector.model_zoo.get_model') as mock_get_model:
            
            # FaceDetector Mock 설정 (얼굴 없음)
            mock_face_analysis.return_value.get.return_value = []
            
            # InswapperDetector Mock 설정
            mock_swapper = Mock()
            mock_get_model.return_value = mock_swapper
            
            # 서비스 초기화
            detector = BuffaloDetector(config)
            swapper = InswapperDetector(config)
            
            # Act
            # 1. 얼굴 탐지
            source_faces = detector.detect_faces(source_image)
            target_faces = detector.detect_faces(target_image)
            
            # 2. 얼굴 교체 (얼굴이 없으므로 원본 이미지 반환)
            result = swapper.swap_faces_in_image(source_image, target_image, source_faces, target_faces)
            
            # Assert
            assert len(source_faces) == 0
            assert len(target_faces) == 0
            assert isinstance(result, np.ndarray)
            assert np.array_equal(result, target_image)
    
    def test_face_detection_with_different_image_sizes(self):
        """다양한 이미지 크기로 얼굴 탐지 및 교체 테스트"""
        # Arrange
        config = Config()
        
        # 다양한 크기의 이미지 생성
        image_sizes = [(100, 100), (200, 200), (500, 500)]
        
        for width, height in image_sizes:
            source_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            target_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # Mock 얼굴 생성
            mock_face = Mock()
            mock_face.bbox = [width//4, height//4, width*3//4, height*3//4]
            mock_face.kps = np.array([[width//2, height//2], [width//2 + 20, height//2], 
                                    [width//2, height//2 + 20], [width//2 - 20, height//2 + 20], 
                                    [width//2 + 20, height//2 + 20]])
            mock_face.embedding = np.random.rand(512)
            mock_face.det_score = 0.95
            mock_face.age = 25
            mock_face.gender = 1
            
            with patch('src.services.buffalo_detector.FaceAnalysis') as mock_face_analysis, \
                 patch('src.services.inswapper_detector.model_zoo.get_model') as mock_get_model:
                
                # FaceDetector Mock 설정
                mock_face_analysis.return_value.get.return_value = [mock_face]
                
                # InswapperDetector Mock 설정
                mock_swapper = Mock()
                mock_swapper.get.return_value = target_image.copy()
                mock_get_model.return_value = mock_swapper
                
                # 서비스 초기화
                detector = BuffaloDetector(config)
                swapper = InswapperDetector(config)
                
                # Act
                # 1. 얼굴 탐지
                source_faces = detector.detect_faces(source_image)
                target_faces = detector.detect_faces(target_image)
                
                # 2. 얼굴 교체
                if source_faces and target_faces:
                    result = swapper.swap_face(source_image, target_image, source_faces[0], target_faces[0])
                
                # Assert
                assert len(source_faces) == 1
                assert len(target_faces) == 1
                assert isinstance(result, np.ndarray)
                assert result.shape == target_image.shape
    
    def test_face_detection_with_rgb_and_bgr_images(self):
        """RGB와 BGR 이미지로 얼굴 탐지 및 교체 테스트"""
        # Arrange
        config = Config()
        
        # RGB 이미지 생성 (빨간색이 파란색보다 높은 값)
        rgb_image = np.zeros((200, 200, 3), dtype=np.uint8)
        rgb_image[:, :, 0] = 200  # Red channel
        rgb_image[:, :, 1] = 100  # Green channel
        rgb_image[:, :, 2] = 50   # Blue channel
        
        # BGR 이미지 생성 (파란색이 빨간색보다 높은 값)
        bgr_image = np.zeros((200, 200, 3), dtype=np.uint8)
        bgr_image[:, :, 0] = 50   # Blue channel
        bgr_image[:, :, 1] = 100  # Green channel
        bgr_image[:, :, 2] = 200  # Red channel
        
        # Mock 얼굴 생성
        mock_face = Mock()
        mock_face.bbox = [50, 50, 150, 150]
        mock_face.kps = np.array([[100, 80], [120, 80], [110, 100], [100, 120], [120, 120]])
        mock_face.embedding = np.random.rand(512)
        mock_face.det_score = 0.95
        mock_face.age = 25
        mock_face.gender = 1
        
        with patch('src.services.buffalo_detector.FaceAnalysis') as mock_face_analysis, \
             patch('src.services.inswapper_detector.model_zoo.get_model') as mock_get_model:
            
            # FaceDetector Mock 설정
            mock_face_analysis.return_value.get.return_value = [mock_face]
            
            # InswapperDetector Mock 설정
            mock_swapper = Mock()
            mock_get_model.return_value = mock_swapper
            
            # 서비스 초기화
            detector = BuffaloDetector(config)
            swapper = InswapperDetector(config)
            
            # Act
            # RGB 이미지로 얼굴 탐지
            rgb_faces = detector.detect_faces(rgb_image)
            
            # BGR 이미지로 얼굴 탐지
            bgr_faces = detector.detect_faces(bgr_image)
            
            # Assert
            assert len(rgb_faces) == 1
            assert len(bgr_faces) == 1
            assert isinstance(rgb_faces[0], Face)
            assert isinstance(bgr_faces[0], Face)
    
    def test_face_swap_error_handling(self, sample_image, mock_face):
        """얼굴 교체 오류 처리 테스트"""
        # Arrange
        config = Config()
        source_image = sample_image.copy()
        target_image = sample_image.copy()
        
        with patch('src.services.buffalo_detector.FaceAnalysis') as mock_face_analysis, \
             patch('src.services.inswapper_detector.model_zoo.get_model') as mock_get_model:
            
            # FaceDetector Mock 설정
            mock_face_analysis.return_value.get.return_value = [mock_face]
            
            # InswapperDetector Mock 설정 (오류 발생)
            mock_swapper = Mock()
            mock_swapper.get.side_effect = Exception("Model error")
            mock_get_model.return_value = mock_swapper
            
            # 서비스 초기화
            detector = BuffaloDetector(config)
            swapper = InswapperDetector(config)
            
            # Act
            # 1. 얼굴 탐지
            source_faces = detector.detect_faces(source_image)
            target_faces = detector.detect_faces(target_image)
            
            # 2. 얼굴 교체 (오류 발생)
            with pytest.raises(RuntimeError, match="Face swap failed"):
                swapper.swap_face(source_image, target_image, source_faces[0], target_faces[0])
            
            # Assert
            assert len(source_faces) == 1
            assert len(target_faces) == 1
    
    def test_face_swap_with_partial_failure(self, sample_image):
        """일부 얼굴 교체 실패 시 테스트"""
        # Arrange
        config = Config()
        source_image = sample_image.copy()
        target_image = sample_image.copy()
        
        # 여러 얼굴 Mock 생성
        mock_faces = []
        for i in range(3):
            mock_face = Mock()
            mock_face.bbox = [100 + i*50, 100 + i*50, 200 + i*50, 200 + i*50]
            mock_face.kps = np.array([[150 + i*50, 120 + i*50], [180 + i*50, 120 + i*50], 
                                    [165 + i*50, 140 + i*50], [150 + i*50, 160 + i*50], 
                                    [180 + i*50, 160 + i*50]])
            mock_face.embedding = np.random.rand(512)
            mock_face.det_score = 0.95
            mock_face.age = 25 + i
            mock_face.gender = i % 2
            mock_faces.append(mock_face)
        
        with patch('src.services.buffalo_detector.FaceAnalysis') as mock_face_analysis, \
             patch('src.services.inswapper_detector.model_zoo.get_model') as mock_get_model:
            
            # FaceDetector Mock 설정
            mock_face_analysis.return_value.get.return_value = mock_faces
            
            # InswapperDetector Mock 설정 (일부 실패)
            mock_swapper = Mock()
            
            # 첫 번째 호출은 성공, 두 번째 호출은 실패, 세 번째 호출은 성공
            mock_swapper.get.side_effect = [
                target_image.copy(),  # 첫 번째 성공
                Exception("Partial failure"),  # 두 번째 실패
                target_image.copy()   # 세 번째 성공
            ]
            mock_get_model.return_value = mock_swapper
            
            # 서비스 초기화
            detector = BuffaloDetector(config)
            swapper = InswapperDetector(config)
            
            # Act
            # 1. 얼굴 탐지
            source_faces = detector.detect_faces(source_image)
            target_faces = detector.detect_faces(target_image)
            
            # 2. 여러 얼굴 교체 (일부 실패)
            result = swapper.swap_faces_in_image(source_image, target_image, source_faces, target_faces)
            
            # Assert
            assert len(source_faces) == 3
            assert len(target_faces) == 3
            assert isinstance(result, np.ndarray)
            assert result.shape == target_image.shape
            # 일부 실패해도 결과는 반환됨