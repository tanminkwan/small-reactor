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
from src.services.codeformer_enhancer import CodeFormerEnhancer
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


class TestCodeFormerIntegration:
    """CodeFormer 통합 테스트"""
    
    def test_codeformer_enhancer_integration(self, sample_image):
        """CodeFormerEnhancer 통합 테스트"""
        # Arrange
        config = Config()
        test_image = sample_image.copy()
        
        # Mock 설정
        with patch('codeformer.basicsr.utils.registry.ARCH_REGISTRY') as mock_arch_registry, \
             patch('codeformer.facelib.utils.face_restoration_helper.FaceRestoreHelper') as mock_face_helper, \
             patch('torch.load') as mock_torch_load, \
             patch('torch.cuda.is_available', return_value=False):
            
            # Mock 모델 설정
            mock_model = Mock()
            mock_model.eval.return_value = None
            mock_model.to.return_value = mock_model
            mock_model.load_state_dict.return_value = None
            # 모델 호출 시 튜플 반환 (output, _) 형태로 설정
            mock_model.return_value = (Mock(), None)  # (output, _) 형태
            mock_arch_registry.get.return_value.return_value = mock_model
            
            # Mock 체크포인트 설정
            mock_checkpoint = {'params_ema': {}}
            mock_torch_load.return_value = mock_checkpoint
            
            # Mock Face Helper 설정
            mock_helper_instance = Mock()
            mock_helper_instance.clean_all.return_value = None
            mock_helper_instance.read_image.return_value = None
            mock_helper_instance.get_face_landmarks_5.return_value = None
            mock_helper_instance.align_warp_face.return_value = None
            mock_helper_instance.cropped_faces = [test_image[50:150, 50:150]]  # Mock cropped face
            mock_helper_instance.add_restored_face.return_value = None
            mock_helper_instance.get_inverse_affine.return_value = None
            mock_helper_instance.paste_faces_to_input_image.return_value = test_image.copy()
            mock_face_helper.return_value = mock_helper_instance
            
            # 서비스 초기화
            enhancer = CodeFormerEnhancer(config)
            
            # Act
            enhanced_image = enhancer.enhance_image(test_image, enhancement_strength=0.5)
            
            # Assert
            assert isinstance(enhanced_image, np.ndarray)
            assert enhanced_image.shape == test_image.shape
            assert enhancer.is_initialized()
            
            # 모델 정보 확인
            model_info = enhancer.get_model_info()
            assert model_info['model_name'] == 'codeformer'
            assert 'model_path' in model_info
            assert 'use_gpu' in model_info
            assert 'device' in model_info
            assert model_info['initialized'] is True
    
    def test_codeformer_face_regions_enhancement(self, sample_image):
        """CodeFormer 얼굴 영역 복원 테스트"""
        # Arrange
        config = Config()
        test_image = sample_image.copy()
        face_regions = [(50, 50, 150, 150), (200, 200, 300, 300)]  # 두 개의 얼굴 영역
        
        # Mock 설정
        with patch('codeformer.basicsr.utils.registry.ARCH_REGISTRY') as mock_arch_registry, \
             patch('codeformer.facelib.utils.face_restoration_helper.FaceRestoreHelper') as mock_face_helper, \
             patch('torch.load') as mock_torch_load, \
             patch('torch.cuda.is_available', return_value=False):
            
            # Mock 모델 설정
            mock_model = Mock()
            mock_model.eval.return_value = None
            mock_model.to.return_value = mock_model
            mock_model.load_state_dict.return_value = None
            # 모델 호출 시 튜플 반환 (output, _) 형태로 설정
            mock_model.return_value = (Mock(), None)  # (output, _) 형태
            mock_arch_registry.get.return_value.return_value = mock_model
            
            # Mock 체크포인트 설정
            mock_checkpoint = {'params_ema': {}}
            mock_torch_load.return_value = mock_checkpoint
            
            # Mock Face Helper 설정
            mock_helper_instance = Mock()
            mock_helper_instance.clean_all.return_value = None
            mock_helper_instance.read_image.return_value = None
            mock_helper_instance.get_face_landmarks_5.return_value = None
            mock_helper_instance.align_warp_face.return_value = None
            mock_helper_instance.cropped_faces = [test_image[50:150, 50:150]]  # Mock cropped face
            mock_helper_instance.add_restored_face.return_value = None
            mock_helper_instance.get_inverse_affine.return_value = None
            mock_helper_instance.paste_faces_to_input_image.return_value = test_image.copy()
            mock_face_helper.return_value = mock_helper_instance
            
            # 서비스 초기화
            enhancer = CodeFormerEnhancer(config)
            
            # Act
            enhanced_image = enhancer.enhance_face_regions(
                test_image, 
                face_regions, 
                enhancement_strength=0.5
            )
            
            # Assert
            assert isinstance(enhanced_image, np.ndarray)
            assert enhanced_image.shape == test_image.shape
    
    def test_codeformer_enhancer_validation(self, sample_image):
        """CodeFormerEnhancer 입력 검증 테스트"""
        # Arrange
        config = Config()
        
        # Mock 설정
        with patch('codeformer.basicsr.utils.registry.ARCH_REGISTRY') as mock_arch_registry, \
             patch('codeformer.facelib.utils.face_restoration_helper.FaceRestoreHelper') as mock_face_helper, \
             patch('torch.load') as mock_torch_load, \
             patch('torch.cuda.is_available', return_value=False):
            
            # Mock 모델 설정
            mock_model = Mock()
            mock_model.eval.return_value = None
            mock_model.to.return_value = mock_model
            mock_model.load_state_dict.return_value = None
            # 모델 호출 시 튜플 반환 (output, _) 형태로 설정
            mock_model.return_value = (Mock(), None)  # (output, _) 형태
            mock_arch_registry.get.return_value.return_value = mock_model
            
            # Mock 체크포인트 설정
            mock_checkpoint = {'params_ema': {}}
            mock_torch_load.return_value = mock_checkpoint
            
            # Mock Face Helper 설정
            mock_helper_instance = Mock()
            mock_face_helper.return_value = mock_helper_instance
            
            # 서비스 초기화
            enhancer = CodeFormerEnhancer(config)
            
            # Act & Assert
            # None 이미지 테스트
            with pytest.raises(ValueError, match="Image cannot be None"):
                enhancer.enhance_image(None, 0.5)
            
            # 빈 이미지 테스트
            empty_image = np.array([])
            with pytest.raises(ValueError, match="Image cannot be empty"):
                enhancer.enhance_image(empty_image, 0.5)
            
            # 잘못된 차원 이미지 테스트
            wrong_dim_image = np.random.rand(100, 100)  # 2D 이미지
            with pytest.raises(ValueError, match="Image must be 3D"):
                enhancer.enhance_image(wrong_dim_image, 0.5)
            
            # 잘못된 enhancement_strength 테스트
            with pytest.raises(ValueError, match="Enhancement strength must be between 0.0 and 1.0"):
                enhancer.enhance_image(sample_image, -0.1)
            
            with pytest.raises(ValueError, match="Enhancement strength must be between 0.0 and 1.0"):
                enhancer.enhance_image(sample_image, 1.1)
    
    def test_codeformer_enhancer_initialization_failure(self):
        """CodeFormerEnhancer 초기화 실패 테스트"""
        # Arrange
        config = Config()
        
        # Mock 설정 (초기화 실패)
        with patch('codeformer.basicsr.utils.registry.ARCH_REGISTRY') as mock_arch_registry:
            mock_arch_registry.get.side_effect = ImportError("CodeFormer not available")
            
            # Act & Assert
            with pytest.raises(RuntimeError, match="CodeFormer dependencies not available"):
                CodeFormerEnhancer(config)
    
    def test_codeformer_enhancer_model_not_initialized(self, sample_image):
        """CodeFormerEnhancer 모델 미초기화 상태 테스트"""
        # Arrange
        config = Config()
        
        # Mock 설정 (모델 초기화 실패)
        with patch('src.services.codeformer_enhancer.ARCH_REGISTRY') as mock_arch_registry, \
             patch('src.services.codeformer_enhancer.FaceRestoreHelper') as mock_face_helper, \
             patch('torch.load') as mock_torch_load, \
             patch('torch.cuda.is_available', return_value=False):
            
            # Mock 모델 설정
            mock_model = Mock()
            mock_model.eval.return_value = None
            mock_model.to.return_value = mock_model
            mock_model.load_state_dict.return_value = None
            # 모델 호출 시 튜플 반환 (output, _) 형태로 설정
            mock_model.return_value = (Mock(), None)  # (output, _) 형태
            mock_arch_registry.get.return_value.return_value = mock_model
            
            # Mock 체크포인트 설정
            mock_checkpoint = {'params_ema': {}}
            mock_torch_load.return_value = mock_checkpoint
            
            # Mock Face Helper 설정
            mock_helper_instance = Mock()
            mock_face_helper.return_value = mock_helper_instance
            
            # 서비스 초기화 (모델을 None으로 설정하여 초기화 실패 시뮬레이션)
            enhancer = CodeFormerEnhancer(config)
            enhancer._model = None  # 모델을 None으로 설정
            
            # Act & Assert
            with pytest.raises(RuntimeError, match="Model not initialized"):
                enhancer.enhance_image(sample_image, 0.5)
            
            assert not enhancer.is_initialized()
            
            model_info = enhancer.get_model_info()
            assert model_info['initialized'] is False


class TestFaceManagerCodeFormerIntegration:
    """FaceManager와 CodeFormerEnhancer 통합 테스트"""
    
    def test_face_manager_with_codeformer_integration(self, sample_image):
        """FaceManager에서 CodeFormerEnhancer 사용 통합 테스트"""
        # Arrange
        from gradio_face_manager import FaceManager
        
        # Mock 설정
        with patch('src.services.buffalo_detector.FaceAnalysis') as mock_face_analysis, \
             patch('src.services.codeformer_enhancer.ARCH_REGISTRY') as mock_arch_registry, \
             patch('src.services.codeformer_enhancer.FaceRestoreHelper') as mock_face_helper, \
             patch('torch.load') as mock_torch_load, \
             patch('torch.cuda.is_available', return_value=False):
            
            # Mock 얼굴 생성
            mock_face = Mock()
            mock_face.bbox = [50, 50, 150, 150]
            mock_face.kps = np.array([[100, 80], [120, 80], [110, 100], [100, 120], [120, 120]])
            mock_face.embedding = np.random.rand(512)
            mock_face.det_score = 0.95
            mock_face.age = 25
            mock_face.gender = 1
            
            # FaceDetector Mock 설정
            mock_face_analysis.return_value.get.return_value = [mock_face]
            
            # CodeFormer Mock 설정
            mock_model = Mock()
            mock_model.eval.return_value = None
            mock_model.to.return_value = mock_model
            mock_model.load_state_dict.return_value = None
            mock_arch_registry.get.return_value.return_value = mock_model
            
            mock_checkpoint = {'params_ema': {}}
            mock_torch_load.return_value = mock_checkpoint
            
            mock_helper_instance = Mock()
            mock_helper_instance.clean_all.return_value = None
            mock_helper_instance.read_image.return_value = None
            mock_helper_instance.get_face_landmarks_5.return_value = None
            mock_helper_instance.align_warp_face.return_value = None
            mock_helper_instance.cropped_faces = [sample_image[50:150, 50:150]]
            mock_helper_instance.add_restored_face.return_value = None
            mock_helper_instance.get_inverse_affine.return_value = None
            mock_helper_instance.paste_faces_to_input_image.return_value = sample_image.copy()
            mock_face_helper.return_value = mock_helper_instance
            
            # FaceManager 초기화
            face_manager = FaceManager()
            
            # Act
            success, message, enhanced_image = face_manager.enhance_faces_with_codeformer(
                sample_image, "1"
            )
            
            # Assert
            assert success is True
            assert "CodeFormer 복원 완료" in message
            assert "복원된 얼굴: 1개" in message
            assert isinstance(enhanced_image, np.ndarray)
            assert enhanced_image.shape == sample_image.shape
    
    def test_face_manager_codeformer_no_faces(self, sample_image):
        """FaceManager CodeFormer - 얼굴이 없는 경우 테스트"""
        # Arrange
        from gradio_face_manager import FaceManager
        
        # Mock 설정 (얼굴 없음)
        with patch('src.services.buffalo_detector.FaceAnalysis') as mock_face_analysis, \
             patch('src.services.codeformer_enhancer.ARCH_REGISTRY') as mock_arch_registry, \
             patch('src.services.codeformer_enhancer.FaceRestoreHelper') as mock_face_helper, \
             patch('torch.load') as mock_torch_load, \
             patch('torch.cuda.is_available', return_value=False):
            
            # FaceDetector Mock 설정 (얼굴 없음)
            mock_face_analysis.return_value.get.return_value = []
            
            # CodeFormer Mock 설정 (초기화는 성공하지만 사용되지 않음)
            mock_model = Mock()
            mock_arch_registry.get.return_value.return_value = mock_model
            mock_checkpoint = {'params_ema': {}}
            mock_torch_load.return_value = mock_checkpoint
            mock_face_helper.return_value = Mock()
            
            # FaceManager 초기화
            face_manager = FaceManager()
            
            # Act
            success, message, enhanced_image = face_manager.enhance_faces_with_codeformer(
                sample_image, "1"
            )
            
            # Assert
            assert success is False
            assert "이미지에서 얼굴을 찾을 수 없습니다" in message
            assert enhanced_image is None
    
    def test_face_manager_codeformer_invalid_indices(self, sample_image):
        """FaceManager CodeFormer - 잘못된 인덱스 테스트"""
        # Arrange
        from gradio_face_manager import FaceManager
        
        # Mock 설정
        with patch('src.services.buffalo_detector.FaceAnalysis') as mock_face_analysis, \
             patch('src.services.codeformer_enhancer.ARCH_REGISTRY') as mock_arch_registry, \
             patch('src.services.codeformer_enhancer.FaceRestoreHelper') as mock_face_helper, \
             patch('torch.load') as mock_torch_load, \
             patch('torch.cuda.is_available', return_value=False):
            
            # Mock 얼굴 생성
            mock_face = Mock()
            mock_face.bbox = [50, 50, 150, 150]
            mock_face.kps = np.array([[100, 80], [120, 80], [110, 100], [100, 120], [120, 120]])
            mock_face.embedding = np.random.rand(512)
            mock_face.det_score = 0.95
            mock_face.age = 25
            mock_face.gender = 1
            
            # FaceDetector Mock 설정
            mock_face_analysis.return_value.get.return_value = [mock_face]
            
            # CodeFormer Mock 설정
            mock_model = Mock()
            mock_arch_registry.get.return_value.return_value = mock_model
            mock_checkpoint = {'params_ema': {}}
            mock_torch_load.return_value = mock_checkpoint
            mock_face_helper.return_value = Mock()
            
            # FaceManager 초기화
            face_manager = FaceManager()
            
            # Act - 잘못된 인덱스 (2번째 얼굴을 요청하지만 1개만 있음)
            success, message, enhanced_image = face_manager.enhance_faces_with_codeformer(
                sample_image, "2"
            )
            
            # Assert
            assert success is False
            assert "유효한 얼굴 인덱스가 없습니다" in message
            assert enhanced_image is None
    
    def test_face_manager_codeformer_multiple_faces(self, sample_image):
        """FaceManager CodeFormer - 여러 얼굴 복원 테스트"""
        # Arrange
        from gradio_face_manager import FaceManager
        
        # Mock 설정
        with patch('src.services.buffalo_detector.FaceAnalysis') as mock_face_analysis, \
             patch('src.services.codeformer_enhancer.ARCH_REGISTRY') as mock_arch_registry, \
             patch('src.services.codeformer_enhancer.FaceRestoreHelper') as mock_face_helper, \
             patch('torch.load') as mock_torch_load, \
             patch('torch.cuda.is_available', return_value=False):
            
            # 여러 Mock 얼굴 생성
            mock_faces = []
            for i in range(3):
                mock_face = Mock()
                mock_face.bbox = [50 + i*50, 50 + i*50, 150 + i*50, 150 + i*50]
                mock_face.kps = np.array([[100 + i*50, 80 + i*50], [120 + i*50, 80 + i*50], 
                                        [110 + i*50, 100 + i*50], [100 + i*50, 120 + i*50], 
                                        [120 + i*50, 120 + i*50]])
                mock_face.embedding = np.random.rand(512)
                mock_face.det_score = 0.95
                mock_face.age = 25 + i
                mock_face.gender = i % 2
                mock_faces.append(mock_face)
            
            # FaceDetector Mock 설정
            mock_face_analysis.return_value.get.return_value = mock_faces
            
            # CodeFormer Mock 설정
            mock_model = Mock()
            mock_arch_registry.get.return_value.return_value = mock_model
            mock_checkpoint = {'params_ema': {}}
            mock_torch_load.return_value = mock_checkpoint
            mock_helper_instance = Mock()
            mock_helper_instance.clean_all.return_value = None
            mock_helper_instance.read_image.return_value = None
            mock_helper_instance.get_face_landmarks_5.return_value = None
            mock_helper_instance.align_warp_face.return_value = None
            mock_helper_instance.cropped_faces = [sample_image[50:150, 50:150]]
            mock_helper_instance.add_restored_face.return_value = None
            mock_helper_instance.get_inverse_affine.return_value = None
            mock_helper_instance.paste_faces_to_input_image.return_value = sample_image.copy()
            mock_face_helper.return_value = mock_helper_instance
            
            # FaceManager 초기화
            face_manager = FaceManager()
            
            # Act - 모든 얼굴 복원
            success, message, enhanced_image = face_manager.enhance_faces_with_codeformer(
                sample_image, ""  # 빈 문자열 = 모든 얼굴
            )
            
            # Assert
            assert success is True
            assert "CodeFormer 복원 완료" in message
            assert "복원된 얼굴: 3개" in message
            assert isinstance(enhanced_image, np.ndarray)
            assert enhanced_image.shape == sample_image.shape
            
            # Act - 특정 얼굴들만 복원
            success, message, enhanced_image = face_manager.enhance_faces_with_codeformer(
                sample_image, "1,3"  # 1번째와 3번째 얼굴
            )
            
            # Assert
            assert success is True
            assert "CodeFormer 복원 완료" in message
            assert "복원된 얼굴: 2개" in message
            assert isinstance(enhanced_image, np.ndarray)
            assert enhanced_image.shape == sample_image.shape