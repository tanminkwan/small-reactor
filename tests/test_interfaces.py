"""
인터페이스 단위 테스트

TDD 방식으로 작성된 인터페이스들의 테스트
"""

import pytest
import numpy as np
from abc import ABC
from unittest.mock import Mock

from src.interfaces.face_detector import IFaceDetector, Face
from src.interfaces.face_swapper import IFaceSwapper
from src.interfaces.image_enhancer import IImageEnhancer
from src.interfaces.image_processor import IImageProcessor


class TestFace:
    """Face 클래스 테스트"""
    
    def test_face_creation(self):
        """Face 객체 생성 테스트"""
        # Arrange
        bbox = [100, 100, 200, 200]
        kps = np.array([[150, 120], [180, 120], [165, 140], [150, 160], [180, 160]])
        embedding = np.random.rand(512)
        det_score = 0.95
        age = 25
        gender = 1
        
        # Act
        face = Face(bbox, kps, embedding, det_score, age, gender)
        
        # Assert
        assert face.bbox == bbox
        assert np.array_equal(face.kps, kps)
        assert np.array_equal(face.embedding, embedding)
        assert face.det_score == det_score
        assert face.age == age
        assert face.gender == gender
    
    def test_face_creation_without_optional_params(self):
        """선택적 매개변수 없이 Face 객체 생성 테스트"""
        # Arrange
        bbox = [100, 100, 200, 200]
        kps = np.array([[150, 120], [180, 120], [165, 140], [150, 160], [180, 160]])
        embedding = np.random.rand(512)
        det_score = 0.95
        
        # Act
        face = Face(bbox, kps, embedding, det_score)
        
        # Assert
        assert face.bbox == bbox
        assert np.array_equal(face.kps, kps)
        assert np.array_equal(face.embedding, embedding)
        assert face.det_score == det_score
        assert face.age is None
        assert face.gender is None
    
    def test_face_repr(self):
        """Face 객체 문자열 표현 테스트"""
        # Arrange
        bbox = [100, 100, 200, 200]
        kps = np.array([[150, 120], [180, 120], [165, 140], [150, 160], [180, 160]])
        embedding = np.random.rand(512)
        det_score = 0.95
        
        face = Face(bbox, kps, embedding, det_score)
        
        # Act
        repr_str = repr(face)
        
        # Assert
        assert "Face" in repr_str
        assert "bbox=" in repr_str
        assert "det_score=" in repr_str


class TestIFaceDetector:
    """IFaceDetector 인터페이스 테스트"""
    
    def test_interface_is_abstract(self):
        """인터페이스가 추상 클래스인지 테스트"""
        # Assert
        assert issubclass(IFaceDetector, ABC)
    
    def test_interface_methods(self):
        """인터페이스 메서드들이 정의되어 있는지 테스트"""
        # Assert
        assert hasattr(IFaceDetector, 'detect_faces')
        assert hasattr(IFaceDetector, 'is_initialized')
        assert hasattr(IFaceDetector, 'get_model_info')
    
    def test_interface_cannot_be_instantiated(self):
        """인터페이스를 직접 인스턴스화할 수 없는지 테스트"""
        # Act & Assert
        with pytest.raises(TypeError):
            IFaceDetector()
    
    def test_concrete_implementation_required(self):
        """구체적 구현이 필요한지 테스트"""
        # Arrange
        class IncompleteDetector(IFaceDetector):
            pass
        
        # Act & Assert
        with pytest.raises(TypeError):
            IncompleteDetector()


class TestIFaceSwapper:
    """IFaceSwapper 인터페이스 테스트"""
    
    def test_interface_is_abstract(self):
        """인터페이스가 추상 클래스인지 테스트"""
        # Assert
        assert issubclass(IFaceSwapper, ABC)
    
    def test_interface_methods(self):
        """인터페이스 메서드들이 정의되어 있는지 테스트"""
        # Assert
        assert hasattr(IFaceSwapper, 'swap_face')
        assert hasattr(IFaceSwapper, 'swap_faces_in_image')
        assert hasattr(IFaceSwapper, 'is_initialized')
        assert hasattr(IFaceSwapper, 'get_model_info')
    
    def test_interface_cannot_be_instantiated(self):
        """인터페이스를 직접 인스턴스화할 수 없는지 테스트"""
        # Act & Assert
        with pytest.raises(TypeError):
            IFaceSwapper()


class TestIImageEnhancer:
    """IImageEnhancer 인터페이스 테스트"""
    
    def test_interface_is_abstract(self):
        """인터페이스가 추상 클래스인지 테스트"""
        # Assert
        assert issubclass(IImageEnhancer, ABC)
    
    def test_interface_methods(self):
        """인터페이스 메서드들이 정의되어 있는지 테스트"""
        # Assert
        assert hasattr(IImageEnhancer, 'enhance_image')
        assert hasattr(IImageEnhancer, 'is_initialized')
        assert hasattr(IImageEnhancer, 'get_model_info')
    
    def test_interface_cannot_be_instantiated(self):
        """인터페이스를 직접 인스턴스화할 수 없는지 테스트"""
        # Act & Assert
        with pytest.raises(TypeError):
            IImageEnhancer()


class TestIImageProcessor:
    """IImageProcessor 인터페이스 테스트"""
    
    def test_interface_is_abstract(self):
        """인터페이스가 추상 클래스인지 테스트"""
        # Assert
        assert issubclass(IImageProcessor, ABC)
    
    def test_interface_methods(self):
        """인터페이스 메서드들이 정의되어 있는지 테스트"""
        # Assert
        assert hasattr(IImageProcessor, 'process_image')
        assert hasattr(IImageProcessor, 'detect_faces_in_image')
        assert hasattr(IImageProcessor, 'get_processing_info')
    
    def test_interface_cannot_be_instantiated(self):
        """인터페이스를 직접 인스턴스화할 수 없는지 테스트"""
        # Act & Assert
        with pytest.raises(TypeError):
            IImageProcessor()


class TestInterfaceIntegration:
    """인터페이스 통합 테스트"""
    
    def test_face_detector_interface_compliance(self):
        """FaceDetector 구현체가 인터페이스를 준수하는지 테스트"""
        # Arrange
        class MockFaceDetector(IFaceDetector):
            def detect_faces(self, image):
                return []
            
            def is_initialized(self):
                return True
            
            def get_model_info(self):
                return {"model": "mock"}
        
        # Act
        detector = MockFaceDetector()
        
        # Assert
        assert detector.is_initialized() is True
        assert detector.detect_faces(np.random.rand(100, 100, 3)) == []
        assert detector.get_model_info()["model"] == "mock"
    
    def test_face_swapper_interface_compliance(self):
        """FaceSwapper 구현체가 인터페이스를 준수하는지 테스트"""
        # Arrange
        class MockFaceSwapper(IFaceSwapper):
            def swap_face(self, source_image, target_image, source_face, target_face):
                return target_image
            
            def swap_faces_in_image(self, source_image, target_image, source_faces, target_faces):
                return target_image
            
            def is_initialized(self):
                return True
            
            def get_model_info(self):
                return {"model": "mock"}
        
        # Act
        swapper = MockFaceSwapper()
        mock_face = Mock()
        
        # Assert
        assert swapper.is_initialized() is True
        result = swapper.swap_face(
            np.random.rand(100, 100, 3),
            np.random.rand(100, 100, 3),
            mock_face,
            mock_face
        )
        assert result is not None
        assert swapper.get_model_info()["model"] == "mock"
    
    def test_image_enhancer_interface_compliance(self):
        """ImageEnhancer 구현체가 인터페이스를 준수하는지 테스트"""
        # Arrange
        class MockImageEnhancer(IImageEnhancer):
            def enhance_image(self, image, enhancement_factor=None):
                return image
            
            def is_initialized(self):
                return True
            
            def get_model_info(self):
                return {"model": "mock"}
        
        # Act
        enhancer = MockImageEnhancer()
        
        # Assert
        assert enhancer.is_initialized() is True
        result = enhancer.enhance_image(np.random.rand(100, 100, 3))
        assert result is not None
        assert enhancer.get_model_info()["model"] == "mock"
    
    def test_image_processor_interface_compliance(self):
        """ImageProcessor 구현체가 인터페이스를 준수하는지 테스트"""
        # Arrange
        class MockImageProcessor(IImageProcessor):
            def process_image(self, source_image, target_image, source_face_index=None, target_face_index=None):
                return target_image
            
            def detect_faces_in_image(self, image):
                return []
            
            def get_processing_info(self):
                return {"processor": "mock"}
        
        # Act
        processor = MockImageProcessor()
        
        # Assert
        result = processor.process_image(
            np.random.rand(100, 100, 3),
            np.random.rand(100, 100, 3)
        )
        assert result is not None
        assert processor.detect_faces_in_image(np.random.rand(100, 100, 3)) == []
        assert processor.get_processing_info()["processor"] == "mock"
