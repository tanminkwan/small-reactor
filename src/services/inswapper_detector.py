"""
Inswapper 얼굴 교체 서비스

Single Responsibility Principle (SRP)에 따라
얼굴 교체만을 담당하는 서비스
"""

import logging
from typing import List, Optional
import numpy as np
import cv2
from pathlib import Path

from src.interfaces.face_swapper import IFaceSwapper
from src.interfaces.face_detector import Face
from src.utils.config import Config


class InswapperDetector(IFaceSwapper):
    """Inswapper 모델을 사용한 얼굴 교체 서비스"""
    
    def __init__(self, config: Config):
        """
        Args:
            config: 설정 객체
        """
        self._config = config
        self._use_gpu = config.is_gpu_available()
        self._model_path = config.get_model_path("inswapper")
        self._swapper = None
        
        self._logger = logging.getLogger(__name__)
        
        # 모델 초기화
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Inswapper 모델을 초기화합니다."""
        try:
            from insightface import model_zoo
            
            # 모델 경로 확인
            if not Path(self._model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self._model_path}")
            
            # InsightFace model_zoo를 사용하여 모델 로드
            self._swapper = model_zoo.get_model(self._model_path)
            
            self._logger.info(f"Inswapper model initialized successfully. GPU: {self._use_gpu}")
            
        except ImportError:
            self._logger.error("InsightFace not installed. Please install with: pip install insightface")
            raise RuntimeError("InsightFace not available")
        except Exception as e:
            self._logger.error(f"Failed to initialize Inswapper model: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")
    
    def swap_face(
        self, 
        source_image: np.ndarray, 
        target_image: np.ndarray,
        source_face: Face,
        target_face: Face
    ) -> np.ndarray:
        """
        소스 얼굴을 타겟 얼굴로 교체합니다.
        
        Args:
            source_image: 소스 이미지
            target_image: 타겟 이미지
            source_face: 소스 얼굴 정보
            target_face: 타겟 얼굴 정보
            
        Returns:
            얼굴이 교체된 이미지
            
        Raises:
            ValueError: 입력 이미지나 얼굴 정보가 유효하지 않은 경우
            RuntimeError: 모델 추론 실패 시
        """
        # 입력 검증
        self._validate_inputs(source_image, target_image, source_face, target_face)
        
        if not self.is_initialized():
            raise RuntimeError("Model not initialized")
        
        try:
            # InsightFace의 swapper.get() 메서드 사용
            # swapper.get(target_image, target_face, source_face)
            result_image = self._swapper.get(target_image, target_face, source_face)
            
            self._logger.debug("Face swap completed successfully")
            return result_image
            
        except Exception as e:
            self._logger.error(f"Face swap failed: {e}")
            raise RuntimeError(f"Face swap failed: {e}")
    
    def swap_faces_in_image(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        source_faces: List[Face],
        target_faces: List[Face]
    ) -> np.ndarray:
        """
        이미지 내의 여러 얼굴을 교체합니다.
        
        Args:
            source_image: 소스 이미지
            target_image: 타겟 이미지
            source_faces: 소스 얼굴 리스트
            target_faces: 타겟 얼굴 리스트
            
        Returns:
            얼굴이 교체된 이미지
        """
        if not source_faces or not target_faces:
            self._logger.warning("No faces provided for swapping")
            return target_image.copy()
        
        if len(source_faces) != len(target_faces):
            self._logger.warning(f"Face count mismatch: source={len(source_faces)}, target={len(target_faces)}")
            # 더 적은 수만큼만 교체
            min_faces = min(len(source_faces), len(target_faces))
            source_faces = source_faces[:min_faces]
            target_faces = target_faces[:min_faces]
        
        result_image = target_image.copy()
        
        for source_face, target_face in zip(source_faces, target_faces):
            try:
                result_image = self.swap_face(source_image, result_image, source_face, target_face)
            except Exception as e:
                self._logger.error(f"Failed to swap face: {e}")
                continue
        
        return result_image
    
    def _validate_inputs(
        self, 
        source_image: np.ndarray, 
        target_image: np.ndarray,
        source_face: Face,
        target_face: Face
    ) -> None:
        """입력 검증"""
        if source_image is None or target_image is None:
            raise ValueError("Images cannot be None")
        
        if source_image.size == 0 or target_image.size == 0:
            raise ValueError("Images cannot be empty")
        
        if len(source_image.shape) != 3 or len(target_image.shape) != 3:
            raise ValueError("Images must be 3D (height, width, channels)")
        
        if source_face is None or target_face is None:
            raise ValueError("Face objects cannot be None")
        
        if source_face.bbox is None or target_face.bbox is None:
            raise ValueError("Face bbox cannot be None")
    
    def is_initialized(self) -> bool:
        """
        모델이 초기화되었는지 확인합니다.
        
        Returns:
            초기화 여부
        """
        return self._swapper is not None
    
    def get_model_info(self) -> dict:
        """
        모델 정보를 반환합니다.
        
        Returns:
            모델 정보 딕셔너리
        """
        return {
            "model_name": "inswapper_128",
            "model_path": self._model_path,
            "use_gpu": self._use_gpu,
            "initialized": self.is_initialized(),
            "device": self._config.get_device()
        }