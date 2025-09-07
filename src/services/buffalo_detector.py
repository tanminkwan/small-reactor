"""
Buffalo_L 얼굴 탐지 서비스

Single Responsibility Principle (SRP)에 따라
얼굴 탐지만을 담당하는 서비스
"""

import logging
from typing import List, Optional
import numpy as np
from pathlib import Path

from src.interfaces.face_detector import IFaceDetector, Face
from src.utils.config import Config


class BuffaloDetector(IFaceDetector):
    """Buffalo_L 모델을 사용한 얼굴 탐지 서비스"""
    
    def __init__(self, config: Config):
        """
        Args:
            config: 설정 객체
        """
        self._config = config
        self._use_gpu = config.is_gpu_available()
        self._face_analysis = None
        self._model_path = config.get_model_path("buffalo_l")
        self._min_det_score = config.get("min_det_score", 0.5)
        
        self._logger = logging.getLogger(__name__)
        
        # 모델 초기화
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Buffalo_L 모델을 초기화합니다."""
        try:
            from insightface.app import FaceAnalysis
            
            # 모델 경로 확인
            if not Path(self._model_path).exists():
                self._logger.warning(f"Model path does not exist: {self._model_path}")
                # 기본 모델 사용
                self._face_analysis = FaceAnalysis(name='buffalo_l')
            else:
                self._face_analysis = FaceAnalysis(name='buffalo_l', root=self._model_path)
            
            # GPU 설정
            ctx_id = 0 if self._use_gpu else -1
            det_size = (640, 640)
            
            self._face_analysis.prepare(ctx_id=ctx_id, det_size=det_size)
            
            self._logger.info(f"Buffalo_L model initialized successfully. GPU: {self._use_gpu}")
            
        except ImportError:
            self._logger.error("InsightFace not installed. Please install with: pip install insightface")
            raise RuntimeError("InsightFace not available")
        except Exception as e:
            self._logger.error(f"Failed to initialize Buffalo_L model: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")
    
    def detect_faces(self, image: np.ndarray) -> List[Face]:
        """
        이미지에서 얼굴을 탐지합니다.
        
        Args:
            image: 입력 이미지 (BGR 또는 RGB)
            
        Returns:
            탐지된 얼굴 리스트
            
        Raises:
            ValueError: 이미지가 유효하지 않은 경우
            RuntimeError: 모델 로딩 또는 추론 실패 시
        """
        # 입력 검증
        if image is None:
            raise ValueError("Invalid image: image cannot be None")
        
        if image.size == 0:
            raise ValueError("Empty image")
        
        if len(image.shape) != 3:
            raise ValueError("Image must be 3D (height, width, channels)")
        
        if not self.is_initialized():
            raise RuntimeError("Model not initialized")
        
        try:
            # InsightFace는 BGR 이미지를 기대함
            if image.shape[2] == 3:
                # RGB를 BGR로 변환 (필요한 경우)
                if self._is_rgb_image(image):
                    image = self._rgb_to_bgr(image)
            
            # 얼굴 탐지
            faces = self._face_analysis.get(image)
            
            # Face 객체로 변환 및 필터링
            converted_faces = []
            for face in faces:
                if face.det_score >= self._min_det_score:
                    converted_face = self._convert_face(face)
                    converted_faces.append(converted_face)
            
            # 좌에서 우로, 위에서 아래 순으로 정렬
            converted_faces = self._sort_faces_by_position(converted_faces)
            
            self._logger.debug(f"Detected {len(converted_faces)} faces")
            return converted_faces
            
        except Exception as e:
            self._logger.error(f"Face detection failed: {e}")
            raise RuntimeError(f"Face detection failed: {e}")
    
    def _convert_face(self, insight_face) -> Face:
        """
        InsightFace Face 객체를 Face 객체로 변환합니다.
        
        Args:
            insight_face: InsightFace Face 객체
            
        Returns:
            Face 객체
        """
        return Face(
            bbox=insight_face.bbox.tolist(),
            kps=insight_face.kps,
            embedding=insight_face.embedding,
            det_score=float(insight_face.det_score),
            age=getattr(insight_face, 'age', None),
            gender=getattr(insight_face, 'gender', None)
        )
    
    def _sort_faces_by_position(self, faces: List[Face]) -> List[Face]:
        """
        얼굴들을 좌에서 우로, x좌표가 동일하면 위에서 아래 순으로 정렬합니다.
        
        Args:
            faces: 얼굴 리스트
            
        Returns:
            정렬된 얼굴 리스트
        """
        def sort_key(face: Face) -> tuple:
            """
            정렬 키를 반환합니다.
            (x 좌표, y 좌표) 순으로 정렬하여 좌에서 우, 위에서 아래 순으로 정렬
            """
            bbox = face.bbox
            # bbox는 [x1, y1, x2, y2] 형식
            x1, y1, x2, y2 = bbox
            # 중심점 계산
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            return (center_x, center_y)
        
        return sorted(faces, key=sort_key)
    
    def _is_rgb_image(self, image: np.ndarray) -> bool:
        """
        이미지가 RGB 형식인지 확인합니다.
        
        Args:
            image: 입력 이미지
            
        Returns:
            RGB 형식 여부
        """
        # 간단한 휴리스틱: 빨간색 채널이 파란색 채널보다 높은 경우 RGB로 간주
        if image.shape[2] == 3:
            red_mean = np.mean(image[:, :, 0])
            blue_mean = np.mean(image[:, :, 2])
            return red_mean > blue_mean
        return False
    
    def _rgb_to_bgr(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        RGB 이미지를 BGR로 변환합니다.
        
        Args:
            rgb_image: RGB 이미지
            
        Returns:
            BGR 이미지
        """
        return rgb_image[:, :, ::-1]
    
    def is_initialized(self) -> bool:
        """
        모델이 초기화되었는지 확인합니다.
        
        Returns:
            초기화 여부
        """
        return self._face_analysis is not None
    
    def get_model_info(self) -> dict:
        """
        모델 정보를 반환합니다.
        
        Returns:
            모델 정보 딕셔너리
        """
        return {
            "model_name": "buffalo_l",
            "model_path": self._model_path,
            "use_gpu": self._use_gpu,
            "initialized": self.is_initialized(),
            "min_det_score": self._min_det_score,
            "device": self._config.get_device()
        }
    
    def set_min_det_score(self, score: float) -> None:
        """
        최소 탐지 점수를 설정합니다.
        
        Args:
            score: 최소 탐지 점수 (0.0 ~ 1.0)
        """
        if not (0.0 <= score <= 1.0):
            raise ValueError("Detection score must be between 0.0 and 1.0")
        
        self._min_det_score = score
        self._logger.info(f"Minimum detection score set to {score}")
    
    def get_min_det_score(self) -> float:
        """
        현재 최소 탐지 점수를 반환합니다.
        
        Returns:
            최소 탐지 점수
        """
        return self._min_det_score
