"""
얼굴 탐지 인터페이스

Interface Segregation Principle (ISP)에 따라
얼굴 탐지 기능만을 위한 작고 구체적인 인터페이스
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np


class Face:
    """얼굴 정보를 담는 데이터 클래스"""
    
    def __init__(
        self,
        bbox: List[float],
        kps: np.ndarray,
        embedding: np.ndarray,
        det_score: float,
        age: Optional[int] = None,
        gender: Optional[int] = None,
        landmark_2d_106: Optional[np.ndarray] = None
    ):
        """
        Args:
            bbox: 바운딩 박스 [x1, y1, x2, y2]
            kps: 키포인트 좌표 (5개 점)
            embedding: 얼굴 임베딩 벡터
            det_score: 탐지 점수
            age: 나이 (선택사항)
            gender: 성별 (선택사항)
            landmark_2d_106: 106개 2D 랜드마크 (선택사항)
        """
        self.bbox = bbox
        self.kps = kps
        self.embedding = embedding
        self.det_score = det_score
        self.age = age
        self.gender = gender
        self.landmark_2d_106 = landmark_2d_106
    
    def __repr__(self) -> str:
        return f"Face(bbox={self.bbox}, det_score={self.det_score:.3f})"


class IFaceDetector(ABC):
    """얼굴 탐지 인터페이스"""
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def is_initialized(self) -> bool:
        """
        모델이 초기화되었는지 확인합니다.
        
        Returns:
            초기화 여부
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> dict:
        """
        모델 정보를 반환합니다.
        
        Returns:
            모델 정보 딕셔너리
        """
        pass
