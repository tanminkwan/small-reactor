"""
얼굴 교체 인터페이스

Interface Segregation Principle (ISP)에 따라
얼굴 교체 기능만을 위한 작고 구체적인 인터페이스
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from .face_detector import Face


class IFaceSwapper(ABC):
    """얼굴 교체 인터페이스"""
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
