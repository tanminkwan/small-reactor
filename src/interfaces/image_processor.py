"""
이미지 처리 워크플로우 인터페이스

Interface Segregation Principle (ISP)에 따라
전체 이미지 처리 워크플로우만을 위한 인터페이스
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np

from .face_detector import Face


class IImageProcessor(ABC):
    """이미지 처리 워크플로우 인터페이스"""
    
    @abstractmethod
    def process_image(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        source_face_index: Optional[int] = None,
        target_face_index: Optional[int] = None
    ) -> np.ndarray:
        """
        전체 얼굴 교체 워크플로우를 실행합니다.
        
        Args:
            source_image: 소스 이미지
            target_image: 타겟 이미지
            source_face_index: 사용할 소스 얼굴 인덱스 (None이면 첫 번째)
            target_face_index: 사용할 타겟 얼굴 인덱스 (None이면 첫 번째)
            
        Returns:
            처리된 이미지
            
        Raises:
            ValueError: 입력 이미지가 유효하지 않은 경우
            RuntimeError: 처리 과정에서 오류 발생 시
        """
        pass
    
    @abstractmethod
    def detect_faces_in_image(self, image: np.ndarray) -> List[Face]:
        """
        이미지에서 얼굴을 탐지합니다.
        
        Args:
            image: 입력 이미지
            
        Returns:
            탐지된 얼굴 리스트
        """
        pass
    
    @abstractmethod
    def get_processing_info(self) -> dict:
        """
        처리 정보를 반환합니다.
        
        Returns:
            처리 정보 딕셔너리
        """
        pass
