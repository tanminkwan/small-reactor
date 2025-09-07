"""
이미지 복구 인터페이스

Interface Segregation Principle (ISP)에 따라
이미지 복구 기능만을 위한 작고 구체적인 인터페이스
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class IImageEnhancer(ABC):
    """이미지 복구 인터페이스"""
    
    @abstractmethod
    def enhance_image(
        self, 
        image: np.ndarray,
        enhancement_factor: Optional[float] = None
    ) -> np.ndarray:
        """
        이미지의 품질을 향상시킵니다.
        
        Args:
            image: 입력 이미지
            enhancement_factor: 향상 정도 (0.0 ~ 1.0)
            
        Returns:
            향상된 이미지
            
        Raises:
            ValueError: 입력 이미지가 유효하지 않은 경우
            RuntimeError: 모델 추론 실패 시
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
