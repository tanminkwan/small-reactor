"""
인터페이스 패키지

SOLID 원칙의 인터페이스 분리 원칙(ISP)에 따라
각 기능별로 작고 구체적인 인터페이스를 정의
"""

from .face_detector import IFaceDetector, Face
from .face_swapper import IFaceSwapper
from .image_enhancer import IImageEnhancer
from .image_processor import IImageProcessor

__all__ = [
    "IFaceDetector",
    "Face", 
    "IFaceSwapper",
    "IImageEnhancer",
    "IImageProcessor"
]
