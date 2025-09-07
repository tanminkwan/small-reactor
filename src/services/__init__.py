"""
서비스 패키지

SOLID 원칙의 단일 책임 원칙(SRP)에 따라
각 서비스는 하나의 명확한 책임만을 담당
"""

from .buffalo_detector import BuffaloDetector
from .inswapper_detector import InswapperDetector

__all__ = [
    "BuffaloDetector",
    "InswapperDetector"
]
