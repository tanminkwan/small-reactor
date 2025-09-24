"""
이미지 처리 유틸리티

Single Responsibility Principle (SRP)에 따라
범용 이미지 처리 기능들을 제공하는 유틸리티
"""

import cv2
import numpy as np
from typing import Tuple


def match_and_blend_images(A: np.ndarray, A2: np.ndarray, border_thickness: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    템플릿 매칭을 사용하여 이미지 A에 A2를 결합하고 경계 마스크를 생성합니다.
    
    Args:
        A: 원본 이미지 (ndarray)
        A2: 붙일 이미지 (ndarray)
        border_thickness: 경계 테두리 굵기 (픽셀)
        
    Returns:
        Tuple[결합된 이미지, 경계 마스크]
        - 결합된 이미지: A2가 붙여진 결과 이미지
        - 경계 마스크: A2 영역 경계에서 border_thickness만큼의 영역 (255: 경계영역, 0: 나머지)
    """
    # 입력 이미지 복사 (원본 보존)
    result_image = A.copy()
    
    # 템플릿 매칭 실행
    match_result = cv2.matchTemplate(A, A2, cv2.TM_CCOEFF_NORMED)
    
    # 가장 잘 맞는 위치 찾기
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)
    
    # 좌상단, 우하단 좌표 계산
    top_left = max_loc
    h, w = A2.shape[:2]
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    # A2를 해당 영역에 붙이기
    result_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = A2
    
    # 경계 마스크 생성 - A2와 A의 경계를 중심으로 얇은 테두리만
    mask = np.zeros(A.shape[:2], dtype=np.uint8)
    
    # 방법 1: 단순한 도넛 모양 테두리 마스크 (기본)
    # 외부 사각형 (A2 영역 + border_thickness)
    outer_top_left = (max(0, top_left[0] - border_thickness), 
                      max(0, top_left[1] - border_thickness))
    outer_bottom_right = (min(A.shape[1], bottom_right[0] + border_thickness),
                          min(A.shape[0], bottom_right[1] + border_thickness))
    
    # 내부 사각형 (A2 영역 - border_thickness)  
    inner_top_left = (max(top_left[0], top_left[0] + border_thickness),
                      max(top_left[1], top_left[1] + border_thickness))
    inner_bottom_right = (min(bottom_right[0], bottom_right[0] - border_thickness),
                          min(bottom_right[1], bottom_right[1] - border_thickness))
    
    # 도넛 모양 마스크: 외부 - 내부 = 얇은 테두리
    cv2.rectangle(mask, outer_top_left, outer_bottom_right, 255, -1)
    
    # 내부 영역이 유효한 경우에만 제거 (테두리가 너무 두꺼우면 전체가 마스크가 됨)
    if (inner_bottom_right[0] > inner_top_left[0] and 
        inner_bottom_right[1] > inner_top_left[1]):
        cv2.rectangle(mask, inner_top_left, inner_bottom_right, 0, -1)
    
    return result_image, mask


def seamless_blend(src_img: np.ndarray, dst_img: np.ndarray, mask: np.ndarray, 
                  center: Tuple[int, int], blend_mode: str = "normal") -> np.ndarray:
    """
    두 이미지를 자연스럽게 블렌딩합니다.
    
    Args:
        src_img: 소스 이미지
        dst_img: 대상 이미지  
        mask: 블렌딩 마스크
        center: 블렌딩 중심점 (x, y)
        blend_mode: 블렌딩 모드 ("normal", "mixed")
        
    Returns:
        블렌딩된 이미지
    """
    try:
        if blend_mode == "mixed":
            return cv2.seamlessClone(src_img, dst_img, mask, center, cv2.MIXED_CLONE)
        else:
            return cv2.seamlessClone(src_img, dst_img, mask, center, cv2.NORMAL_CLONE)
    except Exception as e:
        print(f"Seamless blending 실패: {e}")
        # 실패 시 기본 알파 블렌딩으로 폴백
        return alpha_blend(src_img, dst_img, mask)


def alpha_blend(src_img: np.ndarray, dst_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    알파 블렌딩을 수행합니다.
    
    Args:
        src_img: 소스 이미지
        dst_img: 대상 이미지
        mask: 블렌딩 마스크 (0-255)
        
    Returns:
        블렌딩된 이미지
    """
    # 마스크를 0-1 범위로 정규화
    mask_normalized = mask.astype(np.float32) / 255.0
    
    # 3채널로 확장
    if len(mask_normalized.shape) == 2:
        mask_3d = np.repeat(mask_normalized[:, :, np.newaxis], 3, axis=2)
    else:
        mask_3d = mask_normalized
    
    # 알파 블렌딩
    blended = dst_img * (1 - mask_3d) + src_img * mask_3d
    
    return blended.astype(np.uint8)


def create_smooth_mask(mask: np.ndarray, blur_kernel: Tuple[int, int] = (15, 15)) -> np.ndarray:
    """
    매끄러운 마스크를 생성합니다.
    
    Args:
        mask: 원본 마스크
        blur_kernel: 가우시안 블러 커널 크기
        
    Returns:
        부드러워진 마스크
    """
    # 가우시안 블러로 경계 부드럽게 처리
    smooth_mask = cv2.GaussianBlur(mask, blur_kernel, 0)
    
    # 모폴로지 연산으로 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    smooth_mask = cv2.morphologyEx(smooth_mask, cv2.MORPH_CLOSE, kernel)
    
    return smooth_mask


if __name__ == "__main__":
    print("image_utils.py 모듈이 정상적으로 로드되었습니다.")
    print("템플릿 매칭, 이미지 블렌딩 등의 범용 이미지 처리 기능을 제공합니다.")
