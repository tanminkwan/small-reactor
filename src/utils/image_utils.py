"""
이미지 처리 유틸리티

Single Responsibility Principle (SRP)에 따라
범용 이미지 처리 기능들을 제공하는 유틸리티
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image


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


def load_sam_model(
    model_path: str = "./sam_models",
    model_type: str = "vit_h"
):
    """
    SAM 모델을 로드하여 Predictor를 반환합니다.
    
    Args:
        model_path: SAM 모델 파일들이 있는 경로
        model_type: 사용할 모델 타입 (vit_h, vit_l, vit_b)
    
    Returns:
        SamPredictor 객체 또는 None (실패시)
    """
    try:
        from segment_anything import SamPredictor, sam_model_registry
        
        # 모델 파일 검색 (우선순위: 설정된 모델 > H > L > B)
        model_files = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth", 
            "vit_b": "sam_vit_b_01ec64.pth"
        }
        
        model_names = {
            "vit_h": "ViT-H (최고 품질)",
            "vit_l": "ViT-L (중간 품질)",
            "vit_b": "ViT-B (빠른 처리)"
        }
        
        # 검색할 경로들과 우선순위
        search_paths = [
            Path(model_path),  # 설정된 경로
            Path("."),         # 현재 폴더
            Path("./models"),  # models 폴더
            Path("./sam_models"),  # sam_models 폴더
            Path(".."),        # 상위 폴더
        ]
        
        # 우선순위: 설정된 모델 타입 > vit_h > vit_l > vit_b
        if model_type in model_files:
            model_priority = [model_type, "vit_h", "vit_l", "vit_b"]
        else:
            model_priority = ["vit_h", "vit_l", "vit_b"]
        
        # 중복 제거
        model_priority = list(dict.fromkeys(model_priority))
        
        sam_checkpoint = None
        found_model_type = None
        model_name = None
        
        print("🔍 SAM 모델 파일 검색 중...")
        
        for mtype in model_priority:
            filename = model_files[mtype]
            for search_path in search_paths:
                checkpoint_path = search_path / filename
                if checkpoint_path.exists():
                    sam_checkpoint = str(checkpoint_path)
                    found_model_type = mtype
                    model_name = model_names[mtype]
                    print(f"✅ 모델 파일 발견: {checkpoint_path} ({model_name})")
                    break
            if sam_checkpoint:
                break
        
        if not sam_checkpoint:
            print("❌ SAM 모델 파일이 없습니다.")
            print("\n📥 모델 파일을 다운로드해주세요:")
            print("🔸 ViT-H (2.4GB, 최고 품질): https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
            print("🔸 ViT-L (1.2GB, 중간 품질): https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth")  
            print("🔸 ViT-B (358MB, 빠른 처리): https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
            print("\n💡 다운로드 후 다음 경로에 저장:")
            for search_path in search_paths:
                print(f"   - {search_path.absolute()}")
            return None
        
        # SAM 모델 로드 (CPU/GPU 자동 선택)
        print(f"🤖 {model_name} 모델 로딩 중...")
        
        # CPU 환경에서의 성능 경고
        try:
            import torch
            if not torch.cuda.is_available():
                print("⚠️  GPU를 사용할 수 없어 CPU 모드로 실행됩니다.")
                if found_model_type == "vit_h":
                    print("💡 CPU에서는 ViT-B 모델(358MB)을 권장합니다. 처리 시간이 매우 오래 걸릴 수 있습니다.")
                print("🕐 CPU 처리 예상 시간: 30초~5분 (이미지 크기에 따라)")
        except ImportError:
            pass
        
        sam = sam_model_registry[found_model_type](checkpoint=sam_checkpoint)
        predictor = SamPredictor(sam)
        
        print(f"✅ SAM {model_name} 모델 로딩 완료!")
        return predictor
        
    except ImportError:
        print("❌ segment-anything가 설치되지 않았습니다.")
        print("설치: pip install segment-anything")
        return None
    except Exception as e:
        print(f"❌ SAM 모델 로딩 실패: {e}")
        return None


def extract_object_with_sam(
    image: np.ndarray,
    predictor
) -> Optional[Image.Image]:
    """
    SAM (Segment Anything Model)을 사용하여 객체를 추출합니다.
    
    Args:
        image: 입력 이미지 (BGR numpy array)
        predictor: 이미 로드된 SAM Predictor 객체
    
    Returns:
        추출된 이미지 (PIL Image) 또는 None (실패시)
    """
    try:
        if image is None or image.size == 0:
            print("❌ 유효하지 않은 이미지입니다.")
            return None
        
        if predictor is None:
            print("❌ SAM Predictor가 제공되지 않았습니다.")
            return None
            
        # BGR -> RGB 변환
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        print("🎯 SAM으로 객체 분할 실행 중...")
        
        # SAM에 이미지 설정
        predictor.set_image(image_rgb)
        
        # 중앙 점을 프롬프트로 사용 (객체가 중앙에 있다고 가정)
        h, w = image_rgb.shape[:2]
        input_point = np.array([[w//2, h//2]])
        input_label = np.array([1])  # 전경
        
        # 분할 실행
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        
        # 가장 좋은 마스크 선택
        best_mask = masks[np.argmax(scores)]
        best_score = scores[np.argmax(scores)]
        
        print(f"🎯 최고 마스크 점수: {best_score:.3f}")
        
        # 마스크에서 객체가 포함된 bounding box 계산
        mask_coords = np.where(best_mask > 0)
        
        if len(mask_coords[0]) == 0:
            print("❌ 객체를 찾을 수 없습니다.")
            return None
        
        # bounding box 좌표 계산
        y_min, y_max = mask_coords[0].min(), mask_coords[0].max()
        x_min, x_max = mask_coords[1].min(), mask_coords[1].max()
        
        print(f"📏 객체 영역: ({x_min}, {y_min}) → ({x_max}, {y_max})")
        
        # 객체 영역만 자르기
        cropped_image = image_rgb[y_min:y_max+1, x_min:x_max+1]
        cropped_mask = best_mask[y_min:y_max+1, x_min:x_max+1]
        
        # 결과 이미지 생성 (자른 영역만)
        result_pil = Image.fromarray(cropped_image).convert("RGBA")
        alpha = (cropped_mask * 255).astype(np.uint8)
        result_pil.putalpha(Image.fromarray(alpha))
        
        print(f"✅ SAM으로 추출 완료 (크기: {result_pil.size})")
        return result_pil
        
    except Exception as e:
        print(f"❌ SAM 추출 실패: {e}")
        return None


if __name__ == "__main__":
    print("image_utils.py 모듈이 정상적으로 로드되었습니다.")
    print("템플릿 매칭, 이미지 블렌딩, SAM 객체 추출 등의 범용 이미지 처리 기능을 제공합니다.")
