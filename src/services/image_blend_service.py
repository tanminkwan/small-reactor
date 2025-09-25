"""
이미지 결합 서비스

Single Responsibility Principle (SRP)에 따라
이미지 결합 기능만을 담당하는 서비스 클래스
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
from PIL import Image

from src.utils.config import Config
from src.utils.image_utils import match_and_blend_images, alpha_blend, create_smooth_mask


class ImageBlendService:
    """이미지 결합 서비스 클래스"""
    
    def __init__(self, config: Config):
        """
        초기화
        
        Args:
            config: 설정 관리 객체
        """
        self.config = config
        self._logger = logging.getLogger(__name__)
        
        # 출력 디렉토리 생성
        self._ensure_output_directory()
    
    def _ensure_output_directory(self) -> None:
        """출력 디렉토리가 존재하는지 확인하고 없으면 생성합니다."""
        try:
            output_dir = Path(self.config.get("image_blend_output_path", "./output/image_blend"))
            output_dir.mkdir(parents=True, exist_ok=True)
            self._logger.info(f"이미지 결합 출력 디렉토리 설정: {output_dir}")
        except Exception as e:
            self._logger.error(f"출력 디렉토리 생성 실패: {e}")
    
    def blend_images(
        self,
        original_image: np.ndarray,
        patch_image: np.ndarray,
        border_thickness: int = 15,
        blend_method: str = "alpha"
    ) -> Tuple[np.ndarray, str]:
        """
        두 이미지를 결합합니다.
        
        Args:
            original_image: 원본 이미지 (PIL Image)
            patch_image: 부분 이미지 (numpy array) - RGBA 투명도 지원  
            border_thickness: 경계 테두리 굵기
            blend_method: 경계 보정 방법 ("alpha", "seamless", "basic")
            
        Returns:
            Tuple[결합된 이미지, 상태 메시지]
        """
        try:
            if original_image is None or patch_image is None:
                return None, "❌ 원본 이미지와 부분 이미지를 모두 입력해주세요."
            
            # 원본 이미지 처리 (항상 RGB)
            if hasattr(original_image, 'convert'):
                original_array = np.array(original_image.convert('RGB'))
                original_bgr = cv2.cvtColor(original_array, cv2.COLOR_RGB2BGR)
            else:
                original_bgr = np.array(original_image)
                if len(original_bgr.shape) == 3 and original_bgr.shape[2] == 3:
                    original_bgr = cv2.cvtColor(original_bgr, cv2.COLOR_RGB2BGR)
            
            # 부분 이미지 처리 (numpy array 직접 처리)
            patch_alpha_mask = None
            
            # numpy array 입력 처리
            if isinstance(patch_image, np.ndarray):
                patch_array = patch_image
            else:
                # PIL 이미지인 경우 numpy로 변환
                patch_array = np.array(patch_image)
            
            # Alpha 채널 확인 (4채널 RGBA)
            if len(patch_array.shape) == 3 and patch_array.shape[2] == 4:
                # RGBA 이미지
                patch_alpha_mask = patch_array[:, :, 3]
                patch_rgb = patch_array[:, :, :3]
                patch_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)
                
                # 투명도가 실제로 있는지 확인
                if np.all(patch_alpha_mask == 255):
                    # 모든 픽셀이 불투명하면 alpha 채널 무시
                    patch_alpha_mask = None
            
            elif len(patch_array.shape) == 3 and patch_array.shape[2] == 3:
                # RGB 이미지 - 검은색 영역을 투명으로 간주할지 확인
                patch_bgr = cv2.cvtColor(patch_array, cv2.COLOR_RGB2BGR)
                
                # 검은색 영역이 많으면 투명 이미지가 RGB로 변환된 것으로 간주
                black_pixels = np.sum(np.all(patch_array == [0, 0, 0], axis=2))
                total_pixels = patch_array.shape[0] * patch_array.shape[1]
                black_ratio = black_pixels / total_pixels
                
                # 검은색 픽셀이 10% 이상이면 투명 이미지로 간주 (PNG→RGB 변환 시 투명→검은색)
                if black_ratio > 0.1:
                    # 검은색이 아닌 픽셀을 불투명(255), 검은색 픽셀을 투명(0)으로 설정
                    patch_alpha_mask = np.where(
                        np.all(patch_array == [0, 0, 0], axis=2), 
                        0,    # 검은색 → 투명
                        255   # 비검은색 → 불투명
                    ).astype(np.uint8)
            
            else:
                # 지원되지 않는 형식
                self._logger.error(f"지원되지 않는 이미지 형식: {patch_array.shape}")
                return None, f"❌ 지원되지 않는 이미지 형식입니다: {patch_array.shape}"
            
            self._logger.info(f"이미지 결합 시작 - 원본: {original_bgr.shape}, 부분: {patch_bgr.shape}")
            
            
            # Alpha 채널이 있는 경우 특별 처리
            if patch_alpha_mask is not None:
                # Alpha 채널을 고려한 블렌딩
                result_image, final_mask = self._blend_with_alpha_channel(
                    original_bgr, patch_bgr, patch_alpha_mask, border_thickness
                )
                
                final_result = result_image
                
            else:
                # 1단계: 템플릿 매칭 및 기본 결합 (기존 로직)
                result_image, border_mask = match_and_blend_images(
                    original_bgr, patch_bgr, border_thickness
                )
                
                # 2단계: 선택된 방법으로 경계 보정 (기존 로직)
                if blend_method == "alpha":
                    # Alpha blending
                    smooth_border_mask = create_smooth_mask(border_mask, blur_kernel=(21, 21))
                    final_result = alpha_blend(original_bgr, result_image, smooth_border_mask)
                    
                elif blend_method == "seamless":
                    # Seamless cloning (fallback to alpha if failed)
                    try:
                        from src.utils.image_utils import seamless_blend
                        
                        # 템플릿 매칭 결과 재계산
                        match_result = cv2.matchTemplate(original_bgr, patch_bgr, cv2.TM_CCOEFF_NORMED)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)
                        
                        top_left = max_loc
                        h, w = patch_bgr.shape[:2]
                        center_x = top_left[0] + w // 2
                        center_y = top_left[1] + h // 2
                        center = (center_x, center_y)
                        
                        # Seamless cloning용 마스크 생성
                        seamless_mask = np.zeros(original_bgr.shape[:2], dtype=np.uint8)
                        cv2.rectangle(seamless_mask, top_left, (top_left[0] + w, top_left[1] + h), 255, -1)
                        
                        # Seamless cloning 실행
                        final_result = seamless_blend(result_image, original_bgr, seamless_mask, center)
                        
                    except Exception as e:
                        self._logger.warning(f"Seamless cloning 실패, Alpha blending으로 대체: {e}")
                        smooth_border_mask = create_smooth_mask(border_mask, blur_kernel=(21, 21))
                        final_result = alpha_blend(original_bgr, result_image, smooth_border_mask)
                        
                else:  # basic
                    final_result = result_image
            
            # BGR to RGB 변환 후 PIL Image로 변환
            final_rgb = cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)
            final_pil = Image.fromarray(final_rgb)
            
            # 자동 저장
            saved_path = self.save_image(final_result)
            
            # 상태 메시지 생성
            if patch_alpha_mask is not None:
                status_message = f"✅ PNG 투명 이미지 결합 완료!\n투명 영역에서 원본 이미지 보존됨\n저장 위치: {saved_path}"
            else:
                status_message = f"✅ 이미지 결합 완료!\n방법: {blend_method}, 테두리 굵기: {border_thickness}px\n저장 위치: {saved_path}"
            
            return final_pil, status_message
            
        except Exception as e:
            error_message = f"❌ 이미지 결합 중 오류가 발생했습니다: {str(e)}"
            self._logger.error(error_message)
            return None, error_message
    
    def save_image(self, image: np.ndarray, filename: str = None) -> str:
        """
        결합된 이미지를 파일로 저장합니다.
        
        Args:
            image: 저장할 이미지 (H, W, 3) - BGR 형식
            filename: 저장할 파일명 (None이면 자동 생성)
            
        Returns:
            저장된 파일 경로
        """
        try:
            # 출력 디렉토리 확인/생성
            output_dir = Path(self.config.get("image_blend_output_path", "./output/image_blend"))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"blend_result_{timestamp}.jpg"
            
            # 확장자가 없으면 .jpg 추가
            if not Path(filename).suffix:
                filename += ".jpg"
            
            filepath = output_dir / filename
            
            # 이미지 저장
            success = cv2.imwrite(str(filepath), image)
            
            if not success:
                raise RuntimeError(f"이미지 저장 실패: {filepath}")
            
            self._logger.info(f"이미지 저장 완료: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self._logger.error(f"이미지 저장 중 오류 발생: {e}")
            raise
    
    def delete_latest_result_image(self) -> Tuple[bool, str]:
        """
        가장 최근 생성된 결합 결과 이미지 파일을 삭제합니다.
        
        Returns:
            (삭제 성공여부, 메시지)
        """
        try:
            output_dir = Path(self.config.get("image_blend_output_path", "./output/image_blend"))
            
            if not output_dir.exists():
                return False, "이미지 결합 출력 폴더가 존재하지 않습니다."
            
            # 가장 최근 생성된 blend_result 파일 찾기
            result_files = list(output_dir.glob("blend_result_*.jpg"))
            if not result_files:
                return False, "삭제할 결합 결과 파일이 없습니다."
            
            # 파일 생성 시간으로 정렬하여 가장 최근 파일 선택
            latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
            
            # 파일 삭제
            latest_file.unlink()
            
            self._logger.info(f"이미지 결합 결과 파일 삭제 완료: {latest_file}")
            return True, f"✅ 파일 삭제 완료: {latest_file.name}"
            
        except Exception as e:
            self._logger.error(f"결합 파일 삭제 실패: {e}")
            return False, f"❌ 파일 삭제 실패: {str(e)}"
    
    def get_blend_methods(self) -> list:
        """
        사용 가능한 블렌딩 방법 목록을 반환합니다.
        
        Returns:
            블렌딩 방법 리스트
        """
        return [
            "alpha",    # Alpha Blending (권장)
            "seamless", # Seamless Cloning  
            "basic"     # 기본 결합 (블렌딩 없음)
        ]
    
    def _blend_with_alpha_channel(
        self, 
        original_bgr: np.ndarray, 
        patch_bgr: np.ndarray, 
        patch_alpha: np.ndarray, 
        border_thickness: int = 15
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Alpha 채널을 가진 부분 이미지를 원본 이미지와 블렌딩합니다.
        투명한 영역(alpha=0)에서는 원본이 보이고, 불투명한 영역(alpha=255)에서는 부분 이미지가 보입니다.
        
        Args:
            original_bgr: 원본 이미지 (H, W, 3) - BGR 형식
            patch_bgr: 부분 이미지 (H, W, 3) - BGR 형식  
            patch_alpha: 부분 이미지의 alpha 채널 (H, W) - 0~255
            border_thickness: 경계 부드럽게 처리할 두께
            
        Returns:
            Tuple[블렌딩된 이미지, 사용된 마스크]
        """
        try:
            # 1단계: 템플릿 매칭으로 최적 위치 찾기
            match_result = cv2.matchTemplate(original_bgr, patch_bgr, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)
            
            # 최고 매칭 위치
            top_left = max_loc
            patch_h, patch_w = patch_bgr.shape[:2]
            orig_h, orig_w = original_bgr.shape[:2]
            
            # 2단계: 안전한 블렌딩 영역 계산 (경계 검사)
            start_x = max(0, top_left[0])
            start_y = max(0, top_left[1])
            end_x = min(orig_w, top_left[0] + patch_w)
            end_y = min(orig_h, top_left[1] + patch_h)
            
            valid_w = end_x - start_x
            valid_h = end_y - start_y
            
            if valid_w <= 0 or valid_h <= 0:
                return original_bgr, np.zeros(original_bgr.shape[:2], dtype=np.uint8)
            
            # 3단계: 결과 이미지 생성 (원본 이미지 복사)
            result_image = original_bgr.copy()
            
            # 4단계: 부분 이미지에서 유효한 영역 추출
            patch_offset_x = start_x - top_left[0]
            patch_offset_y = start_y - top_left[1]
            
            valid_patch_bgr = patch_bgr[
                patch_offset_y:patch_offset_y + valid_h,
                patch_offset_x:patch_offset_x + valid_w
            ]
            valid_patch_alpha = patch_alpha[
                patch_offset_y:patch_offset_y + valid_h,
                patch_offset_x:patch_offset_x + valid_w
            ]
            
            # 5단계: 원본 이미지에서 해당 영역 추출
            original_region = result_image[start_y:end_y, start_x:end_x]
            
            # 6단계: Alpha 채널 처리
            # Alpha 값을 0-1 범위로 정규화
            alpha_normalized = valid_patch_alpha.astype(np.float32) / 255.0
            
            # 경계 부드럽게 처리 (약간의 가우시안 블러 적용)
            if border_thickness > 0:
                kernel_size = min(border_thickness * 2 + 1, min(valid_h, valid_w))
                if kernel_size >= 3:
                    alpha_smooth = cv2.GaussianBlur(alpha_normalized, (kernel_size, kernel_size), 0)
                else:
                    alpha_smooth = alpha_normalized
            else:
                alpha_smooth = alpha_normalized
            
            # 3채널로 확장 (BGR에 맞춤)
            alpha_3d = np.repeat(alpha_smooth[:, :, np.newaxis], 3, axis=2)
            
            # 7단계: Alpha 블렌딩 수행
            # 중요: alpha=0일 때 원본이 100% 보이고, alpha=1일 때 부분 이미지가 100% 보임
            blended_region = (
                original_region.astype(np.float32) * (1.0 - alpha_3d) + 
                valid_patch_bgr.astype(np.float32) * alpha_3d
            ).astype(np.uint8)
            
            # 8단계: 결과 이미지에 블렌딩된 영역 적용
            result_image[start_y:end_y, start_x:end_x] = blended_region
            
            # 9단계: 사용된 마스크 반환 (전체 이미지 크기)
            full_alpha_mask = np.zeros(original_bgr.shape[:2], dtype=np.uint8)
            full_alpha_mask[start_y:end_y, start_x:end_x] = (alpha_smooth * 255).astype(np.uint8)
            
            return result_image, full_alpha_mask
            
        except Exception as e:
            self._logger.error(f"Alpha 채널 블렌딩 중 오류: {e}")
            # 오류 발생 시 원본 이미지 반환
            return original_bgr, np.zeros(original_bgr.shape[:2], dtype=np.uint8)


if __name__ == "__main__":
    print("ImageBlendService 모듈이 정상적으로 로드되었습니다.")
    print("이미지 결합 기능을 제공합니다.")
