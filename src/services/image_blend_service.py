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
            patch_image: 부분 이미지 (PIL Image)  
            border_thickness: 경계 테두리 굵기
            blend_method: 경계 보정 방법 ("alpha", "seamless", "basic")
            
        Returns:
            Tuple[결합된 이미지, 상태 메시지]
        """
        try:
            if original_image is None or patch_image is None:
                return None, "❌ 원본 이미지와 부분 이미지를 모두 입력해주세요."
            
            # PIL Image를 numpy array로 변환
            if hasattr(original_image, 'convert'):
                original_array = np.array(original_image.convert('RGB'))
                original_bgr = cv2.cvtColor(original_array, cv2.COLOR_RGB2BGR)
            else:
                original_bgr = np.array(original_image)
                if len(original_bgr.shape) == 3 and original_bgr.shape[2] == 3:
                    original_bgr = cv2.cvtColor(original_bgr, cv2.COLOR_RGB2BGR)
            
            if hasattr(patch_image, 'convert'):
                patch_array = np.array(patch_image.convert('RGB'))
                patch_bgr = cv2.cvtColor(patch_array, cv2.COLOR_RGB2BGR)
            else:
                patch_bgr = np.array(patch_image)
                if len(patch_bgr.shape) == 3 and patch_bgr.shape[2] == 3:
                    patch_bgr = cv2.cvtColor(patch_bgr, cv2.COLOR_RGB2BGR)
            
            self._logger.info(f"이미지 결합 시작 - 원본: {original_bgr.shape}, 부분: {patch_bgr.shape}")
            self._logger.info(f"설정 - border_thickness: {border_thickness}, blend_method: {blend_method}")
            
            # 1단계: 템플릿 매칭 및 기본 결합
            result_image, border_mask = match_and_blend_images(
                original_bgr, patch_bgr, border_thickness
            )
            
            # 2단계: 선택된 방법으로 경계 보정
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


if __name__ == "__main__":
    print("ImageBlendService 모듈이 정상적으로 로드되었습니다.")
    print("이미지 결합 기능을 제공합니다.")
