"""
Inpainting 서비스

StableDiffusionInpaintPipeline을 관리하는 서비스 클래스
Single Responsibility Principle (SRP)에 따라 inpainting 기능만을 담당
"""

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from typing import Optional, Tuple
import logging
from pathlib import Path

from src.utils.config import Config


class InpaintService:
    """Inpainting 서비스 클래스"""
    
    def __init__(self, config: Config):
        """
        초기화
        
        Args:
            config: 설정 관리 객체
        """
        self.config = config
        self._logger = logging.getLogger(__name__)
        self._pipeline: Optional[StableDiffusionInpaintPipeline] = None
        
    def _initialize_pipeline(self) -> None:
        """
        파이프라인을 초기화합니다. (한 번만 실행)
        """
        if self._pipeline is not None:
            return
            
        try:
            model_path = self.config.get("inpaint_model_path")
            device = self.config.get_device()
            
            self._logger.info(f"Inpaint 파이프라인 초기화 중... (모델: {model_path}, 디바이스: {device})")
            
            # 파이프라인 로드
            self._pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            
            # GPU 사용 시 CUDA로 이동
            if device == "cuda":
                self._pipeline = self._pipeline.to("cuda")
            
            self._logger.info("Inpaint 파이프라인 초기화 완료")
            
        except Exception as e:
            self._logger.error(f"Inpaint 파이프라인 초기화 실패: {e}")
            raise
    
    def pad_to_multiple(self, arr: np.ndarray, multiple: int = 64, fill_value: int = 0) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        이미지나 마스크를 지정된 배수로 패딩합니다.
        
        Args:
            arr: 패딩할 배열 (이미지: H,W,3 or 마스크: H,W)
            multiple: 패딩 기준 배수
            fill_value: 패딩 영역에 채울 값
            
        Returns:
            Tuple[패딩된 배열, (원본 높이, 원본 너비)]
        """
        if arr.ndim == 3:  # 컬러 이미지
            h, w, c = arr.shape
            new_h = ((h + multiple - 1) // multiple) * multiple
            new_w = ((w + multiple - 1) // multiple) * multiple
            pad_img = np.full((new_h, new_w, c), fill_value, dtype=arr.dtype)
            pad_img[:h, :w, :] = arr
        elif arr.ndim == 2:  # 마스크 (단일 채널)
            h, w = arr.shape
            new_h = ((h + multiple - 1) // multiple) * multiple
            new_w = ((w + multiple - 1) // multiple) * multiple
            pad_img = np.full((new_h, new_w), fill_value, dtype=arr.dtype)
            pad_img[:h, :w] = arr
        else:
            raise ValueError(f"지원하지 않는 shape: {arr.shape}")

        return pad_img, (h, w)
    
    def generate_inpaint_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str,
        negative_prompt: str = "",
        guidance_scale: float = 9.0,
        num_inference_steps: int = 50,
        strength: float = 1.0
    ) -> np.ndarray:
        """
        Inpainting을 수행하여 최종 이미지를 생성합니다.
        
        Args:
            image: 원본 이미지 (H, W, 3) - BGR 형식
            mask: 마스크 이미지 (H, W) - 255가 inpainting 영역
            prompt: Positive 프롬프트
            negative_prompt: Negative 프롬프트
            guidance_scale: 가이던스 스케일
            num_inference_steps: 추론 스텝 수
            strength: 수정 강도 (0.1=마스크만, 1.0=주변까지)
            
        Returns:
            생성된 이미지 (H, W, 3) - BGR 형식
        """
        try:
            # 파이프라인 초기화 (한 번만)
            self._initialize_pipeline()
            
            if self._pipeline is None:
                raise RuntimeError("파이프라인 초기화 실패")
            
            # 원본 크기 저장
            h, w = image.shape[:2]
            
            # 64의 배수로 패딩
            padded_img, (orig_h, orig_w) = self.pad_to_multiple(image, 64, fill_value=0)
            padded_mask, _ = self.pad_to_multiple(mask, 64, fill_value=0)
            
            # BGR to RGB 변환 및 PIL Image 변환
            image_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            mask_pil = Image.fromarray(padded_mask).convert("L")
            
            self._logger.info(f"Inpainting 시작 - 크기: {image_pil.size}, 프롬프트: {prompt[:50]}...")
            # Strength에 따른 실제 스텝 수 자동 보정
            # 사용자가 원하는 스텝 수가 실제로 수행되도록 조정
            adjusted_steps = int(num_inference_steps / strength) if strength > 0 else num_inference_steps
            
            self._logger.info(f"파라미터: guidance_scale={guidance_scale}, num_inference_steps={num_inference_steps} -> adjusted_steps={adjusted_steps}, strength={strength}")
            
            # Inpainting 수행
            result = self._pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image_pil,
                mask_image=mask_pil,
                height=image_pil.height,
                width=image_pil.width,
                guidance_scale=guidance_scale,
                num_inference_steps=adjusted_steps,
                strength=strength
            ).images[0]
            
            # 원본 크기로 크롭
            result = result.crop((0, 0, w, h))
            
            # PIL to numpy 변환 및 RGB to BGR
            result_array = np.array(result)
            result_bgr = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
            
            self._logger.info("Inpainting 완료")
            
            return result_bgr
            
        except Exception as e:
            self._logger.error(f"Inpainting 중 오류 발생: {e}")
            raise
    
    def save_image(self, image: np.ndarray, filename: str = None) -> str:
        """
        생성된 이미지를 파일로 저장합니다.
        
        Args:
            image: 저장할 이미지 (H, W, 3) - BGR 형식
            filename: 저장할 파일명 (None이면 자동 생성)
            
        Returns:
            저장된 파일 경로
        """
        try:
            # Inpaint 전용 출력 디렉토리 확인/생성
            output_dir = Path(self.config.get("inpaint_output_path", "./output/inpaint"))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if filename is None:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"inpaint_result_{timestamp}.jpg"
            
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
    
    def cleanup(self) -> None:
        """
        리소스 정리
        """
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self._logger.info("Inpaint 파이프라인 리소스 정리 완료")
    
    def delete_latest_result_image(self) -> Tuple[bool, str]:
        """
        가장 최근 생성된 inpaint 결과 이미지 파일을 삭제합니다.
        
        Returns:
            (삭제 성공여부, 메시지)
        """
        try:
            output_dir = Path(self.config.get("inpaint_output_path", "./output/inpaint"))
            
            if not output_dir.exists():
                return False, "Inpaint 출력 폴더가 존재하지 않습니다."
            
            # 가장 최근 생성된 inpaint_result 파일 찾기
            result_files = list(output_dir.glob("inpaint_result_*.jpg"))
            if not result_files:
                return False, "삭제할 inpaint 결과 파일이 없습니다."
            
            # 파일 생성 시간으로 정렬하여 가장 최근 파일 선택
            latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
            
            # 파일 삭제
            latest_file.unlink()
            
            self._logger.info(f"Inpaint 결과 이미지 파일 삭제 완료: {latest_file}")
            return True, f"✅ 파일 삭제 완료: {latest_file.name}"
            
        except Exception as e:
            self._logger.error(f"Inpaint 파일 삭제 실패: {e}")
            return False, f"❌ 파일 삭제 실패: {str(e)}"
