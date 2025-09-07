"""
CodeFormer 기반 이미지 향상 서비스

Single Responsibility Principle (SRP)에 따라
얼굴 복원만을 담당하는 서비스
"""

import logging
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Optional

from src.interfaces.image_enhancer import IImageEnhancer
from src.utils.config import Config


class CodeFormerEnhancer(IImageEnhancer):
    """CodeFormer 모델을 사용한 이미지 향상 서비스"""
    
    def __init__(self, config: Config):
        """
        Args:
            config: 설정 객체
        """
        self._config = config
        self._use_gpu = config.is_gpu_available()
        self._model_path = config.get_model_path("codeformer")
        self._device = 'cuda' if self._use_gpu and torch.cuda.is_available() else 'cpu'
        
        self._logger = logging.getLogger(__name__)
        
        # 모델 초기화
        self._model = None
        self._face_helper = None
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """CodeFormer 모델을 초기화합니다."""
        try:
            from codeformer.basicsr.utils.registry import ARCH_REGISTRY
            from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper
            
            # 모델 경로 확인
            if not Path(self._model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self._model_path}")
            
            # CodeFormer 모델 로드
            self._model = ARCH_REGISTRY.get('CodeFormer')(
                dim_embd=512, 
                codebook_size=1024, 
                n_head=8, 
                n_layers=9, 
                connect_list=['32', '64', '128', '256']
            ).to(self._device)
            
            # 체크포인트 로드
            checkpoint = torch.load(self._model_path, weights_only=True, map_location=self._device)['params_ema']
            self._model.load_state_dict(checkpoint)
            self._model.eval()
            
            # Face Helper 초기화
            self._face_helper = FaceRestoreHelper(
                upscale_factor=1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                save_ext='png',
                use_parse=True,
                device=self._device
            )
            
            self._logger.info(f"CodeFormer model initialized successfully. Device: {self._device}")
            
        except ImportError as e:
            self._logger.error(f"CodeFormer dependencies not installed: {e}")
            raise RuntimeError("CodeFormer dependencies not available")
        except Exception as e:
            self._logger.error(f"Failed to initialize CodeFormer model: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")
    
    def enhance_image(self, image: np.ndarray, enhancement_strength: float = 0.5) -> np.ndarray:
        """
        이미지의 얼굴을 복원합니다.
        
        Args:
            image: 입력 이미지
            enhancement_strength: 복원 강도 (0.0 ~ 1.0)
            
        Returns:
            복원된 이미지
            
        Raises:
            ValueError: 입력 이미지가 유효하지 않은 경우
            RuntimeError: 모델 추론 실패 시
        """
        # 입력 검증
        self._validate_input(image, enhancement_strength)
        
        if not self.is_initialized():
            raise RuntimeError("Model not initialized")
        
        try:
            # 이미지 크기 저장
            h, w_img, _ = image.shape
            
            # Face Helper 초기화
            self._face_helper.clean_all()
            self._face_helper.read_image(image)
            self._face_helper.get_face_landmarks_5(
                only_center_face=False, 
                resize=512, 
                eye_dist_threshold=5
            )
            self._face_helper.align_warp_face()
            
            # 얼굴 복원
            for idx, cropped_face in enumerate(self._face_helper.cropped_faces):
                cropped_face_t = self._img2tensor(cropped_face / 255., bgr2rgb=True, float32=True).to(self._device)
                self._normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                cropped_face_t = cropped_face_t.unsqueeze(0).to(self._device)
                
                try:
                    with torch.no_grad():
                        output = self._model(cropped_face_t, w=enhancement_strength, adain=True)[0]
                        restored_face = self._tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                    del output
                    if self._use_gpu:
                        torch.cuda.empty_cache()
                except RuntimeError as error:
                    self._logger.error(f'CodeFormer inference error: {error}')
                    self._logger.error('If you encounter CUDA out of memory, try to reduce enhancement_strength or use CPU.')
                    raise RuntimeError(f"Model inference failed: {error}")
                else:
                    restored_face = restored_face.astype('uint8')
                    self._face_helper.add_restored_face(restored_face)
            
            # 결과 생성
            self._face_helper.get_inverse_affine(None)
            restored_img = self._face_helper.paste_faces_to_input_image()
            
            # 최종 이미지 크기 조정 (원본 크기로)
            restored_img = cv2.resize(restored_img, (w_img, h))
            
            self._logger.debug("Image enhancement completed successfully")
            return restored_img
            
        except Exception as e:
            self._logger.error(f"Image enhancement failed: {e}")
            raise RuntimeError(f"Image enhancement failed: {e}")
    
    def enhance_face_regions(self, image: np.ndarray, face_regions: list, enhancement_strength: float = 0.5) -> np.ndarray:
        """
        특정 얼굴 영역들을 복원합니다.
        
        Args:
            image: 입력 이미지
            face_regions: 얼굴 영역 리스트 [(x1, y1, x2, y2), ...]
            enhancement_strength: 복원 강도 (0.0 ~ 1.0)
            
        Returns:
            복원된 이미지
        """
        if not face_regions:
            self._logger.warning("No face regions provided")
            return image.copy()
        
        # 얼굴 영역을 확장
        expanded_regions = self._expand_face_regions(image, face_regions)
        
        # 각 영역에 대해 복원 수행
        result_image = image.copy()
        
        for region in expanded_regions:
            x1, y1, x2, y2 = region
            
            # 영역 추출
            face_crop = image[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                continue
            
            try:
                # 얼굴 복원
                enhanced_crop = self.enhance_image(face_crop, enhancement_strength)
                
                # 원본 이미지에 복원된 영역 적용
                result_image[y1:y2, x1:x2] = enhanced_crop
                
            except Exception as e:
                self._logger.error(f"Failed to enhance face region {region}: {e}")
                continue
        
        return result_image
    
    def _expand_face_regions(self, image: np.ndarray, face_regions: list, expansion_factor: float = 0.2) -> list:
        """
        얼굴 영역을 확장합니다.
        
        Args:
            image: 입력 이미지
            face_regions: 얼굴 영역 리스트
            expansion_factor: 확장 비율 (0.0 ~ 1.0)
            
        Returns:
            확장된 얼굴 영역 리스트
        """
        h, w = image.shape[:2]
        expanded_regions = []
        
        for region in face_regions:
            x1, y1, x2, y2 = region
            
            # 영역 크기 계산
            face_w = x2 - x1
            face_h = y2 - y1
            
            # 확장 크기 계산
            expand_w = int(face_w * expansion_factor)
            expand_h = int(face_h * expansion_factor)
            
            # 확장된 좌표 계산
            new_x1 = max(0, x1 - expand_w)
            new_y1 = max(0, y1 - expand_h)
            new_x2 = min(w, x2 + expand_w)
            new_y2 = min(h, y2 + expand_h)
            
            expanded_regions.append((new_x1, new_y1, new_x2, new_y2))
        
        return expanded_regions
    
    def _validate_input(self, image: np.ndarray, enhancement_strength: float) -> None:
        """입력 검증"""
        if image is None:
            raise ValueError("Image cannot be None")
        
        if image.size == 0:
            raise ValueError("Image cannot be empty")
        
        if len(image.shape) != 3:
            raise ValueError("Image must be 3D (height, width, channels)")
        
        if not (0.0 <= enhancement_strength <= 1.0):
            raise ValueError("Enhancement strength must be between 0.0 and 1.0")
    
    def _img2tensor(self, img, bgr2rgb=True, float32=True):
        """이미지를 텐서로 변환"""
        from codeformer.basicsr.utils import img2tensor
        return img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)
    
    def _tensor2img(self, tensor, rgb2bgr=True, min_max=(0, 1)):
        """텐서를 이미지로 변환"""
        from codeformer.basicsr.utils import tensor2img
        return tensor2img(tensor, rgb2bgr=rgb2bgr, min_max=min_max)
    
    def _normalize(self, tensor, mean, std, inplace=False):
        """텐서 정규화"""
        from torchvision.transforms.functional import normalize
        return normalize(tensor, mean, std, inplace=inplace)
    
    def is_initialized(self) -> bool:
        """
        모델이 초기화되었는지 확인합니다.
        
        Returns:
            초기화 여부
        """
        return self._model is not None and self._face_helper is not None
    
    def get_model_info(self) -> dict:
        """
        모델 정보를 반환합니다.
        
        Returns:
            모델 정보 딕셔너리
        """
        return {
            "model_name": "codeformer",
            "model_path": self._model_path,
            "use_gpu": self._use_gpu,
            "device": self._device,
            "initialized": self.is_initialized()
        }
    
    def __del__(self):
        """소멸자: 리소스 정리"""
        if self._use_gpu:
            torch.cuda.empty_cache()