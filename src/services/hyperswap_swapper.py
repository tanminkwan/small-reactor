"""
HyperSwap 얼굴 교체 서비스

Single Responsibility Principle (SRP)에 따라
HyperSwap(FaceFusion Labs) ONNX 모델을 이용한 얼굴 교체만을 담당하는 서비스

참고: https://github.com/facefusion/facefusion (processors/modules/face_swapper/core.py)
- 입력: source [1, 512] (L2 정규화된 ArcFace 임베딩), target [1, 3, 256, 256] (RGB, mean/std 0.5)
- 출력: output [1, 3, 256, 256], mask [1, 1, 256, 256]
- 라이선스: ResearchRAIL (비상업적 용도)
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.interfaces.face_swapper import IFaceSwapper
from src.interfaces.face_detector import Face
from src.utils.config import Config


# FaceFusion 'arcface_128' 워프 템플릿 (정규화 좌표, 5점 랜드마크)
ARCFACE_128_TEMPLATE = np.array(
    [
        [0.36167656, 0.40387734],
        [0.63696719, 0.40235469],
        [0.50019687, 0.56044219],
        [0.38710391, 0.72160547],
        [0.61507734, 0.72034453],
    ],
    dtype=np.float32,
)


class HyperSwapSwapper(IFaceSwapper):
    """HyperSwap 모델을 사용한 얼굴 교체 서비스"""

    MODEL_SIZE: Tuple[int, int] = (256, 256)
    MEAN = 0.5
    STD = 0.5

    def __init__(
        self,
        config: Config,
        model_path: Optional[str] = None,
        mask_blur: float = 0.3,
        use_model_mask: bool = True,
    ):
        """
        Args:
            config: 설정 객체
            model_path: HyperSwap ONNX 모델 경로 (None이면 설정의 hyperswap_model_path 사용)
            mask_blur: 박스 마스크 블러 강도 (FaceFusion 기본값 0.3)
            use_model_mask: 모델이 출력하는 얼굴 마스크를 합성에 사용할지 여부
        """
        self._config = config
        self._use_gpu = config.is_gpu_available()
        self._model_path = model_path or config.get_model_path("hyperswap")
        self._mask_blur = mask_blur
        self._use_model_mask = use_model_mask
        self._session = None
        self._has_mask_output = False

        self._logger = logging.getLogger(__name__)

        self._initialize_model()

    def _initialize_model(self) -> None:
        """HyperSwap ONNX 세션을 초기화합니다."""
        try:
            import onnxruntime as ort

            if not Path(self._model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self._model_path}")

            available = ort.get_available_providers()
            providers = ["CPUExecutionProvider"]
            if self._use_gpu and "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

            self._session = ort.InferenceSession(self._model_path, providers=providers)
            output_names = [o.name for o in self._session.get_outputs()]
            self._has_mask_output = "mask" in output_names

            self._logger.info(
                f"HyperSwap model initialized: {Path(self._model_path).name}, "
                f"providers={self._session.get_providers()}"
            )

        except ImportError:
            self._logger.error("onnxruntime not installed. Please install with: pip install onnxruntime-gpu")
            raise RuntimeError("onnxruntime not available")
        except Exception as e:
            self._logger.error(f"Failed to initialize HyperSwap model: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")

    # ------------------------------------------------------------------
    # IFaceSwapper 구현
    # ------------------------------------------------------------------
    def swap_face(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        source_face: Face,
        target_face: Face,
    ) -> np.ndarray:
        """
        소스 얼굴을 타겟 얼굴로 교체합니다.

        Args:
            source_image: 소스 이미지 (HyperSwap은 임베딩만 사용하므로 검증용)
            target_image: 타겟 이미지 (BGR)
            source_face: 소스 얼굴 정보 (embedding 필요)
            target_face: 타겟 얼굴 정보 (kps 필요)

        Returns:
            얼굴이 교체된 이미지 (BGR)
        """
        if source_face is None or getattr(source_face, "embedding", None) is None:
            raise ValueError("Source face embedding cannot be None")
        return self.swap_face_with_embedding(target_image, target_face, source_face.embedding)

    def swap_faces_in_image(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        source_faces: List[Face],
        target_faces: List[Face],
    ) -> np.ndarray:
        """이미지 내의 여러 얼굴을 교체합니다."""
        if not source_faces or not target_faces:
            self._logger.warning("No faces provided for swapping")
            return target_image.copy()

        min_faces = min(len(source_faces), len(target_faces))
        result_image = target_image.copy()
        for source_face, target_face in zip(source_faces[:min_faces], target_faces[:min_faces]):
            try:
                result_image = self.swap_face(source_image, result_image, source_face, target_face)
            except Exception as e:
                self._logger.error(f"Failed to swap face: {e}")
                continue
        return result_image

    def is_initialized(self) -> bool:
        """모델이 초기화되었는지 확인합니다."""
        return self._session is not None

    def get_model_info(self) -> dict:
        """모델 정보를 반환합니다."""
        return {
            "model_name": Path(self._model_path).stem,
            "model_path": self._model_path,
            "use_gpu": self._use_gpu,
            "initialized": self.is_initialized(),
            "providers": self._session.get_providers() if self._session else [],
            "device": self._config.get_device(),
        }

    # ------------------------------------------------------------------
    # 핵심 로직
    # ------------------------------------------------------------------
    def swap_face_with_embedding(
        self,
        target_image: np.ndarray,
        target_face: Face,
        source_embedding: np.ndarray,
    ) -> np.ndarray:
        """
        저장된 소스 임베딩으로 타겟 얼굴을 교체합니다.

        Args:
            target_image: 타겟 이미지 (BGR, uint8)
            target_face: 타겟 얼굴 정보 (kps 5점 랜드마크 필요)
            source_embedding: 소스 얼굴 ArcFace 임베딩 (512,)

        Returns:
            얼굴이 교체된 이미지 (BGR, uint8)
        """
        if not self.is_initialized():
            raise RuntimeError("Model not initialized")
        if target_image is None or target_image.size == 0 or target_image.ndim != 3:
            raise ValueError("Invalid target image")
        kps = getattr(target_face, "kps", None)
        if kps is None:
            raise ValueError("Target face kps (5-point landmarks) cannot be None")

        # 1. 타겟 얼굴 정렬 및 크롭 (arcface_128 템플릿, 256x256)
        crop_frame, affine_matrix = self._warp_face(target_image, np.asarray(kps, dtype=np.float32))

        # 2. 입력 준비
        source_input = self._prepare_source_embedding(source_embedding)
        target_input = self._prepare_crop_frame(crop_frame)

        # 3. 추론
        outputs = self._session.run(None, {"source": source_input, "target": target_input})
        swapped_crop = self._normalize_crop_frame(outputs[0][0])

        # 4. 합성 마스크 (박스 마스크 + 선택적 모델 마스크)
        crop_mask = self._create_box_mask(self.MODEL_SIZE, self._mask_blur)
        if self._use_model_mask and self._has_mask_output and len(outputs) > 1:
            model_mask = np.clip(outputs[1][0][0].astype(np.float32), 0.0, 1.0)
            crop_mask = np.minimum(crop_mask, model_mask)

        # 5. 원본 이미지에 붙여넣기
        return self._paste_back(target_image, swapped_crop, crop_mask, affine_matrix)

    def _warp_face(self, image: np.ndarray, kps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """5점 랜드마크 기준으로 얼굴을 정렬하여 크롭합니다."""
        template = ARCFACE_128_TEMPLATE * np.array(self.MODEL_SIZE, dtype=np.float32)
        affine_matrix = cv2.estimateAffinePartial2D(
            kps, template, method=cv2.RANSAC, ransacReprojThreshold=100
        )[0]
        crop = cv2.warpAffine(
            image, affine_matrix, self.MODEL_SIZE,
            borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA
        )
        return crop, affine_matrix

    @staticmethod
    def _prepare_source_embedding(embedding: np.ndarray) -> np.ndarray:
        """ArcFace 임베딩을 L2 정규화하여 [1, 512] float32로 변환합니다."""
        embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    def _prepare_crop_frame(self, crop_frame: np.ndarray) -> np.ndarray:
        """BGR 크롭 이미지를 모델 입력 텐서로 변환합니다."""
        frame = crop_frame[:, :, ::-1].astype(np.float32) / 255.0
        frame = (frame - self.MEAN) / self.STD
        frame = frame.transpose(2, 0, 1)
        return np.expand_dims(frame, axis=0).astype(np.float32)

    def _normalize_crop_frame(self, output: np.ndarray) -> np.ndarray:
        """모델 출력 텐서를 BGR float 이미지(0~255)로 변환합니다."""
        frame = output.transpose(1, 2, 0).astype(np.float32)
        frame = frame * self.STD + self.MEAN
        frame = np.clip(frame, 0.0, 1.0)
        return frame[:, :, ::-1] * 255.0

    @staticmethod
    def _create_box_mask(crop_size: Tuple[int, int], blur: float) -> np.ndarray:
        """가장자리를 부드럽게 처리한 박스 마스크를 생성합니다 (FaceFusion 방식)."""
        width, height = crop_size
        blur_amount = int(width * 0.5 * blur)
        blur_area = max(blur_amount // 2, 1)
        mask = np.ones((height, width), dtype=np.float32)
        mask[:blur_area, :] = 0
        mask[-blur_area:, :] = 0
        mask[:, :blur_area] = 0
        mask[:, -blur_area:] = 0
        if blur_amount > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), blur_amount * 0.25)
        return mask

    @staticmethod
    def _paste_back(
        image: np.ndarray,
        crop_frame: np.ndarray,
        crop_mask: np.ndarray,
        affine_matrix: np.ndarray,
    ) -> np.ndarray:
        """교체된 크롭 얼굴을 원본 이미지에 역변환하여 합성합니다."""
        img_h, img_w = image.shape[:2]
        crop_h, crop_w = crop_frame.shape[:2]
        inverse_matrix = cv2.invertAffineTransform(affine_matrix)

        # 붙여넣을 영역 계산 (전체 이미지가 아닌 얼굴 주변만 처리)
        corners = np.array([[0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]], dtype=np.float32)
        corners = cv2.transform(corners.reshape(1, -1, 2), inverse_matrix).reshape(-1, 2)
        x1, y1 = np.clip(np.floor(corners.min(axis=0)).astype(int), 0, [img_w, img_h])
        x2, y2 = np.clip(np.ceil(corners.max(axis=0)).astype(int), 0, [img_w, img_h])
        if x2 <= x1 or y2 <= y1:
            return image.copy()

        paste_matrix = inverse_matrix.copy()
        paste_matrix[0, 2] -= x1
        paste_matrix[1, 2] -= y1
        paste_w, paste_h = int(x2 - x1), int(y2 - y1)

        inverse_mask = cv2.warpAffine(crop_mask, paste_matrix, (paste_w, paste_h)).clip(0, 1)
        inverse_mask = np.expand_dims(inverse_mask, axis=-1)
        inverse_frame = cv2.warpAffine(
            crop_frame, paste_matrix, (paste_w, paste_h), borderMode=cv2.BORDER_REPLICATE
        )

        result = image.copy()
        region = result[y1:y2, x1:x2].astype(np.float32)
        blended = region * (1 - inverse_mask) + inverse_frame * inverse_mask
        result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(image.dtype)
        return result
