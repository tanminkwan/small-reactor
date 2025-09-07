#!/usr/bin/env python3
"""
Gradio Face Manager UI

기능:
1. 이미지 업로드 및 첫 번째 얼굴 추출
2. 등록된 모든 embedding 파일 목록 조회
"""

import gradio as gr
import numpy as np
import cv2
import json
import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

from src.services.buffalo_detector import BuffaloDetector
from src.utils.config import Config

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# .env 파일 로드
load_dotenv()

class FaceManager:
    """얼굴 관리 클래스"""
    
    def __init__(self):
        """초기화"""
        self.config = Config()
        self.detector = BuffaloDetector(self.config)
        
        # .env에서 경로 설정 가져오기
        self.faces_path = os.getenv("FACES_PATH", "./faces")
        self.output_path = os.getenv("OUTPUT_PATH", "./outputs")
        
        # 디렉토리 생성
        self.faces_dir = Path(self.faces_path)
        self.faces_dir.mkdir(exist_ok=True)
        Path(self.output_path).mkdir(exist_ok=True)
    
    def _get_safe_filename(self, filename: str) -> str:
        """
        안전한 파일명을 생성합니다.
        
        Args:
            filename: 원본 파일명
            
        Returns:
            안전한 파일명
        """
        import re
        import time
        
        # 한글과 특수문자를 제거하고 영문, 숫자, 언더스코어만 허용
        safe_name = re.sub(r'[^\w\-_.]', '_', filename)
        
        # 연속된 언더스코어를 하나로 변경
        safe_name = re.sub(r'_+', '_', safe_name)
        
        # 시작과 끝의 언더스코어 제거
        safe_name = safe_name.strip('_')
        
        # 빈 문자열이면 타임스탬프 사용
        if not safe_name:
            safe_name = f"face_{int(time.time())}"
        
        return safe_name
    
    def draw_face_boxes(self, image: np.ndarray, faces: List) -> np.ndarray:
        """
        이미지에 얼굴 박스와 인덱스를 그립니다.
        
        Args:
            image: 입력 이미지 (BGR)
            faces: 탐지된 얼굴 리스트
            
        Returns:
            박스와 인덱스가 그려진 이미지 (BGR)
        """
        result_image = image.copy()
        
        for i, face in enumerate(faces):
            bbox = face.bbox
            x1, y1, x2, y2 = map(int, bbox)
            
            # 박스 그리기 (초록색)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 인덱스 텍스트 그리기
            label = f"{i+1}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            
            # 텍스트 크기 계산
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # 텍스트 배경 박스 그리기
            cv2.rectangle(result_image, 
                         (x1, y1 - text_height - 10), 
                         (x1 + text_width + 10, y1), 
                         (0, 255, 0), -1)
            
            # 텍스트 그리기 (검은색)
            cv2.putText(result_image, label, 
                       (x1 + 5, y1 - 5), 
                       font, font_scale, (0, 0, 0), thickness)
        
        return result_image
    
    def detect_and_draw_faces(self, image: np.ndarray) -> Tuple[bool, str, np.ndarray]:
        """
        이미지에서 얼굴을 탐지하고 박스와 인덱스를 그려서 반환합니다.
        (얼굴 교체 작업에는 영향을 주지 않는 시각화 전용 함수)
        
        Args:
            image: 입력 이미지 (BGR)
            
        Returns:
            (성공여부, 메시지, 박스가 그려진 이미지)
        """
        try:
            # 얼굴 탐지
            faces = self.detector.detect_faces(image)
            
            if not faces:
                return False, "이미지에서 얼굴을 찾을 수 없습니다.", None
            
            # 박스와 인덱스가 그려진 이미지 생성
            result_image = self.draw_face_boxes(image, faces)
            
            # BGR을 RGB로 변환 (Gradio 표시용)
            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            
            message = f"얼굴 탐지 완료!\n탐지된 얼굴 수: {len(faces)}개\n(좌→우, 위→아래 순으로 인덱스 부여)"
            
            return True, message, result_image_rgb
            
        except Exception as e:
            logger.error(f"얼굴 탐지 실패: {e}")
            return False, f"얼굴 탐지 실패: {str(e)}", None
    
    def extract_first_face(self, image_bgr: np.ndarray, filename: str, image_rgb: np.ndarray = None) -> Tuple[bool, str, str]:
        """
        이미지에서 첫 번째 얼굴을 추출하여 저장합니다.
        
        Args:
            image_bgr: BGR 형식 입력 이미지 (얼굴 탐지용)
            filename: 원본 파일명
            image_rgb: RGB 형식 입력 이미지 (얼굴 추출용, 선택사항)
            
        Returns:
            (성공여부, 메시지, 저장된 파일 경로)
        """
        try:
            # 얼굴 탐지 (BGR 이미지 사용)
            faces = self.detector.detect_faces(image_bgr)
            
            if not faces:
                return False, "이미지에서 얼굴을 찾을 수 없습니다.", ""
            
            # 첫 번째 얼굴 선택
            first_face = faces[0]
            
            # 파일명 생성 (확장자 제거 및 안전한 파일명으로 변환)
            base_name = self._get_safe_filename(Path(filename).stem)
            
            # 얼굴 영역 추출
            bbox = first_face.bbox
            x1, y1, x2, y2 = map(int, bbox)
            
            # 경계 확인
            h, w = image_bgr.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # 얼굴 이미지 추출 (RGB 이미지가 있으면 사용, 없으면 BGR에서 변환)
            if image_rgb is not None:
                # RGB 이미지에서 얼굴 영역 추출
                face_image_rgb = image_rgb[y1:y2, x1:x2]
            else:
                # BGR 이미지에서 얼굴 영역 추출 후 RGB로 변환
                face_image_bgr = image_bgr[y1:y2, x1:x2]
                face_image_rgb = cv2.cvtColor(face_image_bgr, cv2.COLOR_BGR2RGB)
            
            if face_image_rgb.size == 0:
                return False, "얼굴 영역을 추출할 수 없습니다.", ""
            
            # 파일 저장 경로 (안전한 파일명 사용)
            face_image_path = self.faces_dir / f"{base_name}.jpg"
            embedding_path = self.faces_dir / f"{base_name}.json"
            
            # 이미지 저장 (PIL 사용으로 색상 정확성 보장)
            from PIL import Image
            pil_image = Image.fromarray(face_image_rgb)
            pil_image.save(str(face_image_path))
            
            # embedding 저장
            with open(embedding_path, 'w') as f:
                json.dump(first_face.embedding.tolist(), f)
            
            logger.info(f"얼굴 추출 완료: {face_image_path}, {embedding_path}")
            
            return True, f"얼굴 추출 완료!\n저장된 파일:\n- {face_image_path.name}\n- {embedding_path.name}", str(face_image_path)
            
        except Exception as e:
            logger.error(f"얼굴 추출 실패: {e}")
            return False, f"얼굴 추출 실패: {str(e)}", ""
    
    def get_embedding_list(self) -> List[dict]:
        """
        등록된 모든 embedding 파일 목록을 반환합니다.
        
        Returns:
            embedding 파일 정보 리스트
        """
        try:
            embedding_files = []
            
            # faces 디렉토리의 모든 .json 파일 검색
            for json_file in self.faces_dir.glob("*.json"):
                try:
                    # 해당하는 이미지 파일 찾기
                    base_name = json_file.stem
                    image_file = self.faces_dir / f"{base_name}.jpg"
                    
                    # 파일 정보 수집
                    file_info = {
                        "name": base_name,
                        "json_file": json_file.name,
                        "image_file": image_file.name if image_file.exists() else "없음",
                        "json_size": json_file.stat().st_size,
                        "image_size": image_file.stat().st_size if image_file.exists() else 0,
                        "json_path": str(json_file),
                        "image_path": str(image_file) if image_file.exists() else None
                    }
                    
                    embedding_files.append(file_info)
                    
                except Exception as e:
                    logger.warning(f"파일 정보 수집 실패 {json_file}: {e}")
                    continue
            
            # 이름순으로 정렬
            embedding_files.sort(key=lambda x: x["name"])
            
            return embedding_files
            
        except Exception as e:
            logger.error(f"embedding 목록 조회 실패: {e}")
            return []
    
    def swap_faces(self, target_image: np.ndarray, face_indices: str, source_face_name: str) -> Tuple[bool, str, np.ndarray]:
        """
        타겟 이미지의 얼굴들을 소스 얼굴로 교체합니다.
        
        Args:
            target_image: 타겟 이미지 (BGR)
            face_indices: 교체할 얼굴 인덱스 (쉼표로 구분, 비워두면 모든 얼굴)
            source_face_name: 소스 얼굴 이름
            
        Returns:
            (성공여부, 메시지, 교체된 이미지)
        """
        try:
            # source_face_name이 None이거나 빈 문자열인 경우 첫 번째 저장된 얼굴 사용
            if not source_face_name or source_face_name.strip() == "":
                # faces 디렉토리에서 첫 번째 .json 파일 찾기
                json_files = list(self.faces_dir.glob("*.json"))
                if not json_files:
                    return False, "저장된 얼굴이 없습니다. 먼저 얼굴을 추출해주세요.", None
                source_face_name = json_files[0].stem  # .json 확장자 제거
            
            # 소스 얼굴 embedding 로드
            source_embedding_path = self.faces_dir / f"{source_face_name}.json"
            if not source_embedding_path.exists():
                return False, f"소스 얼굴 파일을 찾을 수 없습니다: {source_face_name}", None
            
            with open(source_embedding_path, 'r') as f:
                source_embedding = np.array(json.load(f))
            
            # 타겟 이미지에서 얼굴 탐지
            target_faces = self.detector.detect_faces(target_image)
            if not target_faces:
                return False, "타겟 이미지에서 얼굴을 찾을 수 없습니다.", None
            
            # 얼굴 인덱스와 위치 정보 로그 출력
            logger.info(f"탐지된 얼굴 수: {len(target_faces)}")
            for i, face in enumerate(target_faces):
                bbox = face.bbox
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                logger.info(f"얼굴 {i+1}: 중심점 ({center_x:.1f}, {center_y:.1f}), bbox ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
            
            # 교체할 얼굴 인덱스 파싱
            if face_indices.strip():
                try:
                    indices = [int(x.strip()) - 1 for x in face_indices.split(',')]  # 1-based to 0-based
                    indices = [i for i in indices if 0 <= i < len(target_faces)]
                    if not indices:
                        return False, "유효한 얼굴 인덱스가 없습니다.", None
                except ValueError:
                    return False, "얼굴 인덱스 형식이 올바르지 않습니다.", None
            else:
                # 모든 얼굴 교체
                indices = list(range(len(target_faces)))
            
            # 얼굴 교체 수행 (성공했던 방식 사용)
            from insightface import model_zoo
            from insightface.app.common import Face
            
            # swapper 모델 로드 (성공했던 방식)
            swapper_model_path = self.config.get_model_path("inswapper")
            swapper = model_zoo.get_model(swapper_model_path)
            
            # source_face를 InsightFace의 Face 객체로 변환 (성공했던 방식)
            source_face = Face()
            # embedding을 float32로 변환 (ONNX 모델 요구사항)
            source_face.embedding = source_embedding.astype(np.float32)
            
            # 교체할 타겟 얼굴들 선택
            target_faces_to_swap = [target_faces[i] for i in indices]
            
            # 얼굴 교체 (성공했던 방식)
            result_image = target_image.copy()
            for target_face in target_faces_to_swap:
                try:
                    result_image = swapper.get(result_image, target_face, source_face)
                except Exception as e:
                    logger.error(f"Failed to swap face: {e}")
                    continue
            
            # BGR을 RGB로 변환
            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            
            message = f"얼굴 교체 완료!\n교체된 얼굴: {len(indices)}개\n소스 얼굴: {source_face_name}"
            
            return True, message, result_image_rgb
            
        except Exception as e:
            logger.error(f"얼굴 교체 실패: {e}")
            return False, f"얼굴 교체 실패: {str(e)}", None
    
    def enhance_faces_with_codeformer(self, image: np.ndarray, face_indices: str = "") -> Tuple[bool, str, np.ndarray]:
        """
        CodeFormer를 사용하여 얼굴 영역을 복원합니다.
        
        Args:
            image: 입력 이미지 (BGR)
            face_indices: 복원할 얼굴 인덱스 (쉼표로 구분, 비워두면 모든 얼굴)
            
        Returns:
            (성공여부, 메시지, 복원된 이미지)
        """
        try:
            # 얼굴 탐지
            faces = self.detector.detect_faces(image)
            if not faces:
                return False, "이미지에서 얼굴을 찾을 수 없습니다.", None
            
            # 얼굴 인덱스와 위치 정보 로그 출력
            logger.info(f"CodeFormer 복원 - 탐지된 얼굴 수: {len(faces)}")
            for i, face in enumerate(faces):
                bbox = face.bbox
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                logger.info(f"얼굴 {i+1}: 중심점 ({center_x:.1f}, {center_y:.1f}), bbox ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
            
            # 얼굴 인덱스 파싱
            if face_indices.strip():
                indices = [int(x.strip()) - 1 for x in face_indices.split(',')]
                indices = [i for i in indices if 0 <= i < len(faces)]
                if not indices:
                    return False, "유효한 얼굴 인덱스가 없습니다.", None
            else:
                indices = list(range(len(faces)))
            
            # CodeFormer 초기화
            import torch
            from torchvision.transforms.functional import normalize
            from codeformer.basicsr.utils import img2tensor, tensor2img
            from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper
            from codeformer.basicsr.utils.registry import ARCH_REGISTRY
            
            class FaceRestorer:
                def __init__(self, model_path, use_gpu=True):
                    self.model_path = model_path
                    self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
                    self.use_gpu = use_gpu
                    
                    # 모델 로드
                    self.model = ARCH_REGISTRY.get('CodeFormer')(
                        dim_embd=512, 
                        codebook_size=1024, 
                        n_head=8, 
                        n_layers=9, 
                        connect_list=['32', '64', '128', '256']
                    ).to(self.device)
                    
                    checkpoint = torch.load(model_path, weights_only=True, map_location=self.device)['params_ema']
                    self.model.load_state_dict(checkpoint)
                    self.model.eval()
                    
                    # Face Helper 초기화
                    self.face_helper = FaceRestoreHelper(
                        upscale_factor=1,
                        face_size=512,
                        crop_ratio=(1, 1),
                        det_model='retinaface_resnet50',
                        save_ext='png',
                        use_parse=True,
                        device=self.device
                    )
                    
                    logger.info(f"FaceRestorer initialized using device: {self.device}")
                
                def restore(self, input_image, w=0.5, only_center_face=False):
                    h, w_img, _ = input_image.shape
                    
                    self.face_helper.clean_all()
                    self.face_helper.read_image(input_image)
                    self.face_helper.get_face_landmarks_5(
                        only_center_face=only_center_face, 
                        resize=512, 
                        eye_dist_threshold=5
                    )
                    self.face_helper.align_warp_face()
                    
                    for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
                        cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True).to(self.device)
                        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                        cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)
                        
                        try:
                            with torch.no_grad():
                                output = self.model(cropped_face_t, w=w, adain=True)[0]
                                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                            del output
                            if self.use_gpu:
                                torch.cuda.empty_cache()
                        except RuntimeError as error:
                            logger.error(f'CodeFormer Error: {error}')
                        else:
                            restored_face = restored_face.astype('uint8')
                            self.face_helper.add_restored_face(restored_face)
                    
                    self.face_helper.get_inverse_affine(None)
                    restored_img = self.face_helper.paste_faces_to_input_image()
                    restored_img = cv2.resize(restored_img, (w_img, h))
                    
                    return restored_img
            
            # CodeFormer 모델 로드
            codeformer_model_path = self.config.get_model_path("codeformer")
            restorer = FaceRestorer(model_path=codeformer_model_path, use_gpu=True)
            
            # 얼굴 영역 확장 함수
            def expand_face_regions(image, face_regions, expansion_factor=0.2):
                h, w = image.shape[:2]
                expanded_regions = []
                
                for region in face_regions:
                    x1, y1, x2, y2 = region
                    face_w = x2 - x1
                    face_h = y2 - y1
                    expand_w = int(face_w * expansion_factor)
                    expand_h = int(face_h * expansion_factor)
                    
                    new_x1 = max(0, x1 - expand_w)
                    new_y1 = max(0, y1 - expand_h)
                    new_x2 = min(w, x2 + expand_w)
                    new_y2 = min(h, y2 + expand_h)
                    
                    expanded_regions.append((new_x1, new_y1, new_x2, new_y2))
                
                return expanded_regions
            
            # 선택된 얼굴들의 bbox를 리스트로 변환
            face_regions = []
            for i in indices:
                bbox = faces[i].bbox
                x1, y1, x2, y2 = map(int, bbox)
                face_regions.append((x1, y1, x2, y2))
            
            # 얼굴 영역 확장
            expanded_regions = expand_face_regions(image, face_regions, expansion_factor=0.2)
            
            # 각 확장된 얼굴 영역에 대해 복원 수행
            enhanced_image = image.copy()
            
            for i, region in enumerate(expanded_regions):
                x1, y1, x2, y2 = region
                logger.info(f"얼굴 영역 {i+1} 복원 중... (확장된 영역: {region})")
                
                # 영역 추출
                face_crop = image[y1:y2, x1:x2]
                
                if face_crop.size == 0:
                    continue
                
                try:
                    # 얼굴 복원
                    enhanced_crop = restorer.restore(face_crop, w=0.5)
                    
                    # 원본 이미지에 복원된 영역 적용
                    enhanced_image[y1:y2, x1:x2] = enhanced_crop
                    logger.info(f"얼굴 영역 {i+1} 복원 완료")
                    
                except Exception as e:
                    logger.error(f"얼굴 영역 {i+1} 복원 실패: {e}")
                    continue
            
            # BGR을 RGB로 변환
            enhanced_image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
            
            message = f"CodeFormer 복원 완료!\n복원된 얼굴: {len(indices)}개"
            
            return True, message, enhanced_image_rgb
            
        except Exception as e:
            logger.error(f"CodeFormer 복원 실패: {e}")
            return False, f"CodeFormer 복원 실패: {str(e)}", None

# 전역 FaceManager 인스턴스
face_manager = FaceManager()

def perform_face_swap_with_optional_codeformer(file_path, face_indices, source_face_name, use_codeformer):
    """
    얼굴 교체를 수행하고, 선택적으로 CodeFormer 복원도 수행합니다.
    
    Args:
        file_path: 타겟 이미지 파일 경로
        face_indices: 교체할 얼굴 인덱스
        source_face_name: 소스 얼굴 이름
        use_codeformer: CodeFormer 복원 사용 여부
        
    Returns:
        (최종 이미지, 메시지, 최종 이미지)
    """
    if file_path is None:
        return None, "타겟 이미지를 업로드해주세요.", None
    
    if not source_face_name:
        return None, "바꿀 얼굴을 선택해주세요.", None
    
    try:
        from PIL import Image
        
        # PIL로 이미지 로드
        pil_image = Image.open(file_path)
        image_rgb = np.array(pil_image)
        
        # RGB를 BGR로 변환 (얼굴 교체용)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # 얼굴 교체 수행
        success, message, swapped_image_rgb = face_manager.swap_faces(
            image_bgr, face_indices, source_face_name
        )
        
        if not success:
            return None, message, None
        
        final_image = swapped_image_rgb
        final_message = message
        
        # CodeFormer 복원이 체크되어 있으면 수행
        if use_codeformer:
            try:
                # 얼굴 교체된 이미지를 BGR로 변환 (CodeFormer용)
                swapped_image_bgr = cv2.cvtColor(swapped_image_rgb, cv2.COLOR_RGB2BGR)
                
                # CodeFormer 복원 수행
                cf_success, cf_message, enhanced_image_rgb = face_manager.enhance_faces_with_codeformer(
                    swapped_image_bgr, face_indices
                )
                
                if cf_success:
                    final_image = enhanced_image_rgb
                    final_message = f"{message}\n{cf_message}"
                else:
                    final_message = f"{message}\nCodeFormer 복원 실패: {cf_message}"
                    
            except Exception as e:
                logger.error(f"CodeFormer 복원 실패: {e}")
                final_message = f"{message}\nCodeFormer 복원 실패: {str(e)}"
        
        # 최종 결과 이미지 파일로 저장
        if final_image is not None:
            try:
                from datetime import datetime
                
                # .env에서 출력 경로 가져오기
                output_path = os.getenv("OUTPUT_PATH", "./outputs")
                output_dir = Path(output_path)
                output_dir.mkdir(exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = output_dir / f"final_result_{timestamp}.jpg"
                
                # PIL로 이미지 저장 (RGB 형식)
                from PIL import Image
                pil_image = Image.fromarray(final_image)
                pil_image.save(output_filename, "JPEG", quality=95)
                
                final_message += f"\n\n💾 최종 결과 저장: {output_filename}"
                logger.info(f"최종 결과 이미지 저장: {output_filename}")
                
            except Exception as save_error:
                logger.error(f"이미지 저장 실패: {save_error}")
                final_message += f"\n⚠️ 이미지 저장 실패: {str(save_error)}"
        
        return final_image, final_message, final_image
        
    except Exception as e:
        logger.error(f"얼굴 교체 실패: {e}")
        return None, f"얼굴 교체 실패: {str(e)}", None

def process_uploaded_image(image_bgr, filename, image_rgb=None):
    """
    업로드된 이미지를 처리합니다.
    
    Args:
        image_bgr: BGR 형식 이미지 (얼굴 탐지용)
        filename: 파일명
        image_rgb: RGB 형식 이미지 (표시용, 선택사항)
        
    Returns:
        (성공여부, 메시지, 추출된 얼굴 이미지)
    """
    if image_bgr is None:
        return False, "이미지를 업로드해주세요.", None
    
    # 얼굴 추출 (BGR 이미지로 탐지, RGB 이미지로 추출)
    success, message, face_path = face_manager.extract_first_face(image_bgr, filename, image_rgb)
    
    if success and face_path:
        # 추출된 얼굴 이미지 로드 (PIL 사용으로 색상 정확성 보장)
        try:
            from PIL import Image
            pil_image = Image.open(face_path)
            face_image_rgb = np.array(pil_image)
            return success, message, face_image_rgb
        except Exception as e:
            logger.warning(f"PIL로 이미지 로드 실패: {e}")
            return success, message, None
    
    return success, message, None

def get_embedding_list_display():
    """
    embedding 목록을 표시용 문자열로 변환합니다.
    
    Returns:
        표시용 문자열
    """
    embeddings = face_manager.get_embedding_list()
    
    if not embeddings:
        return "등록된 embedding 파일이 없습니다."
    
    result = "📋 등록된 Embedding 파일 목록:\n\n"
    
    for i, emb in enumerate(embeddings, 1):
        result += f"**{i}. {emb['name']}**\n"
        result += f"   - JSON 파일: `{emb['json_file']}` ({emb['json_size']} bytes)\n"
        result += f"   - 이미지 파일: `{emb['image_file']}` ({emb['image_size']} bytes)\n"
        result += f"   - 경로: `{emb['json_path']}`\n"
        if emb['image_path']:
            result += f"   - 이미지 경로: `{emb['image_path']}`\n"
        result += "\n"
    
    return result

def get_embedding_gallery_data():
    """
    등록된 embedding 파일들의 갤러리 데이터를 반환합니다.
    
    Returns:
        갤러리용 데이터 리스트 [(이미지경로, 파일명), ...]
    """
    faces_dir = Path(face_manager.faces_path)
    if not faces_dir.exists():
        return []
    
    json_files = list(faces_dir.glob("*.json"))
    gallery_data = []
    
    for json_file in sorted(json_files):
        # .json 확장자 제거
        name = json_file.stem
        # 대응하는 이미지 파일 확인
        image_file = json_file.with_suffix('.jpg')
        if image_file.exists():
            # 이미지 파일 경로를 문자열로 변환
            image_path = str(image_file)
            gallery_data.append((image_path, name))
    
    return gallery_data

def get_embedding_choices():
    """
    드롭다운용 embedding 선택지 목록을 반환합니다.
    
    Returns:
        embedding 이름 리스트
    """
    # faces 폴더에서 직접 JSON 파일 목록을 읽어옵니다
    faces_dir = Path(face_manager.faces_path)
    if not faces_dir.exists():
        return []
    
    json_files = list(faces_dir.glob("*.json"))
    return [f.stem for f in json_files]  # .json 확장자 제거

def refresh_face_choices():
    """
    얼굴 목록을 새로고침하고 드롭다운을 업데이트합니다.
    
    Returns:
        업데이트된 드롭다운 설정
    """
    choices = get_embedding_choices()
    # 항상 첫 번째 항목을 선택하도록 강제
    return gr.update(choices=choices, value=choices[0] if choices else None)

def update_embedding_choices():
    """
    embedding 목록을 업데이트합니다 (기존 함수와 호환성 유지).
    
    Returns:
        업데이트된 드롭다운 설정
    """
    choices = get_embedding_choices()
    # 항상 첫 번째 항목을 선택하도록 강제
    return gr.update(choices=choices, value=choices[0] if choices else None)

def process_target_image(file_path):
    """
    타겟 이미지를 처리하고 얼굴 탐지 결과를 박스로 표시합니다.
    
    Args:
        file_path: 파일 경로
        
    Returns:
        (성공여부, 메시지, 박스가 그려진 이미지)
    """
    if file_path is None:
        return False, "이미지를 업로드해주세요.", None
    
    try:
        from PIL import Image
        
        # PIL로 이미지 로드
        pil_image = Image.open(file_path)
        image_rgb = np.array(pil_image)
        
        # RGB를 BGR로 변환 (얼굴 탐지용)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # 얼굴 탐지 및 박스 그리기
        success, message, result_image = face_manager.detect_and_draw_faces(image_bgr)
        
        if success:
            return True, message, result_image
        else:
            # 얼굴을 찾지 못한 경우 원본 이미지 반환
            return True, f"이미지 로드 완료\n{message}", image_rgb
        
    except Exception as e:
        logger.error(f"이미지 로드 실패: {e}")
        return False, f"이미지를 로드할 수 없습니다: {str(e)}", None

def perform_face_swap(file_path, face_indices, source_face_name):
    """
    얼굴 교체를 수행합니다.
    
    Args:
        file_path: 타겟 이미지 파일 경로
        face_indices: 교체할 얼굴 인덱스
        source_face_name: 소스 얼굴 이름
        
    Returns:
        (성공여부, 메시지, 교체된 이미지)
    """
    if file_path is None:
        return False, "타겟 이미지를 업로드해주세요.", None
    
    if not source_face_name:
        return False, "바꿀 얼굴을 선택해주세요.", None
    
    try:
        from PIL import Image
        
        # PIL로 이미지 로드
        pil_image = Image.open(file_path)
        image_rgb = np.array(pil_image)
        
        # RGB를 BGR로 변환 (얼굴 교체용)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # 얼굴 교체 수행
        success, message, result_image = face_manager.swap_faces(image_bgr, face_indices, source_face_name)
        
        return success, message, result_image
        
    except Exception as e:
        logger.error(f"얼굴 교체 실패: {e}")
        return False, f"얼굴 교체 실패: {str(e)}", None

# Gradio 인터페이스 생성
def create_interface():
    """Gradio 인터페이스를 생성합니다."""
    
    with gr.Blocks(title="Face Manager", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🎭 Face Manager")
        gr.Markdown("이미지에서 얼굴을 추출하고 embedding을 관리합니다.")
        
        with gr.Tab("얼굴 교체"):
            gr.Markdown("## 🔄 얼굴 교체")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # 타겟 이미지 업로드
                    target_upload = gr.Image(
                        label="타겟 이미지 업로드",
                        type="filepath"
                    )
                    
                    # 교체할 얼굴 인덱스 입력
                    face_indices_input = gr.Textbox(
                        label="교체할 얼굴 인덱스 (쉼표로 구분, 비워두면 모든 얼굴)",
                        placeholder="예: 1,3,5 또는 비워두기",
                        value=""
                    )
                    
                    # 바꿀 얼굴 선택 (드롭다운과 새로고침 버튼)
                    with gr.Row():
                        source_face_dropdown = gr.Dropdown(
                            label="바꿀 얼굴 선택",
                            choices=get_embedding_choices(),
                            value=get_embedding_choices()[0] if get_embedding_choices() else None,
                            scale=4
                        )
                        refresh_faces_btn = gr.Button(
                            "🔄", 
                            variant="secondary", 
                            size="sm",
                            scale=1
                        )
                    
                    # CodeFormer 복원 체크박스
                    codeformer_checkbox = gr.Checkbox(
                        label="CodeFormer 복원 포함",
                        value=True,
                        info="체크하면 얼굴 교체 후 자동으로 CodeFormer 복원도 수행됩니다"
                    )
                    
                    # 얼굴 교체 버튼
                    swap_btn = gr.Button("얼굴 변경", variant="primary")
                    
                    # 결과 메시지
                    swap_result_text = gr.Textbox(
                        label="결과",
                        lines=3,
                        interactive=False
                    )
                    
                    # 얼굴 교체된 이미지 State (CodeFormer용)
                    swapped_image_state = gr.State()
                
                with gr.Column(scale=1):
                    # 원본 이미지 표시
                    original_image = gr.Image(
                        label="원본 이미지",
                        type="numpy"
                    )
                    
                    # 최종 결과 이미지 표시
                    swapped_image = gr.Image(
                        label="최종 결과",
                        type="numpy"
                    )
        
        with gr.Tab("얼굴 추출"):
            gr.Markdown("## 📸 이미지에서 첫 번째 얼굴 추출")
            
            with gr.Row():
                with gr.Column(scale=1):
                    upload_input = gr.Image(
                        label="이미지 업로드",
                        type="filepath"
                    )
                    
                    process_btn = gr.Button("얼굴 추출", variant="primary")
                    
                    result_text = gr.Textbox(
                        label="결과",
                        lines=4,
                        interactive=False
                    )
                
                with gr.Column(scale=1):
                    extracted_face = gr.Image(
                        label="추출된 얼굴",
                        type="numpy"
                    )
        
        with gr.Tab("Embedding 목록"):
            gr.Markdown("## 📋 등록된 Embedding 파일 목록")
            
            refresh_btn = gr.Button("목록 새로고침", variant="secondary")
            
            embedding_gallery = gr.Gallery(
                value=get_embedding_gallery_data(),
                label="등록된 얼굴 이미지",
                show_label=True,
                elem_id="embedding_gallery",
                columns=4,
                rows=2,
                height="auto",
                allow_preview=True
            )
        
        # 이벤트 핸들러
        def process_image_wrapper(file_path):
            if file_path is None:
                return False, "파일을 선택해주세요.", None
            
            try:
                # 파일 경로를 Path 객체로 변환
                file_path_obj = Path(file_path)
                filename = file_path_obj.name
                
                # 한글 파일명 처리를 위해 numpy array로 직접 읽기
                import numpy as np
                from PIL import Image
                
                # PIL로 이미지 로드 (한글 경로 지원)
                pil_image = Image.open(file_path)
                # PIL Image를 numpy array로 변환 (RGB 유지)
                image_rgb = np.array(pil_image)
                
                # RGB를 BGR로 변환 (OpenCV는 BGR 사용)
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                
                return process_uploaded_image(image_bgr, filename, image_rgb)
                
            except Exception as e:
                logger.error(f"이미지 로드 실패: {e}")
                return False, f"이미지를 로드할 수 없습니다: {str(e)}", None
        
        process_btn.click(
            fn=process_image_wrapper,
            inputs=[upload_input],
            outputs=[gr.State(), result_text, extracted_face]
        )
        
        refresh_btn.click(
            fn=get_embedding_gallery_data,
            inputs=[],
            outputs=[embedding_gallery]
        )
        
        # 얼굴 교체 탭 이벤트 핸들러
        def update_embedding_choices():
            """embedding 선택지를 업데이트합니다."""
            choices = get_embedding_choices()
            return gr.Dropdown(choices=choices, value=choices[0] if choices else None)
        
        def process_target_image_wrapper(file_path):
            """타겟 이미지 처리 래퍼"""
            success, message, image = process_target_image(file_path)
            return success, message, image
        
        def perform_face_swap_wrapper(file_path, face_indices, source_face_name, use_codeformer):
            """얼굴 교체 + CodeFormer 통합 수행 래퍼"""
            final_image, message, result_image = perform_face_swap_with_optional_codeformer(
                file_path, face_indices, source_face_name, use_codeformer
            )
            return final_image, message, result_image  # 첫 번째는 State용, 세 번째는 표시용
        
        # 타겟 이미지 업로드 시 처리
        target_upload.change(
            fn=process_target_image_wrapper,
            inputs=[target_upload],
            outputs=[gr.State(), swap_result_text, original_image]
        )
        
        # 얼굴 교체 버튼 클릭 시 처리 (CodeFormer 포함)
        swap_btn.click(
            fn=perform_face_swap_wrapper,
            inputs=[target_upload, face_indices_input, source_face_dropdown, codeformer_checkbox],
            outputs=[swapped_image_state, swap_result_text, swapped_image]
        )
        
        # embedding 목록 새로고침 시 드롭다운도 업데이트
        refresh_btn.click(
            fn=update_embedding_choices,
            inputs=[],
            outputs=[source_face_dropdown]
        )
        
        # 얼굴 교체 탭의 새로고침 버튼 클릭 시 처리
        refresh_faces_btn.click(
            fn=refresh_face_choices,
            inputs=[],
            outputs=[source_face_dropdown]
        )
    
    return interface

def main():
    """메인 함수"""
    logger.info("Face Manager UI 시작")
    
    # 인터페이스 생성 및 실행
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )

if __name__ == "__main__":
    main()
