"""
Face Manager 핵심 비즈니스 로직

Single Responsibility Principle (SRP)에 따라
얼굴 관련 비즈니스 로직만을 담당하는 서비스
"""

import logging
import numpy as np
import cv2
import json
from pathlib import Path
from typing import List, Tuple, Optional

from src.services.buffalo_detector import BuffaloDetector
from src.services.codeformer_enhancer import CodeFormerEnhancer
from src.utils.config import Config
from src.utils.mouth_mask import create_mouth_mask, smooth_blend_mouth


class FaceManager:
    """얼굴 관리 핵심 비즈니스 로직"""
    
    def __init__(self, config: Config):
        """
        초기화
        
        Args:
            config: 설정 객체
        """
        self.config = config
        self.detector = BuffaloDetector(config)
        self.enhancer = CodeFormerEnhancer(config)
        
        self._logger = logging.getLogger(__name__)
    
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
            
            # 인덱스 텍스트 그리기 (박스 안쪽에 큰 폰트로)
            label = f"{i+1}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # 박스 크기에 비례한 폰트 크기 계산
            box_width = x2 - x1
            box_height = y2 - y1
            font_scale = min(box_width, box_height) / 100.0  # 박스 크기에 비례
            font_scale = max(1.5, min(font_scale, 4.0))  # 최소 1.5, 최대 4.0
            
            thickness = max(2, int(font_scale))  # 폰트 크기에 비례한 두께
            
            # 텍스트 크기 계산
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # 박스 안쪽 중앙에 텍스트 배치
            text_x = x1 + (box_width - text_width) // 2
            text_y = y1 + (box_height + text_height) // 2
            
            # 텍스트 배경 박스 그리기 (반투명한 흰색)
            padding = 5
            cv2.rectangle(result_image, 
                         (text_x - padding, text_y - text_height - padding), 
                         (text_x + text_width + padding, text_y + padding), 
                         (255, 255, 255), -1)
            
            # 텍스트 그리기 (검은색)
            cv2.putText(result_image, label, 
                       (text_x, text_y), 
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
            self._logger.error(f"얼굴 탐지 실패: {e}")
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
            face_image_path = Path("./faces") / f"{base_name}.jpg"
            embedding_path = Path("./faces") / f"{base_name}.json"
            
            # 디렉토리 생성
            face_image_path.parent.mkdir(exist_ok=True)
            
            # 이미지 저장 (PIL 사용으로 색상 정확성 보장)
            from PIL import Image
            pil_image = Image.fromarray(face_image_rgb)
            pil_image.save(str(face_image_path))
            
            # embedding 저장
            with open(embedding_path, 'w') as f:
                json.dump(first_face.embedding.tolist(), f)
            
            self._logger.info(f"얼굴 추출 완료: {face_image_path}, {embedding_path}")
            
            return True, f"얼굴 추출 완료!\n저장된 파일:\n- {face_image_path.name}\n- {embedding_path.name}", str(face_image_path)
            
        except Exception as e:
            self._logger.error(f"얼굴 추출 실패: {e}")
            return False, f"얼굴 추출 실패: {str(e)}", ""
    
    def swap_faces(self, target_image: np.ndarray, face_indices: str, source_face_name: str, faces_dir: Path) -> Tuple[bool, str, np.ndarray]:
        """
        타겟 이미지의 얼굴들을 소스 얼굴로 교체합니다.
        
        Args:
            target_image: 타겟 이미지 (BGR)
            face_indices: 교체할 얼굴 인덱스 (쉼표로 구분, 비워두면 모든 얼굴)
            source_face_name: 소스 얼굴 이름
            faces_dir: 얼굴 파일들이 저장된 디렉토리
            
        Returns:
            (성공여부, 메시지, 교체된 이미지)
        """
        try:
            # source_face_name이 None이거나 빈 문자열인 경우 첫 번째 저장된 얼굴 사용
            if not source_face_name or source_face_name.strip() == "":
                # faces 디렉토리에서 첫 번째 .json 파일 찾기
                json_files = list(faces_dir.glob("*.json"))
                if not json_files:
                    return False, "저장된 얼굴이 없습니다. 먼저 얼굴을 추출해주세요.", None
                source_face_name = json_files[0].stem  # .json 확장자 제거
            
            # 소스 얼굴 embedding 로드
            source_embedding_path = faces_dir / f"{source_face_name}.json"
            if not source_embedding_path.exists():
                return False, f"소스 얼굴 파일을 찾을 수 없습니다: {source_face_name}", None
            
            with open(source_embedding_path, 'r') as f:
                source_embedding = np.array(json.load(f))
            
            # 타겟 이미지에서 얼굴 탐지
            target_faces = self.detector.detect_faces(target_image)
            if not target_faces:
                return False, "타겟 이미지에서 얼굴을 찾을 수 없습니다.", None
            
            # 얼굴 인덱스와 위치 정보 로그 출력
            self._logger.info(f"탐지된 얼굴 수: {len(target_faces)}")
            for i, face in enumerate(target_faces):
                bbox = face.bbox
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                self._logger.info(f"얼굴 {i+1}: 중심점 ({center_x:.1f}, {center_y:.1f}), bbox ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
            
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
                    self._logger.error(f"Failed to swap face: {e}")
                    continue
            
            # BGR을 RGB로 변환
            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            
            message = f"얼굴 교체 완료!\n교체된 얼굴: {len(indices)}개\n소스 얼굴: {source_face_name}"
            
            return True, message, result_image_rgb
            
        except Exception as e:
            self._logger.error(f"얼굴 교체 실패: {e}")
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
            self._logger.info(f"CodeFormer 복원 - 탐지된 얼굴 수: {len(faces)}")
            for i, face in enumerate(faces):
                bbox = face.bbox
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                self._logger.info(f"얼굴 {i+1}: 중심점 ({center_x:.1f}, {center_y:.1f}), bbox ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
            
            # 얼굴 인덱스 파싱
            if face_indices.strip():
                indices = [int(x.strip()) - 1 for x in face_indices.split(',')]
                indices = [i for i in indices if 0 <= i < len(faces)]
                if not indices:
                    return False, "유효한 얼굴 인덱스가 없습니다.", None
            else:
                indices = list(range(len(faces)))
            
            # 선택된 얼굴들의 bbox를 리스트로 변환
            face_regions = []
            for i in indices:
                bbox = faces[i].bbox
                x1, y1, x2, y2 = map(int, bbox)
                face_regions.append((x1, y1, x2, y2))
            
            # CodeFormerEnhancer를 사용하여 얼굴 영역 복원
            enhanced_image = self.enhancer.enhance_face_regions(
                image, 
                face_regions, 
                enhancement_strength=0.5
            )
            
            # BGR을 RGB로 변환
            enhanced_image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
            
            message = f"CodeFormer 복원 완료!\n복원된 얼굴: {len(indices)}개"
            
            return True, message, enhanced_image_rgb
            
        except Exception as e:
            self._logger.error(f"CodeFormer 복원 실패: {e}")
            return False, f"CodeFormer 복원 실패: {str(e)}", None
    
    def apply_mouth_preservation(self, processed_image: np.ndarray, original_image: np.ndarray, face_indices: str, mouth_settings: dict) -> Tuple[bool, str, np.ndarray]:
        """
        CodeFormer 복원 후에 입 원본유지를 적용합니다.
        
        Args:
            processed_image: 처리된 이미지 (BGR)
            original_image: 원본 이미지 (BGR)
            face_indices: 얼굴 인덱스
            mouth_settings: 입 마스크 설정
            
        Returns:
            (성공여부, 메시지, 입 원본유지가 적용된 이미지)
        """
        try:
            # 얼굴 탐지
            faces = self.detector.detect_faces(processed_image)
            if not faces:
                return False, "이미지에서 얼굴을 찾을 수 없습니다.", None
            
            # 얼굴 인덱스 파싱
            if face_indices.strip():
                indices = [int(x.strip()) - 1 for x in face_indices.split(',')]
                indices = [i for i in indices if 0 <= i < len(faces)]
                if not indices:
                    return False, "유효한 얼굴 인덱스가 없습니다.", None
            else:
                indices = list(range(len(faces)))
            
            # 기본 설정값
            expand_ratio = mouth_settings.get('expand_ratio', 0.2)
            expand_weights = {
                'scale_x': mouth_settings.get('scale_x', 1.0),
                'scale_y': mouth_settings.get('scale_y', 1.0),
                'offset_x': mouth_settings.get('offset_x', 0),
                'offset_y': mouth_settings.get('offset_y', 0)
            }
            
            # 결과 이미지 초기화
            result_image = processed_image.copy()
            
            # 선택된 얼굴들에 대해 입 마스크 적용
            for i, face_idx in enumerate(indices):
                try:
                    face = faces[face_idx]
                    
                    # InsightFace Face 객체에서 랜드마크 가져오기
                    landmarks = getattr(face, 'landmark_2d_106', None)
                    
                    # landmark_2d_106이 없으면 landmark_3d_68 시도
                    if landmarks is None:
                        landmarks = getattr(face, 'landmark_3d_68', None)
                        if landmarks is not None:
                            self._logger.info(f"얼굴 {i+1}에서 landmark_3d_68 사용 (포인트 수: {len(landmarks)})")
                    
                    # 여전히 없으면 kps 시도 (5개 포인트)
                    if landmarks is None:
                        landmarks = getattr(face, 'kps', None)
                        if landmarks is not None:
                            self._logger.info(f"얼굴 {i+1}에서 kps 사용 (포인트 수: {len(landmarks)})")
                    
                    if landmarks is not None and len(landmarks) >= 5:
                        # 입 마스크 생성
                        mouth_mask = create_mouth_mask(
                            landmarks, 
                            result_image.shape, 
                            expand_ratio=expand_ratio,
                            expand_weights=expand_weights
                        )

                        # 입 부분을 원본으로 복원
                        #mouth_mask_bool = mouth_mask > 0
                        #result_image[mouth_mask_bool] = original_image[mouth_mask_bool]
                       
                        result_image_poisson = smooth_blend_mouth(result_image, original_image, mouth_mask, "poisson")  # 원본 마스크 사용
                        
                        self._logger.info(f"얼굴 {i+1} 입 원본유지 적용 완료 (랜드마크 수: {len(landmarks)})")
                    else:
                        self._logger.warning(f"얼굴 {i+1}의 충분한 랜드마크를 찾을 수 없습니다 (현재: {len(landmarks) if landmarks is not None else 0}개)")
                        
                except Exception as e:
                    self._logger.error(f"얼굴 {i+1} 입 원본유지 적용 실패: {e}")
                    continue
            
            # BGR을 RGB로 변환
            result_image_rgb = cv2.cvtColor(result_image if result_image_poisson is None else result_image_poisson, cv2.COLOR_BGR2RGB)
            
            message = f"입 원본유지 적용 완료!\n적용된 얼굴: {len(indices)}개"
            
            return True, message, result_image_rgb
            
        except Exception as e:
            self._logger.error(f"입 원본유지 적용 실패: {e}")
            return False, f"입 원본유지 적용 실패: {str(e)}", None
    
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
