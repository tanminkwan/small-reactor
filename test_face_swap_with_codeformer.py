#!/usr/bin/env python3
"""
Face Swap + CodeFormer Enhancement 통합 테스트 프로그램

기능:
1. sosi.jpg에서 얼굴을 찾아
2. face.jpg에서 얼굴을 추출해 embedding 값을 저장해
3. face.jpg에서 추출한 얼굴 embedding으로 sosi.jpg에서 찾은 얼굴들을 바꿔버려
4. 바뀐 얼굴 범위를 조금 확장한 후 해당 영역을 codeformer-v0.1.0.pth로 복원

사용법:
    python test_face_swap_with_codeformer.py

결과 파일:
    - face_embedding.json: 얼굴 임베딩 데이터
    - swapped_result.jpg: 얼굴 교체 결과
    - enhanced_result.jpg: 얼굴 영역 복원 결과
    - full_enhanced_result.jpg: 전체 이미지 복원 결과
"""

import numpy as np
import cv2
import logging
import json
from pathlib import Path
from dotenv import load_dotenv

from src.services.buffalo_detector import BuffaloDetector
from src.utils.config import Config

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_image(image_path: Path) -> np.ndarray:
    """이미지를 로드합니다."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return image

def save_image(image: np.ndarray, output_path: Path) -> None:
    """이미지를 저장합니다."""
    success = cv2.imwrite(str(output_path), image)
    if not success:
        raise RuntimeError(f"Failed to save image: {output_path}")

def save_embedding(embedding: np.ndarray, output_path: Path) -> None:
    """임베딩을 JSON 파일로 저장합니다."""
    embedding_data = {
        "embedding": embedding.tolist(),
        "shape": embedding.shape,
        "dtype": str(embedding.dtype)
    }
    
    with open(output_path, 'w') as f:
        json.dump(embedding_data, f, indent=2)

def expand_face_regions(image: np.ndarray, face_regions: list, expansion_factor: float = 0.2) -> list:
    """
    얼굴 영역을 확장합니다.
    
    Args:
        image: 입력 이미지
        face_regions: 얼굴 영역 리스트 [(x1, y1, x2, y2), ...]
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

def main():
    """메인 테스트 함수"""
    load_dotenv()
    
    logger.info("=== Face Swap + CodeFormer Enhancement 통합 테스트 시작 ===")
    
    # 설정 초기화
    config = Config()
    
    # 서비스 초기화
    detector_service = BuffaloDetector(config)
    
    # 1. 이미지 로드
    face_image_path = Path("tests/images/face.jpg")
    sosi_image_path = Path("tests/images/sosi.jpg")
    
    face_image = load_image(face_image_path)
    sosi_image = load_image(sosi_image_path)
    
    logger.info(f"face.jpg 로드 완료: {face_image.shape}")
    logger.info(f"sosi.jpg 로드 완료: {sosi_image.shape}")
    
    # 2. face.jpg에서 얼굴 추출 및 임베딩 저장
    logger.info("\n=== face.jpg에서 얼굴 추출 및 임베딩 저장 ===")
    source_faces = detector_service.detect_faces(face_image)
    
    if not source_faces:
        logger.error("face.jpg에서 얼굴을 찾을 수 없습니다.")
        return
    
    source_face = source_faces[0]
    logger.info(f"얼굴 탐지 완료: bbox={source_face.bbox}, det_score={source_face.det_score}")
    
    # 임베딩 저장
    embedding_path = Path("face_embedding.json")
    save_embedding(source_face.embedding, embedding_path)
    logger.info(f"임베딩 저장 완료: {embedding_path}")
    
    # 3. sosi.jpg에서 얼굴 찾기
    logger.info("\n=== sosi.jpg에서 얼굴 찾기 ===")
    target_faces = detector_service.detect_faces(sosi_image)
    
    if not target_faces:
        logger.error("sosi.jpg에서 얼굴을 찾을 수 없습니다.")
        return
    
    logger.info(f"얼굴 {len(target_faces)}개 탐지 완료")
    for i, face in enumerate(target_faces):
        logger.info(f"  얼굴 {i+1}: bbox={face.bbox}, det_score={face.det_score}")
    
    # 4. 성공했던 방식으로 얼굴 교체
    logger.info("\n=== 얼굴 교체 수행 ===")
    
    # InsightFace model_zoo 사용 (성공했던 방식)
    from insightface import model_zoo
    from insightface.app.common import Face
    
    # swapper 모델 로드
    swapper_model_path = config.get_model_path("inswapper")
    swapper = model_zoo.get_model(swapper_model_path)
    
    # source_face를 InsightFace의 Face 객체로 변환
    face = Face()
    face.embedding = source_face.embedding
    logger.info("Face 객체 생성 완료")
    
    # 각 얼굴에 대해 교체 수행
    swapped_image = sosi_image.copy()
    
    for i, target_face in enumerate(target_faces):
        logger.info(f"얼굴 {i+1} 교체 중...")
        swapped_image = swapper.get(swapped_image, target_face, face)
        logger.info(f"얼굴 {i+1} 교체 완료")
    
    # 교체된 이미지 저장
    swapped_path = Path("swapped_result.jpg")
    save_image(swapped_image, swapped_path)
    logger.info(f"얼굴 교체 완료! 결과 저장: {swapped_path}")
    
    # 5. 얼굴 영역 확장 및 CodeFormer로 복원
    logger.info("\n=== 얼굴 영역 확장 및 CodeFormer 복원 ===")
    
    # 교체된 얼굴들의 bbox를 리스트로 변환
    face_regions = []
    for target_face in target_faces:
        bbox = target_face.bbox
        x1, y1, x2, y2 = map(int, bbox)
        face_regions.append((x1, y1, x2, y2))
    
    logger.info(f"얼굴 영역 {len(face_regions)}개를 확장하여 복원합니다.")
    
    # 얼굴 영역 확장
    expanded_regions = expand_face_regions(swapped_image, face_regions, expansion_factor=0.2)
    
    # 제공된 코드를 사용한 FaceRestorer 클래스
    import torch
    from torchvision.transforms.functional import normalize
    from codeformer.basicsr.utils import img2tensor, tensor2img
    from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper
    from codeformer.basicsr.utils.registry import ARCH_REGISTRY
    
    class FaceRestorer:
        """
        얼굴 복원을 위한 클래스
        한 번 초기화하고 여러 이미지에 재사용 가능
        """
        def __init__(self, model_path, use_gpu=False):
            """
            FaceRestorer 클래스 초기화
            
            :param model_path: CodeFormer 모델 파일 경로
            :param use_gpu: GPU 사용 여부 (기본값: False)
            """
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
            
            print(f"FaceRestorer initialized using device: {self.device}")
        
        def restore(self, input_image, w=0.5, only_center_face=False):
            """
            입력 이미지의 얼굴을 복원
            
            :param input_image: ndarray 타입의 입력 이미지
            :param w: 복원 강도 가중치 (0~1, 기본값: 0.5)
            :param only_center_face: 중앙 얼굴만 처리할지 여부
            :return: ndarray 타입의 복원된 이미지
            """
            # 이미지 크기 저장
            h, w_img, _ = input_image.shape
            
            # Face Helper 초기화
            self.face_helper.clean_all()
            self.face_helper.read_image(input_image)
            self.face_helper.get_face_landmarks_5(
                only_center_face=only_center_face, 
                resize=512, 
                eye_dist_threshold=5
            )
            self.face_helper.align_warp_face()
            
            # 얼굴 복원
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
                    print(f'Error: {error}')
                    print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
                else:
                    restored_face = restored_face.astype('uint8')
                    self.face_helper.add_restored_face(restored_face)
            
            # 결과 생성
            self.face_helper.get_inverse_affine(None)
            restored_img = self.face_helper.paste_faces_to_input_image()
            
            # 최종 이미지 크기 조정 (원본 크기로)
            restored_img = cv2.resize(restored_img, (w_img, h))
            
            return restored_img
        
        def __del__(self):
            """소멸자: 리소스 정리"""
            if self.use_gpu:
                torch.cuda.empty_cache()
    
    # FaceRestorer 초기화
    codeformer_model_path = config.get_model_path("codeformer")
    restorer = FaceRestorer(model_path=codeformer_model_path, use_gpu=True)
    
    # 각 확장된 얼굴 영역에 대해 복원 수행
    enhanced_image = swapped_image.copy()
    
    for i, region in enumerate(expanded_regions):
        x1, y1, x2, y2 = region
        logger.info(f"얼굴 영역 {i+1} 복원 중... (확장된 영역: {region})")
        
        # 영역 추출
        face_crop = swapped_image[y1:y2, x1:x2]
        
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
    
    # 최종 결과 저장
    final_path = Path("enhanced_result.jpg")
    save_image(enhanced_image, final_path)
    logger.info(f"얼굴 복원 완료! 최종 결과 저장: {final_path}")
    
    # 6. 전체 이미지 복원 (선택사항)
    logger.info("\n=== 전체 이미지 복원 (선택사항) ===")
    try:
        full_enhanced_image = restorer.restore(swapped_image, w=0.3)
        full_enhanced_path = Path("full_enhanced_result.jpg")
        save_image(full_enhanced_image, full_enhanced_path)
        logger.info(f"전체 이미지 복원 완료! 결과 저장: {full_enhanced_path}")
    except Exception as e:
        logger.warning(f"전체 이미지 복원 실패: {e}")
    
    logger.info("\n=== 통합 테스트 완료 ===")
    logger.info("생성된 파일들:")
    logger.info(f"  - {embedding_path}: 얼굴 임베딩")
    logger.info(f"  - {swapped_path}: 얼굴 교체 결과")
    logger.info(f"  - {final_path}: 얼굴 영역 복원 결과")
    logger.info(f"  - full_enhanced_result.jpg: 전체 이미지 복원 결과 (선택사항)")

if __name__ == "__main__":
    main()
