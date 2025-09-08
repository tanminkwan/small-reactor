#!/usr/bin/env python3
"""
입 원본유지 기능 테스트 스크립트
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

from src.services.buffalo_detector import BuffaloDetector
from src.utils.config import Config
from src.utils.mouth_mask import create_mouth_mask

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mouth_preserve():
    """입 원본유지 기능 테스트"""
    
    # 설정 초기화
    config = Config()
    detector = BuffaloDetector(config)
    
    # 테스트 이미지 경로 (실제 이미지 경로로 변경 필요)
    test_image_path = "test_image.jpg"  # 실제 테스트 이미지 경로로 변경
    
    if not Path(test_image_path).exists():
        logger.error(f"테스트 이미지를 찾을 수 없습니다: {test_image_path}")
        return False
    
    try:
        # 이미지 로드
        image = cv2.imread(test_image_path)
        if image is None:
            logger.error("이미지를 로드할 수 없습니다")
            return False
        
        # 얼굴 탐지
        faces = detector.detect_faces(image)
        if not faces:
            logger.error("얼굴을 찾을 수 없습니다")
            return False
        
        logger.info(f"탐지된 얼굴 수: {len(faces)}")
        
        # 첫 번째 얼굴로 테스트
        face = faces[0]
        
        # 랜드마크 확인
        landmarks_106 = getattr(face, 'landmark_2d_106', None)
        landmarks_68 = getattr(face, 'landmark_3d_68', None)
        
        if landmarks_106 is not None:
            landmarks = landmarks_106
            logger.info("106개 랜드마크 사용")
        elif landmarks_68 is not None:
            landmarks = landmarks_68
            logger.info("68개 랜드마크 사용")
        else:
            logger.error("랜드마크를 찾을 수 없습니다")
            return False
        
        # 입 마스크 생성 테스트
        logger.info("입 마스크 생성 테스트 시작...")
        
        # 기본 설정으로 마스크 생성
        mask_basic = create_mouth_mask(landmarks, image.shape)
        logger.info("기본 설정 마스크 생성 완료")
        
        # 커스텀 설정으로 마스크 생성
        custom_settings = {
            'expand_ratio': 0.3,
            'scale_x': 1.5,
            'scale_y': 1.2,
            'offset_x': 5,
            'offset_y': -5
        }
        
        mask_custom = create_mouth_mask(
            landmarks, 
            image.shape, 
            expand_ratio=custom_settings['expand_ratio'],
            expand_weights={
                'scale_x': custom_settings['scale_x'],
                'scale_y': custom_settings['scale_y'],
                'offset_x': custom_settings['offset_x'],
                'offset_y': custom_settings['offset_y']
            }
        )
        logger.info("커스텀 설정 마스크 생성 완료")
        
        # 마스크 시각화
        mask_visualization = np.zeros_like(image)
        mask_visualization[mask_basic > 0] = [0, 255, 0]  # 기본 마스크를 초록색으로
        mask_visualization[mask_custom > 0] = [0, 0, 255]  # 커스텀 마스크를 빨간색으로
        
        # 결과 저장
        output_dir = Path("test_outputs")
        output_dir.mkdir(exist_ok=True)
        
        cv2.imwrite(str(output_dir / "original_image.jpg"), image)
        cv2.imwrite(str(output_dir / "mouth_mask_basic.jpg"), mask_basic)
        cv2.imwrite(str(output_dir / "mouth_mask_custom.jpg"), mask_custom)
        cv2.imwrite(str(output_dir / "mask_visualization.jpg"), mask_visualization)
        
        logger.info("테스트 완료! 결과 이미지가 test_outputs 폴더에 저장되었습니다.")
        logger.info("- original_image.jpg: 원본 이미지")
        logger.info("- mouth_mask_basic.jpg: 기본 설정 마스크")
        logger.info("- mouth_mask_custom.jpg: 커스텀 설정 마스크")
        logger.info("- mask_visualization.jpg: 마스크 시각화")
        
        return True
        
    except Exception as e:
        logger.error(f"테스트 실패: {e}")
        return False

if __name__ == "__main__":
    print("입 원본유지 기능 테스트를 시작합니다...")
    success = test_mouth_preserve()
    
    if success:
        print("✅ 테스트가 성공적으로 완료되었습니다!")
    else:
        print("❌ 테스트가 실패했습니다.")
