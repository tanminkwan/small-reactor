import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

def create_mouth_mask(landmarks, image_shape, expand_ratio=0.2, 
                     expand_weights={'scale_x': 1.0, 'scale_y': 1.0, 'offset_x': 0, 'offset_y': 0}):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)  # 기본을 0(검은색)으로 시작
    
    print(f"랜드마크 개수: {len(landmarks)}")
    
    # InsightFace 106 포인트에서 입 부분 인덱스
    if len(landmarks) == 106:
        # 106개 모델에서 입 부분: 52-71번 인덱스 (20개 포인트)
        mouth_points = landmarks[52:72]
    elif len(landmarks) == 68:
        # 68개 모델에서 입 부분: 48-67번 인덱스 
        mouth_points = landmarks[48:68]
    elif len(landmarks) == 5:
        # 5개 키포인트만 있는 경우 (kps): [왼쪽 눈, 오른쪽 눈, 코, 왼쪽 입, 오른쪽 입]
        # 입 부분을 대략적으로 추정
        left_mouth = landmarks[3]  # 왼쪽 입
        right_mouth = landmarks[4]  # 오른쪽 입
        nose = landmarks[2]  # 코
        
        # 입 중심점 계산
        mouth_center = (left_mouth + right_mouth) / 2
        
        # 입 크기 추정 (코에서 입까지의 거리 기반)
        nose_to_mouth = np.linalg.norm(nose - mouth_center)
        mouth_width = np.linalg.norm(right_mouth - left_mouth)
        
        # 타원형 마스크 생성
        center_x = int(mouth_center[0])
        center_y = int(mouth_center[1])
        
        # 타원 크기 계산
        semi_axis_x = int(mouth_width * 0.8)  # 입 너비의 80%
        semi_axis_y = int(nose_to_mouth * 0.6)  # 코-입 거리의 60%
        
        # expand_weights 적용
        scale_x = expand_weights.get('scale_x', 1.0)
        scale_y = expand_weights.get('scale_y', 1.0)
        offset_x = expand_weights.get('offset_x', 0)
        offset_y = expand_weights.get('offset_y', 0)
        
        # 최종 중심점과 크기
        final_center = (center_x + offset_x, center_y + offset_y)
        final_semi_axis_x = int(semi_axis_x * scale_x * (1 + expand_ratio))
        final_semi_axis_y = int(semi_axis_y * scale_y * (1 + expand_ratio))
        
        # 타원 마스크 생성
        cv2.ellipse(mask, final_center, (final_semi_axis_x, final_semi_axis_y), 0, 0, 360, 255, -1)
        
        print(f"5개 키포인트로 입 마스크 생성 - 중심: {final_center}, 크기: {final_semi_axis_x}x{final_semi_axis_y}")
        return mask
    else:
        print(f"지원되지 않는 랜드마크 개수: {len(landmarks)}")
        return mask
    
    # 디버깅: 입 부분 랜드마크 확인
    print(f"입 랜드마크 좌표 (처음 5개): {mouth_points[:5]}")
    
    # 입술 외곽선과 내부 분리
    if len(landmarks) == 106:
        # 외부 입술 윤곽 (12개 포인트)
        outer_mouth = landmarks[52:64]
        # 내부 입술 윤곽 (8개 포인트) 
        inner_mouth = landmarks[64:72]
    else:  # 68개 포인트
        outer_mouth = landmarks[48:60]  # 외부 12개
        inner_mouth = landmarks[60:68]  # 내부 8개
    
    # 외부 입술 윤곽으로 마스크 생성
    outer_contour = outer_mouth.astype(np.int32)
    cv2.fillPoly(mask, [outer_contour], 255)
    
    # 입 영역 확장 (expand_weights 적용)
    mouth_center = np.mean(outer_mouth, axis=0).astype(int)
    mouth_width = np.max(outer_mouth[:, 0]) - np.min(outer_mouth[:, 0])
    mouth_height = np.max(outer_mouth[:, 1]) - np.min(outer_mouth[:, 1])
    
    # 기본 확장 크기 계산
    base_expand_w = int(mouth_width * expand_ratio)
    base_expand_h = int(mouth_height * expand_ratio)
    
    # expand_weights에서 값 추출
    scale_x = expand_weights.get('scale_x', 1.0)
    scale_y = expand_weights.get('scale_y', 1.0)
    offset_x = expand_weights.get('offset_x', 0)
    offset_y = expand_weights.get('offset_y', 0)
    
    # 타원 중심 계산 (입 중심에서 offset만큼 이동)
    new_center_x = int(mouth_center[0] + offset_x)
    new_center_y = int(mouth_center[1] + offset_y)
    center = (new_center_x, new_center_y)
    
    # 타원의 반축 길이 계산 (scale 적용)
    semi_axis_x = int((mouth_width//2 + base_expand_w) * scale_x)
    semi_axis_y = int((mouth_height//2 + base_expand_h) * scale_y)
    
    axes = (semi_axis_x, semi_axis_y)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    
    print(f"마스크 생성 완료 - 입 중심: {mouth_center}, 크기: {mouth_width}x{mouth_height}")
    print(f"타원 중심: ({new_center_x}, {new_center_y}), 반축: {semi_axis_x}x{semi_axis_y}")
    print(f"확장 설정 - scale_x:{scale_x}, scale_y:{scale_y}, offset_x:{offset_x}, offset_y:{offset_y}")
    
    return mask

def smooth_blend_mouth(result_img, target_img, mask, blend_mode="poisson"):
    """
    입 영역을 자연스럽게 블렌딩하는 함수
    
    Args:
        result_img: 얼굴교체 결과 이미지
        target_img: 원본 타겟 이미지
        mask: 입 마스크
        blend_mode: 블렌딩 방법 ("gaussian", "feather", "poisson")
    """
    
    if blend_mode == "gaussian":
        # 방법 1: 가우시안 블러로 부드러운 경계
        # 마스크를 0-1 범위로 정규화
        mask_smooth = cv2.GaussianBlur(mask, (21, 21), 0)

        mask_normalized = mask_smooth.astype(np.float32) / 255.0
        
        # 3채널로 확장
        mask_3d = np.repeat(mask_normalized[:, :, np.newaxis], 3, axis=2)
        
        # 가중 평균으로 블렌딩
        blended = result_img * (1 - mask_3d) + target_img * mask_3d
        
        return blended.astype(np.uint8)
    
    elif blend_mode == "poisson":
        # 방법 2: Poisson 블렌딩 (가장 자연스럽지만 계산이 오래 걸림)
        try:
            # 마스크에서 중심점 찾기
            moments = cv2.moments(mask)
            if moments['m00'] != 0:
                center_x = int(moments['m10'] / moments['m00'])
                center_y = int(moments['m01'] / moments['m00'])
                center = (center_x, center_y)
                
                # Poisson 블렌딩
                blended = cv2.seamlessClone(target_img, result_img, mask, center, cv2.NORMAL_CLONE)
                return blended
            else:
                print("Poisson 블렌딩 실패 - 중심점 찾기 실패")
                # fallback to gaussian
                return smooth_blend_mouth(result_img, target_img, mask, "gaussian")
        except:
            print("Poisson 블렌딩 실패 - 예외 발생")
            # Poisson 블렌딩 실패 시 gaussian으로 fallback
            return smooth_blend_mouth(result_img, target_img, mask, "gaussian")


if __name__ == "__main__":
    """
    테스트 코드 - 실제 사용 시에는 제거하거나 별도 테스트 파일로 분리
    """
    print("mouth_mask.py 모듈이 정상적으로 로드되었습니다.")
    print("create_mouth_mask 함수를 사용하여 입 마스크를 생성할 수 있습니다.")