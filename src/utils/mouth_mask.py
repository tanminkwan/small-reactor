import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

def create_mouth_mask(landmarks, image_shape, expand_ratio=0.2, blur_size=0, 
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
    
    # 가우시안 블러로 부드러운 경계
    if blur_size > 0:
        if blur_size % 2 == 0:
            blur_size += 1
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    
    print(f"마스크 생성 완료 - 입 중심: {mouth_center}, 크기: {mouth_width}x{mouth_height}")
    print(f"타원 중심: ({new_center_x}, {new_center_y}), 반축: {semi_axis_x}x{semi_axis_y}")
    print(f"확장 설정 - scale_x:{scale_x}, scale_y:{scale_y}, offset_x:{offset_x}, offset_y:{offset_y}")
    
    return mask

if __name__ == "__main__":

    INSWAPPER_PATH = r"C:\models\inswapper_128.onnx"
    BUFFALO_L_PATH = "C:\\"

    source_img = cv2.imread("C:\\images\\f_base.jpg")
    target_img = cv2.imread("C:\\images\\tongue2.jpg")
    #target_img = cv2.imread("C:\\images\\b_1.jpg")

    app = FaceAnalysis(name='buffalo_l', root=BUFFALO_L_PATH)
    app.prepare(ctx_id=-1, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model(INSWAPPER_PATH)

    # 얼굴 감지
    source_faces = app.get(source_img)
    target_faces = app.get(target_img) 
    source_face = source_faces[0]
    target_face = target_faces[0]

    # 일반 스와핑 수행
    result = swapper.get(target_img, target_face, source_face, paste_back=True)

    # 여기서 입 마스크 적용
    landmarks = target_face.landmark_2d_106  # 또는 landmark_3d_68


    #mask = create_mouth_mask(landmarks, target_img.shape)
    #mask = create_mouth_mask(landmarks, target_img.shape, expand_weights={'scale_x': 4.0})
    mask = create_mouth_mask(landmarks, target_img.shape, expand_weights={'scale_x': 4.0, 'offset_x': 10, 'offset_y': 10})
    #

    # 입 부분만 원본으로 교체
    mouth_mask_bool = mask > 0
    result[mouth_mask_bool] = target_img[mouth_mask_bool]


    # 입 부분만 검은색으로 변경
    #mouth_mask_bool = mask == 0  # 마스크에서 0인 부분이 입 영역
    #result[mouth_mask_bool] = [0, 0, 0]  # BGR 형식으로 검은색 (0, 0, 0)


    # 결과 이미지를 파일로 저장
    output_filename = "f_result.jpg"
    cv2.imwrite(output_filename, result)