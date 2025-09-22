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

def create_mouth_mask2(landmarks, image_shape):
    """
    직선 C와 보라색 얼굴 턱쪽 테두리가 만나는 구간을 마스킹하는 함수
    
    Args:
        landmarks: InsightFace 106 포인트 랜드마크
        image_shape: 이미지 크기 (height, width, channels)
    
    Returns:
        mask: 마스크 이미지 (0-255)
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 1. 직선 A: 인덱스 39와 89 사이
    point_39 = landmarks[39]
    point_89 = landmarks[89]
    
    # 직선 A의 기울기 계산
    if point_89[0] != point_39[0]:
        slope_A = (point_89[1] - point_39[1]) / (point_89[0] - point_39[0])
    else:
        slope_A = 0
    
    # 2. 점 B: 인덱스 80과 71 사이의 중간점
    point_80 = landmarks[80]
    point_71 = landmarks[71]
    point_B = ((point_80[0] + point_71[0]) / 2, (point_80[1] + point_71[1]) / 2)
    
    # 3. 직선 C: 점 B를 지나고 직선 A와 같은 기울기
    # 직선 C의 방정식: y = slope_A * (x - x_B) + y_B
    
    # 4. 턱선과의 교점 찾기 (정확히 두 개만)
    jaw_idx = list(range(0, 34))  # 턱선
    jaw_points = landmarks[jaw_idx]
    
    # 직선 C와 턱선의 교점을 찾기
    intersection_points = []
    
    # 턱선의 각 세그먼트에서 교점을 찾되, 유효한 교점만 선택
    for i in range(len(jaw_points) - 1):
        p1 = jaw_points[i]
        p2 = jaw_points[i + 1]
        
        # 턱선 세그먼트와 직선 C의 교점 계산
        if p2[0] != p1[0]:
            m_jaw = (p2[1] - p1[1]) / (p2[0] - p1[0])
            
            # 교점 계산
            if abs(slope_A - m_jaw) > 1e-6:
                x_intersect = (slope_A * point_B[0] - point_B[1] - m_jaw * p1[0] + p1[1]) / (slope_A - m_jaw)
                y_intersect = slope_A * (x_intersect - point_B[0]) + point_B[1]
                
                # 교점이 세그먼트 범위 내에 있는지 확인
                if min(p1[0], p2[0]) <= x_intersect <= max(p1[0], p2[0]):
                    intersection_points.append((x_intersect, y_intersect))
    
    # 교점이 2개보다 많으면 가장 왼쪽과 오른쪽 교점만 선택
    if len(intersection_points) > 2:
        # x좌표로 정렬
        intersection_points.sort(key=lambda p: p[0])
        # 가장 왼쪽과 오른쪽 교점만 선택
        intersection_points = [intersection_points[0], intersection_points[-1]]
    elif len(intersection_points) < 2:
        # 교점이 2개 미만이면 턱선의 양 끝점을 사용
        left_jaw = jaw_points[0]
        right_jaw = jaw_points[-1]
        intersection_points = [(left_jaw[0], left_jaw[1]), (right_jaw[0], right_jaw[1])]
    
    # 5. 직선 C 아래 얼굴의 윤곽선 그리기
    if len(intersection_points) == 2:
        # 교점들을 x좌표로 정렬
        intersection_points.sort(key=lambda p: p[0])
        left_intersect = intersection_points[0]
        right_intersect = intersection_points[1]
        
        # 직선 C 아래에 있는 턱선 포인트들 찾기
        jaw_points_below_c = []
        for i in range(0, 34):  # 턱선 전체
            point = landmarks[i]
            y_on_line = slope_A * (point[0] - point_B[0]) + point_B[1]
            if point[1] >= y_on_line:  # 직선 C 아래 또는 위에 있는 점
                jaw_points_below_c.append((int(point[0]), int(point[1])))
        
        # 윤곽선 포인트들 구성
        outline_points = []
        
        # 1. 왼쪽 교점
        outline_points.append((int(left_intersect[0]), int(left_intersect[1])))
        
        # 2. 왼쪽 교점과 오른쪽 교점 사이의 턱선 포인트들 (x좌표 순으로)
        middle_points = []
        for x, y in jaw_points_below_c:
            if left_intersect[0] < x < right_intersect[0]:
                middle_points.append((x, y))
        
        # x좌표로 정렬
        middle_points.sort(key=lambda p: p[0])
        outline_points.extend(middle_points)
        
        # 3. 오른쪽 교점
        outline_points.append((int(right_intersect[0]), int(right_intersect[1])))
        
        # 6. 직선 C와 보라색 둘레로 감싸진 영역을 마스킹
        if len(outline_points) > 1:
            # 직선 C의 양 끝점 계산 (교점들 사이의 범위만 사용)
            x_left = left_intersect[0]
            y_left = left_intersect[1]
            x_right = right_intersect[0]
            y_right = right_intersect[1]
            
            # 마스킹할 폴리곤 포인트들 생성
            mask_points = []
            
            # 1. 직선 C의 왼쪽 끝점부터 시작
            mask_points.append((int(x_left), int(y_left)))
            
            # 2. 직선 C를 따라 오른쪽 끝점까지
            mask_points.append((int(x_right), int(y_right)))
            
            # 3. 보라색 윤곽선을 역순으로 따라 (오른쪽에서 왼쪽으로)
            for point in reversed(outline_points):
                # 각 포인트도 이미지 경계 내로 제한
                x, y = point
                x = max(0, min(w - 1, x))
                y = max(0, min(h - 1, y))
                mask_points.append((int(x), int(y)))
            
            # 4. 직선 C의 왼쪽 끝점으로 돌아가기
            mask_points.append((int(x_left), int(y_left)))
            
            # 마스킹 실행
            mask_points_array = np.array(mask_points, dtype=np.int32)
            cv2.fillPoly(mask, [mask_points_array], 255)
            
            # 마스크 가장자리를 부드럽게 만들기 (더 강한 블러 적용)
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
            
            # 마스크 경계를 더 부드럽게 만들기 위해 모폴로지 연산 적용
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.GaussianBlur(mask, (9, 9), 0)
            
            # 마스크가 단일 연결된 영역인지 확인
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 1:
                print(f"경고: 마스크가 {len(contours)}개의 분리된 영역으로 구성됨")
                # 가장 큰 컨투어만 사용
                largest_contour = max(contours, key=cv2.contourArea)
                mask = np.zeros_like(mask)
                cv2.fillPoly(mask, [largest_contour], 255)
                mask = cv2.GaussianBlur(mask, (9, 9), 0)
                print("가장 큰 영역만 사용하여 마스크 재생성")
            
            # 마스크의 볼록 껍질(Convex Hull) 계산하여 불규칙한 모양 개선
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                # 볼록 껍질 계산
                hull = cv2.convexHull(largest_contour)
                
                # 원본 마스크와 볼록 껍질의 차이 확인
                original_area = cv2.contourArea(largest_contour)
                hull_area = cv2.contourArea(hull)
                convexity_ratio = original_area / hull_area if hull_area > 0 else 1.0
                
                print(f"마스크 볼록성 비율: {convexity_ratio:.3f} (1.0에 가까울수록 볼록함)")
                
                # 볼록성 비율이 너무 낮으면 (불규칙하면) 볼록 껍질 사용
                if convexity_ratio < 0.7:
                    print("마스크가 너무 불규칙하여 볼록 껍질로 대체")
                    mask = np.zeros_like(mask)
                    cv2.fillPoly(mask, [hull], 255)
                    mask = cv2.GaussianBlur(mask, (15, 15), 0)
            
            print(f"create_mouth_mask2: 마스크 생성 완료 - {len(outline_points)}개 윤곽선 포인트")
            print(f"직선 C 좌표: 왼쪽({int(x_left)}, {int(y_left)}) -> 오른쪽({int(x_right)}, {int(y_right)})")
            
            # 디버깅: 마스크 영역 시각화
            debug_img = np.zeros((h, w, 3), dtype=np.uint8)
            
            # 직선 C 그리기 (빨간색)
            cv2.line(debug_img, (int(x_left), int(y_left)), (int(x_right), int(y_right)), (0, 0, 255), 2)
            
            # 교점들 그리기 (노란색)
            cv2.circle(debug_img, (int(left_intersect[0]), int(left_intersect[1])), 5, (0, 255, 255), -1)
            cv2.circle(debug_img, (int(right_intersect[0]), int(right_intersect[1])), 5, (0, 255, 255), -1)
            
            # 윤곽선 포인트들 그리기 (보라색)
            for i, point in enumerate(outline_points):
                cv2.circle(debug_img, (int(point[0]), int(point[1])), 3, (255, 0, 255), -1)
                if i > 0:
                    cv2.line(debug_img, (int(outline_points[i-1][0]), int(outline_points[i-1][1])), 
                            (int(point[0]), int(point[1])), (255, 0, 255), 1)
            
            # 마스크 영역 그리기 (반투명 흰색)
            mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
            mask_vis[mask > 0] = [255, 255, 255]
            debug_img = cv2.addWeighted(debug_img, 0.7, mask_vis, 0.3, 0)
            
            # 이미지 경계 그리기 (초록색)
            cv2.rectangle(debug_img, (0, 0), (w-1, h-1), (0, 255, 0), 1)
            
            cv2.imwrite("debug_mask_visualization.jpg", debug_img)
            print("디버깅 이미지 저장: debug_mask_visualization.jpg")
            
            # 마스크 중심점 시각화
            moments = cv2.moments(mask)
            if moments['m00'] != 0:
                center_x = int(moments['m10'] / moments['m00'])
                center_y = int(moments['m01'] / moments['m00'])
                cv2.circle(debug_img, (center_x, center_y), 10, (0, 255, 0), -1)  # 초록색 원으로 중심점 표시
                cv2.imwrite("debug_mask_with_center.jpg", debug_img)
                print(f"마스크 중심점: ({center_x}, {center_y})")
                print(f"이미지 크기: {w}x{h}")
                print(f"중심점이 이미지 경계 내에 있는가: {0 <= center_x < w and 0 <= center_y < h}")
            
        else:
            print("create_mouth_mask2: 윤곽선 포인트가 부족합니다.")
    else:
        print("create_mouth_mask2: 교점이 2개가 아니어서 마스크를 생성할 수 없습니다.")
    
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
            # 마스크가 유효한지 확인
            if mask is None or mask.size == 0:
                print("Poisson 블렌딩 실패 - 마스크가 유효하지 않음")
                return smooth_blend_mouth(result_img, target_img, mask, "gaussian")
            
            # 마스크를 3채널로 변환 (seamlessClone은 3채널 마스크 필요)
            #if len(mask.shape) == 2:
            #    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            #else:
            #    mask_3ch = mask
            
            # 마스크에서 중심점 찾기 - 더 정확한 중심점 계산
            # 방법 1: 마스크의 기하학적 중심
            moments = cv2.moments(mask)
            if moments['m00'] != 0:
                center_x = int(moments['m10'] / moments['m00'])
                center_y = int(moments['m01'] / moments['m00'])
                
                # 방법 2: 마스크의 바운딩 박스 중심도 계산
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # 가장 큰 컨투어의 바운딩 박스 중심
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w_bbox, h_bbox = cv2.boundingRect(largest_contour)
                    bbox_center_x = x + w_bbox // 2
                    bbox_center_y = y + h_bbox // 2
                    
                    # 두 중심점의 평균 사용 (더 안정적)
                    center_x = (center_x + bbox_center_x) // 2
                    center_y = (center_y + bbox_center_y) // 2
                
                # 중심점이 이미지 경계 내에 있는지 확인
                if 0 <= center_x < result_img.shape[1] and 0 <= center_y < result_img.shape[0]:
                    center = (center_x, center_y)
                    print(f"Poisson 블렌딩 중심점: ({center_x}, {center_y})")
                    
                    # Poisson 블렌딩 시도
                    try:
                        blended = cv2.seamlessClone(target_img, result_img, mask, center, cv2.NORMAL_CLONE)
                        print("Poisson 블렌딩 성공")
                        return blended
                    except Exception as e:
                        print(f"Poisson 블렌딩 실패: {e}")
                        print("Gaussian 블렌딩으로 대체")
                        return smooth_blend_mouth(result_img, target_img, mask, "gaussian")
                else:
                    print(f"Poisson 블렌딩 실패 - 중심점이 이미지 경계를 벗어남: ({center_x}, {center_y})")
                    return smooth_blend_mouth(result_img, target_img, mask, "gaussian")
            else:
                print("Poisson 블렌딩 실패 - 중심점 찾기 실패")
                # fallback to gaussian
                return smooth_blend_mouth(result_img, target_img, mask, "gaussian")
        except Exception as e:
            print(f"Poisson 블렌딩 실패 - 예외 발생 {e}")
            # Poisson 블렌딩 실패 시 gaussian으로 fallback
            return smooth_blend_mouth(result_img, target_img, mask, "gaussian")

if __name__ == "__main__":

    INSWAPPER_PATH = r"C:\models\inswapper_128.onnx"
    BUFFALO_L_PATH = "C:\\"

    source_img = cv2.imread("C:\\images\\f_base.jpg")
    target_img = cv2.imread("C:\\images\\b_4.jpg")
    #target_img = cv2.imread("C:\\images\\b_4.jpg")

    app = FaceAnalysis(name='buffalo_l', root=BUFFALO_L_PATH)
    app.prepare(ctx_id=-1, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model(INSWAPPER_PATH)

    # 얼굴 감지
    source_faces = app.get(source_img)
    target_faces = app.get(target_img) 
    source_face = source_faces[0]
    target_face = target_faces[0]
    
    print(f"Source 이미지 크기: {source_img.shape}")
    print(f"Target 이미지 크기: {target_img.shape}")
    print(f"Source 얼굴 바운딩 박스: {source_face.bbox}")
    print(f"Target 얼굴 바운딩 박스: {target_face.bbox}")

    # 일반 스와핑 수행
    result = swapper.get(target_img, target_face, source_face, paste_back=True)
    print(f"Result 이미지 크기: {result.shape}")

    # 입 마스크 생성 (원본과 동일한 파라미터)
    landmarks = target_face.landmark_2d_106
    print(f"랜드마크 개수: {len(landmarks)}")
    
    # 랜드마크 범위 확인
    landmark_x_min, landmark_x_max = landmarks[:, 0].min(), landmarks[:, 0].max()
    landmark_y_min, landmark_y_max = landmarks[:, 1].min(), landmarks[:, 1].max()
    print(f"랜드마크 X 범위: {landmark_x_min:.1f} ~ {landmark_x_max:.1f}")
    print(f"랜드마크 Y 범위: {landmark_y_min:.1f} ~ {landmark_y_max:.1f}")
    print(f"이미지 크기: {target_img.shape[1]} x {target_img.shape[0]}")
    
    # 랜드마크가 이미지 경계를 벗어나는지 확인
    if landmark_x_min < 0 or landmark_x_max >= target_img.shape[1] or landmark_y_min < 0 or landmark_y_max >= target_img.shape[0]:
        print("경고: 일부 랜드마크가 이미지 경계를 벗어남!")
    
    print("\n=== 마스크 생성 시작 ===")
    mask = create_mouth_mask2(landmarks, target_img.shape)
    
    if mask is not None:
        print(f"마스크 크기: {mask.shape}")
        print(f"마스크 최대값: {mask.max()}")
        print(f"마스크 최소값: {mask.min()}")
        print(f"마스크에서 0이 아닌 픽셀 수: {np.count_nonzero(mask)}")
        
        # 마스크 영역 비율 계산
        total_pixels = mask.shape[0] * mask.shape[1]
        mask_pixels = np.count_nonzero(mask)
        mask_ratio = mask_pixels / total_pixels
        print(f"마스크 영역 비율: {mask_ratio:.3f} ({mask_ratio*100:.1f}%)")
        
        # 마스크 중심점 계산
        moments = cv2.moments(mask)
        if moments['m00'] != 0:
            center_x = int(moments['m10'] / moments['m00'])
            center_y = int(moments['m01'] / moments['m00'])
            print(f"마스크 중심점: ({center_x}, {center_y})")
            
            # 중심점이 이미지 경계 내에 있는지 확인
            if 0 <= center_x < target_img.shape[1] and 0 <= center_y < target_img.shape[0]:
                print("중심점이 이미지 경계 내에 있음")
            else:
                print(f"경고: 중심점이 이미지 경계를 벗어남! 이미지 크기: {target_img.shape[1]}x{target_img.shape[0]}")
        
        # 마스크 저장
        cv2.imwrite("debug_mask_final.jpg", mask)
        print("마스크 이미지 저장: debug_mask_final.jpg")
    else:
        print("마스크 생성 실패!")

    # 블렌딩용 부드러운 마스크 생성 (원본 마스크에 블러 적용)

    # === 다양한 블렌딩 방법 테스트 ===
    
    # 방법 1: 가우시안 블렌딩 (가장 빠르고 안정적)
    mask_ = mask.copy()
    result_gaussian = smooth_blend_mouth(result, target_img, mask_, "gaussian")
    cv2.imwrite("result_gaussian_blend_.jpg", result_gaussian)
    
    # 방법 2: Poisson 블렌딩 (가장 자연스럽지만 느림)
    mask_ = mask.copy()
    result_poisson = smooth_blend_mouth(result, target_img, mask_, "poisson")  # 원본 마스크 사용
    cv2.imwrite("result_poisson_blend_.jpg", result_poisson)
    
        # 기존 방법과 비교용
    mouth_mask_bool = mask > 0
    result_original = result.copy()
    result_original[mouth_mask_bool] = target_img[mouth_mask_bool]
    cv2.imwrite("result_original_method_.jpg", result_original)
    
    print("모든 블렌딩 결과가 저장되었습니다.")
    print("- result_gaussian_blend.jpg: 가우시안 블렌딩")
    print("- result_poisson_blend.jpg: Poisson 블렌딩")
    print("- result_original_method.jpg: 기존 방법")