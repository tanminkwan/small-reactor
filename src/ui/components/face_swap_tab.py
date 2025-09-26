"""
얼굴 교체 탭 컴포넌트

Single Responsibility Principle (SRP)에 따라
얼굴 교체 UI만을 담당하는 컴포넌트
"""

import gradio as gr
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from src.services.face_manager import FaceManager
from src.services.file_manager import FileManager
from src.utils.file_utils import open_save_location, get_save_location_status
from src.utils import Config


class FaceSwapTab:
    """얼굴 교체 탭 컴포넌트"""
    
    def __init__(self, face_manager: FaceManager, file_manager: FileManager):
        """
        초기화
        
        Args:
            face_manager: 얼굴 관리 서비스
            file_manager: 파일 관리 서비스
        """
        self.face_manager = face_manager
        self.file_manager = file_manager
        self.config = Config()  # Config 인스턴스 추가
        self.last_result_path = None  # 마지막 저장된 결과 파일 경로
    
    def create_interface(self) -> gr.Tab:
        """
        얼굴 교체 탭 인터페이스를 생성합니다.
        
        Returns:
            Gradio Tab 컴포넌트
        """
        with gr.Tab("얼굴 교체") as tab:
            gr.Markdown("## 🔄 얼굴 교체")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # 타겟 이미지 업로드
                    self.target_upload = gr.Image(
                        label="타겟 이미지 업로드",
                        type="filepath"
                    )
                    
                    # 교체할 얼굴 인덱스 입력
                    self.face_indices_input = gr.Textbox(
                        label="교체할 얼굴 인덱스 (쉼표로 구분, 비워두면 모든 얼굴)",
                        placeholder="예: 1,3,5 또는 비워두기",
                        value="",
                        info="💡 원본 이미지의 얼굴 박스를 클릭하면 자동으로 인덱스가 추가/제거됩니다"
                    )
                    
                    # 바꿀 얼굴 선택 (드롭다운과 새로고침 버튼)
                    with gr.Row():
                        self.source_face_dropdown = gr.Dropdown(
                            label="바꿀 얼굴 선택",
                            choices=self.file_manager.get_embedding_choices(),
                            value=self.file_manager.get_embedding_choices()[0] if self.file_manager.get_embedding_choices() else None,
                            scale=4
                        )
                        self.refresh_faces_btn = gr.Button(
                            "🔄", 
                            variant="secondary", 
                            size="sm",
                            scale=1
                        )
                    
                    # CodeFormer 복원 체크박스
                    self.codeformer_checkbox = gr.Checkbox(
                        label="CodeFormer 복원 포함",
                        value=True,
                        info="체크하면 얼굴 교체 후 자동으로 CodeFormer 복원도 수행됩니다"
                    )
                    
                    # CodeFormer Fidelity 설정 (조건부 표시)
                    with gr.Group(visible=True) as self.codeformer_settings_group:
                        self.fidelity_slider = gr.Slider(
                            label="Fidelity (복원 강도)",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.1,
                            info="높을수록 원본에 가깝게 복원됩니다 (0.0: 완전 복원, 1.0: 원본 유지)"
                        )
                    
                    # 입 원본유지 체크박스
                    self.preserve_mouth_checkbox = gr.Checkbox(
                        label="입 원본유지",
                        value=False,
                        info="체크하면 얼굴 교체 후 입과 입주변을 원본 이미지로 복원합니다"
                    )
                    
                    # 입 원본유지 방식 선택
                    self.mouth_preserve_method = gr.Radio(
                        choices=[
                            ("입주변 타원 마스크", "ellipse"),
                            ("입과 턱 마스크", "chin_region")
                        ],
                        value="chin_region",
                        label="입 원본유지 방식",
                        info="입 원본유지 시 사용할 마스크 방식을 선택하세요",
                        visible=False
                    )
                    
                    # 입 마스크 설정 (조건부 표시)
                    with gr.Group(visible=False) as self.mouth_settings_group:
                        gr.Markdown("### 입 마스크 설정")
                        
                        with gr.Row():
                            self.expand_ratio_slider = gr.Slider(
                                label="확장 비율 (expand_ratio)",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.2,
                                step=0.1,
                                info="입 영역 확장 정도"
                            )
                        
                        with gr.Row():
                            self.scale_x_slider = gr.Slider(
                                label="가로 스케일 (scale_x)",
                                minimum=0.1,
                                maximum=5.0,
                                value=1.0,
                                step=0.1,
                                info="가로 방향 확장 배율"
                            )
                            self.scale_y_slider = gr.Slider(
                                label="세로 스케일 (scale_y)",
                                minimum=0.1,
                                maximum=5.0,
                                value=1.0,
                                step=0.1,
                                info="세로 방향 확장 배율"
                            )
                        
                        with gr.Row():
                            self.offset_x_slider = gr.Slider(
                                label="가로 오프셋 (offset_x)",
                                minimum=-50,
                                maximum=50,
                                value=0,
                                step=1,
                                info="가로 방향 이동 픽셀"
                            )
                            self.offset_y_slider = gr.Slider(
                                label="세로 오프셋 (offset_y)",
                                minimum=-50,
                                maximum=50,
                                value=0,
                                step=1,
                                info="세로 방향 이동 픽셀"
                            )
                    
                    # 얼굴 교체 버튼
                    self.swap_btn = gr.Button("얼굴 변경", variant="primary")
                    
                    # 결과 메시지
                    self.swap_result_text = gr.Textbox(
                        label="결과",
                        lines=3,
                        interactive=False
                    )
                    
                    # 얼굴 교체된 이미지 State (CodeFormer용)
                    self.swapped_image_state = gr.State()
                
                with gr.Column(scale=1):
                    # 원본 이미지 표시
                    self.original_image = gr.Image(
                        label="원본 이미지",
                        type="numpy"
                    )
                    
                    # 최종 결과 이미지 표시
                    with gr.Group():
                        self.swapped_image = gr.Image(
                            label="최종 결과",
                            type="numpy"
                        )
                        
                        # 결과 이미지 관리 버튼
                        with gr.Row():
                            self.delete_result_btn = gr.Button(
                                "🗑️ 결과 이미지 삭제",
                                variant="secondary",
                                size="sm"
                            )
                            self.save_location_btn = gr.Button(
                                "📁 저장 위치 확인",
                                variant="secondary",
                                size="sm"
                            )
                            self.move_to_target_btn = gr.Button(
                                "📤 타겟이미지로 이동",
                                variant="secondary",
                                size="sm"
                            )
                        
                        # 삭제 결과 메시지
                        self.delete_result_text = gr.Textbox(
                            label="삭제 결과",
                            lines=1,
                            interactive=False,
                            visible=True
                        )
        
        return tab
    
    def process_target_image(self, file_path: str, current_indices: str = "") -> Tuple[bool, str, Optional[np.ndarray], str, bool, str]:
        """
        타겟 이미지를 처리하고 얼굴 탐지 결과를 박스로 표시합니다.
        인덱스 검증도 함께 수행하고 기본값으로 리셋합니다.
        
        Args:
            file_path: 파일 경로
            current_indices: 현재 얼굴 인덱스
            
        Returns:
            (성공여부, 메시지, 박스가 그려진 이미지, 검증된 인덱스, 입원본유지 체크, 입원본유지 방식)
        """
        if file_path is None:
            return False, "이미지를 업로드해주세요.", None, "", False, "chin_region"
        
        try:
            from PIL import Image
            
            # PIL로 이미지 로드
            pil_image = Image.open(file_path)
            image_rgb = np.array(pil_image)
            
            # RGB를 BGR로 변환 (얼굴 탐지용)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # 얼굴 탐지 및 박스 그리기
            success, message, result_image = self.face_manager.detect_and_draw_faces(image_bgr)
            
            # 인덱스 검증 및 초기화
            validated_indices = self.validate_and_clear_indices(current_indices, image_rgb)
            
            if success:
                return True, message, result_image, validated_indices, False, "chin_region"
            else:
                # 얼굴을 찾지 못한 경우 원본 이미지 반환
                return True, f"이미지 로드 완료\n{message}", image_rgb, validated_indices, False, "chin_region"
            
        except Exception as e:
            return False, f"이미지를 로드할 수 없습니다: {str(e)}", None, "", False, "chin_region"
    
    def perform_face_swap_with_optional_codeformer(
        self, 
        file_path: str, 
        face_indices: str, 
        source_face_name: str, 
        use_codeformer: bool, 
        fidelity: float = 0.5,
        preserve_mouth: bool = False, 
        mouth_settings: Optional[Dict[str, Any]] = None,
        mouth_preserve_method: str = "ellipse"
    ) -> Tuple[Optional[np.ndarray], str, Optional[np.ndarray]]:
        """
        얼굴 교체를 수행하고, 선택적으로 CodeFormer 복원도 수행합니다.
        
        Args:
            file_path: 타겟 이미지 파일 경로
            face_indices: 교체할 얼굴 인덱스
            source_face_name: 소스 얼굴 이름
            use_codeformer: CodeFormer 복원 사용 여부
            fidelity: CodeFormer Fidelity 설정 (0.0-1.0)
            preserve_mouth: 입 원본유지 여부
            mouth_settings: 입 마스크 설정
            mouth_preserve_method: 입 원본유지 방식
            
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
            success, message, swapped_image_rgb = self.face_manager.swap_faces(
                image_bgr, face_indices, source_face_name, self.file_manager.faces_dir
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
                    cf_success, cf_message, enhanced_image_rgb = self.face_manager.enhance_faces_with_codeformer(
                        swapped_image_bgr, face_indices, fidelity
                    )
                    
                    if cf_success:
                        final_image = enhanced_image_rgb
                        final_message = f"{message}\n{cf_message}"
                    else:
                        final_message = f"{message}\nCodeFormer 복원 실패: {cf_message}"
                        
                except Exception as e:
                    final_message = f"{message}\nCodeFormer 복원 실패: {str(e)}"
            
            # 입 원본유지가 체크되어 있으면 CodeFormer 복원 후에 수행
            if preserve_mouth and mouth_settings:
                try:
                    # 최종 이미지를 BGR로 변환 (입 원본유지용)
                    final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
                    
                    # 입 원본유지 적용
                    mouth_success, mouth_message, mouth_restored_image_rgb = self.face_manager.apply_mouth_preservation(
                        final_image_bgr, image_bgr, face_indices, mouth_settings, mouth_preserve_method
                    )
                    
                    if mouth_success:
                        final_image = mouth_restored_image_rgb
                        final_message = f"{final_message}\n{mouth_message}"
                    else:
                        final_message = f"{final_message}\n입 원본유지 실패: {mouth_message}"
                        
                except Exception as e:
                    final_message = f"{final_message}\n입 원본유지 실패: {str(e)}"
            
            # 최종 결과 이미지 파일로 저장
            if final_image is not None:
                try:
                    output_filename = self.file_manager.save_result_image(final_image)
                    self.last_result_path = output_filename  # 마지막 저장 경로 기록
                    final_message += f"\n\n💾 최종 결과 저장: {output_filename}"
                    
                except Exception as save_error:
                    final_message += f"\n⚠️ 이미지 저장 실패: {str(save_error)}"
            
            return final_image, final_message, final_image
            
        except Exception as e:
            return None, f"❌ 얼굴 교체 중 오류가 발생했습니다: {str(e)}", None
    
    def show_save_location(self) -> str:
        """
        저장 위치를 파일 탐색기로 열고 상태 메시지를 반환합니다.
        
        Returns:
            실행 결과 메시지
        """
        try:
            # 얼굴 교체 결과 저장 경로 가져오기
            # FileManager의 output_path 직접 사용
            output_path = self.file_manager.output_path
            
            return open_save_location(output_path, self.last_result_path)
            
        except Exception as e:
            return f"❌ 저장 위치 열기 중 오류가 발생했습니다: {str(e)}"
            
        except Exception as e:
            return None, f"얼굴 교체 실패: {str(e)}", None
    
    def delete_result_image(self) -> Tuple[bool, str, Optional[np.ndarray]]:
        """
        최종 결과 이미지 파일을 삭제하고 다음으로 삭제될 이미지를 반환합니다.
        
        Returns:
            (삭제 성공여부, 메시지, 다음 삭제 대상 이미지)
        """
        success, message = self.file_manager.delete_latest_result_image()
        
        if success:
            # 삭제 성공 시 다음으로 삭제될 이미지 가져오기
            next_success, next_message, next_image = self.file_manager.get_next_result_image()
            if next_success:
                return success, f"{message}\n{next_message}", next_image
            else:
                return success, f"{message}\n{next_message}", None
        else:
            return success, message, None
    
    def move_result_to_target(self) -> Tuple[bool, str, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        결과 이미지를 타겟 이미지로 이동시킵니다.
        
        Returns:
            (성공여부, 메시지, 타겟 이미지, 원본 이미지)
        """
        try:
            # 다음으로 삭제될 이미지 가져오기 (현재 결과 이미지)
            success, message, result_image = self.file_manager.get_next_result_image()
            
            if success and result_image is not None:
                # 결과 이미지를 타겟 이미지로 설정
                # 얼굴 탐지 및 박스 그리기
                import cv2
                image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
                detect_success, detect_message, boxed_image = self.face_manager.detect_and_draw_faces(image_bgr)
                
                if detect_success:
                    return True, f"✅ 타겟 이미지로 이동 완료\n{detect_message}", result_image, boxed_image
                else:
                    return True, f"✅ 타겟 이미지로 이동 완료\n{detect_message}", result_image, result_image
            else:
                return False, "이동할 결과 이미지가 없습니다.", None, None
                
        except Exception as e:
            return False, f"❌ 타겟 이미지로 이동 실패: {str(e)}", None, None
    
    def handle_face_click(self, evt: gr.SelectData, current_indices: str, original_image: np.ndarray) -> str:
        """
        얼굴 박스 클릭 시 인덱스를 추가/제거합니다.
        
        Args:
            evt: Gradio SelectData 이벤트 (좌표값 포함)
            current_indices: 현재 인덱스 문자열
            original_image: 원본 이미지 (얼굴 탐지용)
            
        Returns:
            업데이트된 인덱스 문자열
        """
        if evt.index is None or original_image is None:
            return current_indices
        
        try:
            # 클릭한 좌표 (x, y)
            click_x, click_y = evt.index[0], evt.index[1]
            
            # 원본 이미지를 BGR로 변환 (얼굴 탐지용)
            import cv2
            if len(original_image.shape) == 3 and original_image.shape[2] == 3:
                # RGB를 BGR로 변환
                image_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = original_image
            
            # 얼굴 탐지
            faces = self.face_manager.detector.detect_faces(image_bgr)
            if not faces:
                return current_indices
            
            # 클릭한 좌표가 포함된 얼굴 찾기
            clicked_face_index = None
            for i, face in enumerate(faces):
                bbox = face.bbox
                x1, y1, x2, y2 = map(int, bbox)
                
                # 클릭한 좌표가 이 얼굴 박스 안에 있는지 확인
                if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                    clicked_face_index = i + 1  # 1-based 인덱스
                    break
            
            if clicked_face_index is None:
                return current_indices
            
            # 현재 인덱스들을 파싱
            if current_indices.strip():
                indices = [idx.strip() for idx in current_indices.split(',')]
            else:
                indices = []
            
            clicked_index_str = str(clicked_face_index)
            
            # 클릭된 인덱스가 이미 있으면 제거, 없으면 추가
            if clicked_index_str in indices:
                indices.remove(clicked_index_str)
            else:
                indices.append(clicked_index_str)
            
            # 정렬하여 반환
            indices.sort(key=int)
            return ','.join(indices)
            
        except Exception as e:
            # 오류 발생 시 현재 인덱스 그대로 반환
            return current_indices
    
    def validate_and_clear_indices(self, current_indices: str, original_image: np.ndarray) -> str:
        """
        현재 인덱스가 원본 이미지의 얼굴 수와 일치하는지 검증하고,
        유효하지 않은 인덱스가 있으면 초기화합니다.
        
        Args:
            current_indices: 현재 인덱스 문자열
            original_image: 원본 이미지
            
        Returns:
            검증된 인덱스 문자열
        """
        if not current_indices.strip() or original_image is None:
            return ""
        
        try:
            # 원본 이미지를 BGR로 변환 (얼굴 탐지용)
            import cv2
            if len(original_image.shape) == 3 and original_image.shape[2] == 3:
                # RGB를 BGR로 변환
                image_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = original_image
            
            # 얼굴 탐지
            faces = self.face_manager.detector.detect_faces(image_bgr)
            if not faces:
                return ""  # 얼굴이 없으면 인덱스 초기화
            
            max_face_index = len(faces)
            
            # 현재 인덱스들을 파싱
            indices = [idx.strip() for idx in current_indices.split(',')]
            valid_indices = []
            
            for idx in indices:
                try:
                    idx_num = int(idx)
                    if 1 <= idx_num <= max_face_index:
                        valid_indices.append(idx)
                except ValueError:
                    # 숫자가 아닌 값은 무시
                    continue
            
            # 유효한 인덱스들을 정렬하여 반환
            valid_indices.sort(key=int)
            return ','.join(valid_indices)
            
        except Exception as e:
            # 오류 발생 시 인덱스 초기화
            return ""
