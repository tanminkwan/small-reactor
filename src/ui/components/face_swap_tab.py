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
                        value=""
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
                    
                    # 입 원본유지 체크박스
                    self.preserve_mouth_checkbox = gr.Checkbox(
                        label="입 원본유지",
                        value=False,
                        info="체크하면 얼굴 교체 후 입과 입주변을 원본 이미지로 복원합니다"
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
                        
                        # 결과 이미지 삭제 버튼
                        with gr.Row():
                            self.delete_result_btn = gr.Button(
                                "🗑️ 결과 이미지 삭제",
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
    
    def process_target_image(self, file_path: str) -> Tuple[bool, str, Optional[np.ndarray]]:
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
            success, message, result_image = self.face_manager.detect_and_draw_faces(image_bgr)
            
            if success:
                return True, message, result_image
            else:
                # 얼굴을 찾지 못한 경우 원본 이미지 반환
                return True, f"이미지 로드 완료\n{message}", image_rgb
            
        except Exception as e:
            return False, f"이미지를 로드할 수 없습니다: {str(e)}", None
    
    def perform_face_swap_with_optional_codeformer(
        self, 
        file_path: str, 
        face_indices: str, 
        source_face_name: str, 
        use_codeformer: bool, 
        preserve_mouth: bool = False, 
        mouth_settings: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[np.ndarray], str, Optional[np.ndarray]]:
        """
        얼굴 교체를 수행하고, 선택적으로 CodeFormer 복원도 수행합니다.
        
        Args:
            file_path: 타겟 이미지 파일 경로
            face_indices: 교체할 얼굴 인덱스
            source_face_name: 소스 얼굴 이름
            use_codeformer: CodeFormer 복원 사용 여부
            preserve_mouth: 입 원본유지 여부
            mouth_settings: 입 마스크 설정
            
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
                        swapped_image_bgr, face_indices
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
                        final_image_bgr, image_bgr, face_indices, mouth_settings
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
                    final_message += f"\n\n💾 최종 결과 저장: {output_filename}"
                    
                except Exception as save_error:
                    final_message += f"\n⚠️ 이미지 저장 실패: {str(save_error)}"
            
            return final_image, final_message, final_image
            
        except Exception as e:
            return None, f"얼굴 교체 실패: {str(e)}", None
    
    def delete_result_image(self) -> Tuple[bool, str, None]:
        """
        최종 결과 이미지 파일을 삭제하고 화면을 초기화합니다.
        
        Returns:
            (삭제 성공여부, 메시지, None)
        """
        success, message = self.file_manager.delete_latest_result_image()
        return success, message, None
