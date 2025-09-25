"""
이미지 결합 탭 컴포넌트

Single Responsibility Principle (SRP)에 따라
이미지 결합 UI만을 담당하는 컴포넌트
"""

import gradio as gr
import numpy as np
from typing import Tuple, Optional
from PIL import Image

from src.services.image_blend_service import ImageBlendService
from src.services.file_manager import FileManager
from src.utils.file_utils import open_save_location, get_save_location_status


class ImageBlendTab:
    """이미지 결합 탭 컴포넌트"""
    
    def __init__(self, image_blend_service: ImageBlendService, file_manager: FileManager):
        """
        초기화
        
        Args:
            image_blend_service: 이미지 결합 서비스
            file_manager: 파일 관리 서비스
        """
        self.image_blend_service = image_blend_service
        self.file_manager = file_manager
        self.last_result_path = None  # 마지막 저장된 결과 파일 경로
    
    def create_interface(self) -> gr.Tab:
        """
        이미지 결합 탭 인터페이스를 생성합니다.
        
        Returns:
            Gradio Tab 컴포넌트
        """
        with gr.Tab("이미지 결합") as tab:
            gr.Markdown("## 🔗 이미지 결합")
            gr.Markdown("템플릿 매칭을 사용하여 원본 이미지에 부분 이미지를 자연스럽게 결합합니다.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # 이미지 업로드 섹션
                    gr.Markdown("### 📷 이미지 업로드")
                    
                    # 원본 이미지 업로드
                    self.original_image_upload = gr.Image(
                        label="원본 이미지 (결합될 기본 이미지)",
                        type="pil",
                        height=400
                    )
                    
                    # 부분 이미지 업로드 (투명도 보존을 위해 numpy 사용)
                    self.patch_image_upload = gr.Image(
                        label="부분 이미지 (원본에 붙일 이미지) - PNG 투명 이미지 지원",
                        type="numpy",
                        height=400
                    )
                    
                    # 설정 섹션
                    gr.Markdown("### ⚙️ 결합 설정")
                    
                    # 마스크 두께 설정
                    self.border_thickness_slider = gr.Slider(
                        label="마스크 두께 (픽셀) - 경계 블렌딩 영역의 두께",
                        minimum=3,
                        maximum=50,
                        step=1,
                        value=15
                    )
                    
                    # 경계 보정 방법 선택
                    self.blend_method_radio = gr.Radio(
                        label="경계 보정 방법 - 결합 경계를 부드럽게 처리",
                        choices=[
                            ("Alpha Blending (권장)", "alpha"),
                            ("Seamless Cloning", "seamless"),
                            ("기본 결합", "basic")
                        ],
                        value="alpha"
                    )
                    
                    # 버튼들
                    gr.Markdown("### 🚀 실행")
                    
                    self.blend_button = gr.Button(
                        "🔗 이미지 결합",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=1):
                    # 결과 표시 섹션
                    gr.Markdown("### 📋 결합 결과")
                    
                    # 결과 이미지 표시
                    self.result_image = gr.Image(
                        label="결합된 이미지",
                        type="pil",
                        interactive=False,
                        height=400
                    )
                    
                    # 결과 관리 버튼들
                    with gr.Row():
                        self.delete_result_btn = gr.Button(
                            "🗑️ 결과 삭제",
                            variant="secondary",
                            size="sm"
                        )
                        self.save_status_btn = gr.Button(
                            "📁 저장 위치 확인",
                            variant="secondary", 
                            size="sm"
                        )
                    
                    # 상태 표시
                    self.status_text = gr.Textbox(
                        label="상태",
                        value="원본 이미지와 부분 이미지를 업로드한 후 '이미지 결합' 버튼을 클릭하세요.",
                        interactive=False,
                        lines=3
                    )
                    
                    # 도움말
                    gr.Markdown("""
                    ### 💡 사용 팁
                    
                    **✨ Alpha 채널 자동 지원:**
                    - **RGBA/PNG 투명 이미지**: 부분 이미지가 투명 영역을 가지면 자동으로 Alpha 채널 블렌딩 적용
                    - **투명 영역**: 원본 이미지가 완전히 보임
                    - **반투명 영역**: 원본과 부분 이미지가 비례적으로 블렌딩
                    - **불투명 영역**: 부분 이미지가 완전히 보임
                    
                    **마스크 두께:** (Alpha 채널이 없는 경우에만 사용)
                    - 작은 값(3-10): 얇은 경계, 빠른 처리
                    - 중간 값(10-25): 자연스러운 블렌딩
                    - 큰 값(25-50): 부드러운 경계, 넓은 블렌딩
                    
                    **경계 보정 방법:** (Alpha 채널이 없는 경우에만 사용)
                    - **Alpha Blending**: 가장 안정적이고 자연스러움 (권장)
                    - **Seamless Cloning**: 고품질이지만 느림, 실패 가능성
                    - **기본 결합**: 블렌딩 없음, 빠름
                    
                    **최적 결과를 위한 팁:**
                    - **PNG 투명 이미지**: 최상의 결과 (Alpha 채널 자동 활용)
                    - 부분 이미지가 원본에 잘 맞는 크기여야 함
                    - 조명과 색상이 비슷할 때 최상의 결과
                    """, elem_classes=["help-section"])
        
        return tab
    
    def setup_event_handlers(self):
        """이벤트 핸들러를 설정합니다."""
        
        # 이미지 결합 버튼 클릭
        self.blend_button.click(
            fn=self._blend_images,
            inputs=[
                self.original_image_upload,
                self.patch_image_upload,
                self.border_thickness_slider,
                self.blend_method_radio
            ],
            outputs=[
                self.result_image,
                self.status_text
            ]
        )
        
        # 결과 삭제 버튼 클릭
        self.delete_result_btn.click(
            fn=self._delete_result_image,
            inputs=[],
            outputs=[
                self.result_image,
                self.status_text
            ]
        )
        
        # 저장 위치 확인 버튼 클릭
        self.save_status_btn.click(
            fn=self._show_save_location,
            inputs=[],
            outputs=[self.status_text]
        )
    
    def _blend_images(
        self,
        original_image: Image.Image,
        patch_image: np.ndarray,
        border_thickness: int,
        blend_method: str
    ) -> Tuple[Optional[Image.Image], str]:
        """
        이미지 결합을 수행합니다.
        
        Args:
            original_image: 원본 이미지
            patch_image: 부분 이미지
            border_thickness: 마스크 두께
            blend_method: 블렌딩 방법
            
        Returns:
            Tuple[결합된 이미지, 상태 메시지]
        """
        try:
            if original_image is None:
                return None, "❌ 원본 이미지를 업로드해주세요."
            
            if patch_image is None:
                return None, "❌ 부분 이미지를 업로드해주세요."
            
            # 이미지 결합 서비스 호출
            result_image, status_message = self.image_blend_service.blend_images(
                original_image=original_image,
                patch_image=patch_image,
                border_thickness=border_thickness,
                blend_method=blend_method
            )
            
            return result_image, status_message
            
        except Exception as e:
            error_message = f"❌ 이미지 결합 중 오류가 발생했습니다: {str(e)}"
            return None, error_message
    
    def _delete_result_image(self) -> Tuple[gr.update, str]:
        """
        가장 최근 생성된 결합 결과 이미지를 삭제합니다.
        
        Returns:
            Tuple[이미지 업데이트, 상태 메시지]
        """
        try:
            success, message = self.image_blend_service.delete_latest_result_image()
            
            if success:
                # 삭제 성공 시 결과 이미지도 UI에서 제거
                return gr.update(value=None), message
            else:
                # 삭제 실패 시 UI는 그대로 유지
                return gr.update(), message
                
        except Exception as e:
            return gr.update(), f"❌ 결과 이미지 삭제 중 오류가 발생했습니다: {str(e)}"
    
    def _show_save_location(self) -> str:
        """
        저장 위치를 파일 탐색기로 열고 상태 메시지를 반환합니다.
        
        Returns:
            실행 결과 메시지
        """
        try:
            # 이미지 결합 결과 저장 경로 가져오기
            output_path = self.image_blend_service.config.get(
                "image_blend_output_path", 
                "./output/image_blend"
            )
            
            return open_save_location(output_path, self.last_result_path)
            
        except Exception as e:
            return f"❌ 저장 위치 열기 중 오류가 발생했습니다: {str(e)}"


if __name__ == "__main__":
    print("ImageBlendTab 컴포넌트가 정상적으로 로드되었습니다.")
    print("이미지 결합 UI 기능을 제공합니다.")
