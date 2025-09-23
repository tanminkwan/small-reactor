"""
Inpainting 탭 컴포넌트

Single Responsibility Principle (SRP)에 따라
Inpainting UI만을 담당하는 컴포넌트
"""

import gradio as gr
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import tempfile
import os
from datetime import datetime

from src.services.file_manager import FileManager


class InpaintingTab:
    """Inpainting 탭 컴포넌트"""
    
    def __init__(self, file_manager: FileManager):
        """
        초기화
        
        Args:
            file_manager: 파일 관리 서비스
        """
        self.file_manager = file_manager
        self.temp_dir = Path(tempfile.gettempdir()) / "small_reactor_inpainting"
        self.temp_dir.mkdir(exist_ok=True)
    
    def create_interface(self) -> gr.Tab:
        """
        Inpainting 탭 인터페이스를 생성합니다.
        
        Returns:
            Gradio Tab 컴포넌트
        """
        with gr.Tab("Inpainting") as tab:
            gr.Markdown("## 🎨 Inpainting")
            gr.Markdown("""
            이미지를 업로드하고 마스킹할 영역을 그려보세요.
            
            **사용법:**
            1. 이미지를 업로드하세요
            2. 이미지 위에서 편집 모드를 활성화하세요 (연필 아이콘 클릭)
            3. 마우스로 마스킹할 영역을 그리세요
            4. '임시 저장' 버튼을 눌러 결과를 저장하세요
            
            *참고: 브러시 크기는 Gradio 기본값을 사용합니다.*
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # 이미지 업로드 및 마스킹 영역
                    # Gradio 5.46.1에서 ImageEditor 시도
                    try:
                        # ImageEditor가 있다면 사용
                        self.image_input = gr.ImageEditor(
                            label="이미지 업로드 및 마스킹 (편집 모드에서 브러시로 그리기)",
                            type="pil",
                            interactive=True,
                            brush=gr.Brush(
                                colors=["#808080"],
                                default_size=20
                            )
                        )
                    except:
                        # ImageEditor가 없다면 기본 Image 사용
                        self.image_input = gr.Image(
                            label="이미지 업로드 (우클릭 후 편집 모드 선택)",
                            type="pil",
                            interactive=True,
                            sources=["upload", "webcam", "clipboard"]
                        )
                    
                    # 브러시 크기 조절
                    self.brush_size_slider = gr.Slider(
                        minimum=5,
                        maximum=100,
                        value=20,
                        step=1,
                        label="브러시 크기 (참고용)",
                        interactive=True
                    )
                    
                    # 브러시 크기 표시
                    self.brush_size_display = gr.Textbox(
                        label="브러시 정보",
                        value="현재 브러시 크기: 20px (참고용 - 실제 브러시 크기는 Gradio 기본값 사용)",
                        interactive=False
                    )
                    
                    # 임시 저장 버튼
                    self.tmp_save_button = gr.Button(
                        "임시 저장",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=1):
                    # 결과 표시 영역
                    self.result_display = gr.Image(
                        label="마스킹 결과",
                        type="pil",
                        interactive=False
                    )
                    
                    # 저장 상태 표시
                    self.save_status = gr.Textbox(
                        label="저장 상태",
                        value="마스킹을 그리고 임시 저장 버튼을 눌러주세요.",
                        interactive=False
                    )
                    
                    # 저장된 파일 경로 표시
                    self.saved_path_display = gr.Textbox(
                        label="저장된 파일 경로",
                        value="",
                        interactive=False
                    )
        
        return tab
    
    def setup_event_handlers(self):
        """이벤트 핸들러를 설정합니다."""
        
        # 브러시 크기 변경 시 브러시 정보 업데이트
        # 참고: Gradio 5.46.1에서는 런타임 브러시 크기 변경이 제한적
        self.brush_size_slider.change(
            fn=self._update_brush_size,
            inputs=[self.brush_size_slider],
            outputs=[self.brush_size_display]
        )
        
        # 임시 저장 버튼 클릭 시
        self.tmp_save_button.click(
            fn=self._save_mask_result,
            inputs=[self.image_input],
            outputs=[self.result_display, self.save_status, self.saved_path_display]
        )
    
    def _update_brush_size(self, brush_size: int) -> str:
        """
        브러시 크기를 업데이트합니다.
        
        Args:
            brush_size: 새로운 브러시 크기
            
        Returns:
            브러시 크기 변경 메시지
        """
        # Gradio 5.46.1에서는 런타임 브러시 크기 변경이 제한적
        # 대신 사용자에게 브러시 크기 정보를 제공
        return f"현재 브러시 크기: {brush_size}px (참고용 - 실제 브러시 크기는 Gradio 기본값 사용)"
    
    def _save_mask_result(
        self, 
        image_data: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], str, str]:
        """
        마스킹 결과를 임시 저장합니다.
        
        Args:
            image_data: Gradio Image 컴포넌트에서 반환된 데이터
            
        Returns:
            Tuple[마스킹된 이미지, 저장 상태 메시지, 저장된 파일 경로]
        """
        try:
            if image_data is None:
                return None, "❌ 이미지가 없습니다.", ""
            
            # Gradio Image 컴포넌트에서 이미지와 마스크 추출
            if isinstance(image_data, dict):
                # 편집된 이미지 (마스크 포함)
                if "composite" in image_data:
                    composite_image = image_data["composite"]
                elif "image" in image_data:
                    composite_image = image_data["image"]
                else:
                    return None, "❌ 이미지 데이터를 찾을 수 없습니다.", ""
            else:
                composite_image = image_data
            
            # PIL Image를 numpy array로 변환
            if hasattr(composite_image, 'convert'):
                composite_array = np.array(composite_image.convert('RGB'))
            else:
                composite_array = np.array(composite_image)
            
            # 타임스탬프를 포함한 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"inpainting_mask_{timestamp}.png"
            filepath = self.temp_dir / filename
            
            # 이미지 저장
            cv2.imwrite(
                str(filepath), 
                cv2.cvtColor(composite_array, cv2.COLOR_RGB2BGR)
            )
            
            # 마스크 영역 추출 (선택적)
            mask_filename = f"inpainting_mask_only_{timestamp}.png"
            mask_filepath = self.temp_dir / mask_filename
            
            # 마스크만 별도 저장하는 로직 (필요시)
            if isinstance(image_data, dict) and "mask" in image_data:
                mask_data = image_data["mask"]
                if mask_data is not None:
                    mask_array = np.array(mask_data)
                    cv2.imwrite(str(mask_filepath), mask_array)
            
            status_message = f"✅ 마스킹 결과가 저장되었습니다. ({timestamp})"
            
            return composite_image, status_message, str(filepath)
            
        except Exception as e:
            error_message = f"❌ 저장 중 오류가 발생했습니다: {str(e)}"
            return None, error_message, ""
    
    def _create_masked_preview(
        self, 
        original_image: np.ndarray, 
        mask: np.ndarray
    ) -> np.ndarray:
        """
        마스크가 적용된 미리보기 이미지를 생성합니다.
        
        Args:
            original_image: 원본 이미지
            mask: 마스크 이미지
            
        Returns:
            마스크가 적용된 미리보기 이미지
        """
        # 마스크 영역을 반투명 회색으로 오버레이
        overlay = original_image.copy()
        
        # 마스크가 있는 영역을 회색으로 설정
        mask_area = mask > 0
        overlay[mask_area] = [128, 128, 128]  # 회색
        
        # 원본과 오버레이를 블렌딩 (반투명 효과)
        alpha = 0.5
        result = cv2.addWeighted(original_image, 1-alpha, overlay, alpha, 0)
        
        return result
    
    def cleanup_temp_files(self, max_age_hours: int = 24) -> None:
        """
        오래된 임시 파일들을 정리합니다.
        
        Args:
            max_age_hours: 최대 보관 시간 (시간 단위)
        """
        try:
            import time
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            for file_path in self.temp_dir.glob("inpainting_*"):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        file_path.unlink()
                        
        except Exception as e:
            # 로깅만 하고 계속 진행
            print(f"임시 파일 정리 중 오류: {e}")
