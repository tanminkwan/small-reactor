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
    
    def create_interface(self) -> gr.Tab:
        """
        Inpainting 탭 인터페이스를 생성합니다.
        
        Returns:
            Gradio Tab 컴포넌트
        """
        with gr.Tab("Inpainting") as tab:
            gr.Markdown("## 🎨 Inpainting")
            
            with gr.Row():
                with gr.Column(scale=3):  # 이미지 영역을 더 크게 (scale=3)
                    # 이미지 업로드 및 마스킹 영역
                    # Gradio 5.46.1에서 ImageEditor 시도
                    try:
                        # ImageEditor가 있다면 사용
                        self.image_input = gr.ImageEditor(
                            label="이미지 업로드 및 마스킹 (편집 모드에서 브러시로 그리기)",
                            type="pil",
                            interactive=True,
                            height=600,  # 높이를 600px로 설정
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
                            height=600,  # 높이를 600px로 설정
                            sources=["upload"],
                            show_download_button=False,  # 불필요한 UI 제거
                            show_share_button=False,
                            mirror_webcam=False,
                        )
                    
                    # 사용법 안내
                    gr.Markdown("""
                    **사용법:**
                    1. 이미지를 업로드하세요
                    2. 편집 모드에서 마스킹할 영역을 그리세요 (색칠한 부분이 inpainting 영역이 됩니다)
                    3. '마스크 보기' 버튼을 눌러 바이너리 마스크를 확인하세요
                    
                    **마스크 형식:** 색칠한 부분은 흰색(255), 나머지는 검은색(0)으로 표시
                    """)
                    
                    # 마스크 보기 버튼
                    self.tmp_save_button = gr.Button(
                        "마스크 보기",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=2):  # 결과 영역을 중간 크기로 (scale=2)
                    # 바이너리 마스크 결과 표시
                    self.result_display = gr.Image(
                        label="생성된 바이너리 마스크 (흰색: 마스킹 영역, 검은색: 보존 영역)",
                        type="pil",
                        interactive=False,
                        height=400  # 높이를 400px로 설정
                    )
                    
                    # 마스크 상태 표시
                    self.save_status = gr.Textbox(
                        label="마스크 상태",
                        value="마스킹을 그리고 '마스크 보기' 버튼을 눌러주세요.",
                        interactive=False
                    )
        
        return tab
    
    def setup_event_handlers(self):
        """이벤트 핸들러를 설정합니다."""
        
        # 마스크 보기 버튼 클릭 시
        self.tmp_save_button.click(
            fn=self._show_mask_result,
            inputs=[self.image_input],
            outputs=[self.result_display, self.save_status]
        )
    
    def _show_mask_result(
        self, 
        image_data: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        바이너리 마스크를 생성하고 화면에 표시합니다.
        
        Args:
            image_data: Gradio Image 컴포넌트에서 반환된 데이터
            
        Returns:
            Tuple[바이너리 마스크 이미지, 마스크 상태 메시지]
        """
        try:
            if image_data is None:
                return None, "❌ 이미지가 없습니다."
            
            # Gradio Image 컴포넌트에서 이미지와 마스크 추출
            original_image = None
            mask_data = None
            
            if isinstance(image_data, dict):
                # 원본 이미지
                if "background" in image_data:
                    original_image = image_data["background"]
                elif "image" in image_data:
                    original_image = image_data["image"]
                
                # 마스크 데이터
                if "layers" in image_data and len(image_data["layers"]) > 0:
                    # 첫 번째 레이어를 마스크로 사용
                    mask_data = image_data["layers"][0]
                elif "composite" in image_data and original_image is not None:
                    # composite와 원본 이미지 차이로 마스크 생성
                    composite_image = image_data["composite"]
                    mask_data = self._extract_mask_from_composite(original_image, composite_image)
            else:
                # 단순 이미지인 경우 전체를 마스크로 처리
                original_image = image_data
            
            if original_image is None:
                return None, "❌ 원본 이미지를 찾을 수 없습니다."
            
            # PIL Image를 numpy array로 변환
            if hasattr(original_image, 'convert'):
                original_array = np.array(original_image.convert('RGB'))
            else:
                original_array = np.array(original_image)
            
            # 바이너리 마스크 생성
            if mask_data is not None:
                # 마스크 데이터가 있는 경우
                if hasattr(mask_data, 'convert'):
                    mask_array = np.array(mask_data.convert('L'))
                else:
                    mask_array = np.array(mask_data)
                    if len(mask_array.shape) == 3:
                        mask_array = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
                
                # 바이너리 마스크로 변환 (임계값 적용)
                binary_mask = np.zeros_like(mask_array)
                binary_mask[mask_array > 50] = 255  # 그려진 부분을 흰색으로
            else:
                # 마스크 데이터가 없는 경우 전체를 검은색으로
                binary_mask = np.zeros((original_array.shape[0], original_array.shape[1]), dtype=np.uint8)
            
            # PIL Image로 변환하여 표시용으로 반환 (파일 저장 없이)
            from PIL import Image
            mask_pil = Image.fromarray(binary_mask, mode='L').convert('RGB')
            
            status_message = "✅ 바이너리 마스크가 생성되었습니다."
            
            return mask_pil, status_message
            
        except Exception as e:
            error_message = f"❌ 마스크 생성 중 오류가 발생했습니다: {str(e)}"
            return None, error_message
    
    def _extract_mask_from_composite(
        self, 
        original_image, 
        composite_image
    ) -> Optional[np.ndarray]:
        """
        원본 이미지와 composite 이미지의 차이를 이용해 마스크를 추출합니다.
        
        Args:
            original_image: 원본 이미지
            composite_image: 편집된 composite 이미지
            
        Returns:
            추출된 마스크 (None if 실패)
        """
        try:
            # PIL Image를 numpy array로 변환
            if hasattr(original_image, 'convert'):
                orig_array = np.array(original_image.convert('RGB'))
            else:
                orig_array = np.array(original_image)
                
            if hasattr(composite_image, 'convert'):
                comp_array = np.array(composite_image.convert('RGB'))
            else:
                comp_array = np.array(composite_image)
            
            # 크기가 다른 경우 리사이즈
            if orig_array.shape != comp_array.shape:
                from PIL import Image
                comp_pil = Image.fromarray(comp_array)
                comp_pil = comp_pil.resize((orig_array.shape[1], orig_array.shape[0]))
                comp_array = np.array(comp_pil)
            
            # 차이 계산
            diff = np.abs(orig_array.astype(np.float32) - comp_array.astype(np.float32))
            diff_gray = np.mean(diff, axis=2)  # RGB를 그레이스케일로
            
            # 임계값을 적용하여 마스크 생성
            mask = np.zeros_like(diff_gray, dtype=np.uint8)
            mask[diff_gray > 10] = 255  # 차이가 있는 부분을 마스크로
            
            return mask
            
        except Exception as e:
            print(f"마스크 추출 중 오류: {e}")
            return None
    
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
    
