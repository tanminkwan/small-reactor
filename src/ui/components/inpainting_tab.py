"""
Inpainting 탭 컴포넌트

Single Responsibility Principle (SRP)에 따라
Inpainting UI만을 담당하는 컴포넌트
"""

import gradio as gr
import numpy as np
import cv2
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

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
        self.prompts_dir = Path("./prompts")
        self.prompts_dir.mkdir(exist_ok=True)
    
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
                            height=400,  # 높이를 400px로 설정
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
                    3. 프롬프트를 설정하세요 (등록된 프롬프트 선택 또는 직접 입력)
                    4. '마스크 보기' 버튼을 눌러 바이너리 마스크를 확인하세요
                    
                    **마스크 형식:** 색칠한 부분은 흰색(255), 나머지는 검은색(0)으로 표시
                    """)
                    
                    # 프롬프트 입력 섹션
                    gr.Markdown("### 📝 프롬프트 설정")
                    
                    # 등록된 프롬프트 선택 (자동 로드)
                    with gr.Row():
                        self.prompt_selector = gr.Dropdown(
                            label="등록된 프롬프트 선택 (선택 시 자동 로드)",
                            choices=self._get_saved_prompts(),
                            value=None,
                            interactive=True,
                            scale=6,  # 4 → 6으로 더 길게
                            max_choices=10,  # 드롭다운에서 최대 10개 항목 표시
                            allow_custom_value=True  # 직접 입력도 가능
                        )
                        self.refresh_prompts_btn = gr.Button(
                            "🔄 새로고침",
                            size="sm", 
                            scale=1
                        )
                    
                    # Positive 프롬프트 입력
                    self.positive_prompt = gr.Textbox(
                        label="Positive 프롬프트",
                        placeholder="생성하고 싶은 이미지에 대한 상세한 설명을 입력하세요...",
                        lines=3,
                        max_lines=5,
                        interactive=True  # 편집 가능하도록 설정
                    )
                    
                    # Negative 프롬프트 입력
                    self.negative_prompt = gr.Textbox(
                        label="Negative 프롬프트",
                        placeholder="원하지 않는 요소들을 입력하세요...",
                        lines=2,
                        max_lines=4,
                        interactive=True  # 편집 가능하도록 설정
                    )
                    
                    # 프롬프트 관리 버튼들
                    with gr.Row():
                        self.clear_prompts_btn = gr.Button(
                            "🗑️ 프롬프트 초기화",
                            size="sm"
                        )
                    
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
                    
                    # 상태 표시
                    self.save_status = gr.Textbox(
                        label="상태",
                        value="마스킹과 프롬프트를 설정한 후 '마스크 보기' 버튼을 눌러주세요.",
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
        
        # 프롬프트 관련 이벤트 핸들러
        # 드롭다운 선택 시 자동으로 프롬프트 로드
        self.prompt_selector.change(
            fn=self._load_selected_prompt,
            inputs=[self.prompt_selector],
            outputs=[self.positive_prompt, self.negative_prompt, self.save_status]
        )
        
        self.refresh_prompts_btn.click(
            fn=self._refresh_prompt_list,
            inputs=[],
            outputs=[self.prompt_selector]
        )
        
        # 프롬프트 초기화 버튼 클릭
        self.clear_prompts_btn.click(
            fn=self._clear_prompts,
            inputs=[],
            outputs=[self.positive_prompt, self.negative_prompt, self.prompt_selector, self.save_status]
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
    
    def _get_saved_prompts(self) -> List[str]:
        """
        저장된 프롬프트 파일들의 목록을 반환합니다.
        
        Returns:
            프롬프트 파일명 목록 (확장자 제외)
        """
        try:
            if not self.prompts_dir.exists():
                return []
            
            json_files = list(self.prompts_dir.glob("*.json"))
            return [f.stem for f in json_files]
            
        except Exception as e:
            print(f"프롬프트 목록 조회 실패: {e}")
            return []
    
    def _load_selected_prompt(
        self, 
        selected_prompt: str
    ) -> Tuple[str, str, str]:
        """
        선택된 프롬프트를 로드합니다.
        
        Args:
            selected_prompt: 선택된 프롬프트 파일명
            
        Returns:
            Tuple[positive 프롬프트, negative 프롬프트, 상태 메시지]
        """
        try:
            if not selected_prompt:
                return "", "", "프롬프트를 선택해주세요."
            
            filepath = self.prompts_dir / f"{selected_prompt}.json"
            
            if not filepath.exists():
                return "", "", f"❌ 파일을 찾을 수 없습니다: {selected_prompt}.json"
            
            # JSON 파일 읽기
            with open(filepath, 'r', encoding='utf-8') as f:
                prompt_data = json.load(f)
            
            positive = prompt_data.get("positive", "")
            negative = prompt_data.get("negative", "")
            title = prompt_data.get("title", selected_prompt)
            
            status = f"✅ 프롬프트를 불러왔습니다: {title}"
            
            return positive, negative, status
            
        except Exception as e:
            return "", "", f"❌ 프롬프트 로드 중 오류가 발생했습니다: {str(e)}"
    
    def _refresh_prompt_list(self) -> gr.Dropdown:
        """
        프롬프트 목록을 새로고침합니다.
        
        Returns:
            업데이트된 Dropdown 컴포넌트
        """
        try:
            prompt_choices = self._get_saved_prompts()
            return gr.Dropdown(choices=prompt_choices, value=None)
        except Exception as e:
            print(f"프롬프트 목록 새로고침 실패: {e}")
            return gr.Dropdown(choices=[], value=None)
    
    def _clear_prompts(self) -> Tuple[str, str, gr.Dropdown, str]:
        """
        모든 프롬프트 입력을 초기화합니다.
        
        Returns:
            Tuple[positive 초기화, negative 초기화, 드롭다운 초기화, 상태 메시지]
        """
        return (
            "",  # positive_prompt 초기화
            "",  # negative_prompt 초기화
            gr.Dropdown(value=None),  # prompt_selector 초기화
            "✅ 모든 프롬프트 입력이 초기화되었습니다."  # save_status
        )
    
