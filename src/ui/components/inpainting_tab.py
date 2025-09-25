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
from src.services.inpaint_service import InpaintService
from src.utils.file_utils import open_save_location, get_save_location_status


class InpaintingTab:
    """Inpainting 탭 컴포넌트"""
    
    def __init__(self, file_manager: FileManager, inpaint_service: InpaintService):
        """
        초기화
        
        Args:
            file_manager: 파일 관리 서비스
            inpaint_service: Inpainting 서비스
        """
        self.file_manager = file_manager
        self.inpaint_service = inpaint_service
        self.prompts_dir = Path("./prompts")
        self.prompts_dir.mkdir(exist_ok=True)
        
        # Inpainting 기능 가용성 체크
        self.is_inpaint_available = inpaint_service.is_available()
        self.unavailable_reason = inpaint_service.get_unavailable_reason() if not self.is_inpaint_available else ""
        
        # 마지막 저장된 결과 파일 경로
        self.last_result_path = None
    
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
                    
                    # 생성 파라미터 설정
                    gr.Markdown("### ⚙️ 생성 파라미터")
                    
                    with gr.Row():
                        self.guidance_scale = gr.Slider(
                            label="Guidance Scale",
                            minimum=1.0,
                            maximum=20.0,
                            step=0.5,
                            value=9.0,
                            info="프롬프트 따르기 강도 (7.0~12.0 권장)"
                        )
                        self.num_inference_steps = gr.Slider(
                            label="Inference Steps",
                            minimum=10,
                            maximum=150,
                            step=5,
                            value=50,
                            info="생성 품질 vs 속도 (30~80 권장)"
                        )
                    
                    with gr.Row():
                        self.strength = gr.Slider(
                            label="Strength (변경 강도)",
                            minimum=0.1,
                            maximum=1.0,
                            step=0.05,
                            value=1.0,
                            info="0.1=원본 유지하며 살짝 변경, 1.0=마스크 영역 완전히 새로 생성"
                        )
                        self.mask_blur = gr.Slider(
                            label="Mask Blur (경계 블러)",
                            minimum=0,
                            maximum=20,
                            step=1,
                            value=4,
                            info="마스크 경계를 부드럽게 처리"
                        )
                    
                    # 파라미터 설명
                    gr.Markdown("""
                    **파라미터 가이드:**
                    - **Guidance Scale**: 낮을수록 자유로운 생성, 높을수록 프롬프트에 충실
                    - **Inference Steps**: 실제 수행할 스텝 수 (자동 보정됨)
                    - **Strength**: 마스크 영역 변경 강도 (0.1=원본 유지, 1.0=완전 재생성)
                    - **Mask Blur**: 마스크 경계 부드럽게 처리 (4 = 기본값)
                    
                    ⚠️ **참고**: Strength 값에 따라 실제 스텝 수가 자동 조정됩니다.
                    """, elem_classes=["parameter-guide"])
                    
                    # 버튼들
                    with gr.Row():
                        self.tmp_save_button = gr.Button(
                            "마스크 보기",
                            variant="secondary",
                            size="lg",
                            scale=1
                        )
                        self.generate_button = gr.Button(
                            "🎨 이미지 생성" if self.is_inpaint_available else "❌ Inpainting 사용 불가",
                            variant="primary" if self.is_inpaint_available else "secondary",
                            size="lg",
                            scale=2,
                            interactive=self.is_inpaint_available
                        )
                
                with gr.Column(scale=2):  # 결과 영역을 중간 크기로 (scale=2)
                    # 바이너리 마스크 결과 표시
                    self.result_display = gr.Image(
                        label="생성된 바이너리 마스크 (흰색: 마스킹 영역, 검은색: 보존 영역)",
                        type="pil",
                        interactive=False,
                        height=300  # 높이를 300px로 설정
                    )
                    
                    # 최종 생성된 이미지 표시
                    with gr.Group():
                        self.output_image = gr.Image(
                            label="최종결과",
                            type="pil",
                            interactive=False,
                            height=400,  # 높이를 400px로 설정
                            visible=True  # 항상 보이도록 설정
                        )
                        
                        # 결과 이미지 관리 버튼
                        with gr.Row():
                            self.delete_result_btn = gr.Button(
                                "🗑️ 결과이미지 삭제",
                                variant="secondary",
                                size="sm"
                            )
                            self.move_to_edit_btn = gr.Button(
                                "📝 편집모드로 이동",
                                variant="secondary",
                                size="sm"
                            )
                            self.save_location_btn = gr.Button(
                                "📁 저장 위치 확인",
                                variant="secondary",
                                size="sm"
                            )
                    
                    # 상태 표시
                    initial_status = (
                        self.unavailable_reason if not self.is_inpaint_available 
                        else "마스킹과 프롬프트를 설정한 후 '마스크 보기' 버튼을 눌러주세요."
                    )
                    self.save_status = gr.Textbox(
                        label="상태",
                        value=initial_status,
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
        
        # 이미지 생성 버튼 클릭 (사용 가능할 때만 실제 처리)
        if self.is_inpaint_available:
            self.generate_button.click(
                fn=self._generate_inpaint_image,
                inputs=[
                    self.image_input,
                    self.positive_prompt,
                    self.negative_prompt,
                    self.guidance_scale,
                    self.num_inference_steps,
                    self.strength,
                    self.mask_blur
                ],
                outputs=[
                    self.output_image,
                    self.save_status
                ]
            )
        else:
            # Inpainting 사용 불가 시 에러 메시지만 표시
            self.generate_button.click(
                fn=lambda *args: (None, f"❌ {self.unavailable_reason}"),
                inputs=[
                    self.image_input,
                    self.positive_prompt,
                    self.negative_prompt,
                    self.guidance_scale,
                    self.num_inference_steps,
                    self.strength,
                    self.mask_blur
                ],
                outputs=[
                    self.output_image,
                    self.save_status
                ]
            )
        
        # 결과 이미지 삭제 버튼 클릭
        self.delete_result_btn.click(
            fn=self._delete_result_image_with_visibility,
            inputs=[],
            outputs=[
                self.output_image,
                self.save_status
            ],
            show_progress=False
        )
        
        # 편집모드로 이동 버튼 클릭
        self.move_to_edit_btn.click(
            fn=self._move_to_edit_mode,
            inputs=[self.output_image],
            outputs=[
                self.image_input,
                self.save_status
            ]
        )
        
        # 저장 위치 확인 버튼 클릭
        self.save_location_btn.click(
            fn=self._open_save_location,
            inputs=[],
            outputs=[self.save_status]
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
    
    def _generate_inpaint_image(
        self,
        image_data: Dict[str, Any],
        positive_prompt: str,
        negative_prompt: str,
        guidance_scale: float,
        num_inference_steps: int,
        strength: float,
        mask_blur: int
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        Inpainting을 수행하여 최종 이미지를 생성합니다.
        
        Args:
            image_data: Gradio Image 컴포넌트에서 반환된 데이터
            positive_prompt: Positive 프롬프트
            negative_prompt: Negative 프롬프트
            guidance_scale: 가이던스 스케일 값
            num_inference_steps: 추론 스텝 수
            strength: 수정 강도 (0.1=마스크만, 1.0=주변까지)
            mask_blur: 마스크 경계 블러 정도
            
        Returns:
            Tuple[생성된 이미지, 상태 메시지]
        """
        try:
            if image_data is None:
                return None, "❌ 이미지가 없습니다."
            
            if not positive_prompt.strip():
                return None, "❌ Positive 프롬프트를 입력해주세요."
            
            # 이미지와 마스크 추출
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
                    mask_data = image_data["layers"][0]
                elif "composite" in image_data and original_image is not None:
                    composite_image = image_data["composite"]
                    mask_data = self._extract_mask_from_composite(original_image, composite_image)
            else:
                original_image = image_data
            
            if original_image is None:
                return None, "❌ 원본 이미지를 찾을 수 없습니다."
            
            # PIL Image를 numpy array로 변환 (BGR 형식)
            if hasattr(original_image, 'convert'):
                original_array = np.array(original_image.convert('RGB'))
                original_bgr = cv2.cvtColor(original_array, cv2.COLOR_RGB2BGR)
            else:
                original_bgr = np.array(original_image)
            
            # 마스크 생성
            if mask_data is not None:
                if hasattr(mask_data, 'convert'):
                    mask_array = np.array(mask_data.convert('L'))
                else:
                    mask_array = np.array(mask_data)
                    if len(mask_array.shape) == 3:
                        mask_array = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
                
                # 바이너리 마스크로 변환
                binary_mask = np.zeros_like(mask_array)
                binary_mask[mask_array > 50] = 255
            else:
                return None, "❌ 마스크 영역이 없습니다. 이미지에 마스킹을 해주세요."
            
            # 마스크가 비어있는지 확인
            if np.sum(binary_mask) == 0:
                return None, "❌ 마스크 영역이 없습니다. 이미지에 마스킹을 해주세요."
            
            # 마스크 블러 처리 적용
            if mask_blur > 0:
                binary_mask = cv2.GaussianBlur(binary_mask, (mask_blur*2+1, mask_blur*2+1), 0)
            
            # Inpainting 수행 (사용자 입력 파라미터 사용)
            result_bgr = self.inpaint_service.generate_inpaint_image(
                image=original_bgr,
                mask=binary_mask,
                prompt=positive_prompt,
                negative_prompt=negative_prompt or "",
                guidance_scale=guidance_scale,
                num_inference_steps=int(num_inference_steps),
                strength=strength
            )
            
            # BGR to RGB 변환 후 PIL Image로 변환
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            from PIL import Image
            result_pil = Image.fromarray(result_rgb)
            
            # 자동으로 이미지 저장
            saved_path = self.inpaint_service.save_image(result_bgr)
            self.last_result_path = saved_path  # 마지막 저장 경로 기록
            
            return (
                result_pil,
                f"✅ 이미지 생성 및 저장이 완료되었습니다!\n파라미터: Guidance={guidance_scale}, Steps={int(num_inference_steps)}, Strength={strength}, Blur={mask_blur}\n저장 위치: {saved_path}"
            )
            
        except Exception as e:
            error_message = f"❌ 이미지 생성 중 오류가 발생했습니다: {str(e)}"
            return None, error_message
    
    def _delete_result_image_with_visibility(self) -> Tuple[gr.update, str]:
        """
        가장 최근 생성된 inpaint 결과 이미지 파일을 삭제하고 UI에서도 제거합니다.
        
        Returns:
            Tuple[이미지 업데이트 (값 클리어), 상태 메시지]
        """
        try:
            # 실제 파일 시스템에서 가장 최근 inpaint 결과 파일 삭제
            success, message = self.inpaint_service.delete_latest_result_image()
            
            if success:
                # 삭제 성공 시 UI에서도 이미지 제거
                return gr.update(value=None), message
            else:
                # 삭제 실패 시 UI는 그대로 두고 메시지만 표시
                return gr.update(), message
                
        except Exception as e:
            return gr.update(), f"❌ 이미지 삭제 중 오류가 발생했습니다: {str(e)}"
    
    def _move_to_edit_mode(self, output_image) -> Tuple[gr.update, str]:
        """
        결과 이미지를 편집 모드로 이동합니다 (이미지 업로드 및 마스킹 영역으로).
        
        Args:
            output_image: 결과 이미지
            
        Returns:
            Tuple[이미지 업데이트, 상태 메시지]
        """
        try:
            if output_image is None:
                return gr.update(), "❌ 이동할 결과 이미지가 없습니다."
            
            return gr.update(value=output_image), "✅ 결과 이미지가 편집 모드로 이동되었습니다. 새로운 마스킹을 진행해주세요."

        except Exception as e:
            return gr.update(), f"❌ 편집 모드로 이동 중 오류가 발생했습니다: {str(e)}"
    
    def _open_save_location(self) -> str:
        """
        저장 위치를 파일 탐색기로 열고 상태 메시지를 반환합니다.
        
        Returns:
            실행 결과 메시지
        """
        try:
            # Inpainting 결과 저장 경로 가져오기
            output_path = self.inpaint_service.config.get(
                "inpaint_output_path", 
                "./outputs/inpaint"
            )
            
            return open_save_location(output_path, self.last_result_path)
            
        except Exception as e:
            return f"❌ 저장 위치 열기 중 오류가 발생했습니다: {str(e)}"
    
