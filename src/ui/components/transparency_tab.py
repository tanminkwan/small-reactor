"""
이미지 투명화 탭 컴포넌트

Single Responsibility Principle (SRP)에 따라
이미지 투명화 UI만을 담당하는 컴포넌트
"""

import gradio as gr
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
import os
from datetime import datetime

from src.services.file_manager import FileManager
from src.utils import Config
from src.utils.file_utils import open_save_location, get_save_location_status


class TransparencyTab:
    """이미지 투명화 탭 컴포넌트"""
    
    def __init__(self, file_manager: FileManager):
        """
        초기화
        
        Args:
            file_manager: 파일 관리 서비스
        """
        self.file_manager = file_manager
        self.config = Config()
        self.last_result_path = None
    
    def create_interface(self) -> gr.Tab:
        """
        이미지 투명화 탭 인터페이스를 생성합니다.
        
        Returns:
            Gradio Tab 컴포넌트
        """
        with gr.Tab("이미지 투명화") as tab:
            gr.Markdown("## 🔍 이미지 투명화")
            
            with gr.Row():
                with gr.Column(scale=3):  # 이미지 영역을 더 크게 (scale=3)
                    # 이미지 업로드 및 마스킹 영역
                    try:
                        # ImageEditor가 있다면 사용
                        self.image_input = gr.ImageEditor(
                            label="이미지 업로드 및 마스킹 (편집 모드에서 브러시로 투명화할 영역 그리기)",
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
                    2. 편집 모드에서 투명화할 영역을 브러시로 그리세요 (색칠한 부분이 투명해집니다)
                    3. '투명화 생성' 버튼을 눌러 투명 이미지를 생성하세요
                    
                    **특징:** 브러시로 칠한 영역이 직접 투명화됩니다 (마스크 추출 과정 없음)
                    """)
                    
                    # 파라미터 설정
                    gr.Markdown("### ⚙️ 투명화 설정")
                    
                    with gr.Row():
                        self.mask_blur = gr.Slider(
                            label="경계 부드럽게 (Blur)",
                            minimum=0,
                            maximum=10,
                            step=1,
                            value=2,
                            info="투명 영역 경계를 부드럽게 처리 (0=선명한 경계, 10=매우 부드러운 경계)"
                        )
                        self.transparency_strength = gr.Slider(
                            label="투명도 강도",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=1.0,
                            info="0.0=반투명, 1.0=완전 투명"
                        )
                    
                    # 버튼
                    self.generate_button = gr.Button(
                        "🔍 투명화 생성",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=2):  # 결과 영역을 중간 크기로 (scale=2)
                    # 최종 투명화된 이미지 표시
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
                    self.save_status = gr.Textbox(
                        label="상태",
                        value="이미지를 업로드하고 투명화할 영역을 브러시로 그린 후 '투명화 생성' 버튼을 눌러주세요.",
                        interactive=False
                    )
        
        return tab
    
    def setup_event_handlers(self):
        """이벤트 핸들러를 설정합니다."""
        
        # 투명화 생성 버튼 클릭 시
        self.generate_button.click(
            fn=self._generate_transparent_image,
            inputs=[
                self.image_input,
                self.mask_blur,
                self.transparency_strength
            ],
            outputs=[self.output_image, self.save_status]
        )
        
        # 결과이미지 삭제 버튼 클릭
        self.delete_result_btn.click(
            fn=self._delete_result_image_with_visibility,
            inputs=[],
            outputs=[self.output_image, self.save_status]
        )
        
        # 편집모드로 이동 버튼 클릭
        self.move_to_edit_btn.click(
            fn=self._move_to_edit_mode,
            inputs=[self.output_image],
            outputs=[self.image_input, self.save_status]
        )
        
        # 저장 위치 확인 버튼 클릭
        self.save_location_btn.click(
            fn=self._open_save_location,
            inputs=[],
            outputs=[self.save_status]
        )
    
    def _generate_transparent_image(
        self,
        edited_image,
        blur_value: int,
        transparency_strength: float
    ) -> Tuple[Optional[Image.Image], str]:
        """
        브러시로 칠한 영역을 직접 투명화 처리합니다.
        
        Args:
            edited_image: 편집된 이미지 (브러시 포함)
            blur_value: 경계 블러 값
            transparency_strength: 투명도 강도
            
        Returns:
            Tuple[투명화된 이미지, 상태 메시지]
        """
        try:
            if edited_image is None:
                return None, "❌ 이미지를 먼저 업로드해주세요."
            
            # 브러시로 칠한 영역을 직접 투명화 처리
            transparent_image = self._apply_direct_transparency(
                edited_image, 
                blur_value, 
                transparency_strength
            )
            
            if transparent_image is None:
                return None, "❌ 투명화 처리에 실패했습니다. 편집 모드에서 브러시로 영역을 그려주세요."
            
            # 자동 저장
            save_path = self._save_result(transparent_image)
            if save_path:
                self.last_result_path = save_path
                status = f"✅ 투명화 생성 완료!\n📁 저장 위치: {save_path}"
            else:
                status = "✅ 투명화 생성 완료! (저장 실패)"
            
            return transparent_image, status
            
        except Exception as e:
            return None, f"❌ 투명화 생성 실패: {str(e)}"
    
    def _apply_direct_transparency(
        self,
        edited_image,
        blur_value: int,
        transparency_strength: float
    ) -> Optional[Image.Image]:
        """브러시로 칠한 영역을 직접 투명화 처리합니다."""
        try:
            # 디버깅 정보 출력
            print(f"🔍 편집된 이미지 타입: {type(edited_image)}")
            
            # Inpainting 탭과 동일한 방식으로 처리
            original_image = None
            mask_data = None
            
            if isinstance(edited_image, dict):
                print("🔍 딕셔너리 형태의 데이터")
                print(f"🔍 딕셔너리 키들: {list(edited_image.keys())}")
                
                # 원본 이미진지
                if "background" in edited_image:
                    original_image = edited_image["background"]
                    print("✅ background에서 원본 이미지 추출")
                elif "image" in edited_image:
                    original_image = edited_image["image"]
                    print("✅ image에서 원본 이미지 추출")
                
                # 마스크 데이터 (브러시 영역)
                if "layers" in edited_image and len(edited_image["layers"]) > 0:
                    mask_data = edited_image["layers"][0]  # 첫 번째 레이어를 마스크로 사용
                    print(f"✅ layers에서 마스크 데이터 추출 ({len(edited_image['layers'])}개 레이어)")
                elif "composite" in edited_image and original_image is not None:
                    # composite와 원본 이미지 차이로 마스크 생성
                    composite_image = edited_image["composite"]
                    mask_data = self._extract_mask_from_composite(original_image, composite_image)
                    print("✅ composite와 background 차이로 마스크 생성")
            elif hasattr(edited_image, 'layers') and len(edited_image.layers) >= 2:
                # 기존 방식: PIL 이미지의 layers 속성
                original_image = edited_image.layers[0]
                mask_data = edited_image.layers[1]
                print("✅ PIL 이미지 layers 속성 사용")
            elif hasattr(edited_image, 'background'):
                # background 속성
                original_image = edited_image.background
                if hasattr(edited_image, 'layers') and len(edited_image.layers) > 0:
                    mask_data = edited_image.layers[0]
                print("✅ PIL 이미지 background 속성 사용")
            else:
                # 단순 이미지인 경우
                original_image = edited_image
                print("✅ 단순 이미지 처리")
            
            if original_image is None:
                print("❌ 원본 이미지를 찾을 수 없음")
                return None
            
            # 원본을 RGBA로 변환
            if original_image.mode != 'RGBA':
                rgba_image = original_image.convert('RGBA')
            else:
                rgba_image = original_image.copy()
            
            print(f"🔍 원본 이미지 모드: {rgba_image.mode}, 크기: {rgba_image.size}")
            
            # 마스크 데이터에서 브러시 영역 감지 (Inpainting 탭 방식)
            brush_mask = None
            
            if mask_data is not None:
                print(f"🔍 마스크 데이터 처리: {type(mask_data)}")
                
                # 마스크 데이터를 numpy 배열로 변환
                if hasattr(mask_data, 'convert'):
                    mask_array = np.array(mask_data.convert('L'))
                else:
                    mask_array = np.array(mask_data)
                    if len(mask_array.shape) == 3:
                        mask_array = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
                
                print(f"🔍 마스크 배열 모양: {mask_array.shape}")
                
                # 브러시 영역 감지 (임계값 적용)
                brush_mask = mask_array > 50  # Inpainting 탭과 동일한 임계값
                print(f"🔍 브러시 영역 감지: {np.sum(brush_mask)}개 픽셀")
            else:
                print("❌ 마스크 데이터가 없음")
            
            # 브러시 영역이 없으면 None 반환
            if brush_mask is None or not np.any(brush_mask):
                print("❌ 브러시 영역을 찾을 수 없음")
                return None
            
            print(f"✅ 브러시 영역 감지 완료: {np.sum(brush_mask)}개 픽셀")
            
            # 브러시 마스크를 0-255 범위로 변환
            mask = brush_mask.astype(np.uint8) * 255
            
            # 마스크 블러 적용
            if blur_value > 0:
                mask_blurred = cv2.GaussianBlur(mask, (blur_value*2+1, blur_value*2+1), blur_value/3)
            else:
                mask_blurred = mask
            
            # Alpha 채널 수정
            rgba_array = np.array(rgba_image)
            
            # 브러시로 칠한 부분의 투명도 조정
            alpha_channel = rgba_array[:, :, 3].astype(np.float32)
            mask_normalized = mask_blurred.astype(np.float32) / 255.0
            
            # 투명도 강도 적용
            alpha_reduction = mask_normalized * transparency_strength
            new_alpha = alpha_channel * (1.0 - alpha_reduction)
            
            rgba_array[:, :, 3] = np.clip(new_alpha, 0, 255).astype(np.uint8)
            
            return Image.fromarray(rgba_array, 'RGBA')
            
        except Exception as e:
            print(f"투명화 처리 오류: {e}")
            return None
    
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
            
            # 두 이미지의 차이를 계산
            diff = np.abs(orig_array.astype(np.float32) - comp_array.astype(np.float32))
            
            # 채널별 차이를 합산하여 전체 차이 계산
            total_diff = np.sum(diff, axis=2)
            
            # 임계값을 적용하여 맄스크 생성 (차이가 큰 부분을 마스크로 처리)
            threshold = 30.0  # 임계값 조정 가능
            mask = (total_diff > threshold).astype(np.uint8) * 255
            
            print(f"🔍 composite 차이 기반 마스크: {np.sum(mask > 0)}개 픽셀")
            
            return mask
            
        except Exception as e:
            print(f"마스크 추출 오류: {e}")
            return None
    
    def _save_result(self, result_image: Image.Image) -> Optional[str]:
        """결과 이미지를 자동 저장합니다."""
        try:
            # 저장 경로 설정 (object_extraction과 같은 폴더 사용)
            output_path = self.config.get(
                "object_extraction_output_path", 
                "./output/object_extraction"
            )
            
            # 디렉토리 생성
            os.makedirs(output_path, exist_ok=True)
            
            # 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transparent_{timestamp}.png"
            
            # 전체 경로
            save_path = os.path.join(output_path, filename)
            
            # PNG로 저장 (투명도 보존)
            result_image.save(save_path, "PNG")
            
            return save_path
            
        except Exception as e:
            print(f"❌ 저장 실패: {e}")
            return None
    
    def _delete_result_image_with_visibility(self) -> Tuple[gr.update, str]:
        """결과 이미지를 삭제합니다."""
        try:
            if self.last_result_path and os.path.exists(self.last_result_path):
                os.remove(self.last_result_path)
                message = f"✅ 결과 이미지가 삭제되었습니다.\n📁 삭제된 파일: {self.last_result_path}"
                self.last_result_path = None
                return gr.update(value=None), message
            else:
                return gr.update(), "❌ 삭제할 결과 이미지가 없습니다."
                
        except Exception as e:
            return gr.update(), f"❌ 이미지 삭제 중 오류가 발생했습니다: {str(e)}"
    
    def _move_to_edit_mode(self, output_image) -> Tuple[gr.update, str]:
        """결과 이미지를 편집 모드로 이동합니다. RGBA 이미지의 alpha 채널을 시각적으로 처리합니다."""
        try:
            if output_image is None:
                return gr.update(), "❌ 이동할 결과 이미지가 없습니다."
            
            # RGBA 이미지인 경우 alpha 채널 시각화 처리
            processed_image = self._prepare_image_for_editor(output_image)
            
            return gr.update(value=processed_image), "✅ 결과 이미지가 편집 모드로 이동되었습니다. 추가 마스킹을 진행해주세요."
            
        except Exception as e:
            return gr.update(), f"❌ 편집 모드로 이동 중 오류가 발생했습니다: {str(e)}"
    
    def _open_save_location(self) -> str:
        """
        저장 위치를 파일 탐색기로 열고 상태 메시지를 반환합니다.
        
        Returns:
            실행 결과 메시지
        """
        try:
            # 이미지 투명화 결과 저장 경로 가져오기
            output_path = self.config.get(
                "object_extraction_output_path", 
                "./output/object_extraction"
            )
            
            return open_save_location(output_path, self.last_result_path)
            
        except Exception as e:
            return f"❌ 저장 위치 열기 중 오류가 발생했습니다: {str(e)}"
    
    def _prepare_image_for_editor(self, image: Image.Image) -> Image.Image:
        """
        이미지를 편집기에 적합한 형태로 준비합니다.
        RGBA 이미지의 투명 영역을 체크보드 패턴으로 시각화합니다.
        
        Args:
            image: 처리할 이미지
            
        Returns:
            편집기에 적합한 RGB 이미지
        """
        try:
            if image.mode != 'RGBA':
                # RGBA가 아니면 그대로 반환
                return image
            
            print(f"🔍 RGBA 이미지를 편집기용으로 변환: {image.size}")
            
            # 체크보드 패턴 생성
            checkerboard = self._create_checkerboard_pattern(image.size)
            
            # RGBA 이미지를 numpy 배열로 변환
            rgba_array = np.array(image)
            
            # RGB 채널과 Alpha 채널 분리
            rgb_channels = rgba_array[:, :, :3]
            alpha_channel = rgba_array[:, :, 3]
            
            # Alpha 값을 0-1 범위로 정규화
            alpha_normalized = alpha_channel.astype(np.float32) / 255.0
            
            # 체크보드 패턴과 RGB 이미지를 alpha blending
            blended_image = np.zeros_like(rgb_channels)
            for c in range(3):  # RGB 채널별로 처리
                blended_image[:, :, c] = (
                    rgb_channels[:, :, c] * alpha_normalized + 
                    checkerboard * (1 - alpha_normalized)
                ).astype(np.uint8)
            
            # RGB 이미지로 변환
            result_image = Image.fromarray(blended_image, 'RGB')
            
            print(f"✅ RGBA -> RGB 변환 완료: {result_image.mode}")
            return result_image
            
        except Exception as e:
            print(f"❌ 이미지 처리 오류: {e}")
            # 오류 발생 시 원본 이미지 반환
            return image
    
    def _create_checkerboard_pattern(self, size: tuple, checker_size: int = 20) -> np.ndarray:
        """
        체크보드 패턴을 생성합니다.
        
        Args:
            size: 이미지 크기 (width, height)
            checker_size: 체크보드 간격
            
        Returns:
            체크보드 패턴 배열
        """
        width, height = size
        
        # 체크보드 패턴 생성
        checkerboard = np.zeros((height, width), dtype=np.uint8)
        
        for y in range(height):
            for x in range(width):
                # 체크보드 패턴: 밝은 회색(220)과 어두운 회색(200) 교대
                if (x // checker_size + y // checker_size) % 2 == 0:
                    checkerboard[y, x] = 220  # 밝은 회색
                else:
                    checkerboard[y, x] = 200  # 어두운 회색
        
        return checkerboard