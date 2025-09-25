"""
사물 추출 탭 컴포넌트

Single Responsibility Principle (SRP)에 따라
사물 추출 UI만을 담당하는 컴포넌트
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
from src.utils import Config, load_sam_model, extract_object_with_sam
from src.utils.file_utils import open_save_location, get_save_location_status


class ObjectExtractionTab:
    """사물 추출 탭 컴포넌트"""
    
    def __init__(self, file_manager: FileManager):
        """
        초기화
        
        Args:
            file_manager: 파일 관리 서비스
        """
        self.file_manager = file_manager
        self.config = Config()
        self.sam_predictor = None
        self.last_result_path = None
        self.current_result_image = None  # 현재 결과 이미지 저장
        # 회전 관련 변수 추가
        self.original_rotation_image = None  # 회전용 원본 이미지 저장
        # SAM 시작 좌표 변수 추가
        self.sam_click_point = None  # 사용자 클릭 좌표 (x, y)
    
    def create_interface(self) -> gr.Tab:
        """
        사물 추출 탭 인터페이스를 생성합니다.
        
        Returns:
            Gradio Tab 컴포넌트
        """
        with gr.Tab("사물 추출") as tab:
            gr.Markdown("## ✂️ 사물 추출")
            gr.Markdown("SAM (Segment Anything Model)을 사용하여 이미지에서 객체를 추출하고 배경을 투명하게 만듭니다.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # 원본 이미지 업로드
                    self.image_upload = gr.Image(
                        label="원본 이미지 업로드",
                        type="pil"
                    )
                    
                    # Alpha 채널 적용 버튼
                    self.extract_btn = gr.Button(
                        "Alpha 채널 적용", 
                        variant="primary",
                        size="lg"
                    )
                    
                    # 이미지 회전 컨트롤
                    gr.Markdown("### 🔄 이미지 회전")
                    with gr.Row():
                        self.rotation_angle = gr.Number(
                            label="회전 각도 (도)",
                            value=0,
                            step=5,
                            info="양수: 시계반대방향, 음수: 시계방향"
                        )
                        with gr.Column(min_width=80):
                            self.rotate_ccw_btn = gr.Button(
                                "↻ -5°",
                                variant="secondary",
                                size="sm"
                            )
                            self.rotate_cw_btn = gr.Button(
                                "↺ +5°", 
                                variant="secondary",
                                size="sm"
                            )
                    
                    self.rotate_btn = gr.Button(
                        "🔄 이미지 회전",
                        variant="secondary"
                    )
                
                with gr.Column(scale=1):
                    # 최종결과 표시
                    with gr.Group():
                        self.result_image = gr.Image(
                            label="최종결과",
                            type="pil",
                            interactive=False
                        )
                        
                        # 결과 이미지 관리 버튼
                        with gr.Row():
                            self.delete_result_btn = gr.Button(
                                "🗑️ 결과 이미지 삭제",
                                variant="secondary",
                                size="sm"
                            )
                            self.move_to_original_btn = gr.Button(
                                "📤 원본이미지로 이동",
                                variant="secondary",
                                size="sm"
                            )
                        
                        with gr.Row():
                            self.save_location_btn = gr.Button(
                                "📁 저장 위치 확인",
                                variant="secondary",
                                size="sm",
                                scale=2
                            )
                    
                    # 상태 텍스트 (오른쪽 아래)
                    self.status_text = gr.Textbox(
                        label="상태",
                        lines=4,
                        interactive=False,
                        value="이미지를 업로드한 후:\n1. 원본 이미지를 클릭하여 추출할 객체의 시작지점 선택 (선택사항)\n2. 'Alpha 채널 적용' 버튼 클릭\n\n✨ 클릭하지 않으면 이미지 중앙을 기준으로 추출합니다."
                    )
        
        return tab
    
    def setup_event_handlers(self):
        """이벤트 핸들러를 설정합니다."""
        
        # Alpha 채널 적용 버튼 클릭
        self.extract_btn.click(
            fn=self._extract_object_with_alpha,
            inputs=[self.image_upload],
            outputs=[
                self.result_image,
                self.status_text
            ]
        )
        
        # 회전 각도 조절 버튼들
        self.rotate_ccw_btn.click(
            fn=self._adjust_rotation_angle,
            inputs=[self.rotation_angle, gr.State(-5)],
            outputs=[self.rotation_angle]
        )
        
        self.rotate_cw_btn.click(
            fn=self._adjust_rotation_angle,
            inputs=[self.rotation_angle, gr.State(5)],
            outputs=[self.rotation_angle]
        )
        
        # 이미지 회전 버튼 클릭
        self.rotate_btn.click(
            fn=self._rotate_image,
            inputs=[self.image_upload, self.rotation_angle],
            outputs=[
                self.result_image,  # 최종결과에 표시
                self.status_text
            ]
        )
        
        # 결과 이미지 삭제 버튼 클릭
        self.delete_result_btn.click(
            fn=self._delete_result_image,
            inputs=[],
            outputs=[
                self.result_image,
                self.status_text
            ]
        )
        
        # 원본 이미지로 이동 버튼 클릭
        self.move_to_original_btn.click(
            fn=self._move_result_to_original,
            inputs=[],
            outputs=[
                self.image_upload,
                self.status_text
            ]
        )
        
        # 저장 위치 확인 버튼 클릭
        self.save_location_btn.click(
            fn=self._show_save_location,
            inputs=[],
            outputs=[self.status_text]
        )
        
        # 이미지 업로드 시 회전 상태 리셋
        self.image_upload.change(
            fn=lambda img: self._reset_rotation(),
            inputs=[self.image_upload],
            outputs=[]
        )
        
        # 원본 이미지 클릭 시 SAM 시작 좌표 설정
        self.image_upload.select(
            fn=self._handle_image_click,
            inputs=[self.image_upload],
            outputs=[self.status_text]
        )
    
    def _load_sam_model(self):
        """SAM 모델을 로드합니다 (한 번만 실행)"""
        if self.sam_predictor is None:
            sam_model_path = self.config.get("sam_model_path", "./sam_models")
            sam_model_type = self.config.get("sam_model_type", "vit_h")
            
            self.sam_predictor = load_sam_model(
                model_path=sam_model_path,
                model_type=sam_model_type
            )
        
        return self.sam_predictor
    
    def _extract_object_with_alpha(
        self,
        image_pil: Image.Image
    ) -> Tuple[Optional[Image.Image], str]:
        """
        이미지에서 객체를 추출하고 Alpha 채널을 적용합니다.
        
        Args:
            image_pil: 입력 PIL 이미지
            
        Returns:
            (결과 이미지, 상태 메시지)
        """
        try:
            # 입력 검증
            if image_pil is None:
                return None, "❌ 이미지를 업로드해주세요."
            
            # PIL 이미지를 numpy 배열로 변환 (BGR로 변환)
            image_rgb = np.array(image_pil)
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            status = f"📸 이미지 로드 완료: {image.shape}\n"
            
            # SAM 모델 로드
            status += "🤖 SAM 모델 로딩 중...\n"
            predictor = self._load_sam_model()
            
            if predictor is None:
                return None, status + "❌ SAM 모델을 로드할 수 없습니다."
            
            status += "✅ SAM 모델 로딩 완료\n"
            
            # 객체 추출
            if self.sam_click_point:
                status += f"🎯 객체 추출 중... (클릭지점: {self.sam_click_point})\n"
            else:
                status += "🎯 객체 추출 중... (중앙지점 사용)\n"
            
            result_pil = extract_object_with_sam(
                image=image,
                predictor=predictor,
                click_point=self.sam_click_point
            )
            
            if result_pil is None:
                return None, status + "❌ 객체 추출에 실패했습니다."
            
            # 현재 결과 이미지 저장
            self.current_result_image = result_pil
            
            # 자동 저장
            save_path = self._save_result(result_pil, "extracted_object")
            if save_path:
                self.last_result_path = save_path
                status += f"✅ 추출 완료!\n📁 저장 위치: {save_path}"
            else:
                status += "✅ 추출 완료! (저장 실패)"
            
            return result_pil, status
            
        except Exception as e:
            return None, f"❌ 오류 발생: {str(e)}"
    
    def _save_result(self, result_image: Image.Image, original_path: str) -> Optional[str]:
        """
        결과 이미지를 자동 저장합니다.
        
        Args:
            result_image: 저장할 이미지
            original_path: 원본 이미지 경로
            
        Returns:
            저장된 파일 경로 또는 None (실패 시)
        """
        try:
            # 저장 경로 설정
            output_path = self.config.get(
                "object_extraction_output_path", 
                "./output/object_extraction"
            )
            
            # 디렉토리 생성
            os.makedirs(output_path, exist_ok=True)
            
            # 파일명 생성 (원본 파일명 + 타임스탬프)
            if os.path.exists(original_path):
                original_name = Path(original_path).stem
            else:
                original_name = original_path  # 기본 이름으로 사용
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{original_name}_extracted_{timestamp}.png"
            
            # 전체 경로
            save_path = os.path.join(output_path, filename)
            
            # 저장
            result_image.save(save_path, "PNG")
            
            return save_path
            
        except Exception as e:
            print(f"❌ 저장 실패: {e}")
            return None
    
    def _delete_result_image(self) -> Tuple[Optional[Image.Image], str]:
        """
        결과 이미지를 삭제합니다.
        
        Returns:
            (None, 상태 메시지)
        """
        try:
            if self.last_result_path and os.path.exists(self.last_result_path):
                os.remove(self.last_result_path)
                message = f"✅ 결과 이미지가 삭제되었습니다.\n📁 삭제된 파일: {self.last_result_path}"
                self.last_result_path = None
                self.current_result_image = None  # 현재 결과 이미지도 초기화
            else:
                message = "❌ 삭제할 결과 이미지가 없습니다."
            
            return None, message
            
        except Exception as e:
            return None, f"❌ 삭제 실패: {str(e)}"
    
    def _move_result_to_original(self) -> Tuple[Optional[Image.Image], str]:
        """
        결과 이미지를 원본 이미지 업로드 영역으로 이동합니다.
        
        Returns:
            (결과 이미지, 상태 메시지)
        """
        try:
            if self.current_result_image is not None:
                return self.current_result_image, "✅ 결과 이미지가 원본 이미지 영역으로 이동되었습니다.\n새로운 추출 작업을 진행할 수 있습니다."
            else:
                return None, "❌ 이동할 결과 이미지가 없습니다."
                
        except Exception as e:
            return None, f"❌ 이동 실패: {str(e)}"
    
    def _show_save_location(self) -> str:
        """
        저장 위치를 파일 탐색기로 열고 상태 메시지를 반환합니다.
        
        Returns:
            실행 결과 메시지
        """
        try:
            # 사물 추출 결과 저장 경로 가져오기
            output_path = self.config.get(
                "object_extraction_output_path", 
                "./output/object_extraction"
            )
            
            return open_save_location(output_path, self.last_result_path)
            
        except Exception as e:
            return f"❌ 저장 위치 열기 중 오류가 발생했습니다: {str(e)}"
    
    def _adjust_rotation_angle(self, current_angle: float, adjustment: float) -> float:
        """
        회전 각도를 조절합니다.
        
        Args:
            current_angle: 현재 각도
            adjustment: 조절할 각도
            
        Returns:
            새로운 각도
        """
        # 현재 각도에 조절값을 더한 절대 각도
        new_angle = current_angle + adjustment
        # -360도에서 360도 범위로 제한
        return max(-360, min(360, new_angle))
    
    def _rotate_image(
        self, 
        image_pil: Image.Image, 
        angle: float
    ) -> Tuple[Optional[Image.Image], str]:
        """
        이미지를 회전시킵니다. 항상 원본을 현재 각도로 한 번만 회전합니다.
        
        Args:
            image_pil: 현재 표시된 이미지 (참조용, 원본이 없을 때만 사용)
            angle: 회전할 각도 (절대값)
            
        Returns:
            (회전된 이미지, 상태 메시지)
        """
        try:
            # 원본 이미지가 없으면 오류
            if self.original_rotation_image is None:
                if image_pil is None:
                    return None, "❌ 회전할 이미지가 없습니다."
                
                # 최초 원본 이미지 저장 (Alpha 채널 제거)
                if image_pil.mode == 'RGBA':
                    rgb_image = Image.new('RGB', image_pil.size, (0, 0, 0))
                    rgb_image.paste(image_pil, mask=image_pil.split()[3])
                    self.original_rotation_image = rgb_image
                else:
                    self.original_rotation_image = image_pil.convert('RGB')
                
                print(f"🎯 원본 이미지 저장: {self.original_rotation_image.size}")
            
            if angle == 0:
                return self.original_rotation_image, "🔄 회전 각도가 0도입니다."
            
            # 항상 저장된 원본만 사용하여 회전 (image_pil 무시)
            print(f"🔄 원본({self.original_rotation_image.size})을 {angle}도 회전")
            
            rotated_image = self.original_rotation_image.rotate(
                angle=angle,
                expand=True,
                fillcolor=(0, 0, 0)  # 검은색 배경
            )
            
            # Alpha 채널이 있으면 RGB로 강제 변환 (확실히 제거)
            if rotated_image.mode != 'RGB':
                if rotated_image.mode == 'RGBA':
                    # RGBA -> RGB 변환 (검은색 배경)
                    rgb_result = Image.new('RGB', rotated_image.size, (0, 0, 0))
                    rgb_result.paste(rotated_image, mask=rotated_image.split()[3])
                    rotated_image = rgb_result
                    print("⚠️ RGBA를 RGB로 강제 변환")
                else:
                    rotated_image = rotated_image.convert('RGB')
                    print(f"⚠️ {rotated_image.mode}를 RGB로 변환")
            
            print(f"✅ 최종 회전 결과: {rotated_image.mode} 모드, 크기 {rotated_image.size}")
            
            # 회전된 이미지를 현재 결과로 저장
            self.current_result_image = rotated_image
            
            status = f"✅ 이미지 회전 완료!\n🔄 회전 각도: {angle}도\n📏 원본 크기: {self.original_rotation_image.size}\n📏 새 크기: {rotated_image.size}\n⬛ Alpha 채널 제거됨 (RGB 모드)\n🎯 원본에서 한 번만 회전 (화질 보존)"
            
            return rotated_image, status
            
        except Exception as e:
            return None, f"❌ 회전 실패: {str(e)}"
    
    def _handle_image_click(self, evt: gr.SelectData, image_pil: Image.Image) -> str:
        """
        이미지 클릭 시 SAM 시작 좌표를 설정합니다.
        
        Args:
            evt: Gradio SelectData 이벤트
            image_pil: 클릭된 이미지
            
        Returns:
            상태 메시지
        """
        try:
            if evt.index is None or len(evt.index) < 2:
                return "❌ 잘못된 클릭 좌표입니다."
            
            # 클릭 좌표 저장 (x, y)
            self.sam_click_point = (evt.index[0], evt.index[1])
            
            status = f"🎯 SAM 시작점 설정 완료!\n📍 클릭 좌표: ({self.sam_click_point[0]}, {self.sam_click_point[1]})\n✨ 'Alpha 채널 적용' 버튼을 눈러서 해당 지점에서 객체를 추출하세요."
            
            print(f"📍 SAM 시작점 설정: {self.sam_click_point}")
            
            return status
            
        except Exception as e:
            return f"❌ 클릭 처리 실패: {str(e)}"
    
    def _reset_rotation(self):
        """회전 상태와 SAM 클릭 좌표를 초기화합니다."""
        if self.original_rotation_image is not None:
            print("🔄 회전 상태 리셋: 원본 이미지 삭제")
        if self.sam_click_point is not None:
            print("📍 SAM 클릭 좌표 리셋")
        self.original_rotation_image = None
        self.sam_click_point = None  # SAM 클릭 좌표도 리셋
