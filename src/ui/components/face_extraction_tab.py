"""
얼굴 추출 탭 컴포넌트

Single Responsibility Principle (SRP)에 따라
얼굴 추출 UI만을 담당하는 컴포넌트
"""

import gradio as gr
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional

from src.services.face_manager import FaceManager


class FaceExtractionTab:
    """얼굴 추출 탭 컴포넌트"""
    
    def __init__(self, face_manager: FaceManager):
        """
        초기화
        
        Args:
            face_manager: 얼굴 관리 서비스
        """
        self.face_manager = face_manager
    
    def create_interface(self) -> gr.Tab:
        """
        얼굴 추출 탭 인터페이스를 생성합니다.
        
        Returns:
            Gradio Tab 컴포넌트
        """
        with gr.Tab("얼굴 추출") as tab:
            gr.Markdown("## 📸 이미지에서 첫 번째 얼굴 추출")
            
            with gr.Row():
                with gr.Column(scale=1):
                    self.upload_input = gr.Image(
                        label="이미지 업로드",
                        type="filepath"
                    )
                    
                    self.process_btn = gr.Button("얼굴 추출", variant="primary")
                    
                    self.result_text = gr.Textbox(
                        label="결과",
                        lines=4,
                        interactive=False
                    )
                
                with gr.Column(scale=1):
                    self.extracted_face = gr.Image(
                        label="추출된 얼굴",
                        type="numpy"
                    )
        
        return tab
    
    def process_uploaded_image(self, file_path: str) -> Tuple[bool, str, Optional[np.ndarray]]:
        """
        업로드된 이미지를 처리합니다.
        
        Args:
            file_path: 업로드된 이미지 파일 경로
            
        Returns:
            (성공여부, 메시지, 추출된 얼굴 이미지)
        """
        if file_path is None:
            return False, "이미지를 업로드해주세요.", None
        
        try:
            # 파일 경로를 Path 객체로 변환
            file_path_obj = Path(file_path)
            filename = file_path_obj.name
            
            # 한글 파일명 처리를 위해 numpy array로 직접 읽기
            from PIL import Image
            
            # PIL로 이미지 로드 (한글 경로 지원)
            pil_image = Image.open(file_path)
            # PIL Image를 numpy array로 변환 (RGB 유지)
            image_rgb = np.array(pil_image)
            
            # RGB를 BGR로 변환 (OpenCV는 BGR 사용)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # 얼굴 추출 (BGR 이미지로 탐지, RGB 이미지로 추출)
            success, message, face_path = self.face_manager.extract_first_face(image_bgr, filename, image_rgb)
            
            if success and face_path:
                # 추출된 얼굴 이미지 로드 (PIL 사용으로 색상 정확성 보장)
                try:
                    pil_image = Image.open(face_path)
                    face_image_rgb = np.array(pil_image)
                    return success, message, face_image_rgb
                except Exception as e:
                    return success, message, None
            
            return success, message, None
            
        except Exception as e:
            return False, f"이미지를 로드할 수 없습니다: {str(e)}", None
