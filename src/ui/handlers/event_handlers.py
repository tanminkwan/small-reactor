"""
이벤트 핸들러

Single Responsibility Principle (SRP)에 따라
UI 이벤트 처리만을 담당하는 핸들러
"""

import gradio as gr
from typing import Dict, Any, Optional

from src.services.face_manager import FaceManager
from src.services.file_manager import FileManager
from src.ui.components.face_extraction_tab import FaceExtractionTab
from src.ui.components.face_swap_tab import FaceSwapTab
from src.ui.components.embedding_list_tab import EmbeddingListTab
from src.ui.utils.ui_helpers import UIHelpers


class EventHandlers:
    """이벤트 핸들러 모음"""
    
    def __init__(self, face_manager: FaceManager, file_manager: FileManager):
        """
        초기화
        
        Args:
            face_manager: 얼굴 관리 서비스
            file_manager: 파일 관리 서비스
        """
        self.face_manager = face_manager
        self.file_manager = file_manager
        self.ui_helpers = UIHelpers()
    
    def setup_face_extraction_handlers(self, tab: FaceExtractionTab) -> None:
        """
        얼굴 추출 탭 이벤트 핸들러를 설정합니다.
        
        Args:
            tab: 얼굴 추출 탭 컴포넌트
        """
        # 얼굴 추출 버튼 클릭 이벤트
        tab.process_btn.click(
            fn=tab.process_uploaded_image,
            inputs=[tab.upload_input],
            outputs=[gr.State(), tab.result_text, tab.extracted_face]
        )
    
    def setup_face_swap_handlers(self, tab: FaceSwapTab) -> None:
        """
        얼굴 교체 탭 이벤트 핸들러를 설정합니다.
        
        Args:
            tab: 얼굴 교체 탭 컴포넌트
        """
        # 타겟 이미지 업로드 시 처리
        tab.target_upload.change(
            fn=tab.process_target_image,
            inputs=[tab.target_upload],
            outputs=[gr.State(), tab.swap_result_text, tab.original_image]
        )
        
        # 입 원본유지 체크박스 변경 시 설정 그룹 표시/숨김
        tab.preserve_mouth_checkbox.change(
            fn=self.ui_helpers.toggle_mouth_settings,
            inputs=[tab.preserve_mouth_checkbox],
            outputs=[tab.mouth_settings_group]
        )
        
        # 얼굴 교체 버튼 클릭 시 처리 (CodeFormer 포함)
        tab.swap_btn.click(
            fn=self._perform_face_swap_wrapper,
            inputs=[
                tab.target_upload, 
                tab.face_indices_input, 
                tab.source_face_dropdown, 
                tab.codeformer_checkbox, 
                tab.preserve_mouth_checkbox, 
                tab.expand_ratio_slider, 
                tab.scale_x_slider, 
                tab.scale_y_slider, 
                tab.offset_x_slider, 
                tab.offset_y_slider
            ],
            outputs=[tab.swapped_image_state, tab.swap_result_text, tab.swapped_image]
        )
        
        # 결과 이미지 삭제 버튼 클릭 시 처리
        tab.delete_result_btn.click(
            fn=self._delete_result_wrapper,
            inputs=[],
            outputs=[tab.delete_result_text, tab.delete_result_text, tab.swapped_image]
        )
        
        # 얼굴 교체 탭의 새로고침 버튼 클릭 시 처리
        tab.refresh_faces_btn.click(
            fn=lambda: self.ui_helpers.refresh_face_choices(self.file_manager),
            inputs=[],
            outputs=[tab.source_face_dropdown]
        )
    
    def setup_embedding_list_handlers(self, tab: EmbeddingListTab) -> None:
        """
        Embedding 목록 탭 이벤트 핸들러를 설정합니다.
        
        Args:
            tab: Embedding 목록 탭 컴포넌트
        """
        # 목록 새로고침 버튼 클릭 이벤트
        tab.refresh_btn.click(
            fn=lambda: tab.get_embedding_gallery_data(),
            inputs=[],
            outputs=[tab.embedding_gallery]
        )
    
    def setup_all_handlers(
        self, 
        face_extraction_tab: FaceExtractionTab,
        face_swap_tab: FaceSwapTab,
        embedding_list_tab: EmbeddingListTab
    ) -> None:
        """
        모든 탭의 이벤트 핸들러를 설정합니다.
        
        Args:
            face_extraction_tab: 얼굴 추출 탭
            face_swap_tab: 얼굴 교체 탭
            embedding_list_tab: Embedding 목록 탭
        """
        self.setup_face_extraction_handlers(face_extraction_tab)
        self.setup_face_swap_handlers(face_swap_tab)
        self.setup_embedding_list_handlers(embedding_list_tab)
    
    def _perform_face_swap_wrapper(
        self, 
        file_path: str, 
        face_indices: str, 
        source_face_name: str, 
        use_codeformer: bool, 
        preserve_mouth: bool, 
        expand_ratio: float, 
        scale_x: float, 
        scale_y: float, 
        offset_x: float, 
        offset_y: float
    ) -> tuple:
        """
        얼굴 교체 + CodeFormer 통합 수행 래퍼
        
        Args:
            file_path: 파일 경로
            face_indices: 얼굴 인덱스
            source_face_name: 소스 얼굴 이름
            use_codeformer: CodeFormer 사용 여부
            preserve_mouth: 입 원본유지 여부
            expand_ratio: 확장 비율
            scale_x: 가로 스케일
            scale_y: 세로 스케일
            offset_x: 가로 오프셋
            offset_y: 세로 오프셋
            
        Returns:
            (최종 이미지, 메시지, 결과 이미지)
        """
        # 입 마스크 설정 구성
        mouth_settings = None
        if preserve_mouth:
            mouth_settings = {
                'expand_ratio': expand_ratio,
                'scale_x': scale_x,
                'scale_y': scale_y,
                'offset_x': offset_x,
                'offset_y': offset_y
            }
        
        # FaceSwapTab 인스턴스 생성하여 메서드 호출
        face_swap_tab = FaceSwapTab(self.face_manager, self.file_manager)
        final_image, message, result_image = face_swap_tab.perform_face_swap_with_optional_codeformer(
            file_path, face_indices, source_face_name, use_codeformer, preserve_mouth, mouth_settings
        )
        return final_image, message, result_image
    
    def _delete_result_wrapper(self) -> tuple:
        """
        결과 이미지 삭제 래퍼
        
        Returns:
            (삭제 성공여부, 메시지, None)
        """
        face_swap_tab = FaceSwapTab(self.face_manager, self.file_manager)
        success, message, _ = face_swap_tab.delete_result_image()
        return success, message, None
