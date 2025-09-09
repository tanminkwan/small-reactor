"""
Embedding 목록 탭 컴포넌트

Single Responsibility Principle (SRP)에 따라
Embedding 목록 UI만을 담당하는 컴포넌트
"""

import gradio as gr
from typing import List, Tuple

from src.services.file_manager import FileManager


class EmbeddingListTab:
    """Embedding 목록 탭 컴포넌트"""
    
    def __init__(self, file_manager: FileManager):
        """
        초기화
        
        Args:
            file_manager: 파일 관리 서비스
        """
        self.file_manager = file_manager
    
    def create_interface(self) -> gr.Tab:
        """
        Embedding 목록 탭 인터페이스를 생성합니다.
        
        Returns:
            Gradio Tab 컴포넌트
        """
        with gr.Tab("Embedding 목록") as tab:
            gr.Markdown("## 📋 등록된 Embedding 파일 목록")
            
            self.refresh_btn = gr.Button("목록 새로고침", variant="secondary")
            
            self.embedding_gallery = gr.Gallery(
                value=self.file_manager.get_embedding_gallery_data(),
                label="등록된 얼굴 이미지",
                show_label=True,
                elem_id="embedding_gallery",
                columns=4,
                rows=2,
                height="auto",
                allow_preview=True
            )
        
        return tab
    
    def get_embedding_gallery_data(self) -> List[Tuple[str, str]]:
        """
        등록된 embedding 파일들의 갤러리 데이터를 반환합니다.
        
        Returns:
            갤러리용 데이터 리스트 [(이미지경로, 파일명), ...]
        """
        return self.file_manager.get_embedding_gallery_data()
    
    def get_embedding_list_display(self) -> str:
        """
        embedding 목록을 표시용 문자열로 변환합니다.
        
        Returns:
            표시용 문자열
        """
        return self.file_manager.get_embedding_list_display()
