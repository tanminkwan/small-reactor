"""
UI 공통 유틸리티

Interface Segregation Principle (ISP)에 따라
UI 관련 공통 기능만을 제공하는 유틸리티
"""

import gradio as gr
from pathlib import Path
from typing import List, Tuple

from src.services.file_manager import FileManager


class UIHelpers:
    """UI 공통 유틸리티"""
    
    @staticmethod
    def get_embedding_choices(file_manager: FileManager) -> List[str]:
        """
        드롭다운용 embedding 선택지 목록을 반환합니다.
        
        Args:
            file_manager: 파일 관리 서비스
            
        Returns:
            embedding 이름 리스트
        """
        return file_manager.get_embedding_choices()
    
    @staticmethod
    def get_embedding_gallery_data(file_manager: FileManager) -> List[Tuple[str, str]]:
        """
        등록된 embedding 파일들의 갤러리 데이터를 반환합니다.
        
        Args:
            file_manager: 파일 관리 서비스
            
        Returns:
            갤러리용 데이터 리스트 [(이미지경로, 파일명), ...]
        """
        return file_manager.get_embedding_gallery_data()
    
    @staticmethod
    def refresh_face_choices(file_manager: FileManager) -> gr.Dropdown:
        """
        얼굴 목록을 새로고침하고 드롭다운을 업데이트합니다.
        
        Args:
            file_manager: 파일 관리 서비스
            
        Returns:
            업데이트된 드롭다운 설정
        """
        choices = file_manager.get_embedding_choices()
        # 항상 첫 번째 항목을 선택하도록 강제
        return gr.update(choices=choices, value=choices[0] if choices else None)
    
    @staticmethod
    def update_embedding_choices(file_manager: FileManager) -> gr.Dropdown:
        """
        embedding 목록을 업데이트합니다 (기존 함수와 호환성 유지).
        
        Args:
            file_manager: 파일 관리 서비스
            
        Returns:
            업데이트된 드롭다운 설정
        """
        choices = file_manager.get_embedding_choices()
        # 항상 첫 번째 항목을 선택하도록 강제
        return gr.update(choices=choices, value=choices[0] if choices else None)
    
    @staticmethod
    def toggle_mouth_settings(checked: bool) -> gr.Group:
        """
        입 원본유지 체크박스 상태에 따라 설정 그룹 표시/숨김
        
        Args:
            checked: 체크박스 상태
            
        Returns:
            그룹 표시/숨김 설정
        """
        return gr.update(visible=checked)
