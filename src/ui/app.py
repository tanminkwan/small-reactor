"""
Face Manager 메인 애플리케이션

Single Responsibility Principle (SRP)에 따라
애플리케이션 조합과 실행만을 담당하는 클래스
"""

import gradio as gr
import logging

from src.core.container import DIContainer
from src.ui.components.face_extraction_tab import FaceExtractionTab
from src.ui.components.face_swap_tab import FaceSwapTab
from src.ui.components.embedding_list_tab import EmbeddingListTab
from src.ui.handlers.event_handlers import EventHandlers


class FaceManagerApp:
    """Face Manager 메인 애플리케이션"""
    
    def __init__(self, container: DIContainer):
        """
        초기화
        
        Args:
            container: 의존성 주입 컨테이너
        """
        self.container = container
        self._logger = logging.getLogger(__name__)
        
        # 서비스 인스턴스들 가져오기
        self.face_manager = container.get_face_manager()
        self.file_manager = container.get_file_manager()
        
        # 탭 컴포넌트들 초기화
        self.face_extraction_tab = FaceExtractionTab(self.face_manager)
        self.face_swap_tab = FaceSwapTab(self.face_manager, self.file_manager)
        self.embedding_list_tab = EmbeddingListTab(self.file_manager)
        
        # 이벤트 핸들러 초기화
        self.event_handlers = EventHandlers(self.face_manager, self.file_manager)
        
        self._logger.info("FaceManagerApp 초기화 완료")
    
    def create_interface(self) -> gr.Blocks:
        """
        전체 인터페이스를 생성합니다.
        
        Returns:
            Gradio Blocks 인터페이스
        """
        with gr.Blocks(title="Face Manager", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🎭 Face Manager")
            gr.Markdown("이미지에서 얼굴을 추출하고 embedding을 관리합니다.")
            
            # 각 탭 생성
            with self.face_swap_tab.create_interface():
                pass
            
            with self.face_extraction_tab.create_interface():
                pass
            
            with self.embedding_list_tab.create_interface():
                pass
            
            # 이벤트 핸들러 설정
            self.event_handlers.setup_all_handlers(
                self.face_extraction_tab,
                self.face_swap_tab,
                self.embedding_list_tab
            )
        
        self._logger.info("인터페이스 생성 완료")
        return interface
    
    def run(self, server_name: str = "0.0.0.0", server_port: int = 7860, share: bool = False, debug: bool = True) -> None:
        """
        애플리케이션을 실행합니다.
        
        Args:
            server_name: 서버 이름
            server_port: 서버 포트
            share: 공유 여부
            debug: 디버그 모드 여부
        """
        try:
            self._logger.info("Face Manager UI 시작")
            
            # 인터페이스 생성
            interface = self.create_interface()
            
            # 애플리케이션 실행
            interface.launch(
                server_name=server_name,
                server_port=server_port,
                share=share,
                debug=debug
            )
            
        except Exception as e:
            self._logger.error(f"애플리케이션 실행 실패: {e}")
            raise RuntimeError(f"애플리케이션 실행 실패: {e}")
    
    def get_app_info(self) -> dict:
        """
        애플리케이션 정보를 반환합니다.
        
        Returns:
            애플리케이션 정보 딕셔너리
        """
        return {
            "app_name": "Face Manager",
            "version": "1.0.0",
            "container_info": self.container.get_service_info(),
            "components_initialized": {
                "face_extraction_tab": self.face_extraction_tab is not None,
                "face_swap_tab": self.face_swap_tab is not None,
                "embedding_list_tab": self.embedding_list_tab is not None,
                "event_handlers": self.event_handlers is not None
            }
        }
