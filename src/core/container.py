"""
의존성 주입 컨테이너

Dependency Inversion Principle (DIP)에 따라
의존성 관리를 담당하는 컨테이너
"""

import os
import logging
from dotenv import load_dotenv

from src.services.face_manager import FaceManager
from src.services.file_manager import FileManager
from src.utils.config import Config


class DIContainer:
    """의존성 주입 컨테이너"""
    
    def __init__(self):
        """
        초기화
        """
        # .env 파일 로드
        load_dotenv()
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self._logger = logging.getLogger(__name__)
        
        # 서비스 인스턴스들
        self._config = None
        self._face_manager = None
        self._file_manager = None
        
        # 서비스 초기화
        self._initialize_services()
    
    def _initialize_services(self) -> None:
        """
        서비스들을 초기화합니다.
        """
        try:
            # Config 초기화
            self._config = Config()
            self._logger.info("Config 초기화 완료")
            
            # FaceManager 초기화
            self._face_manager = FaceManager(self._config)
            self._logger.info("FaceManager 초기화 완료")
            
            # FileManager 초기화
            faces_path = os.getenv("FACES_PATH", "./faces")
            output_path = os.getenv("OUTPUT_PATH", "./outputs")
            self._file_manager = FileManager(faces_path, output_path)
            self._logger.info("FileManager 초기화 완료")
            
        except Exception as e:
            self._logger.error(f"서비스 초기화 실패: {e}")
            raise RuntimeError(f"서비스 초기화 실패: {e}")
    
    def get_config(self) -> Config:
        """
        Config 인스턴스를 반환합니다.
        
        Returns:
            Config 인스턴스
        """
        if self._config is None:
            raise RuntimeError("Config가 초기화되지 않았습니다.")
        return self._config
    
    def get_face_manager(self) -> FaceManager:
        """
        FaceManager 인스턴스를 반환합니다.
        
        Returns:
            FaceManager 인스턴스
        """
        if self._face_manager is None:
            raise RuntimeError("FaceManager가 초기화되지 않았습니다.")
        return self._face_manager
    
    def get_file_manager(self) -> FileManager:
        """
        FileManager 인스턴스를 반환합니다.
        
        Returns:
            FileManager 인스턴스
        """
        if self._file_manager is None:
            raise RuntimeError("FileManager가 초기화되지 않았습니다.")
        return self._file_manager
    
    def is_initialized(self) -> bool:
        """
        모든 서비스가 초기화되었는지 확인합니다.
        
        Returns:
            초기화 여부
        """
        return (
            self._config is not None and
            self._face_manager is not None and
            self._file_manager is not None
        )
    
    def get_service_info(self) -> dict:
        """
        서비스 정보를 반환합니다.
        
        Returns:
            서비스 정보 딕셔너리
        """
        return {
            "config_initialized": self._config is not None,
            "face_manager_initialized": self._face_manager is not None,
            "file_manager_initialized": self._file_manager is not None,
            "all_services_initialized": self.is_initialized()
        }

