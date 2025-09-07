"""
설정 관리 유틸리티

환경 변수와 설정을 관리하는 유틸리티 클래스
"""

import os
from typing import Any, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv


class Config:
    """설정 관리 클래스"""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Args:
            env_file: .env 파일 경로 (None이면 기본 경로 사용)
        """
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
        
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """환경 변수에서 설정을 로드합니다."""
        return {
            # GPU 설정
            "use_gpu": self._get_bool("USE_GPU", True),
            "cuda_visible_devices": self._get_str("CUDA_VISIBLE_DEVICES", "0"),
            
            # 모델 경로 설정
            "buffalo_l_model_path": self._get_str("BUFFALO_L_MODEL_PATH", "models/buffalo_l"),
            "inswapper_model_path": self._get_str("INSWAPPER_MODEL_PATH", "models/inswapper_128.onnx"),
            "codeformer_model_path": self._get_str("CODEFORMER_MODEL_PATH", "models/codeformer-v0.1.0.pth"),
            
            # 로깅 설정
            "log_level": self._get_str("LOG_LEVEL", "INFO"),
            "log_file": self._get_str("LOG_FILE", "logs/app.log"),
            
            # Gradio 설정
            "gradio_server_name": self._get_str("GRADIO_SERVER_NAME", "0.0.0.0"),
            "gradio_server_port": self._get_int("GRADIO_SERVER_PORT", 7860),
            "gradio_share": self._get_bool("GRADIO_SHARE", False),
            
            # 성능 설정
            "max_image_size": self._get_int("MAX_IMAGE_SIZE", 1024),
            "batch_size": self._get_int("BATCH_SIZE", 1),
        }
    
    def _get_str(self, key: str, default: str) -> str:
        """문자열 환경 변수를 가져옵니다."""
        return os.getenv(key, default)
    
    def _get_int(self, key: str, default: int) -> int:
        """정수 환경 변수를 가져옵니다."""
        try:
            return int(os.getenv(key, str(default)))
        except ValueError:
            return default
    
    def _get_bool(self, key: str, default: bool) -> bool:
        """불린 환경 변수를 가져옵니다."""
        value = os.getenv(key, str(default)).lower()
        return value in ("true", "1", "yes", "on")
    
    def get(self, key: str, default: Any = None) -> Any:
        """설정 값을 가져옵니다."""
        return self._config.get(key, default)
    
    def get_model_path(self, model_name: str) -> str:
        """모델 경로를 가져옵니다."""
        path_key = f"{model_name}_model_path"
        path = self.get(path_key)
        
        if not path:
            raise ValueError(f"Model path not found for {model_name}")
        
        # 절대 경로로 변환
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        
        return path
    
    def is_gpu_available(self) -> bool:
        """GPU 사용 가능 여부를 확인합니다."""
        if not self.get("use_gpu"):
            return False
        
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def get_device(self) -> str:
        """사용할 디바이스를 반환합니다."""
        return "cuda" if self.is_gpu_available() else "cpu"
    
    def create_directories(self) -> None:
        """필요한 디렉토리를 생성합니다."""
        directories = [
            "models",
            "logs", 
            "output",
            "temp"
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    def validate_config(self) -> bool:
        """설정이 유효한지 검증합니다."""
        try:
            # 모델 경로 검증
            self.get_model_path("buffalo_l")
            self.get_model_path("inswapper")
            self.get_model_path("codeformer")
            
            # 포트 번호 검증
            port = self.get("gradio_server_port")
            if not (1 <= port <= 65535):
                return False
            
            # 이미지 크기 검증
            max_size = self.get("max_image_size")
            if max_size <= 0:
                return False
            
            return True
        except Exception:
            return False
    
    def __getitem__(self, key: str) -> Any:
        """딕셔너리 스타일로 설정에 접근합니다."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """설정 값을 설정합니다."""
        self._config[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 반환합니다."""
        return self._config.copy()
