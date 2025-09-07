"""
Config 유틸리티 단위 테스트

TDD 방식으로 작성된 Config 클래스의 테스트
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from src.utils.config import Config


class TestConfig:
    """Config 클래스 테스트"""
    
    def test_config_initialization_default(self):
        """기본 설정으로 초기화 테스트"""
        # Act
        config = Config()
        
        # Assert
        assert config is not None
        assert isinstance(config, Config)
        assert config.get("use_gpu") is not None
        assert config.get("gradio_server_port") is not None
    
    def test_config_initialization_with_env_file(self):
        """환경 파일로 초기화 테스트"""
        # Arrange
        env_content = """
USE_GPU=false
GRADIO_SERVER_PORT=8080
LOG_LEVEL=DEBUG
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content)
            env_file = f.name
        
        try:
            # Act
            config = Config(env_file)
            
            # Assert
            assert config.get("use_gpu") is False
            assert config.get("gradio_server_port") == 8080
            assert config.get("log_level") == "DEBUG"
        finally:
            os.unlink(env_file)
    
    def test_get_string_value(self):
        """문자열 값 가져오기 테스트"""
        # Arrange
        with patch.dict(os.environ, {'TEST_STRING': 'test_value'}):
            config = Config()
            
            # Act
            value = config._get_str('TEST_STRING', 'default')
            
            # Assert
            assert value == 'test_value'
    
    def test_get_string_value_default(self):
        """문자열 값 기본값 테스트"""
        # Arrange
        config = Config()
        
        # Act
        value = config._get_str('NON_EXISTENT_KEY', 'default_value')
        
        # Assert
        assert value == 'default_value'
    
    def test_get_int_value(self):
        """정수 값 가져오기 테스트"""
        # Arrange
        with patch.dict(os.environ, {'TEST_INT': '42'}):
            config = Config()
            
            # Act
            value = config._get_int('TEST_INT', 0)
            
            # Assert
            assert value == 42
    
    def test_get_int_value_invalid(self):
        """유효하지 않은 정수 값 테스트"""
        # Arrange
        with patch.dict(os.environ, {'TEST_INT': 'invalid'}):
            config = Config()
            
            # Act
            value = config._get_int('TEST_INT', 10)
            
            # Assert
            assert value == 10
    
    def test_get_bool_value_true(self):
        """불린 값 True 테스트"""
        test_cases = ['true', 'TRUE', 'True', '1', 'yes', 'YES', 'on', 'ON']
        
        for test_case in test_cases:
            # Arrange
            with patch.dict(os.environ, {'TEST_BOOL': test_case}):
                config = Config()
                
                # Act
                value = config._get_bool('TEST_BOOL', False)
                
                # Assert
                assert value is True, f"Failed for value: {test_case}"
    
    def test_get_bool_value_false(self):
        """불린 값 False 테스트"""
        test_cases = ['false', 'FALSE', 'False', '0', 'no', 'NO', 'off', 'OFF', 'invalid']
        
        for test_case in test_cases:
            # Arrange
            with patch.dict(os.environ, {'TEST_BOOL': test_case}):
                config = Config()
                
                # Act
                value = config._get_bool('TEST_BOOL', True)
                
                # Assert
                assert value is False, f"Failed for value: {test_case}"
    
    def test_get_method(self):
        """get 메서드 테스트"""
        # Arrange
        config = Config()
        
        # Act
        value = config.get('use_gpu')
        default_value = config.get('non_existent_key', 'default')
        
        # Assert
        assert value is not None
        assert default_value == 'default'
    
    def test_get_model_path(self):
        """모델 경로 가져오기 테스트"""
        # Arrange
        with patch.dict(os.environ, {'BUFFALO_L_MODEL_PATH': '/path/to/model'}):
            config = Config()
            
            # Act
            path = config.get_model_path('buffalo_l')
            
            # Assert
            assert path == '/path/to/model'
    
    def test_get_model_path_not_found(self):
        """모델 경로를 찾을 수 없는 경우 테스트"""
        # Arrange
        config = Config()
        
        # Act & Assert
        with pytest.raises(ValueError, match="Model path not found"):
            config.get_model_path('non_existent_model')
    
    def test_get_model_path_absolute(self):
        """상대 경로를 절대 경로로 변환 테스트"""
        # Arrange
        with patch.dict(os.environ, {'BUFFALO_L_MODEL_PATH': 'models/buffalo_l'}):
            config = Config()
            
            # Act
            path = config.get_model_path('buffalo_l')
            
            # Assert
            assert os.path.isabs(path)
    
    @patch('torch.cuda.is_available')
    def test_is_gpu_available_true(self, mock_cuda_available):
        """GPU 사용 가능 테스트"""
        # Arrange
        mock_cuda_available.return_value = True
        with patch.dict(os.environ, {'USE_GPU': 'true'}):
            config = Config()
            
            # Act
            is_available = config.is_gpu_available()
            
            # Assert
            assert is_available is True
    
    @patch('torch.cuda.is_available')
    def test_is_gpu_available_false_cuda_unavailable(self, mock_cuda_available):
        """CUDA 사용 불가능한 경우 테스트"""
        # Arrange
        mock_cuda_available.return_value = False
        with patch.dict(os.environ, {'USE_GPU': 'true'}):
            config = Config()
            
            # Act
            is_available = config.is_gpu_available()
            
            # Assert
            assert is_available is False
    
    def test_is_gpu_available_false_use_gpu_false(self):
        """USE_GPU가 false인 경우 테스트"""
        # Arrange
        with patch.dict(os.environ, {'USE_GPU': 'false'}):
            config = Config()
            
            # Act
            is_available = config.is_gpu_available()
            
            # Assert
            assert is_available is False
    
    @patch('torch.cuda.is_available')
    def test_get_device_cuda(self, mock_cuda_available):
        """CUDA 디바이스 반환 테스트"""
        # Arrange
        mock_cuda_available.return_value = True
        with patch.dict(os.environ, {'USE_GPU': 'true'}):
            config = Config()
            
            # Act
            device = config.get_device()
            
            # Assert
            assert device == 'cuda'
    
    def test_get_device_cpu(self):
        """CPU 디바이스 반환 테스트"""
        # Arrange
        with patch.dict(os.environ, {'USE_GPU': 'false'}):
            config = Config()
            
            # Act
            device = config.get_device()
            
            # Assert
            assert device == 'cpu'
    
    def test_create_directories(self):
        """디렉토리 생성 테스트"""
        # Arrange
        config = Config()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Act
            with patch('pathlib.Path.mkdir') as mock_mkdir:
                config.create_directories()
                
                # Assert
                assert mock_mkdir.call_count == 4  # models, logs, output, temp
    
    def test_validate_config_valid(self):
        """유효한 설정 검증 테스트"""
        # Arrange
        with patch.dict(os.environ, {
            'BUFFALO_L_MODEL_PATH': '/path/to/buffalo_l',
            'INSWAPPER_MODEL_PATH': '/path/to/inswapper.onnx',
            'CODEFORMER_MODEL_PATH': '/path/to/codeformer.pth',
            'GRADIO_SERVER_PORT': '7860',
            'MAX_IMAGE_SIZE': '1024'
        }):
            config = Config()
            
            # Act
            is_valid = config.validate_config()
            
            # Assert
            assert is_valid is True
    
    def test_validate_config_invalid_port(self):
        """유효하지 않은 포트 검증 테스트"""
        # Arrange
        with patch.dict(os.environ, {
            'BUFFALO_L_MODEL_PATH': '/path/to/buffalo_l',
            'INSWAPPER_MODEL_PATH': '/path/to/inswapper.onnx',
            'CODEFORMER_MODEL_PATH': '/path/to/codeformer.pth',
            'GRADIO_SERVER_PORT': '99999',  # 유효하지 않은 포트
            'MAX_IMAGE_SIZE': '1024'
        }):
            config = Config()
            
            # Act
            is_valid = config.validate_config()
            
            # Assert
            assert is_valid is False
    
    def test_validate_config_invalid_image_size(self):
        """유효하지 않은 이미지 크기 검증 테스트"""
        # Arrange
        with patch.dict(os.environ, {
            'BUFFALO_L_MODEL_PATH': '/path/to/buffalo_l',
            'INSWAPPER_MODEL_PATH': '/path/to/inswapper.onnx',
            'CODEFORMER_MODEL_PATH': '/path/to/codeformer.pth',
            'GRADIO_SERVER_PORT': '7860',
            'MAX_IMAGE_SIZE': '-1'  # 유효하지 않은 크기
        }):
            config = Config()
            
            # Act
            is_valid = config.validate_config()
            
            # Assert
            assert is_valid is False
    
    def test_dict_access(self):
        """딕셔너리 스타일 접근 테스트"""
        # Arrange
        config = Config()
        
        # Act
        value = config['use_gpu']
        config['test_key'] = 'test_value'
        
        # Assert
        assert value is not None
        assert config['test_key'] == 'test_value'
    
    def test_to_dict(self):
        """딕셔너리 변환 테스트"""
        # Arrange
        config = Config()
        
        # Act
        config_dict = config.to_dict()
        
        # Assert
        assert isinstance(config_dict, dict)
        assert 'use_gpu' in config_dict
        assert 'gradio_server_port' in config_dict
        assert 'buffalo_l_model_path' in config_dict
    
    def test_config_immutability(self):
        """설정 불변성 테스트"""
        # Arrange
        config = Config()
        original_dict = config.to_dict()
        
        # Act
        config['test_key'] = 'test_value'
        new_dict = config.to_dict()
        
        # Assert
        assert 'test_key' in new_dict
        assert 'test_key' not in original_dict
        assert new_dict['test_key'] == 'test_value'
