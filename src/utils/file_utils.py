"""
파일 관련 공통 유틸리티 함수들
"""

import os
import subprocess
import platform
from pathlib import Path
from typing import Optional


def open_directory(path: str) -> bool:
    """
    주어진 경로의 디렉토리를 시스템 파일 탐색기로 엽니다.
    
    Args:
        path: 열고자 하는 디렉토리 경로 (파일 경로인 경우 부모 디렉토리를 열음)
        
    Returns:
        성공 시 True, 실패 시 False
    """
    try:
        # 경로가 존재하는지 확인
        path_obj = Path(path)
        
        # 파일인 경우 부모 디렉토리 사용
        if path_obj.is_file():
            directory_path = path_obj.parent
        else:
            directory_path = path_obj
            
        # 디렉토리가 존재하지 않으면 생성
        if not directory_path.exists():
            directory_path.mkdir(parents=True, exist_ok=True)
        
        # 운영체제별로 다른 명령 사용
        system = platform.system()
        
        if system == "Windows":
            # Windows: explorer 사용 (check=False로 설정 - Explorer는 성공해도 non-zero를 반환할 수 있음)
            result = subprocess.run(["explorer", str(directory_path)], check=False)
            # Windows Explorer는 실행만 되면 성공으로 간주
            print(f"✅ 디렉토리 열기 성공: {directory_path} (exit code: {result.returncode})")
            return True
        elif system == "Darwin":  # macOS
            # macOS: open 사용
            result = subprocess.run(["open", str(directory_path)], check=False)
            if result.returncode == 0:
                print(f"✅ 디렉토리 열기 성공: {directory_path}")
                return True
            else:
                print(f"❌ 디렉토리 열기 실패 (exit code: {result.returncode}): {directory_path}")
                return False
        elif system == "Linux":
            # Linux: xdg-open 사용
            result = subprocess.run(["xdg-open", str(directory_path)], check=False)
            if result.returncode == 0:
                print(f"✅ 디렉토리 열기 성공: {directory_path}")
                return True
            else:
                print(f"❌ 디렉토리 열기 실패 (exit code: {result.returncode}): {directory_path}")
                return False
        else:
            print(f"❌ 지원하지 않는 운영체제: {system}")
            return False
        
    except Exception as e:
        print(f"❌ 디렉토리 열기 실패: {e}")
        return False


def get_save_location_status(
    default_path: str, 
    last_saved_file: Optional[str] = None,
    location_name: str = "저장"
) -> str:
    """
    저장 위치 상태 메시지를 생성합니다.
    
    Args:
        default_path: 기본 저장 경로
        last_saved_file: 마지막으로 저장된 파일 경로 (Optional)
        location_name: 저장 위치 이름 (예: "결과", "이미지", "프롬프트")
        
    Returns:
        저장 위치 상태 메시지
    """
    try:
        # 디렉토리 생성 (존재하지 않는 경우)
        Path(default_path).mkdir(parents=True, exist_ok=True)
        
        if last_saved_file and os.path.exists(last_saved_file):
            return (
                f"📁 최근 저장된 파일:\n{last_saved_file}\n\n"
                f"📂 기본 {location_name} 저장 폴더:\n{default_path}"
            )
        else:
            return f"📂 {location_name} 저장 폴더:\n{default_path}"
            
    except Exception as e:
        return f"❌ 저장 위치 정보를 가져오는데 실패했습니다: {str(e)}"


def open_save_location(
    default_path: str, 
    last_saved_file: Optional[str] = None
) -> str:
    """
    저장 위치를 파일 탐색기로 열고 상태 메시지를 반환합니다.
    
    Args:
        default_path: 기본 저장 경로
        last_saved_file: 마지막으로 저장된 파일 경로 (Optional)
        
    Returns:
        실행 결과 메시지
    """
    try:
        # 마지막 저장된 파일이 있고 존재하면 해당 디렉토리 열기
        if last_saved_file and os.path.exists(last_saved_file):
            target_path = last_saved_file
            location_type = "최근 저장된 파일이 있는 폴더"
        else:
            target_path = default_path
            location_type = "기본 저장 폴더"
        
        # 디렉토리 열기
        success = open_directory(target_path)
        
        if success:
            return f"✅ {location_type}를 열었습니다:\n{Path(target_path).parent if os.path.isfile(target_path) else target_path}"
        else:
            return f"❌ {location_type} 열기에 실패했습니다."
            
    except Exception as e:
        return f"❌ 저장 위치 열기 중 오류가 발생했습니다: {str(e)}"
