#!/usr/bin/env python3
"""
Face Manager 메인 진입점

애플리케이션의 시작점으로 의존성 주입과 애플리케이션 실행을 담당
"""

import logging
from dotenv import load_dotenv

from src.core.container import DIContainer
from src.ui.app import FaceManagerApp

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """메인 함수"""
    try:
        logger.info("Face Manager 시작")
        
        # 의존성 주입 컨테이너 생성
        container = DIContainer()
        
        # 컨테이너 초기화 확인
        if not container.is_initialized():
            raise RuntimeError("서비스 초기화에 실패했습니다.")
        
        logger.info("의존성 주입 컨테이너 초기화 완료")
        logger.info(f"서비스 정보: {container.get_service_info()}")
        
        # 애플리케이션 생성
        app = FaceManagerApp(container)
        
        # 애플리케이션 정보 로그
        app_info = app.get_app_info()
        logger.info(f"애플리케이션 정보: {app_info}")
        
        # 애플리케이션 실행
        app.run()
        
    except Exception as e:
        logger.error(f"애플리케이션 실행 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()
