"""
File Manager 서비스

Single Responsibility Principle (SRP)에 따라
파일 관리만을 담당하는 서비스
"""

import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image


class FileManager:
    """파일 관리 전용 클래스"""
    
    def __init__(self, faces_path: str, output_path: str):
        """
        초기화
        
        Args:
            faces_path: 얼굴 파일들이 저장될 경로
            output_path: 결과 이미지가 저장될 경로
        """
        self.faces_path = faces_path
        self.output_path = output_path
        
        # 디렉토리 생성
        self.faces_dir = Path(faces_path)
        self.faces_dir.mkdir(exist_ok=True)
        Path(output_path).mkdir(exist_ok=True)
        
        self._logger = logging.getLogger(__name__)
    
    def get_safe_filename(self, filename: str) -> str:
        """
        안전한 파일명을 생성합니다.
        
        Args:
            filename: 원본 파일명
            
        Returns:
            안전한 파일명
        """
        import re
        import time
        
        # 한글과 특수문자를 제거하고 영문, 숫자, 언더스코어만 허용
        safe_name = re.sub(r'[^\w\-_.]', '_', filename)
        
        # 연속된 언더스코어를 하나로 변경
        safe_name = re.sub(r'_+', '_', safe_name)
        
        # 시작과 끝의 언더스코어 제거
        safe_name = safe_name.strip('_')
        
        # 빈 문자열이면 타임스탬프 사용
        if not safe_name:
            safe_name = f"face_{int(time.time())}"
        
        return safe_name
    
    def get_embedding_list(self) -> List[dict]:
        """
        등록된 모든 embedding 파일 목록을 반환합니다.
        
        Returns:
            embedding 파일 정보 리스트
        """
        try:
            embedding_files = []
            
            # faces 디렉토리의 모든 .json 파일 검색
            for json_file in self.faces_dir.glob("*.json"):
                try:
                    # 해당하는 이미지 파일 찾기
                    base_name = json_file.stem
                    image_file = self.faces_dir / f"{base_name}.jpg"
                    
                    # 파일 정보 수집
                    file_info = {
                        "name": base_name,
                        "json_file": json_file.name,
                        "image_file": image_file.name if image_file.exists() else "없음",
                        "json_size": json_file.stat().st_size,
                        "image_size": image_file.stat().st_size if image_file.exists() else 0,
                        "json_path": str(json_file),
                        "image_path": str(image_file) if image_file.exists() else None
                    }
                    
                    embedding_files.append(file_info)
                    
                except Exception as e:
                    self._logger.warning(f"파일 정보 수집 실패 {json_file}: {e}")
                    continue
            
            # 이름순으로 정렬
            embedding_files.sort(key=lambda x: x["name"])
            
            return embedding_files
            
        except Exception as e:
            self._logger.error(f"embedding 목록 조회 실패: {e}")
            return []
    
    def get_embedding_choices(self) -> List[str]:
        """
        드롭다운용 embedding 선택지 목록을 반환합니다.
        
        Returns:
            embedding 이름 리스트
        """
        if not self.faces_dir.exists():
            return []
        
        json_files = list(self.faces_dir.glob("*.json"))
        return [f.stem for f in json_files]  # .json 확장자 제거
    
    def get_embedding_gallery_data(self) -> List[Tuple[str, str]]:
        """
        등록된 embedding 파일들의 갤러리 데이터를 반환합니다.
        
        Returns:
            갤러리용 데이터 리스트 [(이미지경로, 파일명), ...]
        """
        if not self.faces_dir.exists():
            return []
        
        json_files = list(self.faces_dir.glob("*.json"))
        gallery_data = []
        
        for json_file in sorted(json_files):
            # .json 확장자 제거
            name = json_file.stem
            # 대응하는 이미지 파일 확인
            image_file = json_file.with_suffix('.jpg')
            if image_file.exists():
                # 이미지 파일 경로를 문자열로 변환
                image_path = str(image_file)
                gallery_data.append((image_path, name))
        
        return gallery_data
    
    def save_result_image(self, image: np.ndarray, prefix: str = "final_result") -> str:
        """
        결과 이미지를 파일로 저장합니다.
        
        Args:
            image: 저장할 이미지 (RGB 형식)
            prefix: 파일명 접두사
            
        Returns:
            저장된 파일 경로
        """
        try:
            output_dir = Path(self.output_path)
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = output_dir / f"{prefix}_{timestamp}.jpg"
            
            # PIL로 이미지 저장 (RGB 형식)
            pil_image = Image.fromarray(image)
            pil_image.save(output_filename, "JPEG", quality=95)
            
            self._logger.info(f"결과 이미지 저장: {output_filename}")
            return str(output_filename)
            
        except Exception as e:
            self._logger.error(f"이미지 저장 실패: {e}")
            raise RuntimeError(f"이미지 저장 실패: {e}")
    
    def delete_latest_result_image(self) -> Tuple[bool, str]:
        """
        가장 최근 생성된 결과 이미지 파일을 삭제합니다.
        
        Returns:
            (삭제 성공여부, 메시지)
        """
        try:
            output_dir = Path(self.output_path)
            
            if not output_dir.exists():
                return False, "출력 폴더가 존재하지 않습니다."
            
            # 가장 최근 생성된 final_result 파일 찾기
            result_files = list(output_dir.glob("final_result_*.jpg"))
            if not result_files:
                return False, "삭제할 결과 파일이 없습니다."
            
            # 파일 생성 시간으로 정렬하여 가장 최근 파일 선택
            latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
            
            # 파일 삭제
            latest_file.unlink()
            
            self._logger.info(f"결과 이미지 파일 삭제 완료: {latest_file}")
            return True, f"✅ 파일 삭제 완료: {latest_file.name}"
            
        except Exception as e:
            self._logger.error(f"파일 삭제 실패: {e}")
            return False, f"❌ 파일 삭제 실패: {str(e)}"
    
    def get_embedding_list_display(self) -> str:
        """
        embedding 목록을 표시용 문자열로 변환합니다.
        
        Returns:
            표시용 문자열
        """
        embeddings = self.get_embedding_list()
        
        if not embeddings:
            return "등록된 embedding 파일이 없습니다."
        
        result = "📋 등록된 Embedding 파일 목록:\n\n"
        
        for i, emb in enumerate(embeddings, 1):
            result += f"**{i}. {emb['name']}**\n"
            result += f"   - JSON 파일: `{emb['json_file']}` ({emb['json_size']} bytes)\n"
            result += f"   - 이미지 파일: `{emb['image_file']}` ({emb['image_size']} bytes)\n"
            result += f"   - 경로: `{emb['json_path']}`\n"
            if emb['image_path']:
                result += f"   - 이미지 경로: `{emb['image_path']}`\n"
            result += "\n"
        
        return result


