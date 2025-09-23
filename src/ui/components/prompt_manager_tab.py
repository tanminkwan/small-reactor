"""
Prompt Manager 탭 컴포넌트

Single Responsibility Principle (SRP)에 따라
Stable Diffusion 프롬프트 관리 UI만을 담당하는 컴포넌트
"""

import gradio as gr
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

from src.services.file_manager import FileManager


class PromptManagerTab:
    """Prompt Manager 탭 컴포넌트"""
    
    def __init__(self, file_manager: FileManager):
        """
        초기화
        
        Args:
            file_manager: 파일 관리 서비스
        """
        self.file_manager = file_manager
        self.prompts_dir = Path("./prompts")
        self.prompts_dir.mkdir(exist_ok=True)
        self._logger = logging.getLogger(__name__)
        
        # 현재 선택된 프롬프트 정보
        self.current_prompt_file = None
        
    def create_interface(self) -> gr.Tab:
        """
        Prompt Manager 탭 인터페이스를 생성합니다.
        
        Returns:
            Gradio Tab 컴포넌트
        """
        with gr.Tab("Prompt Manager") as tab:
            gr.Markdown("## 📝 Stable Diffusion Prompt Manager")
            gr.Markdown("Stable Diffusion용 프롬프트를 저장하고 관리할 수 있습니다.")
            
            with gr.Row():
                # 왼쪽: 프롬프트 입력/편집 영역
                with gr.Column(scale=2):
                    gr.Markdown("### 프롬프트 작성/편집")
                    
                    # 프롬프트 제목
                    self.title_input = gr.Textbox(
                        label="프롬프트 제목",
                        placeholder="예: 아름다운 풍경, Beautiful Landscape, Anime Character 등",
                        lines=1
                    )
                    
                    # Positive 프롬프트
                    self.positive_input = gr.Textbox(
                        label="Positive 프롬프트",
                        placeholder="생성하고 싶은 이미지에 대한 상세한 설명을 입력하세요...",
                        lines=6,
                        max_lines=10
                    )
                    
                    # Negative 프롬프트
                    self.negative_input = gr.Textbox(
                        label="Negative 프롬프트",
                        placeholder="원하지 않는 요소들을 입력하세요...",
                        lines=4,
                        max_lines=8
                    )
                    
                    # 버튼들
                    with gr.Row():
                        self.save_button = gr.Button(
                            "💾 저장",
                            variant="primary",
                            size="lg"
                        )
                        self.clear_button = gr.Button(
                            "🗑️ 초기화",
                            variant="secondary"
                        )
                        self.delete_button = gr.Button(
                            "❌ 삭제",
                            variant="stop"
                        )
                
                # 오른쪽: 기존 프롬프트 목록 및 상태
                with gr.Column(scale=1):
                    gr.Markdown("### 저장된 프롬프트")
                    
                    # 새로고침 버튼
                    self.refresh_button = gr.Button(
                        "🔄 목록 새로고침",
                        size="sm"
                    )
                    
                    # 프롬프트 목록
                    initial_choices = self._get_prompt_choices()
                    self.prompt_list = gr.Dropdown(
                        label="프롬프트 선택",
                        choices=initial_choices,
                        value=None,
                        interactive=True
                    )
                    
                    # 선택된 프롬프트 미리보기
                    self.preview_title = gr.Textbox(
                        label="선택된 프롬프트 제목",
                        interactive=False,
                        visible=False
                    )
                    
                    self.preview_positive = gr.Textbox(
                        label="Positive 미리보기",
                        lines=4,
                        max_lines=6,
                        interactive=False,
                        visible=False
                    )
                    
                    self.preview_negative = gr.Textbox(
                        label="Negative 미리보기",
                        lines=3,
                        max_lines=4,
                        interactive=False,
                        visible=False
                    )
                    
                    # 상태 표시
                    self.status_message = gr.Textbox(
                        label="상태",
                        value="새로운 프롬프트를 작성하거나 기존 프롬프트를 선택하세요.",
                        interactive=False
                    )
            
            # 사용법 안내
            with gr.Accordion("사용법 안내", open=False):
                gr.Markdown("""
                **프롬프트 저장:**
                1. 프롬프트 제목을 입력하세요 (한글, 영문, 숫자 사용 가능)
                2. Positive와 Negative 프롬프트를 작성하세요
                3. '저장' 버튼을 클릭하세요
                
                **프롬프트 수정:**
                1. 오른쪽에서 수정할 프롬프트를 선택하세요
                2. 왼쪽 입력란에서 내용을 수정하세요
                3. '저장' 버튼을 클릭하세요
                
                **파일 저장 위치:** `./prompts/<제목>.json`
                **제목 규칙:** 한글, 영문, 숫자 사용 가능, 특수문자는 자동으로 언더스코어(_)로 변환
                """)
        
        return tab
    
    def setup_event_handlers(self):
        """이벤트 핸들러를 설정합니다."""
        
        # 저장 버튼 클릭
        self.save_button.click(
            fn=self._save_prompt,
            inputs=[self.title_input, self.positive_input, self.negative_input],
            outputs=[self.status_message, self.prompt_list]
        )
        
        # 초기화 버튼 클릭
        self.clear_button.click(
            fn=self._clear_inputs,
            inputs=[],
            outputs=[
                self.title_input, self.positive_input, self.negative_input,
                self.preview_title, self.preview_positive, self.preview_negative,
                self.status_message
            ]
        )
        
        # 삭제 버튼 클릭
        self.delete_button.click(
            fn=self._delete_prompt,
            inputs=[self.prompt_list],
            outputs=[self.status_message, self.prompt_list, self.title_input, 
                    self.positive_input, self.negative_input]
        )
        
        # 새로고침 버튼 클릭
        self.refresh_button.click(
            fn=self._refresh_prompt_list,
            inputs=[],
            outputs=[self.prompt_list]
        )
        
        # 프롬프트 선택 시
        self.prompt_list.change(
            fn=self._load_selected_prompt,
            inputs=[self.prompt_list],
            outputs=[
                self.title_input, self.positive_input, self.negative_input,
                self.preview_title, self.preview_positive, self.preview_negative,
                self.status_message
            ]
        )
        
        # 초기 프롬프트 목록 설정은 create_interface에서 처리
    
    def _save_prompt(
        self, 
        title: str, 
        positive: str, 
        negative: str
    ) -> Tuple[str, gr.Dropdown]:
        """
        프롬프트를 JSON 파일로 저장합니다.
        
        Args:
            title: 프롬프트 제목
            positive: Positive 프롬프트
            negative: Negative 프롬프트
            
        Returns:
            Tuple[상태 메시지, 업데이트된 프롬프트 목록]
        """
        try:
            # 입력 검증
            if not title.strip():
                return "❌ 프롬프트 제목을 입력해주세요.", gr.Dropdown()
            
            if not positive.strip():
                return "❌ Positive 프롬프트를 입력해주세요.", gr.Dropdown()
            
            # 파일명으로 사용할 수 있도록 제목 정리
            safe_title = self._sanitize_filename(title.strip())
            if not safe_title:
                return "❌ 유효하지 않은 제목입니다. 한글, 영문, 숫자를 사용해주세요.", gr.Dropdown()
            
            # 프롬프트 데이터 생성
            prompt_data = {
                "title": title.strip(),
                "positive": positive.strip(),
                "negative": negative.strip(),
                "created_at": self._get_current_timestamp(),
                "updated_at": self._get_current_timestamp()
            }
            
            # 파일 경로
            filepath = self.prompts_dir / f"{safe_title}.json"
            
            # 기존 파일이 있으면 업데이트 시간만 변경
            if filepath.exists():
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                    prompt_data["created_at"] = existing_data.get("created_at", prompt_data["created_at"])
                except:
                    pass  # 기존 파일을 읽을 수 없으면 새로 생성
            
            # JSON 파일 저장
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(prompt_data, f, ensure_ascii=False, indent=2)
            
            self.current_prompt_file = safe_title
            self._logger.info(f"프롬프트 저장 완료: {filepath}")
            
            # 업데이트된 프롬프트 목록 반환
            prompt_choices = self._get_prompt_choices()
            updated_dropdown = gr.Dropdown(choices=prompt_choices, value=safe_title)
            
            return f"✅ 프롬프트가 저장되었습니다: {safe_title}.json", updated_dropdown
            
        except Exception as e:
            self._logger.error(f"프롬프트 저장 실패: {e}")
            return f"❌ 프롬프트 저장 중 오류가 발생했습니다: {str(e)}", gr.Dropdown()
    
    def _clear_inputs(self) -> Tuple[str, str, str, str, str, str, str]:
        """
        모든 입력 필드를 초기화합니다.
        
        Returns:
            Tuple[초기화된 입력값들과 상태 메시지]
        """
        self.current_prompt_file = None
        return (
            "",  # title_input
            "",  # positive_input  
            "",  # negative_input
            "",  # preview_title (hidden)
            "",  # preview_positive (hidden)
            "",  # preview_negative (hidden)
            "입력 필드가 초기화되었습니다."  # status_message
        )
    
    def _delete_prompt(
        self, 
        selected_prompt: str
    ) -> Tuple[str, gr.Dropdown, str, str, str]:
        """
        선택된 프롬프트를 삭제합니다.
        
        Args:
            selected_prompt: 선택된 프롬프트 파일명
            
        Returns:
            Tuple[상태 메시지, 업데이트된 목록, 초기화된 입력들]
        """
        try:
            if not selected_prompt:
                return "❌ 삭제할 프롬프트를 선택해주세요.", gr.Dropdown(), "", "", ""
            
            filepath = self.prompts_dir / f"{selected_prompt}.json"
            
            if not filepath.exists():
                return "❌ 선택된 프롬프트 파일을 찾을 수 없습니다.", gr.Dropdown(), "", "", ""
            
            # 파일 삭제
            filepath.unlink()
            self.current_prompt_file = None
            self._logger.info(f"프롬프트 삭제 완료: {filepath}")
            
            # 업데이트된 프롬프트 목록
            prompt_choices = self._get_prompt_choices()
            updated_dropdown = gr.Dropdown(choices=prompt_choices, value=None)
            
            return (
                f"✅ 프롬프트가 삭제되었습니다: {selected_prompt}.json",
                updated_dropdown,
                "",  # title_input
                "",  # positive_input
                ""   # negative_input
            )
            
        except Exception as e:
            self._logger.error(f"프롬프트 삭제 실패: {e}")
            return f"❌ 프롬프트 삭제 중 오류가 발생했습니다: {str(e)}", gr.Dropdown(), "", "", ""
    
    def _refresh_prompt_list(self) -> gr.Dropdown:
        """
        프롬프트 목록을 새로고침합니다.
        
        Returns:
            업데이트된 Dropdown 컴포넌트
        """
        try:
            prompt_choices = self._get_prompt_choices()
            return gr.Dropdown(choices=prompt_choices, value=None)
        except Exception as e:
            self._logger.error(f"프롬프트 목록 새로고침 실패: {e}")
            return gr.Dropdown(choices=[], value=None)
    
    def _load_selected_prompt(
        self, 
        selected_prompt: str
    ) -> Tuple[str, str, str, str, str, str, str]:
        """
        선택된 프롬프트를 로드합니다.
        
        Args:
            selected_prompt: 선택된 프롬프트 파일명
            
        Returns:
            Tuple[로드된 데이터들과 미리보기, 상태 메시지]
        """
        try:
            if not selected_prompt:
                return "", "", "", "", "", "", "프롬프트를 선택해주세요."
            
            filepath = self.prompts_dir / f"{selected_prompt}.json"
            
            if not filepath.exists():
                return "", "", "", "", "", "", f"❌ 파일을 찾을 수 없습니다: {selected_prompt}.json"
            
            # JSON 파일 읽기
            with open(filepath, 'r', encoding='utf-8') as f:
                prompt_data = json.load(f)
            
            self.current_prompt_file = selected_prompt
            
            title = prompt_data.get("title", "")
            positive = prompt_data.get("positive", "")
            negative = prompt_data.get("negative", "")
            
            status = f"✅ 프롬프트를 로드했습니다: {selected_prompt}.json"
            
            return (
                title,     # title_input
                positive,  # positive_input
                negative,  # negative_input
                title,     # preview_title
                positive,  # preview_positive
                negative,  # preview_negative
                status     # status_message
            )
            
        except Exception as e:
            self._logger.error(f"프롬프트 로드 실패: {e}")
            return "", "", "", "", "", "", f"❌ 프롬프트 로드 중 오류가 발생했습니다: {str(e)}"
    
    def _get_prompt_choices(self) -> List[str]:
        """
        저장된 프롬프트 파일들의 목록을 반환합니다.
        
        Returns:
            프롬프트 파일명 목록 (확장자 제외)
        """
        try:
            if not self.prompts_dir.exists():
                return []
            
            json_files = list(self.prompts_dir.glob("*.json"))
            return [f.stem for f in json_files]
            
        except Exception as e:
            self._logger.error(f"프롬프트 목록 조회 실패: {e}")
            return []
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        파일명으로 사용할 수 있도록 문자열을 정리합니다.
        한글, 영문 대소문자, 숫자는 허용하고 파일명에 사용할 수 없는 문자만 치환합니다.
        
        Args:
            filename: 원본 파일명
            
        Returns:
            정리된 파일명
        """
        import re
        # 파일명에 사용할 수 없는 문자들을 언더스코어로 치환
        # Windows: < > : " | ? * / \
        # 추가로 공백도 언더스코어로 치환
        sanitized = re.sub(r'[<>:"|?*\\/\s]', '_', filename)
        # 연속된 언더스코어 제거
        sanitized = re.sub(r'_+', '_', sanitized)
        # 양쪽 언더스코어 제거
        sanitized = sanitized.strip('_')
        return sanitized
    
    def _get_current_timestamp(self) -> str:
        """
        현재 타임스탬프를 반환합니다.
        
        Returns:
            현재 시간 문자열
        """
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
