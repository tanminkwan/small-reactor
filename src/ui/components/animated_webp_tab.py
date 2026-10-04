"""
비디오 WebP 애니메이션 3단계 얼굴 교체 탭 컴포넌트

Single Responsibility Principle (SRP)에 따라
Animated WebP 3단계 UI 처리만을 담당하는 컴포넌트
"""

import gradio as gr
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

from src.services.webp_service import WebPService
from src.services.face_manager import FaceManager
from src.services.file_manager import FileManager
from src.utils.file_utils import open_save_location
from src.utils.config import Config


class AnimatedWebPTab:
    """비디오 WebP 애니메이션 3단계 얼굴 교체 탭 컴포넌트"""
    
    def __init__(self, webp_service: WebPService, face_manager: FaceManager, file_manager: FileManager, config: Optional[Config] = None):
        """
        초기화
        
        Args:
            webp_service: WebP 처리 서비스
            face_manager: 얼굴 관리 서비스
            file_manager: 파일 관리 서비스
            config: 설정 객체
        """
        self.webp_service = webp_service
        self.face_manager = face_manager
        self.file_manager = file_manager
        self.config = config or Config()
        self.last_result_path = None
    
    def create_interface(self) -> gr.Tab:
        """
        Animated WebP 탭 인터페이스를 생성합니다.
        
        Returns:
            Gradio Tab 컴포넌트
        """
        default_extract_path = self.config.get("webp_extract_output_path", "./outputs/webp_frames/extracted")
        default_swap_path = self.config.get("webp_swap_output_path", "./outputs/webp_frames/swapped")
        default_animated_path = self.config.get("webp_animated_output_path", "./outputs/animated_swapped.webp")

        with gr.Tab("비디오 WebP/MP4 얼굴 교체") as tab:
            gr.Markdown("## 🎬 비디오 WebP/MP4 3단계 애니메이션 얼굴 교체")
            gr.Markdown("비디오 (WebP, MP4) 파일의 프레임 분해 -> 일괄 얼굴 변경 -> 비디오 WebP 결합을 순차적으로 수행합니다. (최대 300초)")
            
            # --- 1단계: 비디오 영상 프레임 분해 ---
            with gr.Accordion("1️⃣ Step 1: 비디오 (WebP, MP4) 영상 프레임 분해", open=True):
                with gr.Row():
                    with gr.Column(scale=1):
                        self.input_webp = gr.File(
                            label="비디오 (WebP, MP4) 파일 선택 또는 드래그",
                            file_count="single",
                            type="filepath",
                            file_types=[".webp", ".mp4"]
                        )
                        self.extract_output_dir = gr.Textbox(
                            label="프레임 저장 디렉터리 경로",
                            value=default_extract_path,
                            info="분해된 프레임 이미지가 저장될 디렉터리 경로"
                        )
                        self.extract_btn = gr.Button("🎬 프레임 이미지 분해 실행", variant="primary")
                    
                    with gr.Column(scale=1):
                        self.extract_result_text = gr.Textbox(
                            label="1단계 처리 결과",
                            lines=5,
                            interactive=False
                        )
            
            # --- 2단계: 일괄 얼굴 교체 ---
            with gr.Accordion("2️⃣ Step 2: 프레임 일괄 얼굴 변경", open=True):
                with gr.Row():
                    with gr.Column(scale=1):
                        self.swap_input_dir = gr.Textbox(
                            label="입력 프레임 디렉터리 경로",
                            value=default_extract_path,
                            info="얼굴 교체를 수행할 프레임 이미지가 있는 디렉터리"
                        )
                        self.swap_output_dir = gr.Textbox(
                            label="얼굴 변경 프레임 저장 디렉터리 경로",
                            value=default_swap_path,
                            info="1단계 저장 디렉터리와 다른 디렉터리를 지정하세요"
                        )
                        self.face_indices_input = gr.Textbox(
                            label="교체할 얼굴 인덱스 (쉼표로 구분, 비워두면 모든 얼굴)",
                            placeholder="예: 1,3,5 또는 비워두기",
                            value=""
                        )
                        
                        with gr.Row():
                            choices = self.file_manager.get_embedding_choices()
                            default_val = choices[0] if choices else None
                            self.source_face_dropdown = gr.Dropdown(
                                label="바꿀 얼굴 선택",
                                choices=choices,
                                value=default_val,
                                scale=4
                            )
                            self.refresh_faces_btn = gr.Button("🔄", variant="secondary", size="sm", scale=1)
                        
                        self.codeformer_checkbox = gr.Checkbox(
                            label="CodeFormer 복원 포함",
                            value=True,
                            info="체크하면 얼굴 교체 후 자동으로 CodeFormer 복원도 수행됩니다"
                        )
                        self.fidelity_slider = gr.Slider(
                            label="Fidelity (복원 강도)",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.1,
                            info="높을수록 원본에 가깝게 복원됩니다 (0.0: 완전 복원, 1.0: 원본 유지)"
                        )
                        self.swap_btn = gr.Button("🔄 일괄 얼굴 변경 실행", variant="primary")
                    
                    with gr.Column(scale=1):
                        self.swap_result_text = gr.Textbox(
                            label="2단계 처리 결과",
                            lines=5,
                            interactive=False
                        )
                        self.swapped_sample_image = gr.Image(
                            label="마지막 변환 프레임 미리보기",
                            type="numpy"
                        )
            
            # --- 3단계: 비디오 WebP 결합 ---
            with gr.Accordion("3️⃣ Step 3: 비디오 WebP 결합", open=True):
                with gr.Row():
                    with gr.Column(scale=1):
                        self.combine_input_dir = gr.Textbox(
                            label="프레임 디렉터리 경로",
                            value=default_swap_path,
                            info="결합할 프레임 이미지들이 있는 디렉터리"
                        )
                        self.combine_output_path = gr.Textbox(
                            label="출력 비디오 WebP 파일 경로",
                            value=default_animated_path,
                            info="결과 비디오 WebP 애니메이션 파일 경로"
                        )
                        self.fps_input = gr.Number(
                            label="FPS (초당 프레임 수)",
                            value=25.0,
                            precision=2
                        )
                        self.loop_input = gr.Number(
                            label="루프 반복 횟수 (0: 무한 반복)",
                            value=0,
                            precision=0
                        )
                        self.combine_btn = gr.Button("📽️ 비디오 WebP 결합 실행", variant="primary")
                    
                    with gr.Column(scale=1):
                        self.combine_result_text = gr.Textbox(
                            label="3단계 처리 결과",
                            lines=6,
                            interactive=False
                        )
                        self.save_location_btn = gr.Button(
                            "📁 저장 위치 확인",
                            variant="secondary",
                            size="sm"
                        )
        
        return tab

    def setup_event_handlers(self):
        """이벤트 핸들러를 바인딩합니다."""
        # 바꿀 얼굴 드롭다운 새로고침
        self.refresh_faces_btn.click(
            fn=self._refresh_face_choices,
            inputs=[],
            outputs=[self.source_face_dropdown]
        )
        
        # Step 1: 프레임 분해
        self.extract_btn.click(
            fn=self._on_extract_frames,
            inputs=[self.input_webp, self.extract_output_dir],
            outputs=[self.extract_result_text, self.swap_input_dir, self.fps_input]
        )
        
        # Step 2: 일괄 얼굴 교체
        self.swap_btn.click(
            fn=self._on_batch_face_swap,
            inputs=[
                self.swap_input_dir,
                self.swap_output_dir,
                self.face_indices_input,
                self.source_face_dropdown,
                self.codeformer_checkbox,
                self.fidelity_slider
            ],
            outputs=[self.swap_result_text, self.swapped_sample_image, self.combine_input_dir]
        )
        
        # Step 3: 비디오 WebP 결합
        self.combine_btn.click(
            fn=self._on_combine_frames,
            inputs=[
                self.combine_input_dir,
                self.combine_output_path,
                self.fps_input,
                self.loop_input
            ],
            outputs=[self.combine_result_text]
        )
        
        # 저장 위치 확인
        self.save_location_btn.click(
            fn=self._show_save_location,
            inputs=[],
            outputs=[self.combine_result_text]
        )

    def _refresh_face_choices(self):
        """저장된 얼굴 선택 목록을 새로고침합니다."""
        choices = self.file_manager.get_embedding_choices()
        val = choices[0] if choices else None
        return gr.Dropdown(choices=choices, value=val)

    def _on_extract_frames(self, webp_file: Optional[str], output_dir: str) -> Tuple[str, str, float]:
        """1단계 프레임 추출 이벤트 핸들러"""
        if not webp_file:
            return "❌ 비디오 (WebP, MP4) 파일을 업로드해주세요.", output_dir, 25.0
        
        ok, msg, count, fps = self.webp_service.extract_webp_frames(webp_file, output_dir)
        if ok:
            return msg, output_dir, fps
        return msg, output_dir, 25.0

    def _on_batch_face_swap(
        self,
        input_dir: str,
        output_dir: str,
        face_indices: str,
        source_face_name: str,
        use_codeformer: bool,
        fidelity: float,
        progress=gr.Progress()
    ):
        """2단계 일괄 얼굴 교체 이벤트 핸들러 (실시간 진행 상황 스트리밍)"""
        frame_files = self.webp_service._get_sorted_frame_files(input_dir)
        if not frame_files:
            yield f"❌ 입력 디렉터리에 프레임 이미지 파일이 없습니다: {input_dir}", None, output_dir
            return
        
        total = len(frame_files)
        yield f"⏳ 일괄 얼굴 교체 작업 시작... (총 {total}개 프레임)", None, output_dir
        
        for current, total_count, last_img, is_finished, step_msg in self.webp_service.batch_face_swap_generator(
            input_dir=input_dir,
            output_dir=output_dir,
            face_indices=face_indices,
            source_face_name=source_face_name,
            use_codeformer=use_codeformer,
            fidelity=fidelity,
            face_manager=self.face_manager,
            file_manager=self.file_manager
        ):
            if total_count > 0:
                pct = (current / total_count) * 100.0
                progress(current / total_count, desc=f"{current}/{total_count} 프레임 완료")
                
                if is_finished:
                    status_text = (
                        f"✅ 일괄 얼굴 교체 완료!\n"
                        f"📊 진행률: {current} / {total_count} 프레임 완료 ({pct:.1f}%)\n"
                        f"📂 저장 경로: {Path(output_dir).resolve()}"
                    )
                else:
                    status_text = (
                        f"🔄 일괄 얼굴 교체 진행 중...\n"
                        f"📊 진행률: {current} / {total_count} 프레임 완료 ({pct:.1f}%)\n"
                        f"📂 저장 경로: {Path(output_dir).resolve()}\n"
                        f"📝 {step_msg}"
                    )
                yield status_text, last_img, output_dir
            else:
                yield step_msg, None, output_dir

    def _on_combine_frames(
        self,
        input_dir: str,
        output_path: str,
        fps: float,
        loop: float
    ) -> str:
        """3단계 비디오 WebP 결합 이벤트 핸들러"""
        loop_int = int(loop) if loop is not None else 0
        ok, msg, result_file = self.webp_service.combine_frames_to_webp(
            input_dir=input_dir,
            output_path=output_path,
            fps=float(fps or 25.0),
            loop=loop_int
        )
        
        if ok and result_file:
            self.last_result_path = result_file
            return msg
        return msg

    def _show_save_location(self) -> str:
        """저장 위치 폴더를 엽니다."""
        try:
            if self.last_result_path and Path(self.last_result_path).exists():
                parent_dir = str(Path(self.last_result_path).parent)
                return open_save_location(parent_dir, self.last_result_path)
            output_dir = str(Path("./outputs").resolve())
            return open_save_location(output_dir)
        except Exception as e:
            return f"❌ 저장 위치 열기 실패: {str(e)}"
