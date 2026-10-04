"""
얼굴 교체 2 탭 컴포넌트 (HyperSwap)

Open/Closed Principle (OCP)에 따라
기존 얼굴 교체 탭(FaceSwapTab)의 UI와 기능을 그대로 재사용하고,
얼굴 교체 엔진만 HyperSwap 모델로 교체한 컴포넌트
"""

import gradio as gr
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

from src.ui.components.face_swap_tab import FaceSwapTab


class FaceSwap2Tab(FaceSwapTab):
    """HyperSwap 모델을 사용하는 얼굴 교체 2 탭 컴포넌트"""

    TAB_TITLE = "얼굴 교체 2"
    TAB_HEADER = "## 🔄 얼굴 교체 2 (HyperSwap 256)"

    def _create_extra_controls(self) -> None:
        """HyperSwap 모델 선택 드롭다운을 생성합니다."""
        choices = self.face_manager.get_hyperswap_model_choices()
        self.hyperswap_model_dropdown = gr.Dropdown(
            label="HyperSwap 모델",
            choices=choices,
            value=choices[0] if choices else None,
            info="1a: 속도·안정성 / 1b: 균형 / 1c: 품질·닮음 (얼굴에 따라 결과가 다르니 비교해보세요)"
        )

    def get_extra_swap_inputs(self) -> list:
        """얼굴 교체 시 모델 이름을 함께 전달합니다."""
        return [("model_name", self.hyperswap_model_dropdown)]

    def _swap_faces(self, image_bgr: np.ndarray, face_indices: str, source_face_name: str, **swap_options) -> Tuple[bool, str, Optional[np.ndarray]]:
        """HyperSwap 모델로 얼굴 교체를 수행합니다."""
        return self.face_manager.swap_faces_hyperswap(
            image_bgr,
            face_indices,
            source_face_name,
            self.file_manager.faces_dir,
            model_name=swap_options.get("model_name")
        )

    def _get_result_prefix(self, is_no_face_selected: bool, **swap_options) -> str:
        """결과 파일명 접두사를 반환합니다. (예: hyperswap_1a_256 -> final_hyperswap1a)"""
        if is_no_face_selected:
            return "final_codeformer"
        model_name = swap_options.get("model_name") or Path(
            self.face_manager.config.get_model_path("hyperswap")
        ).stem
        parts = model_name.split("_")
        short_name = "".join(parts[:2]) if len(parts) >= 2 else model_name
        return f"final_{short_name}"
