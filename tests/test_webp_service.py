"""
WebPService 단위 테스트
"""

import os
import shutil
import pytest
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
from PIL import Image

from src.services.webp_service import WebPService


@pytest.fixture
def webp_service():
    return WebPService()


@pytest.fixture
def temp_dir(tmp_path):
    d = tmp_path / "webp_test_dir"
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def create_animated_webp(output_path: Path, frame_count: int = 3, duration: int = 40):
    frames = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i in range(frame_count):
        color = colors[i % len(colors)]
        img = Image.new("RGB", (64, 64), color=color)
        frames.append(img)
    
    frames[0].save(
        output_path,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )


def test_extract_webp_frames_invalid_file(webp_service, temp_dir):
    ok, msg, count, fps = webp_service.extract_webp_frames("non_existent.webp", str(temp_dir / "out"))
    assert not ok
    assert count == 0
    assert "존재하지 않는" in msg


def test_extract_webp_frames_success(webp_service, temp_dir):
    webp_path = temp_dir / "sample_anim.webp"
    create_animated_webp(webp_path, frame_count=3, duration=50)  # 50ms = 20 fps
    
    out_dir = temp_dir / "extracted_frames"
    ok, msg, count, fps = webp_service.extract_webp_frames(str(webp_path), str(out_dir))
    
    assert ok
    assert count == 3
    assert fps == pytest.approx(20.0, abs=3.0)
    
    extracted_files = sorted(list(out_dir.glob("*.webp")))
    assert len(extracted_files) == 3
    assert extracted_files[0].name == "frame_0001.webp"
    assert extracted_files[1].name == "frame_0002.webp"
    assert extracted_files[2].name == "frame_0003.webp"


def test_batch_face_swap_success(webp_service, temp_dir):
    # 1. 입력 디렉터리 및 프레임 생성
    in_dir = temp_dir / "in_frames"
    out_dir = temp_dir / "out_frames"
    in_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(2):
        img = Image.new("RGB", (64, 64), color=(i * 100, 50, 50))
        img.save(in_dir / f"frame_{i+1:04d}.webp", format="WEBP")
    
    # 2. Mock 서비스 설정
    mock_face_manager = MagicMock()
    mock_file_manager = MagicMock()
    mock_file_manager.faces_dir = "./faces"
    
    dummy_swapped_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    mock_face_manager.swap_faces.return_value = (True, "Swap success", dummy_swapped_rgb)
    mock_face_manager.enhance_faces_with_codeformer.return_value = (True, "CF success", dummy_swapped_rgb)
    
    progress_calls = []
    def mock_progress(ratio, desc):
        progress_calls.append((ratio, desc))
    
    # 3. 일괄 얼굴 교체 수행
    ok, msg, count, last_img = webp_service.batch_face_swap(
        input_dir=str(in_dir),
        output_dir=str(out_dir),
        face_indices="1",
        source_face_name="source_face.jpg",
        use_codeformer=True,
        fidelity=0.5,
        face_manager=mock_face_manager,
        file_manager=mock_file_manager,
        progress_fn=mock_progress
    )
    
    assert ok
    assert count == 2
    assert last_img is not None
    assert mock_face_manager.swap_faces.call_count == 2
    assert mock_face_manager.enhance_faces_with_codeformer.call_count == 2
    assert len(progress_calls) == 2
    
    swapped_files = sorted(list(out_dir.glob("*.webp")))
    assert len(swapped_files) == 2


def test_batch_face_swap_generator_success(webp_service, temp_dir):
    in_dir = temp_dir / "gen_in_frames"
    out_dir = temp_dir / "gen_out_frames"
    in_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(3):
        img = Image.new("RGB", (64, 64), color=(i * 50, 50, 50))
        img.save(in_dir / f"frame_{i+1:04d}.webp", format="WEBP")
    
    mock_face_manager = MagicMock()
    mock_file_manager = MagicMock()
    mock_file_manager.faces_dir = "./faces"
    
    dummy_swapped_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    mock_face_manager.swap_faces.return_value = (True, "Swap success", dummy_swapped_rgb)
    mock_face_manager.enhance_faces_with_codeformer.return_value = (True, "CF success", dummy_swapped_rgb)
    
    yielded_results = list(webp_service.batch_face_swap_generator(
        input_dir=str(in_dir),
        output_dir=str(out_dir),
        face_indices="",
        source_face_name="test_face.jpg",
        use_codeformer=True,
        fidelity=0.5,
        face_manager=mock_face_manager,
        file_manager=mock_file_manager
    ))
    
    assert len(yielded_results) == 3
    first_current, first_total, first_img, first_finished, first_msg = yielded_results[0]
    assert first_current == 1
    assert first_total == 3
    assert not first_finished
    
    last_current, last_total, last_img, last_finished, last_msg = yielded_results[-1]
    assert last_current == 3
    assert last_total == 3
    assert last_finished


def test_combine_frames_to_webp(webp_service, temp_dir):
    in_dir = temp_dir / "combine_in"
    in_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(3):
        img = Image.new("RGB", (64, 64), color=(0, i * 80, 100))
        img.save(in_dir / f"frame_{i+1:04d}.webp", format="WEBP")
    
    out_file = temp_dir / "combined_output.webp"
    ok, msg, res_path = webp_service.combine_frames_to_webp(
        input_dir=str(in_dir),
        output_path=str(out_file),
        fps=10.0,
        loop=0
    )
    
    assert ok
    assert res_path == str(out_file.resolve())
    assert out_file.exists()
    
    with Image.open(out_file) as im:
        assert getattr(im, "n_frames", 1) == 3
        count = 0
        for i in range(im.n_frames):
            im.seek(i)
            count += 1
        assert count == 3


def test_extract_webp_frames_over_300s_limit(webp_service, temp_dir, monkeypatch):
    mock_im = MagicMock()
    mock_im.__enter__.return_value = mock_im
    mock_im.__exit__.return_value = None
    mock_im.n_frames = 8000  # 8000 frames * 40ms = 320s > 300s
    mock_im.info = {}
    mock_im.convert.return_value = Image.new("RGB", (8, 8))
    
    monkeypatch.setattr(Image, "open", lambda path: mock_im)

    webp_path = temp_dir / "over_limit.webp"
    webp_path.touch()
    
    out_dir = temp_dir / "extracted_frames_over"
    ok, msg, count, fps = webp_service.extract_webp_frames(str(webp_path), str(out_dir))
    
    assert not ok
    assert count == 0
    assert "300초 초과" in msg
    assert len(list(out_dir.glob("*.webp"))) == 0


def test_extract_mp4_frames_over_300s_limit(webp_service, temp_dir, monkeypatch):
    import cv2
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    
    def mock_get(prop_id):
        if prop_id == cv2.CAP_PROP_FPS:
            return 25.0
        elif prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return 8000  # 8000 / 25 = 320 seconds (> 300s)
        return 0.0

    mock_cap.get.side_effect = mock_get

    monkeypatch.setattr(cv2, "VideoCapture", lambda path: mock_cap)

    mp4_path = temp_dir / "over_limit.mp4"
    mp4_path.touch()

    out_dir = temp_dir / "extracted_mp4_over"
    ok, msg, count, fps = webp_service.extract_webp_frames(str(mp4_path), str(out_dir))

    assert not ok
    assert count == 0
    assert "300초 초과" in msg
    assert mock_cap.release.called

