"""
WebP 비디오 애니메이션 분해, 일괄 얼굴 교체, 결합 서비스

Single Responsibility Principle (SRP)에 따라
WebP 애니메이션 프레임 처리 및 합성만을 담당하는 서비스
"""

import os
import re
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Callable
import numpy as np
import cv2
from PIL import Image

from src.services.face_manager import FaceManager
from src.services.file_manager import FileManager


class WebPService:
    """WebP 비디오 애니메이션 분해 및 합성 서비스"""
    
    def __init__(self):
        """초기화"""
        self._logger = logging.getLogger(__name__)
    
    def _get_sorted_frame_files(self, directory: str) -> List[Path]:
        """
        디렉터리 내의 이미지 파일들을 자연스러운 숫자 정렬 순서로 가져옵니다.
        
        Args:
            directory: 이미지 디렉터리 경로
            
        Returns:
            정렬된 Path 객체 리스트
        """
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            return []
        
        valid_extensions = {".webp", ".png", ".jpg", ".jpeg", ".bmp"}
        files = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in valid_extensions]
        
        # 숫자 정렬 (natural sort)
        def natural_key(path_obj: Path):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', path_obj.name)]
        
        return sorted(files, key=natural_key)

    def extract_webp_frames(self, webp_path: str, output_dir: str) -> Tuple[bool, str, int, float]:
        """
        비디오 (WebP, MP4) 파일에서 프레임 이미지들을 추출하여 지정된 디렉터리에 저장합니다.
        재생 시간이 300초를 초과하면 작업을 중단하고 에러 메시지를 반환합니다.
        
        Args:
            webp_path: 비디오 (WebP, MP4) 파일 경로
            output_dir: 추출된 프레임 이미지를 저장할 디렉터리 경로
            
        Returns:
            (성공 여부, 메시지, 추출된 프레임 수, 계산된 FPS)
        """
        if not webp_path or not os.path.exists(webp_path):
            return False, "❌ 존재하지 않는 비디오 파일입니다.", 0, 0.0
        
        try:
            out_dir_path = Path(output_dir)
            out_dir_path.mkdir(parents=True, exist_ok=True)
            
            ext = Path(webp_path).suffix.lower()
            if ext in [".mp4", ".mkv", ".avi", ".mov", ".webm"]:
                return self._extract_mp4_frames(webp_path, out_dir_path)
            else:
                return self._extract_pil_webp_frames(webp_path, out_dir_path)
                
        except Exception as e:
            err_msg = f"❌ 비디오 프레임 추출 중 오류 발생: {str(e)}"
            self._logger.error(err_msg)
            return False, err_msg, 0, 0.0

    def _extract_mp4_frames(self, video_path: str, out_dir_path: Path) -> Tuple[bool, str, int, float]:
        """MP4 등 비디오 파일에서 프레임을 추출합니다 (300초 초과 검사 포함)."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "❌ 비디오 파일을 열 수 없습니다.", 0, 0.0
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or np.isnan(fps):
                fps = 25.0
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count > 0 and fps > 0:
                duration_sec = frame_count / fps
                if duration_sec > 300.0:
                    cap.release()
                    err_msg = f"❌ 비디오 재생 시간이 너무 깁니다. (300초 초과: {duration_sec:.1f}초) 300초 이하의 영상만 처리 가능합니다."
                    self._logger.warning(err_msg)
                    return False, err_msg, 0, 0.0

            count = 0
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                count += 1

                duration_sec = count / fps
                if duration_sec > 300.0:
                    cap.release()
                    for f in out_dir_path.glob("frame_*.webp"):
                        try:
                            f.unlink()
                        except Exception:
                            pass
                    err_msg = f"❌ 비디오 재생 시간이 너무 깁니다. (300초 초과: {duration_sec:.1f}초) 300초 이하의 영상만 처리 가능합니다."
                    self._logger.warning(err_msg)
                    return False, err_msg, 0, 0.0

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                save_name = f"frame_{count:04d}.webp"
                frame_save_path = out_dir_path / save_name
                Image.fromarray(frame_rgb).save(frame_save_path, format="WEBP")

            cap.release()

            if count == 0:
                return False, "❌ 비디오에서 프레임을 추출할 수 없습니다.", 0, 0.0

            fps_final = round(float(fps), 2)
            msg = f"✅ 성공: 총 {count}개 프레임 추출 완료\n📂 저장 경로: {out_dir_path.resolve()}\n⏱️ 추정 FPS: {fps_final}"
            self._logger.info(f"MP4/비디오 프레임 추출 완료 ({count}개, {fps_final} FPS)")
            return True, msg, count, fps_final

        except Exception as e:
            cap.release()
            err_msg = f"❌ 비디오 프레임 추출 중 오류 발생: {str(e)}"
            self._logger.error(err_msg)
            return False, err_msg, 0, 0.0

    def _extract_pil_webp_frames(self, webp_path: str, out_dir_path: Path) -> Tuple[bool, str, int, float]:
        """WebP 애니메이션 이미지에서 프레임을 추출합니다 (300초 초과 검사 포함)."""
        try:
            with Image.open(webp_path) as im:
                frame_count = getattr(im, "n_frames", 1)
                default_duration = im.info.get("duration", 40)
                if not default_duration or default_duration <= 0:
                    default_duration = 40

                durations = []
                frames_rgb = []
                
                for i in range(frame_count):
                    im.seek(i)
                    frame_duration = im.info.get("duration", default_duration)
                    durations.append(frame_duration if frame_duration > 0 else default_duration)
                    frames_rgb.append(im.convert("RGB"))

                total_duration_sec = sum(durations) / 1000.0
                if total_duration_sec > 300.0:
                    err_msg = f"❌ 비디오 재생 시간이 너무 깁니다. (300초 초과: {total_duration_sec:.1f}초) 300초 이하의 영상만 처리 가능합니다."
                    self._logger.warning(err_msg)
                    return False, err_msg, 0, 0.0

                for i, frame_rgb in enumerate(frames_rgb):
                    save_name = f"frame_{i + 1:04d}.webp"
                    frame_save_path = out_dir_path / save_name
                    frame_rgb.save(frame_save_path, format="WEBP")

                avg_duration = sum(durations) / len(durations) if durations else 40.0
                fps = round(1000.0 / avg_duration, 2) if avg_duration > 0 else 25.0

                msg = f"✅ 성공: 총 {frame_count}개 프레임 추출 완료\n📂 저장 경로: {out_dir_path.resolve()}\n⏱️ 추정 FPS: {fps}"
                self._logger.info(f"WebP 프레임 추출 완료 ({frame_count}개, {fps} FPS)")
                return True, msg, frame_count, fps

        except Exception as e:
            try:
                cap = cv2.VideoCapture(webp_path)
                if cap.isOpened():
                    cap.release()
                    return self._extract_mp4_frames(webp_path, out_dir_path)
            except Exception:
                pass
            
            err_msg = f"❌ WebP/비디오 프레임 추출 중 오류 발생: {str(e)}"
            self._logger.error(err_msg)
            return False, err_msg, 0, 0.0

    def batch_face_swap(
        self,
        input_dir: str,
        output_dir: str,
        face_indices: str,
        source_face_name: str,
        use_codeformer: bool,
        fidelity: float,
        face_manager: FaceManager,
        file_manager: FileManager,
        progress_fn: Optional[Callable[[float, str], None]] = None,
        swap_model: Optional[str] = None
    ) -> Tuple[bool, str, int, Optional[np.ndarray]]:
        """
        지정된 디렉터리의 프레임 이미지들에 대해 일괄 얼굴 교체 및 복원을 수행합니다.
        
        Args:
            input_dir: 입력 프레임 디렉터리 경로
            output_dir: 출력 프레임 디렉터리 경로
            face_indices: 교체할 얼굴 인덱스
            source_face_name: 바꿀 얼굴 embedding 이름
            use_codeformer: CodeFormer 복원 수행 여부
            fidelity: CodeFormer Fidelity 설정
            face_manager: FaceManager 서비스 객체
            file_manager: FileManager 서비스 객체
            progress_fn: 진행 상황 콜백 함수
            
        Returns:
            (성공 여부, 메시지, 처리된 프레임 수, 마지막 프레임 이미지 numpy array)
        """
        frame_files = self._get_sorted_frame_files(input_dir)
        if not frame_files:
            return False, f"❌ 입력 디렉터리에 프레임 이미지 파일이 없습니다: {input_dir}", 0, None
        
        in_path = Path(input_dir).resolve()
        out_path = Path(output_dir).resolve()
        
        if in_path == out_path:
            return False, "❌ 입력 디렉터리와 출력 디렉터리는 서로 달라야 합니다.", 0, None
        
        is_no_face = not source_face_name or source_face_name == "선택 안함"
        if is_no_face and not use_codeformer:
            return False, "❌ 바꿀 얼굴을 선택하거나 CodeFormer 복원을 체크해주세요.", 0, None
        
        out_path.mkdir(parents=True, exist_ok=True)
        
        total = len(frame_files)
        success_count = 0
        last_frame_rgb = None
        
        try:
            for idx, frame_file in enumerate(frame_files):
                if progress_fn:
                    progress_fn((idx + 1) / total, f"프레임 처리 중... ({idx + 1}/{total})")
                
                # 이미지 읽기 (PIL -> RGB)
                pil_img = Image.open(frame_file).convert("RGB")
                img_rgb = np.array(pil_img)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                
                final_rgb = img_rgb
                
                # 1. 얼굴 교체 수행 (선택된 얼굴이 있는 경우)
                if not is_no_face:
                    if swap_model and swap_model.startswith("hyperswap"):
                        swap_ok, swap_msg, swapped_rgb = face_manager.swap_faces_hyperswap(
                            img_bgr, face_indices, source_face_name, file_manager.faces_dir,
                            model_name=swap_model
                        )
                    else:
                        swap_ok, swap_msg, swapped_rgb = face_manager.swap_faces(
                            img_bgr, face_indices, source_face_name, file_manager.faces_dir
                        )
                    if swap_ok and swapped_rgb is not None:
                        final_rgb = swapped_rgb
                
                # 2. CodeFormer 복원 수행 (선택된 경우)
                if use_codeformer:
                    current_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                    cf_ok, cf_msg, enhanced_rgb = face_manager.enhance_faces_with_codeformer(
                        current_bgr, face_indices, fidelity
                    )
                    if cf_ok and enhanced_rgb is not None:
                        final_rgb = enhanced_rgb
                
                # 웹피 프레임 파일로 저장
                dest_file = out_path / f"{frame_file.stem}.webp"
                Image.fromarray(final_rgb).save(dest_file, format="WEBP")
                
                success_count += 1
                last_frame_rgb = final_rgb
            
            msg = f"✅ 일괄 얼굴 교체 완료: 총 {success_count}/{total}개 프레임 처리\n📂 저장 경로: {out_path}"
            return True, msg, success_count, last_frame_rgb
            
        except Exception as e:
            err_msg = f"❌ 일괄 얼굴 교체 중 오류 발생: {str(e)}"
            self._logger.error(err_msg)
            return False, err_msg, success_count, last_frame_rgb

    def batch_face_swap_generator(
        self,
        input_dir: str,
        output_dir: str,
        face_indices: str,
        source_face_name: str,
        use_codeformer: bool,
        fidelity: float,
        face_manager: FaceManager,
        file_manager: FileManager,
        swap_model: Optional[str] = None,
    ):
        """
        지정된 디렉터리의 프레임 이미지들에 대해 일괄 얼굴 교체 및 복원을 수행하면서
        각 프레임마다 실시간 진행 상황을 yield하는 제너레이터 메서드입니다.
        """
        frame_files = self._get_sorted_frame_files(input_dir)
        if not frame_files:
            yield 0, 0, None, True, f"❌ 입력 디렉터리에 프레임 이미지 파일이 없습니다: {input_dir}"
            return
        
        in_path = Path(input_dir).resolve()
        out_path = Path(output_dir).resolve()
        
        if in_path == out_path:
            yield 0, 0, None, True, "❌ 입력 디렉터리와 출력 디렉터리는 서로 달라야 합니다."
            return
        
        is_no_face = not source_face_name or source_face_name == "선택 안함"
        if is_no_face and not use_codeformer:
            yield 0, 0, None, True, "❌ 바꿀 얼굴을 선택하거나 CodeFormer 복원을 체크해주세요."
            return
        
        out_path.mkdir(parents=True, exist_ok=True)
        total = len(frame_files)
        
        try:
            for idx, frame_file in enumerate(frame_files):
                current = idx + 1
                
                # 이미지 읽기 (PIL -> RGB)
                pil_img = Image.open(frame_file).convert("RGB")
                img_rgb = np.array(pil_img)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                
                final_rgb = img_rgb
                
                # 1. 얼굴 교체 수행
                if not is_no_face:
                    if swap_model and swap_model.startswith("hyperswap"):
                        swap_ok, swap_msg, swapped_rgb = face_manager.swap_faces_hyperswap(
                            img_bgr, face_indices, source_face_name, file_manager.faces_dir,
                            model_name=swap_model
                        )
                    else:
                        swap_ok, swap_msg, swapped_rgb = face_manager.swap_faces(
                            img_bgr, face_indices, source_face_name, file_manager.faces_dir
                        )
                    if swap_ok and swapped_rgb is not None:
                        final_rgb = swapped_rgb
                
                # 2. CodeFormer 복원 수행
                if use_codeformer:
                    current_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                    cf_ok, cf_msg, enhanced_rgb = face_manager.enhance_faces_with_codeformer(
                        current_bgr, face_indices, fidelity
                    )
                    if cf_ok and enhanced_rgb is not None:
                        final_rgb = enhanced_rgb
                
                # WebP 프레임 저장
                dest_file = out_path / f"{frame_file.stem}.webp"
                Image.fromarray(final_rgb).save(dest_file, format="WEBP")
                
                is_finished = (current == total)
                step_msg = f"프레임 {current}/{total} 저장 완료: {dest_file.name}"
                yield current, total, final_rgb, is_finished, step_msg
                
        except Exception as e:
            err_msg = f"❌ 일괄 얼굴 교체 중 오류 발생: {str(e)}"
            self._logger.error(err_msg)
            yield 0, total, None, True, err_msg

    def combine_frames_to_webp(self, input_dir: str, output_path: str, fps: float = 25.0, loop: int = 0) -> Tuple[bool, str, Optional[str]]:
        """
        디렉터리의 프레임 이미지들을 하나의 애니메이션 WebP 비디오로 결합합니다.
        
        Args:
            input_dir: 프레임 이미지 디렉터리 경로
            output_path: 출력할 애니메이션 WebP 파일 경로
            fps: 결합 속도 (FPS)
            loop: 루프 반복 횟수 (0: 무한 반복)
            
        Returns:
            (성공 여부, 메시지, 생성된 WebP 파일 경로)
        """
        frame_files = self._get_sorted_frame_files(input_dir)
        if not frame_files:
            return False, f"❌ 결합할 프레임 이미지가 없습니다: {input_dir}", None
        
        if fps <= 0:
            fps = 25.0
        
        duration_ms = int(1000.0 / fps)
        if duration_ms < 1:
            duration_ms = 1
        
        try:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            
            pil_frames = [Image.open(f).convert("RGB") for f in frame_files]
            
            # 애니메이션 WebP 저장
            pil_frames[0].save(
                out_file,
                format="WEBP",
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration_ms,
                loop=loop
            )
            
            msg = f"✅ 비디오 WebP 결합 성공!\n📂 생성 파일: {out_file.resolve()}\n🎞️ 프레임 수: {len(pil_frames)}개 ({fps} FPS)"
            self._logger.info(f"비디오 WebP 결합 완료: {out_file} ({len(pil_frames)} frames)")
            return True, msg, str(out_file.resolve())
            
        except Exception as e:
            err_msg = f"❌ 비디오 WebP 결합 중 오류 발생: {str(e)}"
            self._logger.error(err_msg)
            return False, err_msg, None
