# WebP 비디오 애니메이션 얼굴 교체 기능 분석 및 설계서

## 1. 개요 (Overview)

본 설계서는 기존 Face Manager 애플리케이션의 아키텍처와 개발 방식을 철저히 준수하면서, **WebP 애니메이션(비디오) 분해 -> 일괄 얼굴 교체 -> WebP 비디오 결합**의 3단계 독립 프로세스를 하나의 통합 화면(탭)으로 제공하기 위한 분석 및 설계 내용을 담고 있습니다.

기존 코드베이스의 가치와 안정성을 보장하기 위해 **단일 책임 원칙(SRP)**과 **의존성 역전 원칙(DIP)**을 유지하며, 기존 기능에 절대적인 영향을 주지 않도록 격리된 신규 서비스와 UI 컴포넌트로 구성됩니다.

---

## 2. 요구사항 분석 (Requirements Analysis)

### 2.1 주요 기능 요구사항
1. **1단계: WebP 영상 프레임 분해 (Frame Extraction)**
   - 특정 video webp 파일을 Drag & Drop 또는 파일 선택으로 업로드.
   - 지정된 입력 프레임 디렉터리(기본값: `./outputs/webp_frames/extracted`)에 각 프레임 이미지(`.webp`)로 분해하여 저장.
   - 프레임 수 및 비디오 메타데이터(FPS, 프레임 지연 시간) 자동 측정 및 상태 로그 표시.

2. **2단계: 일괄 얼굴 교체 (Batch Face Swap)**
   - 기존 '얼굴 교체' 화면과 동일한 설정 옵션 제공:
     - `교체할 얼굴 인덱스` (예: "1,3,5" 또는 빈칸 시 전체)
     - `바꿀 얼굴 선택` (저장된 얼굴 embedding 목록 드롭다운 + 🔄 새로고침 버튼)
     - `CodeFormer 복원 포함` (체크박스, 기본값: True)
     - `Fidelity (복원 강도)` (슬라이더: 0.0 ~ 1.0, 기본값: 0.5)
   - 작업 디렉터리 분리:
     - 프레임 읽기 디렉터리 (기본값: 1단계 추출 디렉터리)
     - 프레임 저장 디렉터리 (기본값: `./outputs/webp_frames/swapped`, 1단계 디렉터리와 엄격히 분리)
   - 분해된 프레임 파일들을 일괄 순회하며 얼굴 교체 및 품질 복원 수행.
   - 진행 상황(Progress Bar) 및 처리 현황 실시간 업데이트.

3. **3단계: 비디오 WebP 결합 (Animated WebP Recomposition)**
   - 프레임 읽기 디렉터리 선택 (기본값: 2단계 얼굴 교체 결과 디렉터리).
   - 결합 속도(FPS) 및 루프 설정 지정.
   - 해당 디렉터리의 프레임 이미지들을 순서대로 연결하여 애니메이션 `.webp` 파일 생성 (기본값: `./outputs/animated_swapped.webp`).
   - 생성된 비디오 WebP 애니메이션 미리보기 및 결과 파일 저장 위치 확인/이동 기능 제공.

### 2.2 비기능 요구사항 & 제약사항
- **기존 코드 영향도 0%**: 기존 서비스/컴포넌트(`FaceSwapTab`, `FaceManager`, `FileManager` 등)의 시그니처 및 로직 변경 금지.
- **아키텍처 패턴 유지**: DIContainer를 이용한 서비스 등록, Component 기반 Gradio UI 구현, SRP/DIP 준수.
- **예외 처리 및 안전성**: 프레임 추출/일괄 변환/결합 중 발생하는 오류 발생 시 로그에 상세 출력하고 안전하게 중단/복구.

---

## 3. 시스템 아키텍처 설계 (System Architecture)

```
[UI Layer]
   └── AnimatedWebPTab (src/ui/components/animated_webp_tab.py)
            │
            ▼
[Service Layer]
   ├── WebPService (src/services/webp_service.py)  [신규]
   ├── FaceManager (src/services/face_manager.py)  [기존 재사용]
   └── FileManager (src/services/file_manager.py)  [기존 재사용]
            │
            ▼
[Core Container Layer]
   └── DIContainer (src/core/container.py)
            │
            ▼
[Application Orchestration]
   └── FaceManagerApp (src/ui/app.py)
```

---

## 4. 컴포넌트 상세 설계 (Detailed Component Design)

### 4.1 WebPService (`src/services/webp_service.py`) [신규]

WebP 비디오 처리(분해, 일괄 변환, 결합) 전용 서비스 클래스입니다.

#### 메서드 명세:
1. `extract_webp_frames(self, webp_path: str, output_dir: str) -> Tuple[bool, str, int, float]`
   - PIL `Image.open`을 사용하여 애니메이션 WebP 탐색.
   - 각 프레임을 `frame_0001.webp`, `frame_0002.webp` 형식으로 `output_dir`에 저장.
   - FPS 계산 및 추출된 프레임 수 반환.

2. `batch_face_swap(self, input_dir: str, output_dir: str, face_indices: str, source_face_name: str, use_codeformer: bool, fidelity: float, face_manager: FaceManager, file_manager: FileManager, progress_fn=None) -> Tuple[bool, str, int, Optional[np.ndarray]]`
   - `input_dir`의 이미지 파일 목록을 파일명 정렬 순으로 조회.
   - `output_dir` 생성 및 존재 확인.
   - 각 프레임별로 `FaceManager.swap_faces()` 및 `FaceManager.enhance_faces_with_codeformer()` 호출.
   - 변환 결과를 `output_dir`에 동명의 `.webp` 파일로 저장.
   - 샘플 마지막 프레임 numpy array 및 처리 결과 메시지 반환.

3. `combine_frames_to_webp(self, input_dir: str, output_path: str, fps: float, loop: int = 0) -> Tuple[bool, str, Optional[str]]`
   - `input_dir` 내 프레임 파일 정렬 및 읽기.
   - PIL `Image.save(..., save_all=True, append_images=..., duration=ms, loop=loop)`로 애니메이션 WebP 생성.
   - 결과 파일 경로 및 성공 여부 반환.

---

### 4.2 DIContainer 확장 (`src/core/container.py`)

- `self._webp_service` 필드 추가.
- `_initialize_services()`에서 `WebPService` 초기화.
- `get_webp_service(self) -> WebPService` 접근자 제공.

---

### 4.3 AnimatedWebPTab (`src/ui/components/animated_webp_tab.py`) [신규]

Gradio 기반의 3단계 독립 프로세스 UI 컴포넌트입니다.

#### UI 레이아웃 구조:
- **Title**: `🎬 비디오 WebP 얼굴 교체`
- **Step 1 Accordion / Group: 1️⃣ WebP 영상 프레임 분해**
  - WebP 파일 드롭존 (`gr.File`)
  - 저장 디렉터리 입력 (`gr.Textbox`, 기본값: `./outputs/webp_frames/extracted`)
  - 실행 버튼 (`gr.Button("🎬 프레임 이미지 분해 실행")`)
  - 결과 로그 및 추출 정보 (`gr.Textbox`)
- **Step 2 Accordion / Group: 2️⃣ 프레임 일괄 얼굴 변경**
  - 입력 프레임 디렉터리 (`gr.Textbox`, Step 1 디렉터리와 연동)
  - 출력 프레임 디렉터리 (`gr.Textbox`, 기본값: `./outputs/webp_frames/swapped`)
  - 교체할 얼굴 인덱스 (`gr.Textbox`)
  - 바꿀 얼굴 선택 (`gr.Dropdown` + `gr.Button("🔄")`)
  - CodeFormer 복원 포함 (`gr.Checkbox`)
  - Fidelity (복원 강도) (`gr.Slider`)
  - 실행 버튼 (`gr.Button("🔄 일괄 얼굴 변경 실행")`)
  - 진행 상태 (`gr.Progress`), 샘플 프레임 미리보기 (`gr.Image`)
- **Step 3 Accordion / Group: 3️⃣ 프레임 -> 비디오 WebP 결합**
  - 프레임 디렉터리 (`gr.Textbox`, Step 2 디렉터리와 연동)
  - 출력 파일 경로 (`gr.Textbox`, 기본값: `./outputs/animated_swapped.webp`)
  - FPS 설정 (`gr.Number`, 기본값: 25.0)
  - 루프 설정 (`gr.Number`, 기본값: 0)
  - 실행 버튼 (`gr.Button("📽️ 비디오 WebP 결합 실행")`)
  - 최종 애니메이션 WebP 미리보기 (`gr.Image`) 및 저장 위치 확인 버튼 (`gr.Button("📁 저장 위치 확인")`)

---

### 4.4 Main Application 통합 (`src/ui/app.py`)

- `self.animated_webp_tab = AnimatedWebPTab(self.webp_service, self.face_manager, self.file_manager)` 추가.
- `create_interface()` 내에 새로운 탭 추가 및 이벤트 핸들러 바인딩.

---

## 5. 기존 기능 및 코드 영향성 검증 (Impact Assessment)

| 기존 컴포넌트 / 모듈 | 변경 여부 | 영향도 분석 |
|---|---|---|
| `src/services/face_manager.py` | 없음 | 기존 메서드(`swap_faces`, `enhance_faces_with_codeformer`)를 신규 서비스에서 그대로 호출 |
| `src/services/file_manager.py` | 없음 | 기존 메서드(`get_embedding_choices` 등) 그대로 호출 |
| `src/ui/components/face_swap_tab.py` | 없음 | 코드 수정 없음 |
| `src/core/container.py` | 추가만 발생 | 신규 서비스 getter 및 초기화 로직만 추가되므로 기존 의존성 전이 영향 없음 |
| `src/ui/app.py` | 추가만 발생 | 신규 탭 생성 및 바인딩 코드만 추가됨 |

---

## 6. 검증 계획 (Verification Plan)

1. **단위 테스트 (Unit Tests)**:
   - `tests/test_webp_service.py` 작성
   - WebP 애니메이션 프레임 분해 테스트
   - 프레임 일괄 변환 파이프라인 (Mock FaceManager 사용) 테스트
   - 애니메이션 WebP 합성 및 재생 시간/FPS 검증 테스트

2. **통합 UI 수동 및 기능 검증**:
   - WebP 샘플 파일 업로드 후 Step 1 (분해) 실행 및 저장 디렉터리 확인.
   - Step 2 (일괄 얼굴 교체) 실행 후 변환된 프레임 이미지 확인.
   - Step 3 (WebP 결합) 실행 후 최종 재생되는 애니메이션 WebP 확인.
