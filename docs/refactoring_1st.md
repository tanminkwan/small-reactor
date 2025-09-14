# Gradio Face Manager 리팩토링 계획

## 현재 상황 분석

`gradio_face_manager.py` 파일이 1,185줄로 너무 방대해졌습니다. 다음과 같은 문제점들이 있습니다:

### 문제점
1. **단일 책임 원칙 위반**: 하나의 파일에 UI, 비즈니스 로직, 데이터 처리 등 모든 것이 섞여있음
2. **가독성 저하**: 코드가 길어져서 이해하기 어려움
3. **유지보수성 저하**: 수정 시 전체 파일을 확인해야 함
4. **테스트 어려움**: UI와 로직이 결합되어 단위 테스트 작성이 어려움
5. **재사용성 저하**: UI에 종속된 로직으로 다른 환경에서 사용하기 어려움

## 분리 계획

### 1. 핵심 비즈니스 로직 분리

#### 1.1 FaceManager 클래스 개선
- **현재**: UI와 비즈니스 로직이 혼재
- **개선**: 순수한 비즈니스 로직만 포함하도록 분리

```python
# src/services/face_manager.py
class FaceManager:
    """얼굴 관리 핵심 비즈니스 로직"""
    
    def __init__(self, config: Config):
        self.config = config
        self.detector = BuffaloDetector(config)
        self.enhancer = CodeFormerEnhancer(config)
        # ... 기타 초기화
    
    # UI와 무관한 순수 비즈니스 로직만 포함
    def extract_first_face(self, image_bgr: np.ndarray, filename: str) -> Tuple[bool, str, str]:
        # ... 구현
    
    def swap_faces(self, target_image: np.ndarray, face_indices: str, source_face_name: str) -> Tuple[bool, str, np.ndarray]:
        # ... 구현
    
    def enhance_faces_with_codeformer(self, image: np.ndarray, face_indices: str = "") -> Tuple[bool, str, np.ndarray]:
        # ... 구현
```

#### 1.2 파일 관리 로직 분리
```python
# src/services/file_manager.py
class FileManager:
    """파일 관리 전용 클래스"""
    
    def __init__(self, faces_path: str, output_path: str):
        self.faces_path = faces_path
        self.output_path = output_path
        # ... 초기화
    
    def get_safe_filename(self, filename: str) -> str:
        # ... 구현
    
    def get_embedding_list(self) -> List[dict]:
        # ... 구현
    
    def save_result_image(self, image: np.ndarray, prefix: str = "final_result") -> str:
        # ... 구현
```

### 2. UI 컴포넌트 분리

#### 2.1 Gradio 인터페이스 컴포넌트 분리
```python
# src/ui/components/face_extraction_tab.py
class FaceExtractionTab:
    """얼굴 추출 탭 컴포넌트"""
    
    def __init__(self, face_manager: FaceManager):
        self.face_manager = face_manager
    
    def create_interface(self) -> gr.Tab:
        # ... 구현
    
    def process_uploaded_image(self, file_path) -> Tuple[bool, str, np.ndarray]:
        # ... 구현

# src/ui/components/face_swap_tab.py
class FaceSwapTab:
    """얼굴 교체 탭 컴포넌트"""
    
    def __init__(self, face_manager: FaceManager, file_manager: FileManager):
        self.face_manager = face_manager
        self.file_manager = file_manager
    
    def create_interface(self) -> gr.Tab:
        # ... 구현
    
    def perform_face_swap_with_optional_codeformer(self, file_path, face_indices, source_face_name, use_codeformer, preserve_mouth=False, mouth_settings=None):
        # ... 구현

# src/ui/components/embedding_list_tab.py
class EmbeddingListTab:
    """Embedding 목록 탭 컴포넌트"""
    
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
    
    def create_interface(self) -> gr.Tab:
        # ... 구현
```

#### 2.2 공통 UI 유틸리티 분리
```python
# src/ui/utils/ui_helpers.py
class UIHelpers:
    """UI 공통 유틸리티"""
    
    @staticmethod
    def get_embedding_choices(faces_dir: Path) -> List[str]:
        # ... 구현
    
    @staticmethod
    def get_embedding_gallery_data(faces_dir: Path) -> List[Tuple[str, str]]:
        # ... 구현
    
    @staticmethod
    def refresh_face_choices(faces_dir: Path) -> gr.Dropdown:
        # ... 구현
```

### 3. 이벤트 핸들러 분리

#### 3.1 이벤트 핸들러 클래스
```python
# src/ui/handlers/event_handlers.py
class EventHandlers:
    """이벤트 핸들러 모음"""
    
    def __init__(self, face_manager: FaceManager, file_manager: FileManager):
        self.face_manager = face_manager
        self.file_manager = file_manager
    
    def setup_face_extraction_handlers(self, tab: FaceExtractionTab):
        # ... 구현
    
    def setup_face_swap_handlers(self, tab: FaceSwapTab):
        # ... 구현
    
    def setup_embedding_list_handlers(self, tab: EmbeddingListTab):
        # ... 구현
```

### 4. 설정 및 의존성 주입

#### 4.1 의존성 주입 컨테이너
```python
# src/core/container.py
class DIContainer:
    """의존성 주입 컨테이너"""
    
    def __init__(self):
        self.config = Config()
        self.face_manager = None
        self.file_manager = None
        self._initialize_services()
    
    def _initialize_services(self):
        """서비스 초기화"""
        self.face_manager = FaceManager(self.config)
        self.file_manager = FileManager(
            faces_path=os.getenv("FACES_PATH", "./faces"),
            output_path=os.getenv("OUTPUT_PATH", "./outputs")
        )
    
    def get_face_manager(self) -> FaceManager:
        return self.face_manager
    
    def get_file_manager(self) -> FileManager:
        return self.file_manager
```

### 5. 메인 애플리케이션 구조

#### 5.1 메인 애플리케이션 클래스
```python
# src/ui/app.py
class FaceManagerApp:
    """Face Manager 메인 애플리케이션"""
    
    def __init__(self, container: DIContainer):
        self.container = container
        self.face_manager = container.get_face_manager()
        self.file_manager = container.get_file_manager()
        
        # 탭 컴포넌트들
        self.face_extraction_tab = FaceExtractionTab(self.face_manager)
        self.face_swap_tab = FaceSwapTab(self.face_manager, self.file_manager)
        self.embedding_list_tab = EmbeddingListTab(self.file_manager)
        
        # 이벤트 핸들러
        self.event_handlers = EventHandlers(self.face_manager, self.file_manager)
    
    def create_interface(self) -> gr.Blocks:
        """전체 인터페이스 생성"""
        with gr.Blocks(title="Face Manager", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🎭 Face Manager")
            gr.Markdown("이미지에서 얼굴을 추출하고 embedding을 관리합니다.")
            
            with gr.Tab("얼굴 교체"):
                self.face_swap_tab.create_interface()
            
            with gr.Tab("얼굴 추출"):
                self.face_extraction_tab.create_interface()
            
            with gr.Tab("Embedding 목록"):
                self.embedding_list_tab.create_interface()
            
            # 이벤트 핸들러 설정
            self.event_handlers.setup_all_handlers()
        
        return interface
    
    def run(self):
        """애플리케이션 실행"""
        interface = self.create_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True
        )
```

#### 5.2 메인 진입점
```python
# main.py
#!/usr/bin/env python3
"""
Face Manager 메인 진입점
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
    logger.info("Face Manager UI 시작")
    
    # 의존성 주입 컨테이너 생성
    container = DIContainer()
    
    # 애플리케이션 생성 및 실행
    app = FaceManagerApp(container)
    app.run()

if __name__ == "__main__":
    main()
```

## 파일 구조

```
src/
├── core/
│   └── container.py              # 의존성 주입 컨테이너
├── services/
│   ├── face_manager.py           # 핵심 비즈니스 로직
│   └── file_manager.py           # 파일 관리 로직
├── ui/
│   ├── app.py                    # 메인 애플리케이션
│   ├── components/
│   │   ├── face_extraction_tab.py
│   │   ├── face_swap_tab.py
│   │   └── embedding_list_tab.py
│   ├── handlers/
│   │   └── event_handlers.py
│   └── utils/
│       └── ui_helpers.py
└── utils/
    ├── config.py
    └── mouth_mask.py

main.py                           # 메인 진입점
```

## 리팩토링 단계

### 1단계: 핵심 비즈니스 로직 분리
- [ ] `FaceManager` 클래스를 `src/services/face_manager.py`로 분리
- [ ] UI 관련 코드 제거하고 순수 비즈니스 로직만 유지
- [ ] 파일 관리 로직을 `src/services/file_manager.py`로 분리

### 2단계: UI 컴포넌트 분리
- [ ] 각 탭을 독립적인 컴포넌트 클래스로 분리
- [ ] 공통 UI 유틸리티 함수들을 별도 모듈로 분리
- [ ] 이벤트 핸들러를 별도 클래스로 분리

### 3단계: 의존성 주입 및 메인 애플리케이션 구조화
- [ ] 의존성 주입 컨테이너 구현
- [ ] 메인 애플리케이션 클래스 구현
- [ ] 메인 진입점 분리

### 4단계: 테스트 및 검증
- [ ] 각 분리된 컴포넌트에 대한 단위 테스트 작성
- [ ] 통합 테스트 작성
- [ ] 기능 검증

## 예상 효과

1. **가독성 향상**: 각 파일이 명확한 책임을 가짐
2. **유지보수성 향상**: 수정 시 해당 컴포넌트만 확인하면 됨
3. **테스트 용이성**: 각 컴포넌트를 독립적으로 테스트 가능
4. **재사용성 향상**: 비즈니스 로직을 다른 UI에서도 사용 가능
5. **확장성 향상**: 새로운 기능 추가 시 해당 컴포넌트만 수정하면 됨

## 주의사항

1. **점진적 리팩토링**: 한 번에 모든 것을 바꾸지 말고 단계적으로 진행
2. **기능 보존**: 리팩토링 과정에서 기존 기능이 손실되지 않도록 주의
3. **테스트 우선**: 각 단계마다 기능이 정상 작동하는지 확인
4. **문서화**: 각 컴포넌트의 역할과 인터페이스를 명확히 문서화

