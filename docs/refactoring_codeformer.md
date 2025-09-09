코드를 분석해보겠습니다. 먼저 `codeformer_enhancer`와 `FaceManager` 내부의 `FaceRestore` 클래스가 어떻게 구성되어 있는지 확인해보겠습니다.


[3 tools called]

이제 `gradio_face_manager.py`에서 `FaceRestore` 클래스를 다시 확인해보겠습니다.


[2 tools called]

이제 두 클래스의 차이점과 각각이 필요한 이유를 분석해보겠습니다.

## 분석 결과

### 1. `CodeFormerEnhancer` (src/services/codeformer_enhancer.py)

**역할과 특징:**
- **독립적인 서비스 클래스**: SOLID 원칙의 SRP(단일 책임 원칙)를 준수
- **인터페이스 구현**: `IImageEnhancer` 인터페이스를 구현하여 확장성 보장
- **완전한 기능**: 모델 초기화, 입력 검증, 에러 처리, 리소스 관리 등 모든 기능 포함
- **재사용 가능**: 다른 모듈에서도 사용할 수 있는 독립적인 서비스
- **테스트 가능**: 단위 테스트 작성이 용이한 구조

**주요 기능:**
- 모델 초기화 및 관리
- 입력 검증 (`_validate_input`)
- 전체 이미지 복원 (`enhance_image`)
- 특정 영역 복원 (`enhance_face_regions`)
- 영역 확장 (`_expand_face_regions`)
- 모델 정보 제공 (`get_model_info`)

### 2. `FaceRestorer` (gradio_face_manager.py 내부)

**역할과 특징:**
- **임시 내부 클래스**: `enhance_faces_with_codeformer` 메서드 내부에서만 사용
- **간단한 래퍼**: CodeFormer의 기본 기능만을 간단히 래핑
- **특정 용도**: Gradio UI에서 얼굴 교체 후 복원하는 특정 시나리오에만 사용
- **최소한의 기능**: 모델 로드와 기본 복원 기능만 제공

**주요 기능:**
- 모델 초기화 (간단 버전)
- 기본 복원 (`restore` 메서드)

### 3. 왜 둘 다 필요한가?

#### **아키텍처 관점에서의 필요성:**

1. **계층 분리 (Layered Architecture)**
   - `CodeFormerEnhancer`: **서비스 계층** - 비즈니스 로직과 도메인 서비스
   - `FaceRestorer`: **프레젠테이션 계층** - UI 특화된 임시 구현

2. **의존성 관리**
   - `CodeFormerEnhancer`: 다른 서비스들과의 의존성을 명확히 관리
   - `FaceRestorer`: UI 코드에 직접 임베드되어 의존성 최소화

3. **개발 단계별 접근**
   - `CodeFormerEnhancer`: **장기적 솔루션** - 확장 가능하고 유지보수 가능
   - `FaceRestorer`: **단기적 솔루션** - 빠른 프로토타이핑과 테스트용

#### **실제 사용 시나리오:**

1. **CodeFormerEnhancer 사용 시:**
   ```python
   # 다른 서비스에서 재사용 가능
   enhancer = CodeFormerEnhancer(config)
   enhanced_image = enhancer.enhance_image(image)
   ```

2. **FaceRestorer 사용 시:**
   ```python
   # UI 특화된 특정 시나리오에서만 사용
   restorer = FaceRestorer(model_path, use_gpu=True)
   restored_image = restorer.restore(face_crop)
   ```

### 4. 개선 방향

현재 구조는 **중복 코드**가 존재하므로, 다음과 같은 개선이 필요합니다:

1. **통합**: `FaceRestorer`를 제거하고 `CodeFormerEnhancer`를 사용
2. **의존성 주입**: `FaceManager`에 `CodeFormerEnhancer`를 주입
3. **인터페이스 활용**: `IImageEnhancer` 인터페이스를 통한 느슨한 결합

이렇게 하면 SOLID 원칙을 더 잘 준수하고, 코드 중복을 제거하며, 유지보수성을 향상시킬 수 있습니다.