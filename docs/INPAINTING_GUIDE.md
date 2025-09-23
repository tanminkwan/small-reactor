# 🎨 AI Inpainting 사용 가이드

## 📖 개요

AI Inpainting은 Stable Diffusion 모델을 사용하여 이미지의 특정 영역을 AI가 새롭게 생성하는 기능입니다. 마스킹 도구로 수정할 영역을 지정하고, 프롬프트로 원하는 내용을 설명하면 해당 영역을 고품질로 재생성합니다.

## 🚀 주요 특징

### ✨ 핵심 기능
- **🎯 정밀 마스킹**: 직관적인 마스킹 도구로 수정 영역 지정
- **🤖 AI 생성**: Stable Diffusion 기반 고품질 이미지 생성
- **📝 프롬프트 제어**: Positive/Negative 프롬프트로 생성 내용 제어
- **🔧 화질 보존**: 마스크 이외 영역 원본 화질 100% 보존
- **⚙️ 세밀 조정**: 다양한 파라미터로 생성 품질 최적화

### 🎛️ 고급 기능
- **📋 프롬프트 템플릿**: 미리 저장된 프롬프트로 빠른 설정
- **🔄 편집 모드**: 결과 이미지를 다시 편집하기 위한 원클릭 이동
- **🗑️ 결과 관리**: 생성된 파일 삭제 및 화면 초기화
- **📊 실시간 미리보기**: 마스크 생성 과정 실시간 확인

## 🎮 사용 방법

### 1. 이미지 업로드 및 마스킹

#### 이미지 업로드
1. **드래그 앤 드롭**: 이미지를 업로드 영역으로 드래그
2. **파일 선택**: 클릭하여 파일 탐색기에서 선택
3. **지원 형식**: JPG, PNG, WEBP 등 주요 이미지 형식

#### 마스킹 작업
1. **브러시 도구**: 마우스로 수정할 영역을 칠하기
2. **브러시 크기**: 세밀한 작업을 위한 브러시 크기 조정
3. **지우개**: 실수한 마스킹 영역 제거
4. **실행 취소**: Ctrl+Z로 이전 작업 되돌리기

### 2. 프롬프트 설정

#### Positive 프롬프트
```
# 얼굴 수정 예시
"beautiful young woman, natural skin, soft lighting, photorealistic"

# 객체 교체 예시
"red sports car, shiny paint, modern design, high quality"

# 배경 변경 예시
"beautiful sunset sky, golden hour, dramatic clouds, cinematic"
```

#### Negative 프롬프트
```
# 일반적인 negative 프롬프트
"blurry, distorted, low quality, artifacts, deformed, cartoon"

# 얼굴 관련 negative 프롬프트
"ugly, wrinkled, aged, scars, makeup, artificial"
```

#### 프롬프트 템플릿 활용
1. **템플릿 선택**: 드롭다운에서 미리 저장된 템플릿 선택
2. **자동 입력**: 선택한 템플릿이 프롬프트 입력창에 자동 입력
3. **커스터마이징**: 필요에 따라 프롬프트 수정 및 보완

### 3. 생성 파라미터 조정

#### 🎯 Guidance Scale (1.0-20.0)
- **낮은 값 (1.0-7.0)**: 자유로운 생성, 창의적 결과
- **중간 값 (7.0-12.0)**: 균형잡힌 생성 (권장)
- **높은 값 (12.0-20.0)**: 프롬프트에 매우 충실, 과도하면 아티팩트 발생

#### ⚡ Inference Steps (10-150)
- **낮은 값 (10-30)**: 빠른 생성, 품질 저하 가능
- **중간 값 (30-80)**: 균형잡힌 품질과 속도 (권장)
- **높은 값 (80-150)**: 최고 품질, 긴 생성 시간

#### 💪 Strength (0.1-1.0)
- **0.1-0.3**: 원본 유지하며 살짝 변경
- **0.4-0.7**: 적당한 변경
- **0.8-1.0**: 마스크 영역 완전히 새로 생성 (권장)

#### 🌟 Mask Blur (0-20)
- **0-2**: 선명한 경계
- **4-8**: 자연스러운 경계 (권장)
- **10-20**: 매우 부드러운 경계

#### 📏 Mask Dilation (0-50)
- **0**: 마스크 영역 그대로
- **5-15**: 약간 확장 (권장)
- **20-50**: 큰 영역 확장

### 4. 이미지 생성 및 결과 관리

#### 생성 과정
1. **마스크 미리보기**: "마스크 보기" 버튼으로 마스크 확인
2. **이미지 생성**: "🎨 이미지 생성" 버튼 클릭
3. **진행 상황**: 생성 과정 실시간 모니터링
4. **결과 표시**: 완성된 이미지가 "최종결과" 영역에 표시

#### 결과 관리
- **🗑️ 결과이미지 삭제**: 생성된 파일을 완전히 삭제
- **📝 편집모드로 이동**: 결과 이미지를 입력 영역으로 이동하여 추가 편집

## 🎯 실전 활용 예시

### 얼굴 보정
```
Positive: "beautiful face, natural skin, soft lighting, photorealistic"
Negative: "blurry, distorted, makeup, artificial"
Guidance Scale: 9.0
Strength: 0.8
```

### 객체 교체
```
Positive: "modern smartphone, sleek design, metallic finish"
Negative: "old, damaged, low quality, blurry"
Guidance Scale: 10.0
Strength: 1.0
```

### 배경 변경
```
Positive: "beautiful nature background, mountains, clear sky"
Negative: "urban, buildings, pollution, artificial"
Guidance Scale: 8.0
Strength: 0.9
```

### 의상 변경
```
Positive: "elegant dress, formal wear, high fashion"
Negative: "casual, wrinkled, old, damaged"
Guidance Scale: 11.0
Strength: 0.7
```

## ⚙️ 고급 설정

### 환경 변수 설정
```env
# .env 파일에서 설정
INPAINT_MODEL_PATH=TheImposterImposters/URPM-SD1.5-v2.3.inpainting
INPAINT_OUTPUT_PATH=C:\path\to\output\inpaint
```

### 프롬프트 템플릿 관리
프롬프트 템플릿은 `./prompts/` 디렉토리에 JSON 파일로 저장됩니다:

```json
{
  "face_enhancement": {
    "positive": "beautiful face, natural skin, soft lighting, photorealistic",
    "negative": "blurry, distorted, makeup, artificial"
  },
  "object_replacement": {
    "positive": "high quality object, modern design, detailed",
    "negative": "old, damaged, low quality, blurry"
  }
}
```

## 🔧 기술적 세부사항

### 화질 보존 기술
- **마스크 기반 블렌딩**: 마스크 영역만 AI 생성, 나머지는 원본 유지
- **VAE 손실 방지**: 원본 픽셀 직접 보존으로 화질 저하 방지
- **경계 부드럽게**: 마스크 블러로 자연스러운 경계 처리

### 성능 최적화
- **파이프라인 재사용**: 모델을 한 번만 로딩하여 메모리 효율성 향상
- **이미지 패딩**: 64픽셀 배수로 패딩하여 최적 성능 보장
- **자동 스텝 보정**: Strength 값에 따른 실제 수행 스텝 수 자동 계산

### 메모리 관리
- **지연 로딩**: 첫 사용 시에만 모델 로딩
- **GPU 메모리**: 8GB VRAM 권장 (최소 6GB)
- **시스템 메모리**: 16GB RAM 권장

## 🐛 문제 해결

### 일반적인 문제

#### 1. 모델 로딩 실패
```bash
# HuggingFace 로그인 (필요한 경우)
huggingface-cli login

# 모델 수동 다운로드
python -c "from diffusers import StableDiffusionInpaintPipeline; StableDiffusionInpaintPipeline.from_pretrained('TheImposterImposters/URPM-SD1.5-v2.3.inpainting')"
```

#### 2. 메모리 부족
- GPU VRAM이 부족한 경우 이미지 크기를 줄여보세요
- Inference Steps를 낮춰 메모리 사용량을 줄여보세요
- 다른 GPU 사용 프로그램을 종료해보세요

#### 3. 품질 문제
- **아티팩트 발생**: Guidance Scale을 낮춰보세요 (7.0-10.0)
- **프롬프트 무시**: Guidance Scale을 높여보세요 (10.0-15.0)
- **경계 부자연스러움**: Mask Blur를 조정해보세요 (4-8)
- **변화 부족**: Strength를 높여보세요 (0.8-1.0)

#### 4. 생성 속도 문제
- **너무 느림**: Inference Steps를 줄여보세요 (30-50)
- **품질 저하**: Steps를 적절히 유지하면서 이미지 크기 조정

### 최적 설정 가이드

#### 일반적인 용도
```
Guidance Scale: 9.0
Inference Steps: 50
Strength: 0.8
Mask Blur: 4
Mask Dilation: 0
```

#### 고품질 생성
```
Guidance Scale: 10.0
Inference Steps: 80
Strength: 1.0
Mask Blur: 6
Mask Dilation: 5
```

#### 빠른 생성
```
Guidance Scale: 8.0
Inference Steps: 30
Strength: 0.7
Mask Blur: 4
Mask Dilation: 0
```

## 📚 추가 자료

### 프롬프트 작성 팁
1. **구체적 묘사**: 추상적 표현보다 구체적 묘사 사용
2. **품질 키워드**: "high quality", "detailed", "photorealistic" 등 추가
3. **스타일 지정**: "cinematic", "portrait", "landscape" 등 스타일 명시
4. **조명 설정**: "soft lighting", "natural light", "dramatic lighting" 등

### 마스킹 기법
1. **정확한 경계**: 수정할 영역만 정확히 마스킹
2. **적절한 여백**: 너무 타이트하지 않게 약간의 여백 포함
3. **복잡한 형태**: 복잡한 객체는 여러 번에 나누어 마스킹
4. **경계 처리**: 자연스러운 경계를 위해 Mask Blur 활용

### 성능 최적화 팁
1. **적절한 해상도**: 너무 큰 이미지는 메모리 부족 원인
2. **배치 처리**: 여러 이미지 처리 시 순차적으로 처리
3. **GPU 모니터링**: nvidia-smi로 GPU 사용량 모니터링
4. **메모리 정리**: 장시간 사용 시 주기적으로 애플리케이션 재시작

---

**💡 팁**: 최상의 결과를 위해 다양한 파라미터 조합을 실험해보세요. 각 이미지와 용도에 따라 최적의 설정이 다를 수 있습니다.
