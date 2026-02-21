# CCTV 객체 탐지 및 상황 설명 시스템

## 개요
본 시스템은 세 가지 데이터셋(CrowdHuman, BDD100K, Pest24)을 대상으로 YOLO 모델 성능을 검증하고, 탐지 결과를 기반으로 상황 설명을 생성합니다.

- **데이터 흐름**
  1. YOLO 탐지
  2. 프롬프트 생성
  3. 상황 분류/설명
- **모델 연결**
  - YOLOv5 → DistilBERT (상황 분류)
  - YOLO11 → Florence-2, Qwen2.5-VL (상황 설명)

## 환경
- Ubuntu, Python 3.11, CUDA 12.x, NVIDIA GPU 권장
- 모델별 Conda 가상환경 사용
  - `cctv_research`: YOLO, DistilBERT, Florence-2
  - `cctv_qwen`: Qwen2.5-VL

## 프로젝트 구조
cctv_package/
├─ datasets/ # 입력 데이터, 프롬프트
├─ outputs/ # 탐지, 프롬프트, 상황 설명 결과
├─ src/ # 학습, 추론, 프롬프트 생성, 설명 모델 스크립트
├─ weights/ # YOLO 모델 weight

## 실행 흐름
1. YOLO 객체 탐지 → detection 결과 생성  
2. 프롬프트 생성 → 상황 설명 모델 입력 준비  
3. 상황 분류 및 설명 생성 → `outputs` 폴더에 저장

## 주의 사항
- 가상환경을 모델별로 올바르게 활성화해야 함  
- Florence-2, Qwen2.5-VL은 GPU 권장, CPU 실행 가능하지만 느림  
- 기존 결과 파일 덮어쓰기 주의