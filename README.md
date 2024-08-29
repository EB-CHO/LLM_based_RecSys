# LLM based Recommender System
LLM 기반 Recommender System 구현을 위한 Repository 입니다.

## Data Curation
- [x] Restaurant type (식당/카페/술집) 별 mood labling.
- [x] LLM instruction tuning 용 데이터 준비. 
- [x] RS 모델 예측 기반 Vector Search 용 DB 준비.

## Model Training
- `LLaMA-3-Korean-Bllossom-8B` ([Link](https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B)) 
- [x] GCN 기반 RS 모델 학습
- [x] LLM instruction tuning 코드 구현.

## Inference UI
- [x] LLM 기반 추천시스템 UI 구현
- [x] GCN 기반 모델 추천 검증 코드 구현 

## Command

### training
```
accelerate launch trainer_token.py
```

### inference
```
python inference.py
```


## Supports
- NVIDIA A100(80GB)-8U
- Supported by [LIM Lab](http://sungbin-lim.net) at the Department of Statistics in Korea University.
