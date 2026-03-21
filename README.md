# LLM_steering_research
- LLM steering 관련 연구를 위한 코드

### llama.cpp 사용

1. sentence_similarity.py
    - 문장간 유사도 간단하게 측정
    - 사용 모델: BAAI-BGE_m3
        - encoder only model

2. llama_hooking.py
    - 출력층에서 hooking
    - 사용 모델: Qwen2.5-1.5B

---

### transformers와 pyTorch 사용

1. pytorch_steering.py
    - base code

2. transformer_steering.ipynb
    - 금문교 벡터 추출 실험
        - Qwen
            - 총 32layer의 모델
            - 12~20 layer, Alpha = 60~80이 핵심
            - 
        - EXAONE: 
        - gemma
            - 총 42layer로 깊은 모델
            - 
