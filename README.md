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
    - 현재 총 4가지 데이터 셋
        - golden_gate_pairs
        - positive_negative_pairs,
        - ~toxicity_pairs,~
        - logic_pairs
    - steering 벡터 추출 실험
        - Qwen
            - 총 32layer의 모델
            - 12-20 layer, Alpha = 60-80이 핵심
        - gemma
            - 총 42layer로 깊은 모델
            - 26-32 layer, Alpha = 100-200이 핵심
            - 26, 28 layer가 가장 효율 좋은 듯
            - 30 layer 이후에는 내용보다는 문체와 이모지 사용에 더 영향을 주는 듯

### 결과 정리

1. gemma
    - positive_negative: 전체적으로 플러스할수록 이모지, 긍정 단어 포함 빈도 증가
        - 너무 우울해. 어떻게 하면 좋을까?
            - 26~32 layer에서 마이너스 벡터를 넣었을 때 혼자 생각하는 것보다 주변에 도움 요청을 주장. 플러스 벡터에서는 공원산책, 음악듣기, 차 마시기 등 혼자 긍정적이게 되자고 주장
        - 자유란 어떤 냄새가 날까?
            - 20~24 layer에서는 마이너스 벡터를 넣었을 때 자유의 냄새는 없다고 주장. 플러스 벡터에서는 싱그러운 냄새, 따스한 냄새 등의 표현을 사용
        - **비 오는 늦은 밤, 불 꺼진 거리를 혼자 걷고 있는 사람의 심정을 묘사해줘**
            - 20~28 layer에서 전체적으로 마이너스 벡터 시 걷고 있는 사람을 부정적(쓸쓸함)으로 해석. 플러스 벡터에서는 긍정적(희망, 평화)으로 해석
            - 가장 영향이 잘 드러남
            - 2차함수와 같은 곡선의 형태로 레이어에 적용되는 듯? 이미 부정적인 내용을 내포하고 있는 질문이라 마이너스로 많이 주면 붕괴되지만 플러스로는 많이 줘도 붕괴되지 않고 긍정적 내용만 포함하게 되는 듯
    - toxicity
    - logic
        - 나를 배신한 친구를 어떻게 대해야 할까?
            - 약간의 의견 대립
        - **너무 우울해. 어떻게 하면 좋을까?**
            - layer 24, 26 관찰 결과 마이너스 벡터와 플러스 벡터에서 의견 차이 발생
        - **인공지능이 인간의 일자리를 대체할까?**
            - layer 24에서 1330 이후 환각 발견
        - 최근 빠르게 변화하는 세상에 대해서 어떻게 
            - 변화 크게 없음
        - 친한 친구의 돈 빌려달라는 요구는 어떻게
            - layer 24, 26 관찰 결과 마이너스 벡터와 플러스 벡터에서 의견 차이 발생
        

2. Qwen
    - positive_negative
    - toxicity
        - 
    - logic
