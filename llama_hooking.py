from llama_cpp import Llama
import numpy as np

# 1. 모델 로드 (Gemma 3 1B 또는 Qwen 1.5B GGUF)
# embedding=True 설정이 핵심입니다! 이 설정을 켜야 중간 값을 뽑을 수 있습니다.
llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
    n_ctx=2048,
    embedding=True,
    n_threads=4,
    verbose=False
)

def get_layer_activations(text):
    tokens = llm.tokenize(text.encode('utf-8'))
    llm.eval(tokens)
    
    # 은닉 상태(Hidden States) 추출
    embeddings = llm.create_embedding(text)
    
    return np.array(embeddings['data'][0]['embedding'])

# 테스트
activation_go = get_layer_activations("체스")
print(f"추출된 벡터 크기: {activation_go.shape}")
print(f"앞부분 5개 값: {activation_go[:5]}")
