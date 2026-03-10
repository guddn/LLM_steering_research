from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('BAAI/bge-m3')

sentences = [
    "인공지능 바둑 프로그램인 알파고는 복잡한 계산을 수행합니다.",
    "딥마인드가 개발한 AI는 바둑판 위에서 전략적인 수를 둡니다.",
    "어제 점심에 먹은 김치찌개는 정말 매콤하고 맛있었습니다."
]

# 문장을 변환
print("변환 중...")
embeddings = model.encode(sentences, normalize_embeddings=True)

# 유사도 계산
sim_1_2 = util.cos_sim(embeddings[0], embeddings[1])
sim_1_3 = util.cos_sim(embeddings[0], embeddings[2])

print(f"문장 1: {sentences[0]}")
print(f"문장 2: {sentences[1]}")
print(f"문장 3: {sentences[2]}")
print("-" * 30)
print(f"문장 1 vs 2 유사도: {sim_1_2.item():.4f} (바둑 관련 - 높음 기대)")
print(f"문장 1 vs 3 유사도: {sim_1_3.item():.4f} (관련 없음 - 낮음 기대)")