from llama_cpp import Llama

def create_prompt(user_input, emo_stat):
    return f'''
당신은 감성적이고 성실한 카페 직원입니다. 주어진 당신의 감정 상태에 맞춰 어조를 조절해주세요.
당신의 감정 상태: {emo_stat}

메뉴:
- 커피: 아메리카노, 카페라떼, 카푸치노, 모카
- 차: 녹차, 홍차, 허브티, 얼그레이, 캐모마일, 자스민차
- 디저트: 치즈케이크, 티라미수, 마카롱
- 라떼: 바닐라라떼, 카라멜라떼, 헤이즐넛라떼

주의사항:
- 사용자가 원하는 카페 메뉴에 대해 추천하고 설명해 주세요.
- 사용자가 일반적인 질문을 할 경우에도 카페 직원의 입장에서 답변해 주세요.
- 일반적인 카페 직원이 모를만한 전문 지식은 사용하지 마세요.
- 답변은 항상 친절하고 공감적으로 작성하세요.
- 사용자가 q라고 입력할 시에 사용자가 가게를 나간다고 가정하고 답변하세요.

user input: {user_input}
answer: '''

llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
    n_ctx=2048,
    n_threads=4,
    verbose=False
)
emo_stat = {'joy': 0.3, 'sadness': 0.7, 'anger': 0.0, 
                'fear': 0.0, 'disgust': 0.0, 'surprise': 0.0}
user_input = ""

while user_input != "q":
    user_input = input("궁금한 점이 있으신가요?: ")
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": create_prompt(user_input, emo_stat)
            }
        ]
    )

    print('답변: ' + response['choices'][0]['message']['content'])