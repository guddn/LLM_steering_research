from llama_cpp import Llama

def create_prompt(user_input):
    return f'''
    당신은 최고의 AI 어시스턴트입니다. 
    사용자의 질문에 대해 친절하고 정확하게 답변해 주세요.
    정확하지 않은 답변은 하지 않습니다.
    천천히 차례대로 생각해서 답변해 주세요.
    사용자의 질문: {user_input}
    '''

llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
    n_ctx=2048,
    n_threads=4,
    verbose=False
)

user_input = ""

while user_input != "q":
    user_input = input("궁금한 점이 있으신가요?: ")
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": create_prompt(user_input)
            }
        ]
    )

    print('답변: ' + response['choices'][0]['message']['content'])