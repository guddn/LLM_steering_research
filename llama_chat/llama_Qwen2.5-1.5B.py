from llama_cpp import Llama

def create_prompt(user_input):
    return f'''
    당신은 계산을 정확하게 수행하는 계산기입니다.
    주어진 계산 문제에 대해 정확한 답을 제공해주세요.
    계산 문제에 대한 풀이과정을 항상 포함합니다.
    계산 문제는 다음과 같습니다: {user_input}
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