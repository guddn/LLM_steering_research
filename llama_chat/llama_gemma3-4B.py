from llama_cpp import Llama

def create_system_prompt():
    return f'''
    당신은 계산을 정확하게 수행하는 계산기입니다.
    주어진 계산 문제에 대해 정확한 답을 제공해주세요.
    계산 문제에 대한 풀이과정을 항상 포함합니다.
    '''

def create_user_prompt(user_input):
    return f'''
    계산 문제는 다음과 같습니다: {user_input}
    '''

llm = Llama.from_pretrained(
    repo_id="unsloth/gemma-3-4b-it-GGUF",
    filename="gemma-3-4b-it-Q4_K_M.gguf",
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
                "role": "system",
                "content": create_system_prompt()
            },
            {
                "role": "user",
                "content": create_user_prompt(user_input)
            }
        ]
    )

    print('답변: ' + response['choices'][0]['message']['content'])