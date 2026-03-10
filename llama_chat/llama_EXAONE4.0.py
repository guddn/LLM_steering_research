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

# jinja2 템플릿 에러
# llm = Llama.from_pretrained(
#     repo_id="LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF",
#     filename="EXAONE-4.0-1.2B-BF16.gguf",
#     n_ctx=2048,
#     n_threads=4,
#     verbose=False
# )

path = '/Users/hello/.cache/huggingface/hub/models--LGAI-EXAONE--EXAONE-4.0-1.2B-GGUF/snapshots/162446400ea4596377a3ce1d3ddffa32971af0a6/EXAONE-4.0-1.2B-BF16.gguf'

llm = Llama(
    model_path=path,
    n_ctx=2048,
    verbose=False,
    chat_format="llama-2",
    tokenizer=None,
)

user_input = ""

while user_input != "q":
    # user_input = input("궁금한 점이 있으신가요?: ")
    # response = llm.create_chat_completion(
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": create_system_prompt()
    #         },
    #         {
    #             "role": "user",
    #             "content": create_user_prompt(user_input)
    #         }
    #     ]
    # )
    response = llm(
        user_input,
        max_tokens=128,
        stop=["\n"],
        echo=True
    )

    # print('답변: ' + response['choices'][0]['message']['content'])
    print('답변: ' + response['choices'][0]['text'])