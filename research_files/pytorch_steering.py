import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)
model.eval()

hidden_size = model.config.hidden_size
steering_vector = torch.randn(hidden_size, device=model.device)
steering_vector = steering_vector / steering_vector.norm()

target_layer = 12
alpha = 3.0
target_token = -1

def steering_hook(module, inputs, output):
    if isinstance(output, tuple):
        hidden_states = output[0]
        hidden_states = hidden_states.clone()
        hidden_states[:, target_token, :] += alpha * steering_vector
        return (hidden_states,) + output[1:]
    else:
        hidden_states = output.clone()
        hidden_states[:, target_token, :] += alpha * steering_vector
        return hidden_states

handle = model.model.layers[target_layer].register_forward_hook(steering_hook)

prompt = "상대가 너무 지쳐 보일 때 어떤 식으로 말해주는 게 좋을까?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False
    )

print(tokenizer.decode(out[0], skip_special_tokens=True))

handle.remove()