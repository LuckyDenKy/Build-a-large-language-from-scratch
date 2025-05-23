from ch4 import generate_text_simple
from ch5 import text_to_token_ids, token_ids_to_text
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
import torch

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length":1024,
    "drop_rate":0.0,
    "qkv_bias":True
}
model_configs = {
    "gpt2-small (124M)": {"emb_dim":768,"n_layers":12,"n_heads":12},
    "gpt2-medium (355M)": {"emb_dim":1024,"n_layers":24,"n_heads":16},
    "gpt2-large (774M)": {"emb_dim":1280,"n_layers":36,"n_heads":20},
    "gpt2-xl (1558M)": {"emb_dim":1600,"n_layers":48,"n_heads":25},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

#from gpt_download import download_and_load_gpt2
from ch5 import load_gpt2
from ch4 import GPTModel
from ch5 import load_weights_into_gpt

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
#settings,params = download_and_load_gpt2(model_size,models_dir="gpt2")
settings,params = load_gpt2(model_size,models_dir="gpt2")
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model,params)
model.eval()
for param in model.parameters():
    param.requires_grad = False
torch.manual_seed(123)
num_classes = 2
# replace the last layer of the model with a new one
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],
    out_features=num_classes
)
# Make the final LayerNorm and last transformer block trainable
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True

def getModel():
    return model

def temp():
    text1 = "Every effort moves you"
    token_ids = generate_text_simple(
        model = model,
        idx=text_to_token_ids(text1,tokenizer),
        max_new_tokens=15,
        context_size=BASE_CONFIG["context_length"]
    )
    print(token_ids_to_text(token_ids,tokenizer))

    text2 = (
        "Is the following text 'spam'? Answer with 'yes' or 'no':"
        " 'You are a winner you have been specially"
        " selected to receive $1000 cash or a $ 2000 award.'"
    )
    token_ids = generate_text_simple(
        model = model,
        idx=text_to_token_ids(text2,tokenizer),
        max_new_tokens=23,
        context_size = BASE_CONFIG["context_length"]
    )
    print(token_ids_to_text(token_ids,tokenizer))

def temp2():
    inputs = tokenizer.encode("Do you have time")
    inputs = torch.tensor(inputs).unsqueeze(0)
    print("Inputs:",inputs)
    print("Inputs dimensions:",inputs.shape)

    with torch.no_grad():
        outputs = model(inputs)
    print("Outputs:\n",outputs)
    print("Outputs dimensions:",outputs.shape)
    print("Last output token:",outputs[:,-1,:])

    probas = torch.softmax(outputs[:,-1,:],dim=-1)
    label = torch.argmax(probas)
    print("Class label:",label.item())

if __name__ == "__main__":
    temp2()