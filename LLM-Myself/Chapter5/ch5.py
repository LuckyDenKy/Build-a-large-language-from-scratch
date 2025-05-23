import torch
from ch4 import GPTModel

GPT_CONGIG_124M = {
        "vocab_size":     50257,  # Vocabulary size
        "context_length": 256,    # Context length
        "emb_dim":        768,    # Embedding dimension
        "n_heads":        12,     # Number of attention heads
        "n_layers":       12,     # Number of layers
        "drop_rate":      0.1,    # Dropout rate
        "qkv_bias":       False   # Query-Key-Value bias
    }
torch.manual_seed(123)
model = GPTModel(GPT_CONGIG_124M)
model.eval()

import tiktoken
from ch4 import generate_text_simple

def text_to_token_ids(text,tokenizer):
    encoded = tokenizer.encode(text,allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Adds the batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids,tokenizer):
    flat = token_ids.squeeze(0)   # Removes the batch dimension
    return tokenizer.decode(flat.tolist())

# start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

# token_ids = generate_text_simple(
#     model=model,
#     idx = text_to_token_ids(start_context,tokenizer),
#     max_new_tokens=10,
#     context_size=GPT_CONGIG_124M["context_length"]
# )
# print("Output text:\n",token_ids_to_text(token_ids,tokenizer))

inputs = torch.tensor([
    [16833,3626,6100],  # every effort moves
    [40,1107,588]       # I really like
])
targets = torch.tensor([
    [3626,6100,345],   # effort moves you
    [1107,588,11311]   # really like chocolate
])

# with torch.no_grad():
#     logits = model(inputs)                  # 1.求出logits
# probas = torch.softmax(logits,dim=-1)       # 2.求出所有Probabilities
# print(probas.shape)
# token_ids = torch.argmax(probas,dim=-1,keepdim=True)
# print("Token IDs:\n",token_ids)
# print(f"Targets batch 1:{token_ids_to_text(targets[0],tokenizer)}")
# print(f"Outputs batch 1:{token_ids_to_text(token_ids[0].flatten(),tokenizer)}")
# text_idx = 0
# target_probas_1 = probas[text_idx,[0,1,2],targets[text_idx]]   # 3.求出目标的probability
# print("Text 1:",target_probas_1)

# text_idx = 1
# target_probas_2 = probas[text_idx,[0,1,2],targets[text_idx]]   # 3.求出目标的probability
# print("Text 2:",target_probas_2)

# log_probas = torch.log(torch.cat((target_probas_1,target_probas_2)))   # 4.求出log的probability
# print(log_probas)

# avg_log_probas = torch.mean(log_probas)    # 5.求出平均的probability
# print(avg_log_probas)

# neg_avg_log_probas = -avg_log_probas      # 6.求负的，结果是一个entrop loss
# print(neg_avg_log_probas)

file_path = "Chapter5/the-verdict.txt"
with open(file_path,"r",encoding='utf-8') as file:
    text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:",total_characters)  # 20479
print("Tokens:",total_tokens)          # 5145

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

from ch2 import create_dataloader_v1
torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONGIG_124M["context_length"],
    stride=GPT_CONGIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
)
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONGIG_124M["context_length"],
    stride=GPT_CONGIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
)
# print("Train loader:")
# for x,y in train_loader:
#     print(x.shape,y.shape)

# print("\nValidation loader:")
# for x,y in val_loader:
#     print(x.shape,y.shape)

def calc_loss_batch(input_batch,target_batch,model,device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0,1),target_batch.flatten()
    )
    return loss

def calc_loss_loader(data_loader,model,device,num_batchs=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batchs is None:
        num_batchs = len(data_loader)
    else:
        num_batchs = min(num_batchs,len(data_loader))
    for i,(input_batch,target_batch) in enumerate(data_loader):
        if i<num_batchs:
            loss = calc_loss_batch(input_batch,target_batch,model,device)
            total_loss += loss.item()
        else:
            break
    
    return total_loss / num_batchs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader,model,device)
    val_loss = calc_loss_loader(val_loader,model,device)
print("Training loss:",train_loss)
print("Validation loss:",val_loss)

def evaluate_model(model,train_loader,val_loader,device,eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader,model,device,num_batchs=eval_iter)
        val_loss = calc_loss_loader(val_loader,model,device,num_batchs=eval_iter)
    model.train()
    return train_loss,val_loss

def generate_and_print_sample(model,tokenizer,device,start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context,tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx = encoded,
            max_new_tokens=50,
            context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids,tokenizer)
    print(decoded_text.replace("\n"," "))
    model.train()

def train_model_simple(model,train_loader,val_loader,
                       optimizer,device,num_epochs,
                       eval_freq,eval_iter,start_context,tokenizer):
    train_losses,val_losses,track_tokens_seen = [],[],[]
    tokens_seen,global_step = 0,-1

    for epoch in range(num_epochs):
        model.train()
        for input_batch,target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch,target_batch,model,device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss,val_loss = evaluate_model(model,train_loader,val_loader,device,eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                )
        
        generate_and_print_sample(
            model,tokenizer,device,start_context
        )
    
    return train_losses,val_losses,track_tokens_seen

torch.manual_seed(123)
model = GPTModel(GPT_CONGIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = 0.0004,
    weight_decay=0.1
)
num_epochs = 10
train_losses,val_losses,tokens_seen = train_model_simple(
    model,train_loader,val_loader,optimizer,device,
    num_epochs=num_epochs,eval_freq=5,eval_iter=5,
    start_context="Every effort moves you",tokenizer=tokenizer
)

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
def plot_losses(epochs_seen,tokens_seen,train_losses,val_losses):
    fig,ax1 = plt.subplots(figsize=(5,3))
    ax1.plot(epochs_seen,train_losses,label="Training loss")
    ax1.plot(epochs_seen,val_losses,linestyle="-.",label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen,train_losses,alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.show()

epochs_tensor = torch.linspace(0,num_epochs,len(train_losses))
plot_losses(epochs_tensor,tokens_seen,train_losses,val_losses)