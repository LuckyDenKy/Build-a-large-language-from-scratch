from CreateDataLoader import SpamDataset

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

train_dataset = SpamDataset("train.csv",max_length=None,tokenizer=tokenizer)
val_dataset = SpamDataset("validation.csv",max_length=train_dataset.max_length,tokenizer=tokenizer)
test_dataset = SpamDataset("test.csv",max_length=train_dataset.max_length,tokenizer=tokenizer)

from torch.utils.data import DataLoader
num_workers = 0
batch_size = 8
torch.manual_seed(123)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False
)
# print("Train loader:")
# for input_batch, target_batch in train_loader:
#     pass
# print("Input batch dimensions:", input_batch.shape)
# print("Label batch dimensions", target_batch.shape)
# print(f"{len(train_loader)} training batches")
# print(f"{len(val_loader)} validation batches")
# print(f"{len(test_loader)} test batches")


from gpt_download import download_and_load_gpt2
from ch4 import GPTModel
from ch5 import load_weights_into_gpt
CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(
    model_size=model_size, models_dir="gpt2"
)
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

from ch4 import generate_text_simple
from ch5 import text_to_token_ids,token_ids_to_text
# text_1 = "Every effort moves you"
# token_ids = generate_text_simple(
#     model=model,
#     idx=text_to_token_ids(text_1, tokenizer),
#     max_new_tokens=15,
#     context_size=BASE_CONFIG["context_length"]
# )
# print(token_ids_to_text(token_ids, tokenizer))

num_classes = 2
model.out_head = torch.nn.Linear(in_features=768,out_features=num_classes)
model.to(device)

from CalculateAccAndLoss import calc_accuracy_loader
# train_accuracy = calc_accuracy_loader(
#     train_loader, model, device, num_batches=10
# )
# val_accuracy = calc_accuracy_loader(
#     val_loader, model, device, num_batches=10
# )
# test_accuracy = calc_accuracy_loader(
#     test_loader, model, device, num_batches=10
# )
# print(f"Training accuracy: {train_accuracy*100:.2f}%")
# print(f"Validation accuracy: {val_accuracy*100:.2f}%")
# print(f"Test accuracy: {test_accuracy*100:.2f}%")

### Implementing a LoRA layer
import math
class LoRALayer(torch.nn.Module):
    def __init__(self,in_dim,out_dim,rank,alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim,rank))
        torch.nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
        self.B = torch.nn.Parameter(torch.zeros(rank,out_dim))
        self.alpha = alpha
    
    def forward(self,x):
        x = self.alpha * (x @ self.A @ self.B)
        return x

class LinerWithLoRA(torch.nn.Module):
    def __init__(self,linear,rank,alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features,linear.out_features,rank,alpha)
    
    def forward(self,x):
        return self.linear(x) + self.lora(x)

def replace_linear_with_lora(model,rank,alpha):
    for name,module in model.named_children():
        if isinstance(module,torch.nn.Linear):
            setattr(model,name,LinerWithLoRA(module,rank,alpha))
        else:
            replace_linear_with_lora(module,rank,alpha)

# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Total trainable parameters before:{total_params}")

# for param in model.parameters():
#     param.requires_grad = False
# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Total trainable parameters after: {total_params:,}")

# replace_linear_with_lora(model, rank=16, alpha=16)
# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Total trainable LoRA parameters: {total_params:,}")

from ch6 import train_classifier_simple
import time
start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 5
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=50, eval_iter=5,
    tokenizer=tokenizer
)
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

from ch6 import plot_values
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
plot_values(
    epochs_tensor, examples_seen_tensor,
    train_losses, val_losses, label="loss"
)
train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)
print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")