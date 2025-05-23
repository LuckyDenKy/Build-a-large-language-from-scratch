import matplotlib.pyplot as plt
import time

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
from CreateDataLoader import SpamDataset
train_dataset = SpamDataset(
    csv_file="train.csv",
    max_length=None,
    tokenizer=tokenizer
)
#print("train_dataset.max_length",train_dataset.max_length)
val_dataset = SpamDataset(
    csv_file="validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
test_dataset = SpamDataset(
    csv_file="test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

from torch.utils.data import DataLoader
num_workers = 0
batch_size = 8
torch.manual_seed(123)
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,drop_last=True)
val_loader = DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,drop_last=False)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,drop_last=False)
for input_batch,target_batch in train_loader:
    pass
#print("Input batch dimensions:", input_batch.shape)
#print("Target batch dimensions:", target_batch.shape)
'''
Input batch dimensions: torch.Size([8, 120])
Target batch dimensions: torch.Size([8])
'''
# print(f"{len(train_loader)} training batches")
# print(f"{len(val_loader)} validation batches")
# print(f"{len(test_loader)} test batches")
'''
130 training batches
19 validation batches
38 test batches
'''

from ModelPrepare import getModel
model = getModel().to(device)
#print(model)
from CalculateAccAndLoss import calc_accuracy_loader, calc_loss_loader,calc_loss_batch

def test_calc_accuracy_loader():
    model.to(device)

    torch.manual_seed(123)
    train_accuracy = calc_accuracy_loader(train_loader,model,device,num_batches=10)
    val_accuracy = calc_accuracy_loader(val_loader,model,device,num_batches=10)
    test_accuracy = calc_accuracy_loader(test_loader,model,device,num_batches=10)

    print(f"Train accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")

def test_calc_loss_loader():
    model.to(device)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader,model,device,num_batches=5)
        val_loss = calc_loss_loader(val_loader,model,device,num_batches=5)
        test_loss = calc_loss_loader(test_loader,model,device,num_batches=5)
    
    print(f"Training loss:{train_loss:.3f}")
    print(f"Validation loss:{val_loss:.3f}")
    print(f"Test loss:{test_loss:.3f}")

def train_classifier_simple(model,train_loader,val_loader,optimizer,device,num_epochs,eval_freq,eval_iter):
    train_losses,val_losses,train_accs,val_accs = [],[],[],[]
    examples_seen,global_step = 0,-1

    for epoch in range(num_epochs):
        model.train()
        for input_batch,target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch,target_batch,model,device)
            loss.backward()
            optimizer.step()

            examples_seen += input_batch.shape[0]
            global_step +=1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss,val_loss = evaluate_model(model,train_loader,val_loader,device,eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )
        
        # Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader,model,device,num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader,model,device,num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
    
    return train_losses,val_losses,train_accs,val_accs,examples_seen

def evaluate_model(model,train_loader,val_loader,device,eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader,model,device,num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader,model,device,num_batches=eval_iter)
    model.train()
    return train_loss,val_loss

def plot_values(epochs_seen,examples_seen,train_values,val_values,label="loss"):
    fig,ax1 = plt.subplots(figsize=(5,3))

    ax1.plot(epochs_seen,train_values,label=f"Training {label}")
    ax1.plot(epochs_seen,val_values,linestyle="-.",label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(examples_seen,train_values,alpha=0)
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()

def train_fun():
    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=0.1)
    num_epochs = 5

    train_losses,val_losses,train_accs,val_accs,examples_seen = train_classifier_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs,
        eval_freq=50,
        eval_iter=5
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    epochs_tensor = torch.linspace(0,num_epochs,len(train_losses))
    examples_seen_tensor = torch.linspace(0,examples_seen,len(train_losses))
    plot_values(epochs_tensor,examples_seen_tensor,train_losses,val_losses)

    epochs_tensor = torch.linspace(0,num_epochs,len(train_accs))
    examples_seen_tensor = torch.linspace(0,examples_seen,len(train_accs))
    plot_values(epochs_tensor,examples_seen_tensor,train_accs,val_accs,label="accuracy")

    train_acc = calc_accuracy_loader(train_loader,model,device)
    val_acc = calc_accuracy_loader(val_loader,model,device)
    test_acc = calc_accuracy_loader(test_loader,model,device)
    print(f"Final training accuracy: {train_acc*100:.2f}%")
    print(f"Final validation accuracy: {val_acc*100:.2f}%")
    print(f"Final test accuracy: {test_acc*100:.2f}%")

    # save
    torch.save(model.state_dict(),"Chapter6/SpamClassifierModel.pth")
    
# 定义一个函数，用于加载模型
def load_model():
    pretrained_model = getModel()
    # 加载模型参数
    pretrained_model.load_state_dict(torch.load("Chapter6/SpamClassifierModel.pth",map_location=device))
    # 返回加载后的模型
    return pretrained_model

def classify_review(text,model,tokenizer,device,max_length=None,pad_token_id=50256):
    model.eval()

    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[1]
    input_ids = input_ids[:min(max_length,supported_context_length)]
    input_ids += [pad_token_id] * (max_length - len(input_ids))

    input_tensor = torch.tensor(input_ids,device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:,-1,:]
    predicted_label = torch.argmax(logits,dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"

def test_classify_review():
    model = load_model()
    text_1  = (
        "You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award."
    )
    print(classify_review(text_1,model,tokenizer,device,max_length=train_dataset.max_length))

    text_2 = (
        "Hey, just wanted to check if we're still on"
        " for dinner tonight? Let me know!"
    )
    print(classify_review(text_2,model,tokenizer,device,max_length=train_dataset.max_length))



if __name__ == "__main__":
    test_classify_review()