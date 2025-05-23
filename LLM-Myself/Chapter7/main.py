import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from DataloaderPrepare import Initialize_DataLoader
from PrtrainedModelPrepare import prepare_model

def train():
    import time
    from ch5 import train_model_simple
    from DatasetPrepare import format_input,partition_dataset

    torch.manual_seed(123)

    train_data,test_data,val_data = partition_dataset()
    train_loader,val_loader,test_loader = Initialize_DataLoader(tokenizer)
    model = prepare_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.00005,weight_decay=0.1)
    num_epochs = 2

    start_time = time.time()
    train_losses,val_losses,tokens_seen = train_model_simple(
        model,train_loader,val_loader,optimizer,device,
        num_epochs=num_epochs,eval_freq=5,eval_iter=5,
        start_context=format_input(val_data[0]),tokenizer=tokenizer
    )
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

def main():
    train_loader,val_loader,test_loader = Initialize_DataLoader(tokenizer)
    model = prepare_model()
    print(model)

def test_1():
    from DatasetPrepare import partition_dataset,format_input
    from ch5 import generate, token_ids_to_text, text_to_token_ids

    torch.manual_seed(123)
    train_data,test_data,val_data = partition_dataset()
    input_text = format_input(val_data[0])
    print(input_text)

    model = prepare_model()
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text,tokenizer),
        max_new_tokens=35,
        context_size=1024,
        eos_id=50256,
    )
    generated_text = token_ids_to_text(token_ids,tokenizer)
    response_text = generated_text[len(input_text):].strip()
    print(response_text)

def test_2():
    from ch5 import calc_loss_loader,train_model_simple
    model = prepare_model().to(device)
    train_loader,val_loader,test_loader = Initialize_DataLoader(tokenizer)
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader,model,device,num_batchs=5
        )
        val_loss = calc_loss_loader(
            val_loader,model,device,num_batchs=5
        )
    print("Training loss:",train_loss)
    print("Validation loss:",val_loss)

if __name__ == "__main__":
    train()
    '''Ep 1 (Step 000000): Train loss 2.523, Val loss 2.522
Ep 1 (Step 000005): Train loss 0.917, Val loss 1.091
Ep 1 (Step 000010): Train loss 0.964, Val loss 0.990
Ep 1 (Step 000015): Train loss 0.861, Val loss 0.925
Ep 1 (Step 000020): Train loss 0.825, Val loss 0.892
Ep 1 (Step 000025): Train loss 0.774, Val loss 0.864
Ep 1 (Step 000030): Train loss 0.719, Val loss 0.837
Ep 1 (Step 000035): Train loss 0.690, Val loss 0.810
Ep 1 (Step 000040): Train loss 0.684, Val loss 0.801
Ep 1 (Step 000045): Train loss 0.690, Val loss 0.793
Ep 1 (Step 000050): Train loss 0.623, Val loss 0.775
Ep 1 (Step 000055): Train loss 0.628, Val loss 0.769
Ep 1 (Step 000060): Train loss 0.618, Val loss 0.755
Ep 1 (Step 000065): Train loss 0.596, Val loss 0.730
Ep 1 (Step 000070): Train loss 0.546, Val loss 0.722
Ep 1 (Step 000075): Train loss 0.524, Val loss 0.721
Ep 1 (Step 000080): Train loss 0.568, Val loss 0.722
Ep 1 (Step 000085): Train loss 0.631, Val loss 0.733
Ep 1 (Step 000090): Train loss 0.546, Val loss 0.708
Ep 1 (Step 000095): Train loss 0.592, Val loss 0.697
Ep 1 (Step 000100): Train loss 0.489, Val loss 0.690
Ep 1 (Step 000105): Train loss 0.453, Val loss 0.686
Ep 1 (Step 000110): Train loss 0.472, Val loss 0.666
Ep 1 (Step 000115): Train loss 0.467, Val loss 0.668
Below is an instruction that describes a task. Write a response that appropriately completes the request.  ### Instruction: Convert the active sentence to passive: 'The chef cooks the meal every day.'  ### Response: The chef cooks the meal every day.<|endoftext|>The following is an instruction that describes a task. Write a response that appropriately completes the request.  ### Instruction: What is the opposite of 'fast'?
Ep 2 (Step 000120): Train loss 0.463, Val loss 0.675
Ep 2 (Step 000125): Train loss 0.437, Val loss 0.689
Ep 2 (Step 000130): Train loss 0.376, Val loss 0.698
Ep 2 (Step 000135): Train loss 0.427, Val loss 0.709
Ep 2 (Step 000140): Train loss 0.413, Val loss 0.696
Ep 2 (Step 000145): Train loss 0.413, Val loss 0.680
Ep 2 (Step 000150): Train loss 0.360, Val loss 0.675
Ep 2 (Step 000155): Train loss 0.402, Val loss 0.675
Ep 2 (Step 000160): Train loss 0.362, Val loss 0.679
Ep 2 (Step 000165): Train loss 0.421, Val loss 0.694
Ep 2 (Step 000170): Train loss 0.441, Val loss 0.696
Ep 2 (Step 000175): Train loss 0.361, Val loss 0.688
Ep 2 (Step 000180): Train loss 0.390, Val loss 0.677
Ep 2 (Step 000185): Train loss 0.378, Val loss 0.650
Ep 2 (Step 000190): Train loss 0.319, Val loss 0.634
Ep 2 (Step 000195): Train loss 0.364, Val loss 0.630
Ep 2 (Step 000200): Train loss 0.380, Val loss 0.634
Ep 2 (Step 000205): Train loss 0.318, Val loss 0.641
Ep 2 (Step 000210): Train loss 0.332, Val loss 0.643
Ep 2 (Step 000215): Train loss 0.322, Val loss 0.645
Ep 2 (Step 000220): Train loss 0.332, Val loss 0.640
Ep 2 (Step 000225): Train loss 0.321, Val loss 0.637
Ep 2 (Step 000230): Train loss 0.285, Val loss 0.629
Below is an instruction that describes a task. Write a response that appropriately completes the request.  ### Instruction: Convert the active sentence to passive: 'The chef cooks the meal every day.'  ### Response: The meal is cooked every day by the chef.<|endoftext|>The following is an instruction that describes a task. Write a response that appropriately completes the request.  ### Instruction: Convert the active sentence to passive:
Training completed in 14.37 minutes.'''