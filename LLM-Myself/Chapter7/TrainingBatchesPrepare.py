from DatasetPrepare import format_input

import torch
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    def __init__(self,data,tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))
    
    def __getitem__(self,index):
        return self.encoded_texts[index]
    
    def __len__(self):
        return len(self.data)
    
def custom_collate_draft_1(batch,pad_token_id=50256,device="cpu"):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst = []

    for item in batch:
        new_item = item.copy()
        new_item + [pad_token_id]

        padded = (
            new_item + [pad_token_id] * (batch_max_length-len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        inputs_lst.append(inputs)
    
    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor

def custom_collate_draft_2(batch,pad_token_id=50256,device="cpu"):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst,targets_lst = [],[]

    for item in batch:
        new_item = item.copy()
        new_item + [pad_token_id]
        padded = (
            new_item + [pad_token_id] * (batch_max_length-len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor,targets_tensor

def custom_collate_fn(batch,pad_token_id=50256,ignore_index=-100,allowed_max_length=None,device="cpu"):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst,targets_lst = [],[]

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] * (batch_max_length-len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        # Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        
        # Optional truncates to the maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor,targets_tensor



if __name__ == "__main__":
    inputs_1 = [0,1,2,3,4]
    inputs_2 = [5,6]
    inputs_3 = [7,8,9]
    batch = (inputs_1,inputs_2,inputs_3)
    # print(custom_collate_draft_1(batch))

    #inputs,targets = custom_collate_draft_2(batch)

    inputs,targets = custom_collate_fn(batch)

    print(inputs)
    print(targets)

