from functools import partial
from TrainingBatchesPrepare import custom_collate_fn,InstructionDataset
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
customized_collate_fn = partial(
    custom_collate_fn,
    device = device,
    allowed_max_length = 1024
)

from torch.utils.data import DataLoader
from DatasetPrepare import partition_dataset

def Initialize_DataLoader(tokenizer):
    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)

    train_data,test_data,val_data = partition_dataset()

    train_dataset = InstructionDataset(train_data,tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_dataset = InstructionDataset(val_data,tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    test_dataset = InstructionDataset(test_data,tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    return train_loader,val_loader,test_loader

if __name__ == "__main__":
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    train_loader,val_loader,test_loader = Initialize_DataLoader(tokenizer)
    print("Train loader:")
    for inputs,targets in train_loader:
        print(inputs.shape,targets.shape)