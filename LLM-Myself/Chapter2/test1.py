# 本章过程：Input text(例如：This is an example) -> 
# Tokenized text(分词) -> 
# Token IDs(分词ID) -> 
# Token embeddings + Positional embeddings -> 
# Input embeddings

import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from importlib.metadata import version
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self,txt,tokenizer,max_length,stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        # Tokenized text -> Token IDs
        token_ids = tokenizer.encode(txt)
        # print("tokenizer's n_vocab is ",tokenizer.n_vocab)

        # 创建输入输出文本对
        for i in range(0,len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx]),torch.tensor(self.target_ids[idx])

def create_dataloader_v1(txt,batch_size=4,max_length=256,stride=128,shuffle=True,drop_last=True):
    # 创建分词器
    tokenizer = tiktoken.get_encoding("gpt2") # n_vocab = 50257
    print("tiktoken version:",version("tiktoken"))
    print("tokenizer has been prepared")

    dataset = GPTDatasetV1(txt,tokenizer,max_length,stride)
    dataLoader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,drop_last=drop_last)

    return dataLoader

class TokenEmbedding(nn.Module):
    def __init__(self,vocab_size,output_dim,context_length):
        super(TokenEmbedding,self).__init__()
        self.context_length = context_length
        self.tokenEmbedding_layer = torch.nn.Embedding(vocab_size,output_dim)
        self.positionalEmbedding_layer = torch.nn.Embedding(context_length,output_dim)
    
    def forward(self,token_ids):
        input_embeddings = self.tokenEmbedding_layer(token_ids)+self.positionalEmbedding_layer(torch.arange(self.context_length))
        return input_embeddings


if __name__ == "__main__":
    torch.manual_seed(123)

    with open("Chapter2\\the-verdict.txt","r",encoding="utf-8") as f:
        raw_text = f.read()
    
    batch_size = 1
    max_length = 4
    dataloader = create_dataloader_v1(raw_text,batch_size=batch_size,max_length=max_length,stride=1,shuffle=False,drop_last=False)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print("first batch is: ",first_batch)    
    
    tokenEmbedding = TokenEmbedding(vocab_size=50257,output_dim=4,context_length=max_length)
    print("input embeddings: ",tokenEmbedding(first_batch[0]))
    print("target embeddings: ",tokenEmbedding(first_batch[1]))
    print("finish")


