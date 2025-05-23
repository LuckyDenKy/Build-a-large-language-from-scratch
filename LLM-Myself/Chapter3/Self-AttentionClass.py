import torch
import torch.nn as nn

class SelfAttention_V1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out
        self.w_query = nn.Parameter(torch.rand(d_in,d_out))
        self.w_key = nn.Parameter(torch.rand(d_in,d_out))
        self.w_value = nn.Parameter(torch.rand(d_in,d_out))
    
    def forward(self,x):
        keys = x @ self.w_key
        queries = x@ self.w_query
        values = x @ self.w_value
        
        attention_scores = queries @ keys.T
        attn_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5,dim=-1)
        context_vec = attn_weights @ values
        return context_vec
    

class SelfAttention_V2(nn.Module):
    def __init__(self, d_in, d_out,qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.w_q = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.w_k = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.w_v = nn.Linear(d_in,d_out,bias=qkv_bias)

    def forward(self,x):
        queries = self.w_q(x)
        keys = self.w_k(x)
        values = self.w_v(x)

        attention_scores = queries @ keys.T
        attention_weigths = torch.softmax(attention_scores / keys.shape[-1]**0.5,dim=-1)
        context_vec = attention_weigths @ values
        return context_vec

class CausalAttention(nn.Module):
    def __init__(self,d_in,d_out,context_length,dropout,qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.w_query = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.w_key = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.w_value = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length,context_length),diagonal=1)
        )
    
    def forward(self,x):
        b,num_tokens,d_in = x.shape   # b表示batchsize
        queries = self.w_query(x)
        keys = self.w_key(x)
        values = self.w_value(x)

        attention_scores = queries @ keys.transpose(1,2)
        attention_scores.masked_fill_(
            self.mask.bool()[:num_tokens,:num_tokens],-torch.inf
        )
        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5,dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vec = attention_weights @ values
        return context_vec




if __name__ == "__main__":
    inputs = torch.tensor(
        [[0.43,0.15,0.89],  # Your     (x^1)
        [0.55,0.87,0.66],  # journey  (x^2)
        [0.57,0.85,0.64],  # starts   (x^3)
        [0.22,0.58,0.33],  # with     (x^4)
        [0.77,0.25,0.10],  # one      (x^5)
        [0.05,0.80,0.55]]  # step     (x^6)
    )
    d_in = inputs.shape[1]
    d_out = 2

    # torch.manual_seed(123)
    # sa_v1 = SelfAttention_V1(d_in,d_out)
    # # print(sa_v1(inputs))

    # torch.manual_seed(789)
    # sa_v2 = SelfAttention_V2(d_in,d_out)
    # print(sa_v2(inputs))

    # sa_v1.w_query = nn.Parameter(sa_v2.w_q.weight.T)
    # sa_v1.w_key = nn.Parameter(sa_v2.w_k.weight.T)
    # sa_v1.w_value = nn.Parameter(sa_v2.w_v.weight.T)
    # print(sa_v1(inputs))

    batch = torch.stack((inputs,inputs),dim=0)
    print(batch.shape)
    context_length = batch.shape[1]

    ca = CausalAttention(d_in,d_out,context_length,0.0)
    context_vec = ca(batch)
    print(context_vec.shape)
    print(context_vec)


