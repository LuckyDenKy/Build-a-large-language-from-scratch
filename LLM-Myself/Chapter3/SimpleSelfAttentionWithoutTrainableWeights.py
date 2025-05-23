import time
import torch

# 自定义一句话便于后续程序验证使用
# "Your journey starts with one step"
inputs = torch.tensor(
    [[0.43,0.15,0.89],  # Your     (x^1)
     [0.55,0.87,0.66],  # journey  (x^2)
     [0.57,0.85,0.64],  # starts   (x^3)
     [0.22,0.58,0.33],  # with     (x^4)
     [0.77,0.25,0.10],  # one      (x^5)
     [0.05,0.80,0.55]]  # step     (x^6)
)


if __name__ == "__main__":
    print("---------------------------Strat---------------------------")
    start = time.time()
    print("Here suppose 'journey' as query to calculate the attention scores")
    query = inputs[1]
    print(f"'journey' query (inputs[1]) = {query}")
    
    # 1. Calculate attention scores
    attn_scores_2 = torch.empty(inputs.shape[0])      # 用句子单词个数初始化是因为，最后的attn_scores_2的每个元素代表对query的注意力分数值
    for i,x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(x_i,query)
    print("Attention scores :",attn_scores_2)
    print("Shape of attn_scores_2 :",attn_scores_2.shape)

    # 2. Normalize the attention scores
    attn_weights_2 = torch.softmax(attn_scores_2,dim=0)
    print("After normalize the attention scores, output the attention weights :",attn_weights_2)
    print("Shape of attn_weights_2 :",attn_weights_2.shape)

    # 3. Weighted sum to compute the context vector
    context_vec_2 = torch.zeros(query.shape)
    for i,x_i in enumerate(inputs):
        context_vec_2 += attn_weights_2[i] * x_i
    print("Context vector :",context_vec_2)

    # 4. Compute attention weights for all inputs
    atten_scores = torch.empty(6,6)
    for i,x_i in enumerate(inputs):
        for j,x_j in enumerate(inputs):
            atten_scores[i,j] = torch.dot(x_i,x_j)
    print("All attention scores :\n",atten_scores)
    atten_weights = torch.softmax(atten_scores,dim=1)
    print("All attention weights :\n",atten_weights)
    context_vec = torch.zeros(6,3)
    for j,aw_j in enumerate(atten_weights):
        for i,x_i in enumerate(inputs):
            context_vec[j] += aw_j[i] * x_i
    print("All context vectors :\n",context_vec)
    # 另一种算法
    context_vec = atten_weights @ inputs
    print("All context vectors :\n",context_vec)

    end = time.time()
    print("---------------------------Finish---------------------------")
    print(f"This running cost {end-start} seconds")