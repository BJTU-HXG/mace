import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        # print(self.W_q.bias)
        # print(self.W_k.bias)
        # print(self.W_v.bias)
        
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        # mask应该是一个与attn_scores相同维度的矩阵
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask==0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
    
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1,2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1,2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, X, mask=None):
        # QKV的Linear层使得多头能够关注不同的语义特征
        # print(self.W_q(X))
        # print(self.W_k(X))
        # print(self.W_v(X))
        Q = self.split_heads(self.W_q(X))
        K = self.split_heads(self.W_k(X))
        V = self.split_heads(self.W_v(X))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


# 标准的Attention
print("==============Attention===============")
d_model = 128
num_heads = 4
input = torch.randn(1,144,128)
X = input
model1 = MultiHeadAttention(d_model,num_heads)
output1 = model1(X)
# print(output1)

# file_name = "/home/ana/hhj/mace/workspace/attention/data/attn_3_144_128/input.raw"
# input.numpy().tofile(file_name)

# onnx_file_path = "/home/ana/hhj/mace/workspace/attention/data/attn_3_144_128/attn_3_144_128.onnx"
# torch.onnx.export(
#     model1, 
#     input,
#     onnx_file_path, 
#     export_params=True, 
#     opset_version=12, 
#     input_names=['input'], 
#     output_names=['output'],
#     dynamic_axes=None
# )
print(f'=================Attention完毕=====================')

# 存储模型权重
weights_dict = model1.state_dict()
# print(weights_dict)
q_weight = weights_dict['W_q.weight']
q_bias = weights_dict['W_q.bias']
k_weight = weights_dict['W_k.weight']
k_bias = weights_dict['W_k.bias']
v_weight = weights_dict['W_v.weight']
v_bias = weights_dict['W_v.bias']
# print(X)
# print(q_weight)
# print(k_weight)
# print(v_weight)
Q = torch.add(torch.matmul(X, q_weight.transpose(0,1)), q_bias)
K = torch.add(torch.matmul(X, k_weight.transpose(0,1)), k_bias)
V = torch.add(torch.matmul(X, v_weight.transpose(0,1)), v_bias)
# print(Q)
# print(K)
# print(V)



height = q_weight.shape[0]
width = q_weight.shape[1]
qkv_weight = torch.cat([q_weight.unsqueeze(2), k_weight.unsqueeze(2), v_weight.unsqueeze(2)], dim=2).reshape(q_weight.shape[0], q_weight.shape[1] * 3).transpose(0,1).contiguous()
qkv_bias = torch.cat([q_bias.unsqueeze(1), k_bias.unsqueeze(1), v_bias.unsqueeze(1)], dim=1).reshape(1, q_bias.shape[0] * 3).transpose(0,1).contiguous().reshape(-1)


print(qkv_weight.shape)
print(qkv_bias.shape)



# QKV矩阵合并后的attention split qkv
qkv_weight_split = torch.cat([q_weight, k_weight, v_weight], dim=1).transpose(0,1)
# print(qkv_weight_split)
qkv_bias_split = torch.cat([q_bias, k_bias, v_bias], dim=0)
# print(qkv_bias_split)
class MultiHeadAttention_combine(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention_combine, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # self.W_qkv = nn.Linear(d_model, d_model*3)
        self.W_qkv = nn.Linear(d_model, d_model*3)
        self.W_qkv.weight = torch.nn.Parameter(qkv_weight_split)
        self.W_qkv.bias = torch.nn.Parameter(qkv_bias_split)
        # print(torch.matmul(X, self.W_qkv.weight.transpose(0,1)))
        # print(torch.add(torch.matmul(X, self.W_qkv.weight.transpose(0,1)), self.W_qkv.bias))
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        # mask应该是一个与attn_scores相同维度的矩阵
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask==0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
    
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1,2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1,2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, X, mask=None):
        # QKV的Linear层使得多头能够关注不同的语义特征
        QKV = self.W_qkv(X)
        # print(QKV)
        split_tensor = torch.split(QKV, QKV.shape[2] // 3, dim=2)
        Q = self.split_heads(split_tensor[0])
        K = self.split_heads(split_tensor[1])
        V = self.split_heads(split_tensor[2])
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
    

print("========================Attention Combine===================")
model2 = MultiHeadAttention_combine(d_model, num_heads)
output2 = model2(X)
# print(output2)
file_name = "/home/ana/hhj/mace/workspace/attention/data/qkv_1_144_128/input.raw"
input.numpy().tofile(file_name)

onnx_file_path = "/home/ana/hhj/mace/workspace/attention/data/qkv_1_144_128/qkv_1_144_128.onnx"
torch.onnx.export(
    model2, 
    input,
    onnx_file_path, 
    export_params=True, 
    opset_version=12, 
    input_names=['input'], 
    output_names=['output'],
    dynamic_axes=None
)
print("========================Attention Combine完毕===================")  
