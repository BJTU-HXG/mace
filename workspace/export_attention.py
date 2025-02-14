import torch
import os
import torch.nn as nn
import math
import onnx
import time
import numpy as np
import onnxruntime as ort
from onnxsim import simplify


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
        # print(f'X input shape[{X.shape}]: {X}')
        # print(f'Q weight shape:[{self.W_q.weight.shape}]', self.W_q.weight)
        # print(f'X_q shape[{self.W_q(X).shape}]: {self.W_q(X)}')
        # print(self.W_k(X))
        # print(self.W_v(X))
        Q = self.split_heads(self.W_q(X))
        K = self.split_heads(self.W_k(X))
        V = self.split_heads(self.W_v(X))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        # print(f'model1 attn_output shape: [{attn_output.shape}], {attn_output}')
        output = self.W_o(self.combine_heads(attn_output))
        return output


class MultiHeadAttention_combine(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention_combine, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_qkv = nn.Linear(d_model, d_model*3)
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
        # print(f'QKV shape: [{QKV.shape}]', {QKV})
        split_tensor = torch.split(QKV, QKV.shape[2] // 3, dim=2)
        Q = self.split_heads(split_tensor[0])
        K = self.split_heads(split_tensor[1])
        V = self.split_heads(split_tensor[2])
        # print(f'Q shape: [{Q.shape}], {Q}')
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        # print(f'model2 attn_output shape: [{attn_output.shape}], {attn_output}')
        output = self.W_o(self.combine_heads(attn_output))
        return output

class MultiHeadAttention3(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention3, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 手动初始化权重和偏置
        self.W_q_weight = nn.Parameter(torch.randn(d_model, d_model) * math.sqrt(2.0 / d_model))
        self.W_q_bias = nn.Parameter(torch.zeros(d_model))

        self.W_k_weight = nn.Parameter(torch.randn(d_model, d_model) * math.sqrt(2.0 / d_model))
        self.W_k_bias = nn.Parameter(torch.zeros(d_model))

        self.W_v_weight = nn.Parameter(torch.randn(d_model, d_model) * math.sqrt(2.0 / d_model))
        self.W_v_bias = nn.Parameter(torch.zeros(d_model))

        self.W_o_weight = nn.Parameter(torch.randn(d_model, d_model) * math.sqrt(2.0 / d_model))
        self.W_o_bias = nn.Parameter(torch.zeros(d_model))

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
    
    def linear_transform(self, X, weight, bias):
        # 使用 matmul 和 bias_add 实现线性变换
        return torch.matmul(X, weight) + bias
    
    def forward(self, X, mask=None):
        Q = self.split_heads(self.linear_transform(X, self.W_q_weight, self.W_q_bias))
        K = self.split_heads(self.linear_transform(X, self.W_k_weight, self.W_k_bias))
        V = self.split_heads(self.linear_transform(X, self.W_v_weight, self.W_v_bias))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        # print(f'model1 attn_output shape: [{attn_output.shape}], {attn_output}')
        output = self.linear_transform(self.combine_heads(attn_output), self.W_o_weight, self.W_o_bias)
        return output

def cosine_sim(a,b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a.flatten(),b.flatten()) / (a_norm * b_norm)
    return cos

def combine_weight(model1_weights_dict, model2_weight_dict):
    q_weight = model1_weights_dict['W_q.weight']
    q_bias = model1_weights_dict['W_q.bias']
    k_weight = model1_weights_dict['W_k.weight']
    k_bias = model1_weights_dict['W_k.bias']
    v_weight = model1_weights_dict['W_v.weight']
    v_bias = model1_weights_dict['W_v.bias']
    o_weight = model1_weights_dict['W_o.weight']
    o_bias = model1_weights_dict['W_o.bias']
    print(f'q k v shape: {q_weight.shape}')
    print(q_weight)
    # print(k_weight)
    # print(v_weight)
    height = q_weight.shape[0]
    width = q_weight.shape[1]

    qkv_weight_split = torch.zeros(width*3, height)
    for i in range(width):
        for j in range(height):
            qkv_weight_split[i][j] = q_weight[i][j]
            qkv_weight_split[i+d_model][j] = k_weight[i][j]
            qkv_weight_split[i+2*d_model][j] = v_weight[i][j]
    # print(qkv_weight_split.shape)
    # print(qkv_weight_split)
    qkv_bias_split = torch.cat([q_bias, k_bias, v_bias], dim=0)
    model2_weights_dict['W_qkv.weight'] = qkv_weight_split
    model2_weights_dict['W_qkv.bias'] = qkv_bias_split
    model2_weights_dict['W_o.weight'] = o_weight
    model2_weights_dict['W_o.bias'] = o_bias
    return model2_weights_dict

def combine_weight_transpack(model1_weights_dict, model3_weights_dict):

    q_weight = model1_weights_dict['W_q.weight']
    q_bias = model1_weights_dict['W_q.bias']
    k_weight = model1_weights_dict['W_k.weight']
    k_bias = model1_weights_dict['W_k.bias']
    v_weight = model1_weights_dict['W_v.weight']
    v_bias = model1_weights_dict['W_v.bias']
    o_weight = model1_weights_dict['W_o.weight'].transpose(0,1).contiguous()
    o_bias = model1_weights_dict['W_o.bias']
    
    width = q_weight.shape[0]
    height = q_weight.shape[1]
    
    combine_qkv = torch.cat([q_weight.flatten(), k_weight.flatten(), v_weight.flatten()] )
    
    qkv_weight_split = torch.zeros(3*height, width)
    start_idx = 0
    for k in range(qkv_weight_split.shape[1]):
        qkv_weight_split[:height, k] = combine_qkv[start_idx: start_idx+height]
        qkv_weight_split[height : 2*height, k] = combine_qkv[start_idx+height: start_idx+2*height]
        qkv_weight_split[2*height : 3*height, k] = combine_qkv[start_idx+2*height: start_idx+3*height]
        start_idx += 3*height
    print(qkv_weight_split) 
    
    qkv_bias_split = torch.cat([q_bias, k_bias, v_bias], dim=0)

    model3_weights_dict['W_qkv.weight'] = qkv_weight_split
    model3_weights_dict['W_qkv.bias'] = qkv_bias_split
    model3_weights_dict['W_o.weight'] = o_weight
    model3_weights_dict['W_o.bias'] = o_bias
    return model3_weights_dict

def save_model(model, onnx_path):
    torch.onnx.export(
        model, 
        X,
        onnx_path,
        export_params=True,
        opset_version=12,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None
    )
    onnx_model = onnx.load(onnx_path)
    sim_model, check = simplify(onnx_model)
    onnx.save(sim_model, onnx_path)


# 标准的Attention
batch = 128
d_model = 400
num_heads = 4

# 路径名称
dir1 = '/home/ana/hhj/mace/workspace/attn/attn_1_128_400'
dir2 = '/home/ana/hhj/mace/workspace/attn/qkv_1_128_400'
dir3 = '/home/ana/hhj/mace/workspace/attn/trans_1_128_400'

if not os.path.exists(dir1):
    os.makedirs(dir1)
if not os.path.exists(dir2):
    os.makedirs(dir2)
if not os.path.exists(dir3):
    os.makedirs(dir3)

input_path = os.path.join(dir1, 'input.raw')
input_qkv_path = os.path.join(dir2, 'input.raw')
input_trans_path = os.path.join(dir3, 'input.raw')

weights_path = os.path.join(dir1, 'weights.pth')
weights_qkv_path = os.path.join(dir2, 'weights.pth')
weights_trans_path = os.path.join(dir3, 'weights.pth')

onnx_path = os.path.join(dir1, 'attn.onnx')
onnx_qkv_path = os.path.join(dir2, 'qkv.onnx')
onnx_trans_path = os.path.join(dir3, 'trans.onnx')

# 加载输入和权重
X = torch.zeros(1, batch, d_model)
if os.path.exists(input_path):
    data = np.fromfile(input_path, dtype=np.float32).reshape(1,batch, d_model)
    X = torch.from_numpy(data)
elif not os.path.exists(input_path):
    X = torch.randn(1, batch, d_model) 
    with open(input_path, 'w') as f:
        X.numpy().tofile(input_path)
    with open(input_qkv_path, 'w') as f:
        X.numpy().tofile(input_qkv_path)
    with open(input_trans_path, 'w') as f:
        X.numpy().tofile(input_trans_path)


model1 = MultiHeadAttention(d_model,num_heads)
model1_weights_dict = model1.state_dict()
if os.path.exists(weights_path):
    model1.load_state_dict(torch.load(weights_path))
    print(f'weight load finish!')
elif not os.path.exists(weights_path):
    with open(weights_path, 'w') as f:
        torch.save(model1_weights_dict, weights_path)

model2 = MultiHeadAttention_combine(d_model, num_heads)  
model2_weights_dict = model2.state_dict()    
if os.path.exists(weights_qkv_path):
    model2.load_state_dict(torch.load(weights_qkv_path))
    print('qkv weights load finish!')
elif not os.path.exists(weights_qkv_path):
    weights = combine_weight(model1_weights_dict, model2_weights_dict)
    with open(weights_qkv_path, 'w') as f:
        torch.save(weights, weights_qkv_path)
    model2.load_state_dict(torch.load(weights_qkv_path))
# save model
save_model(model1, onnx_path)
save_model(model2, onnx_qkv_path)

model3 = MultiHeadAttention_combine(d_model, num_heads)
model3_weights_dict = model3.state_dict()
if os.path.exists(weights_trans_path):
    model3.load_state_dict(torch.load(weights_trans_path))
elif not os.path.exists(weights_trans_path):
    weights = combine_weight_transpack(model1_weights_dict, model3_weights_dict)
    with open(weights_trans_path, 'w') as f:
        torch.save(weights, weights_trans_path)
    model3.load_state_dict(torch.load(weights_trans_path))
save_model(model3, onnx_trans_path)

# cpu inference
session1 = ort.InferenceSession(onnx_path)
input_name = session1.get_inputs()[0].name
output_name = session1.get_outputs()[0].name
start1 = time.time()
output1 = session1.run([output_name], {input_name: X.numpy()})[0]
end1 = time.time()
cost1 = end1 - start1
print(f'model1 cost time: {cost1}')


session2 = ort.InferenceSession(onnx_qkv_path)
input_name = session2.get_inputs()[0].name
output_name = session2.get_outputs()[0].name
start2 = time.time()
output2 = session2.run([output_name], {input_name: X.numpy()})[0]
end2 = time.time()
cost2 = end2 - start2
print(f'model2 cost time: {cost2}')
print(f'speed up: {cost1 / cost2}')
print(f'output1 shape: {output1.shape}  output2 shape: {output2.shape}')
print(f'cossim : {cosine_sim(output1.flatten(), output2.flatten())}')

