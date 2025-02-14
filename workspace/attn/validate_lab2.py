import numpy as np
import onnxruntime as ort
import time
import os
import torch

def cosine_sim(a,b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a.flatten(),b.flatten()) / (a_norm * b_norm)
    return cos

output_shape = [1, 128, 64]
dir1 = '/home/ana/hhj/mace/workspace/attn/attn_1_128_64'
dir2 = '/home/ana/hhj/mace/workspace/attn/trans_1_128_64'
input_path = os.path.join(dir1, 'input.raw')
onnx_path = os.path.join(dir1, 'attn.onnx')
output_mace_path = '/home/ana/hhj/mace/workspace/attn/mace_output.raw'

X = torch.from_numpy(np.fromfile(input_path, dtype=np.float32)).reshape(output_shape)
output_mace = np.fromfile(output_mace_path, dtype=np.float32)

session = ort.InferenceSession(onnx_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

output_ort = session.run([output_name], {input_name: X.numpy()})[0]

sim = cosine_sim(output_mace, output_ort)
print(sim)