import torch
import torch.onnx
import torch.nn as nn

# Define the sizes of the matrices
A = torch.randn(1, 144, 5)  # Matrix A: 144x128
B = torch.randn(5, 10)  # Matrix B: 128x384

file_name = "/home/ana/hhj/mace/workspace/attention/data/matmul_144_5_10/input.raw"
A.numpy().tofile(file_name)
# Perform matrix multiplication
result = torch.matmul(A, B)

# Define a model that just returns the result of the matrix multiplication
class MatMulModel(torch.nn.Module):
    def __init__(self, B):
        super(MatMulModel, self).__init__()
        self.B = nn.Parameter(B)  # Register B as a learnable parameter
        
    def forward(self, A):
        return torch.matmul(A, self.B)

# Instantiate the model
model = MatMulModel(B)

# Export the model to ONNX format
onnx_file_path = "/home/ana/hhj/mace/workspace/attention/data/matmul_144_5_10/matmul_144_5_10.onnx"
torch.onnx.export(
    model, 
    A,
    onnx_file_path, 
    export_params=True, 
    opset_version=11, 
    input_names=['input'], 
    output_names=['output'],
    dynamic_axes=None
)

