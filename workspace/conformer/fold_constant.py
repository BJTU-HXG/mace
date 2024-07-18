import onnx
import onnxruntime as ort
import numpy as np
import torch
from onnx import TensorProto, numpy_helper, checker, optimizer
from onnx.helper import make_tensor_value_info


def load_model(path):
    model = onnx.load(path)
    return model

def get_curr_node(net, ndlist):
    outputs = []
    curr_nodes = []
    for node in ndlist:
        outputs.append(node.output[0])
    for node in net.node:
        for ipt in node.input:
            if ipt in outputs:
                curr_nodes.append(node)
    return curr_nodes

def remove_useless_nodes(*args, **kwargs):
    for arg in args:
        for node in arg:
            model.graph.node.remove(node)
    
def fold_constant(model):
    net = model.graph
    mul_nodes = []
    matmul_nodes = []
    reshape_nodes = []
    trans_nodes = []
    bmm_nodes = []
    
    for node in net.node:
        if node.op_type == 'Mul' and node.input[1] == 'pos_emb_helper':
            mul_nodes.append(node)
    matmul_nodes = get_curr_node(net, mul_nodes)
    reshape_nodes = get_curr_node(net, matmul_nodes)
    trans_nodes = get_curr_node(net, reshape_nodes)
    bmm_nodes = get_curr_node(net, trans_nodes)
    
    mat_a_name = mul_nodes[0].input[0]
    mat_b_name = matmul_nodes[0].input[1]
    reshape_name = reshape_nodes[0].input[1]
    axes = tuple(trans_nodes[0].attribute[0].ints)

    mat_a = None
    mat_b = None
    dst_shape = None
    initializer_todel = []
    for tensor in net.initializer:
        if tensor.name == mat_a_name:
            shape = tuple(tensor.dims)
            mat_a = torch.tensor(np.frombuffer(tensor.raw_data, dtype=np.float32)).reshape(shape)
            initializer_todel.append(tensor)
        elif tensor.name == mat_b_name:
            shape = tuple(tensor.dims)
            mat_b = torch.tensor(np.frombuffer(tensor.raw_data, dtype=np.float32)).reshape(shape)
            initializer_todel.append(tensor)
        elif tensor.name == reshape_name:
            dst_shape = tuple(np.frombuffer(tensor.raw_data, dtype=np.int64)) 
    # remove useless initializer
    for tensor in initializer_todel:
        net.initializer.remove(tensor)
    # remove useless node
    remove_useless_nodes(mul_nodes, matmul_nodes, reshape_nodes, trans_nodes, model=model)
    # remove useless input
    for tensor in net.input:
        if tensor.name == mul_nodes[0].input[1]:
            net.input.remove(tensor)
    # fold constant
    print(f'a shape: {mat_a.size()}, b shape: {mat_b.size()}, dst shape: {dst_shape}, transpose: {axes}')
    res_const = torch.matmul(mat_a, mat_b).reshape(dst_shape).permute(axes).numpy()
    initializer = numpy_helper.from_array(res_const, 'pos_initializer')
    net.initializer.append(initializer)
    for node in bmm_nodes:
        node.input[1] = initializer.name

    

if __name__ == '__main__':
    path = '/home/huaijie.h/workspace/models/conformer.onnx'
    model = load_model(path)
    fold_constant(model)
    checker.check_model(model)
    onnx.save(model, 'change_conformer.onnx')
    print(1)