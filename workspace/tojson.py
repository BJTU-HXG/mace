import os
import sys

import argparse
import json
import numpy as np
import onnx
from onnx import numpy_helper


FLAGS = None

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        help="input onnx file path.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="tensor output dir.")
    return parser.parse_known_args()

def save_json_yaml(encoding_file_path: str, encodings_dict: dict):
    """
    Function which saves encoding in YAML and JSON file format
    :param encoding_file_path: file name to use to generate the yaml and json file
    :param encodings_dict: dictionary containing the encoding
    """
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)
    encoding_file_path_json = encoding_file_path
    # encoding_file_path_yaml = encoding_file_path + '.yaml'
    with open(encoding_file_path_json, 'w') as encoding_fp_json:
        json.dump(encodings_dict, encoding_fp_json, sort_keys=True, indent=4, cls=NpEncoder)
    # with open(encoding_file_path_yaml, 'w') as encoding_fp_yaml:
    #     yaml.dump(encodings_dict, encoding_fp_yaml, default_flow_style=False, allow_unicode=True)

def export_encodings(quantize_tensor_dict, encoding_file_path, inits):
    """
    Export encodings to json and yaml file
    :param encoding_file_path: path to save the encoding files
    """
    param_encodings = {}
    activation_encodings = {}
    for name in quantize_tensor_dict:
        if name in inits:
            param_encodings[name] = quantize_tensor_dict[name]
        else:
            activation_encodings[name] = quantize_tensor_dict[name]

    encodings_dict = {'version': '0.5.1',
                      'activation_encodings': activation_encodings,
                      'param_encodings': param_encodings,
                      'quantizer_args': None
                     }

    save_json_yaml(encoding_file_path, encodings_dict)


def list1_to_scalar(data):
    if data.size == 1 and len(data.shape) == 1:
        return data[0]
    else:
        return data


# fused_moving_avg_obs_fake_quant
# Inputs
# 0 X,
# 1 observer_enabled,
# 2 fake_quant_enabled,
# 3 activation_post_process.min_val,
# 4 activation_post_process.max_val,
# 5 scale,
# 6 zero_point,
# 7 activation_post_process.averaging_constant,
# 8 activation_post_process.quant_min,
# 9 activation_post_process.quant_max,
# 10 ch_axis,
# 11 is_per_channel,
# 12 is_symmetric_quant,
def extract_quantize_info(node, quantize_tensor_dict, inits):
    print("Extracting quantize info of", node.input[0])
    fake_quant_enabled = numpy_helper.to_array(inits[node.input[2]])[0]
    assert fake_quant_enabled
    min = list1_to_scalar(numpy_helper.to_array(inits[node.input[3]]))
    if min.size > 1:
        print("Skipping per channel quantization params ", node.name)
        return
    max = list1_to_scalar(numpy_helper.to_array(inits[node.input[4]]))
    scale = list1_to_scalar(numpy_helper.to_array(inits[node.input[5]]))
    zero_point = list1_to_scalar(numpy_helper.to_array(inits[node.input[6]]))
    is_symmetric_quant = str(list1_to_scalar(numpy_helper.to_array(inits[node.input[12]])))

    quantize_tensor_dict[node.input[0]] = \
        [{'min': min,
         'max': max,
        #  'scale': scale,
        #  'offset': -zero_point,
         'bitwidth': 8,
         'is_symmetric': is_symmetric_quant,
         'dtype': 'int'}]


def add_nodes_to_del(nodes_to_del, node, constants):
    nodes_to_del.append(constants[node.input[7]])
    nodes_to_del.append(constants[node.input[8]])
    nodes_to_del.append(constants[node.input[9]])
    nodes_to_del.append(constants[node.input[10]])
    nodes_to_del.append(constants[node.input[11]])
    nodes_to_del.append(constants[node.input[12]])
    nodes_to_del.append(node)


def main(unused_args):
    model = onnx.load(FLAGS.input)

    inits = {}
    constants = {}
    for init in model.graph.initializer:
        # print(init.name)
        inits[init.name] = init
    for node in model.graph.node:
        if node.op_type == "Constant":
            # print(node.name)
            inits[node.output[0]] = node.attribute[0].t
            constants[node.output[0]] = node

    model_inputs = {}
    for index, input in enumerate(model.graph.input):
        # print(input)
        model_inputs[input.name] = input
    model_outputs = {}
    for index, output in enumerate(model.graph.output):
        # print(output)
        model_outputs[output.name] = output


    producers = {}
    consumers = {}
    for node in model.graph.node:
        for input in node.input:
            if input in consumers:
                consumers[input].append(node)
            else:
                consumers[input] = [node]
        for output in node.output:
            producers[output] = node


    quantize_tensor_dict = {}
    nodes_to_del = []
    for node in model.graph.node:
        if node.op_type == "fused_moving_avg_obs_fake_quant":
            # print(node.input[0], node.output[0])
            extract_quantize_info(node, quantize_tensor_dict, inits)
            if node.output[0] in consumers:
                for consumer in consumers[node.output[0]]:
                    for i in range(len(consumer.input)):
                        if consumer.input[i] == node.output[0]:
                            consumer.input[i] = node.input[0]
            elif node.output[0] in model_outputs:
                producer = producers[node.input[0]]
                producer.output[0] = node.output[0]
            add_nodes_to_del(nodes_to_del, node, constants)

    filename = os.path.splitext(os.path.basename(FLAGS.input))[0]
    json_file_path = os.path.join(FLAGS.output_dir, filename + ".json")
    export_encodings(quantize_tensor_dict, json_file_path, inits)

    for node in nodes_to_del:
        model.graph.node.remove(node)

    output_onnx_path = os.path.join(FLAGS.output_dir, filename + "_wo_fake_quant.onnx")
    onnx.save(model, output_onnx_path)

if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    main(unused_args=[sys.argv[0]] + unparsed)

