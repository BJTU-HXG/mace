#include <iostream>
#include "ais_inference.h"
#include "ais_inference_hexagon.h"
#include "ais_inference_base.h"


using namespace std;
using namespace AISRT::INFER;
int main(int argc, char **argv) {
    AisInferenceHexagon test;
    const std::string model_name = "conformer"; 
    AisInferConfig a;
    test.Load(model_name, a);
    IHANDLE ihandle = nullptr;
    const vector<AisTensorParam> input_tensors;
    vector<AisTensorParam> output_tensors;
    test.Run(ihandle, input_tensors, output_tensors);
    return 0;
}
