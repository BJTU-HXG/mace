#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include "ais_inference_base.h"

namespace AISRT {
namespace INFER {
using namespace std;


class AisInferenceHexagon : public AisInferenceBase {
 public:
  StatusCode Load(const string &model_name, const AisInferConfig &config);
  StatusCode Unload();
  StatusCode Run(IHANDLE instance, const vector<AisTensorParam> &input_tensors, vector<AisTensorParam> &output_tensors);
 private:
  int nn_id_;
};

} // namespace INFER
} // namespace AISRT