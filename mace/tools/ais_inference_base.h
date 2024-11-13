#ifndef COMMON_MODULES_INFERENCE_AIS_INFERENCE_BASE_H_
#define COMMON_MODULES_INFERENCE_AIS_INFERENCE_BASE_H_

#include <string>
#include <vector>
#include "ais_inference.h"

namespace AISRT {
namespace INFER {
using namespace std;
class AisInferenceBase {
 public:
  virtual ~AisInferenceBase() = default;
  virtual StatusCode Load(const string &model_name, const AisInferConfig &config) = 0;
  virtual StatusCode Unload() = 0;
  virtual StatusCode CreateInstance(IHANDLE *instance) = 0;
  virtual StatusCode DestroyInstance(IHANDLE instance) = 0;
  virtual StatusCode Run(IHANDLE instance, const vector<AisTensorParam> &input_tensors, vector<AisTensorParam> &output_tensors) = 0;
  virtual StatusCode Reset(IHANDLE instance) = 0;
  virtual int GetInputCount() const = 0;
  virtual int GetOutputCount() const = 0;
  virtual vector<string> GetInputNames() const = 0;
  virtual vector<string> GetOutputNames() const = 0;
  virtual std::unordered_map<std::string, int> GetMetaData() const = 0;
};

}  // namespace SPEECH
}  // namespace AISRT

#endif  // COMMON_MODULES_INFERENCE_AIS_INFERENCE_BASE_H_
