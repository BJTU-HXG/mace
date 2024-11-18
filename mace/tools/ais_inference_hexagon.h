#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <memory>
#include "ais_inference_base.h"
#include "mace/core/runtime/runtime.h"
#include "mace/public/mace.h"
#include "mace/port/env.h"
#include "mace/port/file_system.h"
#include "mace/utils/logging.h"
#include "mace/utils/memory.h"
#include "mace/utils/string_util.h"
#include "mace/utils/statistics.h"
#include "mace/utils/transpose.h"
#include "mace/utils/dbg.h"

namespace AISRT {
namespace INFER {


class AisInferenceHexagon : public AisInferenceBase {
 public:
  ~AisInferenceHexagon() noexcept = default;
  StatusCode Load(const string &model_name, const AisInferConfig &config);
  StatusCode Unload();
  StatusCode CreateInstance(IHANDLE *instance);
  StatusCode DestroyInstance(IHANDLE instance);
  StatusCode Run(IHANDLE instance, const vector<AisTensorParam> &input_tensors, vector<AisTensorParam> &output_tensors);
  StatusCode Reset(IHANDLE instance);
  int GetInputCount() const;
  int GetOutputCount() const;
  vector<string> GetInputNames() const;
  vector<string> GetOutputNames() const;
  std::unordered_map<std::string, int> GetMetaData() const;
 private:
  std::shared_ptr<mace::MaceEngine> engine;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<mace::IDataType> input_data_types;
  std::vector<mace::IDataType> output_data_types;
  std::vector<mace::DataFormat> input_data_formats;
  std::vector<mace::DataFormat> output_data_formats;
};

} // namespace INFER
} // namespace AISRT