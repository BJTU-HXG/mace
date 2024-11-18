#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <memory>
#include <vector>
#include <sys/types.h>
#include <dirent.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>

#include "gflags/gflags.h"
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
#include "third_party/nnlib/hexagon_nn.h"

// #include "mace/codegen/engine/mace_engine_factory.h"

#include "ais_inference_hexagon.h"
#include "status.h"
namespace AISRT {
namespace INFER {
using namespace std;

std::shared_ptr<char> ReadInputDataFromFile(
    const std::string &file_path, const int tensor_size,
    const mace::IDataType input_data_type,
    std::shared_ptr<char> input_data = nullptr) {
  auto file_data_size = tensor_size * mace::GetEnumTypeSize(mace::DataType::DT_FLOAT);
  auto buffer_in = std::shared_ptr<char>(new char[file_data_size],
                                         std::default_delete<char[]>());
  std::ifstream in_file(file_path, std::ios::in | std::ios::binary);
  if (in_file.is_open()) {
    in_file.read(buffer_in.get(), file_data_size);
    in_file.close();
  } else {
    return nullptr;
  }

  auto input_size =
      tensor_size * mace::GetEnumTypeSize(static_cast<mace::DataType>(input_data_type));
  if (input_data == nullptr) {
    input_data = std::shared_ptr<char>(new char[input_size],
                                       std::default_delete<char[]>());
  }
  // CopyDataBetweenSameType and CopyDataBetweenDiffType are not an exported
  // functions, app should not use it, the follow line is only used to
  // transform data from file during testing.
  if (input_data_type == mace::IDataType::IDT_FLOAT || input_data_type == mace::IDataType::IDT_INT32) {
    mace::ops::CopyDataBetweenSameType(
        nullptr, buffer_in.get(), input_data.get(), input_size);
  } 

  return input_data;
}

int64_t WriteOutputDataToFile(const std::string &file_path,
                              const mace::IDataType file_data_type,
                              const std::shared_ptr<void> output_data,
                              const mace::IDataType output_data_type,
                              const std::vector<int64_t> &output_shape) {
  int64_t output_size = std::accumulate(output_shape.begin(),
                                        output_shape.end(), 1,
                                        std::multiplies<int64_t>());
  auto output_bytes = output_size * sizeof(float);
  std::vector<float> tmp_output(output_size);
  // CopyDataBetweenSameType and CopyDataBetweenDiffType are not an exported
  // functions, app should not use it, the follow line is only used to
  // transform data from file during testing.
  if (file_data_type == output_data_type) {
    mace::ops::CopyDataBetweenSameType(
        nullptr, output_data.get(), tmp_output.data(), output_bytes);
  }

  std::ofstream out_file(file_path, std::ios::binary);
  MACE_CHECK(out_file.is_open(), "Open output file failed: ", strerror(errno));
  out_file.write(reinterpret_cast<char *>(tmp_output.data()), output_bytes);
  out_file.flush();
  out_file.close();

  return output_size;
}

std::string FormatName(const std::string input) {
  std::string res = input;
  for (size_t i = 0; i < input.size(); ++i) {
    if (!isalnum(res[i])) res[i] = '_';
  }
  return res;
}

StatusCode AisInferenceHexagon::Load(const string &model_name, const AisInferConfig &config) {

  std::string FLAGS_input_node = "input,icache1,icache2,icache3";
  std::string FLAGS_output_node = "output,output_ce,ocache1,ocache2,ocache3";
  input_names = mace::Split(FLAGS_input_node, ',');
  output_names = mace::Split(FLAGS_output_node, ',');
  std::string FLAGS_model_file = "/fota/shiding/conformer/conformer.pb";
  std::string FLAGS_model_data_file = "/fota/shiding/conformer/conformer.data";
  //std::string FLAGS_model_file = "/fota/shiding/conformer/conformer.pb";
  //std::string FLAGS_model_data_file = "/fota/shiding/conformer/conformer.data";

  mace::MaceStatus status;
  //读取模型.pb文件
  std::unique_ptr<mace::port::ReadOnlyMemoryRegion> model_graph_data = mace::make_unique<mace::port::ReadOnlyBufferMemoryRegion>();
  if (FLAGS_model_file != "") {
    auto fs = mace::GetFileSystem();
    status = fs->NewReadOnlyMemoryRegionFromFile(FLAGS_model_file.c_str(),
                                                 &model_graph_data);
  }
  //读取模型.data文件
  std::unique_ptr<mace::port::ReadOnlyMemoryRegion> model_weights_data = mace::make_unique<mace::port::ReadOnlyBufferMemoryRegion>();
  if (FLAGS_model_data_file != "") {
    auto fs = mace::GetFileSystem();
    status = fs->NewReadOnlyMemoryRegionFromFile(FLAGS_model_data_file.c_str(),
                                                 &model_weights_data);
  }

  bool *model_data_unused = nullptr;
  mace::MaceEngine *tutor = nullptr;

  
  mace::MaceEngineConfig config_;
  status = config_.SetCPUThreadPolicy(-1, static_cast<mace::CPUAffinityPolicy >(1));
  config_.SetHexagonPower(mace::HexagonNNCornerType::HEXAGON_NN_CORNER_TURBO, true, 100);
  mace::MaceStatus create_engine_status;

  create_engine_status =
        CreateMaceEngineFromProto(reinterpret_cast<const unsigned char *>(
                                      model_graph_data->data()),
                                  model_graph_data->length(),
                                  reinterpret_cast<const unsigned char *>(
                                      model_weights_data->data()),
                                  model_weights_data->length(),
                                  input_names,
                                  output_names,
                                  config_,
                                  &engine,
                                  model_data_unused,
                                  tutor,
                                  false);

  return StatusCodeOK;
}

StatusCode AisInferenceHexagon::Run(IHANDLE instance, const vector<AisTensorParam> &input_tensors, vector<AisTensorParam> &output_tensors) {
  const size_t input_count = input_names.size();
  const size_t output_count = output_names.size();
  input_shapes = {{1,19,40},{1,17,128},{5,1,17,128},{5,1,128,3}};
  output_shapes = {{1,4,439},{1,4,439},{1,17,128},{5,1,17,128},{5,1,128,3}};
  input_data_types = {mace::IDataType::IDT_FLOAT, mace::IDataType::IDT_FLOAT, mace::IDataType::IDT_FLOAT, mace::IDataType::IDT_FLOAT};
  output_data_types = {mace::IDataType::IDT_FLOAT, mace::IDataType::IDT_FLOAT, mace::IDataType::IDT_FLOAT, mace::IDataType::IDT_FLOAT, mace::IDataType::IDT_FLOAT};
  input_data_formats = {mace::DataFormat::NONE, mace::DataFormat::NONE, mace::DataFormat::NONE, mace::DataFormat::NONE};
  output_data_formats = {mace::DataFormat::NONE, mace::DataFormat::NONE, mace::DataFormat::NONE, mace::DataFormat::NONE, mace::DataFormat::NONE};

  std::map<std::string, mace::MaceTensor> inputs;
  std::map<std::string, mace::MaceTensor> outputs;
  std::map<std::string, int64_t> inputs_size;
  std::string FLAGS_input_file = "/fota/shiding/conformer/inputs/conformer";                
  std::string FLAGS_output_file = "/fota/shiding/conformer/outputs/conformer";                
  for (size_t i = 0; i < input_count; ++i) {
    // Allocate input and output
    // only support float and int32, use char for generalization
    // sizeof(int) == 4, sizeof(float) == 4
    auto input_tensor_size = std::accumulate(
        input_shapes[i].begin(), input_shapes[i].end(), 1,
        std::multiplies<int64_t>());
    auto file_path = FLAGS_input_file + "_" + FormatName(input_names[i]);
    auto input_data = ReadInputDataFromFile(
        file_path, input_tensor_size, input_data_types[i]);

    inputs[input_names[i]] = mace::MaceTensor(input_shapes[i], input_data,
        input_data_formats[i], input_data_types[i]);
    inputs_size[input_names[i]] = input_tensor_size;
  }

  for (size_t i = 0; i < output_count; ++i) {
    // only support float and int32, use char for generalization
    int64_t output_size =
        std::accumulate(output_shapes[i].begin(), output_shapes[i].end(), 4,
                        std::multiplies<int64_t>());
    auto buffer_out = std::shared_ptr<char>(new char[output_size],
                                            std::default_delete<char[]>());
    outputs[output_names[i]] = mace::MaceTensor(
        output_shapes[i], buffer_out, output_data_formats[i],
        static_cast<mace::IDataType>(output_data_types[i]));
  }

  mace::MaceStatus run_status = engine->Run(inputs, &outputs);
  if (run_status != mace::MaceStatus::MACE_SUCCESS) {
    //报错
  }

  for (size_t i = 0; i < output_count; ++i) {
      std::string output_name =
          FLAGS_output_file + "_" + FormatName(output_names[i]);
      auto output_data_type = outputs[output_names[i]].data_type();
      auto file_data_type =
          output_data_type == mace::IDataType::IDT_INT32 ? mace::IDataType::IDT_INT32 : mace::IDataType::IDT_FLOAT;

      auto output_size = WriteOutputDataToFile(
          output_name, file_data_type, outputs[output_names[i]].data<void>(),
          output_data_type, output_shapes[i]);
    }

  return StatusCodeOK;
}
StatusCode AisInferenceHexagon::Unload() {
  return StatusCodeOK;

}
StatusCode AisInferenceHexagon::CreateInstance(IHANDLE *instance) {
return StatusCodeOK;
}
StatusCode AisInferenceHexagon::DestroyInstance(IHANDLE instance) {
return StatusCodeOK;
}
StatusCode AisInferenceHexagon::Reset(IHANDLE instance) {
return StatusCodeOK;
}
int AisInferenceHexagon::GetInputCount() const {
return StatusCodeOK;
}
int AisInferenceHexagon::GetOutputCount() const{
return StatusCodeOK;
}
vector<string> AisInferenceHexagon::GetInputNames() const{
  return {};
}
vector<string> AisInferenceHexagon::GetOutputNames() const{
  return {};
}
std::unordered_map<std::string, int> AisInferenceHexagon::GetMetaData() const{
  std::unordered_map<std::string, int> a;
 return a;
}
}
}