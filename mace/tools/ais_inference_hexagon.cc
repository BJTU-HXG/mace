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

#include "mace/codegen/engine/mace_engine_factory.h"

#include "ais_inference_hexagon.h"
#include "status.h"
namespace AISRT {
namespace INFER {
using namespace std;

StatusCode AisInferenceHexagon::Load(const string &model_name, const AisInferConfig &config) {

  std::string FLAGS_input_node = "input,icache1,icache2,icache3";
  std::string FLAGS_output_node = "output,output_ce,ocache1,ocache2,ocache3";
  std::vector<std::string> input_names = Split(FLAGS_input_node, ',');
  std::vector<std::string> output_names = Split(FLAGS_output_node, ',');
  std::string FLAGS_model_file = "/";
  std::string FLAGS_model_data_file = "/";
  //读取模型.pb文件
  std::unique_ptr<mace::port::ReadOnlyMemoryRegion> model_graph_data =
      make_unique<mace::port::ReadOnlyBufferMemoryRegion>();
  if (FLAGS_model_file != "") {
    auto fs = GetFileSystem();
    status = fs->NewReadOnlyMemoryRegionFromFile(FLAGS_model_file.c_str(),
                                                 &model_graph_data);
  }
  //读取模型.data文件
  std::unique_ptr<mace::port::ReadOnlyMemoryRegion> model_weights_data =
      make_unique<mace::port::ReadOnlyBufferMemoryRegion>();
  if (FLAGS_model_data_file != "") {
    auto fs = GetFileSystem();
    status = fs->NewReadOnlyMemoryRegionFromFile(FLAGS_model_data_file.c_str(),
                                                 &model_weights_data);
  }

  bool *model_data_unused = nullptr;
  mace::MaceEngine *tutor = nullptr;

  mace::MaceStatus status;
  mace::MaceEngineConfig config;
  status = config.SetCPUThreadPolicy(-1, static_cast<mace::CPUAffinityPolicy >(1));
  #ifdef MACE_ENABLE_HEXAGON
    // SetHexagonToUnsignedPD() can be called for 8150 family(with new cDSP
    // firmware) or 8250 family above to run hexagon nn on unsigned PD.
  #ifdef __ANDROID__
    config.SetHexagonToUnsignedPD();
  #endif
    config.SetHexagonPower(HEXAGON_NN_CORNER_TURBO, true, 100);
  #endif
  std::shared_ptr<mace::MaceEngine> engine;
  mace::MaceStatus create_engine_status;

  return StatusCodeOK;
}

StatusCode Run(IHANDLE instance, const vector<AisTensorParam> &input_tensors, vector<AisTensorParam> &output_tensors) {
  
}

}
}