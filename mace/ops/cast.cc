// Copyright 2018 The MACE Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mace/core/ops/operator.h"
#include "mace/core/registry/ops_registry.h"

#if defined(MACE_ENABLE_NEON) && defined(__ANDROID__)
#include <arm_neon.h>
#endif

namespace mace {
namespace ops {

template <RuntimeType D, typename SrcType>
class CastOp : public Operation {
 public:
  explicit CastOp(OpConstructContext *context)
      : Operation(context) {}

  MaceStatus Run(OpContext *context) override {
    LOG(INFO)<<"op name: "<<this->debug_def().name();
    auto *debug_data = this->Input(0)->data<int32_t>();
    LOG(INFO)<<"cast input: "<<debug_data[0]<<" "<<debug_data[1]<<" "<<debug_data[2]<<" "<<debug_data[3]<<" "<<debug_data[4]<<" "<<debug_data[5];
    MACE_UNUSED(context);
    const Tensor *input = this->Input(INPUT);
    Tensor *output = this->Output(OUTPUT);
    MACE_RETURN_IF_ERROR(output->ResizeLike(input))

    auto dst_dtype = output->dtype();
	LOG(INFO)<<"cast dst_dtype:"<<dst_dtype;
#define MACE_CAST_COPY \
    auto output_data = output->mutable_data<T>();                       \
    auto input_data = input->data<SrcType>();                           \
    for (index_t i = 0; i < output->size(); ++i) {                      \
      output_data[i] = static_cast<T>(input_data[i]);                   \
    }

    MACE_RUN_WITH_TYPE_ENUM(dst_dtype, MACE_CAST_COPY);
    auto *debug_data2 = this->Output(0)->data<float>();
    LOG(INFO)<<"cast output: "<<debug_data2[0]<<" "<<debug_data2[1]<<" "<<debug_data2[2]<<" "<<debug_data2[3]<<" "<<debug_data2[4]<<" "<<debug_data2[5];
    return MaceStatus::MACE_SUCCESS;
  }

 private:
  MACE_OP_INPUT_TAGS(INPUT);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

void RegisterCast(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "Cast", CastOp, RuntimeType::RT_CPU, float);
  MACE_REGISTER_OP(op_registry, "Cast", CastOp, RuntimeType::RT_CPU, int32_t);
#if defined(MACE_ENABLE_NEON) && defined(__ANDROID__)
  MACE_REGISTER_OP(op_registry, "Cast", CastOp,
                   RuntimeType::RT_CPU, float16_t);
#endif
}

}  // namespace ops
}  // namespace mace
