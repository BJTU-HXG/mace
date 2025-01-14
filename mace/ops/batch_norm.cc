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

#include <memory>
#include <string>
#include <vector>

#include "mace/core/ops/operator.h"
#include "mace/core/registry/ops_registry.h"
#include "mace/ops/activation.h"
#include "mace/ops/delegator/activation.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/image/batch_norm.h"
#include "mace/runtimes/opencl/transform/buffer_transformer.h"
#endif  // MACE_ENABLE_OPENCL
#include "mace/utils/memory.h"

namespace mace {
namespace ops {

template<RuntimeType D, class T>
class BatchNormOp;

template<class T>
class BatchNormOp<RuntimeType::RT_CPU, T> : public Operation {
 public:
  explicit BatchNormOp(OpConstructContext *context)
      : Operation(context),
        epsilon_(Operation::GetOptionalArg<float>("epsilon",
                                                  static_cast<float>(1e-4))),
        activation_delegator_(
            delegator::Activation::Create(
                context->workspace(),
                MACE_DELEGATOR_KEY(Activation, RuntimeType::RT_CPU,
                                   T, kCpuImplType),
                delegator::ActivationParam(
                    ops::StringToActivationType(
                        Operation::GetOptionalArg<std::string>("activation",
                                                               "NOOP")),
                    Operation::GetOptionalArg<float>("max_limit", 0.0f),
                    Operation::GetOptionalArg<float>("activation_coefficient",
                                                     0.0f),
                    Operation::GetOptionalArg<float>("hardsigmoid_alpha", 0.f),
                    Operation::GetOptionalArg<float>("hardsigmoid_beta", 0.f)))) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    bool not_folded = this->InputSize() == 5;
    const Tensor *input = this->Input(INPUT);
    const Tensor *scale = this->Input(SCALE);
    const Tensor *offset = this->Input(OFFSET);

    const_cast<Tensor*> (input)->to4dim();
    MACE_CHECK(input->dim_size() == 4, "input must be 4-dimensional. ",
               input->dim_size());
    MACE_CHECK(scale->dim_size() == 1, "scale must be 1-dimensional. ",
               scale->dim_size());
    MACE_CHECK(offset->dim_size() == 1, "offset must be 1-dimensional. ",
               offset->dim_size());

    Tensor *output = this->Output(OUTPUT);
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));

    // Batch normalization in the paper https://arxiv.org/abs/1502.03167 .
    // The calculation formula for inference is
    // Y = \frac{ \scale } { \sqrt{var+\variance_epsilon} } * X +
    //          ( \offset - \frac { \scale * mean } {
    //          \sqrt{var+\variance_epsilon} }
    // new_scale = \frac{ \scale } { \sqrt{var+\variance_epsilon} }
    // new_offset = \offset - mean * common_val;
    // Y = new_scale * X + new_offset;
    const index_t batch = input->dim(0);
    const index_t channels = input->dim(1);
    const index_t height = input->dim(2);
    const index_t width = input->dim(3);

    utils::ThreadPool &thread_pool = context->runtime()->thread_pool();

    {
      const T *input_ptr = input->data<T>();
      const T *scale_ptr = scale->data<T>();
      const T *offset_ptr = offset->data<T>();
      T *output_ptr = output->mutable_data<T>();

      std::vector<T> new_scale;
      std::vector<T> new_offset;
      if (not_folded) {
        const Tensor *mean = this->Input(MEAN);
        const Tensor *var = this->Input(VAR);
        MACE_CHECK(mean->dim_size() == 1, "mean must be 1-dimensional. ",
                   mean->dim_size());
        MACE_CHECK(var->dim_size() == 1, "var must be 1-dimensional. ",
                   var->dim_size());
        new_scale.resize(channels);
        new_offset.resize(channels);
        const T *mean_ptr = mean->data<T>();
        const T *var_ptr = var->data<T>();

        thread_pool.Compute1D([=, &new_scale, &new_offset](index_t start,
                                                           index_t end,
                                                           index_t step) {
          for (index_t c = start; c < end; c += step) {
            new_scale[c] = scale_ptr[c] / std::sqrt(var_ptr[c] + epsilon_);
            new_offset[c] = offset_ptr[c] - mean_ptr[c] * new_scale[c];
          }
        }, 0, channels, 1);
      }

      const T *scale_data = not_folded ? new_scale.data() : scale_ptr;
      const T *offset_data = not_folded ? new_offset.data() : offset_ptr;

      index_t channel_size = height * width;
      index_t batch_size = channels * channel_size;

      thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                                index_t start1, index_t end1, index_t step1) {
        for (index_t b = start0; b < end0; b += step0) {
          for (index_t c = start1; c < end1; c += step1) {
            index_t offset = b * batch_size + c * channel_size;
            for (index_t hw = 0; hw < height * width; ++hw) {
              output_ptr[offset + hw] =
                  scale_data[c] * input_ptr[offset + hw] + offset_data[c];
            }
          }
        }
      }, 0, batch, 1, 0, channels, 1);
    }

    activation_delegator_->Compute(context, output, output);

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  float epsilon_;
  std::unique_ptr<delegator::Activation> activation_delegator_;

 protected:
  MACE_OP_INPUT_TAGS(INPUT, SCALE, OFFSET, MEAN, VAR);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

#ifdef MACE_ENABLE_OPENCL
template<>
class BatchNormOp<RuntimeType::RT_OPENCL, float> : public Operation {
 public:
  explicit BatchNormOp(OpConstructContext *context)
      : Operation(context) {
    float epsilon = Operation::GetOptionalArg<float>(
        "epsilon", static_cast<float>(1e-4));
    ActivationType activation = ops::StringToActivationType(
        Operation::GetOptionalArg<std::string>("activation", "NOOP"));
    float relux_max_limit = Operation::GetOptionalArg<float>("max_limit", 0.0f);
    float activation_coefficient = Operation::GetOptionalArg<float>(
        "activation_coefficient", 0.0f);
    MemoryType mem_type;
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      mem_type = MemoryType::GPU_IMAGE;
      kernel_ = make_unique<opencl::image::BatchNormKernel>(
          epsilon, activation, relux_max_limit, activation_coefficient);
    } else {
      MACE_NOT_IMPLEMENTED;
    }
    // Transform filters
    int input_size = operator_def_->input_size();
    for (int i = 1; i < input_size; ++i) {
      const Tensor *input_tensor = context->workspace()->GetTensor(
          operator_def_->input(i));
      MACE_CHECK(input_tensor != nullptr);
      MACE_CHECK(TransformFilter(
          context,
          operator_def_.get(),
          i,
          BufferContentType::ARGUMENT,
          mem_type) == MaceStatus::MACE_SUCCESS);
    }
  }
  MaceStatus Run(OpContext *context) override {
    bool not_folded = this->InputSize() == 5;
    const Tensor *input = this->Input(INPUT);
    const Tensor *scale = this->Input(SCALE);
    const Tensor *offset = this->Input(OFFSET);
    const Tensor *mean = not_folded ? this->Input(MEAN) : nullptr;
    const Tensor *var = not_folded ? this->Input(VAR) : nullptr;

    MACE_CHECK(input->dim_size() == 4, "input must be 4-dimensional. ",
               input->dim_size());
    MACE_CHECK(scale->dim_size() == 1, "scale must be 1-dimensional. ",
               scale->dim_size());
    MACE_CHECK(offset->dim_size() == 1, "offset must be 1-dimensional. ",
               offset->dim_size());
    if (not_folded) {
      MACE_CHECK(mean->dim_size() == 1, "mean must be 1-dimensional. ",
                 mean->dim_size());
      MACE_CHECK(var->dim_size() == 1, "var must be 1-dimensional. ",
                 var->dim_size());
    }

    Tensor *output = this->Output(OUTPUT);
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));

    return kernel_->Compute(context, input, scale, offset, mean,
                            var, output);
  }

 protected:
  BufferContentType GetInputTensorContentType(size_t idx) const override {
    if (idx > 0) {
      return BufferContentType::ARGUMENT;
    }
    return Operation::GetInputTensorContentType(idx);
  }

 private:
  std::unique_ptr<OpenCLBatchNormKernel> kernel_;

 protected:
  MACE_OP_INPUT_TAGS(INPUT, SCALE, OFFSET, MEAN, VAR);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};
#endif  // MACE_ENABLE_OPENCL

void RegisterBatchNorm(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "BatchNorm", BatchNormOp,
                   RuntimeType::RT_CPU, float);
  MACE_REGISTER_BF16_OP(op_registry, "BatchNorm",
                        BatchNormOp, RuntimeType::RT_CPU);
  MACE_REGISTER_GPU_OP(op_registry, "BatchNorm", BatchNormOp);
}

}  // namespace ops
}  // namespace mace
