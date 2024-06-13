// Copyright 2019 The MACE Authors. All Rights Reserved.
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

#include <algorithm>
#include <cmath>
#include "mace/ops/delegator/activation.h"

namespace mace {
namespace ops {
namespace ref {

template<typename T>
class Activation : public delegator::Activation {
 public:
  explicit Activation(const delegator::ActivationParam &param)
      : delegator::Activation(param) {}
  ~Activation() = default;

  MaceStatus Compute(const OpContext *context, const Tensor *input,
                     Tensor *output) override;

 private:
  void DoActivation(const OpContext *context, const Tensor *input,
                    Tensor *output);
};

template<typename T>
MaceStatus Activation<T>::Compute(const OpContext *context,
                                  const Tensor *input,
                                  Tensor *output) {
  if (input != output) {
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));
    DoActivation(context, input, output);
  } else {
    DoActivation(context, input, output);
  }
  //LOG(INFO) << "Activation is runningggggggggg";
  //LOG(INFO) << "type is:" << type_ ;
  return MaceStatus::MACE_SUCCESS;
}

template<typename T>
void Activation<T>::DoActivation(const OpContext *context,
                                 const Tensor *input,
                                 Tensor *output) {
  MACE_UNUSED(context);
  auto input_ptr = input->data<T>();
  auto output_ptr = output->mutable_data<T>();
  const index_t size = input->size();

  switch (type_) {
    case RELU: {
      for (index_t i = 0; i < size; ++i) {
        *output_ptr++ = std::max(0.f, *input_ptr++);
      }

      break;
    }

    case RELUX: {
      for (index_t i = 0; i < size; ++i) {
        *output_ptr++ = std::max(0.f, std::min(limit_, *input_ptr++));
      }

      break;
    }

    case GELU: {
      for (index_t i = 0; i < size; ++i) {
          const auto in_val = *input_ptr++;
          *output_ptr++ = 0.5 * in_val * (1 + std::tanh(std::sqrt(2/M_PI) * (in_val + 0.044715 * std::pow(in_val, 3))));
      }
      //LOG(INFO) << "GELU is runningggggggggg" ;
      break;
    }
    
    case LEAKYRELU: {
      for (index_t i = 0; i < size; ++i) {
        *output_ptr =
            std::max<float>(*input_ptr, 0.f)
                + std::min(*input_ptr, 0.f) * activation_coefficient_;
        ++input_ptr;
        ++output_ptr;
      }

      break;
    }

    case TANH: {
      for (index_t i = 0; i < size; ++i) {
        *output_ptr++ = std::tanh(*input_ptr++);
      }

      break;
    }

    case SIGMOID: {
      for (index_t i = 0; i < size; ++i) {
        *output_ptr++ = 1 / (1 + std::exp(-(*input_ptr++)));
      }
      break;
    }

    case HARDSIGMOID: {
      for (index_t i = 0; i < size; ++i) {
        *output_ptr++ = std::max(0.0f, std::min(1.0f,
          hardsigmoid_alpha_ * (*input_ptr++) + hardsigmoid_beta_));
      }
      break;
    }

    case ELU: {
      for (index_t i = 0; i < input->size(); ++i) {
        const auto in_val = *input_ptr++;
        if (in_val < 0) {
          *output_ptr = (std::exp(in_val) - 1) * activation_coefficient_;
        } else {
          *output_ptr = in_val;
        }
        output_ptr++;
      }
      break;
    }

    case NOOP:break;

    default:MACE_NOT_IMPLEMENTED;
  }
}

void RegisterActivationDelegator(OpDelegatorRegistry *registry) {
  MACE_REGISTER_DELEGATOR(
      registry, Activation<float>, delegator::ActivationParam,
      MACE_DELEGATOR_KEY(Activation, RuntimeType::RT_CPU,
                         float, ImplType::REF));

  MACE_REGISTER_BF16_DELEGATOR(
      registry, Activation<BFloat16>, delegator::ActivationParam,
      MACE_DELEGATOR_KEY(Activation, RuntimeType::RT_CPU,
                         BFloat16, ImplType::REF));

  MACE_REGISTER_FP16_DELEGATOR(
      registry, Activation<float16_t>, delegator::ActivationParam,
      MACE_DELEGATOR_KEY(Activation, RuntimeType::RT_CPU,
                         float16_t, ImplType::REF));
}

}  // namespace ref
}  // namespace ops
}  // namespace mace
