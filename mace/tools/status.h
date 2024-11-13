#pragma once
namespace AISRT {
namespace INFER {

// align to aisrt
enum StatusCode {
  StatusCodeInvalid = -1,
  StatusCodeOK = 0,
  StatusInferenceError = 2000,
  StatusInferenceModelLoadFailed = 2001,
  StatusInferenceOpenFailed = 2002,
  StatusInferenceCfgJsonError = 2003,
};

}
}