/**
 *Copyright (c) 2022 AIS of NIO All rights reserved.
 *
 *Created by weisheng.han on 2022/06/30
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include "status.h"

namespace AISRT {
namespace INFER {
typedef void* ENGINE_HANDLE;
typedef void* IHANDLE;        // infer_handle

using namespace std;

// 框架类型
enum AisFrameworkType {
  kAisInferNone = -1,
  kAisInferOnnx,
  kAisInferMnn,
  kAisInferAnn,    // ais-nn
  kAisInferSnpe,
  kAisInferQnn,
  kAisInferHexagon
};

// 数据类型
enum AisInferElementDataType {
  kAisInferElementDataNone = -1,
  kAisInferElementDataFloat,
  kAisInferElementDataUint8,
  kAisInferElementDataInt8,
  kAisInferElementDataUint16,
  kAisInferElementDataInt16,
  kAisInferElementDataInt32,
  kAisInferElementDataInt64,
  kAisInferElementDataFloat16,
  kAisInferElementDataDouble,
  kAisInferElementDataUint32,
  kAisInferElementDataUint64,
  kAisInferElementDataBfloat16,
  kAisInferElementDataBool,
};

// 后端类型
enum AisInferForwardType {
  kAisinferForwardCpu,
  kAisinferForwardCuda = 1
};


// 推理配置
struct AisInferConfig {
  AisFrameworkType framework_type;
  AisInferElementDataType data_type;
  AisInferForwardType forward_type;
  int threadnum;

  // using for ann
  string params;
  AisInferConfig():threadnum(1) {
  }
  // for SNPE
  std::vector<std::string> output_names;
};

// 算子参数
struct AisTensorParam {
  AisInferElementDataType data_type;
  string name;
  vector<int64_t> shape;
  const void *data;
};

class AisInference {
 public:
 /**
  *@brief :初始化引擎，加载模型资源
  *@param  model_name:模型名称
  *@param  config:配置信息，包括：框架类型/后端类型/输入数据类型等
  *@return StatusCode :错误码
  */
  StatusCode Load(const string &model_name, const AisInferConfig &config);

  /**
   *@brief :退出引擎，释放引擎资源
   *@return StatusCode :错误码
   */
  StatusCode Unload();

  /**
   *@brief :获取推理版本信息
   *@return string :版本信息，引擎未初始化时，为空
   */
  string GetVersion();

  /**
   *@brief Create a Instance object:初始化一个处理实例
   *@param  instance:实例id
   *@return StatusCode :错误码
   */
  StatusCode CreateInstance(IHANDLE *instance);

  /**
   *@brief :销毁一个处理实例
   *@param  instance:实例id
   *@return StatusCode :错误码
   */
  StatusCode DestroyInstance(IHANDLE instance);

  /**
   *@brief :进行推理
   *@param  instance:进行推理的实例id
   *@param  input_tensor:输入参数，算子名称/维度/数据等
   *@param  output_tensor:通过该参数设置输出算子名称，数据类型等，处理完成后更新数据指针及维度
   *@return StatusCode :错误码
   */
  StatusCode Run(IHANDLE instance, const vector<AisTensorParam> &input_tensors, vector<AisTensorParam> &output_tensors);

  /**
   *@brief :Reset Cache for inhouse inference
   *@param  instance:进行推理的实例id
   *@return StatusCode :错误码
   */
  StatusCode Reset(IHANDLE instance);

  /**
   *@brief Get the Input Count object:获取输入算子个数
   *@param  instance:实例id
   *@return int :算子个数
   */
  int GetInputCount() const;

  /**
   *@brief Get the Output Count object:获取输出算子个数
   *@param  instance:实例id
   *@return int :算子个数
   */
  int GetOutputCount() const;

  /**
   *@brief Get the Input Names object:获取输入算子名称
   *@param  instance:实例id
   *@return vector<string> :算子名称列表
   */
  vector<string> GetInputNames() const;


  /**
   *@brief Get the Output Names object:获取输出算子名称
   *@param  instance:实例id
   *@return vector<string> :算子名称列表
   */
  vector<string> GetOutputNames() const;

  /**
   *@brief Get Metadata from model:从模型获取meta信息
   *@param  instance:实例id
   *@return unordered_map<string, int>:metadata信息列表
   */
  unordered_map<string, int> GetMetaData() const;

  int64_t CalcSize(vector<int64_t> &shape) const;

 private:
  void *info_;
};

}  // namespace SPEECH
}  // namespace AISRT
