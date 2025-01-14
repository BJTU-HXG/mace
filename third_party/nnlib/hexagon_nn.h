/*
 * Copyright (c) 2016-2019, The Linux Foundation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted (subject to the limitations in the
 * disclaimer below) provided that the following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *    * Neither the name of The Linux Foundation nor the names of its
 *      contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 * NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
 * GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
 * HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef THIRD_PARTY_NNLIB_HEXAGON_NN_H_
#define THIRD_PARTY_NNLIB_HEXAGON_NN_H_

#ifndef __QAIC_HEADER
#define __QAIC_HEADER(ff) ff
#endif //__QAIC_HEADER

#ifndef __QAIC_HEADER_EXPORT
#define __QAIC_HEADER_EXPORT
#endif // __QAIC_HEADER_EXPORT

#ifndef __QAIC_HEADER_ATTRIBUTE
#define __QAIC_HEADER_ATTRIBUTE
#endif // __QAIC_HEADER_ATTRIBUTE

#ifndef __QAIC_IMPL
#define __QAIC_IMPL(ff) ff
#endif //__QAIC_IMPL

#ifndef __QAIC_IMPL_EXPORT
#define __QAIC_IMPL_EXPORT
#endif // __QAIC_IMPL_EXPORT

#ifndef __QAIC_IMPL_ATTRIBUTE
#define __QAIC_IMPL_ATTRIBUTE
#endif // __QAIC_IMPL_ATTRIBUTE
#ifdef __cplusplus
extern "C" {
#endif
#if !defined(__QAIC_STRING1_OBJECT_DEFINED__) && !defined(__STRING1_OBJECT__)
#define __QAIC_STRING1_OBJECT_DEFINED__
#define __STRING1_OBJECT__
typedef struct _cstring1_s {
   char* data;
   int dataLen;
} _cstring1_t;

#endif /* __QAIC_STRING1_OBJECT_DEFINED__ */
typedef struct hexagon_nn_input hexagon_nn_input;
struct hexagon_nn_input {
   unsigned int src_id;
   unsigned int output_idx;
};
typedef struct hexagon_nn_output hexagon_nn_output;
struct hexagon_nn_output {
   unsigned int rank;
   unsigned int max_sizes[8];
   unsigned int elementsize;
   int zero_offset;
   float stepsize;
};
typedef struct hexagon_nn_perfinfo hexagon_nn_perfinfo;
struct hexagon_nn_perfinfo {
   unsigned int node_id;
   unsigned int executions;
   unsigned int counter_lo;
   unsigned int counter_hi;
};
typedef int hexagon_nn_nn_id;
typedef struct hexagon_nn_initinfo hexagon_nn_initinfo;
struct hexagon_nn_initinfo {
  int priority;
};
enum hexagon_nn_padding_type {
   NN_PAD_NA,
   NN_PAD_SAME,
   NN_PAD_VALID,
   NN_PAD_MIRROR_REFLECT,
   NN_PAD_MIRROR_SYMMETRIC,
   NN_PAD_SAME_CAFFE,
   _32BIT_PLACEHOLDER_hexagon_nn_padding_type = 0x7fffffff
};
typedef enum hexagon_nn_padding_type hexagon_nn_padding_type;
enum hexagon_nn_corner_type {
   NN_CORNER_RELEASE,
   NN_CORNER_TURBO,
   NN_CORNER_NOMPLUS,
   NN_CORNER_NOMINAL,
   NN_CORNER_SVSPLUS,
   NN_CORNER_SVS,
   NN_CORNER_SVS2,
   _32BIT_PLACEHOLDER_hexagon_nn_corner_type = 0x7fffffff
};
typedef enum hexagon_nn_corner_type hexagon_nn_corner_type;
enum hexagon_nn_dcvs_type {
   NN_DCVS_DEFAULT,
   NN_DCVS_ENABLE,
   NN_DCVS_DISABLE,
   _32BIT_PLACEHOLDER_hexagon_nn_dcvs_type = 0x7fffffff
};
typedef enum hexagon_nn_dcvs_type hexagon_nn_dcvs_type;
typedef struct hexagon_nn_tensordef hexagon_nn_tensordef;
struct hexagon_nn_tensordef {
   unsigned int batches;
   unsigned int height;
   unsigned int width;
   unsigned int depth;
   unsigned char* data;
   int dataLen;
   unsigned int data_valid_len;
   unsigned int unused;
};
enum hexagon_nn_execute_result {
  NN_EXECUTE_SUCCESS,
  NN_EXECUTE_ERROR,
  NN_EXECUTE_BUFFER_SIZE_ERROR,
  _32BIT_PLACEHOLDER_hexagon_nn_execute_result = 0x7fffffff
};
typedef enum hexagon_nn_execute_result hexagon_nn_execute_result;
typedef struct hexagon_nn_execute_info hexagon_nn_execute_info;
struct hexagon_nn_execute_info {
  hexagon_nn_execute_result result;
  unsigned char* extraInfo;
  int extraInfoLen;
  unsigned int extraInfoValidLen;
};
enum hexagon_nn_option_type {
  NN_OPTION_NOSUCHOPTION,
  NN_OPTION_SCALAR_THREADS,
  NN_OPTION_HVX_THREADS,
  NN_OPTION_VTCM_REQ,
  NN_OPTION_ENABLE_GRAPH_PRINT,
  NN_OPTION_ENABLE_TENSOR_PRINT,
  NN_OPTION_TENSOR_PRINT_FILTER,
  NN_OPTION_HAP_MEM_GROW_SIZE,
  NN_OPTION_ENABLE_CONST_PRINT,
  NN_OPTION_LASTPLUSONE,
  _32BIT_PLACEHOLDER_hexagon_nn_option_type = 0x7fffffff
};
typedef enum hexagon_nn_option_type hexagon_nn_option_type;
typedef struct hexagon_nn_uint_option hexagon_nn_uint_option;
struct hexagon_nn_uint_option {
  unsigned int option_id;
  unsigned int uint_value;
};
typedef struct hexagon_nn_string_option hexagon_nn_string_option;
struct hexagon_nn_string_option {
  unsigned int option_id;
  char string_data[256];
};
/**
 * @brief Statistics of user heap memory
 */
struct HAP_mem_stats {
   uint64_t bytes_free; /** number of bytes free from all the segments,
                        * may not be available for a single alloc
                        */
   uint64_t bytes_used; /** number of bytes used */
   uint64_t seg_free;   /** number of segments free */
   uint64_t seg_used;   /** number of segments used */
   uint64_t min_grow_bytes; /** minimum number of bytes to grow the heap by when creating a new segment */
};
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_config)(void) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_config_with_options)(const hexagon_nn_uint_option* uint_options, int uint_optionsLen, const hexagon_nn_string_option* string_options, int string_optionsLen) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_graph_config)(hexagon_nn_nn_id id, const hexagon_nn_uint_option* uint_options, int uint_optionsLen, const hexagon_nn_string_option* string_options, int string_optionsLen) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_get_dsp_offset)(unsigned int* libhexagon_addr, unsigned int* fastrpc_shell_addr) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_init)(hexagon_nn_nn_id* g) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_set_debug_level)(hexagon_nn_nn_id id, int level) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_snpprint)(hexagon_nn_nn_id id, unsigned char* buf, int bufLen) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_getlog)(hexagon_nn_nn_id id, unsigned char* buf, int bufLen) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_append_node)(hexagon_nn_nn_id id, unsigned int node_id, unsigned int operation, hexagon_nn_padding_type padding, const hexagon_nn_input* inputs, int inputsLen, const hexagon_nn_output* outputs, int outputsLen) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_append_const_node)(hexagon_nn_nn_id id, unsigned int node_id, unsigned int batches, unsigned int height, unsigned int width, unsigned int depth, const unsigned char* data, int dataLen) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_append_empty_const_node)(hexagon_nn_nn_id id, unsigned int node_id, unsigned int batches, unsigned int height, unsigned int width, unsigned int depth, unsigned int size) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_populate_const_node)(hexagon_nn_nn_id id, unsigned int node_id, const unsigned char* data, int dataLen, unsigned int target_offset) __QAIC_HEADER_ATTRIBUTE;
typedef struct hexagon_nn_op_node hexagon_nn_op_node;
struct hexagon_nn_op_node {
  unsigned int node_id;
  unsigned int operation;
  hexagon_nn_padding_type padding;
  hexagon_nn_input* inputs;
  int inputsLen;
  hexagon_nn_output* outputs;
  int outputsLen;
};
typedef struct hexagon_nn_const_node hexagon_nn_const_node;
struct hexagon_nn_const_node {
  unsigned int node_id;
  hexagon_nn_tensordef tensor;
};
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_append_node_list)(hexagon_nn_nn_id id, const hexagon_nn_op_node* ops, int opsLen) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_append_const_node_list)(hexagon_nn_nn_id id, const hexagon_nn_const_node* consts, int constsLen) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_prepare)(hexagon_nn_nn_id id) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_execute)(hexagon_nn_nn_id id, unsigned int batches_in, unsigned int height_in, unsigned int width_in, unsigned int depth_in, const unsigned char* data_in, int data_inLen, unsigned int* batches_out, unsigned int* height_out, unsigned int* width_out, unsigned int* depth_out, unsigned char* data_out, int data_outLen, unsigned int* data_len_out) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_teardown)(hexagon_nn_nn_id id) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_variable_read)(hexagon_nn_nn_id id, unsigned int node_id, int output_index, unsigned int* batches_out, unsigned int* height_out, unsigned int* width_out, unsigned int* depth_out, unsigned char* data_out, int data_outLen, unsigned int* data_len_out) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_variable_write)(hexagon_nn_nn_id id, unsigned int node_id, int output_index, unsigned int batches, unsigned int height, unsigned int width, unsigned int depth, const unsigned char* data_in, int data_inLen) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_variable_write_flat)(hexagon_nn_nn_id id, unsigned int node_id, int output_index, const unsigned char* data_in, int data_inLen) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_set_powersave_level)(unsigned int level) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_set_powersave_details)(hexagon_nn_corner_type corner, hexagon_nn_dcvs_type dcvs, unsigned int latency) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_set_clocks)(hexagon_nn_corner_type corner, hexagon_nn_dcvs_type dcvs, unsigned int latency) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_remove_clocks)() __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_get_perfinfo)(hexagon_nn_nn_id id, hexagon_nn_perfinfo* info_out, int info_outLen, unsigned int* n_items) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_reset_perfinfo)(hexagon_nn_nn_id id, unsigned int event) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_last_execution_cycles)(hexagon_nn_nn_id id, unsigned int* cycles_lo, unsigned int* cycles_hi) __QAIC_HEADER_ATTRIBUTE;
// __QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_get_mem_stats)(HAP_mem_stats* stats) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_version)(int* ver) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_op_name_to_id)(const char* name, unsigned int* node_id) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_op_id_to_name)(unsigned int node_id, char* name, int nameLen) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_get_num_nodes_in_graph)(hexagon_nn_nn_id id, unsigned int* num_nodes) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_disable_dcvs)(void) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_GetHexagonBinaryVersion)(int* ver) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_PrintLog)(const unsigned char* buf, int bufLen) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_execute_new)(hexagon_nn_nn_id id, const hexagon_nn_tensordef* inputs, int inputsLen, hexagon_nn_tensordef* outputs, int outputsLen) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_execute_with_info)(hexagon_nn_nn_id id, const hexagon_nn_tensordef* inputs, int inputsLen, hexagon_nn_tensordef* outputs, int outputsLen, hexagon_nn_execute_info* execute_info) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_init_with_info)(hexagon_nn_nn_id* g, const hexagon_nn_initinfo* info) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_get_nodetype)(hexagon_nn_nn_id graph_id, hexagon_nn_nn_id node_id, unsigned int* node_type) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_multi_execution_cycles)(hexagon_nn_nn_id id, unsigned int* cycles_lo, unsigned int* cycles_hi) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_get_power)(int type) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_set_graph_option)(hexagon_nn_nn_id id, const char* name, int value) __QAIC_HEADER_ATTRIBUTE;
__QAIC_HEADER_EXPORT int __QAIC_HEADER(hexagon_nn_populate_graph)(hexagon_nn_nn_id id, const unsigned char* graph_data, int graph_dataLen) __QAIC_HEADER_ATTRIBUTE;

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_NNLIB_HEXAGON_NN_H_
