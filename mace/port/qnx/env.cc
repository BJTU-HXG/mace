#include "mace/port/qnx/env.h"

#include <errno.h>

#include <sys/time.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

#include "mace/port/env.h"
#include "mace/port/posix/file_system.h"
#include "mace/port/posix/time.h"
#include "mace/utils/logging.h"

namespace mace {
namespace port {

int64_t QnxEnv::NowMicros() {
	return mace::port::posix::NowMicros();
}

MaceStatus QnxEnv::AdviseFree(void *addr, size_t length) {
	MACE_UNUSED(addr);
	MACE_UNUSED(length);
  return MaceStatus::MACE_SUCCESS;
}

FileSystem *QnxEnv::GetFileSystem() {
	return &posix_file_system_;
}

MaceStatus QnxEnv::GetCPUMaxFreq(std::vector<float> *max_freqs) {
	MACE_UNUSED(max_freqs);
	return MaceStatus::MACE_UNSUPPORTED;
}

MaceStatus QnxEnv::SchedSetAffinity(const std::vector<size_t> &cpu_ids) {
	MACE_UNUSED(cpu_ids);
  return MaceStatus::MACE_SUCCESS;
}

LogWriter *QnxEnv::GetLogWriter() {
	return &log_writer_;
}

std::vector<std::string> QnxEnv::GetBackTraceUnsafe(int max_steps) {
	return std::vector<std::string>();
}

std::unique_ptr<MallocLogger> QnxEnv::NewMallocLogger(
		std::ostringstream *oss,
		const std::string &name) {
	return make_unique<MallocLogger>();
}

Env *Env::Default() {
  static QnxEnv qnx_env;
  return &qnx_env;
}

}  // namespace port
}  // namespace mace