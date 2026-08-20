// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/bf16_gemm.h"

#include <hip/hip_bfloat16.h>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace aima {
namespace {

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
  }
}

void check_blas(hipblasStatus_t status, const char* operation) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": status=" +
                             std::to_string(static_cast<int>(status)));
  }
}

__global__ void fill_bf16_kernel(hip_bfloat16* values, std::size_t count,
                                 float value) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) values[index] = hip_bfloat16(value);
}

void fill_bf16(void* pointer, std::size_t count, float value) {
  constexpr unsigned threads = 256;
  const unsigned blocks = static_cast<unsigned>((count + threads - 1) / threads);
  hipLaunchKernelGGL(fill_bf16_kernel, dim3(blocks), dim3(threads), 0, nullptr,
                     static_cast<hip_bfloat16*>(pointer), count, value);
  check_hip(hipGetLastError(), "fill_bf16_kernel");
}

class DeviceAllocation {
 public:
  explicit DeviceAllocation(std::size_t bytes) : bytes_(bytes) {
    check_hip(hipMalloc(&pointer_, bytes), "hipMalloc");
  }
  ~DeviceAllocation() {
    if (pointer_) {
      const hipError_t ignored = hipFree(pointer_);
      static_cast<void>(ignored);
    }
  }
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;
  void* get() const { return pointer_; }
  std::size_t bytes() const { return bytes_; }

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0;
};

class Event {
 public:
  Event() { check_hip(hipEventCreate(&event_), "hipEventCreate"); }
  ~Event() {
    if (event_) {
      const hipError_t ignored = hipEventDestroy(event_);
      static_cast<void>(ignored);
    }
  }
  operator hipEvent_t() const { return event_; }

 private:
  hipEvent_t event_ = nullptr;
};

}  // namespace

struct Bf16GemmPlan::Impl {
  std::size_t m = 0;
  std::size_t n = 0;
  std::size_t k = 0;
  hipblasLtHandle_t handle = nullptr;
  hipblasLtMatmulDesc_t operation = nullptr;
  hipblasLtMatrixLayout_t a_layout = nullptr;
  hipblasLtMatrixLayout_t b_layout = nullptr;
  hipblasLtMatrixLayout_t c_layout = nullptr;
  hipblasLtMatrixLayout_t d_layout = nullptr;
  hipblasLtMatmulPreference_t preference = nullptr;
  hipblasLtMatmulAlgo_t algorithm{};
  void* workspace = nullptr;
  std::size_t workspace_bytes = 0;
  int heuristic_count = 0;
  int library_version = 0;
  bool right_operand_is_transposed = false;
  bool torch_n1_layout = false;
  bool bias_epilogue = false;
  bool owns_handle = false;
  bool owns_operation = false;

  ~Impl() { release(); }

  void release() noexcept {
    if (workspace) {
      const hipError_t ignored = hipFree(workspace);
      static_cast<void>(ignored);
      workspace = nullptr;
    }
    if (preference) hipblasLtMatmulPreferenceDestroy(preference);
    if (d_layout) hipblasLtMatrixLayoutDestroy(d_layout);
    if (c_layout) hipblasLtMatrixLayoutDestroy(c_layout);
    if (b_layout) hipblasLtMatrixLayoutDestroy(b_layout);
    if (a_layout) hipblasLtMatrixLayoutDestroy(a_layout);
    if (owns_operation && operation) hipblasLtMatmulDescDestroy(operation);
    if (owns_handle && handle) hipblasLtDestroy(handle);
    preference = nullptr;
    d_layout = c_layout = b_layout = a_layout = nullptr;
    operation = nullptr;
    handle = nullptr;
  }
};

Bf16GemmPlan::Bf16GemmPlan(std::size_t m, std::size_t n, std::size_t k,
                           std::size_t workspace_limit_bytes,
                           bool right_operand_is_transposed,
                           bool bias_epilogue,
                           const Bf16GemmPlan* algorithm_source)
    : impl_(std::make_unique<Impl>()) {
  if (m == 0 || n == 0 || k == 0 || workspace_limit_bytes == 0) {
    throw std::invalid_argument("BF16 GEMM dimensions and workspace must be non-zero");
  }
  if (algorithm_source != nullptr && bias_epilogue) {
    throw std::invalid_argument(
        "biased BF16 GEMM plans cannot borrow an algorithm source");
  }
  if (algorithm_source != nullptr &&
      (!algorithm_source->impl_ || algorithm_source->impl_->m < m ||
       algorithm_source->impl_->n != n ||
       algorithm_source->impl_->k != k ||
       algorithm_source->impl_->right_operand_is_transposed !=
           right_operand_is_transposed ||
       algorithm_source->impl_->bias_epilogue != bias_epilogue)) {
    throw std::invalid_argument(
        "BF16 GEMM algorithm source geometry is incompatible");
  }
  impl_->m = m;
  impl_->n = n;
  impl_->k = k;
  impl_->right_operand_is_transposed = right_operand_is_transposed;
  impl_->torch_n1_layout = right_operand_is_transposed && n == 1;
  impl_->bias_epilogue = bias_epilogue;
  try {
    if (algorithm_source != nullptr) {
      // A derived logical-M plan has the same operation geometry as its
      // resident bucket source. Reuse the source's immutable handle and
      // operation instead of repeatedly initializing hipBLASLt on the request
      // path; only the exact-M matrix layouts below are plan-local.
      impl_->handle = algorithm_source->impl_->handle;
      impl_->operation = algorithm_source->impl_->operation;
      impl_->library_version = algorithm_source->impl_->library_version;
    } else {
      check_blas(hipblasLtCreate(&impl_->handle), "hipblasLtCreate");
      impl_->owns_handle = true;
      check_blas(hipblasLtGetVersion(impl_->handle, &impl_->library_version),
                 "hipblasLtGetVersion");
      check_blas(hipblasLtMatmulDescCreate(&impl_->operation,
                                           HIPBLAS_COMPUTE_32F, HIP_R_32F),
                 "hipblasLtMatmulDescCreate");
      impl_->owns_operation = true;
      if (impl_->bias_epilogue) {
        const hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_BIAS;
        const hipDataType bias_type = HIP_R_16BF;
        check_blas(hipblasLtMatmulDescSetAttribute(
                       impl_->operation, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                       &epilogue, sizeof(epilogue)),
                   "hipblasLtMatmulDescSetAttribute bias epilogue");
        check_blas(hipblasLtMatmulDescSetAttribute(
                       impl_->operation,
                       HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                       &bias_type, sizeof(bias_type)),
                   "hipblasLtMatmulDescSetAttribute bias type");
      }
    }
    // hipBLASLt's gfx1151 row-major heuristic surface contains algorithms that
    // can pass selection and still access outside the output for this large
    // production shape. Use the equivalent, mature column-major formulation:
    // D_row = A_row * B_row  <=>  D_col^T = B_col^T * A_col^T. The underlying
    // bytes are unchanged; only descriptors and argument order are swapped.
    if (impl_->torch_n1_layout) {
      // PyTorch's ROCm matmul uses the native N=1 column-major view directly:
      // W=[1,K], X^T=[K,M], D^T=[1,M], with no transpose attributes.  The
      // generic mathematically-equivalent W^T view selects a different
      // hipBLASLt reduction and changes a small number of BF16 gate values.
      check_blas(hipblasLtMatrixLayoutCreate(
                     &impl_->a_layout, HIP_R_16BF, n, k,
                     static_cast<std::int64_t>(n)),
                 "hipblasLtMatrixLayoutCreate W-n1-view");
    } else if (right_operand_is_transposed) {
      check_blas(hipblasLtMatrixLayoutCreate(
                     &impl_->a_layout, HIP_R_16BF, k, n,
                     static_cast<std::int64_t>(k)),
                 "hipblasLtMatrixLayoutCreate W-row-view");
      if (algorithm_source == nullptr) {
        const hipblasOperation_t transpose = HIPBLAS_OP_T;
        check_blas(hipblasLtMatmulDescSetAttribute(
                       impl_->operation, HIPBLASLT_MATMUL_DESC_TRANSA,
                       &transpose, sizeof(transpose)),
                   "hipblasLtMatmulDescSetAttribute transpose W");
      }
    } else {
      check_blas(hipblasLtMatrixLayoutCreate(
                     &impl_->a_layout, HIP_R_16BF, n, k,
                     static_cast<std::int64_t>(n)),
                 "hipblasLtMatrixLayoutCreate B-transpose-view");
    }
    check_blas(hipblasLtMatrixLayoutCreate(&impl_->b_layout, HIP_R_16BF,
                                            k, m, static_cast<std::int64_t>(k)),
               "hipblasLtMatrixLayoutCreate A-transpose-view");
    check_blas(hipblasLtMatrixLayoutCreate(&impl_->c_layout, HIP_R_16BF,
                                            n, m, static_cast<std::int64_t>(n)),
               "hipblasLtMatrixLayoutCreate C-transpose-view");
    check_blas(hipblasLtMatrixLayoutCreate(&impl_->d_layout, HIP_R_16BF,
                                            n, m, static_cast<std::int64_t>(n)),
               "hipblasLtMatrixLayoutCreate D-transpose-view");
    if (algorithm_source != nullptr) {
      if (algorithm_source->impl_->workspace_bytes > workspace_limit_bytes) {
        throw std::invalid_argument(
            "BF16 GEMM algorithm source runtime is incompatible");
      }
      impl_->algorithm = algorithm_source->impl_->algorithm;
      impl_->workspace_bytes =
          algorithm_source->impl_->workspace_bytes;
      impl_->heuristic_count = 1;
    } else {
      check_blas(hipblasLtMatmulPreferenceCreate(&impl_->preference),
                 "hipblasLtMatmulPreferenceCreate");
      check_blas(hipblasLtMatmulPreferenceSetAttribute(
                     impl_->preference,
                     HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                     &workspace_limit_bytes, sizeof(workspace_limit_bytes)),
                 "hipblasLtMatmulPreferenceSetAttribute workspace");
      std::array<hipblasLtMatmulHeuristicResult_t, 32> heuristics{};
      check_blas(hipblasLtMatmulAlgoGetHeuristic(
                     impl_->handle, impl_->operation, impl_->a_layout,
                     impl_->b_layout, impl_->c_layout, impl_->d_layout,
                     impl_->preference,
                     static_cast<int>(heuristics.size()), heuristics.data(),
                     &impl_->heuristic_count),
                 "hipblasLtMatmulAlgoGetHeuristic");
      const auto selected = std::find_if(
          heuristics.begin(), heuristics.begin() + impl_->heuristic_count,
          [workspace_limit_bytes](const auto& result) {
            return result.state == HIPBLAS_STATUS_SUCCESS &&
                   result.workspaceSize <= workspace_limit_bytes;
          });
      if (selected == heuristics.begin() + impl_->heuristic_count) {
        throw std::runtime_error(
            "hipBLASLt returned no supported BF16 GEMM algorithm");
      }
      impl_->algorithm = selected->algo;
      impl_->workspace_bytes = selected->workspaceSize;
    }
    if (impl_->workspace_bytes != 0) {
      check_hip(hipMalloc(&impl_->workspace, impl_->workspace_bytes),
                "hipMalloc hipBLASLt workspace");
    }
  } catch (...) {
    impl_->release();
    throw;
  }
}

Bf16GemmPlan::~Bf16GemmPlan() = default;
Bf16GemmPlan::Bf16GemmPlan(Bf16GemmPlan&&) noexcept = default;
Bf16GemmPlan& Bf16GemmPlan::operator=(Bf16GemmPlan&&) noexcept = default;

void Bf16GemmPlan::launch(const void* a, const void* b, void* d,
                          void* stream) const {
  if (!impl_ || !a || !b || !d || impl_->bias_epilogue) {
    throw std::invalid_argument("BF16 GEMM launch requires initialized non-null inputs");
  }
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  check_blas(hipblasLtMatmul(
                 impl_->handle, impl_->operation, &alpha,
                 b, impl_->a_layout, a, impl_->b_layout, &beta,
                 d, impl_->c_layout, d, impl_->d_layout, &impl_->algorithm,
                 impl_->workspace, impl_->workspace_bytes,
                 static_cast<hipStream_t>(stream)),
             "hipblasLtMatmul");
}

void Bf16GemmPlan::launch_with_bias(const void* a, const void* b,
                                    const void* bias, void* d,
                                    void* stream) const {
  if (!impl_ || !a || !b || !bias || !d || !impl_->bias_epilogue) {
    throw std::invalid_argument(
        "biased BF16 GEMM launch requires a biased plan and non-null inputs");
  }
  check_blas(hipblasLtMatmulDescSetAttribute(
                 impl_->operation, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                 &bias, sizeof(bias)),
             "hipblasLtMatmulDescSetAttribute bias pointer");
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  check_blas(hipblasLtMatmul(
                 impl_->handle, impl_->operation, &alpha, b,
                 impl_->a_layout, a, impl_->b_layout, &beta, d,
                 impl_->c_layout, d, impl_->d_layout, &impl_->algorithm,
                 impl_->workspace, impl_->workspace_bytes,
                 static_cast<hipStream_t>(stream)),
             "hipblasLtMatmul bias");
}

std::size_t Bf16GemmPlan::m() const { return impl_->m; }
std::size_t Bf16GemmPlan::n() const { return impl_->n; }
std::size_t Bf16GemmPlan::k() const { return impl_->k; }
std::size_t Bf16GemmPlan::workspace_bytes() const { return impl_->workspace_bytes; }
int Bf16GemmPlan::heuristic_count() const { return impl_->heuristic_count; }
int Bf16GemmPlan::library_version() const { return impl_->library_version; }
bool Bf16GemmPlan::bias_epilogue() const { return impl_->bias_epilogue; }

Bf16GemmProbeResult probe_bf16_gemm() {
  Bf16GemmProbeResult result;
  result.m = 8192;
  result.n = 12352;
  result.k = 2048;
  hipDeviceProp_t properties{};
  check_hip(hipGetDeviceProperties(&properties, 0), "hipGetDeviceProperties");
  result.gpu_arch = properties.gcnArchName;
  if (result.gpu_arch.find("gfx1151") != 0) {
    throw std::runtime_error("native BF16 GEMM probe requires gfx1151, got " +
                             result.gpu_arch);
  }

  DeviceAllocation a(result.m * result.k * sizeof(hip_bfloat16));
  DeviceAllocation b(result.k * result.n * sizeof(hip_bfloat16));
  DeviceAllocation d(result.m * result.n * sizeof(hip_bfloat16));
  fill_bf16(a.get(), result.m * result.k, 1.0f);
  fill_bf16(b.get(), result.k * result.n, 1.0f);
  Bf16GemmPlan plan(result.m, result.n, result.k);
  result.workspace_bytes = plan.workspace_bytes();
  result.heuristic_count = plan.heuristic_count();
  result.library_version = plan.library_version();

  for (int index = 0; index < 2; ++index) plan.launch(a.get(), b.get(), d.get());
  check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize warmup");
  Event start;
  Event stop;
  for (int index = 0; index < 5; ++index) {
    check_hip(hipEventRecord(start), "hipEventRecord start");
    plan.launch(a.get(), b.get(), d.get());
    check_hip(hipEventRecord(stop), "hipEventRecord stop");
    check_hip(hipEventSynchronize(stop), "hipEventSynchronize stop");
    float milliseconds = 0.0f;
    check_hip(hipEventElapsedTime(&milliseconds, start, stop),
              "hipEventElapsedTime");
    result.measured_ms.push_back(milliseconds);
  }

  constexpr std::size_t sample_elements = 4096;
  std::array<std::uint16_t, sample_elements> sample{};
  check_hip(hipMemcpy(sample.data(), d.get(), sizeof(sample), hipMemcpyDeviceToHost),
            "hipMemcpy GEMM sample");
  constexpr std::uint16_t expected = 0x4500;  // BF16(2048.0)
  result.expected_bf16_elements = sample.size();
  for (const std::uint16_t value : sample) {
    result.exact_bf16_elements += value == expected ? 1 : 0;
  }
  if (result.exact_bf16_elements != result.expected_bf16_elements) {
    throw std::runtime_error("native BF16 GEMM exact sample comparison failed");
  }
  std::vector<double> sorted = result.measured_ms;
  std::sort(sorted.begin(), sorted.end());
  result.median_ms = sorted[sorted.size() / 2];
  result.tflops = (2.0 * static_cast<double>(result.m) * result.n * result.k) /
                  (result.median_ms * 1.0e9);
  return result;
}

}  // namespace aima
