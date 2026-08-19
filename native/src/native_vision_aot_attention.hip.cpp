// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vision_aot_attention.h"

#include "aima/aot_kernel.h"
#include "aima/native_vl_processor.h"
#include "aima/sha256.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace aima {
namespace {

constexpr char kVisionAttentionImageSha256[] =
    "8327e42d99f5d34667b59d481dabc8e1d7cf9675361df974d85f5d6005109a9e";
constexpr char kVisionAttentionKernelSymbol[] = "_fwd_kernel";
constexpr std::uint32_t kVisionHeads = 16;
constexpr std::uint32_t kVisionHeadDimension = 72;
constexpr std::uint32_t kVisionHidden = 1152;
constexpr std::uint32_t kAttentionBlock = 128;
constexpr float kSoftmaxScaleLog2 = 0.17002323269844055f;

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

struct VerifiedAttentionExecutable {
  std::shared_ptr<AotKernel> kernel;
  std::string image_sha256;
};

VerifiedAttentionExecutable load_verified_attention_executable(
    const std::filesystem::path& image_path) {
  VerifiedAttentionExecutable result;
  result.image_sha256 = sha256_file(image_path);
  if (result.image_sha256 != kVisionAttentionImageSha256) {
    throw std::runtime_error("native vision attention AOT image hash mismatch");
  }
  result.kernel = std::make_shared<AotKernel>(
      AotKernel::from_file(image_path, kVisionAttentionKernelSymbol));
  return result;
}

}  // namespace

struct NativeVisionAotAttentionPlan::Impl {
  Impl(const std::filesystem::path& image_path, std::size_t patches,
       const std::vector<std::uint32_t>& cu_seqlens)
      : Impl(load_verified_attention_executable(image_path), patches,
             cu_seqlens) {}

  Impl(VerifiedAttentionExecutable executable, std::size_t patches,
       const std::vector<std::uint32_t>& cu_seqlens)
      : patch_count_value(patches),
        image_sha256_value(std::move(executable.image_sha256)),
        kernel(std::move(executable.kernel)) {
    if (patches == 0 || patches > kNativeVlVisionBatchPatchLimit) {
      throw std::invalid_argument(
          "native AOT vision attention patch count is outside the budget");
    }
    if (cu_seqlens.size() < 2 || cu_seqlens.front() != 0 ||
        cu_seqlens.back() != patches) {
      throw std::invalid_argument(
          "native AOT vision attention sequence boundaries are invalid");
    }
    segment_count_value = cu_seqlens.size() - 1;
    std::vector<std::int32_t> starts(segment_count_value);
    std::vector<std::int32_t> lengths(segment_count_value);
    for (std::size_t segment = 0; segment < segment_count_value; ++segment) {
      const std::uint32_t start = cu_seqlens[segment];
      const std::uint32_t end = cu_seqlens[segment + 1];
      if (end <= start || end > patches) {
        throw std::invalid_argument(
            "native AOT vision attention sequences must be non-empty");
      }
      starts[segment] = static_cast<std::int32_t>(start);
      lengths[segment] = static_cast<std::int32_t>(end - start);
      max_segment_length =
          std::max(max_segment_length, static_cast<std::uint32_t>(end - start));
    }
    workspace_bytes_value =
        2 * segment_count_value * sizeof(std::int32_t);
    if (!kernel || image_sha256_value != kVisionAttentionImageSha256) {
      throw std::runtime_error("native vision attention AOT image hash mismatch");
    }
    try {
      check_hip(hipMalloc(&metadata_device, workspace_bytes_value),
                "hipMalloc AOT vision attention metadata");
      starts_device = static_cast<std::int32_t*>(metadata_device);
      lengths_device = starts_device + segment_count_value;
      check_hip(hipMemcpy(starts_device, starts.data(),
                          starts.size() * sizeof(std::int32_t),
                          hipMemcpyHostToDevice),
                "hipMemcpy AOT vision attention starts");
      check_hip(hipMemcpy(lengths_device, lengths.data(),
                          lengths.size() * sizeof(std::int32_t),
                          hipMemcpyHostToDevice),
                "hipMemcpy AOT vision attention lengths");
    } catch (...) {
      release();
      throw;
    }
  }

  ~Impl() { release(); }

  void release() noexcept {
    if (metadata_device != nullptr) {
      const hipError_t ignored = hipFree(metadata_device);
      static_cast<void>(ignored);
      metadata_device = nullptr;
    }
    starts_device = nullptr;
    lengths_device = nullptr;
    kernel.reset();
  }

  std::size_t patch_count_value = 0;
  std::size_t segment_count_value = 0;
  std::size_t workspace_bytes_value = 0;
  std::uint32_t max_segment_length = 0;
  std::string image_sha256_value;
  void* metadata_device = nullptr;
  std::int32_t* starts_device = nullptr;
  std::int32_t* lengths_device = nullptr;
  std::shared_ptr<AotKernel> kernel;
};

NativeVisionAotAttentionPlan::NativeVisionAotAttentionPlan(
    const std::filesystem::path& image_path, std::size_t patch_count,
    const std::vector<std::uint32_t>& cu_seqlens)
    : impl_(std::make_unique<Impl>(image_path, patch_count, cu_seqlens)) {}
NativeVisionAotAttentionPlan::NativeVisionAotAttentionPlan(
    std::shared_ptr<AotKernel> kernel, std::string image_sha256,
    std::size_t patch_count,
    const std::vector<std::uint32_t>& cu_seqlens)
    : impl_(std::make_unique<Impl>(
          VerifiedAttentionExecutable{std::move(kernel),
                                      std::move(image_sha256)},
          patch_count,
          cu_seqlens)) {}
NativeVisionAotAttentionPlan::~NativeVisionAotAttentionPlan() = default;
NativeVisionAotAttentionPlan::NativeVisionAotAttentionPlan(
    NativeVisionAotAttentionPlan&&) noexcept = default;
NativeVisionAotAttentionPlan& NativeVisionAotAttentionPlan::operator=(
    NativeVisionAotAttentionPlan&&) noexcept = default;

void NativeVisionAotAttentionPlan::launch(
    const void* query_device, const void* key_device, const void* value_device,
    void* output_device, void* stream_pointer) const {
  if (!impl_ || !impl_->kernel || query_device == nullptr ||
      key_device == nullptr || value_device == nullptr ||
      output_device == nullptr || query_device == output_device ||
      key_device == output_device || value_device == output_device) {
    throw std::invalid_argument("native AOT vision attention launch is invalid");
  }
  void* query = const_cast<void*>(query_device);
  void* key = const_cast<void*>(key_device);
  void* value = const_cast<void*>(value_device);
  void* starts = impl_->starts_device;
  void* lengths = impl_->lengths_device;
  void* output = output_device;
  float scale = kSoftmaxScaleLog2;
  std::int32_t stride_patch = static_cast<std::int32_t>(kVisionHidden);
  std::int32_t stride_head = static_cast<std::int32_t>(kVisionHeadDimension);
  std::vector<void*> parameters{
      &query,        &key,          &value,         &scale,
      &starts,       &lengths,      &output,        &stride_patch,
      &stride_head,  &stride_patch, &stride_head,   &stride_patch,
      &stride_head,  &stride_patch, &stride_head,
  };
  AotLaunchConfig config;
  config.grid_x = static_cast<std::uint32_t>(impl_->segment_count_value);
  config.grid_y = kVisionHeads;
  config.grid_z =
      (impl_->max_segment_length + kAttentionBlock - 1) / kAttentionBlock;
  config.num_warps = 8;
  config.warp_size = 32;
  config.shared_memory_bytes = 32768;
  impl_->kernel->launch(config, parameters,
                        reinterpret_cast<hipStream_t>(stream_pointer));
}

std::size_t NativeVisionAotAttentionPlan::patch_count() const {
  return impl_->patch_count_value;
}

std::size_t NativeVisionAotAttentionPlan::segment_count() const {
  return impl_->segment_count_value;
}

std::size_t NativeVisionAotAttentionPlan::workspace_bytes() const {
  return impl_->workspace_bytes_value;
}

std::shared_ptr<const NativeVisionAotAttentionPlan>
NativeVisionAotAttentionPlan::rebind(
    std::size_t patch_count,
    const std::vector<std::uint32_t>& cu_seqlens) const {
  if (!impl_ || !impl_->kernel) {
    throw std::runtime_error(
        "native vision attention executable is unavailable");
  }
  return std::shared_ptr<const NativeVisionAotAttentionPlan>(
      new NativeVisionAotAttentionPlan(
          impl_->kernel, impl_->image_sha256_value, patch_count,
          cu_seqlens));
}

const std::string& NativeVisionAotAttentionPlan::image_sha256() const {
  return impl_->image_sha256_value;
}

}  // namespace aima
