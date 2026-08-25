// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_http_support.h"

#include <chrono>
#include <cstdint>
#include <ctime>
#include <limits>
#include <stdexcept>
#include <utility>

namespace aima {
namespace {

[[noreturn]] void throw_invalid_timeout(const char* name) {
  throw std::runtime_error(std::string(name == nullptr ? "timeout" : name) +
                           ": invalid timeout in milliseconds");
}

}  // namespace

std::size_t parse_native_http_timeout_ms(const std::string& value,
                                         const char* name) {
  if (value.empty()) throw_invalid_timeout(name);

  std::size_t result = 0;
  constexpr std::size_t kSizeMaximum =
      std::numeric_limits<std::size_t>::max();
  for (const char character : value) {
    if (character < '0' || character > '9') throw_invalid_timeout(name);
    const std::size_t digit = static_cast<std::size_t>(character - '0');
    if (result > (kSizeMaximum - digit) / 10) throw_invalid_timeout(name);
    result = result * 10 + digit;
  }

  using Milliseconds = std::chrono::milliseconds;
  using SteadyDuration = std::chrono::steady_clock::duration;
  const auto maximum_steady_milliseconds =
      std::chrono::duration_cast<Milliseconds>(SteadyDuration::max()).count();
  if (maximum_steady_milliseconds < 0 ||
      static_cast<std::uintmax_t>(result) >
          static_cast<std::uintmax_t>(maximum_steady_milliseconds)) {
    throw_invalid_timeout(name);
  }

  const std::uintmax_t seconds =
      static_cast<std::uintmax_t>(result / 1000);
  const std::uintmax_t maximum_time_t_seconds =
      static_cast<std::uintmax_t>(std::numeric_limits<time_t>::max());
  if (seconds > maximum_time_t_seconds) throw_invalid_timeout(name);

  return result;
}

NativeSerialExecutor::NativeSerialExecutor(std::size_t maximum_pending)
    : maximum_pending_(maximum_pending) {
  if (maximum_pending_ == 0) {
    throw std::invalid_argument("maximum_pending must be positive");
  }
  worker_ = std::thread(&NativeSerialExecutor::worker_loop, this);
}

NativeSerialExecutor::~NativeSerialExecutor() { shutdown(); }

bool NativeSerialExecutor::submit(Task task) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_ || pending_.size() >= maximum_pending_) return false;
    pending_.push_back(std::move(task));
  }
  condition_.notify_one();
  return true;
}

bool NativeSerialExecutor::busy() const noexcept {
  return busy_.load();
}

void NativeSerialExecutor::shutdown() {
  std::lock_guard<std::mutex> shutdown_lock(shutdown_mutex_);
  std::deque<Task> cancelled;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
    cancelled.swap(pending_);
  }

  for (Task& task : cancelled) {
    try {
      if (task.cancel) task.cancel();
    } catch (...) {
    }
  }

  condition_.notify_all();
  if (worker_.joinable()) worker_.join();
}

void NativeSerialExecutor::worker_loop() {
  while (true) {
    Task task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      condition_.wait(lock, [this]() { return stopping_ || !pending_.empty(); });
      if (stopping_ && pending_.empty()) return;
      task = std::move(pending_.front());
      pending_.pop_front();
    }

    busy_.store(true);
    try {
      if (task.run) task.run();
    } catch (...) {
    }
    busy_.store(false);
  }
}

}  // namespace aima
