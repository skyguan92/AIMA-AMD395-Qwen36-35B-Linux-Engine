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

std::size_t require_positive_capacity(std::size_t maximum_pending) {
  if (maximum_pending == 0) {
    throw std::invalid_argument("maximum_pending must be positive");
  }
  return maximum_pending;
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

  using SteadyDuration = std::chrono::steady_clock::duration;
  if (!native_http_detail::milliseconds_fit_duration<SteadyDuration>(result)) {
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
    : maximum_pending_(require_positive_capacity(maximum_pending)),
      worker_(&NativeSerialExecutor::worker_loop, this),
      worker_id_(worker_.get_id()) {}

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
  std::deque<Task> cancelled;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
    cancelled.swap(pending_);
  }
  condition_.notify_all();

  for (Task& task : cancelled) {
    try {
      if (task.cancel) task.cancel();
    } catch (...) {
    }
  }

  if (std::this_thread::get_id() == worker_id_) return;

  std::unique_lock<std::mutex> lock(mutex_);
  while (join_state_ == JoinState::kJoining) {
    condition_.wait(lock,
                    [this]() { return join_state_ != JoinState::kJoining; });
  }
  if (join_state_ == JoinState::kJoined) return;
  join_state_ = JoinState::kJoining;
  lock.unlock();
  try {
    worker_.join();
  } catch (...) {
    lock.lock();
    join_state_ = JoinState::kUnclaimed;
    lock.unlock();
    condition_.notify_all();
    throw;
  }
  lock.lock();
  join_state_ = JoinState::kJoined;
  lock.unlock();
  condition_.notify_all();
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
