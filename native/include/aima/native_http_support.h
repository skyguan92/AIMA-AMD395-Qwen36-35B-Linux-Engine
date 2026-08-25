#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

namespace aima {

std::size_t parse_native_http_timeout_ms(const std::string& value,
                                         const char* name);

class NativeSerialExecutor {
 public:
  struct Task {
    std::function<void()> run;
    std::function<void()> cancel;
  };

  explicit NativeSerialExecutor(std::size_t maximum_pending);
  ~NativeSerialExecutor();

  NativeSerialExecutor(const NativeSerialExecutor&) = delete;
  NativeSerialExecutor& operator=(const NativeSerialExecutor&) = delete;

  bool submit(Task task);
  bool busy() const noexcept;
  void shutdown();

 private:
  void worker_loop();

  const std::size_t maximum_pending_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<Task> pending_;
  bool stopping_ = false;
  std::atomic<bool> busy_{false};
  std::thread worker_;
  std::mutex shutdown_mutex_;
};

}  // namespace aima
