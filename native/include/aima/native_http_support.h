#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>

namespace aima {

namespace native_http_detail {
namespace duration_range_detail {

class UnsignedDurationProduct {
 public:
  explicit UnsignedDurationProduct(std::uintmax_t value = 0) noexcept {
    words_[0] = value;
  }

  void multiply(std::uintmax_t factor) noexcept {
    UnsignedDurationProduct multiplicand = *this;
    UnsignedDurationProduct product;
    while (factor != 0) {
      if ((factor & 1U) != 0) product.add(multiplicand);
      factor >>= 1U;
      if (factor != 0) multiplicand.shift_left();
    }
    *this = product;
  }

  bool less_than_or_equal(const UnsignedDurationProduct& other) const
      noexcept {
    for (std::size_t index = words_.size(); index != 0; --index) {
      if (words_[index - 1] < other.words_[index - 1]) return true;
      if (words_[index - 1] > other.words_[index - 1]) return false;
    }
    return true;
  }

 private:
  void add(const UnsignedDurationProduct& other) noexcept {
    std::uintmax_t carry = 0;
    for (std::size_t index = 0; index < words_.size(); ++index) {
      const std::uintmax_t partial = words_[index] + other.words_[index];
      const std::uintmax_t first_carry = partial < words_[index] ? 1U : 0U;
      const std::uintmax_t sum = partial + carry;
      const std::uintmax_t second_carry = sum < partial ? 1U : 0U;
      words_[index] = sum;
      carry = first_carry | second_carry;
    }
  }

  void shift_left() noexcept {
    std::uintmax_t carry = 0;
    constexpr int kHighBit =
        std::numeric_limits<std::uintmax_t>::digits - 1;
    for (std::size_t index = 0; index < words_.size(); ++index) {
      const std::uintmax_t next_carry = words_[index] >> kHighBit;
      words_[index] = (words_[index] << 1U) | carry;
      carry = next_carry;
    }
  }

  // A duration range comparison multiplies at most three uintmax_t-sized
  // factors. Four words leave enough space without relying on extensions.
  std::array<std::uintmax_t, 4> words_{};
};

}  // namespace duration_range_detail

template <typename Duration>
bool milliseconds_fit_duration(std::size_t value) noexcept {
  using DestinationRep = typename Duration::rep;
  using MillisecondRep = std::chrono::milliseconds::rep;
  static_assert(std::is_integral<DestinationRep>::value,
                "duration representation must be integral");
  static_assert(std::is_integral<MillisecondRep>::value,
                "millisecond representation must be integral");

  const std::uintmax_t count = static_cast<std::uintmax_t>(value);
  const std::uintmax_t maximum_millisecond_count =
      static_cast<std::uintmax_t>(std::numeric_limits<MillisecondRep>::max());
  if (count > maximum_millisecond_count) return false;

  // Compare value * period.den with max(rep) * period.num * 1000 exactly.
  // The fixed-width multiword products avoid overflow even for coarse periods.
  duration_range_detail::UnsignedDurationProduct requested(count);
  requested.multiply(static_cast<std::uintmax_t>(Duration::period::den));
  duration_range_detail::UnsignedDurationProduct maximum(
      static_cast<std::uintmax_t>(
          std::numeric_limits<DestinationRep>::max()));
  maximum.multiply(static_cast<std::uintmax_t>(Duration::period::num));
  maximum.multiply(1000);
  return requested.less_than_or_equal(maximum);
}

}  // namespace native_http_detail

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
  enum class JoinState { kUnclaimed, kJoining, kJoined };

  void worker_loop();

  const std::size_t maximum_pending_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<Task> pending_;
  bool stopping_ = false;
  JoinState join_state_ = JoinState::kUnclaimed;
  std::atomic<bool> busy_{false};
  std::thread worker_;
  const std::thread::id worker_id_;
};

}  // namespace aima
