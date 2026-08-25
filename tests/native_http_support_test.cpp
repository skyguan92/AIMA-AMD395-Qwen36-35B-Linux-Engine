// SPDX-License-Identifier: Apache-2.0

#include "aima/native_http_support.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iostream>
#include <limits>
#include <ratio>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

namespace {

using namespace std::chrono_literals;

void require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "native_http_support_test: " << message << '\n';
    std::exit(1);
  }
}

template <typename Function>
void require_throws(Function&& function, const char* message) {
  try {
    function();
  } catch (...) {
    return;
  }
  require(false, message);
}

void note_running(std::atomic<int>& running, std::atomic<int>& maximum) {
  const int current = running.fetch_add(1) + 1;
  int observed = maximum.load();
  while (current > observed &&
         !maximum.compare_exchange_weak(observed, current)) {
  }
}

void test_timeout_parser() {
  require(aima::parse_native_http_timeout_ms("0", "--timeout") == 0,
          "zero timeout was rejected");
  require(aima::parse_native_http_timeout_ms("600001", "--timeout") ==
              600001,
          "timeout above the legacy request cap changed");
  require(aima::parse_native_http_timeout_ms("86400000", "--timeout") ==
              86400000,
          "one-day timeout was rejected");
  require_throws(
      []() { (void)aima::parse_native_http_timeout_ms("-1", "--timeout"); },
      "negative timeout was admitted");
  require_throws(
      []() { (void)aima::parse_native_http_timeout_ms("", "--timeout"); },
      "empty timeout was admitted");
  require_throws(
      []() { (void)aima::parse_native_http_timeout_ms("+1", "--timeout"); },
      "signed timeout was admitted");
  require_throws(
      []() { (void)aima::parse_native_http_timeout_ms(" 1", "--timeout"); },
      "leading whitespace timeout was admitted");
  require_throws(
      []() { (void)aima::parse_native_http_timeout_ms("1junk", "--timeout"); },
      "trailing junk timeout was admitted");
  require_throws(
      []() {
        (void)aima::parse_native_http_timeout_ms("18446744073709551615",
                                                  "--timeout");
      },
      "unrepresentable timeout was admitted");
  require_throws(
      []() { (void)aima::parse_native_http_timeout_ms("abc", "--timeout"); },
      "non-numeric timeout was admitted");
  require_throws(
      []() {
        (void)aima::parse_native_http_timeout_ms("184467440737095516160",
                                                  "--timeout");
      },
      "overflowing timeout was admitted");
}

void test_duration_range_checking() {
  using CoarseDuration = std::chrono::duration<std::int8_t, std::ratio<2>>;
  require(aima::native_http_detail::milliseconds_fit_duration<CoarseDuration>(
              254000),
          "coarse duration rejected its exact millisecond maximum");
  require(!aima::native_http_detail::milliseconds_fit_duration<CoarseDuration>(
              254001),
          "coarse duration admitted a value beyond its exact maximum");

  using FineDuration = std::chrono::duration<std::int32_t, std::nano>;
  require(aima::native_http_detail::milliseconds_fit_duration<FineDuration>(
              2147),
          "fine duration rejected its whole-millisecond maximum");
  require(!aima::native_http_detail::milliseconds_fit_duration<FineDuration>(
              2148),
          "fine duration admitted a value beyond its maximum");

  using HugeCoarseDuration =
      std::chrono::duration<std::uint64_t, std::ratio<2>>;
  using MillisecondRep = std::chrono::milliseconds::rep;
  const std::uintmax_t maximum_millisecond_count =
      static_cast<std::uintmax_t>(std::numeric_limits<MillisecondRep>::max());
  if (maximum_millisecond_count <=
      static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max())) {
    const std::size_t maximum =
        static_cast<std::size_t>(maximum_millisecond_count);
    require(aima::native_http_detail::milliseconds_fit_duration<
                HugeCoarseDuration>(maximum),
            "coarse duration overflowed while checking a large value");
    if (maximum < std::numeric_limits<std::size_t>::max()) {
      require(!aima::native_http_detail::milliseconds_fit_duration<
                  HugeCoarseDuration>(maximum + 1),
              "duration check admitted a count that cannot construct "
              "milliseconds");
    }
  }
}

void test_executor_capacity_and_serial_execution() {
  aima::NativeSerialExecutor executor(1);
  std::atomic<int> running{0};
  std::atomic<int> maximum_running{0};
  std::promise<void> active_started;
  std::future<void> active_started_future = active_started.get_future();
  std::promise<void> release_active;
  std::shared_future<void> release_active_future =
      release_active.get_future().share();
  std::promise<void> pending_finished;
  std::future<void> pending_finished_future = pending_finished.get_future();

  aima::NativeSerialExecutor::Task active_task;
  active_task.run = [&]() {
    note_running(running, maximum_running);
    active_started.set_value();
    require(release_active_future.wait_for(10s) == std::future_status::ready,
            "active task release was not signaled");
    running.fetch_sub(1);
  };
  active_task.cancel = []() {};
  require(executor.submit(std::move(active_task)),
          "first task was rejected");
  require(active_started_future.wait_for(5s) == std::future_status::ready,
          "first task did not start");
  require(executor.busy(), "executor was not busy while task was active");

  aima::NativeSerialExecutor::Task pending_task;
  pending_task.run = [&]() {
    note_running(running, maximum_running);
    running.fetch_sub(1);
    pending_finished.set_value();
  };
  pending_task.cancel = []() {};
  require(executor.submit(std::move(pending_task)),
          "pending task at capacity was rejected");
  aima::NativeSerialExecutor::Task rejected_task;
  rejected_task.run = []() {};
  rejected_task.cancel = []() {};
  require(!executor.submit(std::move(rejected_task)),
          "task beyond pending capacity was admitted");

  release_active.set_value();
  require(pending_finished_future.wait_for(5s) == std::future_status::ready,
          "pending task did not run after active task released");
  require(maximum_running.load() == 1,
          "executor ran more than one task simultaneously");
  executor.shutdown();
  require(!executor.busy(),
          "executor remained busy after normal task completion");
}

void test_executor_survives_throw_and_clears_busy() {
  aima::NativeSerialExecutor executor(1);
  std::promise<void> throw_started;
  std::future<void> throw_started_future = throw_started.get_future();
  std::promise<void> marker_finished;
  std::future<void> marker_finished_future = marker_finished.get_future();

  aima::NativeSerialExecutor::Task throwing_task;
  throwing_task.run = [&]() {
    throw_started.set_value();
    throw std::runtime_error("expected task failure");
  };
  throwing_task.cancel = []() {};
  require(executor.submit(std::move(throwing_task)),
          "throwing task was rejected");
  require(throw_started_future.wait_for(5s) == std::future_status::ready,
          "throwing task did not start");

  aima::NativeSerialExecutor::Task marker_task;
  marker_task.run = [&]() { marker_finished.set_value(); };
  marker_task.cancel = []() {};
  require(executor.submit(std::move(marker_task)),
          "marker after throwing task was rejected");
  require(marker_finished_future.wait_for(5s) == std::future_status::ready,
          "worker did not survive throwing task");
  executor.shutdown();
  require(!executor.busy(),
          "executor remained busy after throwing task completion");
}

void test_shutdown_cancels_pending_before_active_completes() {
  aima::NativeSerialExecutor executor(1);
  std::promise<void> active_started;
  std::future<void> active_started_future = active_started.get_future();
  std::promise<void> release_active;
  std::shared_future<void> release_active_future =
      release_active.get_future().share();
  std::promise<void> pending_cancelled;
  std::future<void> pending_cancelled_future = pending_cancelled.get_future();
  std::atomic<bool> pending_ran{false};
  std::promise<void> shutdown_returned;
  std::future<void> shutdown_returned_future = shutdown_returned.get_future();
  std::promise<void> shutdown_thread_ready;
  std::future<void> shutdown_thread_ready_future =
      shutdown_thread_ready.get_future();
  std::promise<void> permit_shutdown;
  std::shared_future<void> permit_shutdown_future =
      permit_shutdown.get_future().share();
  std::promise<void> shutdown_entered;
  std::future<void> shutdown_entered_future = shutdown_entered.get_future();

  aima::NativeSerialExecutor::Task active_task;
  active_task.run = [&]() {
    active_started.set_value();
    require(release_active_future.wait_for(10s) == std::future_status::ready,
            "shutdown test active task release was not signaled");
  };
  active_task.cancel = []() {};
  require(executor.submit(std::move(active_task)),
          "shutdown test active task was rejected");
  require(active_started_future.wait_for(5s) == std::future_status::ready,
          "shutdown test active task did not start");
  aima::NativeSerialExecutor::Task pending_task;
  pending_task.run = [&]() { pending_ran.store(true); };
  pending_task.cancel = [&]() { pending_cancelled.set_value(); };
  require(executor.submit(std::move(pending_task)),
          "shutdown test pending task was rejected");

  std::thread shutdown_thread([&]() {
    shutdown_thread_ready.set_value();
    require(permit_shutdown_future.wait_for(2s) == std::future_status::ready,
            "shutdown thread was not released");
    shutdown_entered.set_value();
    executor.shutdown();
    shutdown_returned.set_value();
  });
  require(shutdown_thread_ready_future.wait_for(2s) ==
              std::future_status::ready,
          "shutdown thread did not reach its start barrier");
  permit_shutdown.set_value();
  require(shutdown_entered_future.wait_for(2s) == std::future_status::ready,
          "shutdown call was not entered");
  require(pending_cancelled_future.wait_for(2s) == std::future_status::ready,
          "shutdown did not cancel pending work while active task ran");
  require(shutdown_returned_future.wait_for(50ms) ==
              std::future_status::timeout,
          "shutdown returned before the active task completed");

  release_active.set_value();
  require(shutdown_returned_future.wait_for(5s) == std::future_status::ready,
          "shutdown did not return after active task completed");
  shutdown_thread.join();
  require(!pending_ran.load(), "shutdown-cancelled task was run");

  executor.shutdown();
  aima::NativeSerialExecutor::Task stopped_task;
  stopped_task.run = []() {};
  stopped_task.cancel = []() {};
  require(!executor.submit(std::move(stopped_task)),
          "submission was admitted after shutdown");
}

void test_cancel_callback_can_recursively_shutdown() {
  aima::NativeSerialExecutor executor(1);
  std::promise<void> active_started;
  std::future<void> active_started_future = active_started.get_future();
  std::promise<void> release_active;
  std::shared_future<void> release_active_future =
      release_active.get_future().share();
  std::promise<void> cancel_started;
  std::future<void> cancel_started_future = cancel_started.get_future();
  std::promise<void> recursive_returned;
  std::future<void> recursive_returned_future = recursive_returned.get_future();
  std::promise<void> outer_returned;
  std::future<void> outer_returned_future = outer_returned.get_future();
  std::atomic<bool> pending_ran{false};

  aima::NativeSerialExecutor::Task active_task;
  active_task.run = [&]() {
    active_started.set_value();
    require(release_active_future.wait_for(10s) == std::future_status::ready,
            "recursive shutdown active task was not released");
  };
  active_task.cancel = []() {};
  require(executor.submit(std::move(active_task)),
          "recursive shutdown active task was rejected");
  require(active_started_future.wait_for(5s) == std::future_status::ready,
          "recursive shutdown active task did not start");

  aima::NativeSerialExecutor::Task pending_task;
  pending_task.run = [&]() { pending_ran.store(true); };
  pending_task.cancel = [&]() {
    cancel_started.set_value();
    executor.shutdown();
    recursive_returned.set_value();
  };
  require(executor.submit(std::move(pending_task)),
          "recursive shutdown pending task was rejected");

  std::thread shutdown_thread([&]() {
    executor.shutdown();
    outer_returned.set_value();
  });
  require(cancel_started_future.wait_for(5s) == std::future_status::ready,
          "recursive shutdown cancel callback did not start");
  release_active.set_value();
  require(recursive_returned_future.wait_for(5s) == std::future_status::ready,
          "shutdown called recursively from cancel did not return");
  require(outer_returned_future.wait_for(5s) == std::future_status::ready,
          "outer shutdown did not return after recursive shutdown");
  shutdown_thread.join();
  require(!pending_ran.load(), "recursively cancelled task was run");
}

void test_worker_and_external_shutdown_do_not_form_join_cycle() {
  aima::NativeSerialExecutor executor(1);
  std::promise<void> active_started;
  std::future<void> active_started_future = active_started.get_future();
  std::promise<void> permit_worker_shutdown;
  std::shared_future<void> permit_worker_shutdown_future =
      permit_worker_shutdown.get_future().share();
  std::promise<void> worker_shutdown_returned;
  std::future<void> worker_shutdown_returned_future =
      worker_shutdown_returned.get_future();
  std::promise<void> pending_cancelled;
  std::future<void> pending_cancelled_future = pending_cancelled.get_future();
  std::promise<void> external_shutdown_returned;
  std::future<void> external_shutdown_returned_future =
      external_shutdown_returned.get_future();
  std::atomic<bool> pending_ran{false};

  aima::NativeSerialExecutor::Task active_task;
  active_task.run = [&]() {
    active_started.set_value();
    require(permit_worker_shutdown_future.wait_for(10s) ==
                std::future_status::ready,
            "worker shutdown task was not released");
    executor.shutdown();
    worker_shutdown_returned.set_value();
  };
  active_task.cancel = []() {};
  require(executor.submit(std::move(active_task)),
          "worker shutdown active task was rejected");
  require(active_started_future.wait_for(5s) == std::future_status::ready,
          "worker shutdown active task did not start");

  aima::NativeSerialExecutor::Task pending_task;
  pending_task.run = [&]() { pending_ran.store(true); };
  pending_task.cancel = [&]() { pending_cancelled.set_value(); };
  require(executor.submit(std::move(pending_task)),
          "worker shutdown pending task was rejected");

  std::thread external_shutdown_thread([&]() {
    executor.shutdown();
    external_shutdown_returned.set_value();
  });
  require(pending_cancelled_future.wait_for(5s) == std::future_status::ready,
          "external shutdown did not cancel pending work");
  permit_worker_shutdown.set_value();
  require(worker_shutdown_returned_future.wait_for(5s) ==
              std::future_status::ready,
          "shutdown called from worker did not return");
  require(external_shutdown_returned_future.wait_for(5s) ==
              std::future_status::ready,
          "external shutdown did not finish after worker shutdown");
  external_shutdown_thread.join();
  require(!pending_ran.load(), "worker shutdown cancelled task was run");
}

}  // namespace

int main() {
  static_assert(!std::is_copy_constructible<aima::NativeSerialExecutor>::value,
                "NativeSerialExecutor must not be copy constructible");
  static_assert(!std::is_copy_assignable<aima::NativeSerialExecutor>::value,
                "NativeSerialExecutor must not be copy assignable");
  test_timeout_parser();
  test_duration_range_checking();
  test_executor_capacity_and_serial_execution();
  test_executor_survives_throw_and_clears_busy();
  test_shutdown_cancels_pending_before_active_completes();
  test_cancel_callback_can_recursively_shutdown();
  test_worker_and_external_shutdown_do_not_form_join_cycle();
  std::cout << "native_http_support_test: PASS\n";
  return 0;
}
