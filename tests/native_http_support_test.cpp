// SPDX-License-Identifier: Apache-2.0

#include "aima/native_http_support.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <string>
#include <thread>

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
      []() { (void)aima::parse_native_http_timeout_ms("abc", "--timeout"); },
      "non-numeric timeout was admitted");
  require_throws(
      []() {
        (void)aima::parse_native_http_timeout_ms("184467440737095516160",
                                                  "--timeout");
      },
      "overflowing timeout was admitted");
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

  require(executor.submit({
              [&]() {
                note_running(running, maximum_running);
                active_started.set_value();
                release_active_future.wait();
                running.fetch_sub(1);
              },
              []() {}}),
          "first task was rejected");
  require(active_started_future.wait_for(5s) == std::future_status::ready,
          "first task did not start");
  require(executor.busy(), "executor was not busy while task was active");

  require(executor.submit({
              [&]() {
                note_running(running, maximum_running);
                running.fetch_sub(1);
                pending_finished.set_value();
              },
              []() {}}),
          "pending task at capacity was rejected");
  require(!executor.submit({[]() {}, []() {}}),
          "task beyond pending capacity was admitted");

  release_active.set_value();
  require(pending_finished_future.wait_for(5s) == std::future_status::ready,
          "pending task did not run after active task released");
  require(maximum_running.load() == 1,
          "executor ran more than one task simultaneously");
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
  std::promise<void> shutdown_returned;
  std::future<void> shutdown_returned_future = shutdown_returned.get_future();

  require(executor.submit({
              [&]() {
                active_started.set_value();
                release_active_future.wait();
              },
              []() {}}),
          "shutdown test active task was rejected");
  require(active_started_future.wait_for(5s) == std::future_status::ready,
          "shutdown test active task did not start");
  require(executor.submit({[]() {}, [&]() { pending_cancelled.set_value(); }}),
          "shutdown test pending task was rejected");

  std::thread shutdown_thread([&]() {
    executor.shutdown();
    shutdown_returned.set_value();
  });
  require(pending_cancelled_future.wait_for(5s) == std::future_status::ready,
          "shutdown did not cancel pending work while active task ran");
  require(shutdown_returned_future.wait_for(0ms) == std::future_status::timeout,
          "shutdown returned before the active task completed");

  release_active.set_value();
  require(shutdown_returned_future.wait_for(5s) == std::future_status::ready,
          "shutdown did not return after active task completed");
  shutdown_thread.join();

  executor.shutdown();
  require(!executor.submit({[]() {}, []() {}}),
          "submission was admitted after shutdown");
}

}  // namespace

int main() {
  test_timeout_parser();
  test_executor_capacity_and_serial_execution();
  test_shutdown_cancels_pending_before_active_completes();
  std::cout << "native_http_support_test: PASS\n";
  return 0;
}
