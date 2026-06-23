#pragma once

#include "Types.hpp"
#include "SPSCQueue.hpp"
#include "OrderBook.hpp"
#include <thread>
#include <atomic>
#include <array>
#include <memory>
#include <string>

// Forward declarations for TensorRT classes
namespace nvinfer1 {
    class IRuntime;
    class ICudaEngine;
    class IExecutionContext;
}

namespace astraea {

class RuntimeOrderBook : public OrderBook<RuntimeOrderBook> {
public:
    inline void on_order_added(uint64_t) noexcept {}
    inline void on_order_canceled(uint64_t) noexcept {}
    inline void on_trade(uint64_t, uint32_t, uint32_t) noexcept {}
};

struct FeatureFrame {
    uint64_t timestamp;
    std::array<float, 40> features;
};

class EngineRuntime {
public:
    EngineRuntime();
    ~EngineRuntime();

    EngineRuntime(const EngineRuntime&) = delete;
    EngineRuntime& operator=(const EngineRuntime&) = delete;
    EngineRuntime(EngineRuntime&&) = delete;
    EngineRuntime& operator=(EngineRuntime&&) = delete;

    bool initialize(const std::string& engine_path);
    void start();
    void stop();

    bool enqueue_event(const MarketEvent& event);
    
    // Performance Tracking
    uint64_t get_processed_count() const { return processed_count_.load(std::memory_order_acquire); }
    uint64_t get_latency_ns(size_t index) const { return latencies_[index]; }
    void allocate_latency_tracking(size_t max_events) { latencies_.resize(max_events, 0); }

private:
    void matching_loop();
    void inference_loop();

    void pin_thread_to_core(std::thread& t, int core_id);

    std::atomic<bool> running_{false};

    SPSCQueue<MarketEvent, 4096> ingestion_queue_;
    SPSCQueue<FeatureFrame, 1024> feature_queue_;

    RuntimeOrderBook order_book_;

    std::thread matching_thread_;
    std::thread inference_thread_;

    nvinfer1::IRuntime* trt_runtime_{nullptr};
    nvinfer1::ICudaEngine* trt_engine_{nullptr};
    nvinfer1::IExecutionContext* trt_context_{nullptr};

    void* gpu_input_buffer_{nullptr};
    void* gpu_output_buffer_{nullptr};

    std::atomic<uint64_t> processed_count_{0};
    std::vector<uint64_t> latencies_;
};

} // namespace astraea
