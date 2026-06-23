#include "EngineRuntime.hpp"
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

namespace astraea {

class LocalLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kERROR) {
            std::cerr << "[EngineRuntime TRT] " << msg << std::endl;
        }
    }
} gLocalLogger;

EngineRuntime::EngineRuntime() {}

EngineRuntime::~EngineRuntime() {
    stop();
    if (trt_context_) delete trt_context_;
    if (trt_engine_) delete trt_engine_;
    if (trt_runtime_) delete trt_runtime_;
    if (gpu_input_buffer_) cudaFree(gpu_input_buffer_);
    if (gpu_output_buffer_) cudaFree(gpu_output_buffer_);
}

void EngineRuntime::pin_thread_to_core(std::thread& t, int core_id) {
#ifdef _WIN32
    HANDLE hThread = t.native_handle();
    DWORD_PTR mask = (1ULL << core_id);
    SetThreadAffinityMask(hThread, mask);
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
#endif
}

bool EngineRuntime::initialize(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open TRT engine file: " << engine_path << std::endl;
        return false;
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) return false;

    trt_runtime_ = nvinfer1::createInferRuntime(gLocalLogger);
    if (!trt_runtime_) return false;

    trt_engine_ = trt_runtime_->deserializeCudaEngine(buffer.data(), size);
    if (!trt_engine_) return false;

    trt_context_ = trt_engine_->createExecutionContext();
    if (!trt_context_) return false;

    cudaMalloc(&gpu_input_buffer_, 1 * 10 * 40 * sizeof(float));
    cudaMalloc(&gpu_output_buffer_, 1 * 3 * sizeof(float));

    return true;
}

bool EngineRuntime::enqueue_event(const MarketEvent& event) {
    return ingestion_queue_.try_push(event);
}

void EngineRuntime::start() {
    running_.store(true, std::memory_order_release);
    
    matching_thread_ = std::thread(&EngineRuntime::matching_loop, this);
    pin_thread_to_core(matching_thread_, 1);
    
    inference_thread_ = std::thread(&EngineRuntime::inference_loop, this);
    pin_thread_to_core(inference_thread_, 2);
}

void EngineRuntime::stop() {
    running_.store(false, std::memory_order_release);
    if (matching_thread_.joinable()) matching_thread_.join();
    if (inference_thread_.joinable()) inference_thread_.join();
}

void EngineRuntime::matching_loop() {
    MarketEvent event;
    while (running_.load(std::memory_order_acquire)) {
        if (ingestion_queue_.try_pop(event)) {
            order_book_.process(event);
            
            FeatureFrame frame{};
            frame.timestamp = event.timestamp;
            // Omitted: populating the 40 features
            
            while (!feature_queue_.try_push(frame) && running_.load(std::memory_order_acquire)) {}
        }
    }
}

void EngineRuntime::inference_loop() {
    FeatureFrame frame;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    void* bindings[] = { gpu_input_buffer_, gpu_output_buffer_ };

    while (running_.load(std::memory_order_acquire)) {
        if (feature_queue_.try_pop(frame)) {
            cudaMemcpyAsync(gpu_input_buffer_, frame.features.data(), frame.features.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
            trt_context_->enqueueV2(bindings, stream, nullptr);
            
            float out[3];
            cudaMemcpyAsync(out, gpu_output_buffer_, 3 * sizeof(float), cudaMemcpyDeviceToHost, stream);
            
            cudaStreamSynchronize(stream);

            auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            uint64_t latency = now - frame.timestamp;
            
            uint64_t idx = processed_count_.fetch_add(1, std::memory_order_acq_rel);
            if (idx < latencies_.size()) {
                latencies_[idx] = latency;
            }
        }
    }
    cudaStreamDestroy(stream);
}

} // namespace astraea
