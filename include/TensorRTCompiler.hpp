#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <string>
#include <memory>
#include <iostream>

namespace astraea {
namespace ml {

// RAII Deleter for TensorRT objects
struct InferDeleter {
    template <typename T>
    void operator()(T* obj) const {
        delete obj;
    }
};

template <typename T>
using UniquePtr = std::unique_ptr<T, InferDeleter>;

// Custom Logger extending nvinfer1::ILogger
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Suppress info-level messages
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << std::endl;
        } else if (severity == Severity::kINFO) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

class TensorRTCompiler {
public:
    TensorRTCompiler() = default;
    ~TensorRTCompiler() = default;

    // Disallow copy/move
    TensorRTCompiler(const TensorRTCompiler&) = delete;
    TensorRTCompiler& operator=(const TensorRTCompiler&) = delete;
    TensorRTCompiler(TensorRTCompiler&&) = delete;
    TensorRTCompiler& operator=(TensorRTCompiler&&) = delete;

    [[nodiscard]] bool compile(const std::string& onnx_path, const std::string& engine_path);

private:
    TRTLogger logger_;
};

} // namespace ml
} // namespace astraea
