#include "TensorRTCompiler.hpp"
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace astraea {
namespace ml {

bool TensorRTCompiler::compile(const std::string& onnx_path, const std::string& engine_path) {
    // 1. Create Builder
    UniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(logger_)};
    if (!builder) {
        logger_.log(nvinfer1::ILogger::Severity::kERROR, "Failed to create IBuilder.");
        return false;
    }

    // 2. Create Network with explicit batch flag
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    UniquePtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(explicitBatch)};
    if (!network) {
        logger_.log(nvinfer1::ILogger::Severity::kERROR, "Failed to create INetworkDefinition.");
        return false;
    }

    // 3. Parse ONNX
    UniquePtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, logger_)};
    if (!parser) {
        logger_.log(nvinfer1::ILogger::Severity::kERROR, "Failed to create IParser.");
        return false;
    }

    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        logger_.log(nvinfer1::ILogger::Severity::kERROR, "Failed to parse ONNX file.");
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << parser->getError(i)->desc() << std::endl;
        }
        return false;
    }

    // 4. Create Builder Config
    UniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    if (!config) {
        logger_.log(nvinfer1::ILogger::Severity::kERROR, "Failed to create IBuilderConfig.");
        return false;
    }

    // Allocate 1GB fixed workspace memory pool limit
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

    // 5. Setup Optimization Profile for dynamic batch size
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    auto input = network->getInput(0);
    nvinfer1::Dims dims = input->getDimensions();
    
    // Set dynamic batch profile: min=1, opt=32, max=128
    nvinfer1::Dims minDims = dims; minDims.d[0] = 1;
    nvinfer1::Dims optDims = dims; optDims.d[0] = 32;
    nvinfer1::Dims maxDims = dims; maxDims.d[0] = 128;
    
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, minDims);
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, optDims);
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, maxDims);
    config->addOptimizationProfile(profile);

    // 6. Query hardware platform capability for quantization
    if (builder->platformHasFastInt8()) {
        logger_.log(nvinfer1::ILogger::Severity::kINFO, "Hardware supports fast INT8. Enabling INT8 quantization.");
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        // In a real production deployment, we would attach an Int8EntropyCalibrator here.
    } else if (builder->platformHasFastFp16()) {
        logger_.log(nvinfer1::ILogger::Severity::kINFO, "Hardware supports fast FP16. Enabling FP16 compilation.");
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    } else {
        logger_.log(nvinfer1::ILogger::Severity::kINFO, "Hardware does not support fast INT8 or FP16. Defaulting to FP32.");
    }

    // 7. Build Engine
    logger_.log(nvinfer1::ILogger::Severity::kINFO, "Building serialized engine... This may take a while.");
    UniquePtr<nvinfer1::IHostMemory> serializedModel{builder->buildSerializedNetwork(*network, *config)};
    if (!serializedModel) {
        logger_.log(nvinfer1::ILogger::Severity::kERROR, "Failed to build serialized engine.");
        return false;
    }

    // 8. Write to Disk
    std::ofstream engineFile(engine_path, std::ios::binary);
    if (!engineFile) {
        logger_.log(nvinfer1::ILogger::Severity::kERROR, ("Failed to open engine output file: " + engine_path).c_str());
        return false;
    }
    
    engineFile.write(static_cast<const char*>(serializedModel->data()), serializedModel->size());
    engineFile.close();

    logger_.log(nvinfer1::ILogger::Severity::kINFO, ("Successfully compiled and saved engine to " + engine_path).c_str());
    return true;
}

} // namespace ml
} // namespace astraea
