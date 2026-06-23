#include "TensorRTCompiler.hpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: astraea_compiler <input.onnx> <output.engine>\n";
        return 1;
    }

    std::string onnx_path = argv[1];
    std::string engine_path = argv[2];

    astraea::ml::TensorRTCompiler compiler;
    if (compiler.compile(onnx_path, engine_path)) {
        return 0;
    } else {
        std::cerr << "Compilation failed.\n";
        return 1;
    }
}
