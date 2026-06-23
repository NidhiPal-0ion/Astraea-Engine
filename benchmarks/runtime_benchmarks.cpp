#include "EngineRuntime.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace astraea;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: astraea_runtime <engine_path>\n";
        return 1;
    }

    std::string engine_path = argv[1];
    EngineRuntime runtime;
    
    if (!runtime.initialize(engine_path)) {
        std::cerr << "Engine Initialization Failed.\n";
        return 1;
    }

    constexpr size_t NumEvents = 1000000;
    runtime.allocate_latency_tracking(NumEvents);

    std::cout << "Starting Astraea-Engine Runtime Simulation...\n";
    runtime.start();

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < NumEvents; ++i) {
        MarketEvent event{};
        event.timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        event.order_id = i + 1; // 1-indexed to avoid 0 if needed
        event.price = 50000 + (i % 100);
        event.quantity = 10;
        event.side = (i % 2 == 0) ? 'B' : 'A';
        event.action = 'A';

        while (!runtime.enqueue_event(event)) {
            // spin if queue is full
        }
    }

    // Wait for all to be processed
    while (runtime.get_processed_count() < NumEvents) {
        std::this_thread::yield();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    runtime.stop();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    double events_per_sec = (static_cast<double>(NumEvents) / duration) * 1000000.0;

    std::vector<uint64_t> latencies;
    for (size_t i = 0; i < NumEvents; ++i) {
        latencies.push_back(runtime.get_latency_ns(i));
    }

    std::sort(latencies.begin(), latencies.end());

    uint64_t p50 = latencies[NumEvents * 0.5];
    uint64_t p95 = latencies[NumEvents * 0.95];
    uint64_t p99 = latencies[NumEvents * 0.99];

    std::cout << "\n| Metric | Result |\n";
    std::cout << "|---|---|\n";
    std::cout << "| Total Throughput | " << events_per_sec << " Events/sec |\n";
    std::cout << "| Median Latency (p50) | " << p50 << " ns |\n";
    std::cout << "| p95 Latency | " << p95 << " ns |\n";
    std::cout << "| p99 Latency | " << p99 << " ns |\n";

    return 0;
}
