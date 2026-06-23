#include <benchmark/benchmark.h>
#include <SPSCQueue.hpp>
#include <Types.hpp>

static void BM_QueuePushPop(benchmark::State& state) {
    SPSCQueue<MarketEvent, 1024> queue;
    MarketEvent event{123456789ULL, 1ULL, 100, 50000, 'B', 'A'};
    for (auto _ : state) {
        if (queue.try_push(event)) {
            MarketEvent popped;
            queue.try_pop(popped);
            benchmark::DoNotOptimize(popped);
        }
    }
}
BENCHMARK(BM_QueuePushPop);

BENCHMARK_MAIN();
