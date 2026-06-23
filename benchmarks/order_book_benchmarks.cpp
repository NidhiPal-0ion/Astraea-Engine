#include <benchmark/benchmark.h>
#include <OrderBook.hpp>
#include <vector>

using namespace astraea;

class BenchOrderBook : public OrderBook<BenchOrderBook> {
public:
    inline void on_order_added(uint64_t /*order_id*/) noexcept {}
    inline void on_order_canceled(uint64_t /*order_id*/) noexcept {}
    inline void on_trade(uint64_t /*incoming_id*/, uint32_t /*price*/, uint32_t /*qty*/) noexcept {}
};

static void BM_OrderBook_AddCancel(benchmark::State& state) {
    BenchOrderBook book;
    MarketEvent add_ev{0, 1, 100, 50000, 'B', 'A'};
    MarketEvent cancel_ev{0, 1, 0, 0, 'B', 'C'};
    for (auto _ : state) {
        book.handle_add(add_ev);
        book.handle_cancel(cancel_ev);
    }
}
BENCHMARK(BM_OrderBook_AddCancel);

static void BM_OrderBook_Cross(benchmark::State& state) {
    BenchOrderBook book;
    uint64_t id = 1;
    for (auto _ : state) {
        state.PauseTiming();
        MarketEvent bid{0, id++, 100, 50000, 'B', 'A'};
        MarketEvent ask{0, id++, 100, 50000, 'A', 'A'};
        state.ResumeTiming();
        
        book.handle_add(bid);
        book.handle_add(ask);
    }
}
BENCHMARK(BM_OrderBook_Cross);

BENCHMARK_MAIN();
