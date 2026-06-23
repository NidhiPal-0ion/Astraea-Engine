#include "OrderBook.hpp"

using namespace astraea;

class TestBook : public OrderBook<TestBook> {
public:
    void on_order_added(uint64_t ) {}
    void on_order_canceled(uint64_t ) {}
    void on_trade(uint64_t , uint32_t , uint32_t ) {}
};

int main() {
    TestBook book;
    MarketEvent event{};
    book.process(event);
    return 0;
}
