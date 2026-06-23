#include <gtest/gtest.h>
#include <OrderBook.hpp>
#include <vector>

using namespace astraea;

struct Trade {
    uint64_t incoming_id;
    uint32_t price;
    uint32_t qty;
};

class TestOrderBook : public OrderBook<TestOrderBook> {
public:
    std::vector<uint64_t> added;
    std::vector<uint64_t> canceled;
    std::vector<Trade> trades;

    void on_order_added(uint64_t order_id) {
        added.push_back(order_id);
    }
    void on_order_canceled(uint64_t order_id) {
        canceled.push_back(order_id);
    }
    void on_trade(uint64_t incoming_id, uint32_t price, uint32_t qty) {
        trades.push_back({incoming_id, price, qty});
    }
};

TEST(OrderBookTest, AddOrder) {
    TestOrderBook book;
    MarketEvent ev{0, 1, 100, 50000, 'B', 'A'};
    EXPECT_TRUE(book.process(ev));
    EXPECT_EQ(book.added.size(), 1);
    EXPECT_EQ(book.added[0], 1);
}

TEST(OrderBookTest, CancelOrder) {
    TestOrderBook book;
    MarketEvent ev{0, 1, 100, 50000, 'B', 'A'};
    EXPECT_TRUE(book.process(ev));
    
    MarketEvent cancel_ev{0, 1, 0, 0, 'B', 'C'};
    EXPECT_TRUE(book.process(cancel_ev));
    EXPECT_EQ(book.canceled.size(), 1);
    EXPECT_EQ(book.canceled[0], 1);
}

TEST(OrderBookTest, CrossOrders) {
    TestOrderBook book;
    MarketEvent b1{0, 1, 100, 50000, 'B', 'A'};
    EXPECT_TRUE(book.process(b1));

    MarketEvent a1{0, 2, 50, 49999, 'A', 'A'};
    EXPECT_TRUE(book.process(a1));
    
    EXPECT_EQ(book.trades.size(), 1);
    EXPECT_EQ(book.trades[0].incoming_id, 2);
    EXPECT_EQ(book.trades[0].price, 50000); // executed at resting price
    EXPECT_EQ(book.trades[0].qty, 50);

    // B1 has 50 left
    MarketEvent a2{0, 3, 100, 50000, 'A', 'A'};
    EXPECT_TRUE(book.process(a2));

    EXPECT_EQ(book.trades.size(), 2);
    EXPECT_EQ(book.trades[1].incoming_id, 3);
    EXPECT_EQ(book.trades[1].price, 50000);
    EXPECT_EQ(book.trades[1].qty, 50);
}
