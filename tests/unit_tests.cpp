#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <SPSCQueue.hpp>
#include <Types.hpp>

TEST(SPSCQueueTest, SingleThreadedBasic) {
    SPSCQueue<int, 4> queue;
    EXPECT_TRUE(queue.empty());
    EXPECT_EQ(queue.size(), static_cast<size_t>(0));

    EXPECT_TRUE(queue.try_push(42));
    EXPECT_TRUE(queue.try_push(43));
    EXPECT_TRUE(queue.try_push(44));
    EXPECT_TRUE(queue.try_push(45));
    EXPECT_FALSE(queue.try_push(46)); // Full

    EXPECT_EQ(queue.size(), static_cast<size_t>(4));

    int val = 0;
    EXPECT_TRUE(queue.try_pop(val));
    EXPECT_EQ(val, 42);
    EXPECT_TRUE(queue.try_pop(val));
    EXPECT_EQ(val, 43);

    EXPECT_EQ(queue.size(), static_cast<size_t>(2));
}

TEST(SPSCQueueTest, ConcurrentPushPop) {
    constexpr size_t Capacity = 4096;
    constexpr uint64_t NumEvents = 1000000;
    SPSCQueue<MarketEvent, Capacity> queue;

    std::thread producer([&]() {
        for (uint64_t i = 0; i < NumEvents; ++i) {
            MarketEvent event{
                .timestamp = 1000 + i,
                .order_id = i,
                .quantity = static_cast<uint32_t>(10 + i % 100),
                .price = static_cast<uint32_t>(50000 + i % 1000),
                .side = (i % 2 == 0) ? 'B' : 'A',
                .action = 'A'
            };
            while (!queue.try_push(event)) {
                std::this_thread::yield();
            }
        }
    });

    std::thread consumer([&]() {
        for (uint64_t i = 0; i < NumEvents; ++i) {
            MarketEvent event{};
            while (!queue.try_pop(event)) {
                std::this_thread::yield();
            }
            EXPECT_EQ(event.order_id, i);
            EXPECT_EQ(event.timestamp, 1000 + i);
            EXPECT_EQ(event.quantity, static_cast<uint32_t>(10 + i % 100));
            EXPECT_EQ(event.price, static_cast<uint32_t>(50000 + i % 1000));
            EXPECT_EQ(event.side, (i % 2 == 0) ? 'B' : 'A');
            EXPECT_EQ(event.action, 'A');
        }
    });

    producer.join();
    consumer.join();
}
