#pragma once

#include "Types.hpp"
#include <array>
#include <cstdint>
#include <cstddef>
#include <algorithm>

namespace astraea {

struct Order {
    uint64_t order_id;
    uint32_t price;
    uint32_t quantity;
    char side;

    int32_t prev = -1;
    int32_t next = -1;
};

struct PriceLevel {
    int32_t head = -1;
    int32_t tail = -1;
};

// CRTP base for zero-cost polymorphism.
// Derived must implement:
// void on_order_added(uint64_t order_id);
// void on_order_canceled(uint64_t order_id);
// void on_trade(uint64_t incoming_id, uint32_t price, uint32_t qty);
template <typename Derived, size_t MaxOrders = 1000000, size_t MaxPriceLevels = 100000>
class OrderBook {
public:
    OrderBook() {
        for (size_t i = 0; i < MaxOrders; ++i) {
            free_indices_[i] = static_cast<int32_t>(MaxOrders - 1 - i);
        }
        free_count_ = MaxOrders;
        lookup_table_.fill(-1);
    }

    [[nodiscard]] inline bool process(const MarketEvent& event) noexcept {
        switch (event.action) {
            case 'A': return handle_add(event);
            case 'C': return handle_cancel(event);
            case 'E': return handle_execute(event);
            default: return false;
        }
    }

    [[nodiscard]] inline bool handle_add(const MarketEvent& event) noexcept {
        if (event.quantity == 0 || free_count_ == 0 || event.price >= MaxPriceLevels) {
            return false;
        }

        uint32_t remaining_qty = event.quantity;

        if (event.side == 'B') {
            while (remaining_qty > 0 && best_ask_ < MaxPriceLevels && best_ask_ <= event.price) {
                PriceLevel& level = asks_[best_ask_];
                if (!match_level(level, remaining_qty, event.order_id, best_ask_)) {
                    find_next_best_ask();
                }
            }
        } else if (event.side == 'A') {
            while (remaining_qty > 0 && best_bid_ > 0 && best_bid_ >= event.price) {
                PriceLevel& level = bids_[best_bid_];
                if (!match_level(level, remaining_qty, event.order_id, best_bid_)) {
                    find_next_best_bid();
                }
            }
        }

        if (remaining_qty > 0) {
            add_order_to_book(event.order_id, event.price, remaining_qty, event.side);
            static_cast<Derived*>(this)->on_order_added(event.order_id);
        }
        
        return true;
    }

    [[nodiscard]] inline bool handle_cancel(const MarketEvent& event) noexcept {
        uint32_t hash = event.order_id % MaxOrders;
        int32_t order_idx = lookup_table_[hash];
        if (order_idx == -1) return false;

        Order& order = order_pool_[order_idx];
        if (order.order_id != event.order_id || order.quantity == 0) return false;

        remove_from_list(order);
        order.quantity = 0;
        
        free_indices_[free_count_++] = order_idx;
        lookup_table_[hash] = -1;

        static_cast<Derived*>(this)->on_order_canceled(event.order_id);
        return true;
    }

    [[nodiscard]] inline bool handle_execute(const MarketEvent& event) noexcept {
        uint32_t hash = event.order_id % MaxOrders;
        int32_t order_idx = lookup_table_[hash];
        if (order_idx == -1) return false;

        Order& order = order_pool_[order_idx];
        if (order.order_id != event.order_id || order.quantity == 0) return false;

        uint32_t exec_qty = std::min(order.quantity, event.quantity);
        order.quantity -= exec_qty;

        static_cast<Derived*>(this)->on_trade(event.order_id, order.price, exec_qty);

        if (order.quantity == 0) {
            remove_from_list(order);
            free_indices_[free_count_++] = order_idx;
            lookup_table_[hash] = -1;
        }
        return true;
    }

private:
    std::array<Order, MaxOrders> order_pool_;
    std::array<int32_t, MaxOrders> free_indices_;
    size_t free_count_ = 0;
    
    // Flat array mapping: assumes order_id % MaxOrders is unique for active orders
    std::array<int32_t, MaxOrders> lookup_table_;

    std::array<PriceLevel, MaxPriceLevels> bids_{};
    std::array<PriceLevel, MaxPriceLevels> asks_{};

    uint32_t best_bid_ = 0;
    uint32_t best_ask_ = MaxPriceLevels;

    inline bool match_level(PriceLevel& level, uint32_t& remaining_qty, uint64_t incoming_order_id, uint32_t exec_price) noexcept {
        int32_t curr_idx = level.head;
        while (curr_idx != -1 && remaining_qty > 0) {
            Order& resting = order_pool_[curr_idx];
            uint32_t trade_qty = std::min(resting.quantity, remaining_qty);
            
            resting.quantity -= trade_qty;
            remaining_qty -= trade_qty;

            static_cast<Derived*>(this)->on_trade(incoming_order_id, exec_price, trade_qty);

            if (resting.quantity == 0) {
                int32_t next_idx = resting.next;
                remove_from_list(resting);
                free_indices_[free_count_++] = curr_idx;
                lookup_table_[resting.order_id % MaxOrders] = -1;
                curr_idx = next_idx;
            } else {
                break;
            }
        }
        return level.head != -1; // return false if level is depleted
    }

    inline void add_order_to_book(uint64_t order_id, uint32_t price, uint32_t quantity, char side) noexcept {
        int32_t order_idx = free_indices_[--free_count_];
        Order& order = order_pool_[order_idx];
        order.order_id = order_id;
        order.price = price;
        order.quantity = quantity;
        order.side = side;
        order.prev = -1;
        order.next = -1;

        lookup_table_[order_id % MaxOrders] = order_idx;

        if (side == 'B') {
            append_to_level(bids_[price], order_idx);
            if (price > best_bid_) best_bid_ = price;
        } else {
            append_to_level(asks_[price], order_idx);
            if (price < best_ask_) best_ask_ = price;
        }
    }

    inline void append_to_level(PriceLevel& level, int32_t order_idx) noexcept {
        Order& order = order_pool_[order_idx];
        if (level.tail == -1) {
            level.head = level.tail = order_idx;
        } else {
            order.prev = level.tail;
            order_pool_[level.tail].next = order_idx;
            level.tail = order_idx;
        }
    }

    inline void remove_from_list(Order& order) noexcept {
        PriceLevel* level = nullptr;
        if (order.side == 'B') {
            level = &bids_[order.price];
        } else {
            level = &asks_[order.price];
        }

        if (order.prev != -1) {
            order_pool_[order.prev].next = order.next;
        } else {
            level->head = order.next;
        }

        if (order.next != -1) {
            order_pool_[order.next].prev = order.prev;
        } else {
            level->tail = order.prev;
        }

        if (level->head == -1) {
            if (order.side == 'B' && order.price == best_bid_) {
                find_next_best_bid();
            } else if (order.side == 'A' && order.price == best_ask_) {
                find_next_best_ask();
            }
        }
    }

    inline void find_next_best_bid() noexcept {
        while (best_bid_ > 0 && bids_[best_bid_].head == -1) {
            --best_bid_;
        }
    }

    inline void find_next_best_ask() noexcept {
        while (best_ask_ < MaxPriceLevels && asks_[best_ask_].head == -1) {
            ++best_ask_;
        }
    }
};

} // namespace astraea
