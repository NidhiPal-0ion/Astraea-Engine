#pragma once
#include <cstdint>

#pragma pack(push, 1)
struct MarketEvent {
    uint64_t timestamp;
    uint64_t order_id;
    uint32_t quantity;
    uint32_t price;
    char side;      // 'B' = Bid, 'A' = Ask
    char action;    // 'A' = Add, 'C' = Cancel, 'E' = Execute
};
#pragma pack(pop)

// Safety static assert to ensure memory footprint is exactly 26 bytes
static_assert(sizeof(MarketEvent) == 26, "MarketEvent must be packed to exactly 26 bytes");
