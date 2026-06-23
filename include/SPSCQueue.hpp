#pragma once
#include <atomic>
#include <cstddef>
#include <new>
#include <utility>
#include <array>
#include <type_traits>

template <typename T, size_t Capacity>
class SPSCQueue {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");

    // Standard aligned storage slot for C++23 (replacing deprecated std::aligned_storage)
    struct alignas(T) StorageSlot {
        std::byte data[sizeof(T)];
    };

public:
    SPSCQueue() : head_(0), tail_(0) {}
    ~SPSCQueue() = default;

    // Disable copying and moving to preserve memory pinning
    SPSCQueue(const SPSCQueue&) = delete;
    SPSCQueue& operator=(const SPSCQueue&) = delete;
    SPSCQueue(SPSCQueue&&) = delete;
    SPSCQueue& operator=(SPSCQueue&&) = delete;

    template <typename... Args>
    bool try_emplace(Args&&... args) {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        const size_t current_head = head_.load(std::memory_order_acquire);

        if ((current_tail - current_head) >= Capacity) {
            return false; // Queue is full
        }

        const size_t index = current_tail & (Capacity - 1);
        new (storage_[index].data) T(std::forward<Args>(args)...);
        tail_.store(current_tail + 1, std::memory_order_release);
        return true;
    }

    bool try_push(const T& value) {
        return try_emplace(value);
    }

    bool try_push(T&& value) {
        return try_emplace(std::move(value));
    }

    bool try_pop(T& value) {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        const size_t current_tail = tail_.load(std::memory_order_acquire);

        if (current_head == current_tail) {
            return false; // Queue is empty
        }

        const size_t index = current_head & (Capacity - 1);
        T* item_ptr = reinterpret_cast<T*>(storage_[index].data);
        value = std::move(*item_ptr);
        
        // Destruct the moved-from element in-place if not trivially destructible
        if constexpr (!std::is_trivially_destructible_v<T>) {
            item_ptr->~T();
        }
        
        head_.store(current_head + 1, std::memory_order_release);
        return true;
    }

    bool empty() const noexcept {
        return head_.load(std::memory_order_relaxed) == tail_.load(std::memory_order_relaxed);
    }

    size_t size() const noexcept {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t tail = tail_.load(std::memory_order_relaxed);
        return (tail >= head) ? (tail - head) : 0;
    }

private:
    // Strict hardware cache line isolation layouts to prevent adjacent-line false sharing
    alignas(64) std::array<StorageSlot, Capacity> storage_;
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
    alignas(64) char padding_[64]; 
};
