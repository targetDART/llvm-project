//===- ThreadLocalCache.h - ThreadLocalCache class --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a definition of the ThreadLocalCache class. This class
// provides support for defining thread local objects with non-static duration.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_THREADLOCALCACHE_H
#define MLIR_SUPPORT_THREADLOCALCACHE_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Mutex.h"

namespace mlir {
/// This class provides support for defining a thread local object with non
/// static storage duration. This is very useful for situations in which a data
/// cache has very large lock contention.
template <typename ValueT>
class ThreadLocalCache {
  // Keep a separate shared_ptr protected state that can be acquired atomically
  // instead of using shared_ptr's for each value. This avoids a problem
  // where the instance shared_ptr is locked() successfully, and then the
  // ThreadLocalCache gets destroyed before remove() can be called successfully.
  struct PerInstanceState {
    /// Remove the given value entry. This is generally called when a thread
    /// local cache is destructing.
    void remove(ValueT *value) {
      // Erase the found value directly, because it is guaranteed to be in the
      // list.
      llvm::sys::SmartScopedLock<true> threadInstanceLock(instanceMutex);
      auto it =
          llvm::find_if(instances, [&](std::unique_ptr<ValueT> &instance) {
            return instance.get() == value;
          });
      assert(it != instances.end() && "expected value to exist in cache");
      instances.erase(it);
    }

    /// Owning pointers to all of the values that have been constructed for this
    /// object in the static cache.
    SmallVector<std::unique_ptr<ValueT>, 1> instances;

    /// A mutex used when a new thread instance has been added to the cache for
    /// this object.
    llvm::sys::SmartMutex<true> instanceMutex;
  };

  /// The type used for the static thread_local cache. This is a map between an
  /// instance of the non-static cache and a weak reference to an instance of
  /// ValueT. We use a weak reference here so that the object can be destroyed
  /// without needing to lock access to the cache itself.
  struct CacheType
      : public llvm::SmallDenseMap<PerInstanceState *,
                                   std::pair<std::weak_ptr<ValueT>, ValueT *>> {
    ~CacheType() {
      // Remove the values of this cache that haven't already expired.
      for (auto &it : *this)
        if (std::shared_ptr<ValueT> value = it.second.first.lock())
          it.first->remove(value.get());
    }

    /// Clear out any unused entries within the map. This method is not
    /// thread-safe, and should only be called by the same thread as the cache.
    void clearExpiredEntries() {
      for (auto it = this->begin(), e = this->end(); it != e;) {
        auto curIt = it++;
        if (curIt->second.first.expired())
          this->erase(curIt);
      }
    }
  };

public:
  ThreadLocalCache() = default;
  ~ThreadLocalCache() {
    // No cleanup is necessary here as the shared_pointer memory will go out of
    // scope and invalidate the weak pointers held by the thread_local caches.
  }

  /// Return an instance of the value type for the current thread.
  ValueT &get() {
    // Check for an already existing instance for this thread.
    CacheType &staticCache = getStaticCache();
    std::pair<std::weak_ptr<ValueT>, ValueT *> &threadInstance =
        staticCache[perInstanceState.get()];
    if (ValueT *value = threadInstance.second)
      return *value;

    // Otherwise, create a new instance for this thread.
    {
      llvm::sys::SmartScopedLock<true> threadInstanceLock(
          perInstanceState->instanceMutex);
      threadInstance.second =
          perInstanceState->instances.emplace_back(std::make_unique<ValueT>())
              .get();
    }
    threadInstance.first =
        std::shared_ptr<ValueT>(perInstanceState, threadInstance.second);

    // Before returning the new instance, take the chance to clear out any used
    // entries in the static map. The cache is only cleared within the same
    // thread to remove the need to lock the cache itself.
    staticCache.clearExpiredEntries();
    return *threadInstance.second;
  }
  ValueT &operator*() { return get(); }
  ValueT *operator->() { return &get(); }

private:
  ThreadLocalCache(ThreadLocalCache &&) = delete;
  ThreadLocalCache(const ThreadLocalCache &) = delete;
  ThreadLocalCache &operator=(const ThreadLocalCache &) = delete;

  /// Return the static thread local instance of the cache type.
  static CacheType &getStaticCache() {
    static LLVM_THREAD_LOCAL CacheType cache;
    return cache;
  }

  std::shared_ptr<PerInstanceState> perInstanceState =
      std::make_shared<PerInstanceState>();
};
} // namespace mlir

#endif // MLIR_SUPPORT_THREADLOCALCACHE_H
