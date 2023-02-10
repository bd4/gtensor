#ifndef GTENSOR_BACKEND_SYCL_DEVICE_H
#define GTENSOR_BACKEND_SYCL_DEVICE_H

#include <cstdlib>
#include <exception>
#include <forward_list>
#include <iostream>
#include <unordered_map>

#include "backend_sycl_compat.h"

// ======================================================================
// gt::backend::sycl::device

namespace gt
{
namespace backend
{

namespace sycl
{

namespace device
{

inline auto get_exception_handler()
{
  static auto exception_handler = [](::sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (::sycl::exception const& e) {
        std::cerr << "Caught asynchronous SYCL exception:" << std::endl
                  << e.what() << std::endl;
        abort();
      }
    }
  };
  return exception_handler;
}

inline bool device_per_tile_enabled()
{
  static bool init = false;
  static bool enabled;
  if (!init) {
    enabled =
      (std::getenv("GTENSOR_DEVICE_SYCL_DISABLE_SUB_DEVICES") == nullptr);
  }
  return enabled;
}

inline uint32_t get_unique_device_id_sycl(int device_index,
                                          const ::sycl::device& d)
{
#if __INTEL_CLANG_COMPILER
  if (d.has(::sycl::aspect::ext_intel_device_info_uuid)) {
    auto UUID = d.get_info<::sycl::ext::intel::info::device::uuid>();
    uint32_t unique_id = UUID[0];
    unique_id |= UUID[1] << 8;
    unique_id |= UUID[2] << 16;
    unique_id |= UUID[3] << 24;
    return unique_id;
  } else if (d.has(::sycl::aspect::ext_intel_pci_address)) {
    uint32_t unique_id = 0;
    auto BDF = d.get_info<::sycl::ext::intel::info::device::pci_address>();
    std::cout << "bdf " << BDF << std::endl;
    unique_id |= (std::stoi(BDF.substr(0, 2)) << 16);
    unique_id |= (std::stoi(BDF.substr(3, 5)) << 8);
    unique_id |= std::stoi(BDF.substr(6, 8));
    return unique_id;
  }
#endif
  // NOTE: this will be unique, but is not useful for it's intended
  // purpose of varifying the MPI -> GPU mapping, since it would work
  // even if the runtime returned the same device multiple times.
  return d.get_info<::sycl::info::device::vendor_id>() + device_index;
}

inline std::vector<::sycl::device> get_devices_with_numa_sub(
  const ::sycl::platform& p)
{
  std::vector<::sycl::device> result;
  for (auto root_dev : p.get_devices()) {
    // Handle GPUs with multiple tiles, which can be partitioned based on numa
    // domain
    if (device_per_tile_enabled()) {
      auto max_sub_devices =
        root_dev.get_info<::sycl::info::device::partition_max_sub_devices>();
      auto aff_domains =
        root_dev.get_info<::sycl::info::device::partition_affinity_domains>();
      // NB: device type check is a workaround for bug in host backend, where
      // max > 0 but it's not supported
      if (max_sub_devices > 0 && aff_domains.size() > 0 &&
          (root_dev.is_gpu() || root_dev.is_cpu())) {
        auto sub_devices = root_dev.create_sub_devices<
          ::sycl::info::partition_property::partition_by_affinity_domain>(
          ::sycl::info::partition_affinity_domain::numa);
        for (auto sub_dev : sub_devices) {
          result.push_back(sub_dev);
        }
      }
    }
    if (result.size() == 0) {
      result.push_back(root_dev);
    }
  }
  return result;
}

class SyclQueues
{
public:
  SyclQueues() : current_device_id_(0)
  {
    // Get devices from default platform, which for Intel implementation
    // can be controlled with SYCL_DEVICE_FILTER env variable. This
    // allows flexible selection at runtime.
#if (__INTEL_CLANG_COMPILER && __INTEL_CLANG_COMPILER < 20230000)
    ::sycl::platform p{::sycl::default_selector()};
#else
    ::sycl::platform p{::sycl::default_selector_v};
#endif

    devices_ = get_devices_with_numa_sub(p);

    // Use global singleton context for all queues, to make sure memory
    // allocated on different queues with same device work as expected (similar
    // to CUDA). Note that l0 backend does not need this, but OpenCL does,
    // so more portable to do it this way.
    context_ = ::sycl::context(devices_, get_exception_handler());
  }

  void valid_device_id_or_throw(int device_id)
  {
    if (device_id >= devices_.size() || device_id < 0) {
      throw std::runtime_error("No such device");
    }
  }

  ::sycl::queue& get_queue(int device_id)
  {
    valid_device_id_or_throw(device_id);
    if (default_queue_map_.count(device_id) == 0) {
      default_queue_map_[device_id] = ::sycl::queue{
        context_, devices_[device_id], ::sycl::property::queue::in_order()};
    }
    return default_queue_map_[device_id];
  }

  ::sycl::queue& new_stream_queue(int device_id)
  {
    valid_device_id_or_throw(device_id);
    stream_queue_map_[device_id].emplace_front(
      context_, devices_[device_id], ::sycl::property::queue::in_order());
    return stream_queue_map_[device_id].front();
  }

  ::sycl::queue& new_stream_queue()
  {
    return new_stream_queue(current_device_id_);
  }

  void delete_stream_queue(int device_id, ::sycl::queue& q)
  {
    valid_device_id_or_throw(device_id);
    stream_queue_map_[device_id].remove(q);
  }

  void delete_stream_queue(::sycl::queue& q)
  {
    delete_stream_queue(current_device_id_, q);
  }

  int get_device_count() { return devices_.size(); }

  void set_device_id(int device_id)
  {
    valid_device_id_or_throw(device_id);
    current_device_id_ = device_id;
  }

  uint32_t get_device_vendor_id(int device_id)
  {
    valid_device_id_or_throw(device_id);
    const ::sycl::device& sycl_dev = devices_[device_id];
    return get_unique_device_id_sycl(device_id, sycl_dev);
  }

  int get_device_id() { return current_device_id_; }

  bool has_open_stream_queues(int device_id)
  {
    return !stream_queue_map_[device_id].empty();
  }

  bool has_open_stream_queues()
  {
    return has_open_stream_queues(current_device_id_);
  }

  ::sycl::queue& get_queue() { return get_queue(current_device_id_); }

  bool is_host_backend()
  {
    return (devices_[0].get_info<::sycl::info::device::device_type>() ==
            ::sycl::info::device_type::host);
  }

private:
  ::sycl::context context_;
  std::vector<::sycl::device> devices_;
  std::unordered_map<int, ::sycl::queue> default_queue_map_;
  std::unordered_map<int, std::forward_list<::sycl::queue>> stream_queue_map_;
  int current_device_id_;
};

inline SyclQueues& get_sycl_queues_instance()
{
  static SyclQueues queues;
  return queues;
}

} // namespace device

/*! Get the global singleton queue object used for all device operations.  */
inline ::sycl::queue& get_queue()
{
  return device::get_sycl_queues_instance().get_queue();
}

inline ::sycl::queue& get_queue(int device_id)
{
  return device::get_sycl_queues_instance().get_queue(device_id);
}

/*! Get a new queue different from the default, for use like alternate streams.
 */
inline ::sycl::queue& new_stream_queue()
{
  return device::get_sycl_queues_instance().new_stream_queue();
}

inline ::sycl::queue& new_stream_queue(int device_id)
{
  return device::get_sycl_queues_instance().new_stream_queue(device_id);
}

/*! Get a new queue different from the default, for use like alternate streams.
 */
inline void delete_stream_queue(::sycl::queue& q)
{
  device::get_sycl_queues_instance().delete_stream_queue(q);
}

inline void delete_stream_queue(int device_id, ::sycl::queue& q)
{
  device::get_sycl_queues_instance().delete_stream_queue(device_id, q);
}

/*! Get a new queue different from the default, for use like alternate streams.
 */
inline bool has_open_stream_queues(int device_id)
{
  return device::get_sycl_queues_instance().has_open_stream_queues(device_id);
}

inline bool has_open_stream_queues()
{
  return device::get_sycl_queues_instance().has_open_stream_queues();
}

inline bool is_host_backend()
{
  return device::get_sycl_queues_instance().is_host_backend();
}

#ifdef GTENSOR_DEVICE_SYCL_L0

inline void mem_info(size_t* free, size_t* total)
{
  zes_mem_state_t memory_props{
    ZES_STRUCTURE_TYPE_MEM_PROPERTIES,
  };

  auto q = get_queue();
  auto d = q.get_device();

  // Get level-zero device handle
  auto ze_dev = ::sycl::get_native<::sycl::backend::ext_oneapi_level_zero>(d);

  uint32_t n_mem_modules = 1;
  std::vector<zes_mem_handle_t> module_list(n_mem_modules);
  zesDeviceEnumMemoryModules(ze_dev, &n_mem_modules, module_list.data());

  zesMemoryGetState(module_list[0], &memory_props);
  *total = memory_props.size;
  *free = memory_props.free;
}

#else // no GTENSOR_DEVICE_SYCL_L0

inline void mem_info(size_t* free, size_t* total)
{
  *free = 0;
  *total = 0;
}

#endif // GTENSOR_DEVICE_SYCL_L0

} // namespace sycl

} // namespace backend

} // namespace gt

#endif // GTENSOR_BACKEND_SYCL_DEVICE_H
