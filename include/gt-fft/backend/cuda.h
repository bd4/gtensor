#ifndef GTENSOR_FFT_CUDA_H
#define GTENSOR_FFT_CUDA_H

#include <numeric>
#include <stdexcept>
#include <vector>

// Note: this file is included by fft/hip.h after redef/type aliasing
// all the necessary types and functions.
#ifdef GTENSOR_DEVICE_CUDA
#include <cufft.h>
#endif

namespace gt
{

namespace fft
{

namespace detail
{

template <gt::fft::Domain D, typename R>
struct fft_config;

template <>
struct fft_config<gt::fft::Domain::COMPLEX, double>
{
  constexpr static cufftType type_forward = CUFFT_Z2Z;
  constexpr static cufftType type_inverse = CUFFT_Z2Z;
  constexpr static auto exec_fn_forward = &cufftExecZ2Z;
  constexpr static auto exec_fn_inverse = &cufftExecZ2Z;
  using Tin = gt::complex<double>;
  using Tout = gt::complex<double>;
  using Bin = cufftDoubleComplex;
  using Bout = cufftDoubleComplex;
};

template <>
struct fft_config<gt::fft::Domain::COMPLEX, float>
{
  constexpr static cufftType type_forward = CUFFT_C2C;
  constexpr static cufftType type_inverse = CUFFT_C2C;
  constexpr static auto exec_fn_forward = &cufftExecC2C;
  constexpr static auto exec_fn_inverse = &cufftExecC2C;
  using Tin = gt::complex<float>;
  using Tout = gt::complex<float>;
  using Bin = cufftComplex;
  using Bout = cufftComplex;
};

template <>
struct fft_config<gt::fft::Domain::REAL, double>
{
  constexpr static cufftType type_forward = CUFFT_D2Z;
  constexpr static cufftType type_inverse = CUFFT_Z2D;
  constexpr static auto exec_fn_forward = &cufftExecD2Z;
  constexpr static auto exec_fn_inverse = &cufftExecZ2D;
  using Tin = double;
  using Tout = gt::complex<double>;
  using Bin = cufftDoubleReal;
  using Bout = cufftDoubleComplex;
};

template <>
struct fft_config<gt::fft::Domain::REAL, float>
{
  constexpr static cufftType type_forward = CUFFT_R2C;
  constexpr static cufftType type_inverse = CUFFT_C2R;
  constexpr static auto exec_fn_forward = &cufftExecR2C;
  constexpr static auto exec_fn_inverse = &cufftExecC2R;
  using Tin = float;
  using Tout = gt::complex<float>;
  using Bin = cufftReal;
  using Bout = cufftComplex;
};

} // namespace detail

template <gt::fft::Domain D, typename R>
class FFTPlanManyCUDA;

template <typename R>
class FFTPlanManyCUDA<gt::fft::Domain::REAL, R>
{
  constexpr static gt::fft::Domain D = gt::fft::Domain::REAL;

public:
  FFTPlanManyCUDA(std::vector<int> real_lengths, int batch_size = 1)
    : is_valid_(true)
  {
    int rank = real_lengths.size();
    int* nreal = real_lengths.data();

    std::vector<int> complex_lengths = real_lengths;
    complex_lengths[rank - 1] = real_lengths[rank - 1] / 2 + 1;
    int* ncomplex = complex_lengths.data();

    int idist = std::accumulate(real_lengths.begin(), real_lengths.end(), 1,
                                std::multiplies<int>());
    int odist = std::accumulate(complex_lengths.begin(), complex_lengths.end(),
                                1, std::multiplies<int>());
    auto type_forward = detail::fft_config<D, R>::type_forward;
    auto type_inverse = detail::fft_config<D, R>::type_inverse;
    auto result = cufftPlanMany(&plan_forward_, rank, nreal, nreal, 1, idist,
                                ncomplex, 1, odist, type_forward, batch_size);
    assert(result == CUFFT_SUCCESS);
    auto result2 =
      cufftPlanMany(&plan_inverse_, rank, nreal, ncomplex, 1, odist, nreal, 1,
                    idist, type_inverse, batch_size);
    assert(result2 == CUFFT_SUCCESS);
  }

  // move only
  // delete copy ctor/assign
  FFTPlanManyCUDA(const FFTPlanManyCUDA& other) = delete;
  FFTPlanManyCUDA& operator=(const FFTPlanManyCUDA& other) = delete;

  // custom move to avoid double destroy in moved-from object
  FFTPlanManyCUDA(FFTPlanManyCUDA&& other) : is_valid_(true)
  {
    plan_forward_ = other.plan_forward_;
    plan_inverse_ = other.plan_inverse_;
    other.is_valid_ = false;
  }

  FFTPlanManyCUDA& operator=(FFTPlanManyCUDA&& other)
  {
    plan_forward_ = other.plan_forward_;
    plan_inverse_ = other.plan_inverse_;
    other.is_valid_ = false;
    return *this;
  }

  virtual ~FFTPlanManyCUDA()
  {
    if (is_valid_) {
      cufftDestroy(plan_forward_);
      cufftDestroy(plan_inverse_);
    }
  }

  void operator()(typename detail::fft_config<D, R>::Tin* indata,
                  typename detail::fft_config<D, R>::Tout* outdata) const
  {
    if (!is_valid_) {
      throw std::runtime_error("can't use a moved-from plan");
    }
    using Bin = typename detail::fft_config<D, R>::Bin;
    using Bout = typename detail::fft_config<D, R>::Bout;
    auto bin = reinterpret_cast<Bin*>(indata);
    auto bout = reinterpret_cast<Bout*>(outdata);
    auto fn = detail::fft_config<D, R>::exec_fn_forward;
    auto result = fn(plan_forward_, bin, bout);
    assert(result == CUFFT_SUCCESS);
  }

  void inverse(typename detail::fft_config<D, R>::Tout* indata,
               typename detail::fft_config<D, R>::Tin* outdata) const
  {
    if (!is_valid_) {
      throw std::runtime_error("can't use a moved-from plan");
    }
    using Bin = typename detail::fft_config<D, R>::Bin;
    using Bout = typename detail::fft_config<D, R>::Bout;
    auto bin = reinterpret_cast<Bout*>(indata);
    auto bout = reinterpret_cast<Bin*>(outdata);
    auto fn = detail::fft_config<D, R>::exec_fn_inverse;
    auto result = fn(plan_inverse_, bin, bout);
    assert(result == CUFFT_SUCCESS);
  }

private:
  cufftHandle plan_forward_;
  cufftHandle plan_inverse_;
  bool is_valid_;
};

template <typename R>
class FFTPlanManyCUDA<gt::fft::Domain::COMPLEX, R>
{
  constexpr static gt::fft::Domain D = gt::fft::Domain::COMPLEX;

public:
  FFTPlanManyCUDA(std::vector<int> lengths, int batch_size = 1)
    : is_valid_(true)
  {
    auto type_forward = detail::fft_config<D, R>::type_forward;
    int dist = std::accumulate(lengths.begin(), lengths.end(), 1,
                               std::multiplies<int>());
    auto result =
      cufftPlanMany(&plan_, lengths.size(), lengths.data(), nullptr, 1, dist,
                    nullptr, 1, dist, type_forward, batch_size);
    assert(result == CUFFT_SUCCESS);
  }

  // move only
  // delete copy ctor/assign
  FFTPlanManyCUDA(const FFTPlanManyCUDA& other) = delete;
  FFTPlanManyCUDA& operator=(const FFTPlanManyCUDA& other) = delete;

  // custom move to avoid double destroy in moved-from object
  FFTPlanManyCUDA(FFTPlanManyCUDA&& other) : is_valid_(true)
  {
    plan_ = other.plan_;
    other.is_valid_ = false;
  }

  FFTPlanManyCUDA& operator=(FFTPlanManyCUDA&& other)
  {
    plan_ = other.plan_;
    other.is_valid_ = false;
    return *this;
  }

  virtual ~FFTPlanManyCUDA()
  {
    if (is_valid_) {
      cufftDestroy(plan_);
    }
  }

  void operator()(typename detail::fft_config<D, R>::Tin* indata,
                  typename detail::fft_config<D, R>::Tout* outdata) const
  {
    if (!is_valid_) {
      throw std::runtime_error("can't use a moved-from plan");
    }
    using Bin = typename detail::fft_config<D, R>::Bin;
    using Bout = typename detail::fft_config<D, R>::Bout;
    auto bin = reinterpret_cast<Bin*>(indata);
    auto bout = reinterpret_cast<Bout*>(outdata);
    auto fn = detail::fft_config<D, R>::exec_fn_forward;
    auto result = fn(plan_, bin, bout, CUFFT_FORWARD);
    assert(result == CUFFT_SUCCESS);
  }

  void inverse(typename detail::fft_config<D, R>::Tout* indata,
               typename detail::fft_config<D, R>::Tin* outdata) const
  {
    if (!is_valid_) {
      throw std::runtime_error("can't use a moved-from plan");
    }
    using Bin = typename detail::fft_config<D, R>::Bin;
    using Bout = typename detail::fft_config<D, R>::Bout;
    auto bin = reinterpret_cast<Bout*>(indata);
    auto bout = reinterpret_cast<Bin*>(outdata);
    auto fn = detail::fft_config<D, R>::exec_fn_inverse;
    auto result = fn(plan_, bin, bout, CUFFT_INVERSE);
    assert(result == CUFFT_SUCCESS);
  }

private:
  cufftHandle plan_;
  bool is_valid_;
};

template <gt::fft::Domain D, typename R>
using FFTPlanManyBackend = FFTPlanManyCUDA<D, R>;

} // namespace fft

} // namespace gt

#endif // GTENSOR_FFT_CUDA_H
