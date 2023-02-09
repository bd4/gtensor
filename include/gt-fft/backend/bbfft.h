#ifndef GTENSOR_FFT_BBFFT_H
#define GTENSOR_FFT_BBFFT_H

#include <cassert>
#include <memory>
#include <numeric>
#include <stdexcept>

#include "gtensor/backend_sycl.h"
#include "gtensor/complex.h"

#include "bbfft/sycl/make_plan.hpp"

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
  using Tin = gt::complex<double>;
  using Tout = gt::complex<double>;
  using Bin = std::complex<double>;
  using Bout = std::complex<double>;
  static const bbfft::transform_type transform_forward =
    bbfft::transform_type::c2c;
  static const bbfft::transform_type transform_inverse =
    bbfft::transform_type::c2c;
};

template <>
struct fft_config<gt::fft::Domain::COMPLEX, float>
{
  using Tin = gt::complex<float>;
  using Tout = gt::complex<float>;
  using Bin = std::complex<float>;
  using Bout = std::complex<float>;
  static const bbfft::transform_type transform_forward =
    bbfft::transform_type::c2c;
  static const bbfft::transform_type transform_inverse =
    bbfft::transform_type::c2c;
};

template <>
struct fft_config<gt::fft::Domain::REAL, double>
{
  using Tin = double;
  using Tout = gt::complex<double>;
  using Bin = double;
  using Bout = std::complex<double>;
  static const bbfft::transform_type transform_forward =
    bbfft::transform_type::r2c;
  static const bbfft::transform_type transform_inverse =
    bbfft::transform_type::c2r;
};

template <>
struct fft_config<gt::fft::Domain::REAL, float>
{
  using Tin = float;
  using Tout = gt::complex<float>;
  using Bin = float;
  using Bout = std::complex<float>;
  static const bbfft::transform_type transform_forward =
    bbfft::transform_type::r2c;
  static const bbfft::transform_type transform_inverse =
    bbfft::transform_type::c2r;
};

} // namespace detail

template <gt::fft::Domain D, typename R>
class FFTPlanManyBBFFT
{
  using Config = typename detail::fft_config<D, R>;

public:
  FFTPlanManyBBFFT(std::vector<int> lengths, int batch_size = 1,
                   gt::stream_view stream = gt::stream_view{})
    : q_(stream.get_backend_stream())
  {
    init(lengths, 1, 0, 1, 0, batch_size);
  }

  FFTPlanManyBBFFT(std::vector<int> lengths, int istride, int idist,
                   int ostride, int odist, int batch_size = 1,
                   gt::stream_view stream = gt::stream_view{})
    : q_(stream.get_backend_stream())
  {
    init(lengths, istride, idist, ostride, odist, batch_size);
  }

  // move only
  FFTPlanManyBBFFT(FFTPlanManyBBFFT&& other) = default;
  FFTPlanManyBBFFT& operator=(FFTPlanManyBBFFT&& other) = default;

  // delete copy ctor/assign
  FFTPlanManyBBFFT(const FFTPlanManyBBFFT& other) = delete;
  FFTPlanManyBBFFT& operator=(const FFTPlanManyBBFFT& other) = delete;

  void operator()(typename detail::fft_config<D, R>::Tin* indata,
                  typename detail::fft_config<D, R>::Tout* outdata) const
  {
    if (!plan_forward_) {
      throw std::runtime_error("can't use a moved-from plan");
    }
    using Bin = typename detail::fft_config<D, R>::Bin;
    using Bout = typename detail::fft_config<D, R>::Bout;
    auto bin = reinterpret_cast<Bin*>(indata);
    auto bout = reinterpret_cast<Bout*>(outdata);
    if (rank_ > 1) {
      // only inplace supported by bbfft for 2d/3d
      q_.copy(bin, reinterpret_cast<Bin*>(outdata), rsize_).wait();
      plan_forward_.execute(bout).wait();
      // only inplace supported by bbfft for 2d/3d
      if (cbytes_ >= rbytes_fwd_) {
        std::cout << "cbytes >= rbytes" << std::endl;
        q_.copy(bin, reinterpret_cast<Bin*>(outdata), rsize_).wait();
        plan_forward_.execute(bout).wait();
      } else {
        std::cout << "cbytes < rbytes, using temp" << std::endl;
        auto tmp = sycl::malloc_device<uint8_t>(max_bytes_, q_);
        q_.copy(reinterpret_cast<uint8_t*>(indata), tmp, rbytes_).wait();
        auto e = plan_forward_.execute(tmp);
        q_.copy(tmp, reinterpret_cast<uint8_t*>(outdata), cbytes_, e).wait();
        sycl::free(tmp, q_);
      }
    } else {
      plan_forward_.execute(bin, bout).wait();
    }
  }

  void inverse(typename detail::fft_config<D, R>::Tout* indata,
               typename detail::fft_config<D, R>::Tin* outdata) const
  {
    if (!plan_forward_) {
      throw std::runtime_error("can't use a moved-from plan");
    }
    using Breal = typename detail::fft_config<D, R>::Bin;
    using Bcmplx = typename detail::fft_config<D, R>::Bout;
    auto bin = reinterpret_cast<Bcmplx*>(indata);
    auto bout = reinterpret_cast<Breal*>(outdata);
    if (rank_ > 1) {
      // only inplace supported by bbfft for 2d/3d
      if (rbytes_inv_ >= cbytes_) {
        std::cout << "rbytes >= cbytes" << std::endl;
        q_.copy(bin, reinterpret_cast<Bcmplx*>(bout), csize_).wait();
        plan_inverse_.execute(bout).wait();
      } else {
        std::cout << "rbytes < cbytes, using temp" << std::endl;
        auto tmp = sycl::malloc_device<uint8_t>(max_bytes_, q_);
        q_.copy(reinterpret_cast<uint8_t*>(indata), tmp, cbytes_).wait();
        auto e = plan_inverse_.execute(bout);
        q_.copy(tmp, reinterpret_cast<uint8_t*>(outdata), rbytes_inv_, e).wait();
        sycl::free(tmp, q_);
      }
    } else {
      plan_inverse_.execute(bin, bout).wait();
    }
  }

private:
  void init(std::vector<int> lengths, int istride, int idist, int ostride,
            int odist, int batch_size)
  {
    rank_ = lengths.size();

    // shape is { M, N_1, N_2, N_3, K }. We always use M=1, K is number of
    // batches, and the N's are the dimensions from length vector arg
    std::array<std::size_t, bbfft::max_tensor_dim> shape;
    shape[0] = 1;
    for (int i = 1; i <= rank_; i++) {
      // shape is reverse of lengths, column major vs row major
      shape[i] = lengths[rank_ - i];
    }
    shape[rank_ + 1] = batch_size;
    for (int i = rank_ + 2; i < bbfft::max_tensor_dim; i++) {
      shape[i] = 0;
    }

    std::size_t cshape1;
    std::size_t istrideu = istride;
    std::size_t ostrideu = ostride;
    if constexpr (D == gt::fft::Domain::REAL) {
      cshape1 = shape[1] / 2 + 1;
    } else {
      cshape1 = shape[1];
    }
    std::array<std::size_t, bbfft::max_tensor_dim> rstride = {
      1u, istrideu, istrideu * shape[1]};
    // special input strides for in place r2c for rank > 1
    std::array<std::size_t, bbfft::max_tensor_dim> rstride_fwd = {
      1u, istrideu, istrideu * cshape1 * 2};
    std::array<std::size_t, bbfft::max_tensor_dim> cstride = {
      1u, ostrideu, ostrideu * cshape1};

    if (rank_ > 1 && D == gt::fft::Domain::REAL) {
      // special case, we need to use in place plans which
      // have a different stride structure for r2c
      rstride_fwd[2] = rstride_fwd[1] * cshape1 * 2;
    }

    for (int i = 1; i < rank_; i++) {
      rstride[i + 2] = shape[i + 1] * rstride[i + 1];
      rstride_fwd[i + 2] = shape[i + 1] * rstride_fwd[i + 1];
      cstride[i + 2] = shape[i + 1] * cstride[i + 1];
    }

    if (idist == 0) {
      idist = rstride[rank_ + 1];
    } else {
      rstride[rank_ + 1] = idist;
    }
    if (odist == 0) {
      odist = cstride[rank_ + 1];
    } else {
      cstride[rank_ + 1] = odist;
    }
    int idist_fwd = rstrides_fwd[rank_ + 1];

    std::cout << "rstride {";
    for (int i = 0; i < bbfft::max_tensor_dim; i++) {
      std::cout << " " << rstride[i];
    }
    std::cout << "}" << std::endl;
    std::cout << "cstride {";
    for (int i = 0; i < bbfft::max_tensor_dim; i++) {
      std::cout << " " << cstride[i];
    }
    std::cout << "}" << std::endl;

    using Bin = typename detail::fft_config<D, R>::Bin;
    using Bout = typename detail::fft_config<D, R>::Bout;
    rsize_fwd_ = idist_fwd * batch_size;
    rsize_inv_ = idist * batch_size;
    csize_ = odist * batch_size;
    rbytes_inv_ = rsize_inv_ * sizeof(Bin);
    rbytes_fwd_ = rsize_fwd_ * sizeof(Bin);
    cbytes_ = csize_ * sizeof(Bout);
    max_bytes_ = std::max(rbytes_fwd_, cbytes_);
    std::cout << "shape " << shape[0] << " " << shape[1] << " " << shape[2]
              << " " << shape[3] << shape[4] << std::endl;
    std::cout << "batch " << batch_size << std::endl;
    std::cout << "dist " << idist << " " << idist_fwd << " " << odist << std::endl;
    std::cout << "size " << rsize_inv_ << " " << rsize_fwd_
              << " " << csize_ << std::endl;
    std::cout << "bytes " << rbytes_inv_ << " "
              << rbytes_fwd_ << " " << cbytes_ << std::endl;

    bbfft::configuration cfg_fwd = {
      rank_,                     // dim
      shape,                     // { M, N_1, N_2, ..., K }
      bbfft::to_precision_v<R>,  // precision
      bbfft::direction::forward, // direction
      Config::transform_forward, // type
      rstride_fwd,               // input strides
      cstride                    // output stride
    };
    if (rank_ > 1) {
      cfg_fwd.set_strides_default(true);
    }
    bbfft::configuration cfg_inv = {rank_,
                                    shape,
                                    bbfft::to_precision_v<R>,
                                    bbfft::direction::backward,
                                    Config::transform_inverse,
                                    cstride,
                                    rstride};
    if (rank_ > 1) {
      cfg_inv.set_strides_default(true);
    }

    plan_forward_ = bbfft::make_plan(cfg_fwd, q_);
    plan_inverse_ = bbfft::make_plan(cfg_inv, q_);
  }

  sycl::queue& q_;
  mutable bbfft::plan<::sycl::event> plan_forward_;
  mutable bbfft::plan<::sycl::event> plan_inverse_;
  unsigned int rank_;
  std::size_t rbytes_;
  std::size_t rsize_fwd_;
  std::size_t rsize_inv_;
  std::size_t cbytes_;
  std::size_t csize_;
  std::size_t max_bytes_;
};

template <gt::fft::Domain D, typename R>
using FFTPlanManyBackend = FFTPlanManyBBFFT<D, R>;

} // namespace fft

} // namespace gt

#endif // GTENSOR_FFT_BBFFT_H
