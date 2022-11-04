#ifndef GTENSOR_COMPLEX_OPS_H
#define GTENSOR_COMPLEX_OPS_H

#include "device_runtime.h"
#include "macros.h"

#if defined(GTENSOR_DEVICE_SYCL)
#include "sycl_ext_complex.h"
#endif

namespace gt
{

#if defined(GTENSOR_DEVICE_CUDA) || defined(GTENSOR_DEVICE_HIP)

template <typename T>
GT_INLINE T norm(const complex<T>& a)
{
  return thrust::norm(a);
}

template <typename T>
GT_INLINE complex<T> conj(const complex<T>& a)
{
  return thrust::conj(a);
}

template <typename T>
GT_INLINE T norm(thrust::device_reference<complex<T>> a)
{
  return thrust::norm(thrust::raw_reference_cast(a));
}

using std::abs;
using thrust::abs;

template <typename T>
GT_INLINE T abs(thrust::device_reference<complex<T>> a)
{
  return thrust::abs(thrust::raw_reference_cast(a));
}

template <typename T>
GT_INLINE complex<T> conj(thrust::device_reference<complex<T>> a)
{
  return thrust::conj(thrust::raw_reference_cast(a));
}

using std::exp;
using thrust::exp;

template <typename T>
GT_INLINE complex<T> exp(thrust::device_reference<thrust::complex<T>> a)
{
  return thrust::exp(thrust::raw_reference_cast(a));
}

template <typename T>
GT_INLINE complex<T> exp(thrust::device_reference<const thrust::complex<T>> a)
{
  return thrust::exp(thrust::raw_reference_cast(a));
}

#elif defined(GTENSOR_DEVICE_SYCL)

using gt::backend::sycl::ext::cplx::abs;
using gt::backend::sycl::ext::cplx::norm;
using gt::backend::sycl::ext::cplx::exp;
using gt::backend::sycl::ext::cplx::conj;

#else // host

template <typename T>
GT_INLINE T norm(const complex<T>& a)
{
  return std::norm(a);
}

using std::abs;

template <typename T>
GT_INLINE complex<T> conj(const complex<T>& a)
{
  return std::conj(a);
}

using std::exp;

#endif

} // namespace gt

#endif
