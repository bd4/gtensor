
#ifndef GTENSOR_GSTRIDED_H
#define GTENSOR_GSTRIDED_H

#include "defs.h"
#include "expression.h"
#include "strides.h"

namespace gt
{

template <typename D>
struct gtensor_inner_types;

// ======================================================================
// gstrided

template <typename D>
class gstrided : public expression<D>
{
public:
  using derived_type = D;
  using base_type = expression<D>;
  using inner_types = gtensor_inner_types<D>;
  using space_type = typename inner_types::space_type;

  using value_type = typename inner_types::value_type;
  using reference = typename inner_types::reference;
  using const_reference = typename inner_types::const_reference;

  constexpr static size_type dimension() { return inner_types::dimension; }

  using shape_type = gt::shape_type<dimension()>;
  using strides_type = gt::shape_type<dimension()>;

  using base_type::derived;

  gstrided() = default;
  gstrided(const shape_type& shape, const strides_type& strides);

  GT_INLINE int shape(int i) const;
  GT_INLINE const shape_type& shape() const;
  GT_INLINE const strides_type& strides() const;
  GT_INLINE size_type size() const;

  template <typename... Args>
  GT_INLINE const_reference operator()(Args&&... args) const;
  template <typename... Args>
  GT_INLINE reference operator()(Args&&... args);

  GT_INLINE const_reference data_access(size_type i) const;
  GT_INLINE reference data_access(size_type i);

protected:
  template <typename... Args>
  GT_INLINE size_type index(Args&&... args) const;

protected:
  shape_type shape_;
  strides_type strides_;
};

// ----------------------------------------------------------------------
// gstrided implementation

template <typename D>
inline gstrided<D>::gstrided(const shape_type& shape,
                             const strides_type& strides)
  : shape_(shape), strides_(strides)
{}

template <typename D>
inline int gstrided<D>::shape(int i) const
{
  return shape_[i];
}

template <typename D>
inline auto gstrided<D>::shape() const -> const shape_type&
{
  return shape_;
}

template <typename D>
inline auto gstrided<D>::strides() const -> const strides_type&
{
  return strides_;
}

template <typename D>
inline size_type gstrided<D>::size() const
{
  return calc_size(shape());
}

template <typename D>
template <typename... Args>
inline auto gstrided<D>::operator()(Args&&... args) const -> const_reference
{
  return data_access(index(std::forward<Args>(args)...));
}

template <typename D>
template <typename... Args>
inline auto gstrided<D>::operator()(Args&&... args) -> reference
{
  return data_access(index(std::forward<Args>(args)...));
}

template <typename D>
inline auto gstrided<D>::data_access(size_type i) const -> const_reference
{
  return derived().data_access_impl(i);
}

template <typename D>
inline auto gstrided<D>::data_access(size_type i) -> reference
{
  return derived().data_access_impl(i);
}

template <typename D>
template <typename... Args>
inline size_type gstrided<D>::index(Args&&... args) const
{
#ifdef GT_BOUNDSCHECK
  bounds_check(this->shape(), std::forward<Args>(args)...);
#endif
  return calc_index(this->strides_, std::forward<Args>(args)...);
}

} // namespace gt

#endif