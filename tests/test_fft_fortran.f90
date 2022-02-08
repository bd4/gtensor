#include "redef.h"

program test_fft_fortran
  use gpufft_m
  use gpu_axpy
  implicit none

  integer :: i
  type(C_PTR) :: d_A, d_A2, d_B
  real, dimension(:), contiguous, pointer :: A, A2
  complex, dimension(:), contiguous, pointer :: B
  integer(C_SIZE_T) :: Abytes, Bbytes

  type(C_PTR) :: plan
  integer(C_INT) :: N, Nout, rank, batch_size
  integer(C_INT), dimension(1), target :: lengths

  N = 4
  Nout = N / 2 + 1
  rank = 1
  batch_size = 2

  Abytes = N * batch_size
  Bbytes = Nout * batch_size

  lengths(1) = N

  print *, "Abytes", Abytes
  print *, "Bbytes", Bbytes

  d_A = gt_backend_managed_allocate(Abytes)
  d_A2 = gt_backend_managed_allocate(Abytes)
  d_B = gt_backend_managed_allocate(Bbytes)

  call C_F_POINTER(d_A, A, [8])
  call C_F_POINTER(d_A2, A2, [8])
  call C_F_POINTER(d_B, B, [6])

  A(1) = 2
  A(2) = 3
  A(3) = -1
  A(4) = 4

  A(5) = 7
  A(6) = -21
  A(7) = 11
  A(8) = 1

  plan = gpufft_new_real_float(rank, c_loc(lengths), 1, N, 1, Nout, batch_size)

  call gpufft_rc(plan, d_A, d_B)
  call gpufft_inverse_cr(plan, d_B, d_A2)

  call gt_synchronize()

  call gpufft_delete_real_float(plan)

  print *, "A ", A
  print *, "B ", B
  print *, "A2", A2 / N

end program test_fft_fortran
