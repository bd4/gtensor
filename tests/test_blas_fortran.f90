!#include "redef.h"
#ifdef DOUBLE_PREC
#define gpublas_saxpy gpublas_daxpy
#define gpublas_caxpy gpublas_zaxpy
#endif

program test_blas_fortran
  use gpu_axpy
  implicit none

  integer :: i
  integer(C_INT) :: N
  real :: a
  !real(C_REAL_TYPE), dimension(10) :: x, y
  type(C_PTR) :: d_x, d_y
  real, dimension(:), contiguous, pointer :: x, y
  real, dimension(10) :: cpu_axpy 
  integer(C_SIZE_T) :: bytes

  complex :: ca
  !complex(C_REAL_TYPE), dimension(10) :: x, y
  type(C_PTR) :: d_cx, d_cy
  complex, dimension(:), contiguous, pointer :: cx, cy
  complex, dimension(10) :: cpu_caxpy 
  integer(C_SIZE_T) :: cbytes

  N = 10

  bytes = int(sizeof(a), C_SIZE_T) * N
  cbytes = int(sizeof(ca), C_SIZE_T) * N

  print *, "bytes", bytes
  print *, "cbytes", cbytes

  a = 0.5
  ca = cmplx(0.5, 1.0)

  print *, sizeof(a), c_sizeof(a), storage_size(a)
  print *, sizeof(ca), c_sizeof(ca), storage_size(ca)

  d_x = gt_backend_managed_allocate(bytes)
  d_y = gt_backend_managed_allocate(bytes)

  d_cx = gt_backend_managed_allocate(cbytes)
  d_cy = gt_backend_managed_allocate(cbytes)

  call C_F_POINTER(d_x, x, [N])
  call C_F_POINTER(d_y, y, [N])

  call C_F_POINTER(d_cx, cx, [N])
  call C_F_POINTER(d_cy, cy, [N])

  do i=1,N
    x(i) = i + 0.0
    y(i) = 1.0
    cx(i) = cmplx(2.0 * i, -2.0 * i);
    cy(i) = cmplx(1.0 * i, -1.0 * i);
  end do

  cpu_axpy = a * x + y
  cpu_caxpy = ca * cx + cy

  call gpublas_create()
  call gpublas_saxpy(N, a, x, 1, y, 1)
  call gt_synchronize()

  print *, "real  ", sum(cpu_axpy - y)

  call gpublas_caxpy(N, ca, cx, 1, cy, 1)
  call gt_synchronize()
  call gpublas_destroy()

  print *, "cmplx ", sum(cpu_caxpy - cy)

  !print *, "a", a
  !print *, "x", x
  !print *, "y", y
  !print *, "expected", cpu_axpy
  !print *, ""


  !print *, "ca", ca
  !print *, "cx", cx
  !print *, "cy", cy
  !print *, "cexpected", cpu_caxpy

end program test_blas_fortran
