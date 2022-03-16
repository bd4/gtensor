program test_managed_fortran
  use,intrinsic :: iso_c_binding
  use gpu_api_m
  implicit none

  integer :: i
  type(C_PTR) :: d_rhs, d_p_g
  complex, dimension(:), contiguous, pointer :: f_rhs, f_p_g
  complex, dimension(:,:,:,:,:,:), contiguous, pointer :: rhs, p_g
  complex, dimension(:,:,:,:,:,:), allocatable :: cfgamma
  complex :: a_complex
  integer(C_SIZE_T) :: bytes
  integer(C_INT) :: N, N1, N2, N3, N4, N5, N6

  N1 = 35
  N2 = 16
  N3 = 24
  N4 = 24
  N5 = 16
  N6 = 2
  N = N1*N2*N3*N4*N5*N6

  ! allocate inputs
  bytes = N * C_SIZEOF(a_complex)

  call gpuAllocateManaged(d_rhs, bytes)
  call gpuAllocateManaged(d_p_g, bytes)

  allocate(cfgamma(N1,N2,N3,N4,N5,N6))

  call C_F_POINTER(d_rhs, f_rhs, [N])
  call C_F_POINTER(d_p_g, f_p_g, [N])

  call C_F_POINTER(d_rhs, rhs, [N1,N2,N3,N4,N5,N6])
  call C_F_POINTER(d_p_g, p_g, [N1,N2,N3,N4,N5,N6])

  cfgamma = 2 * N

  ! init inputs
  do i=1,N
    f_p_g(i) = i * 1.0 / N
  end do

  ! compute result
  ! segfaults in ifort unless stack size is set to unlimited:
  ! ulimit -s unlimited
  rhs = 0.5 * cfgamma * p_g

  print *,f_rhs(1:12)

end program test_managed_fortran
