program test_managed_fortran
  use,intrinsic :: iso_c_binding
  use gpu_api_m
  implicit none

  integer :: i
  type(C_PTR) :: d_rhs, d_p_g
  complex, dimension(:), contiguous, pointer :: rhs, p_g
  complex, dimension(:), allocatable :: cfgamma
  complex :: a_complex
  integer(C_SIZE_T) :: bytes
  integer(C_INT) :: N

  ! allocate inputs
  N = 16

  bytes = N * C_SIZEOF(a_complex)

  call gpuAllocateManaged(d_rhs, bytes)
  call gpuAllocateManaged(d_p_g, bytes)

  allocate(cfgamma(N))

  call C_F_POINTER(d_rhs, rhs, [N])
  call C_F_POINTER(d_p_g, p_g, [N])

  ! init inputs
  do i=1,N
    cfgamma(i) = i * 1.0 / N
    p_g(i) = 2 * N
  end do

  ! compute result
  rhs = 0.5 * cfgamma * p_g

  print *,rhs

end program test_managed_fortran
