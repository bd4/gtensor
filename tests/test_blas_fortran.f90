!#include "redef.h"
#ifdef DOUBLE_PREC
#define gpublas_saxpy gpublas_daxpy
#define gpublas_caxpy gpublas_zaxpy
#endif

#ifdef GPU_BLAS_I64
#define SIZE_OF_GPU_BLAS_INT 8
#else
#define SIZE_OF_GPU_BLAS_INT 4
#endif

#define SIZE_OF_INTEGER 4
#define SIZE_OF_LONG    8
#define SIZE_OF_LOGICAL 4
#ifdef DOUBLE_PREC
#define SIZE_OF_REAL 8
#else
#define SIZE_OF_REAL 4
#endif
#define SIZE_OF_COMPLEX (2*SIZE_OF_REAL)

program test_blas_fortran
  !use gpu_axpy
  use gpublas_m

#ifdef GTENSOR_DEVICE_SYCL
  ! Need for ifx (and ifort?), see
  ! https://www.intel.com/content/www/us/en/develop/documentation/fortran-compiler-oneapi-dev-guide-and-reference/top/language-reference/a-to-z-reference/q-to-r/rand-random.html
  use ifport
#endif

  implicit none

  print*,'== test_axpy =='
  call test_axpy()
  print
  print*,'== test_zgetrf =='
  call test_zgetrf()
  print
  print*,'== test_zgetrs =='
  call test_zgetrs()
  print
contains

subroutine test_axpy

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
end subroutine test_axpy

subroutine test_zgetrf_batched()
  type a_matrix
     complex, dimension(:,:), contiguous, pointer :: mat
  end type a_matrix

  integer, parameter :: N = 4;
  integer, parameter :: batch_size = 4;
  integer, dimension(:), contiguous, pointer :: info
#ifdef GPU_BLAS_I64
  integer(C_LONG), dimension(:,:), contiguous, pointer :: pivot
#else
  integer(C_INT), dimension(:,:), contiguous, pointer :: pivot
#endif
  type(C_PTR), dimension(:), contiguous, pointer :: mat_ptr_arr
  type(C_PTR) :: info_ptr, pivot_ptr, mat_ptr
  integer(C_SIZE_T) :: infobytes, matbytes, pivotbytes
  type(a_matrix), dimension(batch_size) :: b_matrix
  complex,dimension(N,N,batch_size) :: mat_in
  complex,dimension(N,N) :: L,U,A,dum
  real,dimension(N,N) :: P
  real, dimension(N) :: tmp
  integer :: r,c,b

  infobytes =  batch_size * SIZE_OF_INTEGER
  pivotbytes = batch_size * N * SIZE_OF_GPU_BLAS_INT
  matbytes = N**2 * SIZE_OF_COMPLEX

  !print *, "allocating infobytes", infobytes
  !print *, "allocating pivotbytes", pivotbytes
  !print *, "allocating matbytes", matbytes

  call gpuAllocateManaged(info_ptr, infobytes)
  call gpuAllocateManaged(pivot_ptr, pivotbytes)
  call gpuAllocateManaged(mat_ptr, int(c_sizeof(mat_ptr)*batch_size,kind=C_LONG))

  call C_F_POINTER(info_ptr, info, [batch_size])
  call C_F_POINTER(pivot_ptr, pivot, [N,batch_size])
  call C_F_POINTER(mat_ptr, mat_ptr_arr, [batch_size])

  do b=1,batch_size
     call gpuAllocateManaged(mat_ptr_arr(b), matbytes)
     call C_F_POINTER(mat_ptr_arr(b), b_matrix(b)%mat, [N, N])
     do r=1,N
        do c=1,N
           b_matrix(b)%mat(r, c) = cmplx(rand(), rand())
        end do
     end do
     mat_in(:,:,b)=b_matrix(b)%mat
  end do

  call gpublas_create()

  call gpublas_zgetrf_batched(N, mat_ptr_arr, N, pivot_ptr, info, batch_size)

  call gpuSynchronize()

  call gpublas_destroy()

  do b=1,batch_size
     A=mat_in(:,:,b)
     L=(0.0,0.0)
     U=(0.0,0.0)
     forall(r=1:N, c=1:N) P(r,c) = (r/c)*(c/r)
     !print*, pivot(:,b)

     do r=1,N
        L(r,1:r-1)=b_matrix(b)%mat(r,1:r-1)
        L(r,r)=(1.0,0.0)
        U(r,r:N)=b_matrix(b)%mat(r,r:N)
        tmp = P(r,:)
        P(r,:)=P(pivot(r,b),:)
        P(pivot(r,b),:)=tmp
     end do

     !print*,'original matrix'
     !call print2d_complex(A)
     !print *, 'L'
     !call print2d_complex(L)
     !print *, 'U'
     !call print2d_complex(U)
     !print *, 'P'
     !call print2d_real(P)

     A=matmul(P,A)
     dum=matmul(L,U)
     @assertLessThan(maxval(abs(A - dum)), MAX_ERR)
  end do

end subroutine test_zgetrf_batched

subroutine test_zgetrf_npvt_batched()
  type a_matrix
     complex, dimension(:,:), contiguous, pointer :: mat
  end type a_matrix

  integer, parameter :: N = 4;
  integer, parameter :: batch_size = 1;
  integer, dimension(:), contiguous, pointer :: info
  type(C_PTR), dimension(:), contiguous, pointer :: mat_ptr_arr
  type(C_PTR) :: info_ptr, mat_ptr
  integer(C_SIZE_T) :: infobytes, matbytes
  type(a_matrix), dimension(batch_size) :: b_matrix
  complex,dimension(N,N,batch_size) :: mat_in
  complex,dimension(N,N) :: L,U,A,dum
  integer :: r,c,b
  complex :: a_complex

  infobytes =  batch_size*STORAGE_SIZE(N) / 8
  matbytes = N**2 * STORAGE_SIZE(a_complex) / 8

  !print *, "allocating infobytes", infobytes
  !print *, "allocating matbytes", matbytes

  call gpuAllocateManaged(info_ptr, infobytes)
  call gpuAllocateManaged(mat_ptr, int(c_sizeof(mat_ptr)*batch_size,kind=C_LONG))

  call C_F_POINTER(info_ptr, info, [batch_size])
  call C_F_POINTER(mat_ptr, mat_ptr_arr, [batch_size])

  do b=1,batch_size
     call gpuAllocateManaged(mat_ptr_arr(b), matbytes)
     call C_F_POINTER(mat_ptr_arr(b), b_matrix(b)%mat, [N, N])
     b_matrix(b)%mat=0.
     do r=1,N
        b_matrix(b)%mat(r, r)=real(r)
        if (r > 1) then
           b_matrix(b)%mat(r, r-1)=real(r)/(batch_size+2)
        end if
     end do

     !     	do r=1,N
     !          do c=1,N
     !             b_matrix(b)%mat(r, c) = cmplx(rand(), rand())
     !          end do
     !        end do
     mat_in(:,:,b)=b_matrix(b)%mat
  end do

  call gpublas_create()

  call gpublas_zgetrf_npvt_batched(N, mat_ptr_arr, N, info, batch_size)

  call gpuSynchronize()

  call gpublas_destroy()

  do b=1,batch_size
     A=mat_in(:,:,b)
     L=(0.0,0.0)
     U=(0.0,0.0)

     do r=1,N
        L(r,1:r-1)=b_matrix(b)%mat(r,1:r-1)
        L(r,r)=(1.0,0.0)
        U(r,r:N)=b_matrix(b)%mat(r,r:N)
     end do

     !print*,'original matrix'
     !call print2d_complex(A)
     !print *, 'L'
     !call print2d_complex(L)
     !print *, 'U'
     !call print2d_complex(U)

     dum=matmul(L,U)
     do r=1,N
        do c=1,N
           @assertLessThan(abs(A(r,c) - dum(r,c)), 10*MAX_ERR)
        end do
     end do
  end do

end subroutine test_zgetrf_npvt_batched

subroutine test_zgetrs_batched()
  integer, parameter :: N = 4;
  integer, parameter :: NRHS = 2;
  integer, parameter :: batch_size = 2;
  integer(GPU_BLAS_INT), dimension(:,:), contiguous, pointer :: pivot
  type(C_PTR), dimension(:), contiguous, pointer :: Aptr, Bptr
  type(C_PTR) :: d_pivot, d_A, d_B, d_Aptr, d_Bptr
  integer(C_SIZE_T) :: Abytes, Bbytes, pivotbytes, ptrbytes
  complex,dimension(:,:,:), contiguous, pointer :: A, B

  pivotbytes = N * batch_size * SIZE_OF_GPU_BLAS_INT
  Abytes = N * N * SIZE_OF_COMPLEX
  Bbytes = N * NRHS * SIZE_OF_COMPLEX
  ptrbytes = batch_size * STORAGE_SIZE(d_A) / 8

  call gpuAllocateManaged(d_pivot, pivotbytes)
  call gpuAllocateManaged(d_A, Abytes)
  call gpuAllocateManaged(d_B, Bbytes)
  call gpuAllocateManaged(d_Aptr, ptrbytes)
  call gpuAllocateManaged(d_Bptr, ptrbytes)

  call C_F_POINTER(d_pivot, pivot, [N,batch_size])
  call C_F_POINTER(d_A, A, [N,N,batch_size])
  call C_F_POINTER(d_B, B, [N,NRHS,batch_size])
  call C_F_POINTER(d_Aptr, Aptr, [batch_size])
  call C_F_POINTER(d_Bptr, Bptr, [batch_size])

  ! setup input for first batch
  ! first column factored
  A(1, 1, 1) = 4.0;
  A(2, 1, 1) = 1.0;
  A(3, 1, 1) = 0.25;
  ! second column
  A(1, 2, 1) = 4.0;
  A(2, 2, 1) = 2.0;
  A(3, 2, 1) = 0.5;
  ! thrid column
  A(1, 3, 1) = 2.0;
  A(2, 3, 1) = 2.0;
  A(3, 3, 1) = 0.5;

  pivot(1, 1) = 2;
  pivot(2, 1) = 3;
  pivot(3, 1) = 3;
  Aptr(1) = c_loc(A(1,1,1))

  ! setup input for second batch
  ! first column
  A(1, 1, 2) = cmplx(0, 4);
  A(2, 1, 2) = cmplx(0, -1);
  A(3, 1, 2) = cmplx(0.25, -0.25);
  ! second column factored
  A(1, 2, 2) = cmplx(4, 0);
  A(2, 2, 2) = cmplx(0, 10);
  A(3, 2, 2) = cmplx(0, -0.1);
  ! third column factored
  A(1, 3, 2) = cmplx(2, 0);
  A(2, 3, 2) = cmplx(4, 2);
  A(3, 3, 2) = cmplx(1.3, 0.9);

  pivot(1, 2) = 2;
  pivot(2, 2) = 3;
  pivot(3, 2) = 3;
  Aptr(2) = c_loc(A(1,1,2))

  ! first batch, first rhs col vector   (11; 18; 28)
  B(1, 1, 1) = 11;
  B(2, 1, 1) = 18;
  B(3, 1, 1) = 28;
  ! first batch, second rhs col vector  (83; 78; 154)
  B(1, 2, 1) = 73;
  B(2, 2, 1) = 78;
  B(3, 2, 1) = 154;
  ! second batch, first rhs col vector  (73; 78; 154)
  B(1, 1, 2) = cmplx(11, -1);
  B(2, 1, 2) = cmplx(14, 4);
  B(3, 1, 2) = cmplx(16, 12);
  ! second batch, second rhs col vector (73-10i; 90-12i; 112 + 42i)
  B(1, 2, 2) = cmplx(73, -10);
  B(2, 2, 2) = cmplx(90, -12);
  B(3, 2, 2) = cmplx(112, 42);

  Bptr(1) = c_loc(B(1,1,1))
  Bptr(2) = c_loc(B(1,1,2))

  call gpublas_create()

  call gpublas_zgetrs_batched(N, NRHS, Aptr, N, d_pivot, Bptr, N, batch_size)

  call gpuSynchronize()

  call gpublas_destroy()

end subroutine test_zgetrs_batched

end program test_blas_fortran
