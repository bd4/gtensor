!#include "redef.h"

program test_blas_fortran
    use gpu_api_m
    use gpublas_m

    ! Need for ifx (and ifort?), see
    ! https://www.intel.com/content/www/us/en/develop/documentation/fortran-compiler-oneapi-dev-guide-and-reference/top/language-reference/a-to-z-reference/q-to-r/rand-random.html
    use ifport

    type a_matrix
       complex, dimension(:,:), contiguous, pointer :: mat
    end type a_matrix

    integer, parameter :: N = 4;
    integer, parameter :: batch_size = 4;
    integer, dimension(:), contiguous, pointer :: info
    integer, dimension(:,:), contiguous, pointer :: pivot
    type(C_PTR), dimension(:), contiguous, pointer :: mat_ptr_arr
    type(C_PTR) :: info_ptr, pivot_ptr, mat_ptr
    integer(C_SIZE_T) :: infobytes, matbytes, pivotbytes
    type(a_matrix), dimension(batch_size) :: b_matrix
    complex,dimension(N,N,batch_size) :: mat_in
    complex,dimension(N,N) :: L,U,A,dum
    real,dimension(N,N) :: P
    real, dimension(N) :: tmp
    integer :: r,c,b
    complex :: a_complex

    infobytes =  batch_size*STORAGE_SIZE(N) / 8
    pivotbytes = batch_size*N*STORAGE_SIZE(N) / 8
    matbytes = N**2 * STORAGE_SIZE(a_complex) / 8

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
       if (maxval(abs(A - dum)) > MAX_ERR) then
         print *, "big error in batch ", b, " ", maxval(abs(A - dum))
       endif
    end do


end program test_blas_fortran
