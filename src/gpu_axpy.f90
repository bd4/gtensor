module gpu_axpy
  use, intrinsic :: iso_c_binding
  implicit none

  !type(C_PTR), target :: handle = 0

  interface
     subroutine gpublas_create() bind(c, name="gtblas_create")
       import
     end subroutine gpublas_create

     subroutine gpublas_destroy() bind(c, name="gtblas_destroy")
       import
     end subroutine gpublas_destroy

     function gt_backend_managed_allocate(bytes) bind(c)
       import
       integer(C_SIZE_T),value :: bytes
       type(C_PTR) :: gt_backend_managed_allocate
     end function gt_backend_managed_allocate

     subroutine gt_synchronize() bind(c)
       import
     end subroutine gt_synchronize

  end interface

  interface
     subroutine gpublas_caxpy(n,a,x,incx,y,incy) bind(c, name="gtblas_caxpy")
       import
       integer(C_INT),intent(IN),value :: n,incx,incy
       complex(C_FLOAT_COMPLEX), intent(IN) :: a
       complex(C_FLOAT_COMPLEX), dimension(*),intent(IN) :: x
       complex(C_FLOAT_COMPLEX), dimension(*),intent(INOUT) :: y
     end subroutine gpublas_caxpy

     subroutine gpublas_zaxpy(n,a,x,incx,y,incy) bind(c, name="gtblas_zaxpy")
       import
       integer(C_INT),intent(IN),value :: n,incx,incy
       complex(C_DOUBLE_COMPLEX), intent(IN) :: a
       complex(C_DOUBLE_COMPLEX), dimension(*),intent(IN) :: x
       complex(C_DOUBLE_COMPLEX), dimension(*),intent(INOUT) :: y
     end subroutine gpublas_zaxpy

     subroutine gpublas_saxpy(n,a,x,incx,y,incy) bind(c, name="gtblas_saxpy")
       import
       integer(C_INT),intent(IN),value :: n,incx,incy
       real(C_FLOAT), intent(IN) :: a
       real(C_FLOAT), dimension(*),intent(IN) :: x
       real(C_FLOAT), dimension(*),intent(INOUT) :: y
     end subroutine gpublas_saxpy

     subroutine gpublas_daxpy(n,a,x,incx,y,incy) bind(c, name="gtblas_daxpy")
       import
       integer(C_INT),intent(IN),value :: n,incx,incy
       real(C_DOUBLE), intent(IN) :: a
       real(C_DOUBLE), dimension(*),intent(IN) :: x
       real(C_DOUBLE), dimension(*),intent(INOUT) :: y
     end subroutine gpublas_daxpy
   end interface
end module gpu_axpy
