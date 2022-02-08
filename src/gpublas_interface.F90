module gpublas_m
  use,intrinsic :: iso_c_binding
  implicit none

  interface
    subroutine gpublas_create() bind(c, name="gtblas_create")
      import
    end subroutine gpublas_create

    subroutine gpublas_destroy() bind(c, name="gtblas_destroy")
      import
    end subroutine gpublas_destroy

    subroutine gpublas_set_stream(streamId) bind(c, name="gtblas_set_stream")
      import
      type(C_PTR),value,intent(IN) :: streamId
    end subroutine gpublas_set_stream

    subroutine gpublas_get_stream(streamId) bind(c, name="gtblas_get_stream")
      import
      type(C_PTR) :: streamId
    end subroutine gpublas_get_stream

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

    subroutine gpublas_ccopy(n,x,incx,y,incy) bind(c, name="gtblas_ccopy")
      import
      integer(C_INT),intent(IN),value :: n,incx,incy
      type(C_PTR), intent(IN),value :: x
      type(C_PTR), value :: y
    end subroutine gpublas_ccopy

    subroutine gpublas_zcopy(n,x,incx,y,incy) bind(c, name="gtblas_zcopy")
      import
      integer(C_INT),intent(IN),value :: n,incx,incy
      type(C_PTR), intent(IN),value :: x
      type(C_PTR), value :: y
    end subroutine gpublas_zcopy

    subroutine gpublas_scopy(n,x,incx,y,incy) bind(c, name="gtblas_scopy")
      import
      integer(C_INT),intent(IN),value :: n,incx,incy
      type(C_PTR), intent(IN),value :: x
      type(C_PTR), value :: y
    end subroutine gpublas_scopy

    subroutine gpublas_dcopy(n,x,incx,y,incy) bind(c, name="gtblas_dcopy")
      import
      integer(C_INT),intent(IN),value :: n,incx,incy
      type(C_PTR), intent(IN),value :: x
      type(C_PTR), value :: y
    end subroutine gpublas_dcopy

    subroutine gpublas_csscal(n,fac,arr,incx) bind(c, name="gtblas_csscal")
      import
      integer(C_INT),intent(IN),value :: n,incx
      real(C_FLOAT),intent(IN) :: fac
      complex(C_FLOAT_COMPLEX),dimension(*) :: arr
    end subroutine gpublas_csscal

    subroutine gpublas_zdscal(n,fac,arr,incx) bind(c, name="gtblas_zdscal")
      import
      integer(C_INT),intent(IN),value :: n,incx
      real(C_DOUBLE),intent(IN) :: fac
      complex(C_DOUBLE_COMPLEX),dimension(*) :: arr
    end subroutine gpublas_zdscal

    subroutine gpublas_cscal(n,fac,arr,incx) bind(c, name="gtblas_cscal")
      import
      integer(C_INT),intent(IN),value :: n,incx
      complex(C_FLOAT_COMPLEX),intent(IN) :: fac
      complex(C_FLOAT_COMPLEX),dimension(*) :: arr
    end subroutine gpublas_cscal

    subroutine gpublas_zscal(n,fac,arr,incx) bind(c, name="gtblas_zscal")
      import
      integer(C_INT),intent(IN),value :: n,incx
      complex(C_DOUBLE_COMPLEX),intent(IN) :: fac
      complex(C_DOUBLE_COMPLEX),dimension(*) :: arr
    end subroutine gpublas_zscal

    subroutine gpublas_sscal(n,fac,arr,incx) bind(c, name="gtblas_sscal")
      import
      integer(C_INT),intent(IN),value :: n,incx
      real(C_FLOAT),intent(IN) :: fac
      real(C_FLOAT),dimension(*) :: arr
    end subroutine gpublas_sscal

    subroutine gpublas_dscal(n,fac,arr,incx) bind(c, name="gtblas_dscal")
      import
      integer(C_INT),intent(IN),value :: n,incx
      real(C_DOUBLE),intent(IN) :: fac
      real(C_DOUBLE),dimension(*) :: arr
    end subroutine gpublas_dscal

    subroutine gpublas_zgetrf_batched(n, d_Aarray, lda, d_pivotArray, &
     & d_infoArray, batchSize) bind(c, name="gtblas_zgetrf_batched")
      import
      integer(C_INT), intent(in), value :: n, lda, batchSize
      type(C_PTR), dimension(batchSize) :: d_Aarray ! has c_loc of batchsize cmplx arrs
      type(C_PTR), value :: d_pivotArray
      integer(C_INT), dimension(batchSize), intent(out) :: d_infoArray
    end subroutine gpublas_zgetrf_batched

    subroutine gpublas_zgetrs_batched(n, nrhs, d_Aarray, lda, d_pivotArray, &
      & d_Barray, ldb, batchSize) bind(c, name="gtblas_zgetrs_batched")
       import
       integer(C_INT), intent(in), value :: n, nrhs, lda, ldb, batchSize
       ! c_loc of 'batchsize' cmplx arrays, each array is a n * n matrix with LU factorization
       type(C_PTR), dimension(batchSize), intent(in) :: d_Aarray
       type(C_PTR), value :: d_pivotArray
       !integer(C_INT), dimension(n*batchSize), intent(in) :: d_pivotArray
       ! c_loc of 'batchsize' cmplx arrays, each with n * n_rhs entries
       type(C_PTR), dimension(batchSize), intent(inout) :: d_Barray
     end subroutine gpublas_zgetrs_batched

    subroutine gpublas_cgetrf_batched(n, d_Aarray, lda, d_pivotArray, &
     & d_infoArray, batchSize) bind(c, name="gtblas_cgetrf_batched")
      import
      integer(C_INT), intent(in), value :: n, lda, batchSize
      type(C_PTR), dimension(batchSize) :: d_Aarray ! has c_loc of batchsize cmplx arrs
      type(C_PTR), value :: d_pivotArray
      integer(C_INT), dimension(batchSize), intent(out) :: d_infoArray
    end subroutine gpublas_cgetrf_batched

    subroutine gpublas_cgetrs_batched(n, nrhs, d_Aarray, lda, d_pivotArray, &
      & d_Barray, ldb, batchSize) bind(c, name="gtblas_cgetrs_batched")
       import
       integer(C_INT), intent(in), value :: n, nrhs, lda, ldb, batchSize
       ! c_loc of 'batchsize' cmplx arrays, each array is a n * n matrix with LU factorization
       type(C_PTR), dimension(batchSize), intent(in) :: d_Aarray
       type(C_PTR), value :: d_pivotArray
       !integer(C_INT), dimension(n*batchSize), intent(in) :: d_pivotArray
       ! c_loc of 'batchsize' cmplx arrays, each with n * n_rhs entries
       type(C_PTR), dimension(batchSize), intent(inout) :: d_Barray
     end subroutine gpublas_cgetrs_batched

    subroutine gpublas_dgetrf_batched(n, d_Aarray, lda, d_pivotArray, &
     & d_infoArray, batchSize) bind(c, name="gtblas_dgetrf_batched")
      import
      integer(C_INT), intent(in), value :: n, lda, batchSize
      type(C_PTR), dimension(batchSize) :: d_Aarray ! has c_loc of batchsize cmplx arrs
      type(C_PTR), value :: d_pivotArray
      integer(C_INT), dimension(batchSize), intent(out) :: d_infoArray
    end subroutine gpublas_dgetrf_batched

    subroutine gpublas_dgetrs_batched(n, nrhs, d_Aarray, lda, d_pivotArray, &
      & d_Barray, ldb, batchSize) bind(c, name="gtblas_dgetrs_batched")
       import
       integer(C_INT), intent(in), value :: n, nrhs, lda, ldb, batchSize
       ! c_loc of 'batchsize' cmplx arrays, each array is a n * n matrix with LU factorization
       type(C_PTR), dimension(batchSize), intent(in) :: d_Aarray
       type(C_PTR), value :: d_pivotArray
       !integer(C_INT), dimension(n*batchSize), intent(in) :: d_pivotArray
       ! c_loc of 'batchsize' cmplx arrays, each with n * n_rhs entries
       type(C_PTR), dimension(batchSize), intent(inout) :: d_Barray
     end subroutine gpublas_dgetrs_batched

    subroutine gpublas_sgetrf_batched(n, d_Aarray, lda, d_pivotArray, &
     & d_infoArray, batchSize) bind(c, name="gtblas_sgetrf_batched")
      import
      integer(C_INT), intent(in), value :: n, lda, batchSize
      type(C_PTR), dimension(batchSize) :: d_Aarray ! has c_loc of batchsize cmplx arrs
      type(C_PTR), value :: d_pivotArray
      integer(C_INT), dimension(batchSize), intent(out) :: d_infoArray
    end subroutine gpublas_sgetrf_batched

    subroutine gpublas_sgetrs_batched(n, nrhs, d_Aarray, lda, d_pivotArray, &
      & d_Barray, ldb, batchSize) bind(c, name="gtblas_sgetrs_batched")
       import
       integer(C_INT), intent(in), value :: n, nrhs, lda, ldb, batchSize
       ! c_loc of 'batchsize' cmplx arrays, each array is a n * n matrix with LU factorization
       type(C_PTR), dimension(batchSize), intent(in) :: d_Aarray
       type(C_PTR), value :: d_pivotArray
       !integer(C_INT), dimension(n*batchSize), intent(in) :: d_pivotArray
       ! c_loc of 'batchsize' cmplx arrays, each with n * n_rhs entries
       type(C_PTR), dimension(batchSize), intent(inout) :: d_Barray
     end subroutine gpublas_sgetrs_batched

     subroutine gpublas_cgemv(m,n,alpha,A,lda,x,incx,beta,y,incy) bind(c, name="gtblas_cgemv")
       import
       integer(C_INT), intent(IN), value :: m,n,lda,incx,incy
       complex(C_FLOAT_COMPLEX), dimension(*), intent(IN) :: A,x
       complex(C_FLOAT_COMPLEX), intent(IN) :: alpha,beta
       complex(C_FLOAT_COMPLEX), dimension(*), intent(INOUT) :: y
     end subroutine gpublas_cgemv

     subroutine gpublas_zgemv(m,n,alpha,A,lda,x,incx,beta,y,incy) bind(c, name="gtblas_zgemv")
       import
       integer(C_INT), intent(IN), value :: m,n,lda,incx,incy
       complex(C_DOUBLE_COMPLEX), dimension(*), intent(IN) :: A,x
       complex(C_DOUBLE_COMPLEX), intent(IN) :: alpha,beta
       complex(C_DOUBLE_COMPLEX), dimension(*), intent(INOUT) :: y
     end subroutine gpublas_zgemv

     subroutine gpublas_sgemv(m,n,alpha,A,lda,x,incx,beta,y,incy) bind(c, name="gtblas_sgemv")
       import
       integer(C_INT), intent(IN), value :: m,n,lda,incx,incy
       real(C_FLOAT), dimension(*), intent(IN) :: A,x
       real(C_FLOAT), intent(IN) :: alpha,beta
       real(C_FLOAT), dimension(*), intent(INOUT) :: y
     end subroutine gpublas_sgemv

     subroutine gpublas_dgemv(m,n,alpha,A,lda,x,incx,beta,y,incy) bind(c, name="gtblas_dgemv")
       import
       integer(C_INT), intent(IN), value :: m,n,lda,incx,incy
       real(C_DOUBLE), dimension(*), intent(IN) :: A,x
       real(C_DOUBLE), intent(IN) :: alpha,beta
       real(C_DOUBLE), dimension(*), intent(INOUT) :: y
     end subroutine gpublas_dgemv

     subroutine gpublas_cgetrf_npvt_batched(n, d_Aarray, lda, &
     & d_infoArray, batchSize) bind(c, name="gtblas_cgetrf_npvt_batched")
      import
      integer(C_INT), intent(in), value :: n, lda, batchSize
      type(C_PTR), dimension(batchSize) :: d_Aarray ! has c_loc of batchsize cmplx arrs
      integer(C_INT), dimension(batchSize), intent(out) :: d_infoArray
    end subroutine gpublas_cgetrf_npvt_batched

    subroutine gpublas_zgetrf_npvt_batched(n, d_Aarray, lda, &
     & d_infoArray, batchSize) bind(c, name="gtblas_zgetrf_npvt_batched")
      import
      integer(C_INT), intent(in), value :: n, lda, batchSize
      type(C_PTR), dimension(batchSize) :: d_Aarray ! has c_loc of batchsize cmplx arrs
      integer(C_INT), dimension(batchSize), intent(out) :: d_infoArray
    end subroutine gpublas_zgetrf_npvt_batched

 end interface

end module gpublas_m
