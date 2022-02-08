module gpufft_m
  use,intrinsic :: iso_c_binding
  implicit none

  interface
    function gpufft_new_complex_float(rank, n, istride, idist, &
                                      ostride, odist, batch_size) &
             & bind(c, name="gtfft_new_complex_float")
      import
      integer(C_INT), intent(IN), value :: rank, batch_size
      integer(C_INT), intent(IN), value :: istride, idist, ostride, odist
      type(C_PTR), intent(IN), value :: n
      type(C_PTR) :: gpufft_new_complex_float
    end function gpufft_new_complex_float

    function gpufft_new_complex_double(rank, n, istride, idist, &
                                       ostride, odist, batch_size) &
             & bind(c, name="gtfft_new_complex_double")
      import
      integer(C_INT), intent(IN), value :: rank, batch_size
      integer(C_INT), intent(IN), value :: istride, idist, ostride, odist
      type(C_PTR), intent(IN), value :: n
      type(C_PTR) :: gpufft_new_complex_double
    end function gpufft_new_complex_double

    function gpufft_new_real_float(rank, n, istride, idist, &
                                   ostride, odist, batch_size) &
             & bind(c, name="gtfft_new_real_float")
      import
      integer(C_INT), intent(IN), value :: rank, batch_size
      integer(C_INT), intent(IN), value :: istride, idist, ostride, odist
      type(C_PTR), intent(IN), value :: n
      type(C_PTR) :: gpufft_new_real_float
    end function gpufft_new_real_float

    function gpufft_new_real_double(rank, n, istride, idist, &
                                    ostride, odist, batch_size) &
             & bind(c, name="gtfft_new_real_double")
      import
      integer(C_INT), intent(IN), value :: rank, batch_size
      integer(C_INT), intent(IN), value :: istride, idist, ostride, odist
      type(C_PTR), intent(IN), value :: n
      type(C_PTR) :: gpufft_new_real_double
    end function gpufft_new_real_double

    subroutine gpufft_delete_complex_float(plan) &
               & bind(c, name="gtfft_delete_complex_float")
      import
      type(C_PTR), intent(IN), value :: plan
    end subroutine gpufft_delete_complex_float

    subroutine gpufft_delete_complex_double(plan) &
               & bind(c, name="gtfft_delete_complex_double")
      import
      type(C_PTR), intent(IN), value :: plan
    end subroutine gpufft_delete_complex_double

    subroutine gpufft_delete_real_float(plan) &
               & bind(c, name="gtfft_delete_real_float")
      import
      type(C_PTR), intent(IN), value :: plan
    end subroutine gpufft_delete_real_float

    subroutine gpufft_delete_real_double(plan) &
               & bind(c, name="gtfft_delete_real_double")
      import
      type(C_PTR), intent(IN), value :: plan
    end subroutine gpufft_delete_real_double

    subroutine gpufft_zz(plan, indata, outdata) bind(c, name="gtfft_zz")
      import
      type(C_PTR), intent(IN), value :: plan, indata, outdata
    end subroutine gpufft_zz

    subroutine gpufft_inverse_zz(plan, indata, outdata) &
               & bind(c, name="gtfft_inverse_zz")
      import
      type(C_PTR), intent(IN), value :: plan, indata, outdata
    end subroutine gpufft_inverse_zz

    subroutine gpufft_cc(plan, indata, outdata) bind(c, name="gtfft_cc")
      import
      type(C_PTR), intent(IN), value :: plan, indata, outdata
    end subroutine gpufft_cc

    subroutine gpufft_inverse_cc(plan, indata, outdata) &
               & bind(c, name="gtfft_inverse_cc")
      import
      type(C_PTR), intent(IN), value :: plan, indata, outdata
    end subroutine gpufft_inverse_cc

    subroutine gpufft_dz(plan, indata, outdata) bind(c, name="gtfft_dz")
      import
      type(C_PTR), intent(IN), value :: plan, indata, outdata
    end subroutine gpufft_dz

    subroutine gpufft_inverse_zd(plan, indata, outdata) &
               & bind(c, name="gtfft_inverse_zd")
      import
      type(C_PTR), intent(IN), value :: plan, indata, outdata
    end subroutine gpufft_inverse_zd

    subroutine gpufft_rc(plan, indata, outdata) bind(c, name="gtfft_rc")
      import
      type(C_PTR), intent(IN), value :: plan, indata, outdata
    end subroutine gpufft_rc

    subroutine gpufft_inverse_cr(plan, indata, outdata) &
               & bind(c, name="gtfft_inverse_cr")
      import
      type(C_PTR), intent(IN), value :: plan, indata, outdata
    end subroutine gpufft_inverse_cr
  end interface

end module gpufft_m
