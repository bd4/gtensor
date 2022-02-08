FORTRAN_FLAGS = -cpp -g

FC = ifort
BUILD_DIR = build-sycl-gpu
SYCL_LINK = -lsycl -lOpenCL -lze_loader -lmkl_sycl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lstdc++ -lpthread -lm -ldl -lipgo -ldecimal -lcilkrts -lifport -lifcoremt

.PHONY: all
all: $(BUILD_DIR)/test_blas_fortran $(BUILD_DIR)/test_blas_fortran_d

$(BUILD_DIR)/test_zgetrf_fortran_d: tests/test_zgetrf_fortran.f90 src/gpu_api_interface.o src/gpublas_interface.o
	ifort $(FORTRAN_FLAGS) -DDOUBLE_PREC -r8 -g -o $(BUILD_DIR)/test_zgetrf_fortran_d tests/test_zgetrf_fortran.f90 $(BUILD_DIR)/libcgtblas.a $(BUILD_DIR)/libcgtensor.a src/gpu_api_interface.o src/gpublas_interface.o $(SYCL_LINK)

$(BUILD_DIR)/test_blas_fortran: tests/test_blas_fortran.f90 src/gpu_axpy.o
	ifort $(FORTRAN_FLAGS) -o $(BUILD_DIR)/test_blas_fortran tests/test_blas_fortran.f90 $(BUILD_DIR)/libcgtblas.a $(BUILD_DIR)/libcgtensor.a src/gpu_axpy.o $(SYCL_LINK) -lstdc++

$(BUILD_DIR)/test_blas_fortran_d: tests/test_blas_fortran.f90 src/gpu_axpy.o
	ifort $(FORTRAN_FLAGS) -DDOUBLE_PREC -r8 -g -o $(BUILD_DIR)/test_blas_fortran_d tests/test_blas_fortran.f90 $(BUILD_DIR)/libcgtblas.a $(BUILD_DIR)/libcgtensor.a src/gpu_axpy.o $(SYCL_LINK) -lstdc++

$(BUILD_DIR)/test_fft_fortran: tests/test_fft_fortran.f90 src/gpu_axpy.o src/gpufft_interface.o
	ifort $(FORTRAN_FLAGS) -g -o $(BUILD_DIR)/test_fft_fortran tests/test_fft_fortran.f90 $(BUILD_DIR)/libcgtfft.a $(BUILD_DIR)/libcgtensor.a src/gpu_axpy.o src/gpufft_interface.o $(SYCL_LINK) -lstdc++

$(BUILD_DIR)/test_fft_fortran_d: tests/test_fft_fortran.f90 src/gpu_axpy.o src/gpufft_interface.o
	ifort $(FORTRAN_FLAGS) -DDOUBLE_PREC -r8 -g -o $(BUILD_DIR)/test_fft_fortran_d tests/test_fft_fortran.f90 $(BUILD_DIR)/libcgtfft.a $(BUILD_DIR)/libcgtensor.a src/gpu_axpy.o src/gpufft_interface.o $(SYCL_LINK) -lstdc++

src/gpu_axpy.o: src/gpu_axpy.f90
	ifort -g -c -o $@ $<

src/gpufft_interface.o: src/gpufft_interface.f90
	ifort -g -c -o $@ $<

src/gpu_api_interface.o: src/gpu_api_interface.F90
	ifort -g -c -o $@ $<

src/gpublas_interface.o: src/gpublas_interface.F90
	ifort -g -c -o $@ $<
