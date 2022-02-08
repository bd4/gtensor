#!/bin/bash

module load oneapi/eng-compiler
module load cmake

export EnableWalkerPartition=0
export SYCL_DEVICE_FILTER=level_zero

CXX=$(which dpcpp) CC=$(which gcc) \
 cmake -S . -B build-sycl-gpu -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGTENSOR_DEVICE=sycl -DGTENSOR_DEVICE_SYCL_SELECTOR=gpu \
  -DBUILD_TESTING=ON -DGTENSOR_BUILD_EXAMPLES=OFF -DGTENSOR_TEST_DEBUG=ON \
  -DCMAKE_INSTALL_PREFIX=$HOME/soft/gtensor/sycl \
  -DONEAPI_PATH=$A21_SDK_ROOT -DMKL_PATH=$MKLROOT \
  -DGTENSOR_ENABLE_CLIB=ON -DGTENSOR_ENABLE_BLAS=ON -DGTENSOR_ENABLE_FFT=ON \
  -DGTENSOR_BUILD_BENCHMARKS=OFF

cmake --build build-sycl-gpu -v -j10

make

./build-sycl-gpu/test_zgetrf_fortran_d
