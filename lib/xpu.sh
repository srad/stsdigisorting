git clone https://github.com/fweig/xpu.git
cd xpu
mkdir build -p && cd build

export CUDACXX=/usr/local/cuda/bin/nvcc
export hip_INCLUDE_DIR=/opt/rocm-5.4.2/include

cmake -DXPU_BUILD_TESTS=OFF -DXPU_BUILD_EXAMPLES=OFF -DXPU_ENABLE_CUDA=ON -DXPU_DEBUG=ON ..
make
