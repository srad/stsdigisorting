git clone https://github.com/fweig/xpu.git
cd xpu
mkdir build -p && cd build
cmake -DXPU_BUILD_TESTS=OFF -DXPU_BUILD_EXAMPLES=OFF -DXPU_ENABLE_CUDA=ON -DXPU_DEBUG=OFF ..
make
