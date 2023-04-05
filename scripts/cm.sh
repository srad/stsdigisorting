#export CUDACXX=/usr/local/cuda/bin/nvcc
#export hip_INCLUDE_DIR=/opt/rocm-5.4.2/include

snap run cmake --fresh -DXPU_ENABLE_CUDA=ON -DXPU_DEBUG=OFF -DXPU_ENABLE_HIP=ON ..