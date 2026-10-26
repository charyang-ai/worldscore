from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os.path as osp
ROOT = osp.dirname(osp.abspath(__file__))

# Lietorch is built from thirdparty/lietorch. A second setup() call in this
# file makes current setuptools emit two egg-info directories and abort.
# nvcc -gencode flags are CUDA-only; hipcc takes PYTORCH_ROCM_ARCH instead.
setup(
    name='droid_backends',
    ext_modules=[
        CUDAExtension('droid_backends',
            include_dirs=[osp.join(ROOT, 'thirdparty/lietorch/eigen')],
            sources=[
                'src/droid.cpp',
                'src/droid_kernels.cu',
                'src/correlation_kernels.cu',
                'src/altcorr_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3'],
            }),
    ],
    cmdclass={'build_ext': BuildExtension}
)
