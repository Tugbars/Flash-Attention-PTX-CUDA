"""Build: pip install -e python/   (or python setup.py build_ext --inplace)

Compiles the flash-attention kernels + torch bindings into fa_ptx._C.
Requires the CUDA toolkit (nvcc) and a CUDA-enabled torch.
"""
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)

setup(
    name="fa_ptx",
    version="0.1.0",
    description="Hand-written PTX flash attention for consumer Blackwell "
                "(prefill, varlen, paged decode, FP8/INT4 KV cache)",
    packages=["fa_ptx"],
    ext_modules=[
        CUDAExtension(
            name="fa_ptx._C",
            sources=[
                os.path.join(ROOT, "csrc", "fa_torch.cpp"),
                os.path.join(REPO, "kernels", "flash_attention.cu"),
                os.path.join(REPO, "kernels", "flash_attention_decode.cu"),
                os.path.join(REPO, "kernels", "fa_autotune.cu"),
                os.path.join(REPO, "kernels", "fa_api.cu"),
            ],
            include_dirs=[os.path.join(REPO, "include")],
            extra_compile_args={
                "cxx": ["/O2"] if os.name == "nt" else ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "--expt-relaxed-constexpr",
                    "-std=c++17",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
)
