"""Build the isolated GDN post-conv extension for a local vLLM process.

This is intentionally separate from vLLM's CMake build.  Set
TORCH_CUDA_ARCH_LIST=8.6 (the default in build_sm86.sh) to produce an
Ampere/SM86-only shared object.
"""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

Path("build/temp.linux-x86_64-cpython-312").mkdir(parents=True, exist_ok=True)

setup(
    name="gdn_fused_sm86",
    ext_modules=[
        CUDAExtension(
            # Keep the dispatcher-only library distinct from the Python
            # wrapper module (`gdn_fused_sm86.py`). Python prefers extension
            # modules over .py files when both share a basename.
            name="gdn_fused_sm86_ext",
            sources=["bindings.cpp", "fused_gdn_decode_sm86.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
