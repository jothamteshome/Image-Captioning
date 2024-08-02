from setuptools import setup
from torch.utils import cpp_extension

setup(name="decode_captions",
      ext_modules=[
          cpp_extension.CppExtension(
              "decode_captions",
                ["cpp_operators/decode_captions.cpp"]
                )
          ],
      cmdclass={'build_ext': cpp_extension.BuildExtension})