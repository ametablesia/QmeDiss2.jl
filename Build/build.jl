
using PackageCompiler

# 나중에는 패키지화 해서, 자동으로 하도록 만들어야 함.
# create_sysimage(
#     [
#         :LinearAlgebra,
#         :LoopVectorization,
#         :StaticArrays,
#         :MKL
#     ],
#     sysimage_path = "../Entrypoint/package_sysimage.so"
# )

# 그냥 일단 지금은 전체 환경을 통채로 sysimage로 만듦.
create_sysimage(sysimage_path = "../Entrypoint/package_sysimage.so")

# using MKL
# using LinearAlgebra
# using LoopVectorization
# using Base.Threads
# using BenchmarkTools