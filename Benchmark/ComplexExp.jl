using Random
using BenchmarkTools

# 설정
Random.seed!(1234)

N = 10^6
zr = randn(Float64, N)
zi = randn(Float64, N)
zs = ComplexF64.(zr, zi)

out = similar(zs)


# method 1: 그냥 exp(z)
function exp_direct!(out, zs)
    @inbounds @simd for i in eachindex(zs)
        out[i] = exp(zs[i])
    end
    return nothing
end

# method 2: real/imag 분리
function exp_split!(out, zs)
    @inbounds @simd for i in eachindex(zs)
        z = zs[i]
        out[i] = exp(real(z)) * cis(imag(z))
    end
    return nothing
end

# warm-up (반드시 필요하긴 함)
exp_direct!(out, zs)
exp_split!(out, zs)


# benchmark
println("Benchmark: exp(z)")
@btime exp_direct!($out, $zs)

println("\nBenchmark: exp(real) * cis(imag)")
@btime exp_split!($out, $zs)


# benchmark 결과... 거의 차이 안남. 1e6개 하는데 0.5ms 정도면. ns scale임.
# Benchmark: exp(z)
#   13.280 ms (0 allocations: 0 bytes)

# Benchmark: exp(real) * cis(imag)
#   12.770 ms (0 allocations: 0 bytes)