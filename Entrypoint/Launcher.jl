
using Cthulhu
@time using MKL
@time using LinearAlgebra
@time using Base.Threads
@time using BenchmarkTools

@time include("../Physics/Physics.jl")
@time include("../PerturbationTheory/Fret.jl")
@time include("../PerturbationTheory/Mrt.jl")

# spectral_density_1 = DrudeLorentzSpectralDensity(0.05, 1.0)
# decompose_info_1 = SpectralDensityDecomposeInfo(2000, 30.0, 1.0, [1.0 0.0; 0.0 -1.0])

# spectral_density_2 = DrudeLorentzSpectralDensity(0.05, 1.0)
# decompose_info_2 = SpectralDensityDecomposeInfo(2000, 300.0, 1.0, [1.0 0.0; 0.0 -1.0])

# env = Environment(0.0, Vector(), Vector())

# add__spectral_density!(env, spectral_density_1, decompose_info_1)
# add__spectral_density!(env, spectral_density_2, decompose_info_2)

# @printf(stderr, "total count is %d \n", length(env.effective_oscillators))

function test__FMO_7_sites()
    env = Environment()
    # add__spectral_density!(
    #     env,
    #     DrudeLorentzSpectralDensity(0.05, 1.0)
    # )

end



function test__fret()
    @time println("starting fret calculation")
    env = Environment()

    add__spectral_density!(
        env, 
        DrudeLorentzSpectralDensity(0.05, 1.0), 
        SpectralDensityDecomposeInfo(2000, 30.0, 1.0, [1.0 0.0; 0.0 -1.0])
    )
    add__spectral_density!(
        env,
        DrudeLorentzSpectralDensity(0.2, 0.5), 
        SpectralDensityDecomposeInfo(2000, 15.0, 1.0, [0.0 0.0; 0.0 1.0])
    )

    @printf(stderr, "total count is %d \n", length(env.effective_oscillators))

    fret_ctx = create__fret_context(
        System(;
            size_of_system=2, 
            system_hamiltonian=[1.0+0.0im 0.25+0.0im ; 0.25+0.0im -1.0+0.0im]
        ),
        env,
        SimulationDetails(0.01, 0.1, 1000.0, Int64(1000.0/0.01))
    )

    # init__fret_context!()
    calc__λ!(fret_ctx)

    if Threads.nthreads() == 1
        calc__g!(fret_ctx)
    else
        calc__g_with_threads!(fret_ctx)
    end

    calc__rates!(fret_ctx)
end


function test__mrt()
    env = Environment()

    add__spectral_density!(
        env, 
        DrudeLorentzSpectralDensity(0.05, 1.0), 
        SpectralDensityDecomposeInfo(2000, 30.0, 1.0, [1.0 0.0; 0.0 -1.0])
    )
    add__spectral_density!(
        env,
        DrudeLorentzSpectralDensity(0.2, 0.5), 
        SpectralDensityDecomposeInfo(2000, 15.0, 1.0, [0.0 0.0; 0.0 1.0])
    )

    @printf(stderr, "total count is %d \n", length(env.effective_oscillators))

    mrt_ctx = create__mrt_context(
        System(;
            size_of_system=2, 
            system_hamiltonian=[1.0+0.0im 0.25+0.0im ; 0.25+0.0im -1.0+0.0im]
        ),
        env,
        SimulationDetails(0.01, 0.1, 1000.0, Int64(1000.0/0.01))
    )

    calc__Λ!(mrt_ctx)
    calc__Γ!(mrt_ctx)

    if Threads.nthreads() == 1
        calc__g_g′_and_g″!(mrt_ctx)
    else
        calc__g_g′_and_g″_with_threads!(mrt_ctx)
    end

    calc__rates!(mrt_ctx)

    if Threads.nthreads() == 1
        calc__dissipations!(mrt_ctx)
    else
        # @descend calc__dissipations_with_threads!(mrt_ctx)
        calc__dissipations_with_threads!(mrt_ctx)
    end

    check__physics(mrt_ctx)
end



function test__cmrt()
    env = Environment()

    add__spectral_density!(
        env, 
        DrudeLorentzSpectralDensity(0.05, 1.0), 
        SpectralDensityDecomposeInfo(2000, 30.0, 1.0, [1.0 0.0; 0.0 -1.0])
    )
    add__spectral_density!(
        env,
        DrudeLorentzSpectralDensity(0.2, 0.5), 
        SpectralDensityDecomposeInfo(2000, 15.0, 1.0, [0.0 0.0; 0.0 1.0])
    )

    @printf(stderr, "total count is %d \n", length(env.effective_oscillators))

    cmrt_ctx = create__cmrt_context(
        System(;
            size_of_system=2, 
            system_hamiltonian=[1.0+0.0im 0.25+0.0im ; 0.25+0.0im -1.0+0.0im]
        ),
        env,
        SimulationDetails(0.01, 0.1, 1000.0, Int64(1000.0/0.01))
    )

    calc__Λ!(cmrt_ctx)
    calc__Γ!(cmrt_ctx)

    if Threads.nthreads() == 1
        calc__g_g′_and_g″!(cmrt_ctx)
    else
        calc__g_g′_and_g″_with_threads!(cmrt_ctx)
    end

    calc__rates!(cmrt_ctx)
end




BenchmarkTools.DEFAULT_PARAMETERS.samples = 10
# @btime test__fret()
# @btime test__mrt()

# test__fret()
test__mrt()