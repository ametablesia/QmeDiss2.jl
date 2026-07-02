
using Cthulhu
@time using MKL
@time using LinearAlgebra
@time using Base.Threads
@time using BenchmarkTools
using Printf

include("../Physics/Physics.jl")


@time include("../PerturbationTheory/Fret.jl")
@time include("../PerturbationTheory/Mrt.jl")
@time include("../Utils/HighDimensionalDataContainer.jl")
@time include("../PerturbationTheory/Mrt_HighDim.jl")
using .MrtHighDim
@time include("../PerturbationTheory/Cmrt.jl")
@time include("../PerturbationTheory/CoherenceMrt.jl")
@time include("../PerturbationTheory/Rmrt.jl")


using .Physics
using .CoherenceMrt
using .Rmrt

println(filter(x -> occursin("Patternized", String(x)), string.(names(Main, all=true))))

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
    # CONTROLLER
    SCALE_UP        = 1.0 ### DO NOT USE NOT FULLY IMPLEMENTED!!!
    IS_MARKOVIAN    = false
    MAX_TIME        = Int64(floor(25 / SCALE_UP))
    TIMESTEP        = (0.01 / SCALE_UP)
    GAMMA           = 0.20 * SCALE_UP ## 말이 gamma지 reorg임. 0.20 기본값
    TEMPERATURE     = 1.0
    RMRT_METHOD     = :euler # :euler, :rk2, :rk4
    ASYM_C          = 0.0
    IS_SECULAR      = false
    INCLUDING_G1    = false
    IS_G1_MARKOVIAN = false


    ω_c     = 2.0 * SCALE_UP    # default: 1
    # β       = 1.0 / ω_c        # default: beta = (1 / ω_c)  
    β       = 1.0
    J_0     = ComplexF64(0.5 * ω_c) ## Default is 0.5
    Δ       = ComplexF64(0.5 * ω_c)
    # γ       = 0.2
    γ       = GAMMA * SCALE_UP
    ω_max   = ω_c * 50.0

    env = Environment()

    add__spectral_density!(
        env,
        DrudeLorentzSpectralDensity(γ, ω_c),
        # SuperOhmicDebyeSpectralDensity(γ, ω_c),
        SpectralDensityDecomposeInfo(
            # 2000,
            2000,
            ω_max,
            1 / β,
            [1.0 0.0; 0.0 0.0]
        )
    )

    add__spectral_density!(
        env,
        DrudeLorentzSpectralDensity(γ, ω_c),
        # SuperOhmicDebyeSpectralDensity(γ, ω_c),
        SpectralDensityDecomposeInfo(
            # 2000,
            2000,
            ω_max,
            1 / β,
            [0.0 0.0; 0.0 1.0 - ASYM_C]
        )
    )

    @printf(stderr, "total count is %d \n", length(env.effective_oscillators))


    rmrt_ctx = Rmrt.create__rmrt_context(
        System(;
            size_of_system = 2,
            system_hamiltonian = [
                Δ    J_0
                J_0 -Δ
            ] 
        ),
        env,
        SimulationDetails(
            TIMESTEP,
            TIMESTEP,
            MAX_TIME,
            Int64(MAX_TIME / TIMESTEP)
        )
    )


    @printf(stderr, "total count is %d \n", length(env.effective_oscillators))

    mrt_ctx = create__mrt_context(
        System(;
            size_of_system = 2,
            system_hamiltonian = [
                Δ    J_0
                J_0 -Δ
            ] 
        ),
        env,
        SimulationDetails(TIMESTEP, TIMESTEP, MAX_TIME, Int64(MAX_TIME / TIMESTEP))
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




BenchmarkTools.DEFAULT_PARAMETERS.samples = 10
# @btime test__fret()
# @btime test__mrt()

# test__fret()
test__mrt()
