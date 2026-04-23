
using Cthulhu
@time using MKL
@time using LinearAlgebra
@time using Base.Threads
@time using BenchmarkTools

@time include("../Physics/Physics.jl")
@time include("../PerturbationTheory/Fret.jl")
@time include("../PerturbationTheory/Mrt.jl")
@time include("../Utils/HighDimensionalDataContainer.jl")
@time include("../PerturbationTheory/Mrt_HighDim.jl")
using .MrtHighDim

# @time include("../PerturbationTheory/Cmrt.jl")

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



function test__mrt_high_dim()
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

    mrt_ctx = MrtHighDim.create__mrt_context(
        System(;
            size_of_system=2, 
            system_hamiltonian=[1.0+0.0im 0.25+0.0im ; 0.25+0.0im -1.0+0.0im]
        ),
        env,
        SimulationDetails(0.01, 0.1, 1000.0, Int64(1000.0/0.01))
    )

    MrtHighDim.calc__Λ!(mrt_ctx)
    MrtHighDim.calc__Γ!(mrt_ctx)

    if Threads.nthreads() == 1
        MrtHighDim.calc__g_g′_and_g″!(mrt_ctx)
    else
        MrtHighDim.calc__g_g′_and_g″_with_threads!(mrt_ctx)
    end

    MrtHighDim.calc__rates!(mrt_ctx)

    if Threads.nthreads() == 1
        MrtHighDim.calc__dissipations!(mrt_ctx)
    else
        # @descend calc__dissipations_with_threads!(mrt_ctx)
        MrtHighDim.calc__dissipations_with_threads!(mrt_ctx)
    end

    MrtHighDim.check__physics(mrt_ctx)
end



function test__cmrt()
    ω_c     = 1
    β       = 1 / ω_c
    J_0     = ComplexF64(0.5 * ω_c)
    Δ       = ComplexF64(0.5 * ω_c)
    γ       = 0.2
    ω_max   = ω_c * 2

    env = Environment()


    # 오이이잉 왜 freq_max = 2* omega_c 로 해야하는 거지? 이해 할 수가 없네?????
    add__spectral_density!(
        env, 
        SuperOhmicDebyeSpectralDensity(γ, ω_c), 
        SpectralDensityDecomposeInfo(2000, ω_max, 1/β, [1.0 0.0; 0.0 0.0])
    )
    add__spectral_density!(
        env, 
        SuperOhmicDebyeSpectralDensity(γ, ω_c), 
        SpectralDensityDecomposeInfo(2000, ω_max, 1/β, [0.0 0.0; 0.0 1.0])
    )
    # add__spectral_density!(
    #     env,
    #     DrudeLorentzSpectralDensity(0.2, 0.5), 
    #     SpectralDensityDecomposeInfo(2000, 15.0, 1.0, [0.0 0.0; 0.0 1.0])
    # )

    @printf(stderr, "total count is %d \n", length(env.effective_oscillators))

    cmrt_ctx = create__cmrt_context(
        System(;
            size_of_system=2, 
            system_hamiltonian=[Δ J_0 ; J_0 -Δ]
        ),
        env,
        SimulationDetails(0.1, 0.1, 5000.0, Int64(5000.0/0.1))
    )

    calc__Λ!(cmrt_ctx)
    calc__Γ!(cmrt_ctx)

    if Threads.nthreads() == 1
        calc__g_g′_and_g″!(cmrt_ctx)
    else
        calc__g_g′_and_g″_with_threads!(cmrt_ctx)
    end

    Rpd_hist = calc__R_pd!(cmrt_ctx)   # (ti, a, b) Complex
    R_hist   = calc__R_hist!(cmrt_ctx)     # (ti, a, b) Float64, cumulative trapezoid

    U_sys = cmrt_ctx.U_sys

    n = cmrt_ctx.system.n_sys
    σ0 = zeros(ComplexF64, n, n)

    σ0_site = [1.0+0.0im 0.0+0.0im; 0.0+0.0im 0.0+0.0im]

    σ0 = U_sys' * σ0_site * U_sys

    σ_hist = simulate__sigma_cmrt_time_dependent(
        cmrt_ctx;
        σ0=σ0,
        R_hist=R_hist,
        Rpd_hist=Rpd_hist,
        method=:rk4,
    )

    n_itr = cmrt_ctx.simulation_details.num_of_iteration
    dt    = cmrt_ctx.simulation_details.Δt

    save_filename   = "cmrt.txt"
    file_id         = open(save_filename, "w")

    @printf(file_id, "\n---- CMRT reduced dynamics (populations) ----\n")
    for ti in 1:n_itr
        t = (ti-1)*dt
        
        # 지금 σ_hist는 exciton basis 기준이니까, site basis 기준으로 바꿔줘야지.
        σ_exci = @view σ_hist[:,:,ti]
        σ_site = U_sys * σ_exci * U_sys' 
        p1 = real(σ_site[1,1])
        p2 = real(σ_site[2,2])
        @printf(file_id, "t=%10.4f  p1=% .6e  p2=% .6e  trace=% .6e\n", t, p1, p2, p1+p2)
    end

    K_final = @view R_hist[end, :, :]
    @printf(file_id, "\n---- Effective rate matrix at final time (R(t_end)) ----\n")
    @inbounds for a in 1:n
        for b in 1:n
            @printf(file_id, "%15.6e", K_final[a,b])
        end
        @printf(file_id, "\n")
    end

    @printf(stderr, "Program terminated")
end




BenchmarkTools.DEFAULT_PARAMETERS.samples = 10
# @btime test__fret()
# @btime test__mrt()

# test__fret()
test__mrt()
# test__cmrt()