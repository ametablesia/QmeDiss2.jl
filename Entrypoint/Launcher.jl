
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
    γ       = 1             ## DEFAULT = 0.20
    ω_max   = ω_c * 50

    env = Environment()


    # 오이이잉 왜 freq_max = 2* omega_c 로 해야하는 거지? 이해 할 수가 없네?????
    add__spectral_density!(
        env, 
        # SuperOhmicDebyeSpectralDensity(γ, ω_c), 
        DrudeLorentzSpectralDensity(γ, ω_c),
        SpectralDensityDecomposeInfo(2000, ω_max, 1/β, [1.0 0.0; 0.0 0.0])
    )
    add__spectral_density!(
        env,
        DrudeLorentzSpectralDensity(γ, ω_c),
        # SuperOhmicDebyeSpectralDensity(γ, ω_c), 
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
        SimulationDetails(0.01, 0.01, 35, Int64(35/0.01))
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
        @printf(file_id, "t=%10.4f  p1=%.6e  p2=%.6e  trace=%.6e p1_exci=%.6e  p2_exci=%.6e c12_re_exci=%.6e c12_im_exci=%.6e\n", t, p1, p2, p1+p2, real(σ_exci[1,1]), real(σ_exci[2,2]), real(σ_exci[1,2]), imag(σ_exci[1,2]))
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

function test__rmrt()

    # CONTROLLER
    SCALE_UP        = 1.0 ### DO NOT USE NOT FULLY IMPLEMENTED!!!
    IS_MARKOVIAN    = false
    MAX_TIME        = Int64(floor(25 / SCALE_UP))
    TIMESTEP        = (0.01 / SCALE_UP)
    GAMMA           = 0.20 * SCALE_UP ## 말이 gamma지 reorg임. 0.20 기본값
    TEMPERATURE     = 1.0
    RMRT_METHOD     = :rk4 # :euler, :rk2, :rk4
    ASYM_C          = 0.0
    IS_SECULAR      = false
    INCLUDING_G1    = false
    IS_G1_MARKOVIAN = false


    ω_c     = 5 * SCALE_UP    # default: 1
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


    # 1. Exciton basis 계산
    @printf(stderr, "1. exicton basis 계산 \n")
    Rmrt.calc__exciton_energy!(rmrt_ctx)

    # 2. site-basis coupling을 exciton-basis coupling으로 변환
    @printf(stderr, "2. site-basis coupling을 exciton-basis coupling으로 변환 \n")
    Rmrt.calc__exciton_basis_and_γ_exci!(rmrt_ctx)

    @printf(stderr, "Calculated energy gap")
    @show rmrt_ctx.ϵ_exci[2] - rmrt_ctx.ϵ_exci[1]
    @show rmrt_ctx.ϵ_exci_0[2] - rmrt_ctx.ϵ_exci_0[1]


    # 3. g, g′, g″ 계산
    @printf(stderr, "3. g, g′, g″ 계산 \n")
    if Threads.nthreads() == 1
        Rmrt.calc__g_g′_g″!(rmrt_ctx)
    else
        Rmrt.calc__g_g′_g″_with_threads!(rmrt_ctx)
    end

    # 3.1. RK2/RK4용 half-shifted g, g′, g″ 계산
    #
    # 생성자에서는 메모리를 잡지 않음.
    # 여기서 실제로 필요할 때만 allocate + compute.
    if RMRT_METHOD in (:rk2, :rk4)
        @printf(stderr, "3.1. half-shifted g, g′, g″ 계산 \n")

        Rmrt.ensure__half_shifted_grid!(
            rmrt_ctx;
            recompute = false,
            use_threads = Threads.nthreads() > 1,
            verbose = true,
        )
    end

    # 3.2 Λ 곧 reorganization energy 계산. (뭐 Markovian일때만 계산하면 될듯?)
    if IS_MARKOVIAN
        @printf(stderr, "3.2. Λ 계산 \n")
        if Threads.nthreads() == 1
            Rmrt.calc__Λ!(rmrt_ctx)
        else
            Rmrt.calc__Λ_with_threads!(rmrt_ctx)
        end
    end

    # 4. 초기 density matrix
    @printf(stderr, "4. 초기 density matrix \n")
    Rmrt.set__initial_σ_site!(rmrt_ctx; init_site = 1)

    # 5. 전체 σ, σ′ time propagation
    @printf(stderr, "5. 전체 σ, σ′ time propagation \n")
    # if Threads.nthreads() == 1
    #     Rmrt.calc__σ_σ′!(rmrt_ctx)
    # else
    #     Rmrt.calc__σ_σ′_with_threads!(rmrt_ctx)
    # end
    # Rmrt.calc__σ_σ′_with_population_closure!(rmrt_ctx)
    # Rmrt.calc__σ_σ′_with_population_closure_secular!(rmrt_ctx; use_secular = false, secular_tol = 1.0e-10,)
    # Rmrt.calc__σ_σ′_with_population_closure_secular!(rmrt_ctx; use_secular = true, secular_tol = 1.0e-10,)
    if IS_MARKOVIAN
        Rmrt.calc__σ_σ′_with_markovian!(rmrt_ctx; use_secular = false)
    else
        # Rmrt.calc__σ_σ′_secular_core!(rmrt_ctx; method=:euler)
        # 이걸로 돌리면 됨 완성본
        # Rmrt.calc__σ_σ′_secular_core!(
        #     rmrt_ctx;
        #     method = RMRT_METHOD,
        #     use_secular=IS_SECULAR,
        #     # markovianize_PL0QG1QL1P_G1 = IS_G1_MARKOVIAN,
        #     collapse_tau_PL0QG1QL1P_G1 = IS_G1_MARKOVIAN,
        #     include_PL0QG1QL1P = INCLUDING_G1,
        #     auto_prepare_half_shifted_grid = false,
        # )

        result = Rmrt.calc__σ_σ′_secular_core!(
            rmrt_ctx;
            method = RMRT_METHOD,
            use_secular=IS_SECULAR,
        )
        
        # kr = result.kernel_rowsum

        # # branch order: 1=LL, 2=LR, 3=RL, 4=RR
        # rowsum_total = kr[:, :, :, 1] .+ kr[:, :,:, 2] .+ kr[ :, :,:, 3] .+ kr[:, :,:, 4]

        # maximum(abs.(rowsum_total))
        # maximum(abs.(imag.(rowsum_total)))

        # for a in 1:2, b in 1:2
        #     println("input=($a,$b), max rowsum = ", maximum(abs.(rowsum_total[a,b,:])))
        # end

        # for br in 1:4
        #     println("branch $br max rowsum = ", maximum(abs.(kr[:,:,:,br])))
        # end

        # ko = result.kernel_output

        # # input (1,1)
        # loss_11_from_11 = ko[1, 1, 1, :, 1] .+ ko[1, 1, 1, :, 4]  # output 11: LL + RR
        # gain_22_from_11 = ko[2, 1, 1, :, 2] .+ ko[2, 1, 1, :, 3]  # output 22: LR + RL

        # println("input (1,1): max |loss+gain| = ",
        #     maximum(abs.(loss_11_from_11 .+ gain_22_from_11)))
        # println("input (1,1): max |loss-gain| = ",
        #     maximum(abs.(loss_11_from_11 .- gain_22_from_11)))
        # println("input (1,1): max Im(loss+gain) = ",
        #     maximum(abs.(imag.(loss_11_from_11 .+ gain_22_from_11))))

        # # input (2,2)
        # gain_11_from_22 = ko[1, 2, 2, :, 2] .+ ko[1, 2, 2, :, 3]  # output 11: LR + RL
        # loss_22_from_22 = ko[2, 2, 2, :, 1] .+ ko[2, 2, 2, :, 4]  # output 22: LL + RR

        # println("input (2,2): max |gain+loss| = ",
        #     maximum(abs.(gain_11_from_22 .+ loss_22_from_22)))
        # println("input (2,2): max |gain-loss| = ",
        #     maximum(abs.(gain_11_from_22 .- loss_22_from_22)))
        # println("input (2,2): max Im(gain+loss) = ",
        #     maximum(abs.(imag.(gain_11_from_22 .+ loss_22_from_22))))


        # Rmrt.calc__σ_σ′_secular_core!(rmrt_ctx; method=:rk4)

        # println("c12log")
        # open("phase_trace_c12.log", "w") do io
        #     Rmrt.calc__σ_σ′_secular_core!(
        #         rmrt_ctx;
        #         trace_phase_terms = true,
        #         trace_phase_pair = (1, 2),
        #         trace_phase_every = 1,
        #         trace_phase_eps = 1.0e-6,
        #         trace_phase_io = io,
        #         use_L0Q_memory_return = true,
        #         use_local_population_to_coherence = true,
        #         method=:euler
        #     )
        # end

        # Rmrt.calc__σ_σ′_secular_core!(
        #     rmrt_ctx;
        #     heom_file = "heom.txt",
        #     use_heom_input = false,
        #     heom_teacher_forcing_cutoff = 10.0,

        #     use_population_closure = false,
        #     use_local_population_to_coherence = true,
        #     use_L0Q_memory_return = true,

        #     use_secular = false,
        #     method = :rk2,
        #     verbose = true,
        # )

        # println("c12log")
        # open("phase_trace_c12.log", "w") do io
        #     Rmrt.calc__σ_σ′_secular_core!(
        #         rmrt_ctx;
        #         heom_file = "heom.txt",
        #         use_heom_input = true,
        #         trace_phase_terms = true,
        #         trace_phase_pair = (1, 2),
        #         trace_phase_every = 1,
        #         trace_phase_eps = 1.0e-6,
        #         trace_phase_io = io,
        #         use_L0Q_memory_return = true,
        #         use_local_population_to_coherence = true,
        #         method=:euler
        #     )
        # end

        # println("p11log")
        # open("population_rhs_trace.log", "w") do io
        #     Rmrt.calc__σ_σ′_secular_core!(
        #         rmrt_ctx;
        #         heom_file = "heom.txt",
        #         use_heom_input = true,
        #         trace_population_rhs = true,
        #         trace_population_indices = (1),
        #         trace_population_every = 1,
        #         trace_population_io = io,
        #         use_L0Q_memory_return = true,
        #         use_local_population_to_coherence = true,
        #         method=:euler
        #     )
        # end

        # println("p22log")
        # open("population_rhs_trace_p22.log", "w") do io
        #     Rmrt.calc__σ_σ′_secular_core!(
        #         rmrt_ctx;
        #         heom_file = "heom.txt",
        #         use_heom_input = true,
        #         trace_population_rhs = true,
        #         trace_population_indices = (2),
        #         trace_population_every = 1,
        #         trace_population_io = io,
        #         use_L0Q_memory_return = true,
        #         use_local_population_to_coherence = true,
        #         method=:euler
        #     )
        # end


        # println("p1 population decomposition log")
        # open("pop_decomp_p1.log", "w") do io
        #     Rmrt.calc__σ_σ′_secular_core!(
        #         rmrt_ctx;
        #         heom_file = "heom.txt",
        #         use_heom_input = true,

        #         trace_population_decomp = true,
        #         trace_population_decomp_indices = (1,),
        #         trace_population_decomp_every = 1,
        #         trace_population_decomp_io = io,

        #         use_local_population_to_coherence = true,
        #         use_local_coherence_to_population = true,
        #         use_population_memory_population_input = true,
        #         use_population_memory_coherence_input = true,
        #         use_L0Q_memory_return = true,

        #         method = :euler,
        #         use_threads = false,
        #     )
        # end
    end


    Rmrt.save__rmrt_reduced_dynamics_serialized!(
        rmrt_ctx;
        save_filename = "rmrt.txt",
        basis = :both,
        write_derivative = true,
    )

    return rmrt_ctx
end


function test__coherence_mrt()
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

    @printf(stderr, "total oscillator count = %d\n", length(env.effective_oscillators))

    mrt_ctx = CoherenceMrt.create__mrt_context(
        System(;
            size_of_system=2,
            system_hamiltonian=[1.0+0.0im 0.25+0.0im;
                                0.25+0.0im -1.0+0.0im]
        ),
        env,
        SimulationDetails(0.01, 0.1, 1000.0, Int64(1000.0 / 0.01))
    )

    CoherenceMrt.calc__Λ!(mrt_ctx)
    CoherenceMrt.calc__Γ!(mrt_ctx)

    if Threads.nthreads() == 1
        CoherenceMrt.calc__g_g′_and_g″!(mrt_ctx)
    else
        CoherenceMrt.calc__g_g′_and_g″_with_threads!(mrt_ctx)
    end

    CoherenceMrt.calc__rates!(mrt_ctx)

    if Threads.nthreads() == 1
        CoherenceMrt.calc__dissipations!(mrt_ctx)
    else
        CoherenceMrt.calc__dissipations_with_threads!(mrt_ctx)
    end

    CoherenceMrt.check__physics(mrt_ctx)

    # ------------------------------------------------------------
    # coherence test (exciton-basis sigma_{alpha beta})
    # ------------------------------------------------------------
    n_sys = mrt_ctx.system.n_sys
    n_itr = mrt_ctx.simulation_details.num_of_iteration
    Δt    = mrt_ctx.simulation_details.Δt
    U     = mrt_ctx.U_sys

    # site-basis initial density matrix (Hermitian, trace = 1)
    ρ_site0 = ComplexF64[
        0.70 + 0.00im   0.20 + 0.10im
        0.20 - 0.10im   0.30 + 0.00im
    ]

    # transform to exciton basis: sigma = U^\dagger rho_site U
    σ0 = U' * ρ_site0 * U

    σ_history    = zeros(ComplexF64, n_sys, n_sys, n_itr)
    dotσ_history = zeros(ComplexF64, n_sys, n_sys, n_itr)

    σ_history[:, :, 1] .= σ0

    @printf(stderr, "\n================ coherence test ================\n")
    @printf(stderr, "Initial sigma in exciton basis\n")
    for a in 1:n_sys
        for b in 1:n_sys
            z = σ_history[a, b, 1]
            @printf(stderr, "σ0[%d,%d] = % .8e %+.8ei\n", a, b, real(z), imag(z))
        end
    end

    print_indices = unique(sort!(Int[
        1,
        min(10, n_itr),
        min(100, n_itr),
        min(1000, n_itr),
        n_itr,
    ]))

    # simple forward Euler propagation of coherence sector only
    # note: the current RHS fills only off-diagonal entries.
    for time_idx in 1:n_itr
        σ_now = @view σ_history[:, :, time_idx]
        dotσ_now = @view dotσ_history[:, :, time_idx]

        CoherenceMrt.calc__coherence_rhs!(dotσ_now, mrt_ctx, σ_now, time_idx)

        @printf(stderr, "CURRENT TIMEIDX: %d\n", time_idx)
        if time_idx in print_indices
            t = (time_idx - 1) * Δt
            @printf(stderr, "\n---- coherence snapshot: time_idx = %d, t = %.6f ----\n", time_idx, t)
            for a in 1:n_sys
                for b in 1:n_sys
                    zσ   = σ_now[a, b]
                    zdot = dotσ_now[a, b]
                    @printf(stderr,
                        "σ[%d,%d] = % .8e %+.8ei    dotσ[%d,%d] = % .8e %+.8ei\n",
                        a, b, real(zσ), imag(zσ), a, b,
                        real(zdot), imag(zdot)
                    )
                end
            end
        end

        if time_idx < n_itr
            σ_next = @view σ_history[:, :, time_idx + 1]
            @inbounds for b in 1:n_sys, a in 1:n_sys
                σ_next[a, b] = σ_now[a, b] + Δt * dotσ_now[a, b]
            end

            # keep Hermiticity numerically
            σ_next[:, :] .= 0.5 .* (σ_next[:, :] .+ adjoint(σ_next[:, :]))

            # keep trace normalized
            trσ = real(sum(diag(σ_next)))
            if abs(trσ) > 1e-14
                σ_next[:, :] ./= trσ
            end
        end
    end

    @printf(stderr, "\nFinal sigma in exciton basis\n")
    for a in 1:n_sys
        for b in 1:n_sys
            z = σ_history[a, b, n_itr]
            @printf(stderr, "σ_final[%d,%d] = % .8e %+.8ei\n", a, b, real(z), imag(z))
        end
    end

    @printf(stderr, "\nFinal dot sigma\n")
    for a in 1:n_sys
        for b in 1:n_sys
            z = dotσ_history[a, b, n_itr]
            @printf(stderr, "dotσ_final[%d,%d] = % .8e %+.8ei\n", a, b, real(z), imag(z))
        end
    end

    return (; mrt_ctx, ρ_site0, σ0, σ_history, dotσ_history)
end



BenchmarkTools.DEFAULT_PARAMETERS.samples = 10
# @btime test__fret()
# @btime test__mrt()

# test__fret()
# test__mrt()
# test__cmrt()
# test__coherence_mrt()

test__rmrt()