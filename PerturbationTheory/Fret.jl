

# include("../Physics/SpectralDensity.jl")
# include("../Physics/System.jl")
# include("../Physics/Environment.jl")
# include("../Physics/Simulation.jl")
# include("../Physics/Dissipation.jl")
using LinearAlgebra
using LoopVectorization
using Base.Threads
include("../Physics/Physics.jl")

# computing context
mutable struct FretContext
    # input context
    system              ::System
    environment         ::Environment
    simulation_details  ::SimulationDetails

    # computing context
    # H_sys0      ::Vector{ComplexF64}         # site energy 
    H_sys0      ::Matrix{ComplexF64}
    λ           ::Matrix{ComplexF64}         # reorganization energy
    g           ::Array{ComplexF64, 3}       # time_idx, a, b
    g′          ::Array{ComplexF64, 3}       # time_idx, a, b
    g″          ::Array{ComplexF64, 3}       # time_idx, a, b
    G           ::Array{ComplexF64, 3}       # G is for calculating K_BA^i (dissipation rate) 추후에 이거 계산을 돌린 후에 rate나 dissipation 구하는게 나을지도.
    
    # output context
    transition_rate ::Array{Float64, 2}
    dissipation     ::Vector{Dissipation}

    function FretContext(system::System, environment::Environment, simulation_details::SimulationDetails)
        
        n_itr = simulation_details.num_of_iteration
        n_sys = system.n_sys
        
        # H_sys0  = diag(system.H_sys) # 대각성분만 추출.
        H_sys0  = zeros(ComplexF64, (n_sys, n_sys))
        λ       = zeros(ComplexF64, (n_sys, n_sys))
        g       = zeros(ComplexF64, (n_itr, n_sys, n_sys))
        g′      = zeros(ComplexF64, (n_itr, n_sys, n_sys))
        g″      = zeros(ComplexF64, (n_itr, n_sys, n_sys))
        G       = zeros(ComplexF64, (n_itr, n_sys, n_sys)) 

        transition_rate     = zeros(Float64, (n_sys, n_sys))
        dissipation         = Vector{Dissipation}()

        new(system, environment, simulation_details, H_sys0, λ, g, g′, g″, G, transition_rate, dissipation)
    end
end


function create__fret_context(
    system      ::System,
    environment ::Environment,
    simulation_details  ::SimulationDetails
)
    return FretContext(system, environment, simulation_details)
end

function init__fret_context!(context::FretContext)
    n_sys       = context.system.n_sys
    n_itr       = context.simulation_details.num_of_iteration
    
    context.H_sys0  = zeros(ComplexF64, (n_sys, n_sys))
    context.g       = zeros(ComplexF64, (n_itr, n_sys, n_sys))
    context.g′      = zeros(ComplexF64, (n_itr, n_sys, n_sys))
    context.g″      = zeros(ComplexF64, (n_itr, n_sys, n_sys))
    context.G       = zeros(ComplexF64, (n_itr, n_sys, n_sys))
end


function calc__λ!(context::FretContext)
    
    n_sys       = context.system.n_sys
    H_sys       = context.system.H_sys
    H_sys0      = context.H_sys0
    λ           = context.λ
    oscs        = context.environment.effective_oscillators
    n_osc       = context.environment.num_of_effective_oscillators

    @inbounds for osc_idx in 1:n_osc
        ω       = oscs[osc_idx].freq
        γ       = oscs[osc_idx].site_bath_coupling_strength

        @inbounds for b in 1:n_sys, a in b:n_sys
            λ[a,b] += ω * γ[a,a] * γ[b,b]
        end
    end

    # 대칭행렬화
    @inbounds for b in 1:n_sys, a in b:n_sys
        λ[b,a] = λ[a,b]
    end

    # H_sys0 생성 (reorg 빼서 dressing)
    H_sys0 .= H_sys # 값 복사, element wise (단순히 = 하면 안됨. ref 복사라서... 아니면 copy 쓰던가)
    @inbounds for k in 1:n_sys
        H_sys0[k,k] = H_sys[k,k] - λ[k,k]
    end
end

# g를 모든 시간에 대해서 만든다.
function calc__g!(context::FretContext)

    n_sys   = context.system.n_sys
    n_osc   = context.environment.num_of_effective_oscillators
    oscs    = context.environment.effective_oscillators

    n_itr   = context.simulation_details.num_of_iteration
    Δt      = context.simulation_details.Δt
    g       = context.g

    @inbounds for osc_idx in 1:n_osc
        ω       = oscs[osc_idx].freq
        coth    = oscs[osc_idx].coth
        γ       = oscs[osc_idx].site_bath_coupling_strength

        @inbounds for b in 1:n_sys, a in b:n_sys
            common = γ[a,a] * γ[b,b]
            
            @inbounds for time_idx = 1:n_itr
                t = Δt * (time_idx-1)
                ωt = ω * t

                g[time_idx,a,b] += common * ( (coth * (1.0 - cos(ωt))) + 1.0im*(sin(ωt) - ωt) )
            end
        end

        if osc_idx % 100 == 0
            @printf(stderr, "OSC %6d / %6d\n", osc_idx, n_osc)
        end
    end

    # 대칭 복사를 수행 (비대각 상삼각행렬만 붙이면 됨.)
    @inbounds for b in 1:n_sys, a in (b+1):n_sys
        @views g[:,b,a]  .= g[:,a,b]
    end
end

# g를 모든 시간에 대해서 만든다.
function calc__g_with_threads!(context::FretContext)

    n_sys   = context.system.n_sys
    n_osc   = context.environment.num_of_effective_oscillators
    oscs    = context.environment.effective_oscillators

    n_itr   = context.simulation_details.num_of_iteration
    Δt      = context.simulation_details.Δt
    g       = context.g


    # thread 경쟁상태 방지 위한, local sum 변수 메모리 많이 잡아먹음. 주의.
    n_ths = Threads.maxthreadid()
    g_locals = [zeros(ComplexF64, n_itr, n_sys, n_sys) for _ in 1:n_ths]

    @inbounds @threads for osc_idx in 1:n_osc
        ω       = oscs[osc_idx].freq
        coth    = oscs[osc_idx].coth
        γ       = oscs[osc_idx].site_bath_coupling_strength

        tid = threadid()
        g_local = g_locals[tid]

        @inbounds for b in 1:n_sys, a in b:n_sys
            common = γ[a,a] * γ[b,b]
            
            @inbounds for time_idx = 1:n_itr
                t = Δt * (time_idx-1)
                ωt = ω * t

                g_local[time_idx,a,b] += common * ( (coth * (1.0 - cos(ωt))) + 1.0im*(sin(ωt) - ωt) )
            end
        end

        if osc_idx % 100 == 0
            @printf(stderr, "OSC %6d / %6d\n", osc_idx, n_osc)
        end
    end

    # reduction (single-thread, cheap)
    fill!(g, 0)
    for tid in 1:n_ths
        g .+= g_locals[tid]
    end


    # 대칭 복사를 수행 (비대각 상삼각행렬만 붙이면 됨.)
    @inbounds for b in 1:n_sys, a in (b+1):n_sys
        @views g[:,b,a]  .= g[:,a,b]
    end
end


function calc__g′_and_g″!(context::FretContext)

    n_sys   = context.system.n_sys
    n_osc   = context.environment.num_of_effective_oscillators
    oscs    = context.environment.effective_oscillators

    n_itr   = context.simulation_details.num_of_iteration
    Δt      = context.simulation_details.Δt

    g′      = context.g′
    g″     = context.g″

    # 모든 oscillator 돌면서, 모든 시간에 대해서 g', g''를 미리 계산.
    @inbounds for osc_idx in 1:n_osc
        ω       = oscs[osc_idx].freq
        coth    = oscs[osc_idx].coth
        γ       = oscs[osc_idx].site_bath_coupling_strength

        @inbounds for b in 1:n_sys, a in b:n_sys
            common = γ[a,a] * γ[b,b]
            
            @inbounds for time_idx in 1:n_itr

                t = Δt * (time_idx-1)
                ωt = ω * t
                
                g′[time_idx,a,b]   += common * ( (ω * coth * sin(ωt)) + 1.0im*(ω * (cos(ωt) - 1.0)) )
                g″[time_idx,a,b]   += common * ( (ω*ω * coth * cos(ωt)) + 1.0im*(-ω*ω * sin(ωt)) )

            end
        end
    end
    
    #대칭 복사를 수행.
    @inbounds for b in 1:n_sys, a in (b+1):n_sys
        @views g′[:,b,a] .= g′[:,a,b]
        @views g″[:,b,a] .= g″[:,a,b]
    end
end

function calc__rates!(context::FretContext)

    n_sys   = context.system.n_sys
    H_sys0  = context.H_sys0
    rate    = context.transition_rate

    n_itr   = context.simulation_details.num_of_iteration
    Δt      = context.simulation_details.Δt

    G       = context.G
    g       = context.g   
    λ       = context.λ

    fill!(rate, 0.0)

    # 대칭이므로, 절반만 해도 됨.
    @inbounds for b in 1:n_sys, a in (b+1):n_sys
        ϵ_a     = H_sys0[a,a]
        ϵ_b     = H_sys0[b,b]
        V_ba    = H_sys0[b,a]   # coupling

        @inbounds for time_idx in 1:n_itr
            t = (time_idx - 1) * Δt

            g_aa, g_bb, g_ab = g[time_idx,a,a], g[time_idx,b,b], g[time_idx,a,b]
            λ_aa, λ_ab, λ_bb = λ[a,a], λ[a,b], λ[b,b]

            common = -( g_aa - 2*g_ab + g_bb ) - 1.0im*( λ_aa - 2*λ_ab + λ_bb )*t
            exponent_ab = common + 1.0im*( ϵ_b - ϵ_a )*t
            exponent_ba = common - 1.0im*( ϵ_b - ϵ_a )*t

            # rate 계산하는 김에 커널 재활용 (dissipation 구할때 사용.)
            G[time_idx,a,b] = exp(real(exponent_ab)) * (cos(imag(exponent_ab)) + 1.0im*sin(imag(exponent_ab)))
            G[time_idx,b,a] = exp(real(exponent_ba)) * (cos(imag(exponent_ba)) + 1.0im*sin(imag(exponent_ba)))

            # trapezoidal method
            trapezoidal_weight = (time_idx == 1 || time_idx == n_itr) ? 0.5 : 1.0

            # rate[a,b] += trapezoidal_weight * real(exp(exponent_ab))
            # rate[b,a] += trapezoidal_weight * real(exp(exponent_ba)) # 더 빠르게는 exp real로 바꾸는...
            rate[a,b] += trapezoidal_weight * real(G[time_idx,a,b])
            rate[b,a] += trapezoidal_weight * real(G[time_idx,b,a])
        end

        # prefactor = 2.0 * V_ba * V_ba * Δt
        prefactor = 2.0 * abs2(V_ba) * Δt
        rate[a,b] *= prefactor
        rate[b,a] *= prefactor
    end

    @printf(stderr, "---- Population Transfer Rate Constants (a.u.) ----\n")
    @inbounds for a in 1:n_sys
        @inbounds for b in 1:n_sys
            @printf(stderr, "%15.6e", rate[a,b])
        end
        @printf(stderr, "\n")
    end
    @printf(stderr, "\n")
end

function calc__dissipations!(context::FretContext)
    n_sys   = context.system.n_sys
    n_osc   = context.environment.num_of_effective_oscillators
    oscs    = context.environment.effective_oscillators
    
    Δt      = context.simulation_details.Δt
    n_itr   = context.simulation_details.num_of_iteration

    H_sys0  = context.H_sys0
    G       = context.G

    for osc_idx in 1:n_osc
        osc     = oscs[osc_idx]
        ω       = osc.freq
        coth    = osc.coth
        γ       = osc.site_bath_coupling_strength
        spread  = osc.spread

        i_dissipation = context.dissipation[osc_idx].i_dissipation
        j_dissipation = context.dissipation[osc_idx].j_dissipation
        k_dissipation = context.dissipation[osc_idx].k_dissipation 

        for b in 1:n_sys, a in (b+1):n_sys

            # trapezoidal 적분용 acc 변수
            integral_ab = 0.0
            integral_ba = 0.0

            for time_idx in 1:n_itr
                # 커널 재활용
                G_ab = G[time_idx,a,b]
                G_ba = G[time_idx,b,a]

                t   = (time_idx - 1)*Δt
                ωt  = ω*t

                # rate kernel을 dissipation kernel로 바꾸기 위해서 factor를 곱함.
                factor = (cos(ωt) + 1.0im(-coth * sin(ωt)))
                integrand_ab = real(G_ab * factor)
                integrand_ba = real(G_ba * factor)

                # trapezoidal method 적분 시작!
                trapezoidal_weight = (time_idx == 1 || time_idx == n_itr) ? 0.5 : 1.0

                integral_ab += trapezoidal_weight * integrand_ab
                integral_ba += trapezoidal_weight * integrand_ba
            end

            # I dissipation
            i_dissipation[a,b] = integral_ab * Δt
            i_dissipation[b,a] = integral_ba * Δt

            ## J / K dissipation
            V_ab        ::ComplexF64    = H_sys0[a,b]
            two_V_sq    ::Float64       = 2 * abs2(V_ab)
            γ_diff      ::Float64       = γ[b,b] - γ[a,a]
            λ_gen       ::Float64       = ω * γ_diff * γ_diff

            k_dissipation[a,b] = two_V_sq * λ_gen * i_dissipation[a,b]
            k_dissipation[b,a] = two_V_sq * λ_gen * i_dissipation[b,a]

            j_dissipation[a,b] = k_dissipation[a,b] / spread
            j_dissipation[b,a] = k_dissipation[b,a] / spread
        end
    end
end

# 대각 계산 해야하는데 b+1:n_sys로 되어 있어서 대각은 계산이 안되는 버그가 있엇음
# b:n_sys로 바꿔놓음.