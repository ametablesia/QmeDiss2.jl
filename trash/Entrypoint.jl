
# module Entrypoint
# end

# mutable struct CalculationOption
# aa::Int64
# end

# function calc__(CalculationOption)
# return 0
# end


using LinearAlgebra
using BenchmarkTools

# H_sys = rand(ComplexF64, (1000, 1000))
# H_sys = H_sys + H_sys'

# H = Hermitian(H_sys)
# F = eigen(H)

# @btime eigen(H)

# eigvals = F.values
# println(typeof(eigvals), eigvals)
# eigvecs = F.vectors
# println(typeof(eigvecs), eigvecs)


mutable struct System
    n_sys   ::Int64
    H_sys   ::Matrix{ComplexF64}
end

function set__system_hamiltonian!(
    system  ::System,
    H_sys   ::Matrix{ComplexF64}
)
    if ishermitian(H_sys) == false
        println("Hsys is not Hermitian!")
    end

    system.H_sys = H_sys
end

# H_sys = rand(ComplexF64, (1000, 1000))
# system ::System = System(H_sys)
# set__system_hamiltonian!(system, H_sys)




# mutable struct Interaction
    
# end

# mutable struct Bath

# end


mutable struct SpectralDensity
    aa
end

mutable struct EffectiveOscillator
    spectral_density_id  ::Int64
    
    freq    ::Float64
    coth    ::Float64
    spread  ::Float64

    site_bath_coupling_strength       ::Matrix{ComplexF64}
end

mutable struct Dissipation
    effective_oscillator_id ::Int64
    
    i_dissipation
    j_dissipation
    k_dissipation
end

mutable struct Environment
    # internal
    num_of_effective_oscillators::Int64
    spectral_densities      ::Vector{SpectralDensity}
    effective_oscillators    ::Vector{EffectiveOscillator}

    # # Output
    # dissipation             ::Vector{Dissipation}
end


function add__spectral_density!(
    environment::Environment
)

    return nothing
end







mutable struct OpenQuantumSystem
    system      ::System
    environment ::Environment
end

mutable struct SimulationDetails
    Δt                  ::Float64
    Δt_print            ::Float64
    t_max               ::Float64
    num_of_iteration    ::Int64
end

# computing context
mutable struct FretContext
    # input context
    system              ::System
    environment         ::Environment
    simulation_details  ::SimulationDetails

    # computing context
    H_sys0      ::Vector{ComplexF64}         # site energy 
    λ           ::Matrix{ComplexF64}         # reorganization energy
    g           ::Array{ComplexF64, 3}       # a, b, time_idx
    g′          ::Array{ComplexF64, 3}       # a, b, time_idx
    g′′         ::Array{ComplexF64, 3}       # a, b, time_idx
    G           ::Array{ComplexF64, 3}       # G is for calculating K_BA^i (dissipation rate) 추후에 이거 계산을 돌린 후에 rate나 dissipation 구하는게 나을지도.
    
    # output context
    transition_rate ::Array{Float64, 2}
    dissipation     ::Vector{Dissipation}
end

function init__fret_context!(context::FretContext)
    n_sys       = context.system.n_sys
    n_itr       = context.simulation_details.num_of_iteration
    
    context.H_sys0  = context.system.H_sys
    context.g       = zeros(ComplexF64, (n_sys, n_sys, n_itr))
    context.g′      = zeros(ComplexF64, (n_sys, n_sys, n_itr))
    context.g′′     = zeros(ComplexF64, (n_sys, n_sys, n_itr))
    context.G       = zeros(ComplexF64, (n_sys, n_sys, n_itr))

end


# g를 모든 시간에 대해서 만든다.
function calc__g!(context::FretContext)

    n_sys   = context.system.n_sys
    n_osc   = context.environment.num_of_effective_oscillators
    oscs    = context.environment.effective_oscillators

    n_itr   = context.simulation_details.num_of_iteration
    Δt      = context.simulation_details.Δt

    @inbounds for osc_idx in 1:n_osc
        ω       = oscs[osc_idx].freq
        coth    = oscs[osc_idx].coth
        γ       = oscs[osc_idx].site_bath_coupling_strength

        @inbounds for b in 1:n_sys, a in (b+1):n_sys
            common = γ[a,a] * γ[b,b]
            
            for time_idx = 1:n_itr
                t = Δt * (time_idx-1)
                ωt = ω * t

                g[a,b,time_idx] += common * ( (coth * (1.0 - cos(ωt))) + 1.0im*(sin(ωt) - ωt) )
            end
        end
    end

    # 대칭 복사를 수행.
    @inbounds for b in 1:n_sys, a in (b+1):n_sys
        g[b,a,:]  .= g[a,b,:]
    end
end

function calc__g′_and_g′′!(context::FretContext)

    n_sys   = context.system.n_sys
    n_osc   = context.environment.num_of_effective_oscillators
    oscs    = context.environment.effective_oscillators

    n_itr   = context.simulation_details.num_of_iteration
    Δt      = context.simulation_details.Δt

    g′      = context.g′
    g′′     = context.g′′

    # 모든 oscillator 돌면서, 모든 시간에 대해서 g', g''를 미리 계산.
    @inbounds for osc_idx in 1:n_osc
        ω       = oscs[osc_idx].freq
        coth    = oscs[osc_idx].coth
        γ       = oscs[osc_idx].site_bath_coupling_strength

        @inbounds for b in 1:n_sys, a in (b+1):n_sys
            common = γ[a,a] * γ[b,b]
            
            @inbounds for time_idx in 1:n_itr

                t = Δt * (time_idx-1)
                ωt = ω * t
                
                g′[a,b,time_idx]    += common * ( (ω * coth * sin(ωt)) + 1.0im*(ω * (cos(ωt) - 1.0)) )
                g′′[a,b,time_idx]   += common * ( (ω*ω * coth * cos(ωt)) + 1.0im*(-ω*ω * sin(ωt)) )

            end
        end
    end
    
    # 대칭 복사를 수행.
    @inbounds for b in 1:n_sys, a in (b+1):n_sys
        g′[b,a,:]  .= g′[a,b,:]
        g′′[b,a,:] .= g′′[a,b,:]
    end
end

function calc__rates!(context::FretContext)

    n_sys   = context.system.n_sys
    H_sys0  = context.H_sys0
    rate    = context.transition_rate

    n_itr   = context.simulation_details.num_of_iteration
    Δt      = context.simulation_details.Δt

    G    = context.G

    fill!(rate, 0.0)

    # 대칭이므로, 절반만 해도 됨.
    @inbounds for b in 1:n_sys, a in (b+1):n_sys
        ϵ_a     = H_sys0[a,a]
        ϵ_b     = H_sys0[b,b]
        V_ba    = H_sys0[b,a]   # coupling

        @inbounds for time_idx in 1:n_itr
            t = (time_idx - 1) * Δt

            g_aa, g_bb, g_ab = g[a,a,time_idx], g[b,b,time_idx], g[a,b,time_idx]
            λ_aa, λ_ab, λ_bb = λ[a,a], λ[a,b], λ[b,b]

            common = -( g_aa - 2*g_ab + g_bb ) - 1.0im*( λ_aa - 2*λ_ab + λ_bb )*t
            exponent_ab = common + 1.0im*( ϵ_b - ϵ_a )*t
            exponent_ba = common - 1.0im*( ϵ_b - ϵ_a )*t

            # rate 계산하는 김에 커널 재활용
            G[a,b,time_idx] = exp(real(exponent_ab)) * (cos(imag(exponent_ab)) + 1.0im*sin(exponent_ab))
            G[a,b,time_idx] = exp(real(exponent_ba)) * (cos(imag(exponent_ba)) + 1.0im*sin(exponent_ba))

            # trapezoidal method
            trapezoidal_weight = (time_idx == 1 || time_idx == n_itr) ? 0.5 : 1.0

            # rate[a,b] += trapezoidal_weight * real(exp(exponent_ab))
            # rate[b,a] += trapezoidal_weight * real(exp(exponent_ba)) # 더 빠르게는 exp real로 바꾸는...
            rate[a,b] += trapezoidal_weight * exp(real(exponent_ab)) * cos(imag(exponent_ab))
            rate[b,a] += trapezoidal_weight * exp(real(exponent_ba)) * cos(imag(exponent_ba))
        end

        prefactor = 2.0 * V_ba * V_ba * Δt
        rate[a,b] *= prefactor
        rate[b,a] *= prefactor
    end
end

function calc__dissipations!(context::FretContext)
    n_sys   = context.system.n_sys
    n_osc   = context.environment.num_of_effective_oscillators
    oscs    = context.environment.effective_oscillators
    
    Δt      = context.simulation_details.Δt
    n_itr   = context.simulation_details.num_of_iteration

    H_sys0  = context.H_sys0
    G       = context.G

    i_dissipation = context.dissipation.i_dissipation
    j_dissipation = context.dissipation.j_dissipation
    k_dissipation = context.dissipation.k_dissipation 

    for osc_idx in 1:n_osc
        osc     = oscs[osc_idx]
        ω       = osc.freq
        coth    = osc.coth
        γ       = osc.site_bath_coupling_strength
        spread  = osc.spread

        for b in 1:n_sys, a in (b+1):n_sys

            # trapezoidal 적분용 acc 변수
            integral_ab = 0.0
            integral_ba = 0.0

            for time_idx in 1:n_itr
                # 커널 재활용
                G_ab = G[a,b,time_idx]
                G_ba = G[b,a,time_idx]

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
            V_ab        ::Float64   = H_sys0[a,b]
            two_V_sq    ::Float64   = 2 * V_ab * V_ab
            γ_diff      ::Float64   = γ[b,b] - γ[a,a]
            λ_gen       ::Float64   = ω * γ_diff * γ_diff

            k_dissipation[a,b] = two_V_sq * λ_gen * i_dissipation[a,b]
            k_dissipation[b,a] = two_V_sq * λ_gen * i_dissipation[b,a]

            j_dissipation[a,b] = k_dissipation[a,b] / spread
            j_dissipation[b,a] = k_dissipation[b,a] / spread
        end
    end
end