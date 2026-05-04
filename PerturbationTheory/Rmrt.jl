
module Rmrt

include("../Utils/HighDimensionalDataContainer.jl")
include("../Physics/Physics.jl")

using Base.Threads
using LinearAlgebra
import Base: getindex
using Printf

using ..Physics:
    System,
    Environment,
    SimulationDetails,
    Dissipation,
    DrudeLorentzSpectralDensity,
    SpectralDensityDecomposeInfo,
    add__spectral_density!

# 가용가능한 Canonical Pattern들은...
# g:
#     aaaa
#     aabb

# dot_g:
#     aaaa
#     aaab
#     aabb
#     aabc
#     abaa
#     abbb
#     abcc

# ddot_g:
#     abba
#     abbc
#     abcd


# 매크로로 타입 정의
@patternized Patternized_g (n_sys::Int, n_itr::Int) (t::Int, a::Int, b::Int, c::Int, d::Int) begin
    rule(αααα, Matrix{T},   zeros(T, n_itr, n_sys), (t, a), a == b && b == c && c == d)
    rule(ααββ, Array{T,3},  zeros(T, n_itr, n_sys, n_sys), (t, a, c), a == b && c == d && a != c)
end

@patternized Patternized_g′ (n_sys::Int, n_itr::Int) (t::Int, a::Int, b::Int, c::Int, d::Int) begin
    rule(αααα, Matrix{T},   zeros(T, n_itr, n_sys), (t, a), a == b && b == c && c == d)
    rule(αααβ, Array{T,3},  zeros(T, n_itr, n_sys, n_sys), (t, a, d), a == b && b == c && a != d)
    rule(ααββ, Array{T,3},  zeros(T, n_itr, n_sys, n_sys), (t, a, c), a == b && c == d && a != c)
    rule(ααβγ, Array{T,4},  zeros(T, n_itr, n_sys, n_sys, n_sys), (t, a, c, d), a == b && a != c && a != d && c != d)
    rule(αβαα, Array{T,3},  zeros(T, n_itr, n_sys, n_sys), (t, a, b), a == c && c == d && a != b)
    rule(αβββ, Array{T,3},  zeros(T, n_itr, n_sys, n_sys), (t, a, b), b == c && c == d && a != b)
    rule(αβγγ, Array{T,4},  zeros(T, n_itr, n_sys, n_sys, n_sys), (t, a, b, c), c == d && a != b && a != c && b != c)
end

@patternized Patternized_g″ (n_sys::Int, n_itr::Int) (t::Int, a::Int, b::Int, c::Int, d::Int) begin
    rule(αββα, Array{T,3},  zeros(T, n_itr, n_sys, n_sys), (t, a, b), a == d && b == c && a != b)
    rule(αββγ, Array{T,4},  zeros(T, n_itr, n_sys, n_sys, n_sys), (t, a, b, d), b == c && a != b && a != d && b != d)
    rule(αβγδ, Array{T,5},  zeros(T, n_itr, n_sys, n_sys, n_sys, n_sys), (t, a, b, c, d), a != b && a != c && a != d && b != c && b != d && c != d)
end

mutable struct RmrtContext
    # input
    system              ::System
    environment         ::Environment
    simulation_details  ::SimulationDetails

    # computing context
    γ_exci              ::Array{ComplexF64, 3}    # 원래는 oscillator 안에 들어가는 coupling strength 의 exciton verison인데, 음.
    ϵ_exci              ::Vector{Float64}               # energy in exciton basis
    ϵ_exci_0            ::Vector{Float64}               # energy - reorganization energy
    U_sys               ::Matrix{ComplexF64}            # eigenvector matrix
    g                   ::Patternized_g{ComplexF64}
    g′                  ::Patternized_g′{ComplexF64}
    g″                  ::Patternized_g″{ComplexF64}

    # Reduced Density Matrix and its Time-derivatives
    σ                   ::Array{ComplexF64, 3}
    σ′                  ::Array{ComplexF64, 3}

    # output
    transition_rate     ::Array{Float64, 2}
    dissipation         ::Array{Dissipation, 1}


    function RmrtContext(system::System, environment::Environment, simulation_details::SimulationDetails)
        
        n_itr = simulation_details.num_of_iteration
        n_sys = system.n_sys
        n_osc = environment.num_of_effective_oscillators

        # Hamiltonian structures
        γ_exci  = zeros(ComplexF64, (n_osc, n_sys, n_sys))
        ϵ_exci  = zeros(Float64, n_sys)
        ϵ_exci_0= zeros(Float64, n_sys)
        U_sys   = zeros(ComplexF64, (n_sys, n_sys))

        # Line-broadening functions
        g       = Patternized_g{ComplexF64}(n_sys, n_itr)
        g′      = Patternized_g′{ComplexF64}(n_sys, n_itr)
        g″      = Patternized_g″{ComplexF64}(n_sys, n_itr)

        # Reduced density matrix and its time-derivatives
        σ       = zeros(ComplexF64, n_sys, n_sys, n_itr)
        σ′      = zeros(ComplexF64, n_sys, n_sys, n_itr)

        new(system, environment, simulation_details, γ_exci, ϵ_exci, ϵ_exci_0, U_sys, g, g′, g″, σ, σ′)
    end
end

create__rmrt_context(system::System, environment::Environment, simulation_details::SimulationDetails) = RmrtContext(system, environment, simulation_details)
create__rmrt_context(;system::System, environment::Environment, simulation_details::SimulationDetails) = RmrtContext(system, environment, simulation_details)



function calc__ϵ_exci!(context::RmrtContext)

    H_sys       = context.system.H_sys
    U_sys       = context.U_sys
    ϵ_exci      = context.ϵ_exci 
    #########################################

    eigen_result = eigen!(Hermitian(H_sys))
    ϵ_exci      .= eigen_result.values      # copy
    U_sys       .= eigen_result.vectors     # copy
end



function calc__g_g′_g″!(context::RmrtContext)
    n_sys   = context.system.n_sys
    n_osc   = context.environment.num_of_effective_oscillators
    oscs    = context.environment.effective_oscillators

    n_itr   = context.simulation_details.num_of_iteration
    Δt      = context.simulation_details.Δt

    g       = context.g
    g′      = context.g′
    g″      = context.g″

    γ_exci  = context.γ_exci

    @inbounds for osc_idx = 1:n_osc
    
        ω       = oscs[osc_idx].freq
        coth    = oscs[osc_idx].coth

        @inbounds for β in 1:n_sys, α in 1:n_sys

            γʲ_ααββ         = γ_exci[osc_idx,α,α] * γ_exci[osc_idx,β,β]
            ω²_γʲ_αβαβ      = γ_exci[osc_idx,α,β] * γ_exci[osc_idx,α,β] * (ω^2)
            ω_γʲ_αβββ       = γ_exci[osc_idx,α,β] * γ_exci[osc_idx,β,β] * ω
            ω_γʲ_ααββ       = γ_exci[osc_idx,α,α] * γ_exci[osc_idx,β,β] * ω

            @inbounds for time_idx = 1:n_itr
                t   = (time_idx - 1) * Δt
                ωt  = ω * t
                sin_ωt, cos_ωt = sincos(ωt)
                
                g[time_idx,α,α,β,β]         += (γʲ_ααββ      * ((coth * (1.0 - cos_ωt)) + 1.0im*(sin_ωt - ωt)))

                # 같을 때만 따로 처리... 중복되거든.
                if α == β
                    g′[time_idx,α,α,α,α]    += ω_γʲ_αβββ    * ((coth * sin_ωt) + 1.0im*(cos_ωt - 1.0))
                else
                    g′[time_idx,α,α,β,β]    += ω_γʲ_ααββ    * ((coth * sin_ωt) + 1.0im*(cos_ωt - 1.0))
                    g′[time_idx,α,β,β,β]    += ω_γʲ_αβββ    * ((coth * sin_ωt) + 1.0im*(cos_ωt - 1.0))
                    g′[time_idx,β,α,β,β]    += ω_γʲ_αβββ    * ((coth * sin_ωt) + 1.0im*(cos_ωt - 1.0)) 
                end
           
                g″[time_idx,α,β,β,α]    += ω²_γʲ_αβαβ   * ((coth * cos_ωt) - 1.0im*(sin_ωt))
            end
        end

        if (osc_idx-1) % 100 == 0
            @printf(stderr, "OSC %6d / %6d\n", osc_idx, n_osc)
        end
    end
end

function calc__g_g′_g″_with_threads!(context::RmrtContext)
    n_sys   = context.system.n_sys
    n_osc   = context.environment.num_of_effective_oscillators
    oscs    = context.environment.effective_oscillators

    n_itr   = context.simulation_details.num_of_iteration
    Δt      = context.simulation_details.Δt

    g       = context.g
    g′      = context.g′
    g″      = context.g″

    γ_exci  = context.γ_exci

    # thread 경쟁상태 방지 위한, local sum 변수 메모리 많이 잡아먹음. 주의.
    n_ths = Threads.maxthreadid()
    g_locals    = [Patternized_g{ComplexF64}(n_sys, n_itr) for _ in 1:n_ths]
    g′_locals   = [Patternized_g′{ComplexF64}(n_sys, n_itr) for _ in 1:n_ths]
    g″_locals   = [Patternized_g″{ComplexF64}(n_sys, n_itr) for _ in 1:n_ths]
    
    for tid in 1:n_ths
        fill!(g_locals[tid].ααββ, 0)
        
        fill!(g′_locals[tid].ααββ, 0)
        fill!(g′_locals[tid].αβαα, 0)
        fill!(g′_locals[tid].αβββ, 0)

        fill!(g″_locals[tid].αββα, 0)
    end

    @inbounds @threads for osc_idx = 1:n_osc
    
        ω       = oscs[osc_idx].freq
        coth    = oscs[osc_idx].coth

        # g_local 변수 만들기.
        tid = threadid()
        g_local     = g_locals[tid]
        g′_local    = g′_locals[tid]
        g″_local    = g″_locals[tid]

        @inbounds for β in 1:n_sys, α in 1:n_sys

            hr_ααββ         = γ_exci[osc_idx,α,α] * γ_exci[osc_idx,β,β]
            hr_αβαβ_ω²      = γ_exci[osc_idx,α,β] * γ_exci[osc_idx,α,β] * (ω^2)
            hr_αβββ_ω       = γ_exci[osc_idx,α,β] * γ_exci[osc_idx,β,β] * ω
            ω_γʲ_ααββ       = γ_exci[osc_idx,α,α] * γ_exci[osc_idx,β,β] * ω

            @inbounds for time_idx = 1:n_itr
                t   = (time_idx - 1) * Δt
                ωt  = ω * t

                sin_ωt, cos_ωt = sin(ωt), cos(ωt)
                
                g_local[time_idx,α,α,β,β]    += (hr_ααββ      * ((coth * (1.0 - cos_ωt)) + 1.0im*(sin_ωt - ωt)))

                if α == β
                    g′_local[time_idx,α,α,α,α]    += hr_αβββ_ω    * ((coth * sin_ωt) + 1.0im*(cos_ωt - 1.0))
                else
                    g′_local[time_idx,α,α,β,β]    += ω_γʲ_ααββ    * ((coth * sin_ωt) + 1.0im*(cos_ωt - 1.0))
                    g′_local[time_idx,α,β,β,β]    += hr_αβββ_ω    * ((coth * sin_ωt) + 1.0im*(cos_ωt - 1.0))
                    g′_local[time_idx,β,α,β,β]    += hr_αβββ_ω    * ((coth * sin_ωt) + 1.0im*(cos_ωt - 1.0)) 
                end
           
                g″_local[time_idx,α,β,β,α]    += hr_αβαβ_ω²   * ((coth * cos_ωt) - 1.0im*(sin_ωt))
            end
        end

        if (osc_idx-1) % 100 == 0
            @printf(stderr, "OSC %6d / %6d\n", osc_idx, n_osc)
        end
    end
    

    # reduction (single-thread)
    # fill!(g, 0)
    for tid in 1:n_ths
        inplace_add!(g, g_locals[tid])
        inplace_add!(g′, g′_locals[tid])
        inplace_add!(g″, g″_locals[tid])
    end
end

calc__exciton_energy!(context::RmrtContext)                             = calc__ϵ_exci!(context)
calc__line_broadening_functions!(context::RmrtContext)                  = calc__g_g′_g″!(context)
calc__line_broadening_functions_with_threads!(context::RmrtContext)     = calc__g_g′_g″_with_threads!(context)





@inline function _coherence_transition_frequency(context::MrtContext, a::Int, b::Int; use_shifted_energy::Bool=false)
    if use_shifted_energy
        return context.ϵ_exci_0[a] - context.ϵ_exci_0[b]
    else
        return context.ϵ_exci[a] - context.ϵ_exci[b]
    end
end

@inline function _coherence_g_generic(context::MrtContext, time_idx::Int, a::Int, b::Int, c::Int, d::Int)
    γ_exci  = context.γ_exci
    oscs    = context.environment.effective_oscillators
    n_osc   = context.environment.num_of_effective_oscillators
    Δt      = context.simulation_details.Δt

    t = (time_idx - 1) * Δt
    value = 0.0 + 0.0im

    @inbounds for osc_idx in 1:n_osc
        ω    = oscs[osc_idx].freq
        coth = oscs[osc_idx].coth
        γab  = γ_exci[osc_idx, a, b]
        γcd  = γ_exci[osc_idx, c, d]

        sin_ωt, cos_ωt = sincos(ω * t)
        value += γab * γcd * ((coth * (1.0 - cos_ωt)) + 1.0im * (sin_ωt - ω * t))
    end

    return value
end

@inline function _coherence_g′_generic(context::MrtContext, time_idx::Int, a::Int, b::Int, c::Int, d::Int)
    γ_exci  = context.γ_exci
    oscs    = context.environment.effective_oscillators
    n_osc   = context.environment.num_of_effective_oscillators
    Δt      = context.simulation_details.Δt

    t = (time_idx - 1) * Δt
    value = 0.0 + 0.0im

    @inbounds for osc_idx in 1:n_osc
        ω    = oscs[osc_idx].freq
        coth = oscs[osc_idx].coth
        γab  = γ_exci[osc_idx, a, b]
        γcd  = γ_exci[osc_idx, c, d]

        sin_ωt, cos_ωt = sincos(ω * t)
        value += (ω * γab * γcd) * ((coth * sin_ωt) + 1.0im * (cos_ωt - 1.0))
    end

    return value
end

@inline function _coherence_g″_generic(context::MrtContext, time_idx::Int, a::Int, b::Int, c::Int, d::Int)
    γ_exci  = context.γ_exci
    oscs    = context.environment.effective_oscillators
    n_osc   = context.environment.num_of_effective_oscillators
    Δt      = context.simulation_details.Δt

    t = (time_idx - 1) * Δt
    value = 0.0 + 0.0im

    @inbounds for osc_idx in 1:n_osc
        ω    = oscs[osc_idx].freq
        coth = oscs[osc_idx].coth
        γab  = γ_exci[osc_idx, a, b]
        γcd  = γ_exci[osc_idx, c, d]

        sin_ωt, cos_ωt = sincos(ω * t)
        value += ((ω^2) * γab * γcd) * ((coth * cos_ωt) - 1.0im * sin_ωt)
    end

    return value
end

"""
    calc__coherence_rhs!(dotσ, context, σ, time_idx; use_shifted_energy=false)

Compute the uploaded coherence equation for dot sigma_{\alpha\beta}(t) at a
single discrete time index `time_idx`.

Implementation notes:
- The six memory-kernel blocks in the uploaded formula are kept explicitly.
- Generic 4-index g, g′, g″ objects are evaluated on the fly from γ_exci and the
  effective oscillators, so this routine is not limited by the patternized caches.
- Only off-diagonal entries are filled. Diagonal entries are left at zero.
"""
function calc__coherence_rhs!(
    dotσ               ::AbstractMatrix{ComplexF64},
    context             ::RmrtContext,
    σ                   ::AbstractMatrix{<:Complex},
    time_idx            ::Int;
    use_shifted_energy  ::Bool=false,
)
    n_sys = context.system.n_sys
    n_itr = context.simulation_details.num_of_iteration
    Δt    = context.simulation_details.Δt

    size(dotσ, 1) == n_sys || error("dotσ row size does not match n_sys")
    size(dotσ, 2) == n_sys || error("dotσ column size does not match n_sys")
    size(σ, 1) == n_sys    || error("σ row size does not match n_sys")
    size(σ, 2) == n_sys    || error("σ column size does not match n_sys")
    1 <= time_idx <= n_itr || error("time_idx is out of range")

    fill!(dotσ, 0.0 + 0.0im)

    g   = (idx, a, b, c, d) -> _coherence_g_generic(context, idx, a, b, c, d)
    g′  = (idx, a, b, c, d) -> _coherence_g′_generic(context, idx, a, b, c, d)
    g″  = (idx, a, b, c, d) -> _coherence_g″_generic(context, idx, a, b, c, d)

    @inbounds for β in 1:n_sys, α in 1:n_sys
        if α == β
            dotσ[α, β] = 0.0 + 0.0im
            continue
        end

        ω_αβ = _coherence_transition_frequency(context, α, β; use_shifted_energy=use_shifted_energy)

        value = (
            -1.0im * ω_αβ
            - g′(time_idx, α, α, α, α)
            + conj(g′(time_idx, α, α, β, β))
            + g′(time_idx, β, β, α, α)
            - conj(g′(time_idx, β, β, β, β))
        ) * σ[α, β]

        for αbar in 1:n_sys
            αbar == α && continue
            value -= σ[αbar, β] * (
                g′(time_idx, α, αbar, αbar, αbar)
                - conj(g′(time_idx, αbar, α, β, β))
            )
        end

        for βbar in 1:n_sys
            βbar == β && continue
            value += σ[α, βbar] * (
                g′(time_idx, βbar, β, α, α)
                - conj(g′(time_idx, β, βbar, βbar, βbar))
            )
        end

        if time_idx > 1
            integral_sum = 0.0 + 0.0im

            for s_idx in 1:time_idx
                Δ_idx = time_idx - s_idx + 1
                Δ     = (Δ_idx - 1) * Δt
                weight = (s_idx == 1 || s_idx == time_idx) ? 0.5 : 1.0

                kernel_value = 0.0 + 0.0im

                # 1) - sum_{αbar ≠ α} σ_{αβ}(t) ...
                for αbar in 1:n_sys
                    αbar == α && continue

                    E1 = (
                        g(time_idx, α, α, α, α)
                        - g(s_idx, α, α, α, α)
                        - g(Δ_idx, αbar, αbar, αbar, αbar)
                        - (g(time_idx, αbar, αbar, α, α) - g(s_idx, αbar, αbar, α, α) - g(Δ_idx, αbar, αbar, α, α))
                        - g(Δ_idx, β, β, α, α)
                        + conj(g(s_idx, α, α, β, β) - g(time_idx, α, α, β, β))
                        + g(Δ_idx, β, β, αbar, αbar)
                        + conj(g(time_idx, αbar, αbar, β, β) - g(s_idx, αbar, αbar, β, β))
                    )

                    A1 = (
                        -g′(time_idx, α, αbar, α, α)
                        + g′(Δ_idx, α, αbar, α, α)
                        - g′(Δ_idx, α, αbar, αbar, αbar)
                        + g′(time_idx, α, αbar, αbar, αbar)
                    )

                    B1 = (
                        -g′(s_idx, αbar, α, α, α)
                        - g′(Δ_idx, αbar, αbar, αbar, α)
                        + conj(g′(s_idx, α, αbar, β, β))
                        + g′(Δ_idx, β, β, αbar, α)
                    )

                    kernel_value += -σ[α, β] * exp(-1.0im * _coherence_transition_frequency(context, αbar, α; use_shifted_energy=use_shifted_energy) * Δ) * exp(E1) * (g″(Δ_idx, α, αbar, αbar, α) - A1 * B1)
                end

                # 2) - sum_{βbar ≠ β} σ_{αβ}(t) ...
                for βbar in 1:n_sys
                    βbar == β && continue

                    E2 = (
                        conj(g(time_idx, β, β, β, β) - g(s_idx, β, β, β, β))
                        - conj(g(Δ_idx, βbar, βbar, βbar, βbar))
                        - conj(g(time_idx, βbar, βbar, β, β) - g(s_idx, βbar, βbar, β, β) - g(Δ_idx, βbar, βbar, β, β))
                        - g(Δ_idx, β, β, α, α)
                        + conj(g(s_idx, α, α, β, β) - g(time_idx, α, α, β, β))
                        + g(Δ_idx, βbar, βbar, α, α)
                        + conj(g(time_idx, α, α, βbar, βbar) - g(s_idx, α, α, βbar, βbar))
                    )

                    A2 = (
                        conj(g′(time_idx, β, βbar, β, β))
                        - conj(g′(Δ_idx, β, βbar, β, β))
                        + conj(g′(Δ_idx, β, βbar, βbar, βbar))
                        - conj(g′(time_idx, β, βbar, βbar, βbar))
                    )

                    B2 = (
                        -g′(s_idx, β, βbar, α, α)
                        - conj(g′(Δ_idx, α, α, βbar, β))
                        + conj(g′(s_idx, βbar, β, β, β))
                        + conj(g′(Δ_idx, βbar, βbar, βbar, β))
                    )

                    kernel_value += -σ[α, β] * exp(-1.0im * _coherence_transition_frequency(context, β, βbar; use_shifted_energy=use_shifted_energy) * Δ) * exp(E2) * (conj(g″(Δ_idx, β, βbar, βbar, β)) - A2 * B2)
                end

                # 3) - sum_{αbar ≠ α} sum_{αdbar ≠ αbar, α} σ_{αdbar β}(t) ...
                for αbar in 1:n_sys
                    αbar == α && continue
                    for αdbar in 1:n_sys
                        (αdbar == αbar || αdbar == α) && continue

                        E3 = (
                            g(time_idx, αdbar, αdbar, αdbar, αdbar)
                            - g(s_idx, αdbar, αdbar, αdbar, αdbar)
                            - g(Δ_idx, αbar, αbar, αbar, αbar)
                            - (g(time_idx, αbar, αbar, αdbar, αdbar) - g(s_idx, αbar, αbar, αdbar, αdbar) - g(Δ_idx, αbar, αbar, αdbar, αdbar))
                            - g(Δ_idx, β, β, αdbar, αdbar)
                            + conj(g(s_idx, αdbar, αdbar, β, β) - g(time_idx, αdbar, αdbar, β, β))
                            + g(Δ_idx, β, β, αbar, αbar)
                            + conj(g(time_idx, αbar, αbar, β, β) - g(s_idx, αbar, αbar, β, β))
                        )

                        A3 = (
                            -g′(time_idx, α, αbar, αdbar, αdbar)
                            + g′(Δ_idx, α, αbar, αdbar, αdbar)
                            - g′(Δ_idx, α, αbar, αbar, αbar)
                            + g′(time_idx, α, αbar, αbar, αbar)
                        )

                        B3 = (
                            -g′(s_idx, αbar, αdbar, αdbar, αdbar)
                            - g′(Δ_idx, αbar, αbar, αbar, αdbar)
                            + conj(g′(s_idx, αdbar, αbar, β, β))
                            + g′(Δ_idx, β, β, αbar, αdbar)
                        )

                        kernel_value += -σ[αdbar, β] * exp(-1.0im * _coherence_transition_frequency(context, αbar, αdbar; use_shifted_energy=use_shifted_energy) * Δ)  * exp(E3) * (g″(Δ_idx, α, αbar, αbar, αdbar) - A3 * B3)
                    end
                end

                # 4) - sum_{βbar ≠ β} sum_{βdbar ≠ βbar, β} σ_{α βdbar}(t) ...
                for βbar in 1:n_sys
                    βbar == β && continue
                    for βdbar in 1:n_sys
                        (βdbar == βbar || βdbar == β) && continue

                        E4 = (
                            conj(g(time_idx, βdbar, βdbar, βdbar, βdbar) - g(s_idx, βdbar, βdbar, βdbar, βdbar))
                            - conj(g(Δ_idx, βbar, βbar, βbar, βbar))
                            - conj(g(time_idx, βbar, βbar, βdbar, βdbar) - g(s_idx, βbar, βbar, βdbar, βdbar) - g(Δ_idx, βbar, βbar, βdbar, βdbar))
                            - g(Δ_idx, βdbar, βdbar, α, α)
                            + conj(g(s_idx, α, α, βdbar, βdbar) - g(time_idx, α, α, βdbar, βdbar))
                            + g(Δ_idx, βbar, βbar, α, α)
                            + conj(g(time_idx, α, α, βbar, βbar) - g(s_idx, α, α, βbar, βbar))
                        )

                        A4 = (
                            conj(g′(time_idx, β, βbar, βdbar, βdbar))
                            - conj(g′(Δ_idx, β, βbar, βdbar, βdbar))
                            + conj(g′(Δ_idx, β, βbar, βbar, βbar))
                            - conj(g′(time_idx, β, βbar, βbar, βbar))
                        )

                        B4 = (
                            -g′(s_idx, βdbar, βbar, α, α)
                            - conj(g′(Δ_idx, α, α, βbar, βdbar))
                            + conj(g′(s_idx, βbar, βdbar, βdbar, βdbar))
                            + conj(g′(Δ_idx, βbar, βbar, βbar, βdbar))
                        )

                        kernel_value += -σ[α, βdbar] * exp(-1.0im * _coherence_transition_frequency(context, βdbar, βbar; use_shifted_energy=use_shifted_energy) * Δ) * exp(E4) * (conj(g″(Δ_idx, β, βbar, βbar, βdbar)) - A4 * B4)
                    end
                end

                # 5) + sum_{αbar ≠ α} sum_{βbar ≠ β} σ_{αbar βbar}(t) e^{-iω_{α αbar}Δ} ...
                for αbar in 1:n_sys
                    αbar == α && continue
                    for βbar in 1:n_sys
                        βbar == β && continue

                        E5 = (
                            g(time_idx, αbar, αbar, αbar, αbar)
                            - g(s_idx, αbar, αbar, αbar, αbar)
                            - g(Δ_idx, α, α, α, α)
                            - (g(time_idx, α, α, αbar, αbar) - g(s_idx, α, α, αbar, αbar) - g(Δ_idx, α, α, αbar, αbar))
                            - g(Δ_idx, βbar, βbar, αbar, αbar)
                            + conj(g(s_idx, αbar, αbar, βbar, βbar) - g(time_idx, αbar, αbar, βbar, βbar))
                            + g(Δ_idx, βbar, βbar, α, α)
                            + conj(g(time_idx, α, α, βbar, βbar) - g(s_idx, α, α, βbar, βbar))
                        )

                        A5 = (
                            -g′(s_idx, α, αbar, αbar, αbar)
                            - g′(Δ_idx, α, α, α, αbar)
                            + conj(g′(s_idx, αbar, α, βbar, βbar))
                            + g′(Δ_idx, βbar, βbar, α, αbar)
                        )

                        B5 = (
                            g′(time_idx, βbar, β, α, α)
                            - g′(Δ_idx, βbar, β, α, α)
                            - g′(time_idx, βbar, β, αbar, αbar)
                            + g′(Δ_idx, βbar, β, αbar, αbar)
                        )

                        kernel_value += σ[αbar, βbar] * exp(-1.0im * _coherence_transition_frequency(context, α, αbar; use_shifted_energy=use_shifted_energy) * Δ) * exp(E5) * (g″(Δ_idx, βbar, β, α, αbar) - A5 * B5)
                    end
                end

                # 6) + sum_{αbar ≠ α} sum_{βbar ≠ β} σ_{αbar βbar}(t) e^{-iω_{βbar β}Δ} ...
                for αbar in 1:n_sys
                    αbar == α && continue
                    for βbar in 1:n_sys
                        βbar == β && continue

                        E6 = (
                            conj(g(time_idx, βbar, βbar, βbar, βbar) - g(s_idx, βbar, βbar, βbar, βbar))
                            - conj(g(Δ_idx, β, β, β, β))
                            - conj(g(time_idx, β, β, βbar, βbar) - g(s_idx, β, β, βbar, βbar) - g(Δ_idx, β, β, βbar, βbar))
                            - g(Δ_idx, βbar, βbar, αbar, αbar)
                            + conj(g(s_idx, αbar, αbar, βbar, βbar) - g(time_idx, αbar, αbar, βbar, βbar))
                            + g(Δ_idx, β, β, αbar, αbar)
                            + conj(g(time_idx, αbar, αbar, β, β) - g(s_idx, αbar, αbar, β, β))
                        )

                        A6 = (
                            conj(g′(time_idx, αbar, α, βbar, βbar))
                            - conj(g′(Δ_idx, αbar, α, βbar, βbar))
                            - conj(g′(time_idx, αbar, α, β, β))
                            + conj(g′(Δ_idx, αbar, α, β, β))
                        )

                        B6 = (
                            -g′(s_idx, βbar, β, αbar, αbar)
                            - conj(g′(Δ_idx, αbar, αbar, β, βbar))
                            + conj(g′(s_idx, β, βbar, βbar, βbar))
                            + conj(g′(Δ_idx, β, β, β, βbar))
                        )

                        kernel_value += σ[αbar, βbar] * exp(-1.0im * _coherence_transition_frequency(context, βbar, β; use_shifted_energy=use_shifted_energy) * Δ) * exp(E6) * (conj(g″(Δ_idx, αbar, α, β, βbar)) - A6 * B6)
                    end
                end

                integral_sum += weight * kernel_value
            end

            value += Δt * integral_sum
        end

        dotσ[α, β] = value
    end

    return dotσ
end

function calc__coherence_rhs(
    context             ::MrtContext,
    σ                   ::AbstractMatrix{<:Complex},
    time_idx            ::Int;
    use_shifted_energy  ::Bool=false,
)
    dotσ = zeros(ComplexF64, size(σ, 1), size(σ, 2))
    calc__coherence_rhs!(dotσ, context, σ, time_idx; use_shifted_energy=use_shifted_energy)
    return dotσ
end

function calc__coherence_rhs_history!(
    dotσ_history        ::Array{ComplexF64,3},
    context             ::MrtContext,
    σ_history           ::Array{ComplexF64,3};
    use_shifted_energy  ::Bool=false,
)
    n_sys = context.system.n_sys
    n_itr = context.simulation_details.num_of_iteration

    size(dotσ_history, 1) == n_sys || error("dotσ_history first dimension does not match n_sys")
    size(dotσ_history, 2) == n_sys || error("dotσ_history second dimension does not match n_sys")
    size(dotσ_history, 3) == n_itr || error("dotσ_history third dimension does not match num_of_iteration")
    size(σ_history, 1) == n_sys    || error("σ_history first dimension does not match n_sys")
    size(σ_history, 2) == n_sys    || error("σ_history second dimension does not match n_sys")
    size(σ_history, 3) == n_itr    || error("σ_history third dimension does not match num_of_iteration")

    for time_idx in 1:n_itr
        calc__coherence_rhs!(
            @view(dotσ_history[:, :, time_idx]),
            context,
            @view(σ_history[:, :, time_idx]),
            time_idx;
            use_shifted_energy=use_shifted_energy,
        )
    end

    return dotσ_history
end

export MrtContext, create__mrt_context, calc__Λ!, calc__Γ!, calc__g_g′_and_g″!, calc__g_g′_and_g″_with_threads!, calc__rates!, calc__dissipations!, calc__dissipations_with_threads!, check__physics, calc__coherence_rhs!, calc__coherence_rhs, calc__coherence_rhs_history!
end 

