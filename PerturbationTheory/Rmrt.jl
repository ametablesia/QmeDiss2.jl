
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
    curr_itr            ::UInt64
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

        new(system, environment, simulation_details, γ_exci, ϵ_exci, ϵ_exci_0, U_sys, g, g′, g″, 1, σ, σ′)
    end
end

create__rmrt_context(system::System, environment::Environment, simulation_details::SimulationDetails) = RmrtContext(system, environment, simulation_details)
create__rmrt_context(;system::System, environment::Environment, simulation_details::SimulationDetails) = RmrtContext(system, environment, simulation_details)


# Exciton basis 에서 에너지를 읽자.
function calc__ϵ_exci!(context::RmrtContext)

    H_sys       = context.system.H_sys
    U_sys       = context.U_sys
    ϵ_exci      = context.ϵ_exci 
    #########################################

    eigen_result = eigen!(Hermitian(H_sys))
    ϵ_exci      .= eigen_result.values      # copy
    U_sys       .= eigen_result.vectors     # copy
end

function calc__γ_exci!(
    context::RmrtContext;
    update_energy::Bool = false,
    update_shifted_energy::Bool = true,
)
    if update_energy
        calc__ϵ_exci!(context)
    end

    n_sys       = context.system.n_sys
    n_osc       = context.environment.num_of_effective_oscillators
    oscs        = context.environment.effective_oscillators

    U_sys       = context.U_sys
    ϵ_exci      = context.ϵ_exci
    ϵ_exci_0    = context.ϵ_exci_0
    γ_exci      = context.γ_exci

    fill!(γ_exci, 0.0 + 0.0im)

    # λ_diag[α] corresponds to Λ_{αααα}
    λ_diag = zeros(ComplexF64, n_sys)

    @inbounds for osc_idx in 1:n_osc
        ω      = oscs[osc_idx].freq
        γ_site = oscs[osc_idx].site_bath_coupling_strength

        # site basis -> exciton basis
        γ_exci[osc_idx, :, :] .= U_sys' * γ_site * U_sys

        if update_shifted_energy
            for α in 1:n_sys
                λ_diag[α] += ω * γ_exci[osc_idx, α, α] * γ_exci[osc_idx, α, α]
            end
        end
    end

    if update_shifted_energy
        @inbounds for α in 1:n_sys
            ϵ_exci_0[α] = ϵ_exci[α] - real(λ_diag[α])
        end
    end

    return γ_exci
end

function calc__exciton_basis_and_γ_exci!(context::RmrtContext)
    calc__ϵ_exci!(context)
    calc__γ_exci!(context; update_energy = false, update_shifted_energy = true)
    return context
end


@inline function zero__g_g′_g″!(
    g  ::Patternized_g{ComplexF64},
    g′ ::Patternized_g′{ComplexF64},
    g″ ::Patternized_g″{ComplexF64},
)
    # g canonical patterns
    fill!(g.αααα, 0.0 + 0.0im)
    fill!(g.ααββ, 0.0 + 0.0im)

    # g′ canonical patterns
    fill!(g′.αααα, 0.0 + 0.0im)
    fill!(g′.αααβ, 0.0 + 0.0im)
    fill!(g′.ααββ, 0.0 + 0.0im)
    fill!(g′.ααβγ, 0.0 + 0.0im)
    fill!(g′.αβαα, 0.0 + 0.0im)
    fill!(g′.αβββ, 0.0 + 0.0im)
    fill!(g′.αβγγ, 0.0 + 0.0im)

    # g″ canonical patterns
    fill!(g″.αββα, 0.0 + 0.0im)
    fill!(g″.αββγ, 0.0 + 0.0im)
    fill!(g″.αβγδ, 0.0 + 0.0im)

    return nothing
end

@inline function accumulate__g_g′_g″__one_oscillator!(
    g       ::Patternized_g{ComplexF64},
    g′      ::Patternized_g′{ComplexF64},
    g″      ::Patternized_g″{ComplexF64},
    γ_exci  ::Array{ComplexF64, 3},
    osc_idx ::Int,
    ω       ::Float64,
    coth    ::Float64,
    n_sys   ::Int,
    n_itr   ::Int,
    Δt      ::Float64,
)
    γ = @view γ_exci[osc_idx, :, :]

    @inbounds for time_idx in 1:n_itr
        t   = (time_idx - 1) * Δt
        ωt  = ω * t
        sin_ωt, cos_ωt = sincos(ωt)

        # Common scalar factors.
        # g_{abcd}(t)   = γ_ab γ_cd * F_g(t)
        # g′_{abcd}(t)  = γ_ab γ_cd * F_g′(t)
        # g″_{abcd}(t)  = γ_ab γ_cd * F_g″(t)
        F_g  = (coth * (1.0 - cos_ωt)) + 1.0im * (sin_ωt - ωt)
        F_g′ = ω * ((coth * sin_ωt) + 1.0im * (cos_ωt - 1.0))
        F_g″ = (ω^2) * ((coth * cos_ωt) - 1.0im * sin_ωt)

        # ---------------------------------------------------------------------
        # g canonical patterns
        #   αααα, ααββ
        # ---------------------------------------------------------------------
        for β in 1:n_sys, α in 1:n_sys
            g[time_idx, α, α, β, β] += γ[α, α] * γ[β, β] * F_g
        end

        # ---------------------------------------------------------------------
        # g′ canonical patterns
        #   αααα, αααβ, ααββ, ααβγ, αβαα, αβββ, αβγγ
        # ---------------------------------------------------------------------
        for α in 1:n_sys
            γ_αα = γ[α, α]

            # αααα
            g′[time_idx, α, α, α, α] += γ_αα * γ_αα * F_g′

            for α⁻ in 1:n_sys
                α⁻ == α && continue

                γ_αα⁻ = γ[α, α⁻]
                γ_α⁻α⁻ = γ[α⁻, α⁻]

                # αααβ : g′_{αααα⁻}
                g′[time_idx, α, α, α, α⁻] += γ_αα * γ_αα⁻ * F_g′

                # ααββ : g′_{ααα⁻α⁻}
                g′[time_idx, α, α, α⁻, α⁻] += γ_αα * γ_α⁻α⁻ * F_g′

                # αβαα : g′_{αα⁻αα}
                g′[time_idx, α, α⁻, α, α] += γ_αα⁻ * γ_αα * F_g′

                # αβββ : g′_{αα⁻α⁻α⁻}
                g′[time_idx, α, α⁻, α⁻, α⁻] += γ_αα⁻ * γ_α⁻α⁻ * F_g′

                for α⁼ in 1:n_sys
                    (α⁼ == α || α⁼ == α⁻) && continue

                    # ααβγ : g′_{αα α⁻ α⁼}
                    g′[time_idx, α, α, α⁻, α⁼] += γ_αα * γ[α⁻, α⁼] * F_g′

                    # αβγγ : g′_{α α⁻ α⁼ α⁼}
                    g′[time_idx, α, α⁻, α⁼, α⁼] += γ_αα⁻ * γ[α⁼, α⁼] * F_g′
                end
            end
        end

        # ---------------------------------------------------------------------
        # g″ canonical patterns
        #   αββα, αββγ, αβγδ
        # ---------------------------------------------------------------------
        for α in 1:n_sys
            for α⁻ in 1:n_sys
                α⁻ == α && continue

                γ_αα⁻ = γ[α, α⁻]

                # αββα : g″_{α α⁻ α⁻ α}
                # Use γ_{αα⁻} γ_{α⁻α}, not γ_{αα⁻}^2, so the formula remains
                # valid even when γ is complex/Hermitian rather than real-symmetric.
                g″[time_idx, α, α⁻, α⁻, α] += γ_αα⁻ * γ[α⁻, α] * F_g″

                for α⁼ in 1:n_sys
                    (α⁼ == α || α⁼ == α⁻) && continue

                    # αββγ : g″_{α α⁻ α⁻ α⁼}
                    g″[time_idx, α, α⁻, α⁻, α⁼] += γ_αα⁻ * γ[α⁻, α⁼] * F_g″

                    for α⁺ in 1:n_sys
                        (α⁺ == α || α⁺ == α⁻ || α⁺ == α⁼) && continue

                        # αβγδ : g″_{α α⁻ α⁼ α⁺}
                        g″[time_idx, α, α⁻, α⁼, α⁺] += γ_αα⁻ * γ[α⁼, α⁺] * F_g″
                    end
                end
            end
        end
    end

    return nothing
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

    zero__g_g′_g″!(g, g′, g″)

    @inbounds for osc_idx in 1:n_osc
        ω       = oscs[osc_idx].freq
        coth    = oscs[osc_idx].coth

        accumulate__g_g′_g″__one_oscillator!(
            g, g′, g″,
            γ_exci,
            osc_idx,
            ω,
            coth,
            n_sys,
            n_itr,
            Δt,
        )

        if (osc_idx - 1) % 100 == 0
            @printf(stderr, "OSC %6d / %6d\n", osc_idx, n_osc)
        end
    end

    return g, g′, g″
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

    # Thread 경쟁상태 방지용 local containers.
    # 모든 canonical field를 채우므로 메모리 사용량이 커진다.
    n_ths = Threads.maxthreadid()
    g_locals    = [Patternized_g{ComplexF64}(n_sys, n_itr)  for _ in 1:n_ths]
    g′_locals   = [Patternized_g′{ComplexF64}(n_sys, n_itr) for _ in 1:n_ths]
    g″_locals   = [Patternized_g″{ComplexF64}(n_sys, n_itr) for _ in 1:n_ths]

    zero__g_g′_g″!(g, g′, g″)

    for tid in 1:n_ths
        zero__g_g′_g″!(g_locals[tid], g′_locals[tid], g″_locals[tid])
    end

    @inbounds @threads for osc_idx in 1:n_osc
        ω       = oscs[osc_idx].freq
        coth    = oscs[osc_idx].coth

        tid = threadid()

        accumulate__g_g′_g″__one_oscillator!(
            g_locals[tid],
            g′_locals[tid],
            g″_locals[tid],
            γ_exci,
            osc_idx,
            ω,
            coth,
            n_sys,
            n_itr,
            Δt,
        )

        if (osc_idx - 1) % 100 == 0
            @printf(stderr, "OSC %6d / %6d\n", osc_idx, n_osc)
        end
    end

    # reduction: single-thread
    for tid in 1:n_ths
        inplace_add!(g,  g_locals[tid])
        inplace_add!(g′, g′_locals[tid])
        inplace_add!(g″, g″_locals[tid])
    end

    return g, g′, g″
end


function calc__σ_σ′!(context::RmrtContext)
    start_itr = Int(context.curr_itr)

    n_sys    = context.system.n_sys
    n_itr    = context.simulation_details.num_of_iteration
    Δt       = context.simulation_details.Δt
    ϵ        = context.ϵ_exci

    σ        = context.σ
    σ′       = context.σ′
    g        = context.g
    g′       = context.g′
    g″       = context.g″

    start_itr < n_itr || return @view σ[:, :, n_itr]

    @inline ω(a::Int, b::Int) = ϵ[a] - ϵ[b]

    @inline function ∫weight(s_itr::Int, curr_itr::Int)
        # trapezoidal rule on [0,t]
        return (s_itr == 1 || s_itr == curr_itr) ? 0.5 * Δt : Δt
    end

    @inline function phase(a::Int, b::Int, Δ_itr::Int)
        Δ = (Δ_itr - 1) * Δt
        return exp(-1.0im * ω(a, b) * Δ)
    end

    @inline function gen__exponent_type_1(s_itr::Int, Δ_itr::Int, t_itr::Int,
                         α⁻::Int, α⁼::Int, β::Int)
        return (
            -g[s_itr, α⁼, α⁼, α⁼, α⁼]
            +g[s_itr, α⁻, α⁻, α⁼, α⁼]
            +conj(g[s_itr, α⁼, α⁼, β, β])
            -conj(g[s_itr, α⁻, α⁻, β, β])

            -g[Δ_itr, α⁻, α⁻, α⁻, α⁻]
            +g[Δ_itr, α⁻, α⁻, α⁼, α⁼]
            -g[Δ_itr, β, β, α⁼, α⁼]
            +g[Δ_itr, β, β, α⁻, α⁻]

            +g[t_itr, α⁼, α⁼, α⁼, α⁼]
            -g[t_itr, α⁻, α⁻, α⁼, α⁼]
            -conj(g[t_itr, α⁼, α⁼, β, β])
            +conj(g[t_itr, α⁻, α⁻, β, β])
        )
    end

    @inline function gen__exponent_type_2(s_itr::Int, Δ_itr::Int, t_itr::Int,
                         a::Int, β⁼::Int, β⁻::Int)
        return (
            -conj(g[s_itr, β⁼, β⁼, β⁼, β⁼])
            +conj(g[s_itr, β⁻, β⁻, β⁼, β⁼])
            +g[s_itr, β⁼, β⁼, a, a]
            -g[s_itr, β⁻, β⁻, a, a]

            -conj(g[Δ_itr, β⁻, β⁻, β⁻, β⁻])
            +conj(g[Δ_itr, β⁻, β⁻, β⁼, β⁼])
            -conj(g[Δ_itr, a, a, β⁼, β⁼])
            +conj(g[Δ_itr, a, a, β⁻, β⁻])

            +conj(g[t_itr, β⁼, β⁼, β⁼, β⁼])
            -conj(g[t_itr, β⁻, β⁻, β⁼, β⁼])
            -g[t_itr, β⁼, β⁼, a, a]
            +g[t_itr, β⁻, β⁻, a, a]
        )
    end

    @inline function gen_coef_block_type_1(
        s_itr::Int, Δ_itr::Int, t_itr::Int,
        α::Int, α⁻::Int, α⁼::Int, β::Int)

        left_one_point = (
             g′[Δ_itr, α, α⁻, α⁼, α⁼]
            -g′[Δ_itr, α, α⁻, α⁻, α⁻]
            -g′[t_itr, α, α⁻, α⁼, α⁼]
            +g′[t_itr, α, α⁻, α⁻, α⁻]
        )

        right_one_point = (
            -g′[s_itr, α⁻, α⁼, α⁼, α⁼]
            +conj(g′[s_itr, α⁼, α⁻, β, β])
            -g′[Δ_itr, α⁻, α⁻, α⁻, α⁼]
            +g′[Δ_itr, β, β, α⁻, α⁼]
        )

        return g″[Δ_itr, α, α⁻, α⁻, α⁼] - left_one_point * right_one_point
    end

    @inline function gen_coef_block_type_2(
        s_itr::Int, Δ_itr::Int, t_itr::Int,
        β::Int, β⁻::Int, β⁼::Int, α::Int)

        left_one_point = (
            -conj(g′[Δ_itr, β, β⁻, β⁼, β⁼])
            +conj(g′[Δ_itr, β, β⁻, β⁻, β⁻])
            +conj(g′[t_itr, β, β⁻, β⁼, β⁼])
            -conj(g′[t_itr, β, β⁻, β⁻, β⁻])
        )

        right_one_point = (
            -g′[s_itr, β⁼, β⁻, α, α]
            +conj(g′[s_itr, β⁻, β⁼, β⁼, β⁼])
            -conj(g′[Δ_itr, α, α, β⁻, β⁼])
            +conj(g′[Δ_itr, β⁻, β⁻, β⁻, β⁼])
        )

        return conj(g″[Δ_itr, β, β⁻, β⁻, β⁼]) - left_one_point * right_one_point
    end

    @inline function gen_coef_block_type_3(
        s_itr::Int, Δ_itr::Int, t_itr::Int,
        α::Int, β::Int, α⁻::Int, β⁻::Int)

        left_one_point = (
            -g′[s_itr, α, α⁻, α⁻, α⁻]
            +conj(g′[s_itr, α⁻, α, β⁻, β⁻])
            -g′[Δ_itr, α, α, α, α⁻]
            +g′[Δ_itr, β⁻, β⁻, α, α⁻]
        )

        right_one_point = (
            -g′[Δ_itr, β⁻, β, α, α]
            +g′[Δ_itr, β⁻, β, α⁻, α⁻]
            +g′[t_itr, β⁻, β, α, α]
            -g′[t_itr, β⁻, β, α⁻, α⁻]
        )

        return g″[Δ_itr, β⁻, β, α, α⁻] - left_one_point * right_one_point
    end

    @inline function gen_coef_block_type_4(
        s_itr::Int, Δ_itr::Int, t_itr::Int,
        α::Int, β::Int, α⁻::Int, β⁻::Int)

        left_one_point = (
            -conj(g′[Δ_itr, α⁻, α, β⁻, β⁻])
            +conj(g′[Δ_itr, α⁻, α, β, β])
            +conj(g′[t_itr, α⁻, α, β⁻, β⁻])
            -conj(g′[t_itr, α⁻, α, β, β])
        )

        right_one_point = (
            -g′[s_itr, β⁻, β, α⁻, α⁻]
            +conj(g′[s_itr, β, β⁻, β⁻, β⁻])
            -conj(g′[Δ_itr, α⁻, α⁻, β, β⁻])
            +conj(g′[Δ_itr, β, β, β, β⁻])
        )

        return conj(g″[Δ_itr, α⁻, α, β, β⁻]) - left_one_point * right_one_point
    end

    # -------------------------------------------------------------------------
    # Time propagation loop
    # -------------------------------------------------------------------------
    @inbounds for curr_itr in start_itr:(n_itr - 1)
        σ_t  = @view σ[:, :, curr_itr]
        σ′_t = @view σ′[:, :, curr_itr]

        fill!(σ′_t, 0.0 + 0.0im)

        # -------------------------------------------------------------------------
        # Main loop
        # -------------------------------------------------------------------------
        @inbounds for β in 1:n_sys, α in 1:n_sys
            # The present equation is the off-diagonal coherence equation.
            # Population dynamics should be handled by a separate population closure.
            # if α == β
            #     σ′_t[α, β] = 0.0 + 0.0im
            #     continue
            # end

            # t-local diagonal/coherence phase block
            rhs = (
                -1.0im * ω(α, β)
                -g′[curr_itr, α, α, α, α] +conj(g′[curr_itr, α, α, β, β])
                +g′[curr_itr, β, β, α, α] -conj(g′[curr_itr, β, β, β, β])
            ) * σ_t[α, β]

            # t-local left mixing
            for α⁻ in 1:n_sys
                α⁻ == α && continue

                rhs -= σ_t[α⁻, β] * (g′[curr_itr, α, α⁻, α⁻, α⁻] -conj(g′[curr_itr, α⁻, α, β, β]))
            end

            # t-local right mixing
            for β⁻ in 1:n_sys
                β⁻ == β && continue
                rhs += σ_t[α, β⁻] * (g′[curr_itr, β⁻, β, α, α] -conj(g′[curr_itr, β, β⁻, β⁻, β⁻]))
            end

            # memory integral
            if curr_itr > 1
                integral = 0.0 + 0.0im

                for s_itr in 1:curr_itr
                    Δ_itr = curr_itr - s_itr + 1
                    kernel = 0.0 + 0.0im

                    # -------------------------------------------------------------
                    #   -Σ_{α⁻≠α} Σ_{α⁼≠α⁻} σ_{α⁼β} ...
                    # -------------------------------------------------------------
                    for α⁻ in 1:n_sys
                        α⁻ == α && continue
                        for α⁼ in 1:n_sys
                            α⁼ == α⁻ && continue

                            kernel -= (
                                σ_t[α⁼, β]
                                * phase(α⁻, α⁼, Δ_itr)
                                * exp(gen__exponent_type_1(s_itr, Δ_itr, curr_itr, α⁻, α⁼, β))
                                * gen_coef_block_type_1(s_itr, Δ_itr, curr_itr, α, α⁻, α⁼, β)
                            )
                        end
                    end

                    # -------------------------------------------------------------
                    #   -Σ_{β⁻≠β} Σ_{β⁼≠β⁻} σ_{αβ⁼} ...
                    # -------------------------------------------------------------
                    for β⁻ in 1:n_sys
                        β⁻ == β && continue
                        for β⁼ in 1:n_sys
                            β⁼ == β⁻ && continue

                            kernel -= (
                                σ_t[α, β⁼]
                                * phase(β⁼, β⁻, Δ_itr)
                                * exp(gen__exponent_type_2(s_itr, Δ_itr, curr_itr, α, β⁼, β⁻))
                                * gen_coef_block_type_2(s_itr, Δ_itr, curr_itr, β, β⁻, β⁼, α)
                            )
                        end
                    end

                    # -------------------------------------------------------------
                    #   +Σ_{α⁻≠α} Σ_{β⁻≠β} σ_{α⁻β⁻} ...
                    # -------------------------------------------------------------
                    for α⁻ in 1:n_sys
                        α⁻ == α && continue
                        for β⁻ in 1:n_sys
                            β⁻ == β && continue

                            kernel += (
                                σ_t[α⁻, β⁻]
                                * phase(α, α⁻, Δ_itr)
                                * exp(gen__exponent_type_1(s_itr, Δ_itr, curr_itr, α, α⁻, β⁻))
                                * gen_coef_block_type_3(s_itr, Δ_itr, curr_itr, α, β, α⁻, β⁻)
                            )
                        end
                    end

                    # -------------------------------------------------------------
                    #   +Σ_{α⁻≠α} Σ_{β⁻≠β} σ_{α⁻β⁻} ...
                    # -------------------------------------------------------------
                    for α⁻ in 1:n_sys
                        α⁻ == α && continue
                        for β⁻ in 1:n_sys
                            β⁻ == β && continue

                            kernel += (
                                σ_t[α⁻, β⁻]
                                * phase(β⁻, β, Δ_itr)
                                * exp(gen__exponent_type_2(s_itr, Δ_itr, curr_itr, α⁻, β⁻, β))
                                * gen_coef_block_type_4(s_itr, Δ_itr, curr_itr, α, β, α⁻, β⁻)
                            )
                        end
                    end

                    integral += ∫weight(s_itr, curr_itr) * kernel
                end

                rhs += integral
            end

            σ′_t[α, β] = rhs
        end


        @views σ[:, :, curr_itr + 1] .= σ[:, :, curr_itr] .+ Δt .* σ′[:, :, curr_itr]
        context.curr_itr = UInt64(curr_itr + 1)
    end

    return @view σ[:, :, Int(context.curr_itr)]
end

function set__initial_σ!(
    context::RmrtContext,
    σ_initial::AbstractMatrix{<:Complex}
)
    n_sys = context.system.n_sys

    size(σ_initial, 1) == n_sys || error("σ_initial row size does not match n_sys")
    size(σ_initial, 2) == n_sys || error("σ_initial column size does not match n_sys")

    context.curr_itr = UInt64(1)

    σ  = context.σ
    σ′ = context.σ′

    @views σ[:, :, 1]  .= σ_initial
    @views σ′[:, :, 1] .= 0.0 + 0.0im

    return @view σ[:, :, 1]
end

# 특정 population을 1로 만들어 initial state로 만듦.
function set__initial_σ!(
    context::RmrtContext;
    init_state::Integer = 1
)
    n_sys = context.system.n_sys

    1 <= init_state <= n_sys || error("init_state is out of range")

    context.curr_itr = UInt64(1)

    σ  = context.σ
    σ′ = context.σ′

    @views σ[:, :, 1]  .= 0.0 + 0.0im
    @views σ′[:, :, 1] .= 0.0 + 0.0im

    σ[init_state, init_state, 1] = 1.0 + 0.0im

    return @view σ[:, :, 1]
end



function save__rmrt_reduced_dynamics!(
    context::RmrtContext;
    save_filename::AbstractString = "rmrt.txt",
    basis::Symbol = :both,
)
    n_sys = context.system.n_sys
    n_itr_total = context.simulation_details.num_of_iteration
    Δt = context.simulation_details.Δt

    # If the propagation has not reached the final time, save only computed slices.
    curr_itr = Int(context.curr_itr)
    n_save = clamp(curr_itr, 1, n_itr_total)

    σ_hist = context.σ
    σ′_hist = context.σ′
    U_sys = context.U_sys

    basis in (:exciton, :site, :both) || error("basis must be one of :exciton, :site, or :both")

    open(save_filename, "w") do file_id
        @printf(file_id, "\n---- RMRT reduced dynamics ----\n")
        @printf(file_id, "n_sys=%d  n_saved=%d  Δt=% .8e  basis=%s\n", n_sys, n_save, Δt, String(basis))

        if basis == :exciton || basis == :both
            @printf(file_id, "\n---- Exciton-basis reduced dynamics ----\n")
            _write__rmrt_basis_dynamics!(file_id, σ_hist, σ′_hist, n_sys, n_save, Δt, :exciton)
        end

        if basis == :site || basis == :both
            @printf(file_id, "\n---- Site-basis reduced dynamics ----\n")
            _write__rmrt_site_basis_dynamics!(file_id, σ_hist, σ′_hist, U_sys, n_sys, n_save, Δt)
        end
    end

    @printf(stderr, "RMRT reduced dynamics saved to %s\n", save_filename)
    return save_filename
end

@inline function _write__rmrt_basis_dynamics!(
    file_id::IO,
    σ_hist::Array{ComplexF64,3},
    σ′_hist::Array{ComplexF64,3},
    n_sys::Int,
    n_save::Int,
    Δt::Real,
    basis_name::Symbol,
)
    @inbounds for ti in 1:n_save
        t = (ti - 1) * Δt
        σ_t = @view σ_hist[:, :, ti]
        σ′_t = @view σ′_hist[:, :, ti]

        trσ = tr(σ_t)
        trσ′ = tr(σ′_t)

        @printf(file_id, "\nt=%12.6f  trace=(% .8e,% .8e)  trace_prime=(% .8e,% .8e)\n",
            t, real(trσ), imag(trσ), real(trσ′), imag(trσ′))

        @printf(file_id, "  populations[%s]:", String(basis_name))
        for α in 1:n_sys
            @printf(file_id, "  p%d=% .8e", α, real(σ_t[α, α]))
        end
        @printf(file_id, "\n")

        @printf(file_id, "  population_derivatives[%s]:", String(basis_name))
        for α in 1:n_sys
            @printf(file_id, "  dp%d=% .8e", α, real(σ′_t[α, α]))
        end
        @printf(file_id, "\n")

        if n_sys >= 2
            @printf(file_id, "  coherences[%s]:\n", String(basis_name))
            for β in 1:n_sys, α in 1:n_sys
                α == β && continue
                c = σ_t[α, β]
                c′ = σ′_t[α, β]
                @printf(file_id,
                    "    σ[%d,%d]=(% .8e,% .8e)  abs=% .8e  phase=% .8e  σprime[%d,%d]=(% .8e,% .8e)\n",
                    α, β,
                    real(c), imag(c), abs(c), angle(c),
                    α, β,
                    real(c′), imag(c′),
                )
            end
        end
    end
end

@inline function _write__rmrt_site_basis_dynamics!(
    file_id::IO,
    σ_hist::Array{ComplexF64,3},
    σ′_hist::Array{ComplexF64,3},
    U_sys::AbstractMatrix{ComplexF64},
    n_sys::Int,
    n_save::Int,
    Δt::Real,
)
    σ_site = Matrix{ComplexF64}(undef, n_sys, n_sys)
    σ′_site = Matrix{ComplexF64}(undef, n_sys, n_sys)

    @inbounds for ti in 1:n_save
        t = (ti - 1) * Δt
        σ_exci = @view σ_hist[:, :, ti]
        σ′_exci = @view σ′_hist[:, :, ti]

        # σ_site = U σ_exci U†
        mul!(σ_site, U_sys, σ_exci)
        σ_site .= σ_site * U_sys'

        mul!(σ′_site, U_sys, σ′_exci)
        σ′_site .= σ′_site * U_sys'

        trσ = tr(σ_site)
        trσ′ = tr(σ′_site)

        @printf(file_id, "\nt=%12.6f  trace=(% .8e,% .8e)  trace_prime=(% .8e,% .8e)\n",
            t, real(trσ), imag(trσ), real(trσ′), imag(trσ′))

        @printf(file_id, "  populations[site]:")
        for i in 1:n_sys
            @printf(file_id, "  p%d=% .8e", i, real(σ_site[i, i]))
        end
        @printf(file_id, "\n")

        @printf(file_id, "  population_derivatives[site]:")
        for i in 1:n_sys
            @printf(file_id, "  dp%d=% .8e", i, real(σ′_site[i, i]))
        end
        @printf(file_id, "\n")

        if n_sys >= 2
            @printf(file_id, "  coherences[site]:\n")
            for j in 1:n_sys, i in 1:n_sys
                i == j && continue
                c = σ_site[i, j]
                c′ = σ′_site[i, j]
                @printf(file_id,
                    "    σ[%d,%d]=(% .8e,% .8e)  abs=% .8e  phase=% .8e  σprime[%d,%d]=(% .8e,% .8e)\n",
                    i, j,
                    real(c), imag(c), abs(c), angle(c),
                    i, j,
                    real(c′), imag(c′),
                )
            end
        end
    end
end

calc__exciton_energy!(context::RmrtContext)                             = calc__ϵ_exci!(context)
calc__line_broadening_functions!(context::RmrtContext)                  = calc__g_g′_g″!(context)
calc__line_broadening_functions_with_threads!(context::RmrtContext)     = calc__g_g′_g″_with_threads!(context)
calc__reduced_density_matrix!(context::RmrtContext)                     = calc__σ_σ′!(context)

end 

