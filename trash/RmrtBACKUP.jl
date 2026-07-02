
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

# @patternized Patternized_g′ (n_sys::Int, n_itr::Int) (t::Int, a::Int, b::Int, c::Int, d::Int) begin
#     rule(αααα, Matrix{T},   zeros(T, n_itr, n_sys), (t, a), a == b && b == c && c == d)
#     rule(αααβ, Array{T,3},  zeros(T, n_itr, n_sys, n_sys), (t, a, d), a == b && b == c && a != d)
#     rule(ααββ, Array{T,3},  zeros(T, n_itr, n_sys, n_sys), (t, a, c), a == b && c == d && a != c)
#     rule(ααβγ, Array{T,4},  zeros(T, n_itr, n_sys, n_sys, n_sys), (t, a, c, d), a == b && a != c && a != d && c != d)
#     rule(αβαα, Array{T,3},  zeros(T, n_itr, n_sys, n_sys), (t, a, b), a == c && c == d && a != b)
#     rule(αβββ, Array{T,3},  zeros(T, n_itr, n_sys, n_sys), (t, a, b), b == c && c == d && a != b)
#     rule(αβγγ, Array{T,4},  zeros(T, n_itr, n_sys, n_sys, n_sys), (t, a, b, c), c == d && a != b && a != c && b != c)
# end

# @patternized Patternized_g″ (n_sys::Int, n_itr::Int) (t::Int, a::Int, b::Int, c::Int, d::Int) begin
#     rule(αββα, Array{T,3},  zeros(T, n_itr, n_sys, n_sys), (t, a, b), a == d && b == c && a != b)
#     rule(αββγ, Array{T,4},  zeros(T, n_itr, n_sys, n_sys, n_sys), (t, a, b, d), b == c && a != b && a != d && b != d)
#     rule(αβγδ, Array{T,5},  zeros(T, n_itr, n_sys, n_sys, n_sys, n_sys), (t, a, b, c, d), a != b && a != c && a != d && b != c && b != d && c != d)
# end

@patternized Patternized_g′ (n_sys::Int, n_itr::Int) (t::Int, a::Int, b::Int, c::Int, d::Int) begin
    rule(αααα, Matrix{T}, zeros(T, n_itr, n_sys), (t, a), a == b && b == c && c == d)
    rule(αααβ, Array{T,3}, zeros(T, n_itr, n_sys, n_sys), (t, a, d), a == b && b == c && a != d)
    rule(ααβα, Array{T,3}, zeros(T, n_itr, n_sys, n_sys), (t, a, c), a == b && a == d && a != c)
    rule(ααββ, Array{T,3}, zeros(T, n_itr, n_sys, n_sys), (t, a, c), a == b && c == d && a != c)
    rule(ααβγ, Array{T,4}, zeros(T, n_itr, n_sys, n_sys, n_sys), (t, a, c, d), a == b && a != c && a != d && c != d)
    rule(αβαα, Array{T,3}, zeros(T, n_itr, n_sys, n_sys), (t, a, b), a == c && c == d && a != b)
    rule(αβββ, Array{T,3}, zeros(T, n_itr, n_sys, n_sys), (t, a, b), b == c && c == d && a != b)
    rule(αβγγ, Array{T,4}, zeros(T, n_itr, n_sys, n_sys, n_sys), (t, a, b, c), c == d && a != b && a != c && b != c)
end

@patternized Patternized_g″ (n_sys::Int, n_itr::Int) (t::Int, a::Int, b::Int, c::Int, d::Int) begin
    rule(αββα, Array{T,3}, zeros(T, n_itr, n_sys, n_sys), (t, a, b), a == d && b == c && a != b)
    rule(αβαβ, Array{T,3}, zeros(T, n_itr, n_sys, n_sys), (t, a, b), a == c && b == d && a != b)
    rule(αββγ, Array{T,4}, zeros(T, n_itr, n_sys, n_sys, n_sys), (t, a, b, d), b == c && a != b && a != d && b != d)
    rule(αβαγ, Array{T,4}, zeros(T, n_itr, n_sys, n_sys, n_sys), (t, a, b, d), a == c && a != b && a != d && b != d)
    rule(αβγα, Array{T,4}, zeros(T, n_itr, n_sys, n_sys, n_sys), (t, a, b, c), a == d && a != b && a != c && b != c)
    rule(αβγβ, Array{T,4}, zeros(T, n_itr, n_sys, n_sys, n_sys), (t, a, b, c), b == d && a != b && a != c && b != c)
    rule(αβγδ, Array{T,5}, zeros(T, n_itr, n_sys, n_sys, n_sys, n_sys), (t, a, b, c, d), a != b && a != c && a != d && b != c && b != d && c != d)
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

function zero__g_g′_g″!(g::Patternized_g{T}, g′::Patternized_g′{T}, g″::Patternized_g″{T}) where {T}
    z = zero(T)

    # g canonical patterns
    fill!(g.αααα, z)
    fill!(g.ααββ, z)

    # g′ canonical patterns
    fill!(g′.αααα, z)
    fill!(g′.αααβ, z)
    fill!(g′.ααβα, z)
    fill!(g′.ααββ, z)
    fill!(g′.ααβγ, z)
    fill!(g′.αβαα, z)
    fill!(g′.αβββ, z)
    fill!(g′.αβγγ, z)

    # g″ canonical patterns
    fill!(g″.αββα, z)
    fill!(g″.αβαβ, z)
    fill!(g″.αββγ, z)
    fill!(g″.αβαγ, z)
    fill!(g″.αβγα, z)
    fill!(g″.αβγβ, z)
    fill!(g″.αβγδ, z)
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

                # ααβα : g′_{αα α⁻ α}
                g′[time_idx, α, α, α⁻, α] += γ_αα * γ[α⁻, α] * F_g′

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
        #   αββα, αβαβ, αββγ, αβαγ, αβγα, αβγβ, αβγδ
        # ---------------------------------------------------------------------
        for α in 1:n_sys
            for α⁻ in 1:n_sys
                α⁻ == α && continue

                γ_αα⁻ = γ[α, α⁻]

                # αββα : g″_{α α⁻ α⁻ α}
                g″[time_idx, α, α⁻, α⁻, α] += γ_αα⁻ * γ[α⁻, α] * F_g″

                # αβαβ : g″_{α α⁻ α α⁻}
                g″[time_idx, α, α⁻, α, α⁻] += γ_αα⁻ * γ[α, α⁻] * F_g″

                for α⁼ in 1:n_sys
                    (α⁼ == α || α⁼ == α⁻) && continue

                    # αββγ : g″_{α α⁻ α⁻ α⁼}
                    g″[time_idx, α, α⁻, α⁻, α⁼] += γ_αα⁻ * γ[α⁻, α⁼] * F_g″

                    # αβαγ : g″_{α α⁻ α α⁼}
                    g″[time_idx, α, α⁻, α, α⁼] += γ_αα⁻ * γ[α, α⁼] * F_g″

                    # αβγα : g″_{α α⁻ α⁼ α}
                    g″[time_idx, α, α⁻, α⁼, α] += γ_αα⁻ * γ[α⁼, α] * F_g″

                    # αβγβ : g″_{α α⁻ α⁼ α⁻}
                    g″[time_idx, α, α⁻, α⁼, α⁻] += γ_αα⁻ * γ[α⁼, α⁻] * F_g″

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

        @printf(stderr, "Current iteration: %6d / %6d \n", curr_itr, n_itr)

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


function calc__σ_σ′_with_population_closure!(context::RmrtContext)
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
    # Population-closed NZ path kernel
    #
    # This is NOT the same coefficient as the coherence equation.
    #
    # For a transfer src -> dst:
    #
    #   K_{dst <- src}(t,s)
    #     = 2 Re[
    #         exp(-i ω_{dst,src} Δ)
    #         exp(E_{src->dst}(t,s))
    #         C_{src->dst}(t,s)
    #       ]
    #
    # where C has the ddot + one-point*one-point structure from the
    # population-closed path expression.
    # -------------------------------------------------------------------------

    @inline function gen__population_transfer_exponent(
        s_itr::Int,
        Δ_itr::Int,
        t_itr::Int,
        src::Int,
        dst::Int,
    )
        return (
            g[t_itr, src, src, src, src]
            - g[s_itr, src, src, src, src]

            - g[Δ_itr, dst, dst, dst, dst]

            - (
                g[t_itr, dst, dst, src, src]
                - g[s_itr, dst, dst, src, src]
                - g[Δ_itr, dst, dst, src, src]
            )

            - g[Δ_itr, src, src, src, src]

            + conj(
                g[s_itr, src, src, src, src]
                - g[t_itr, src, src, src, src]
            )

            + g[Δ_itr, src, src, dst, dst]

            + conj(
                g[t_itr, dst, dst, src, src]
                - g[s_itr, dst, dst, src, src]
            )
        )
    end

    @inline function gen__population_transfer_coef(
        s_itr::Int,
        Δ_itr::Int,
        t_itr::Int,
        src::Int,
        dst::Int,
    )
        left_one_point = (
            -1.0im * g′[t_itr, src, dst, src, src]
            +1.0im * g′[Δ_itr, src, dst, src, src]
            -1.0im * g′[Δ_itr, src, dst, dst, dst]
            +1.0im * conj(g′[t_itr, dst, src, src, src])
        )

        right_one_point = (
            -1.0im * g′[s_itr, dst, src, src, src]
            -1.0im * g′[Δ_itr, dst, dst, dst, src]
            +1.0im * conj(g′[s_itr, src, dst, src, src])
            +1.0im * g′[Δ_itr, src, src, dst, src]
        )

        return (
            g″[Δ_itr, src, dst, dst, src]
            + left_one_point * right_one_point
        )
    end

    @inline function gen__population_transfer_kernel(
        s_itr::Int,
        Δ_itr::Int,
        t_itr::Int,
        src::Int,
        dst::Int,
    )
        return 2.0 * real(
            phase(dst, src, Δ_itr)
            * exp(gen__population_transfer_exponent(s_itr, Δ_itr, t_itr, src, dst))
            * gen__population_transfer_coef(s_itr, Δ_itr, t_itr, src, dst)
        )
    end

    @inline function calc__population_closed_rhs(
        α::Int,
        curr_itr::Int,
        σ_t::AbstractMatrix{ComplexF64},
    )
        curr_itr > 1 || return (
            0.0 + 0.0im,
            0.0 + 0.0im,
            0.0 + 0.0im,
        )

        rhs  = 0.0
        loss = 0.0
        gain = 0.0

        pα = real(σ_t[α, α])

        for s_itr in 1:curr_itr
            Δ_itr = curr_itr - s_itr + 1
            w_int = ∫weight(s_itr, curr_itr)

            for f in 1:n_sys
                f == α && continue

                pf = real(σ_t[f, f])

                # α -> f loss
                k_loss = gen__population_transfer_kernel(
                    s_itr,
                    Δ_itr,
                    curr_itr,
                    α,
                    f,
                )

                # f -> α gain
                k_gain = gen__population_transfer_kernel(
                    s_itr,
                    Δ_itr,
                    curr_itr,
                    f,
                    α,
                )

                loss_contrib = -pα * k_loss
                gain_contrib =  pf * k_gain

                rhs  += w_int * (loss_contrib + gain_contrib)
                loss += w_int * loss_contrib
                gain += w_int * gain_contrib
            end
        end

        return (
            rhs + 0.0im,
            loss + 0.0im,
            gain + 0.0im,
        )
    end

    # -------------------------------------------------------------------------
    # Time propagation loop
    # -------------------------------------------------------------------------
    @inbounds for curr_itr in start_itr:(n_itr - 1)

        @printf(stderr, "Current iteration: %6d / %6d \n", curr_itr, n_itr)

        σ_t  = @view σ[:, :, curr_itr]
        σ′_t = @view σ′[:, :, curr_itr]

        fill!(σ′_t, 0.0 + 0.0im)

        # -------------------------------------------------------------------------
        # Main loop
        # -------------------------------------------------------------------------
        @inbounds for β in 1:n_sys, α in 1:n_sys

            # 먼저 Population closure로 계산해
            if α == β
                rhs_pop, pop_loss, pop_gain = calc__population_closed_rhs(
                    α,
                    curr_itr,
                    σ_t,
                )

                σ′_t[α, α] = rhs_pop
                continue
            end

            # The present equation is the off-diagonal coherence equation.
            # Population dynamics should be handled by a separate population closure.

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

        # 일단 Hermitian 한번 조질게요
        # 이 미친놈이 오차를 누적시켜요
        for i in 1:n_sys
            σ[i, i, curr_itr + 1] = real(σ[i, i, curr_itr + 1]) + 0.0im
        end
        for i in 1:n_sys-1
            for j in i+1:n_sys
                c = 0.5 * (σ[i, j, curr_itr + 1] + conj(σ[j, i, curr_itr + 1]))
                σ[i, j, curr_itr + 1] = c
                σ[j, i, curr_itr + 1] = conj(c)
            end
        end

        context.curr_itr = UInt64(curr_itr + 1)
    end

    return @view σ[:, :, Int(context.curr_itr)]
end

# function calc__σ_σ′!(
#     context::RmrtContext;
#     debug_population_flow::Bool = true,
#     debug_stride::Int = 100,
#     debug_io::IO = stderr,
# )
#     start_itr = Int(context.curr_itr)

#     n_sys    = context.system.n_sys
#     n_itr    = context.simulation_details.num_of_iteration
#     Δt       = context.simulation_details.Δt
#     ϵ        = context.ϵ_exci

#     σ        = context.σ
#     σ′       = context.σ′
#     g        = context.g
#     g′       = context.g′
#     g″       = context.g″

#     start_itr < n_itr || return @view σ[:, :, n_itr]

#     debug_stride = max(debug_stride, 1)

#     @inline ω(a::Int, b::Int) = ϵ[a] - ϵ[b]

#     @inline function ∫weight(s_itr::Int, curr_itr::Int)
#         return (s_itr == 1 || s_itr == curr_itr) ? 0.5 * Δt : Δt
#     end

#     @inline function phase(a::Int, b::Int, Δ_itr::Int)
#         Δ = (Δ_itr - 1) * Δt
#         return exp(-1.0im * ω(a, b) * Δ)
#     end

#     @inline function gen__exponent_type_1(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α⁻::Int,
#         α⁼::Int,
#         β::Int,
#     )
#         return (
#             -g[s_itr, α⁼, α⁼, α⁼, α⁼]
#             +g[s_itr, α⁻, α⁻, α⁼, α⁼]
#             +conj(g[s_itr, α⁼, α⁼, β, β])
#             -conj(g[s_itr, α⁻, α⁻, β, β])

#             -g[Δ_itr, α⁻, α⁻, α⁻, α⁻]
#             +g[Δ_itr, α⁻, α⁻, α⁼, α⁼]
#             -g[Δ_itr, β, β, α⁼, α⁼]
#             +g[Δ_itr, β, β, α⁻, α⁻]

#             +g[t_itr, α⁼, α⁼, α⁼, α⁼]
#             -g[t_itr, α⁻, α⁻, α⁼, α⁼]
#             -conj(g[t_itr, α⁼, α⁼, β, β])
#             +conj(g[t_itr, α⁻, α⁻, β, β])
#         )
#     end

#     @inline function gen__exponent_type_2(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         a::Int,
#         β⁼::Int,
#         β⁻::Int,
#     )
#         return (
#             -conj(g[s_itr, β⁼, β⁼, β⁼, β⁼])
#             +conj(g[s_itr, β⁻, β⁻, β⁼, β⁼])
#             +g[s_itr, β⁼, β⁼, a, a]
#             -g[s_itr, β⁻, β⁻, a, a]

#             -conj(g[Δ_itr, β⁻, β⁻, β⁻, β⁻])
#             +conj(g[Δ_itr, β⁻, β⁻, β⁼, β⁼])
#             -conj(g[Δ_itr, a, a, β⁼, β⁼])
#             +conj(g[Δ_itr, a, a, β⁻, β⁻])

#             +conj(g[t_itr, β⁼, β⁼, β⁼, β⁼])
#             -conj(g[t_itr, β⁻, β⁻, β⁼, β⁼])
#             -g[t_itr, β⁼, β⁼, a, a]
#             +g[t_itr, β⁻, β⁻, a, a]
#         )
#     end

#     @inline function gen_coef_block_type_1(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         α⁻::Int,
#         α⁼::Int,
#         β::Int,
#     )
#         left_one_point = (
#              g′[Δ_itr, α, α⁻, α⁼, α⁼]
#             -g′[Δ_itr, α, α⁻, α⁻, α⁻]
#             -g′[t_itr, α, α⁻, α⁼, α⁼]
#             +g′[t_itr, α, α⁻, α⁻, α⁻]
#         )

#         right_one_point = (
#             -g′[s_itr, α⁻, α⁼, α⁼, α⁼]
#             +conj(g′[s_itr, α⁼, α⁻, β, β])
#             -g′[Δ_itr, α⁻, α⁻, α⁻, α⁼]
#             +g′[Δ_itr, β, β, α⁻, α⁼]
#         )

#         return g″[Δ_itr, α, α⁻, α⁻, α⁼] - left_one_point * right_one_point
#     end

#     @inline function gen_coef_block_type_2(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         β::Int,
#         β⁻::Int,
#         β⁼::Int,
#         α::Int,
#     )
#         left_one_point = (
#             -conj(g′[Δ_itr, β, β⁻, β⁼, β⁼])
#             +conj(g′[Δ_itr, β, β⁻, β⁻, β⁻])
#             +conj(g′[t_itr, β, β⁻, β⁼, β⁼])
#             -conj(g′[t_itr, β, β⁻, β⁻, β⁻])
#         )

#         right_one_point = (
#             -g′[s_itr, β⁼, β⁻, α, α]
#             +conj(g′[s_itr, β⁻, β⁼, β⁼, β⁼])
#             -conj(g′[Δ_itr, α, α, β⁻, β⁼])
#             +conj(g′[Δ_itr, β⁻, β⁻, β⁻, β⁼])
#         )

#         return conj(g″[Δ_itr, β, β⁻, β⁻, β⁼]) - left_one_point * right_one_point
#     end

#     @inline function gen_coef_block_type_3(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         α⁻::Int,
#         β⁻::Int,
#     )
#         left_one_point = (
#             -g′[s_itr, α, α⁻, α⁻, α⁻]
#             +conj(g′[s_itr, α⁻, α, β⁻, β⁻])
#             -g′[Δ_itr, α, α, α, α⁻]
#             +g′[Δ_itr, β⁻, β⁻, α, α⁻]
#         )

#         right_one_point = (
#             -g′[Δ_itr, β⁻, β, α, α]
#             +g′[Δ_itr, β⁻, β, α⁻, α⁻]
#             +g′[t_itr, β⁻, β, α, α]
#             -g′[t_itr, β⁻, β, α⁻, α⁻]
#         )

#         return g″[Δ_itr, β⁻, β, α, α⁻] - left_one_point * right_one_point
#     end

#     @inline function gen_coef_block_type_4(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         α⁻::Int,
#         β⁻::Int,
#     )
#         left_one_point = (
#             -conj(g′[Δ_itr, α⁻, α, β⁻, β⁻])
#             +conj(g′[Δ_itr, α⁻, α, β, β])
#             +conj(g′[t_itr, α⁻, α, β⁻, β⁻])
#             -conj(g′[t_itr, α⁻, α, β, β])
#         )

#         right_one_point = (
#             -g′[s_itr, β⁻, β, α⁻, α⁻]
#             +conj(g′[s_itr, β, β⁻, β⁻, β⁻])
#             -conj(g′[Δ_itr, α⁻, α⁻, β, β⁻])
#             +conj(g′[Δ_itr, β, β, β, β⁻])
#         )

#         return conj(g″[Δ_itr, α⁻, α, β, β⁻]) - left_one_point * right_one_point
#     end

#     # -------------------------------------------------------------------------
#     # Debug buffers
#     # -------------------------------------------------------------------------
#     debug_local_diagpart = debug_population_flow ? zeros(ComplexF64, n_sys) : ComplexF64[]
#     debug_local_left     = debug_population_flow ? zeros(ComplexF64, n_sys) : ComplexF64[]
#     debug_local_right    = debug_population_flow ? zeros(ComplexF64, n_sys) : ComplexF64[]
#     debug_local_total    = debug_population_flow ? zeros(ComplexF64, n_sys) : ComplexF64[]

#     debug_loss_diag      = debug_population_flow ? zeros(ComplexF64, n_sys) : ComplexF64[]
#     debug_gain_diag      = debug_population_flow ? zeros(ComplexF64, n_sys) : ComplexF64[]
#     debug_total_diag     = debug_population_flow ? zeros(ComplexF64, n_sys) : ComplexF64[]

#     if debug_population_flow
#         @printf(debug_io, "\n# ---- RMRT population local/loss/gain debug ----\n")
#         @printf(
#             debug_io,
#             "# itr time alpha p_alpha diag_re left_re right_re local_re loss_re loss_im gain_re gain_im net_mem_re dp_re abs_loss abs_gain k_loss k_gain kgain_over_kloss\n"
#         )
#     end

#     # -------------------------------------------------------------------------
#     # Time propagation loop
#     # -------------------------------------------------------------------------
#     @inbounds for curr_itr in start_itr:(n_itr - 1)

#         # @printf(stderr, "Current iteration: %6d / %6d \n", curr_itr, n_itr)

#         σ_t  = @view σ[:, :, curr_itr]
#         σ′_t = @view σ′[:, :, curr_itr]

#         fill!(σ′_t, 0.0 + 0.0im)

#         if debug_population_flow
#             fill!(debug_local_diagpart, 0.0 + 0.0im)
#             fill!(debug_local_left,     0.0 + 0.0im)
#             fill!(debug_local_right,    0.0 + 0.0im)
#             fill!(debug_local_total,    0.0 + 0.0im)

#             fill!(debug_loss_diag,      0.0 + 0.0im)
#             fill!(debug_gain_diag,      0.0 + 0.0im)
#             fill!(debug_total_diag,     0.0 + 0.0im)
#         end

#         # ---------------------------------------------------------------------
#         # Main loop
#         # ---------------------------------------------------------------------
#         @inbounds for β in 1:n_sys, α in 1:n_sys

#             # -------------------------------------------------------------
#             # t-local diagonal/coherence phase block
#             # -------------------------------------------------------------
#             local_diag = (
#                 -1.0im * ω(α, β)
#                 -g′[curr_itr, α, α, α, α] + conj(g′[curr_itr, α, α, β, β])
#                 +g′[curr_itr, β, β, α, α] - conj(g′[curr_itr, β, β, β, β])
#             ) * σ_t[α, β]

#             rhs = local_diag

#             # # -------------------------------------------------------------
#             # # t-local left mixing
#             # # -------------------------------------------------------------
#             # local_left = 0.0 + 0.0im

#             # for α⁻ in 1:n_sys
#             #     α⁻ == α && continue

#             #     term_left = -σ_t[α⁻, β] * (
#             #         g′[curr_itr, α, α⁻, α⁻, α⁻]
#             #         - conj(g′[curr_itr, α⁻, α, β, β])
#             #     )

#             #     local_left += term_left
#             #     rhs += term_left
#             # end

#             # # -------------------------------------------------------------
#             # # t-local right mixing
#             # # -------------------------------------------------------------
#             # local_right = 0.0 + 0.0im

#             # for β⁻ in 1:n_sys
#             #     β⁻ == β && continue

#             #     term_right = σ_t[α, β⁻] * (
#             #         g′[curr_itr, β⁻, β, α, α]
#             #         - conj(g′[curr_itr, β, β⁻, β⁻, β⁻])
#             #     )

#             #     local_right += term_right
#             #     rhs += term_right
#             # end

#             # if debug_population_flow && α == β
#             #     debug_local_diagpart[α] = local_diag
#             #     debug_local_left[α]     = local_left
#             #     debug_local_right[α]    = local_right
#             #     debug_local_total[α]    = local_diag + local_left + local_right
#             # end

#             # t-local left mixing
#             local_left = 0.0 + 0.0im

#             for α⁻ in 1:n_sys
#                 α⁻ == α && continue

#                 term_left = -σ_t[α⁻, β] * (
#                     g′[curr_itr, α, α⁻, α⁻, α⁻]
#                     - conj(g′[curr_itr, α⁻, α, β, β])
#                 )

#                 local_left += term_left

#                 # DEBUG TEST:
#                 # population diagonal에서는 local coherence feedback을 끈다.
#                 # if !(α == β)
#                 #     rhs += term_left
#                 # end
#                 rhs += term_left
#             end

#             # t-local right mixing
#             local_right = 0.0 + 0.0im

#             for β⁻ in 1:n_sys
#                 β⁻ == β && continue

#                 term_right = σ_t[α, β⁻] * (
#                     g′[curr_itr, β⁻, β, α, α]
#                     - conj(g′[curr_itr, β, β⁻, β⁻, β⁻])
#                 )

#                 local_right += term_right

#                 # DEBUG TEST:
#                 # population diagonal에서는 local coherence feedback을 끈다.
#                 # if !(α == β)
#                 #     rhs += term_right
#                 # end
#                 rhs += term_right
#             end

#             # -------------------------------------------------------------
#             # memory integral
#             # -------------------------------------------------------------
#             if curr_itr > 1
#                 integral = 0.0 + 0.0im

#                 debug_loss_integral = 0.0 + 0.0im
#                 debug_gain_integral = 0.0 + 0.0im
#                 is_diag_for_debug = debug_population_flow && (α == β)

#                 for s_itr in 1:curr_itr
#                     Δ_itr = curr_itr - s_itr + 1
#                     w_int = ∫weight(s_itr, curr_itr)

#                     kernel = 0.0 + 0.0im

#                     # -----------------------------------------------------
#                     # Block 1: loss-like for diagonal population
#                     # -----------------------------------------------------
#                     for α⁻ in 1:n_sys
#                         α⁻ == α && continue

#                         for α⁼ in 1:n_sys
#                             α⁼ == α⁻ && continue

#                             term = (
#                                 σ_t[α⁼, β]
#                                 * phase(α⁻, α⁼, Δ_itr)
#                                 * exp(gen__exponent_type_1(s_itr, Δ_itr, curr_itr, α⁻, α⁼, β))
#                                 * gen_coef_block_type_1(s_itr, Δ_itr, curr_itr, α, α⁻, α⁼, β)
#                             )

#                             kernel -= term

#                             if is_diag_for_debug
#                                 debug_loss_integral += w_int * (-term)
#                             end
#                         end
#                     end

#                     # -----------------------------------------------------
#                     # Block 2: loss-like for diagonal population
#                     # -----------------------------------------------------
#                     for β⁻ in 1:n_sys
#                         β⁻ == β && continue

#                         for β⁼ in 1:n_sys
#                             β⁼ == β⁻ && continue

#                             term = (
#                                 σ_t[α, β⁼]
#                                 * phase(β⁼, β⁻, Δ_itr)
#                                 * exp(gen__exponent_type_2(s_itr, Δ_itr, curr_itr, α, β⁼, β⁻))
#                                 * gen_coef_block_type_2(s_itr, Δ_itr, curr_itr, β, β⁻, β⁼, α)
#                             )

#                             kernel -= term

#                             if is_diag_for_debug
#                                 debug_loss_integral += w_int * (-term)
#                             end
#                         end
#                     end

#                     # -----------------------------------------------------
#                     # Block 3: gain-like for diagonal population
#                     # -----------------------------------------------------
#                     for α⁻ in 1:n_sys
#                         α⁻ == α && continue

#                         for β⁻ in 1:n_sys
#                             β⁻ == β && continue

#                             term = (
#                                 σ_t[α⁻, β⁻]
#                                 * phase(α, α⁻, Δ_itr)
#                                 * exp(gen__exponent_type_1(s_itr, Δ_itr, curr_itr, α, α⁻, β⁻))
#                                 * gen_coef_block_type_3(s_itr, Δ_itr, curr_itr, α, β, α⁻, β⁻)
#                             )

#                             kernel += term

#                             if is_diag_for_debug
#                                 debug_gain_integral += w_int * term
#                             end
#                         end
#                     end

#                     # -----------------------------------------------------
#                     # Block 4: gain-like for diagonal population
#                     # -----------------------------------------------------
#                     for α⁻ in 1:n_sys
#                         α⁻ == α && continue

#                         for β⁻ in 1:n_sys
#                             β⁻ == β && continue

#                             term = (
#                                 σ_t[α⁻, β⁻]
#                                 * phase(β⁻, β, Δ_itr)
#                                 * exp(gen__exponent_type_2(s_itr, Δ_itr, curr_itr, α⁻, β⁻, β))
#                                 * gen_coef_block_type_4(s_itr, Δ_itr, curr_itr, α, β, α⁻, β⁻)
#                             )

#                             kernel += term

#                             if is_diag_for_debug
#                                 debug_gain_integral += w_int * term
#                             end
#                         end
#                     end

#                     integral += w_int * kernel
#                 end

#                 if is_diag_for_debug
#                     debug_loss_diag[α] = debug_loss_integral
#                     debug_gain_diag[α] = debug_gain_integral
#                 end

#                 rhs += integral
#             end

#             σ′_t[α, β] = rhs

#             if debug_population_flow && α == β
#                 debug_total_diag[α] = rhs
#             end
#         end

#         # ---------------------------------------------------------------------
#         # Debug print per iteration
#         # ---------------------------------------------------------------------
#         if debug_population_flow && ((curr_itr - start_itr) % debug_stride == 0)
#             t_now = (curr_itr - 1) * Δt

#             for α in 1:n_sys
#                 pα = real(σ_t[α, α])

#                 diag_val  = debug_local_diagpart[α]
#                 left_val  = debug_local_left[α]
#                 right_val = debug_local_right[α]
#                 local_val = debug_local_total[α]

#                 loss_val  = debug_loss_diag[α]
#                 gain_val  = debug_gain_diag[α]
#                 mem_val   = loss_val + gain_val
#                 total_val = debug_total_diag[α]

#                 k_loss = NaN
#                 k_gain = NaN
#                 ratio  = NaN

#                 if abs(pα) > 1e-14
#                     k_loss = -real(loss_val) / pα
#                 end

#                 if n_sys == 2
#                     μ = 3 - α
#                     pμ = real(σ_t[μ, μ])

#                     if abs(pμ) > 1e-14
#                         k_gain = real(gain_val) / pμ
#                     end

#                     if isfinite(k_loss) && abs(k_loss) > 1e-14
#                         ratio = k_gain / k_loss
#                     end
#                 end

#                 @printf(
#                     debug_io,
#                     "%10d %12.6f %4d %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e\n",
#                     curr_itr,
#                     t_now,
#                     α,
#                     pα,
#                     real(diag_val),
#                     real(left_val),
#                     real(right_val),
#                     real(local_val),
#                     real(loss_val),
#                     imag(loss_val),
#                     real(gain_val),
#                     imag(gain_val),
#                     real(mem_val),
#                     real(total_val),
#                     abs(loss_val),
#                     abs(gain_val),
#                     k_loss,
#                     k_gain,
#                     ratio,
#                 )
#             end
#         end

#         @views σ[:, :, curr_itr + 1] .= σ[:, :, curr_itr] .+ Δt .* σ′[:, :, curr_itr]
#         context.curr_itr = UInt64(curr_itr + 1)
#     end

#     return @view σ[:, :, Int(context.curr_itr)]
# end

function calc__σ_σ′_with_threads!(context::RmrtContext)
    start_itr = Int(context.curr_itr)
    n_itr     = context.simulation_details.num_of_iteration
    n_sys     = context.system.n_sys
    Δt        = context.simulation_details.Δt
    ϵ         = context.ϵ_exci
    σ         = context.σ
    σ′        = context.σ′
    g         = context.g
    g′        = context.g′
    g″        = context.g″
    start_itr < n_itr || return @view σ[:, :, n_itr]

    @inline ω(a::Int, b::Int) = ϵ[a] - ϵ[b]
    @inline ∫weight(s_itr::Int, curr_itr::Int) = (s_itr == 1 || s_itr == curr_itr) ? 0.5 * Δt : Δt
    @inline function phase(a::Int, b::Int, Δ_itr::Int)
        Δ = (Δ_itr - 1) * Δt
        return exp(-1.0im * ω(a, b) * Δ)
    end

    @inline function gen__exponent_type_1(s_itr::Int, Δ_itr::Int, t_itr::Int, α⁻::Int, α⁼::Int, β::Int)
        return -g[s_itr, α⁼, α⁼, α⁼, α⁼] + g[s_itr, α⁻, α⁻, α⁼, α⁼] + conj(g[s_itr, α⁼, α⁼, β, β]) - conj(g[s_itr, α⁻, α⁻, β, β]) - g[Δ_itr, α⁻, α⁻, α⁻, α⁻] + g[Δ_itr, α⁻, α⁻, α⁼, α⁼] - g[Δ_itr, β, β, α⁼, α⁼] + g[Δ_itr, β, β, α⁻, α⁻] + g[t_itr, α⁼, α⁼, α⁼, α⁼] - g[t_itr, α⁻, α⁻, α⁼, α⁼] - conj(g[t_itr, α⁼, α⁼, β, β]) + conj(g[t_itr, α⁻, α⁻, β, β])
    end

    @inline function gen__exponent_type_2(s_itr::Int, Δ_itr::Int, t_itr::Int, a::Int, β⁼::Int, β⁻::Int)
        return -conj(g[s_itr, β⁼, β⁼, β⁼, β⁼]) + conj(g[s_itr, β⁻, β⁻, β⁼, β⁼]) + g[s_itr, β⁼, β⁼, a, a] - g[s_itr, β⁻, β⁻, a, a] - conj(g[Δ_itr, β⁻, β⁻, β⁻, β⁻]) + conj(g[Δ_itr, β⁻, β⁻, β⁼, β⁼]) - conj(g[Δ_itr, a, a, β⁼, β⁼]) + conj(g[Δ_itr, a, a, β⁻, β⁻]) + conj(g[t_itr, β⁼, β⁼, β⁼, β⁼]) - conj(g[t_itr, β⁻, β⁻, β⁼, β⁼]) - g[t_itr, β⁼, β⁼, a, a] + g[t_itr, β⁻, β⁻, a, a]
    end

    @inline function gen_coef_block_type_1(s_itr::Int, Δ_itr::Int, t_itr::Int, α::Int, α⁻::Int, α⁼::Int, β::Int)
        left_one_point = g′[Δ_itr, α, α⁻, α⁼, α⁼] - g′[Δ_itr, α, α⁻, α⁻, α⁻] - g′[t_itr, α, α⁻, α⁼, α⁼] + g′[t_itr, α, α⁻, α⁻, α⁻]
        right_one_point = -g′[s_itr, α⁻, α⁼, α⁼, α⁼] + conj(g′[s_itr, α⁼, α⁻, β, β]) - g′[Δ_itr, α⁻, α⁻, α⁻, α⁼] + g′[Δ_itr, β, β, α⁻, α⁼]
        return g″[Δ_itr, α, α⁻, α⁻, α⁼] - left_one_point * right_one_point
    end

    @inline function gen_coef_block_type_2(s_itr::Int, Δ_itr::Int, t_itr::Int, β::Int, β⁻::Int, β⁼::Int, α::Int)
        left_one_point = -conj(g′[Δ_itr, β, β⁻, β⁼, β⁼]) + conj(g′[Δ_itr, β, β⁻, β⁻, β⁻]) + conj(g′[t_itr, β, β⁻, β⁼, β⁼]) - conj(g′[t_itr, β, β⁻, β⁻, β⁻])
        right_one_point = -g′[s_itr, β⁼, β⁻, α, α] + conj(g′[s_itr, β⁻, β⁼, β⁼, β⁼]) - conj(g′[Δ_itr, α, α, β⁻, β⁼]) + conj(g′[Δ_itr, β⁻, β⁻, β⁻, β⁼])
        return conj(g″[Δ_itr, β, β⁻, β⁻, β⁼]) - left_one_point * right_one_point
    end

    @inline function gen_coef_block_type_3(s_itr::Int, Δ_itr::Int, t_itr::Int, α::Int, β::Int, α⁻::Int, β⁻::Int)
        left_one_point = -g′[s_itr, α, α⁻, α⁻, α⁻] + conj(g′[s_itr, α⁻, α, β⁻, β⁻]) - g′[Δ_itr, α, α, α, α⁻] + g′[Δ_itr, β⁻, β⁻, α, α⁻]
        right_one_point = -g′[Δ_itr, β⁻, β, α, α] + g′[Δ_itr, β⁻, β, α⁻, α⁻] + g′[t_itr, β⁻, β, α, α] - g′[t_itr, β⁻, β, α⁻, α⁻]
        return g″[Δ_itr, β⁻, β, α, α⁻] - left_one_point * right_one_point
    end

    @inline function gen_coef_block_type_4(s_itr::Int, Δ_itr::Int, t_itr::Int, α::Int, β::Int, α⁻::Int, β⁻::Int)
        left_one_point = -conj(g′[Δ_itr, α⁻, α, β⁻, β⁻]) + conj(g′[Δ_itr, α⁻, α, β, β]) + conj(g′[t_itr, α⁻, α, β⁻, β⁻]) - conj(g′[t_itr, α⁻, α, β, β])
        right_one_point = -g′[s_itr, β⁻, β, α⁻, α⁻] + conj(g′[s_itr, β, β⁻, β⁻, β⁻]) - conj(g′[Δ_itr, α⁻, α⁻, β, β⁻]) + conj(g′[Δ_itr, β, β, β, β⁻])
        return conj(g″[Δ_itr, α⁻, α, β, β⁻]) - left_one_point * right_one_point
    end

    @inbounds for curr_itr in start_itr:(n_itr - 1)

        @printf(stderr, "Current iteration: %6d / %6d \n", curr_itr, n_itr)

        σ_t = @view σ[:, :, curr_itr]
        σ′_t = @view σ′[:, :, curr_itr]
        fill!(σ′_t, 0.0 + 0.0im)

        Threads.@threads for linear_idx in 1:(n_sys * n_sys)
            α = (linear_idx - 1) % n_sys + 1
            β = (linear_idx - 1) ÷ n_sys + 1

            # if α == β
            #     σ′_t[α, β] = 0.0 + 0.0im
            #     continue
            # end

            rhs = (-1.0im * ω(α, β) - g′[curr_itr, α, α, α, α] + conj(g′[curr_itr, α, α, β, β]) + g′[curr_itr, β, β, α, α] - conj(g′[curr_itr, β, β, β, β])) * σ_t[α, β]

            for α⁻ in 1:n_sys
                α⁻ == α && continue
                rhs -= σ_t[α⁻, β] * (g′[curr_itr, α, α⁻, α⁻, α⁻] - conj(g′[curr_itr, α⁻, α, β, β]))
            end

            for β⁻ in 1:n_sys
                β⁻ == β && continue
                rhs += σ_t[α, β⁻] * (g′[curr_itr, β⁻, β, α, α] - conj(g′[curr_itr, β, β⁻, β⁻, β⁻]))
            end

            integral = 0.0 + 0.0im

            if curr_itr > 1
                for s_itr in 1:curr_itr
                    Δ_itr = curr_itr - s_itr + 1
                    kernel = 0.0 + 0.0im

                    for α⁻ in 1:n_sys
                        α⁻ == α && continue
                        for α⁼ in 1:n_sys
                            α⁼ == α⁻ && continue
                            kernel -= σ_t[α⁼, β] * phase(α⁻, α⁼, Δ_itr) * exp(gen__exponent_type_1(s_itr, Δ_itr, curr_itr, α⁻, α⁼, β)) * gen_coef_block_type_1(s_itr, Δ_itr, curr_itr, α, α⁻, α⁼, β)
                        end
                    end

                    for β⁻ in 1:n_sys
                        β⁻ == β && continue
                        for β⁼ in 1:n_sys
                            β⁼ == β⁻ && continue
                            kernel -= σ_t[α, β⁼] * phase(β⁼, β⁻, Δ_itr) * exp(gen__exponent_type_2(s_itr, Δ_itr, curr_itr, α, β⁼, β⁻)) * gen_coef_block_type_2(s_itr, Δ_itr, curr_itr, β, β⁻, β⁼, α)
                        end
                    end

                    for α⁻ in 1:n_sys
                        α⁻ == α && continue
                        for β⁻ in 1:n_sys
                            β⁻ == β && continue
                            kernel += σ_t[α⁻, β⁻] * phase(α, α⁻, Δ_itr) * exp(gen__exponent_type_1(s_itr, Δ_itr, curr_itr, α, α⁻, β⁻)) * gen_coef_block_type_3(s_itr, Δ_itr, curr_itr, α, β, α⁻, β⁻)
                        end
                    end

                    for α⁻ in 1:n_sys
                        α⁻ == α && continue
                        for β⁻ in 1:n_sys
                            β⁻ == β && continue
                            kernel += σ_t[α⁻, β⁻] * phase(β⁻, β, Δ_itr) * exp(gen__exponent_type_2(s_itr, Δ_itr, curr_itr, α⁻, β⁻, β)) * gen_coef_block_type_4(s_itr, Δ_itr, curr_itr, α, β, α⁻, β⁻)
                        end
                    end

                    integral += ∫weight(s_itr, curr_itr) * kernel
                end
            end

            σ′_t[α, β] = rhs + integral
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


function set__initial_σ_site!(
    context::RmrtContext;
    init_site::Integer = 1
)
    n_sys = context.system.n_sys
    1 <= init_site <= n_sys || error("init_site is out of range")

    context.curr_itr = UInt64(1)

    σ     = context.σ
    σ′    = context.σ′
    U_sys = context.U_sys

    @views σ[:, :, 1]  .= 0.0 + 0.0im
    @views σ′[:, :, 1] .= 0.0 + 0.0im

    @inbounds for β in 1:n_sys, α in 1:n_sys
        σ[α, β, 1] = conj(U_sys[init_site, α]) * U_sys[init_site, β]
    end

    return @view σ[:, :, 1]
end



@inline function _rmrt__col!(file_id::IO, name::AbstractString)
    @printf(file_id, " %18s", name)
end

@inline function _rmrt__val!(file_id::IO, value::Real)
    @printf(file_id, " %18.8e", value)
end

@inline function _write__rmrt_upper_tri_header!(
    file_id::IO,
    n_sys::Int,
    basis_name::Symbol;
    write_derivative::Bool=true,
)
    @printf(file_id, "\n---- RMRT reduced dynamics: serialized upper-triangular matrix [%s basis] ----\n", String(basis_name))
    @printf(file_id, "%14s", "time")
    _rmrt__col!(file_id, "tr_re")
    _rmrt__col!(file_id, "tr_im")

    for i in 1:n_sys
        _rmrt__col!(file_id, "p$(i)")
    end

    for i in 1:n_sys-1, j in i+1:n_sys
        _rmrt__col!(file_id, "c$(i)$(j)_re")
        _rmrt__col!(file_id, "c$(i)$(j)_im")
        _rmrt__col!(file_id, "c$(i)$(j)_abs")
        _rmrt__col!(file_id, "c$(i)$(j)_phase")
    end

    if write_derivative
        _rmrt__col!(file_id, "trp_re")
        _rmrt__col!(file_id, "trp_im")

        for i in 1:n_sys
            _rmrt__col!(file_id, "dp$(i)")
        end

        for i in 1:n_sys-1, j in i+1:n_sys
            _rmrt__col!(file_id, "cp$(i)$(j)_re")
            _rmrt__col!(file_id, "cp$(i)$(j)_im")
            _rmrt__col!(file_id, "cp$(i)$(j)_abs")
            _rmrt__col!(file_id, "cp$(i)$(j)_phase")
        end
    end

    @printf(file_id, "\n")
    return nothing
end

@inline function _write__rmrt_upper_tri_row!(
    file_id::IO,
    σ_t::AbstractMatrix{ComplexF64},
    σ′_t::AbstractMatrix{ComplexF64},
    n_sys::Int,
    t::Real;
    write_derivative::Bool=true,
)
    trσ = tr(σ_t)

    @printf(file_id, "%14.6f", t)
    _rmrt__val!(file_id, real(trσ))
    _rmrt__val!(file_id, imag(trσ))

    @inbounds for i in 1:n_sys
        _rmrt__val!(file_id, real(σ_t[i, i]))
    end

    @inbounds for i in 1:n_sys-1, j in i+1:n_sys
        c = σ_t[i, j]
        _rmrt__val!(file_id, real(c))
        _rmrt__val!(file_id, imag(c))
        _rmrt__val!(file_id, abs(c))
        _rmrt__val!(file_id, angle(c))
    end

    if write_derivative
        trσ′ = tr(σ′_t)

        _rmrt__val!(file_id, real(trσ′))
        _rmrt__val!(file_id, imag(trσ′))

        @inbounds for i in 1:n_sys
            _rmrt__val!(file_id, real(σ′_t[i, i]))
        end

        @inbounds for i in 1:n_sys-1, j in i+1:n_sys
            c′ = σ′_t[i, j]
            _rmrt__val!(file_id, real(c′))
            _rmrt__val!(file_id, imag(c′))
            _rmrt__val!(file_id, abs(c′))
            _rmrt__val!(file_id, angle(c′))
        end
    end

    @printf(file_id, "\n")
    return nothing
end

@inline function _write__rmrt_basis_dynamics!(
    file_id::IO,
    σ_hist::Array{ComplexF64,3},
    σ′_hist::Array{ComplexF64,3},
    n_sys::Int,
    n_save::Int,
    Δt::Real,
    basis_name::Symbol;
    write_derivative::Bool=true,
)
    _write__rmrt_upper_tri_header!(
        file_id,
        n_sys,
        basis_name;
        write_derivative=write_derivative,
    )

    @inbounds for ti in 1:n_save
        t = (ti - 1) * Δt
        σ_t = @view σ_hist[:, :, ti]
        σ′_t = @view σ′_hist[:, :, ti]

        _write__rmrt_upper_tri_row!(
            file_id,
            σ_t,
            σ′_t,
            n_sys,
            t;
            write_derivative=write_derivative,
        )
    end

    return nothing
end

@inline function _write__rmrt_site_basis_dynamics!(
    file_id::IO,
    σ_hist::Array{ComplexF64,3},
    σ′_hist::Array{ComplexF64,3},
    U_sys::AbstractMatrix{ComplexF64},
    n_sys::Int,
    n_save::Int,
    Δt::Real;
    write_derivative::Bool=true,
)
    σ_tmp   = Matrix{ComplexF64}(undef, n_sys, n_sys)
    σ_site  = Matrix{ComplexF64}(undef, n_sys, n_sys)
    σ′_tmp  = Matrix{ComplexF64}(undef, n_sys, n_sys)
    σ′_site = Matrix{ComplexF64}(undef, n_sys, n_sys)

    _write__rmrt_upper_tri_header!(
        file_id,
        n_sys,
        :site;
        write_derivative=write_derivative,
    )

    @inbounds for ti in 1:n_save
        t = (ti - 1) * Δt
        σ_exci = @view σ_hist[:, :, ti]
        σ′_exci = @view σ′_hist[:, :, ti]

        mul!(σ_tmp, U_sys, σ_exci)
        mul!(σ_site, σ_tmp, U_sys')

        mul!(σ′_tmp, U_sys, σ′_exci)
        mul!(σ′_site, σ′_tmp, U_sys')

        _write__rmrt_upper_tri_row!(
            file_id,
            σ_site,
            σ′_site,
            n_sys,
            t;
            write_derivative=write_derivative,
        )
    end

    return nothing
end

function save__rmrt_reduced_dynamics_serialized!(
    context::RmrtContext;
    save_filename::AbstractString="rmrt.txt",
    basis::Symbol=:both,
    write_derivative::Bool=true,
    n_save::Integer=Int(context.curr_itr),
)
    n_sys = context.system.n_sys
    n_itr = context.simulation_details.num_of_iteration
    Δt    = context.simulation_details.Δt
    σ     = context.σ
    σ′    = context.σ′
    U_sys = context.U_sys

    n_save_eff = min(Int(n_save), n_itr, size(σ, 3), size(σ′, 3))

    open(save_filename, "w") do file_id
        @printf(file_id, "# RMRT reduced dynamics serialized by upper-triangular density-matrix entries\n")
        @printf(file_id, "# diagonal entries: p_i = real(σ[i,i])\n")
        @printf(file_id, "# off-diagonal entries for i<j: c_ij = σ[i,j], with real, imag, abs, phase\n")
        @printf(file_id, "# derivative columns use prefix p -> prime: trp, dp, cp\n")

        if basis == :exciton
            _write__rmrt_basis_dynamics!(
                file_id,
                σ,
                σ′,
                n_sys,
                n_save_eff,
                Δt,
                :exciton;
                write_derivative=write_derivative,
            )
        elseif basis == :site
            _write__rmrt_site_basis_dynamics!(
                file_id,
                σ,
                σ′,
                U_sys,
                n_sys,
                n_save_eff,
                Δt;
                write_derivative=write_derivative,
            )
        elseif basis == :both
            _write__rmrt_basis_dynamics!(
                file_id,
                σ,
                σ′,
                n_sys,
                n_save_eff,
                Δt,
                :exciton;
                write_derivative=write_derivative,
            )

            _write__rmrt_site_basis_dynamics!(
                file_id,
                σ,
                σ′,
                U_sys,
                n_sys,
                n_save_eff,
                Δt;
                write_derivative=write_derivative,
            )
        else
            error("save__rmrt_reduced_dynamics_serialized!: basis must be :exciton, :site, or :both")
        end
    end

    return save_filename
end

calc__exciton_energy!(context::RmrtContext)                             = calc__ϵ_exci!(context)
calc__line_broadening_functions!(context::RmrtContext)                  = calc__g_g′_g″!(context)
calc__line_broadening_functions_with_threads!(context::RmrtContext)     = calc__g_g′_g″_with_threads!(context)
calc__reduced_density_matrix!(context::RmrtContext)                     = calc__σ_σ′!(context)

end 

