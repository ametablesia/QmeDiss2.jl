
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

# 매크로로 타입 정의
@patternized Patternized_g (n_sys::Int, n_itr::Int) (t::Int, a::Int, b::Int, c::Int, d::Int) begin
    rule(αααα, Matrix{T},   zeros(T, n_itr, n_sys), (t, a), a == b && b == c && c == d)
    rule(ααββ, Array{T,3},  zeros(T, n_itr, n_sys, n_sys), (t, a, c), a == b && c == d && a != c)
end

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

    # 이건 PL0Q U QL1P 에 의해서 추가된 것들.
    rule(αααβ, Array{T,3}, zeros(T, n_itr, n_sys, n_sys), (t, a, d), a == b && b == c && c != d)
    rule(ααβα, Array{T,3}, zeros(T, n_itr, n_sys, n_sys), (t, a, c), a == b && b == d && a != c)
    rule(ααβγ, Array{T,4}, zeros(T, n_itr, n_sys, n_sys, n_sys), (t, a, c, d), a == b && a != c && a != d && c != d)
    rule(αβββ, Array{T,3}, zeros(T, n_itr, n_sys, n_sys), (t, a, b), b == c && c == d && a != b)
    rule(αβαα, Array{T,3}, zeros(T, n_itr, n_sys, n_sys), (t, a, b), a == c && c == d && a != b)
    rule(αβγγ, Array{T,4}, zeros(T, n_itr, n_sys, n_sys, n_sys), (t, a, b, c), c == d && a != b && a != c && b != c)
end

# Lambda는 g′의 long time limit 임 (정확히는 ig′(infty))
@patternized Patternized_Λ (n_sys::Int) (a::Int, b::Int, c::Int, d::Int) begin
    rule(αααα, Vector{T},   zeros(T, n_sys),                   (a,),       a == b && b == c && c == d)
    rule(αααβ, Matrix{T},   zeros(T, n_sys, n_sys),            (a, d),     a == b && b == c && a != d)
    rule(ααβα, Matrix{T},   zeros(T, n_sys, n_sys),            (a, c),     a == b && a == d && a != c)
    rule(ααββ, Matrix{T},   zeros(T, n_sys, n_sys),            (a, c),     a == b && c == d && a != c)
    rule(ααβγ, Array{T,3},  zeros(T, n_sys, n_sys, n_sys),     (a, c, d),  a == b && a != c && a != d && c != d)
    rule(αβαα, Matrix{T},   zeros(T, n_sys, n_sys),            (a, b),     a == c && c == d && a != b)
    rule(αβββ, Matrix{T},   zeros(T, n_sys, n_sys),            (a, b),     b == c && c == d && a != b)
    rule(αβγγ, Array{T,3},  zeros(T, n_sys, n_sys, n_sys),     (a, b, c),  c == d && a != b && a != c && b != c)
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
    Λ                   ::Patternized_Λ{ComplexF64}

    # Reduced Density Matrix and its Time-derivatives
    curr_itr            ::UInt64
    σ                   ::Array{ComplexF64, 3}
    σ′                  ::Array{ComplexF64, 3}

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

        # Long time limit of g′
        Λ       = Patternized_Λ{ComplexF64}(n_sys)

        # Reduced density matrix and its time-derivatives
        σ       = zeros(ComplexF64, n_sys, n_sys, n_itr)
        σ′      = zeros(ComplexF64, n_sys, n_sys, n_itr)

        new(system, environment, simulation_details, γ_exci, ϵ_exci, ϵ_exci_0, U_sys, g, g′, g″, Λ, 1, σ, σ′)
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

function zero__Λ!(Λ::Patternized_Λ{T}) where {T}
    z = zero(T)

    fill!(Λ.αααα, z)
    fill!(Λ.αααβ, z)
    fill!(Λ.ααβα, z)
    fill!(Λ.ααββ, z)
    fill!(Λ.ααβγ, z)
    fill!(Λ.αβαα, z)
    fill!(Λ.αβββ, z)
    fill!(Λ.αβγγ, z)

    return nothing
end

@inline function accumulate__Λ__one_oscillator!(
    Λ       ::Patternized_Λ{ComplexF64},
    γ_exci  ::Array{ComplexF64, 3},
    osc_idx ::Int,
    ω       ::Float64,
    n_sys   ::Int,
)
    γ = @view γ_exci[osc_idx, :, :]

    @inbounds for α in 1:n_sys
        γ_αα = γ[α, α]

        # αααα
        Λ[α, α, α, α] += ω * γ_αα * γ_αα

        for α⁻ in 1:n_sys
            α⁻ == α && continue

            γ_αα⁻ = γ[α, α⁻]
            γ_α⁻α = γ[α⁻, α]
            γ_α⁻α⁻ = γ[α⁻, α⁻]

            # αααβ : Λ_{αααα⁻}
            Λ[α, α, α, α⁻] += ω * γ_αα * γ_αα⁻

            # ααβα : Λ_{ααα⁻α}
            Λ[α, α, α⁻, α] += ω * γ_αα * γ_α⁻α

            # ααββ : Λ_{ααα⁻α⁻}
            Λ[α, α, α⁻, α⁻] += ω * γ_αα * γ_α⁻α⁻

            # αβαα : Λ_{αα⁻αα}
            Λ[α, α⁻, α, α] += ω * γ_αα⁻ * γ_αα

            # αβββ : Λ_{αα⁻α⁻α⁻}
            Λ[α, α⁻, α⁻, α⁻] += ω * γ_αα⁻ * γ_α⁻α⁻

            for α⁼ in 1:n_sys
                (α⁼ == α || α⁼ == α⁻) && continue

                # ααβγ : Λ_{ααα⁻α⁼}
                Λ[α, α, α⁻, α⁼] += ω * γ_αα * γ[α⁻, α⁼]

                # αβγγ : Λ_{αα⁻α⁼α⁼}
                Λ[α, α⁻, α⁼, α⁼] += ω * γ_αα⁻ * γ[α⁼, α⁼]
            end
        end
    end

    return nothing
end

function calc__Λ!(context::RmrtContext)
    n_sys  = context.system.n_sys
    n_osc  = context.environment.num_of_effective_oscillators
    oscs   = context.environment.effective_oscillators

    Λ      = context.Λ
    γ_exci = context.γ_exci

    zero__Λ!(Λ)

    @inbounds for osc_idx in 1:n_osc
        ω = oscs[osc_idx].freq

        accumulate__Λ__one_oscillator!(
            Λ,
            γ_exci,
            osc_idx,
            ω,
            n_sys,
        )
    end

    return Λ
end

function calc__Λ_with_threads!(context::RmrtContext)
    n_sys  = context.system.n_sys
    n_osc  = context.environment.num_of_effective_oscillators
    oscs   = context.environment.effective_oscillators

    Λ      = context.Λ
    γ_exci = context.γ_exci

    n_ths = Threads.maxthreadid()

    Λ_locals = [
        Patternized_Λ{ComplexF64}(n_sys)
        for _ in 1:n_ths
    ]

    zero__Λ!(Λ)

    for tid in 1:n_ths
        zero__Λ!(Λ_locals[tid])
    end

    @inbounds Threads.@threads for osc_idx in 1:n_osc
        tid = Threads.threadid()

        ω = oscs[osc_idx].freq

        accumulate__Λ__one_oscillator!(
            Λ_locals[tid],
            γ_exci,
            osc_idx,
            ω,
            n_sys,
        )
    end

    for tid in 1:n_ths
        inplace_add!(Λ, Λ_locals[tid])
    end

    return Λ
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
    # PL0Q U QL1P를 위한
    fill!(g″.αααβ, z)
    fill!(g″.ααβα, z)
    fill!(g″.ααβγ, z)
    fill!(g″.αβββ, z)
    fill!(g″.αβαα, z)
    fill!(g″.αβγγ, z)

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

        # ---------------------------------------------------------------------
        # Additional g″ patterns needed by PL0Q exp(ΔL0) QL1P
        #   αααβ, ααβα, ααβγ, αβββ, αβαα, αβγγ
        # ---------------------------------------------------------------------
        for α in 1:n_sys
            γ_αα = γ[α, α]

            for β in 1:n_sys
                β == α && continue

                # αααβ : g″_{α α α β}
                g″[time_idx, α, α, α, β] += γ_αα * γ[α, β] * F_g″

                # ααβα : g″_{α α β α}
                g″[time_idx, α, α, β, α] += γ_αα * γ[β, α] * F_g″

                # αβββ : g″_{α β β β}
                g″[time_idx, α, β, β, β] += γ[α, β] * γ[β, β] * F_g″

                # αβαα : g″_{α β α α}
                g″[time_idx, α, β, α, α] += γ[α, β] * γ_αα * F_g″

                for γ_idx in 1:n_sys
                    (γ_idx == α || γ_idx == β) && continue

                    # ααβγ : g″_{α α β γ}
                    g″[time_idx, α, α, β, γ_idx] += γ_αα * γ[β, γ_idx] * F_g″

                    # αβγγ : g″_{α β γ γ}
                    g″[time_idx, α, β, γ_idx, γ_idx] += γ[α, β] * γ[γ_idx, γ_idx] * F_g″
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


### 원래 수정 전 수식이 맞을 지도 모릅니다....?? 음 phase shift 된건가요?
function calc__σ_σ′_secular_core!(
    context::RmrtContext;
    use_population_closure::Bool = false,
    use_local_population_to_coherence::Bool = true,
    use_L0Q_memory_return::Bool = true,
    verify_L0Q_terms::Bool = false,
    verify_L0Q_every::Int = 1,
    verify_L0Q_pair::Union{Nothing,Tuple{Int,Int}} = nothing,
    verify_L0Q_io::IO = stderr,
    verify_L0Q_header::Bool = true,
    use_secular::Bool = false,
    secular_tol::Float64 = 1.0e-10,
    method::Union{Symbol,String} = :rk4,
    use_threads::Bool = true,
    verbose::Bool = true,
    
)
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

    method_sym = Symbol(lowercase(String(method)))
    method_sym in (:euler, :rk2, :rk4) || error(
        "Unsupported integration method: $(method). " *
        "Use :euler, :rk2, or :rk4."
    )
    verify_L0Q_every >= 1 || error("verify_L0Q_every must be >= 1")

    start_itr < n_itr || return @view σ[:, :, n_itr]

    @inline ω(a::Int, b::Int) = ϵ[a] - ϵ[b]

    @inline function is_secular_pair(
        out_a::Int,
        out_b::Int,
        in_a::Int,
        in_b::Int,
    )
        if !use_secular
            return true
        end

        return abs(ω(out_a, out_b) - ω(in_a, in_b)) <= secular_tol
    end

    @inline function is_local_population_to_coherence(
        out_a::Int,
        out_b::Int,
        in_a::Int,
        in_b::Int,
    )
        return out_a != out_b && in_a == in_b
    end

    @inline function ∫weight(s_itr::Int, curr_itr::Int)
        return (s_itr == 1 || s_itr == curr_itr) ? 0.5 * Δt : Δt
    end

    @inline function phase(a::Int, b::Int, Δ_itr::Int)
        Δ = (Δ_itr - 1) * Δt
        return exp(-1.0im * ω(a, b) * Δ)
    end

    # -------------------------------------------------------------------------
    # Verification diagnostics for the first-order coherence-source package.
    # These diagnostics are intentionally separated from the applied RHS.
    # They report the raw, secular-filtered local PL1P pop -> coh source and
    # the raw, secular-filtered PL0Q exp(ΔL0) QL1P return source.
    # If use_local_population_to_coherence = false, the applied RHS may skip
    # these pop -> coh sources, but the verification values still show what
    # would have been present before that switch removes them.
    # -------------------------------------------------------------------------

    verify_header_printed = Ref(false)
    verify_lock = ReentrantLock()

    @inline function should_verify_L0Q(curr_itr::Int, α::Int, β::Int)
        if !verify_L0Q_terms || α == β
            return false
        end
        ((curr_itr - start_itr) % verify_L0Q_every == 0) || return false
        verify_L0Q_pair === nothing && return true
        return verify_L0Q_pair == (α, β)
    end

    function print_verify_L0Q_header_if_needed!()
        if verify_L0Q_header && !verify_header_printed[]
            lock(verify_lock)
            try
                if !verify_header_printed[]
                    println(
                        verify_L0Q_io,
                        "# L0Q_VERIFY columns: itr t α β " *
                        "local_A_pop local_B_pop local_pop " *
                        "L0Q_A_pop L0Q_B_pop L0Q_pop " *
                        "local_A_plus_L0Q_A_pop local_B_plus_L0Q_B_pop local_plus_L0Q_pop " *
                        "L0Q_A_all L0Q_B_all L0Q_all rhs"
                    )
                    verify_header_printed[] = true
                end
            finally
                unlock(verify_lock)
            end
        end
        return nothing
    end

    @inline function fmt__real(x)
        return Printf.@sprintf("%+.12e", Float64(real(x)))
    end

    @inline function fmt__time(x)
        return Printf.@sprintf("%.10e", Float64(x))
    end

    @inline function fmt__complex(z)
        return "(" * fmt__real(real(z)) * "," * fmt__real(imag(z)) * ")"
    end

    @inline function print__L0Q_verify_line!(
        io::IO,
        curr_itr::Int,
        t::Float64,
        α::Int,
        β::Int,
        local_A_pop,
        local_B_pop,
        local_pop,
        L0Q_A_pop,
        L0Q_B_pop,
        L0Q_pop,
        local_A_plus_L0Q_A_pop,
        local_B_plus_L0Q_B_pop,
        local_plus_L0Q_pop,
        L0Q_A_all,
        L0Q_B_all,
        L0Q_all,
        rhs,
    )
        println(
            io,
            "L0Q_VERIFY",
            " itr=", curr_itr,
            " t=", fmt__time(t),
            " a=", α,
            " b=", β,
            " local_A_pop=", fmt__complex(local_A_pop),
            " local_B_pop=", fmt__complex(local_B_pop),
            " local_pop=", fmt__complex(local_pop),
            " L0Q_A_pop=", fmt__complex(L0Q_A_pop),
            " L0Q_B_pop=", fmt__complex(L0Q_B_pop),
            " L0Q_pop=", fmt__complex(L0Q_pop),
            " local_A_plus_L0Q_A_pop=", fmt__complex(local_A_plus_L0Q_A_pop),
            " local_B_plus_L0Q_B_pop=", fmt__complex(local_B_plus_L0Q_B_pop),
            " local_plus_L0Q_pop=", fmt__complex(local_plus_L0Q_pop),
            " L0Q_A_all=", fmt__complex(L0Q_A_all),
            " L0Q_B_all=", fmt__complex(L0Q_B_all),
            " L0Q_all=", fmt__complex(L0Q_all),
            " rhs=", fmt__complex(rhs),
        )
        return nothing
    end

    @inline function add_if_finite(z::Complex)
        return (isfinite(real(z)) && isfinite(imag(z))) ? z : (NaN + NaN * im)
    end

    @inline function gen__exponent_type_1(
        s_itr::Int,
        Δ_itr::Int,
        t_itr::Int,
        α⁻::Int,
        α⁼::Int,
        β::Int,
    )
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

    @inline function gen__exponent_type_2(
        s_itr::Int,
        Δ_itr::Int,
        t_itr::Int,
        a::Int,
        β⁼::Int,
        β⁻::Int,
    )
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
        s_itr::Int,
        Δ_itr::Int,
        t_itr::Int,
        α::Int,
        α⁻::Int,
        α⁼::Int,
        β::Int,
    )
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
        s_itr::Int,
        Δ_itr::Int,
        t_itr::Int,
        β::Int,
        β⁻::Int,
        β⁼::Int,
        α::Int,
    )
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
        s_itr::Int,
        Δ_itr::Int,
        t_itr::Int,
        α::Int,
        β::Int,
        α⁻::Int,
        β⁻::Int,
    )
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
        s_itr::Int,
        Δ_itr::Int,
        t_itr::Int,
        α::Int,
        β::Int,
        α⁻::Int,
        β⁻::Int,
    )
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
    # Missing first-order memory-return term
    #   P L0 Q exp(Δ L0) Q L1 P
    # Time-localized, g-closed form.
    # External output indices: (α, β)
    # Branch A: input (f, β) -> output (α, β), coefficient sign = -
    # Branch B: input (α, f) -> output (α, β), coefficient sign = +
    # -------------------------------------------------------------------------

    @inline function gen__L0Q_exponent_A(
        s_itr::Int,
        Δ_itr::Int,
        t_itr::Int,
        α::Int,
        β::Int,
        f::Int,
    )
        # Transported time-localization for Branch A:
        #   σ[f,β](s) ≈ exp(+iω[f,β]Δ) * N[f,β](s)/N[f,β](t) * σ[f,β](t)
        # Therefore the normalization ratio N[f,β](t)/N[f,β](s) cancels,
        # and only the Λ_A[f, α | β](t,s) prefactor remains here.
        return (
            +(g[t_itr, f, f, f, f] - g[s_itr, f, f, f, f])
            -g[Δ_itr, α, α, α, α]
            -(g[t_itr, α, α, f, f] - g[s_itr, α, α, f, f] - g[Δ_itr, α, α, f, f])
            -g[Δ_itr, β, β, f, f]
            +conj(g[s_itr, f, f, β, β] - g[t_itr, f, f, β, β])
            +g[Δ_itr, β, β, α, α]
            +conj(g[t_itr, α, α, β, β] - g[s_itr, α, α, β, β])
        )
    end

    @inline function gen__L0Q_coef_A(
        s_itr::Int,
        Δ_itr::Int,
        t_itr::Int,
        α::Int,
        β::Int,
        f::Int,
    )
        left_one_point = (
             g′[Δ_itr, α, α, f, f]
            -g′[t_itr, α, α, f, f]
            -g′[Δ_itr, α, α, α, α]
            +g′[t_itr, α, α, α, α]
            -g′[Δ_itr, β, β, f, f]
            +g′[t_itr, β, β, f, f]
            +g′[Δ_itr, β, β, α, α]
            -g′[t_itr, β, β, α, α]
        )

        right_one_point = (
            -g′[s_itr, α, f, f, f]
            -g′[Δ_itr, α, α, α, f]
            +conj(g′[s_itr, f, α, β, β])
            +g′[Δ_itr, β, β, α, f]
        )

        return (
            g″[Δ_itr, α, α, α, f]
            -g″[Δ_itr, β, β, α, f]
            -left_one_point * right_one_point
        )
    end

    @inline function gen__L0Q_exponent_B(
        s_itr::Int,
        Δ_itr::Int,
        t_itr::Int,
        α::Int,
        β::Int,
        f::Int,
    )
        # Transported time-localization for Branch B:
        #   σ[α,f](s) ≈ exp(+iω[α,f]Δ) * N[α,f](s)/N[α,f](t) * σ[α,f](t)
        # Therefore the normalization ratio N[α,f](t)/N[α,f](s) cancels,
        # and only the corrected Λ_B[α | f, β](t,s) prefactor remains here.
        return (
            +conj(g[t_itr, f, f, f, f] - g[s_itr, f, f, f, f])
            -conj(g[Δ_itr, β, β, β, β])
            -conj(g[t_itr, β, β, f, f] - g[s_itr, β, β, f, f] - g[Δ_itr, β, β, f, f])
            -(g[t_itr, f, f, α, α] - g[s_itr, f, f, α, α])
            +(g[t_itr, β, β, α, α] - g[s_itr, β, β, α, α])
            -conj(g[Δ_itr, α, α, f, f])
            +conj(g[Δ_itr, α, α, β, β])
        )
    end

    @inline function gen__L0Q_coef_B(
        s_itr::Int,
        Δ_itr::Int,
        t_itr::Int,
        α::Int,
        β::Int,
        f::Int,
    )
        left_one_point = (
            conj(
                g′[t_itr, α, α, f, f]
                -g′[Δ_itr, α, α, f, f]
                +g′[Δ_itr, α, α, β, β]
                -g′[t_itr, α, α, β, β]
            )
            -conj(
                g′[t_itr, β, β, f, f]
                -g′[Δ_itr, β, β, f, f]
                +g′[Δ_itr, β, β, β, β]
                -g′[t_itr, β, β, β, β]
            )
        )

        right_one_point = (
            -g′[s_itr, f, β, α, α]
            -conj(g′[Δ_itr, α, α, β, f])
            +conj(g′[s_itr, β, f, f, f])
            +conj(g′[Δ_itr, β, β, β, f])
        )

        return (
            conj(g″[Δ_itr, β, f, α, α])
            -conj(g″[Δ_itr, β, f, β, β])
            -left_one_point * right_one_point
        )
    end

    # -------------------------------------------------------------------------
    # Population-closed path kernel
    # Used only when use_population_closure = true.
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
            * exp(gen__population_transfer_exponent(
                s_itr,
                Δ_itr,
                t_itr,
                src,
                dst,
            ))
            * gen__population_transfer_coef(
                s_itr,
                Δ_itr,
                t_itr,
                src,
                dst,
            )
        )
    end

    @inline function calc__population_closed_rhs(
        α::Int,
        curr_itr::Int,
        σ_t::AbstractMatrix,
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
    # RHS evaluator
    # -------------------------------------------------------------------------
    # The memory kernel, g/g′/g″, and secular filters are evaluated at curr_itr.
    # For RK stages, only the trial density matrix σ_t is changed.
    # This avoids requiring half-step interpolation of precomputed g-arrays.
    # -------------------------------------------------------------------------

    @inline function calc__rhs_element!(
        rhs_mat::AbstractMatrix,
        curr_itr::Int,
        σ_t::AbstractMatrix,
        α::Int,
        β::Int,
    )

        # -----------------------------------------------------------------
        # Optional population closure.
        # If disabled, populations are propagated by the secularized
        # full RMRT memory equation below.
        # -----------------------------------------------------------------
        if use_population_closure && α == β
            rhs_pop, pop_loss, pop_gain = calc__population_closed_rhs(
                α,
                curr_itr,
                σ_t,
            )

            rhs_mat[α, α] = rhs_pop
            return nothing
        end

        # -----------------------------------------------------------------
        # t-local diagonal / coherence phase block.
        # This is self-coupling, so it is always secular.
        # For α == β this block algebraically vanishes.
        # -----------------------------------------------------------------
        rhs = (
            -1.0im * ω(α, β)
            -g′[curr_itr, α, α, α, α]
            +conj(g′[curr_itr, α, α, β, β])
            +g′[curr_itr, β, β, α, α]
            -conj(g′[curr_itr, β, β, β, β])
        ) * σ_t[α, β]

        verify_this = should_verify_L0Q(curr_itr, α, β)
        verify_local_A_pop_to_coh = 0.0 + 0.0im
        verify_local_B_pop_to_coh = 0.0 + 0.0im
        verify_local_pop_to_coh   = 0.0 + 0.0im
        verify_L0Q_A_pop          = 0.0 + 0.0im
        verify_L0Q_B_pop          = 0.0 + 0.0im
        verify_L0Q_A_all          = 0.0 + 0.0im
        verify_L0Q_B_all          = 0.0 + 0.0im

        # -----------------------------------------------------------------
        # t-local left mixing:
        # input component  (α⁻, β)
        # output component (α, β)
        # keep if ω(α,β) ≈ ω(α⁻,β).
        #
        # If use_local_population_to_coherence = false, remove only the
        # local population -> coherence source σ[β,β] -> σ[α,β], i.e.
        # output α != β and input α⁻ == β.
        # -----------------------------------------------------------------
        for α⁻ in 1:n_sys
            α⁻ == α && continue

            if is_secular_pair(α, β, α⁻, β)
                local_left_term = σ_t[α⁻, β] * (
                    g′[curr_itr, α, α⁻, α⁻, α⁻]
                    - conj(g′[curr_itr, α⁻, α, β, β])
                )

                if verify_this && is_local_population_to_coherence(α, β, α⁻, β)
                    verify_local_A_pop_to_coh -= local_left_term
                    verify_local_pop_to_coh   -= local_left_term
                end

                if !use_local_population_to_coherence &&
                   is_local_population_to_coherence(α, β, α⁻, β)
                    continue
                end

                rhs -= local_left_term
            end
        end

        # -----------------------------------------------------------------
        # t-local right mixing:
        # input component  (α, β⁻)
        # output component (α, β)
        # keep if ω(α,β) ≈ ω(α,β⁻).
        #
        # If use_local_population_to_coherence = false, remove only the
        # local population -> coherence source σ[α,α] -> σ[α,β], i.e.
        # output α != β and input β⁻ == α.
        # -----------------------------------------------------------------
        for β⁻ in 1:n_sys
            β⁻ == β && continue

            if is_secular_pair(α, β, α, β⁻)
                local_right_term = σ_t[α, β⁻] * (
                    g′[curr_itr, β⁻, β, α, α]
                    - conj(g′[curr_itr, β, β⁻, β⁻, β⁻])
                )

                if verify_this && is_local_population_to_coherence(α, β, α, β⁻)
                    verify_local_B_pop_to_coh += local_right_term
                    verify_local_pop_to_coh   += local_right_term
                end

                if !use_local_population_to_coherence &&
                   is_local_population_to_coherence(α, β, α, β⁻)
                    continue
                end

                rhs += local_right_term
            end
        end

        # -----------------------------------------------------------------
        # Memory integral
        # -----------------------------------------------------------------
        if curr_itr > 1
            integral = 0.0 + 0.0im

            for s_itr in 1:curr_itr
                Δ_itr = curr_itr - s_itr + 1
                w_int = ∫weight(s_itr, curr_itr)
                kernel = 0.0 + 0.0im

                # ---------------------------------------------------------
                # Branch 1
                #
                # input component  (α⁼, β)
                # output component (α, β)
                # keep if ω(α,β) ≈ ω(α⁼,β).
                # ---------------------------------------------------------
                for α⁻ in 1:n_sys
                    α⁻ == α && continue

                    for α⁼ in 1:n_sys
                        α⁼ == α⁻ && continue

                        if is_secular_pair(α, β, α⁼, β)
                            kernel -= (
                                σ_t[α⁼, β]
                                * phase(α⁻, α⁼, Δ_itr)
                                * exp(gen__exponent_type_1(
                                    s_itr,
                                    Δ_itr,
                                    curr_itr,
                                    α⁻,
                                    α⁼,
                                    β,
                                ))
                                * gen_coef_block_type_1(
                                    s_itr,
                                    Δ_itr,
                                    curr_itr,
                                    α,
                                    α⁻,
                                    α⁼,
                                    β,
                                )
                            )
                        end
                    end
                end

                # ---------------------------------------------------------
                # Branch 2
                #
                # input component  (α, β⁼)
                # output component (α, β)
                # keep if ω(α,β) ≈ ω(α,β⁼).
                # ---------------------------------------------------------
                for β⁻ in 1:n_sys
                    β⁻ == β && continue

                    for β⁼ in 1:n_sys
                        β⁼ == β⁻ && continue

                        if is_secular_pair(α, β, α, β⁼)
                            kernel -= (
                                σ_t[α, β⁼]
                                * phase(β⁼, β⁻, Δ_itr)
                                * exp(gen__exponent_type_2(
                                    s_itr,
                                    Δ_itr,
                                    curr_itr,
                                    α,
                                    β⁼,
                                    β⁻,
                                ))
                                * gen_coef_block_type_2(
                                    s_itr,
                                    Δ_itr,
                                    curr_itr,
                                    β,
                                    β⁻,
                                    β⁼,
                                    α,
                                )
                            )
                        end
                    end
                end

                # ---------------------------------------------------------
                # Branch 3
                #
                # input component  (α⁻, β⁻)
                # output component (α, β)
                # keep if ω(α,β) ≈ ω(α⁻,β⁻).
                # ---------------------------------------------------------
                for α⁻ in 1:n_sys
                    α⁻ == α && continue

                    for β⁻ in 1:n_sys
                        β⁻ == β && continue

                        if is_secular_pair(α, β, α⁻, β⁻)
                            kernel += (
                                σ_t[α⁻, β⁻]
                                * phase(α, α⁻, Δ_itr)
                                * exp(gen__exponent_type_1(
                                    s_itr,
                                    Δ_itr,
                                    curr_itr,
                                    α,
                                    α⁻,
                                    β⁻,
                                ))
                                * gen_coef_block_type_3(
                                    s_itr,
                                    Δ_itr,
                                    curr_itr,
                                    α,
                                    β,
                                    α⁻,
                                    β⁻,
                                )
                            )
                        end
                    end
                end

                # ---------------------------------------------------------
                # Branch 4
                #
                # input component  (α⁻, β⁻)
                # output component (α, β)
                # keep if ω(α,β) ≈ ω(α⁻,β⁻).
                # ---------------------------------------------------------
                for α⁻ in 1:n_sys
                    α⁻ == α && continue

                    for β⁻ in 1:n_sys
                        β⁻ == β && continue

                        if is_secular_pair(α, β, α⁻, β⁻)
                            kernel += (
                                σ_t[α⁻, β⁻]
                                * phase(β⁻, β, Δ_itr)
                                * exp(gen__exponent_type_2(
                                    s_itr,
                                    Δ_itr,
                                    curr_itr,
                                    α⁻,
                                    β⁻,
                                    β,
                                ))
                                * gen_coef_block_type_4(
                                    s_itr,
                                    Δ_itr,
                                    curr_itr,
                                    α,
                                    β,
                                    α⁻,
                                    β⁻,
                                )
                            )
                        end
                    end
                end

                # ---------------------------------------------------------
                # Missing first-order memory-return term
                #   P L0 Q exp(Δ L0) Q L1 P
                # This term vanishes for population output (α == β), but is
                # generally needed for coherence output.
                #
                # Transported time-localization is used here:
                #   Branch A: σ[f,β](s) -> exp(+iω[f,β]Δ) N[f,β](s)/N[f,β](t) σ[f,β](t)
                #   Branch B: σ[α,f](s) -> exp(+iω[α,f]Δ) N[α,f](s)/N[α,f](t) σ[α,f](t)
                # Hence N-ratios cancel the dressed-trace N(t)/N(s) factors,
                # and the scalar phases become phase(α,f,Δ) and phase(f,β,Δ).
                # ---------------------------------------------------------
                if (use_L0Q_memory_return || verify_this) && α != β
                    # Branch A: input (f, β) -> output (α, β), sign = -
                    for f in 1:n_sys
                        f == α && continue

                        if is_secular_pair(α, β, f, β)
                            L0Q_A_term = (
                                σ_t[f, β]
                                * phase(α, f, Δ_itr)
                                * exp(gen__L0Q_exponent_A(
                                    s_itr,
                                    Δ_itr,
                                    curr_itr,
                                    α,
                                    β,
                                    f,
                                ))
                                * gen__L0Q_coef_A(
                                    s_itr,
                                    Δ_itr,
                                    curr_itr,
                                    α,
                                    β,
                                    f,
                                )
                            )
                            L0Q_A_contrib = -L0Q_A_term

                            if verify_this
                                verify_L0Q_A_all += w_int * L0Q_A_contrib
                                if is_local_population_to_coherence(α, β, f, β)
                                    verify_L0Q_A_pop += w_int * L0Q_A_contrib
                                end
                            end

                            if !use_local_population_to_coherence &&
                               is_local_population_to_coherence(α, β, f, β)
                                continue
                            end

                            if use_L0Q_memory_return
                                kernel += L0Q_A_contrib
                            end
                        end
                    end

                    # Branch B: input (α, f) -> output (α, β), sign = +
                    for f in 1:n_sys
                        f == β && continue

                        if is_secular_pair(α, β, α, f)
                            L0Q_B_contrib = (
                                σ_t[α, f]
                                * phase(f, β, Δ_itr)
                                * exp(gen__L0Q_exponent_B(
                                    s_itr,
                                    Δ_itr,
                                    curr_itr,
                                    α,
                                    β,
                                    f,
                                ))
                                * gen__L0Q_coef_B(
                                    s_itr,
                                    Δ_itr,
                                    curr_itr,
                                    α,
                                    β,
                                    f,
                                )
                            )

                            if verify_this
                                verify_L0Q_B_all += w_int * L0Q_B_contrib
                                if is_local_population_to_coherence(α, β, α, f)
                                    verify_L0Q_B_pop += w_int * L0Q_B_contrib
                                end
                            end

                            if !use_local_population_to_coherence &&
                               is_local_population_to_coherence(α, β, α, f)
                                continue
                            end

                            if use_L0Q_memory_return
                                kernel += L0Q_B_contrib
                            end
                        end
                    end
                end


                integral += w_int * kernel
            end

            rhs += integral
        end

        if verify_this
            L0Q_pop = verify_L0Q_A_pop + verify_L0Q_B_pop
            L0Q_all = verify_L0Q_A_all + verify_L0Q_B_all
            local_A_plus_L0Q_A_pop = verify_local_A_pop_to_coh + verify_L0Q_A_pop
            local_B_plus_L0Q_B_pop = verify_local_B_pop_to_coh + verify_L0Q_B_pop
            local_plus_L0Q_pop = verify_local_pop_to_coh + L0Q_pop
            time = (curr_itr - 1) * Δt

            print_verify_L0Q_header_if_needed!()
            lock(verify_lock)
            try
                print__L0Q_verify_line!(
                    verify_L0Q_io,
                    curr_itr,
                    time,
                    α,
                    β,
                    verify_local_A_pop_to_coh,
                    verify_local_B_pop_to_coh,
                    verify_local_pop_to_coh,
                    verify_L0Q_A_pop,
                    verify_L0Q_B_pop,
                    L0Q_pop,
                    local_A_plus_L0Q_A_pop,
                    local_B_plus_L0Q_B_pop,
                    local_plus_L0Q_pop,
                    verify_L0Q_A_all,
                    verify_L0Q_B_all,
                    L0Q_all,
                    rhs,
                )
            finally
                unlock(verify_lock)
            end
        end

        rhs_mat[α, β] = rhs

        return nothing
    end

    @inline function calc__rhs!(
        rhs_mat::AbstractMatrix,
        curr_itr::Int,
        σ_t::AbstractMatrix,
    )
        fill!(rhs_mat, 0.0 + 0.0im)

        # ---------------------------------------------------------------------
        # Main loop
        # ---------------------------------------------------------------------
        # curr_itr must remain sequential because σ[:,:,curr_itr+1] depends on
        # previous time slices. The independent output components (α, β) of the
        # RHS can be computed safely in parallel because each thread writes to a
        # distinct rhs_mat[α, β] entry and only reads g/g′/g″ and σ_t.
        # ---------------------------------------------------------------------
        n_components = n_sys * n_sys

        if use_threads && Threads.nthreads() > 1 && n_components > 1
            Threads.@threads for linear_idx in 1:n_components
                @inbounds begin
                    α = ((linear_idx - 1) % n_sys) + 1
                    β = ((linear_idx - 1) ÷ n_sys) + 1

                    calc__rhs_element!(
                        rhs_mat,
                        curr_itr,
                        σ_t,
                        α,
                        β,
                    )
                end
            end
        else
            @inbounds for β in 1:n_sys, α in 1:n_sys
                calc__rhs_element!(
                    rhs_mat,
                    curr_itr,
                    σ_t,
                    α,
                    β,
                )
            end
        end

        return rhs_mat
    end

    @inline function enforce_hermiticity!(
        σ_next::AbstractMatrix,
    )
        for i in 1:n_sys
            σ_next[i, i] = real(σ_next[i, i]) + 0.0im
        end

        for i in 1:n_sys-1
            for j in i+1:n_sys
                c = 0.5 * (
                    σ_next[i, j]
                    +
                    conj(σ_next[j, i])
                )

                σ_next[i, j] = c
                σ_next[j, i] = conj(c)
            end
        end

        return σ_next
    end

    # -------------------------------------------------------------------------
    # RK work buffers
    # -------------------------------------------------------------------------

    σ_stage = Matrix{ComplexF64}(undef, n_sys, n_sys)
    k2      = Matrix{ComplexF64}(undef, n_sys, n_sys)
    k3      = Matrix{ComplexF64}(undef, n_sys, n_sys)
    k4      = Matrix{ComplexF64}(undef, n_sys, n_sys)

    # -------------------------------------------------------------------------
    # Time propagation loop
    # -------------------------------------------------------------------------

    @inbounds for curr_itr in start_itr:(n_itr - 1)

        if verbose
            @printf(
                stderr,
                "Current iteration: %6d / %6d  method=%s  threads=%d  use_threads=%s  local_pop_to_coh=%s  L0Q_return=%s  verify_L0Q=%s\n",
                curr_itr,
                n_itr,
                String(method_sym),
                Threads.nthreads(),
                string(use_threads),
                string(use_local_population_to_coherence),
                string(use_L0Q_memory_return),
                string(verify_L0Q_terms),
            )
        end

        σ_t    = @view σ[:, :, curr_itr]
        σ_next = @view σ[:, :, curr_itr + 1]
        k1     = @view σ′[:, :, curr_itr]

        if method_sym == :euler
            calc__rhs!(k1, curr_itr, σ_t)

            @. σ_next = σ_t + Δt * k1

        elseif method_sym == :rk2
            calc__rhs!(k1, curr_itr, σ_t)

            @. σ_stage = σ_t + 0.5 * Δt * k1
            calc__rhs!(k2, curr_itr, σ_stage)

            @. σ_next = σ_t + Δt * k2

        elseif method_sym == :rk4
            calc__rhs!(k1, curr_itr, σ_t)

            @. σ_stage = σ_t + 0.5 * Δt * k1
            calc__rhs!(k2, curr_itr, σ_stage)

            @. σ_stage = σ_t + 0.5 * Δt * k2
            calc__rhs!(k3, curr_itr, σ_stage)

            @. σ_stage = σ_t + Δt * k3
            calc__rhs!(k4, curr_itr, σ_stage)

            @. σ_next = σ_t + (Δt / 6.0) * (
                k1 + 2.0 * k2 + 2.0 * k3 + k4
            )
        end

        enforce_hermiticity!(σ_next)

        context.curr_itr = UInt64(curr_itr + 1)
    end

    return @view σ[:, :, Int(context.curr_itr)]
end


### MARKOVIAN 전용
    # -------------------------------------------------------------------------
    # Markovian endpoint identities
    #
    #   g′_{abcd}(∞)       = -i Λ_{abcd}
    #   conj(g′_{abcd}(∞)) = +i conj(Λ_{abcd})
    #
    #   g_{abcd}(t) - g_{abcd}(t - Δ)       ≈ -i Λ_{abcd} Δ
    #   conj(g_{abcd}(t) - g_{abcd}(t - Δ)) ≈ +i conj(Λ_{abcd}) Δ
    # -------------------------------------------------------------------------
###

function calc__markovian_generator!(
    R::AbstractMatrix{ComplexF64},
    context::RmrtContext;
    use_population_closure::Bool = false,
    use_local_population_to_coherence::Bool = true,
    use_secular::Bool = false,
    secular_tol::Float64 = 1.0e-10,
    recompute_Λ::Bool = true,
    markovian_max_itr::Union{Nothing,Int} = nothing,
    use_threads::Bool = true,
    verbose::Bool = true,
)
    n_sys = context.system.n_sys
    n_itr = context.simulation_details.num_of_iteration
    Δt    = context.simulation_details.Δt
    ϵ     = context.ϵ_exci

    g  = context.g
    g′ = context.g′
    g″ = context.g″
    Λ  = context.Λ

    n_vec = n_sys * n_sys

    size(R, 1) == n_vec && size(R, 2) == n_vec || error(
        "R must have size ($(n_vec), $(n_vec)); got $(size(R))."
    )

    if recompute_Λ
        calc__Λ!(context)
    end

    fill!(R, 0.0 + 0.0im)

    Δ_max_itr = isnothing(markovian_max_itr) ? n_itr : min(Int(markovian_max_itr), n_itr)
    Δ_max_itr >= 2 || error("markovian_max_itr must be at least 2.")

    if verbose
        @printf(
            stderr,
            "Building Markovian generator: n_sys=%d  n_vec=%d  Δ_max_itr=%d  use_threads=%s\n",
            n_sys,
            n_vec,
            Δ_max_itr,
            string(use_threads),
        )
    end

    @inline vecidx(a::Int, b::Int) = a + (b - 1) * n_sys

    @inline ω(a::Int, b::Int) = ϵ[a] - ϵ[b]

    @inline function is_secular_pair(
        out_a::Int,
        out_b::Int,
        in_a::Int,
        in_b::Int,
    )
        if !use_secular
            return true
        end

        return abs(ω(out_a, out_b) - ω(in_a, in_b)) <= secular_tol
    end

    @inline function is_local_population_to_coherence(
        out_a::Int,
        out_b::Int,
        in_a::Int,
        in_b::Int,
    )
        return out_a != out_b && in_a == in_b
    end

    @inline function ∫Δweight(Δ_itr::Int)
        return (Δ_itr == 1 || Δ_itr == Δ_max_itr) ? 0.5 * Δt : Δt
    end

    @inline function Δtime(Δ_itr::Int)
        return (Δ_itr - 1) * Δt
    end

    @inline function phase(a::Int, b::Int, Δ_itr::Int)
        return exp(-1.0im * ω(a, b) * Δtime(Δ_itr))
    end

    # -------------------------------------------------------------------------
    # Markovian endpoint identities
    # -------------------------------------------------------------------------

    @inline function gprime_inf(a::Int, b::Int, c::Int, d::Int)
        return -1.0im * Λ[a, b, c, d]
    end

    @inline function conj_gprime_inf(a::Int, b::Int, c::Int, d::Int)
        return 1.0im * conj(Λ[a, b, c, d])
    end

    @inline function gdiff_inf(
        a::Int,
        b::Int,
        c::Int,
        d::Int,
        Δ::Float64,
    )
        return -1.0im * Λ[a, b, c, d] * Δ
    end

    @inline function conj_gdiff_inf(
        a::Int,
        b::Int,
        c::Int,
        d::Int,
        Δ::Float64,
    )
        return 1.0im * conj(Λ[a, b, c, d]) * Δ
    end

    # -------------------------------------------------------------------------
    # Markovianized exponent blocks.
    # They depend only on Δ, not on s_itr, t_itr, or curr_itr.
    # -------------------------------------------------------------------------

    @inline function gen__markovian_exponent_type_1(
        Δ_itr::Int,
        α⁻::Int,
        α⁼::Int,
        β::Int,
    )
        Δ = Δtime(Δ_itr)

        return (
            gdiff_inf(α⁼, α⁼, α⁼, α⁼, Δ)
            -gdiff_inf(α⁻, α⁻, α⁼, α⁼, Δ)
            -conj_gdiff_inf(α⁼, α⁼, β, β, Δ)
            +conj_gdiff_inf(α⁻, α⁻, β, β, Δ)

            -g[Δ_itr, α⁻, α⁻, α⁻, α⁻]
            +g[Δ_itr, α⁻, α⁻, α⁼, α⁼]
            -g[Δ_itr, β, β, α⁼, α⁼]
            +g[Δ_itr, β, β, α⁻, α⁻]
        )
    end

    @inline function gen__markovian_exponent_type_2(
        Δ_itr::Int,
        a::Int,
        β⁼::Int,
        β⁻::Int,
    )
        Δ = Δtime(Δ_itr)

        return (
            conj_gdiff_inf(β⁼, β⁼, β⁼, β⁼, Δ)
            -conj_gdiff_inf(β⁻, β⁻, β⁼, β⁼, Δ)
            -gdiff_inf(β⁼, β⁼, a, a, Δ)
            +gdiff_inf(β⁻, β⁻, a, a, Δ)

            -conj(g[Δ_itr, β⁻, β⁻, β⁻, β⁻])
            +conj(g[Δ_itr, β⁻, β⁻, β⁼, β⁼])
            -conj(g[Δ_itr, a, a, β⁼, β⁼])
            +conj(g[Δ_itr, a, a, β⁻, β⁻])
        )
    end

    @inline function gen_coef_block_type_1_markovian(
        Δ_itr::Int,
        α::Int,
        α⁻::Int,
        α⁼::Int,
        β::Int,
    )
        left_one_point = (
             g′[Δ_itr, α, α⁻, α⁼, α⁼]
            -g′[Δ_itr, α, α⁻, α⁻, α⁻]
            -gprime_inf(α, α⁻, α⁼, α⁼)
            +gprime_inf(α, α⁻, α⁻, α⁻)
        )

        right_one_point = (
            -gprime_inf(α⁻, α⁼, α⁼, α⁼)
            +conj_gprime_inf(α⁼, α⁻, β, β)
            -g′[Δ_itr, α⁻, α⁻, α⁻, α⁼]
            +g′[Δ_itr, β, β, α⁻, α⁼]
        )

        return g″[Δ_itr, α, α⁻, α⁻, α⁼] - left_one_point * right_one_point
    end

    @inline function gen_coef_block_type_2_markovian(
        Δ_itr::Int,
        β::Int,
        β⁻::Int,
        β⁼::Int,
        α::Int,
    )
        left_one_point = (
            -conj(g′[Δ_itr, β, β⁻, β⁼, β⁼])
            +conj(g′[Δ_itr, β, β⁻, β⁻, β⁻])
            +conj_gprime_inf(β, β⁻, β⁼, β⁼)
            -conj_gprime_inf(β, β⁻, β⁻, β⁻)
        )

        right_one_point = (
            -gprime_inf(β⁼, β⁻, α, α)
            +conj_gprime_inf(β⁻, β⁼, β⁼, β⁼)
            -conj(g′[Δ_itr, α, α, β⁻, β⁼])
            +conj(g′[Δ_itr, β⁻, β⁻, β⁻, β⁼])
        )

        return conj(g″[Δ_itr, β, β⁻, β⁻, β⁼]) - left_one_point * right_one_point
    end

    @inline function gen_coef_block_type_3_markovian(
        Δ_itr::Int,
        α::Int,
        β::Int,
        α⁻::Int,
        β⁻::Int,
    )
        left_one_point = (
            -gprime_inf(α, α⁻, α⁻, α⁻)
            +conj_gprime_inf(α⁻, α, β⁻, β⁻)
            -g′[Δ_itr, α, α, α, α⁻]
            +g′[Δ_itr, β⁻, β⁻, α, α⁻]
        )

        right_one_point = (
            -g′[Δ_itr, β⁻, β, α, α]
            +g′[Δ_itr, β⁻, β, α⁻, α⁻]
            +gprime_inf(β⁻, β, α, α)
            -gprime_inf(β⁻, β, α⁻, α⁻)
        )

        return g″[Δ_itr, β⁻, β, α, α⁻] - left_one_point * right_one_point
    end

    @inline function gen_coef_block_type_4_markovian(
        Δ_itr::Int,
        α::Int,
        β::Int,
        α⁻::Int,
        β⁻::Int,
    )
        left_one_point = (
            -conj(g′[Δ_itr, α⁻, α, β⁻, β⁻])
            +conj(g′[Δ_itr, α⁻, α, β, β])
            +conj_gprime_inf(α⁻, α, β⁻, β⁻)
            -conj_gprime_inf(α⁻, α, β, β)
        )

        right_one_point = (
            -gprime_inf(β⁻, β, α⁻, α⁻)
            +conj_gprime_inf(β, β⁻, β⁻, β⁻)
            -conj(g′[Δ_itr, α⁻, α⁻, β, β⁻])
            +conj(g′[Δ_itr, β, β, β, β⁻])
        )

        return conj(g″[Δ_itr, α⁻, α, β, β⁻]) - left_one_point * right_one_point
    end

    # -------------------------------------------------------------------------
    # Population-closed path kernel.
    # -------------------------------------------------------------------------

    @inline function gen__population_transfer_exponent_markovian(
        Δ_itr::Int,
        src::Int,
        dst::Int,
    )
        Δ = Δtime(Δ_itr)

        return (
            gdiff_inf(src, src, src, src, Δ)

            -g[Δ_itr, dst, dst, dst, dst]

            -gdiff_inf(dst, dst, src, src, Δ)
            +g[Δ_itr, dst, dst, src, src]

            -g[Δ_itr, src, src, src, src]

            -conj_gdiff_inf(src, src, src, src, Δ)

            +g[Δ_itr, src, src, dst, dst]

            +conj_gdiff_inf(dst, dst, src, src, Δ)
        )
    end

    @inline function gen__population_transfer_coef_markovian(
        Δ_itr::Int,
        src::Int,
        dst::Int,
    )
        left_one_point = (
            -1.0im * gprime_inf(src, dst, src, src)
            +1.0im * g′[Δ_itr, src, dst, src, src]
            -1.0im * g′[Δ_itr, src, dst, dst, dst]
            +1.0im * conj_gprime_inf(dst, src, src, src)
        )

        right_one_point = (
            -1.0im * gprime_inf(dst, src, src, src)
            -1.0im * g′[Δ_itr, dst, dst, dst, src]
            +1.0im * conj_gprime_inf(src, dst, src, src)
            +1.0im * g′[Δ_itr, src, src, dst, src]
        )

        return (
            g″[Δ_itr, src, dst, dst, src]
            + left_one_point * right_one_point
        )
    end

    @inline function gen__population_transfer_kernel_markovian(
        Δ_itr::Int,
        src::Int,
        dst::Int,
    )
        return 2.0 * real(
            phase(dst, src, Δ_itr)
            * exp(gen__population_transfer_exponent_markovian(
                Δ_itr,
                src,
                dst,
            ))
            * gen__population_transfer_coef_markovian(
                Δ_itr,
                src,
                dst,
            )
        )
    end

    # -------------------------------------------------------------------------
    # Fill one output row of R.
    # Each output component (α, β) writes only to one row R[out, :], so this is
    # safe to thread over output components.
    # -------------------------------------------------------------------------

    @inline function add_generator_row!(
        α::Int,
        β::Int,
    )
        out = vecidx(α, β)

        # ---------------------------------------------------------------------
        # Optional population closure: Pauli gain-loss row.
        # ---------------------------------------------------------------------
        if use_population_closure && α == β
            for Δ_itr in 1:Δ_max_itr
                w_int = ∫Δweight(Δ_itr)

                for f in 1:n_sys
                    f == α && continue

                    k_loss = gen__population_transfer_kernel_markovian(
                        Δ_itr,
                        α,
                        f,
                    )

                    k_gain = gen__population_transfer_kernel_markovian(
                        Δ_itr,
                        f,
                        α,
                    )

                    R[out, vecidx(α, α)] -= w_int * k_loss
                    R[out, vecidx(f, f)] += w_int * k_gain
                end
            end

            return nothing
        end

        # ---------------------------------------------------------------------
        # Markovian local diagonal / coherence phase block.
        # ---------------------------------------------------------------------
        R[out, vecidx(α, β)] += (
            -1.0im * ω(α, β)
            -gprime_inf(α, α, α, α)
            +conj_gprime_inf(α, α, β, β)
            +gprime_inf(β, β, α, α)
            -conj_gprime_inf(β, β, β, β)
        )

        # ---------------------------------------------------------------------
        # Markovian local left mixing.
        # ---------------------------------------------------------------------
        for α⁻ in 1:n_sys
            α⁻ == α && continue

            if !use_local_population_to_coherence &&
               is_local_population_to_coherence(α, β, α⁻, β)
                continue
            end

            if is_secular_pair(α, β, α⁻, β)
                R[out, vecidx(α⁻, β)] -= (
                    gprime_inf(α, α⁻, α⁻, α⁻)
                    -conj_gprime_inf(α⁻, α, β, β)
                )
            end
        end

        # ---------------------------------------------------------------------
        # Markovian local right mixing.
        # ---------------------------------------------------------------------
        for β⁻ in 1:n_sys
            β⁻ == β && continue

            if !use_local_population_to_coherence &&
               is_local_population_to_coherence(α, β, α, β⁻)
                continue
            end

            if is_secular_pair(α, β, α, β⁻)
                R[out, vecidx(α, β⁻)] += (
                    gprime_inf(β⁻, β, α, α)
                    -conj_gprime_inf(β, β⁻, β⁻, β⁻)
                )
            end
        end

        # ---------------------------------------------------------------------
        # Markovian memory contribution: integrate over Δ once.
        # ---------------------------------------------------------------------
        for Δ_itr in 1:Δ_max_itr
            w_int = ∫Δweight(Δ_itr)

            # -----------------------------------------------------------------
            # Branch 1
            #
            # input component  (α⁼, β)
            # output component (α, β)
            # -----------------------------------------------------------------
            for α⁻ in 1:n_sys
                α⁻ == α && continue

                for α⁼ in 1:n_sys
                    α⁼ == α⁻ && continue

                    if is_secular_pair(α, β, α⁼, β)
                        factor = (
                            phase(α⁻, α⁼, Δ_itr)
                            * exp(gen__markovian_exponent_type_1(
                                Δ_itr,
                                α⁻,
                                α⁼,
                                β,
                            ))
                            * gen_coef_block_type_1_markovian(
                                Δ_itr,
                                α,
                                α⁻,
                                α⁼,
                                β,
                            )
                        )

                        R[out, vecidx(α⁼, β)] -= w_int * factor
                    end
                end
            end

            # -----------------------------------------------------------------
            # Branch 2
            #
            # input component  (α, β⁼)
            # output component (α, β)
            # -----------------------------------------------------------------
            for β⁻ in 1:n_sys
                β⁻ == β && continue

                for β⁼ in 1:n_sys
                    β⁼ == β⁻ && continue

                    if is_secular_pair(α, β, α, β⁼)
                        factor = (
                            phase(β⁼, β⁻, Δ_itr)
                            * exp(gen__markovian_exponent_type_2(
                                Δ_itr,
                                α,
                                β⁼,
                                β⁻,
                            ))
                            * gen_coef_block_type_2_markovian(
                                Δ_itr,
                                β,
                                β⁻,
                                β⁼,
                                α,
                            )
                        )

                        R[out, vecidx(α, β⁼)] -= w_int * factor
                    end
                end
            end

            # -----------------------------------------------------------------
            # Branch 3
            #
            # input component  (α⁻, β⁻)
            # output component (α, β)
            # -----------------------------------------------------------------
            for α⁻ in 1:n_sys
                α⁻ == α && continue

                for β⁻ in 1:n_sys
                    β⁻ == β && continue

                    if is_secular_pair(α, β, α⁻, β⁻)
                        factor = (
                            phase(α, α⁻, Δ_itr)
                            * exp(gen__markovian_exponent_type_1(
                                Δ_itr,
                                α,
                                α⁻,
                                β⁻,
                            ))
                            * gen_coef_block_type_3_markovian(
                                Δ_itr,
                                α,
                                β,
                                α⁻,
                                β⁻,
                            )
                        )

                        R[out, vecidx(α⁻, β⁻)] += w_int * factor
                    end
                end
            end

            # -----------------------------------------------------------------
            # Branch 4
            #
            # input component  (α⁻, β⁻)
            # output component (α, β)
            # -----------------------------------------------------------------
            for α⁻ in 1:n_sys
                α⁻ == α && continue

                for β⁻ in 1:n_sys
                    β⁻ == β && continue

                    if is_secular_pair(α, β, α⁻, β⁻)
                        factor = (
                            phase(β⁻, β, Δ_itr)
                            * exp(gen__markovian_exponent_type_2(
                                Δ_itr,
                                α⁻,
                                β⁻,
                                β,
                            ))
                            * gen_coef_block_type_4_markovian(
                                Δ_itr,
                                α,
                                β,
                                α⁻,
                                β⁻,
                            )
                        )

                        R[out, vecidx(α⁻, β⁻)] += w_int * factor
                    end
                end
            end
        end

        return nothing
    end

    n_components = n_sys * n_sys

    if use_threads && Threads.nthreads() > 1 && n_components > 1
        Threads.@threads for linear_idx in 1:n_components
            @inbounds begin
                α = ((linear_idx - 1) % n_sys) + 1
                β = ((linear_idx - 1) ÷ n_sys) + 1
                add_generator_row!(α, β)
            end
        end
    else
        @inbounds for β in 1:n_sys, α in 1:n_sys
            add_generator_row!(α, β)
        end
    end

    return R
end

function calc__markovian_generator(
    context::RmrtContext;
    use_population_closure::Bool = false,
    use_local_population_to_coherence::Bool = true,
    use_secular::Bool = false,
    secular_tol::Float64 = 1.0e-10,
    recompute_Λ::Bool = true,
    markovian_max_itr::Union{Nothing,Int} = nothing,
    use_threads::Bool = true,
    verbose::Bool = true,
)
    n_sys = context.system.n_sys
    R = zeros(ComplexF64, n_sys * n_sys, n_sys * n_sys)

    calc__markovian_generator!(
        R,
        context;
        use_population_closure = use_population_closure,
        use_local_population_to_coherence = use_local_population_to_coherence,
        use_secular = use_secular,
        secular_tol = secular_tol,
        recompute_Λ = recompute_Λ,
        markovian_max_itr = markovian_max_itr,
        use_threads = use_threads,
        verbose = verbose,
    )

    return R
end


function calc__σ_σ′_with_markovian_generator!(
    context::RmrtContext,
    R::AbstractMatrix{ComplexF64};
    method::Union{Symbol,String} = :rk4,
    verbose::Bool = false,
)
    start_itr = Int(context.curr_itr)

    n_sys = context.system.n_sys
    n_itr = context.simulation_details.num_of_iteration
    Δt    = context.simulation_details.Δt

    σ  = context.σ
    σ′ = context.σ′

    n_vec = n_sys * n_sys

    size(R, 1) == n_vec && size(R, 2) == n_vec || error(
        "R must have size ($(n_vec), $(n_vec)); got $(size(R))."
    )

    method_sym = Symbol(lowercase(String(method)))
    method_sym in (:euler, :rk2, :rk4) || error(
        "Unsupported integration method: $(method). " *
        "Use :euler, :rk2, or :rk4."
    )

    start_itr < n_itr || return @view σ[:, :, n_itr]

    σ_vec     = Vector{ComplexF64}(undef, n_vec)
    rhs_vec   = Vector{ComplexF64}(undef, n_vec)
    stage_vec = Vector{ComplexF64}(undef, n_vec)
    k2_vec    = Vector{ComplexF64}(undef, n_vec)
    k3_vec    = Vector{ComplexF64}(undef, n_vec)
    k4_vec    = Vector{ComplexF64}(undef, n_vec)

    @inline function enforce_hermiticity!(
        σ_next::AbstractMatrix,
    )
        for i in 1:n_sys
            σ_next[i, i] = real(σ_next[i, i]) + 0.0im
        end

        for i in 1:n_sys-1
            for j in i+1:n_sys
                c = 0.5 * (
                    σ_next[i, j]
                    +
                    conj(σ_next[j, i])
                )

                σ_next[i, j] = c
                σ_next[j, i] = conj(c)
            end
        end

        return σ_next
    end

    @inbounds for curr_itr in start_itr:(n_itr - 1)

        if verbose
            @printf(
                stderr,
                "Current iteration: %6d / %6d  method=%s  markovian_generator=true\n",
                curr_itr,
                n_itr,
                String(method_sym),
            )
        end

        σ_t    = @view σ[:, :, curr_itr]
        σ_next = @view σ[:, :, curr_itr + 1]
        k1_mat = @view σ′[:, :, curr_itr]

        copyto!(σ_vec, vec(σ_t))

        if method_sym == :euler
            mul!(rhs_vec, R, σ_vec)

            @. σ_vec = σ_vec + Δt * rhs_vec

        elseif method_sym == :rk2
            mul!(rhs_vec, R, σ_vec)

            @. stage_vec = σ_vec + 0.5 * Δt * rhs_vec
            mul!(k2_vec, R, stage_vec)

            @. σ_vec = σ_vec + Δt * k2_vec

        elseif method_sym == :rk4
            mul!(rhs_vec, R, σ_vec)

            @. stage_vec = σ_vec + 0.5 * Δt * rhs_vec
            mul!(k2_vec, R, stage_vec)

            @. stage_vec = σ_vec + 0.5 * Δt * k2_vec
            mul!(k3_vec, R, stage_vec)

            @. stage_vec = σ_vec + Δt * k3_vec
            mul!(k4_vec, R, stage_vec)

            @. σ_vec = σ_vec + (Δt / 6.0) * (
                rhs_vec + 2.0 * k2_vec + 2.0 * k3_vec + k4_vec
            )
        end

        copyto!(vec(σ_next), σ_vec)
        copyto!(vec(k1_mat), rhs_vec)

        enforce_hermiticity!(σ_next)

        context.curr_itr = UInt64(curr_itr + 1)
    end

    return @view σ[:, :, Int(context.curr_itr)]
end


function calc__σ_σ′_with_markovian!(
    context::RmrtContext;
    use_population_closure::Bool = false,
    use_local_population_to_coherence::Bool = true,
    use_secular::Bool = false,
    secular_tol::Float64 = 1.0e-10,
    method::Union{Symbol,String} = :rk4,
    use_threads::Bool = true,
    verbose::Bool = false,
    recompute_Λ::Bool = true,
    markovian_max_itr::Union{Nothing,Int} = nothing,
    return_generator::Bool = false,
)
    R = calc__markovian_generator(
        context;
        use_population_closure = use_population_closure,
        use_local_population_to_coherence = use_local_population_to_coherence,
        use_secular = use_secular,
        secular_tol = secular_tol,
        recompute_Λ = recompute_Λ,
        markovian_max_itr = markovian_max_itr,
        use_threads = use_threads,
        verbose = verbose,
    )

    result = calc__σ_σ′_with_markovian_generator!(
        context,
        R;
        method = method,
        verbose = verbose,
    )

    if return_generator
        return result, R
    end

    return result
end



# =============================================================================
# Wrapper 1:
# Secular RMRT with population closure
# =============================================================================
function calc__σ_σ′_with_population_closure_secular!(
    context::RmrtContext;
    use_secular::Bool = true,
    secular_tol::Float64 = 1.0e-10,
    verbose::Bool = false,
)
    return calc__σ_σ′_secular_core!(
        context;
        use_population_closure = true,
        use_secular = use_secular,
        secular_tol = secular_tol,
        verbose = verbose,
    )
end


# =============================================================================
# Wrapper 2:
# Secular RMRT without population closure
# =============================================================================
function calc__σ_σ′_without_population_closure_secular!(
    context::RmrtContext;
    use_secular::Bool = true,
    secular_tol::Float64 = 1.0e-10,
    verbose::Bool = false,
)
    return calc__σ_σ′_secular_core!(
        context;
        use_population_closure = false,
        use_secular = use_secular,
        secular_tol = secular_tol,
        verbose = verbose,
    )
end


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
calc__reorganization_energy!(context::RmrtContext)                      = calc__Λ!(context)
calc__reorganization_energy_with_threads!(context::RmrtContext)         = calc__Λ_with_threads!(context)
calc__line_broadening_functions!(context::RmrtContext)                  = calc__g_g′_g″!(context)
calc__line_broadening_functions_with_threads!(context::RmrtContext)     = calc__g_g′_g″_with_threads!(context)
calc__reduced_density_matrix!(context::RmrtContext)                     = calc__σ_σ′!(context)

end 

