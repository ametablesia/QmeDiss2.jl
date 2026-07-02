
module Rmrt

include("../Utils/HighDimensionalDataContainer.jl")
include("../Physics/Physics.jl")

using Base.Threads
using LinearAlgebra
import Base: getindex
using Printf

using HDF5
using Printf
using Dates

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
    # full grid
    g                   ::Patternized_g{ComplexF64}
    g′                  ::Patternized_g′{ComplexF64}
    g″                  ::Patternized_g″{ComplexF64}
    Λ                   ::Patternized_Λ{ComplexF64}

    # half-shifted grid for RK2 and RK4.
    #
    # These caches are intentionally lazy.  The constructor does not allocate
    # them.  Call ensure__half_shifted_grid!(context) or
    # calc__g_g′_g″_half_shifted!(context) before a non-Markovian RK2/RK4 run
    # that needs stage-time g-series values.
    using_half_shifted_grid ::Bool
    g_half_shifted      ::Union{Nothing,Patternized_g{ComplexF64}}
    g′_half_shifted     ::Union{Nothing,Patternized_g′{ComplexF64}}
    g″_half_shifted     ::Union{Nothing,Patternized_g″{ComplexF64}}

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

        new(system, environment, simulation_details, γ_exci, ϵ_exci, ϵ_exci_0, U_sys, g, g′, g″, Λ, false, nothing, nothing, nothing, UInt64(1), σ, σ′)
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

    invalidate__half_shifted_grid!(context)

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

# =============================================================================
# HDF5 save utilities for g, g′, g″ grids
# =============================================================================

function _h5write_array!(
    parent,
    name::AbstractString,
    A::AbstractArray;
    split_complex::Bool = true,
)
    if split_complex && eltype(A) <: Complex
        parent[name * "__re"] = real.(A)
        parent[name * "__im"] = imag.(A)

        attrs(parent[name * "__re"])["complex_part"] = "real"
        attrs(parent[name * "__im"])["complex_part"] = "imag"
        attrs(parent[name * "__re"])["complex_name"] = name
        attrs(parent[name * "__im"])["complex_name"] = name
    else
        parent[name] = A
    end

    return nothing
end


function _h5write_patternized_container!(
    parent,
    group_name::AbstractString,
    container;
    split_complex::Bool = true,
)
    grp = create_group(parent, group_name)

    attrs(grp)["julia_type"] = string(typeof(container))

    for fname in fieldnames(typeof(container))
        value = getfield(container, fname)
        name = String(fname)

        if value isa AbstractArray
            _h5write_array!(
                grp,
                name,
                value;
                split_complex = split_complex,
            )

        elseif value isa Number || value isa AbstractString || value isa Bool
            attrs(grp)[name] = value

        elseif value === nothing
            attrs(grp)[name] = "__nothing__"

        else
            # 재구성용이 아니라 디버깅/추적용으로 repr만 저장
            attrs(grp)[name * "__repr"] = repr(value)
        end
    end

    return grp
end


function save__g_g′_g″__h5!(
    context::RmrtContext,
    h5_path::AbstractString;
    include_full_grid::Bool = true,
    include_half_shifted_grid::Bool = false,
    split_complex::Bool = true,
    overwrite::Bool = true,
)
    mode = overwrite ? "w" : "cw"

    n_sys = context.system.n_sys
    n_itr = context.simulation_details.num_of_iteration
    Δt    = context.simulation_details.Δt

    h5open(h5_path, mode) do h5
        attrs(h5)["created_at"] = string(now())
        attrs(h5)["format"] = "RMRT g-grid dump"
        attrs(h5)["n_sys"] = n_sys
        attrs(h5)["n_itr"] = n_itr
        attrs(h5)["Δt"] = Δt
        attrs(h5)["using_half_shifted_grid"] = context.using_half_shifted_grid

        meta = create_group(h5, "metadata")

        attrs(meta)["n_sys"] = n_sys
        attrs(meta)["n_itr"] = n_itr
        attrs(meta)["Δt"] = Δt
        attrs(meta)["num_of_effective_oscillators"] =
            context.environment.num_of_effective_oscillators

        osc_freq = [
            context.environment.effective_oscillators[i].freq
            for i in 1:context.environment.num_of_effective_oscillators
        ]

        osc_coth = [
            context.environment.effective_oscillators[i].coth
            for i in 1:context.environment.num_of_effective_oscillators
        ]

        meta["osc_freq"] = osc_freq
        meta["osc_coth"] = osc_coth

        # γ_exci도 같이 저장해두면 나중에 g 재현/검산이 쉬움
        if hasproperty(context, :γ_exci)
            _h5write_array!(
                meta,
                "gamma_exci",
                context.γ_exci;
                split_complex = split_complex,
            )
        end

        if include_full_grid
            full = create_group(h5, "full_grid")
            attrs(full)["time_convention"] = "t = (itr - 1) * Δt"

            full["time"] = collect(0:(n_itr - 1)) .* Δt

            _h5write_patternized_container!(
                full,
                "g",
                context.g;
                split_complex = split_complex,
            )

            _h5write_patternized_container!(
                full,
                "g_prime",
                context.g′;
                split_complex = split_complex,
            )

            _h5write_patternized_container!(
                full,
                "g_dprime",
                context.g″;
                split_complex = split_complex,
            )
        end

        if include_half_shifted_grid
            if !has__half_shifted_grid(context)
                error("Cannot save half-shifted g-grid because it has not been allocated.")
            end

            half = create_group(h5, "half_shifted_grid")
            attrs(half)["time_convention"] = "t = (itr - 1/2) * Δt"

            half["time"] = (collect(0:(n_itr - 1)) .+ 0.5) .* Δt

            _h5write_patternized_container!(
                half,
                "g",
                context.g_half_shifted;
                split_complex = split_complex,
            )

            _h5write_patternized_container!(
                half,
                "g_prime",
                context.g′_half_shifted;
                split_complex = split_complex,
            )

            _h5write_patternized_container!(
                half,
                "g_dprime",
                context.g″_half_shifted;
                split_complex = split_complex,
            )
        end
    end

    @printf(stderr, "Saved g-grid HDF5 dump: %s\n", h5_path)

    return h5_path
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
    Δt      ::Float64;
    time_shift::Float64 = 0.0,
)
    γ = @view γ_exci[osc_idx, :, :]

    @inbounds for time_idx in 1:n_itr
        t   = (time_idx - 1) * Δt + time_shift
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
        #   αααα, αααβ, ααβα, ααββ, ααβγ, αβαα, αβββ, αβγγ
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


function calc__g_g′_g″!(
    context::RmrtContext;

    save_h5::Bool = false,
    h5_path::AbstractString = "rmrt_g_grid.h5",
    overwrite_h5::Bool = true,
    split_complex::Bool = true,
)
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

    context.using_half_shifted_grid = false

    ## 저장할거면 하시던가!
    if save_h5
        save__g_g′_g″__h5!(
            context,
            h5_path;
            include_full_grid = true,
            include_half_shifted_grid = false,
            overwrite = overwrite_h5,
            split_complex = split_complex,
        )
    end

    return g, g′, g″
end

function calc__g_g′_g″_with_threads!(
    context::RmrtContext;
    save_h5::Bool = true,
    h5_path::AbstractString = "rmrt_g_grid.h5",
    overwrite_h5::Bool = true,
    split_complex::Bool = true,
)
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

    context.using_half_shifted_grid = false

    # 저장할거면 하시던가.
    if save_h5
        save__g_g′_g″__h5!(
            context,
            h5_path;
            include_full_grid = true,
            include_half_shifted_grid = false,
            overwrite = overwrite_h5,
            split_complex = split_complex,
        )
    end

    return g, g′, g″
end

# =============================================================================
# Lazy half-shifted g-grid management
# =============================================================================
#
# Full grid convention:
#   context.g[itr, ...]              = g((itr - 1) * Δt)
#
# Half-shifted grid convention:
#   context.g_half_shifted[itr, ...] = g((itr - 1/2) * Δt)
#
# The half-shifted containers are not allocated by the RmrtContext constructor.
# Allocate/compute them only when RK2/RK4 stage-time non-Markovian RHS
# evaluation needs them.

@inline function has__half_shifted_grid(context::RmrtContext)
    return (
        context.g_half_shifted !== nothing &&
        context.g′_half_shifted !== nothing &&
        context.g″_half_shifted !== nothing
    )
end

function invalidate__half_shifted_grid!(context::RmrtContext)
    context.using_half_shifted_grid = false
    return context
end

function free__half_shifted_grid!(context::RmrtContext)
    context.g_half_shifted  = nothing
    context.g′_half_shifted = nothing
    context.g″_half_shifted = nothing
    context.using_half_shifted_grid = false
    return context
end

function allocate__half_shifted_grid!(
    context::RmrtContext;
    force::Bool = false,
    verbose::Bool = false,
)
    n_sys = context.system.n_sys
    n_itr = context.simulation_details.num_of_iteration

    if force || !has__half_shifted_grid(context)
        if verbose
            @printf(
                stderr,
                "Allocating half-shifted g-grid: n_sys=%d  n_itr=%d\n",
                n_sys,
                n_itr,
            )
        end

        context.g_half_shifted  = Patternized_g{ComplexF64}(n_sys, n_itr)
        context.g′_half_shifted = Patternized_g′{ComplexF64}(n_sys, n_itr)
        context.g″_half_shifted = Patternized_g″{ComplexF64}(n_sys, n_itr)
        context.using_half_shifted_grid = false
    end

    return (
        context.g_half_shifted::Patternized_g{ComplexF64},
        context.g′_half_shifted::Patternized_g′{ComplexF64},
        context.g″_half_shifted::Patternized_g″{ComplexF64},
    )
end

function calc__g_g′_g″_half_shifted!(
    context::RmrtContext;
    force_allocate::Bool = false,
    verbose::Bool = true,
)
    n_sys   = context.system.n_sys
    n_osc   = context.environment.num_of_effective_oscillators
    oscs    = context.environment.effective_oscillators

    n_itr   = context.simulation_details.num_of_iteration
    Δt      = context.simulation_details.Δt

    g_h, g′_h, g″_h = allocate__half_shifted_grid!(
        context;
        force = force_allocate,
        verbose = verbose,
    )

    γ_exci  = context.γ_exci

    zero__g_g′_g″!(g_h, g′_h, g″_h)

    @inbounds for osc_idx in 1:n_osc
        ω       = oscs[osc_idx].freq
        coth    = oscs[osc_idx].coth

        accumulate__g_g′_g″__one_oscillator!(
            g_h,
            g′_h,
            g″_h,
            γ_exci,
            osc_idx,
            ω,
            coth,
            n_sys,
            n_itr,
            Δt;
            time_shift = 0.5 * Δt,
        )

        if verbose && (osc_idx - 1) % 100 == 0
            @printf(stderr, "HALF-SHIFTED OSC %6d / %6d\n", osc_idx, n_osc)
        end
    end

    context.using_half_shifted_grid = true

    return g_h, g′_h, g″_h
end

function calc__g_g′_g″_half_shifted_with_threads!(
    context::RmrtContext;
    force_allocate::Bool = false,
    verbose::Bool = true,
)
    n_sys   = context.system.n_sys
    n_osc   = context.environment.num_of_effective_oscillators
    oscs    = context.environment.effective_oscillators

    n_itr   = context.simulation_details.num_of_iteration
    Δt      = context.simulation_details.Δt

    g_h, g′_h, g″_h = allocate__half_shifted_grid!(
        context;
        force = force_allocate,
        verbose = verbose,
    )

    γ_exci  = context.γ_exci

    n_ths = Threads.maxthreadid()
    g_locals    = [Patternized_g{ComplexF64}(n_sys, n_itr)  for _ in 1:n_ths]
    g′_locals   = [Patternized_g′{ComplexF64}(n_sys, n_itr) for _ in 1:n_ths]
    g″_locals   = [Patternized_g″{ComplexF64}(n_sys, n_itr) for _ in 1:n_ths]

    zero__g_g′_g″!(g_h, g′_h, g″_h)

    for tid in 1:n_ths
        zero__g_g′_g″!(g_locals[tid], g′_locals[tid], g″_locals[tid])
    end

    @inbounds @threads for osc_idx in 1:n_osc
        ω       = oscs[osc_idx].freq
        coth    = oscs[osc_idx].coth
        tid     = threadid()

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
            Δt;
            time_shift = 0.5 * Δt,
        )

        if verbose && (osc_idx - 1) % 100 == 0
            @printf(stderr, "HALF-SHIFTED OSC %6d / %6d\n", osc_idx, n_osc)
        end
    end

    for tid in 1:n_ths
        inplace_add!(g_h,  g_locals[tid])
        inplace_add!(g′_h, g′_locals[tid])
        inplace_add!(g″_h, g″_locals[tid])
    end

    context.using_half_shifted_grid = true

    return g_h, g′_h, g″_h
end

function ensure__half_shifted_grid!(
    context::RmrtContext;
    recompute::Bool = false,
    use_threads::Bool = true,
    verbose::Bool = true,
)
    if !recompute && context.using_half_shifted_grid && has__half_shifted_grid(context)
        return (
            context.g_half_shifted::Patternized_g{ComplexF64},
            context.g′_half_shifted::Patternized_g′{ComplexF64},
            context.g″_half_shifted::Patternized_g″{ComplexF64},
        )
    end

    if use_threads && Threads.nthreads() > 1
        return calc__g_g′_g″_half_shifted_with_threads!(
            context;
            force_allocate = !has__half_shifted_grid(context),
            verbose = verbose,
        )
    end

    return calc__g_g′_g″_half_shifted!(
        context;
        force_allocate = !has__half_shifted_grid(context),
        verbose = verbose,
    )
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

# function calc__markovian_generator!(
#     R::AbstractMatrix{ComplexF64},
#     context::RmrtContext;
#     use_population_closure::Bool = false,
#     use_local_population_to_coherence::Bool = true,
#     use_secular::Bool = false,
#     secular_tol::Float64 = 1.0e-10,
#     recompute_Λ::Bool = true,
#     markovian_max_itr::Union{Nothing,Int} = nothing,
#     use_threads::Bool = true,
#     verbose::Bool = true,
# )
#     n_sys = context.system.n_sys
#     n_itr = context.simulation_details.num_of_iteration
#     Δt    = context.simulation_details.Δt
#     ϵ     = context.ϵ_exci

#     g  = context.g
#     g′ = context.g′
#     g″ = context.g″
#     Λ  = context.Λ

#     n_vec = n_sys * n_sys

#     size(R, 1) == n_vec && size(R, 2) == n_vec || error(
#         "R must have size ($(n_vec), $(n_vec)); got $(size(R))."
#     )

#     if recompute_Λ
#         calc__Λ!(context)
#     end

#     fill!(R, 0.0 + 0.0im)

#     Δ_max_itr = isnothing(markovian_max_itr) ? n_itr : min(Int(markovian_max_itr), n_itr)
#     Δ_max_itr >= 2 || error("markovian_max_itr must be at least 2.")

#     if verbose
#         @printf(
#             stderr,
#             "Building Markovian generator: n_sys=%d  n_vec=%d  Δ_max_itr=%d  use_threads=%s\n",
#             n_sys,
#             n_vec,
#             Δ_max_itr,
#             string(use_threads),
#         )
#     end

#     @inline vecidx(a::Int, b::Int) = a + (b - 1) * n_sys

#     @inline ω(a::Int, b::Int) = ϵ[a] - ϵ[b]

#     @inline function is_secular_pair(
#         out_a::Int,
#         out_b::Int,
#         in_a::Int,
#         in_b::Int,
#     )
#         if !use_secular
#             return true
#         end

#         return abs(ω(out_a, out_b) - ω(in_a, in_b)) <= secular_tol
#     end

#     @inline function is_local_population_to_coherence(
#         out_a::Int,
#         out_b::Int,
#         in_a::Int,
#         in_b::Int,
#     )
#         return out_a != out_b && in_a == in_b
#     end

#     @inline function ∫Δweight(Δ_itr::Int)
#         return (Δ_itr == 1 || Δ_itr == Δ_max_itr) ? 0.5 * Δt : Δt
#     end

#     @inline function Δtime(Δ_itr::Int)
#         return (Δ_itr - 1) * Δt
#     end

#     @inline function phase(a::Int, b::Int, Δ_itr::Int)
#         return exp(-1.0im * ω(a, b) * Δtime(Δ_itr))
#     end

#     # -------------------------------------------------------------------------
#     # Markovian endpoint identities
#     # -------------------------------------------------------------------------

#     @inline function gprime_inf(a::Int, b::Int, c::Int, d::Int)
#         return -1.0im * Λ[a, b, c, d]
#     end

#     @inline function conj_gprime_inf(a::Int, b::Int, c::Int, d::Int)
#         return 1.0im * conj(Λ[a, b, c, d])
#     end

#     @inline function gdiff_inf(
#         a::Int,
#         b::Int,
#         c::Int,
#         d::Int,
#         Δ::Float64,
#     )
#         return -1.0im * Λ[a, b, c, d] * Δ
#     end

#     @inline function conj_gdiff_inf(
#         a::Int,
#         b::Int,
#         c::Int,
#         d::Int,
#         Δ::Float64,
#     )
#         return 1.0im * conj(Λ[a, b, c, d]) * Δ
#     end

#     # -------------------------------------------------------------------------
#     # Markovianized exponent blocks.
#     # They depend only on Δ, not on s_itr, t_itr, or curr_itr.
#     # -------------------------------------------------------------------------

#     @inline function gen__markovian_exponent_type_1(
#         Δ_itr::Int,
#         α⁻::Int,
#         α⁼::Int,
#         β::Int,
#     )
#         Δ = Δtime(Δ_itr)

#         return (
#             gdiff_inf(α⁼, α⁼, α⁼, α⁼, Δ)
#             -gdiff_inf(α⁻, α⁻, α⁼, α⁼, Δ)
#             -conj_gdiff_inf(α⁼, α⁼, β, β, Δ)
#             +conj_gdiff_inf(α⁻, α⁻, β, β, Δ)

#             -g[Δ_itr, α⁻, α⁻, α⁻, α⁻]
#             +g[Δ_itr, α⁻, α⁻, α⁼, α⁼]
#             -g[Δ_itr, β, β, α⁼, α⁼]
#             +g[Δ_itr, β, β, α⁻, α⁻]
#         )
#     end

#     @inline function gen__markovian_exponent_type_2(
#         Δ_itr::Int,
#         a::Int,
#         β⁼::Int,
#         β⁻::Int,
#     )
#         Δ = Δtime(Δ_itr)

#         return (
#             conj_gdiff_inf(β⁼, β⁼, β⁼, β⁼, Δ)
#             -conj_gdiff_inf(β⁻, β⁻, β⁼, β⁼, Δ)
#             -gdiff_inf(β⁼, β⁼, a, a, Δ)
#             +gdiff_inf(β⁻, β⁻, a, a, Δ)

#             -conj(g[Δ_itr, β⁻, β⁻, β⁻, β⁻])
#             +conj(g[Δ_itr, β⁻, β⁻, β⁼, β⁼])
#             -conj(g[Δ_itr, a, a, β⁼, β⁼])
#             +conj(g[Δ_itr, a, a, β⁻, β⁻])
#         )
#     end

#     @inline function gen_coef_block_type_1_markovian(
#         Δ_itr::Int,
#         α::Int,
#         α⁻::Int,
#         α⁼::Int,
#         β::Int,
#     )
#         left_one_point = (
#              g′[Δ_itr, α, α⁻, α⁼, α⁼]
#             -g′[Δ_itr, α, α⁻, α⁻, α⁻]
#             -gprime_inf(α, α⁻, α⁼, α⁼)
#             +gprime_inf(α, α⁻, α⁻, α⁻)
#         )

#         right_one_point = (
#             -gprime_inf(α⁻, α⁼, α⁼, α⁼)
#             +conj_gprime_inf(α⁼, α⁻, β, β)
#             -g′[Δ_itr, α⁻, α⁻, α⁻, α⁼]
#             +g′[Δ_itr, β, β, α⁻, α⁼]
#         )

#         return g″[Δ_itr, α, α⁻, α⁻, α⁼] - left_one_point * right_one_point
#     end

#     @inline function gen_coef_block_type_2_markovian(
#         Δ_itr::Int,
#         β::Int,
#         β⁻::Int,
#         β⁼::Int,
#         α::Int,
#     )
#         left_one_point = (
#             -conj(g′[Δ_itr, β, β⁻, β⁼, β⁼])
#             +conj(g′[Δ_itr, β, β⁻, β⁻, β⁻])
#             +conj_gprime_inf(β, β⁻, β⁼, β⁼)
#             -conj_gprime_inf(β, β⁻, β⁻, β⁻)
#         )

#         right_one_point = (
#             -gprime_inf(β⁼, β⁻, α, α)
#             +conj_gprime_inf(β⁻, β⁼, β⁼, β⁼)
#             -conj(g′[Δ_itr, α, α, β⁻, β⁼])
#             +conj(g′[Δ_itr, β⁻, β⁻, β⁻, β⁼])
#         )

#         return conj(g″[Δ_itr, β, β⁻, β⁻, β⁼]) - left_one_point * right_one_point
#     end

#     @inline function gen_coef_block_type_3_markovian(
#         Δ_itr::Int,
#         α::Int,
#         β::Int,
#         α⁻::Int,
#         β⁻::Int,
#     )
#         left_one_point = (
#             -gprime_inf(α, α⁻, α⁻, α⁻)
#             +conj_gprime_inf(α⁻, α, β⁻, β⁻)
#             -g′[Δ_itr, α, α, α, α⁻]
#             +g′[Δ_itr, β⁻, β⁻, α, α⁻]
#         )

#         right_one_point = (
#             -g′[Δ_itr, β⁻, β, α, α]
#             +g′[Δ_itr, β⁻, β, α⁻, α⁻]
#             +gprime_inf(β⁻, β, α, α)
#             -gprime_inf(β⁻, β, α⁻, α⁻)
#         )

#         return g″[Δ_itr, β⁻, β, α, α⁻] - left_one_point * right_one_point
#     end

#     @inline function gen_coef_block_type_4_markovian(
#         Δ_itr::Int,
#         α::Int,
#         β::Int,
#         α⁻::Int,
#         β⁻::Int,
#     )
#         left_one_point = (
#             -conj(g′[Δ_itr, α⁻, α, β⁻, β⁻])
#             +conj(g′[Δ_itr, α⁻, α, β, β])
#             +conj_gprime_inf(α⁻, α, β⁻, β⁻)
#             -conj_gprime_inf(α⁻, α, β, β)
#         )

#         right_one_point = (
#             -gprime_inf(β⁻, β, α⁻, α⁻)
#             +conj_gprime_inf(β, β⁻, β⁻, β⁻)
#             -conj(g′[Δ_itr, α⁻, α⁻, β, β⁻])
#             +conj(g′[Δ_itr, β, β, β, β⁻])
#         )

#         return conj(g″[Δ_itr, α⁻, α, β, β⁻]) - left_one_point * right_one_point
#     end

#     # -------------------------------------------------------------------------
#     # Population-closed path kernel.
#     # -------------------------------------------------------------------------

#     @inline function gen__population_transfer_exponent_markovian(
#         Δ_itr::Int,
#         src::Int,
#         dst::Int,
#     )
#         Δ = Δtime(Δ_itr)

#         return (
#             gdiff_inf(src, src, src, src, Δ)

#             -g[Δ_itr, dst, dst, dst, dst]

#             -gdiff_inf(dst, dst, src, src, Δ)
#             +g[Δ_itr, dst, dst, src, src]

#             -g[Δ_itr, src, src, src, src]

#             -conj_gdiff_inf(src, src, src, src, Δ)

#             +g[Δ_itr, src, src, dst, dst]

#             +conj_gdiff_inf(dst, dst, src, src, Δ)
#         )
#     end

#     @inline function gen__population_transfer_coef_markovian(
#         Δ_itr::Int,
#         src::Int,
#         dst::Int,
#     )
#         left_one_point = (
#             -1.0im * gprime_inf(src, dst, src, src)
#             +1.0im * g′[Δ_itr, src, dst, src, src]
#             -1.0im * g′[Δ_itr, src, dst, dst, dst]
#             +1.0im * conj_gprime_inf(dst, src, src, src)
#         )

#         right_one_point = (
#             -1.0im * gprime_inf(dst, src, src, src)
#             -1.0im * g′[Δ_itr, dst, dst, dst, src]
#             +1.0im * conj_gprime_inf(src, dst, src, src)
#             +1.0im * g′[Δ_itr, src, src, dst, src]
#         )

#         return (
#             g″[Δ_itr, src, dst, dst, src]
#             + left_one_point * right_one_point
#         )
#     end

#     @inline function gen__population_transfer_kernel_markovian(
#         Δ_itr::Int,
#         src::Int,
#         dst::Int,
#     )
#         return 2.0 * real(
#             phase(dst, src, Δ_itr)
#             * exp(gen__population_transfer_exponent_markovian(
#                 Δ_itr,
#                 src,
#                 dst,
#             ))
#             * gen__population_transfer_coef_markovian(
#                 Δ_itr,
#                 src,
#                 dst,
#             )
#         )
#     end

#     # -------------------------------------------------------------------------
#     # Fill one output row of R.
#     # Each output component (α, β) writes only to one row R[out, :], so this is
#     # safe to thread over output components.
#     # -------------------------------------------------------------------------

#     @inline function add_generator_row!(
#         α::Int,
#         β::Int,
#     )
#         out = vecidx(α, β)

#         # ---------------------------------------------------------------------
#         # Optional population closure: Pauli gain-loss row.
#         # ---------------------------------------------------------------------
#         if use_population_closure && α == β
#             for Δ_itr in 1:Δ_max_itr
#                 w_int = ∫Δweight(Δ_itr)

#                 for f in 1:n_sys
#                     f == α && continue

#                     k_loss = gen__population_transfer_kernel_markovian(
#                         Δ_itr,
#                         α,
#                         f,
#                     )

#                     k_gain = gen__population_transfer_kernel_markovian(
#                         Δ_itr,
#                         f,
#                         α,
#                     )

#                     R[out, vecidx(α, α)] -= w_int * k_loss
#                     R[out, vecidx(f, f)] += w_int * k_gain
#                 end
#             end

#             return nothing
#         end

#         # ---------------------------------------------------------------------
#         # Markovian local diagonal / coherence phase block.
#         # ---------------------------------------------------------------------
#         R[out, vecidx(α, β)] += (
#             -1.0im * ω(α, β)
#             -gprime_inf(α, α, α, α)
#             +conj_gprime_inf(α, α, β, β)
#             +gprime_inf(β, β, α, α)
#             -conj_gprime_inf(β, β, β, β)
#         )

#         # ---------------------------------------------------------------------
#         # Markovian local left mixing.
#         # ---------------------------------------------------------------------
#         for α⁻ in 1:n_sys
#             α⁻ == α && continue

#             if !use_local_population_to_coherence &&
#                is_local_population_to_coherence(α, β, α⁻, β)
#                 continue
#             end

#             if is_secular_pair(α, β, α⁻, β)
#                 R[out, vecidx(α⁻, β)] -= (
#                     gprime_inf(α, α⁻, α⁻, α⁻)
#                     -conj_gprime_inf(α⁻, α, β, β)
#                 )
#             end
#         end

#         # ---------------------------------------------------------------------
#         # Markovian local right mixing.
#         # ---------------------------------------------------------------------
#         for β⁻ in 1:n_sys
#             β⁻ == β && continue

#             if !use_local_population_to_coherence &&
#                is_local_population_to_coherence(α, β, α, β⁻)
#                 continue
#             end

#             if is_secular_pair(α, β, α, β⁻)
#                 R[out, vecidx(α, β⁻)] += (
#                     gprime_inf(β⁻, β, α, α)
#                     -conj_gprime_inf(β, β⁻, β⁻, β⁻)
#                 )
#             end
#         end

#         # ---------------------------------------------------------------------
#         # Markovian memory contribution: integrate over Δ once.
#         # ---------------------------------------------------------------------
#         for Δ_itr in 1:Δ_max_itr
#             w_int = ∫Δweight(Δ_itr)

#             # -----------------------------------------------------------------
#             # Branch 1
#             #
#             # input component  (α⁼, β)
#             # output component (α, β)
#             # -----------------------------------------------------------------
#             for α⁻ in 1:n_sys
#                 α⁻ == α && continue

#                 for α⁼ in 1:n_sys
#                     α⁼ == α⁻ && continue

#                     if is_secular_pair(α, β, α⁼, β)
#                         factor = (
#                             phase(α⁻, α⁼, Δ_itr)
#                             * exp(gen__markovian_exponent_type_1(
#                                 Δ_itr,
#                                 α⁻,
#                                 α⁼,
#                                 β,
#                             ))
#                             * gen_coef_block_type_1_markovian(
#                                 Δ_itr,
#                                 α,
#                                 α⁻,
#                                 α⁼,
#                                 β,
#                             )
#                         )

#                         R[out, vecidx(α⁼, β)] -= w_int * factor
#                     end
#                 end
#             end

#             # -----------------------------------------------------------------
#             # Branch 2
#             #
#             # input component  (α, β⁼)
#             # output component (α, β)
#             # -----------------------------------------------------------------
#             for β⁻ in 1:n_sys
#                 β⁻ == β && continue

#                 for β⁼ in 1:n_sys
#                     β⁼ == β⁻ && continue

#                     if is_secular_pair(α, β, α, β⁼)
#                         factor = (
#                             phase(β⁼, β⁻, Δ_itr)
#                             * exp(gen__markovian_exponent_type_2(
#                                 Δ_itr,
#                                 α,
#                                 β⁼,
#                                 β⁻,
#                             ))
#                             * gen_coef_block_type_2_markovian(
#                                 Δ_itr,
#                                 β,
#                                 β⁻,
#                                 β⁼,
#                                 α,
#                             )
#                         )

#                         R[out, vecidx(α, β⁼)] -= w_int * factor
#                     end
#                 end
#             end

#             # -----------------------------------------------------------------
#             # Branch 3
#             #
#             # input component  (α⁻, β⁻)
#             # output component (α, β)
#             # -----------------------------------------------------------------
#             for α⁻ in 1:n_sys
#                 α⁻ == α && continue

#                 for β⁻ in 1:n_sys
#                     β⁻ == β && continue

#                     if is_secular_pair(α, β, α⁻, β⁻)
#                         factor = (
#                             phase(α, α⁻, Δ_itr)
#                             * exp(gen__markovian_exponent_type_1(
#                                 Δ_itr,
#                                 α,
#                                 α⁻,
#                                 β⁻,
#                             ))
#                             * gen_coef_block_type_3_markovian(
#                                 Δ_itr,
#                                 α,
#                                 β,
#                                 α⁻,
#                                 β⁻,
#                             )
#                         )

#                         R[out, vecidx(α⁻, β⁻)] += w_int * factor
#                     end
#                 end
#             end

#             # -----------------------------------------------------------------
#             # Branch 4
#             #
#             # input component  (α⁻, β⁻)
#             # output component (α, β)
#             # -----------------------------------------------------------------
#             for α⁻ in 1:n_sys
#                 α⁻ == α && continue

#                 for β⁻ in 1:n_sys
#                     β⁻ == β && continue

#                     if is_secular_pair(α, β, α⁻, β⁻)
#                         factor = (
#                             phase(β⁻, β, Δ_itr)
#                             * exp(gen__markovian_exponent_type_2(
#                                 Δ_itr,
#                                 α⁻,
#                                 β⁻,
#                                 β,
#                             ))
#                             * gen_coef_block_type_4_markovian(
#                                 Δ_itr,
#                                 α,
#                                 β,
#                                 α⁻,
#                                 β⁻,
#                             )
#                         )

#                         R[out, vecidx(α⁻, β⁻)] += w_int * factor
#                     end
#                 end
#             end
#         end

#         return nothing
#     end

#     n_components = n_sys * n_sys

#     if use_threads && Threads.nthreads() > 1 && n_components > 1
#         Threads.@threads for linear_idx in 1:n_components
#             @inbounds begin
#                 α = ((linear_idx - 1) % n_sys) + 1
#                 β = ((linear_idx - 1) ÷ n_sys) + 1
#                 add_generator_row!(α, β)
#             end
#         end
#     else
#         @inbounds for β in 1:n_sys, α in 1:n_sys
#             add_generator_row!(α, β)
#         end
#     end

#     return R
# end

# function calc__markovian_generator(
#     context::RmrtContext;
#     use_population_closure::Bool = false,
#     use_local_population_to_coherence::Bool = true,
#     use_secular::Bool = false,
#     secular_tol::Float64 = 1.0e-10,
#     recompute_Λ::Bool = true,
#     markovian_max_itr::Union{Nothing,Int} = nothing,
#     use_threads::Bool = true,
#     verbose::Bool = true,
# )
#     n_sys = context.system.n_sys
#     R = zeros(ComplexF64, n_sys * n_sys, n_sys * n_sys)

#     calc__markovian_generator!(
#         R,
#         context;
#         use_population_closure = use_population_closure,
#         use_local_population_to_coherence = use_local_population_to_coherence,
#         use_secular = use_secular,
#         secular_tol = secular_tol,
#         recompute_Λ = recompute_Λ,
#         markovian_max_itr = markovian_max_itr,
#         use_threads = use_threads,
#         verbose = verbose,
#     )

#     return R
# end

# function calc__σ_σ′_with_markovian_generator!(
#     context::RmrtContext,
#     R::AbstractMatrix{ComplexF64};
#     method::Union{Symbol,String} = :rk4,
#     verbose::Bool = false,
# )
#     start_itr = Int(context.curr_itr)

#     n_sys = context.system.n_sys
#     n_itr = context.simulation_details.num_of_iteration
#     Δt    = context.simulation_details.Δt

#     σ  = context.σ
#     σ′ = context.σ′

#     n_vec = n_sys * n_sys

#     size(R, 1) == n_vec && size(R, 2) == n_vec || error(
#         "R must have size ($(n_vec), $(n_vec)); got $(size(R))."
#     )

#     method_sym = Symbol(lowercase(String(method)))
#     method_sym in (:euler, :rk2, :rk4) || error(
#         "Unsupported integration method: $(method). " *
#         "Use :euler, :rk2, or :rk4."
#     )

#     start_itr < n_itr || return @view σ[:, :, n_itr]

#     σ_vec     = Vector{ComplexF64}(undef, n_vec)
#     rhs_vec   = Vector{ComplexF64}(undef, n_vec)
#     stage_vec = Vector{ComplexF64}(undef, n_vec)
#     k2_vec    = Vector{ComplexF64}(undef, n_vec)
#     k3_vec    = Vector{ComplexF64}(undef, n_vec)
#     k4_vec    = Vector{ComplexF64}(undef, n_vec)

#     @inline function enforce_hermiticity!(
#         σ_next::AbstractMatrix,
#     )
#         for i in 1:n_sys
#             σ_next[i, i] = real(σ_next[i, i]) + 0.0im
#         end

#         for i in 1:n_sys-1
#             for j in i+1:n_sys
#                 c = 0.5 * (
#                     σ_next[i, j]
#                     +
#                     conj(σ_next[j, i])
#                 )

#                 σ_next[i, j] = c
#                 σ_next[j, i] = conj(c)
#             end
#         end

#         return σ_next
#     end

#     @inbounds for curr_itr in start_itr:(n_itr - 1)

#         if verbose
#             @printf(
#                 stderr,
#                 "Current iteration: %6d / %6d  method=%s  markovian_generator=true\n",
#                 curr_itr,
#                 n_itr,
#                 String(method_sym),
#             )
#         end

#         σ_t    = @view σ[:, :, curr_itr]
#         σ_next = @view σ[:, :, curr_itr + 1]
#         k1_mat = @view σ′[:, :, curr_itr]

#         copyto!(σ_vec, vec(σ_t))

#         if method_sym == :euler
#             mul!(rhs_vec, R, σ_vec)

#             @. σ_vec = σ_vec + Δt * rhs_vec

#         elseif method_sym == :rk2
#             mul!(rhs_vec, R, σ_vec)

#             @. stage_vec = σ_vec + 0.5 * Δt * rhs_vec
#             mul!(k2_vec, R, stage_vec)

#             @. σ_vec = σ_vec + Δt * k2_vec

#         elseif method_sym == :rk4
#             mul!(rhs_vec, R, σ_vec)

#             @. stage_vec = σ_vec + 0.5 * Δt * rhs_vec
#             mul!(k2_vec, R, stage_vec)

#             @. stage_vec = σ_vec + 0.5 * Δt * k2_vec
#             mul!(k3_vec, R, stage_vec)

#             @. stage_vec = σ_vec + Δt * k3_vec
#             mul!(k4_vec, R, stage_vec)

#             @. σ_vec = σ_vec + (Δt / 6.0) * (
#                 rhs_vec + 2.0 * k2_vec + 2.0 * k3_vec + k4_vec
#             )
#         end

#         copyto!(vec(σ_next), σ_vec)
#         copyto!(vec(k1_mat), rhs_vec)

#         enforce_hermiticity!(σ_next)

#         context.curr_itr = UInt64(curr_itr + 1)
#     end

#     return @view σ[:, :, Int(context.curr_itr)]
# end

# function calc__σ_σ′_with_markovian!(
#     context::RmrtContext;
#     use_population_closure::Bool = false,
#     use_local_population_to_coherence::Bool = true,
#     use_secular::Bool = false,
#     secular_tol::Float64 = 1.0e-10,
#     method::Union{Symbol,String} = :rk4,
#     use_threads::Bool = true,
#     verbose::Bool = false,
#     recompute_Λ::Bool = true,
#     markovian_max_itr::Union{Nothing,Int} = nothing,
#     return_generator::Bool = false,
# )
#     R = calc__markovian_generator(
#         context;
#         use_population_closure = use_population_closure,
#         use_local_population_to_coherence = use_local_population_to_coherence,
#         use_secular = use_secular,
#         secular_tol = secular_tol,
#         recompute_Λ = recompute_Λ,
#         markovian_max_itr = markovian_max_itr,
#         use_threads = use_threads,
#         verbose = verbose,
#     )

#     result = calc__σ_σ′_with_markovian_generator!(
#         context,
#         R;
#         method = method,
#         verbose = verbose,
#     )

#     if return_generator
#         return result, R
#     end

#     return result
# end

# =============================================================================
# Markovian RMRT propagation with precomputed Λ-generator
#
# Convention:
#
#     g′_{abcd}(∞)       = -im * Λ_{abcd}
#     conj(g′_{abcd}(∞)) = +im * conj(Λ_{abcd})
#
# The generator R is built once:
#
#     d vec(σ) / dt = R * vec(σ)
#
# and then σ is propagated by Euler / RK2 / RK4 using BLAS mul!.
# =============================================================================

function calc__markovian_generator!(
    R::AbstractMatrix{ComplexF64},
    context::RmrtContext;
    use_population_closure::Bool = false,
    use_local_population_to_coherence::Bool = true,
    use_local_coherence_to_population::Bool = true,
    use_population_memory_population_input::Bool = true,
    use_population_memory_coherence_input::Bool = true,
    use_L0Q_memory_return::Bool = true,
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

    @inline function is_secular_pair(out_a::Int, out_b::Int, in_a::Int, in_b::Int)
        !use_secular && return true
        return abs(ω(out_a, out_b) - ω(in_a, in_b)) <= secular_tol
    end

    @inline is_population_input(a::Int, b::Int) = a == b
    @inline is_coherence_input(a::Int, b::Int) = a != b

    @inline function is_local_population_to_coherence(
        out_a::Int,
        out_b::Int,
        in_a::Int,
        in_b::Int,
    )
        return out_a != out_b && in_a == in_b
    end

    @inline function is_local_coherence_to_population(
        out_a::Int,
        out_b::Int,
        in_a::Int,
        in_b::Int,
    )
        return out_a == out_b && in_a != in_b
    end

    @inline function should_skip_population_memory_input(
        out_a::Int,
        out_b::Int,
        in_a::Int,
        in_b::Int,
    )
        out_a == out_b || return false

        if is_population_input(in_a, in_b) && !use_population_memory_population_input
            return true
        end

        if is_coherence_input(in_a, in_b) && !use_population_memory_coherence_input
            return true
        end

        return false
    end

    @inline function Δtime(Δ_itr::Int)
        return (Δ_itr - 1) * Δt
    end

    @inline function ∫Δweight(Δ_itr::Int)
        return (Δ_itr == 1 || Δ_itr == Δ_max_itr) ? 0.5 * Δt : Δt
    end

    @inline function phase(a::Int, b::Int, Δ_itr::Int)
        return exp(-1.0im * ω(a, b) * Δtime(Δ_itr))
    end

    # -------------------------------------------------------------------------
    # Markovian endpoint helpers
    # -------------------------------------------------------------------------

    @inline function gprime_inf(a::Int, b::Int, c::Int, d::Int)
        return -1.0im * Λ[a, b, c, d]
    end

    @inline function conj_gprime_inf(a::Int, b::Int, c::Int, d::Int)
        return 1.0im * conj(Λ[a, b, c, d])
    end

    @inline function gdiff_inf(a::Int, b::Int, c::Int, d::Int, Δ::Float64)
        return -1.0im * Λ[a, b, c, d] * Δ
    end

    @inline function conj_gdiff_inf(a::Int, b::Int, c::Int, d::Int, Δ::Float64)
        return 1.0im * conj(Λ[a, b, c, d]) * Δ
    end

    # -------------------------------------------------------------------------
    # Markovianized exponent blocks
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
            - gdiff_inf(α⁻, α⁻, α⁼, α⁼, Δ)
            - conj_gdiff_inf(α⁼, α⁼, β, β, Δ)
            + conj_gdiff_inf(α⁻, α⁻, β, β, Δ)

            - g[Δ_itr, α⁻, α⁻, α⁻, α⁻]
            + g[Δ_itr, α⁻, α⁻, α⁼, α⁼]
            - g[Δ_itr, β, β, α⁼, α⁼]
            + g[Δ_itr, β, β, α⁻, α⁻]
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
            - conj_gdiff_inf(β⁻, β⁻, β⁼, β⁼, Δ)
            - gdiff_inf(β⁼, β⁼, a, a, Δ)
            + gdiff_inf(β⁻, β⁻, a, a, Δ)

            - conj(g[Δ_itr, β⁻, β⁻, β⁻, β⁻])
            + conj(g[Δ_itr, β⁻, β⁻, β⁼, β⁼])
            - conj(g[Δ_itr, a, a, β⁼, β⁼])
            + conj(g[Δ_itr, a, a, β⁻, β⁻])
        )
    end

    @inline function gen__markovian_L0Q_exponent_A(
        Δ_itr::Int,
        α::Int,
        β::Int,
        f::Int,
    )
        Δ = Δtime(Δ_itr)

        return (
            -conj_gdiff_inf(β, β, β, β, Δ)
            +conj_gdiff_inf(α, α, β, β, Δ)

            +gdiff_inf(β, β, f, f, Δ)
            -g[Δ_itr, β, β, f, f]

            -gdiff_inf(α, α, f, f, Δ)
            +g[Δ_itr, α, α, f, f]

            -g[Δ_itr, α, α, α, α]
            +g[Δ_itr, β, β, α, α]
        )
    end

    @inline function gen__markovian_L0Q_exponent_B(
        Δ_itr::Int,
        α::Int,
        β::Int,
        f::Int,
    )
        Δ = Δtime(Δ_itr)

        return (
            -gdiff_inf(α, α, α, α, Δ)
            +gdiff_inf(β, β, α, α, Δ)

            +conj_gdiff_inf(α, α, f, f, Δ)
            -conj(g[Δ_itr, α, α, f, f])

            -conj_gdiff_inf(β, β, f, f, Δ)
            +conj(g[Δ_itr, β, β, f, f])

            -conj(g[Δ_itr, β, β, β, β])
            +conj(g[Δ_itr, α, α, β, β])
        )
    end

    @inline function gen__markovian_population_transfer_exponent(
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

    # -------------------------------------------------------------------------
    # Markovianized coefficient blocks
    # -------------------------------------------------------------------------

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

    @inline function gen__markovian_L0Q_coef_A(
        Δ_itr::Int,
        α::Int,
        β::Int,
        f::Int,
    )
        left_one_point = (
             g′[Δ_itr, α, α, f, f]
            -gprime_inf(α, α, f, f)
            -g′[Δ_itr, α, α, α, α]
            +gprime_inf(α, α, α, α)
            -g′[Δ_itr, β, β, f, f]
            +gprime_inf(β, β, f, f)
            +g′[Δ_itr, β, β, α, α]
            -gprime_inf(β, β, α, α)
        )

        right_one_point = (
            -gprime_inf(α, f, f, f)
            -g′[Δ_itr, α, α, α, f]
            +conj_gprime_inf(f, α, β, β)
            +g′[Δ_itr, β, β, α, f]
        )

        return (
            g″[Δ_itr, α, α, α, f]
            -g″[Δ_itr, β, β, α, f]
            -left_one_point * right_one_point
        )
    end

    @inline function gen__markovian_L0Q_coef_B(
        Δ_itr::Int,
        α::Int,
        β::Int,
        f::Int,
    )
        left_one_point = (
            conj_gprime_inf(α, α, f, f)
            -conj(g′[Δ_itr, α, α, f, f])
            +conj(g′[Δ_itr, α, α, β, β])
            -conj_gprime_inf(α, α, β, β)

            -conj_gprime_inf(β, β, f, f)
            +conj(g′[Δ_itr, β, β, f, f])
            -conj(g′[Δ_itr, β, β, β, β])
            +conj_gprime_inf(β, β, β, β)
        )

        right_one_point = (
            -gprime_inf(f, β, α, α)
            -conj(g′[Δ_itr, α, α, β, f])
            +conj_gprime_inf(β, f, f, f)
            +conj(g′[Δ_itr, β, β, β, f])
        )

        return (
            conj(g″[Δ_itr, β, f, α, α])
            -conj(g″[Δ_itr, β, f, β, β])
            -left_one_point * right_one_point
        )
    end

    @inline function gen__markovian_population_transfer_coef(
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
            +left_one_point * right_one_point
        )
    end

    @inline function gen__markovian_population_transfer_kernel(
        Δ_itr::Int,
        src::Int,
        dst::Int,
    )
        return 2.0 * real(
            phase(dst, src, Δ_itr)
            * exp(gen__markovian_population_transfer_exponent(
                Δ_itr,
                src,
                dst,
            ))
            * gen__markovian_population_transfer_coef(
                Δ_itr,
                src,
                dst,
            )
        )
    end

    # -------------------------------------------------------------------------
    # One row of generator R
    # -------------------------------------------------------------------------

    function add_generator_row!(α::Int, β::Int)
        out = vecidx(α, β)

        # ---------------------------------------------------------------------
        # Population-closed Markovian Pauli block
        # ---------------------------------------------------------------------
        if use_population_closure && α == β
            for Δ_itr in 1:Δ_max_itr
                w_int = ∫Δweight(Δ_itr)

                for f in 1:n_sys
                    f == α && continue

                    k_loss = gen__markovian_population_transfer_kernel(
                        Δ_itr,
                        α,
                        f,
                    )

                    k_gain = gen__markovian_population_transfer_kernel(
                        Δ_itr,
                        f,
                        α,
                    )

                    R[out, vecidx(α, α)] += -w_int * k_loss
                    R[out, vecidx(f, f)] +=  w_int * k_gain
                end
            end

            return nothing
        end

        # ---------------------------------------------------------------------
        # t-local diagonal / coherence phase block
        # ---------------------------------------------------------------------
        R[out, out] += (
            -1.0im * ω(α, β)
            -gprime_inf(α, α, α, α)
            +conj_gprime_inf(α, α, β, β)
            +gprime_inf(β, β, α, α)
            -conj_gprime_inf(β, β, β, β)
        )

        # ---------------------------------------------------------------------
        # t-local left mixing
        # input: (α⁻, β) -> output: (α, β)
        # ---------------------------------------------------------------------
        for α⁻ in 1:n_sys
            α⁻ == α && continue

            is_secular_pair(α, β, α⁻, β) || continue

            if !use_local_population_to_coherence &&
               is_local_population_to_coherence(α, β, α⁻, β)
                continue
            end

            if !use_local_coherence_to_population &&
               is_local_coherence_to_population(α, β, α⁻, β)
                continue
            end

            coeff = -(
                gprime_inf(α, α⁻, α⁻, α⁻)
                -conj_gprime_inf(α⁻, α, β, β)
            )

            R[out, vecidx(α⁻, β)] += coeff
        end

        # ---------------------------------------------------------------------
        # t-local right mixing
        # input: (α, β⁻) -> output: (α, β)
        # ---------------------------------------------------------------------
        for β⁻ in 1:n_sys
            β⁻ == β && continue

            is_secular_pair(α, β, α, β⁻) || continue

            if !use_local_population_to_coherence &&
               is_local_population_to_coherence(α, β, α, β⁻)
                continue
            end

            if !use_local_coherence_to_population &&
               is_local_coherence_to_population(α, β, α, β⁻)
                continue
            end

            coeff = (
                gprime_inf(β⁻, β, α, α)
                -conj_gprime_inf(β, β⁻, β⁻, β⁻)
            )

            R[out, vecidx(α, β⁻)] += coeff
        end

        # ---------------------------------------------------------------------
        # Markovianized memory generator
        # ---------------------------------------------------------------------
        for Δ_itr in 1:Δ_max_itr
            w_int = ∫Δweight(Δ_itr)

            # Branch 1
            # input: (α⁼, β) -> output: (α, β)
            for α⁻ in 1:n_sys
                α⁻ == α && continue

                for α⁼ in 1:n_sys
                    α⁼ == α⁻ && continue

                    is_secular_pair(α, β, α⁼, β) || continue
                    should_skip_population_memory_input(α, β, α⁼, β) && continue

                    factor = -(
                        phase(α⁻, β, Δ_itr)
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

                    R[out, vecidx(α⁼, β)] += w_int * factor
                end
            end

            # Branch 2
            # input: (α, β⁼) -> output: (α, β)
            for β⁻ in 1:n_sys
                β⁻ == β && continue

                for β⁼ in 1:n_sys
                    β⁼ == β⁻ && continue

                    is_secular_pair(α, β, α, β⁼) || continue
                    should_skip_population_memory_input(α, β, α, β⁼) && continue

                    factor = -(
                        phase(α, β⁻, Δ_itr)
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

                    R[out, vecidx(α, β⁼)] += w_int * factor
                end
            end

            # Branch 3
            # input: (α⁻, β⁻) -> output: (α, β)
            for α⁻ in 1:n_sys
                α⁻ == α && continue

                for β⁻ in 1:n_sys
                    β⁻ == β && continue

                    is_secular_pair(α, β, α⁻, β⁻) || continue
                    should_skip_population_memory_input(α, β, α⁻, β⁻) && continue

                    factor = (
                        phase(α, β⁻, Δ_itr)
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

            # Branch 4
            # input: (α⁻, β⁻) -> output: (α, β)
            for α⁻ in 1:n_sys
                α⁻ == α && continue

                for β⁻ in 1:n_sys
                    β⁻ == β && continue

                    is_secular_pair(α, β, α⁻, β⁻) || continue
                    should_skip_population_memory_input(α, β, α⁻, β⁻) && continue

                    factor = (
                        phase(α⁻, β, Δ_itr)
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

            # L0Q memory return
            if use_L0Q_memory_return && α != β
                # Branch A
                # input: (f, β) -> output: (α, β), sign = -
                for f in 1:n_sys
                    f == α && continue

                    is_secular_pair(α, β, f, β) || continue

                    if !use_local_population_to_coherence &&
                       is_local_population_to_coherence(α, β, f, β)
                        continue
                    end

                    factor = -(
                        phase(α, β, Δ_itr)
                        * exp(gen__markovian_L0Q_exponent_A(
                            Δ_itr,
                            α,
                            β,
                            f,
                        ))
                        * gen__markovian_L0Q_coef_A(
                            Δ_itr,
                            α,
                            β,
                            f,
                        )
                    )

                    R[out, vecidx(f, β)] += w_int * factor
                end

                # Branch B
                # input: (α, f) -> output: (α, β), sign = +
                for f in 1:n_sys
                    f == β && continue

                    is_secular_pair(α, β, α, f) || continue

                    if !use_local_population_to_coherence &&
                       is_local_population_to_coherence(α, β, α, f)
                        continue
                    end

                    factor = (
                        phase(α, β, Δ_itr)
                        * exp(gen__markovian_L0Q_exponent_B(
                            Δ_itr,
                            α,
                            β,
                            f,
                        ))
                        * gen__markovian_L0Q_coef_B(
                            Δ_itr,
                            α,
                            β,
                            f,
                        )
                    )

                    R[out, vecidx(α, f)] += w_int * factor
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
    use_local_coherence_to_population::Bool = true,
    use_population_memory_population_input::Bool = true,
    use_population_memory_coherence_input::Bool = true,
    use_L0Q_memory_return::Bool = true,
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
        use_local_coherence_to_population = use_local_coherence_to_population,
        use_population_memory_population_input = use_population_memory_population_input,
        use_population_memory_coherence_input = use_population_memory_coherence_input,
        use_L0Q_memory_return = use_L0Q_memory_return,
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
        "Unsupported integration method: $(method). Use :euler, :rk2, or :rk4."
    )

    start_itr < n_itr || return @view σ[:, :, n_itr]

    σ_vec     = Vector{ComplexF64}(undef, n_vec)
    rhs_vec   = Vector{ComplexF64}(undef, n_vec)
    stage_vec = Vector{ComplexF64}(undef, n_vec)
    k2_vec    = Vector{ComplexF64}(undef, n_vec)
    k3_vec    = Vector{ComplexF64}(undef, n_vec)
    k4_vec    = Vector{ComplexF64}(undef, n_vec)

    @inline function enforce_hermiticity!(σ_next::AbstractMatrix)
        for i in 1:n_sys
            σ_next[i, i] = real(σ_next[i, i]) + 0.0im
        end

        for i in 1:n_sys-1
            for j in i+1:n_sys
                c = 0.5 * (σ_next[i, j] + conj(σ_next[j, i]))
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
    use_local_coherence_to_population::Bool = true,
    use_population_memory_population_input::Bool = true,
    use_population_memory_coherence_input::Bool = true,
    use_L0Q_memory_return::Bool = true,
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
        use_local_coherence_to_population = use_local_coherence_to_population,
        use_population_memory_population_input = use_population_memory_population_input,
        use_population_memory_coherence_input = use_population_memory_coherence_input,
        use_L0Q_memory_return = use_L0Q_memory_return,
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

# function calc__σ_σ′_secular_core!(
#     context::RmrtContext;
#     use_population_closure::Bool = false,
#     use_local_population_to_coherence::Bool = true,
#     use_local_coherence_to_population::Bool = true,
#     use_population_memory_population_input::Bool = true,
#     use_population_memory_coherence_input::Bool = false,
#     use_L0Q_memory_return::Bool = true,

#     # ---------------------------------------------------------------------
#     # Optional HEOM-reference injection mode.
#     # If use_heom_input=true, the function reads heom_file and evaluates the
#     # RMRT RHS/diagnostics along the HEOM reduced trajectory instead of the
#     # self-propagated trajectory.  In this mode σ[:, :, itr] is overwritten by
#     # the HEOM density at each grid point, while σ′[:, :, itr] stores the RMRT
#     # RHS evaluated on that HEOM input.
#     #
#     # Expected HEOM text columns:
#     #   time, p1, p2, ..., c12_re, c12_im, ...
#     # Optional derivative columns:
#     #   dp1, dp2, ..., cp12_re, cp12_im, ...
#     # ---------------------------------------------------------------------
#     heom_file::Union{Nothing,String} = nothing,
#     use_heom_input::Bool = false,
#     heom_time_tol::Float64 = 1.0e-8,

#     # ---------------------------------------------------------------------
#     # Teacher-forcing cutoff mode.
#     # If heom_teacher_forcing_cutoff is not nothing, HEOM history is used up
#     # to t_cut, and after t_cut the dynamics is self-propagated.
#     #
#     # Difference from use_heom_input=true:
#     #   use_heom_input=true                 : force HEOM for all times.
#     #   heom_teacher_forcing_cutoff = t_cut : force only t <= t_cut.
#     #
#     # With 1-based grid indexing t=(itr-1)*Δt, the cutoff index is the largest
#     # index satisfying t <= t_cut.  The first self-propagated step starts from
#     # that HEOM cutoff state.
#     # ---------------------------------------------------------------------
#     heom_teacher_forcing_cutoff::Union{Nothing,Real} = nothing,

#     verify_L0Q_terms::Bool = false,
#     verify_L0Q_every::Int = 1,
#     verify_L0Q_pair::Union{Nothing,Tuple{Int,Int}} = nothing,
#     verify_L0Q_io::IO = stderr,
#     verify_L0Q_header::Bool = true,

#     # ---------------------------------------------------------------------
#     # Phase / instantaneous-frequency diagnostics for a coherence component.
#     # This traces RHS contributions R_k through
#     #
#     #     omega_k(t) = Im(conj(c_ab) * R_k) / |c_ab|^2
#     #
#     # which is the RHS-based instantaneous phase velocity contribution.
#     # Use trace_phase_pair=(1,2) to diagnose c12, or nothing for all coherences.
#     # ---------------------------------------------------------------------
#     trace_phase_terms::Bool = false,
#     trace_phase_every::Int = 1,
#     trace_phase_pair::Union{Nothing,Tuple{Int,Int}} = (1,2),
#     trace_phase_eps::Float64 = 1.0e-8,
#     trace_phase_io::IO = stderr,
#     trace_phase_header::Bool = true,

#     # ---------------------------------------------------------------------
#     # Population-RHS diagnostics.
#     # This prints R_aa^RMRT(t) = [dσ_aa/dt]_RMRT evaluated on the current
#     # input history.  If use_heom_input=true and heom_file contains derivative
#     # columns dp1, dp2, ..., the same line also prints the HEOM finite-difference
#     # dp_a(t) and the RMRT-minus-HEOM derivative error.
#     #
#     # Use trace_population_indices=nothing for all populations, or e.g.
#     # trace_population_indices=(1, 2) / [1, 2] for selected diagonal entries.
#     # ---------------------------------------------------------------------
#     trace_population_rhs::Bool = false,
#     trace_population_every::Int = 1,
#     trace_population_indices = nothing,
#     trace_population_io::IO = stderr,
#     trace_population_header::Bool = true,

#     # ---------------------------------------------------------------------
#     # Population-output RHS decomposition diagnostics.
#     # This is intended for HEOM-forced RHS checks.  For each selected
#     # population output (a,a), it prints the applied local P L1 P piece,
#     # the applied P L1 Q G0 Q L1 P memory branches split by population vs
#     # coherence input, and the residual against HEOM dp_a when available.
#     #
#     # The diagnostic is independent of trace_population_rhs; use a separate
#     # IO stream if you want a clean table.
#     # ---------------------------------------------------------------------
#     trace_population_decomp::Bool = false,
#     trace_population_decomp_every::Int = 1,
#     trace_population_decomp_indices = nothing,
#     trace_population_decomp_io::IO = stderr,
#     trace_population_decomp_header::Bool = true,

#     verify_L0P_transport::Bool = false,
#     verify_L0P_every::Int = 100,
#     verify_L0P_pair::Union{Nothing,Tuple{Int,Int}} = (1,2),
#     verify_L0P_s_offset::Int = 100,
#     verify_L0P_io::IO = stderr,
#     verify_L0P_header::Bool = true,

#     use_secular::Bool = false,
#     secular_tol::Float64 = 1.0e-10,
#     method::Union{Symbol,String} = :rk4,
#     use_threads::Bool = true,

#     # Half-shifted g-grid is allocated/calculated lazily.  This flag only
#     # prepares the cache for stage-time RK2/RK4 support; the RHS refactor should
#     # use the prepared cache explicitly rather than allocating it in inner loops.
#     auto_prepare_half_shifted_grid::Bool = true,
#     recompute_half_shifted_grid::Bool = false,

#     verbose::Bool = true,
    
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

#     function load__heom_reference_density_and_derivative(
#         path::String,
#         n_sys::Int,
#         n_itr::Int,
#         Δt::Float64;
#         time_tol::Float64 = 1.0e-8,
#     )
#         isfile(path) || error("HEOM reference file not found: $(path)")

#         lines = readlines(path)
#         header = String[]
#         rows = Vector{Vector{Float64}}()
#         header_found = false

#         for raw_line in lines
#             line = strip(raw_line)
#             isempty(line) && continue
#             startswith(line, "#") && continue
#             startswith(line, "----") && continue

#             parts = split(line)
#             isempty(parts) && continue

#             if !header_found
#                 if any(x -> x == "time", parts) && any(x -> x == "p1", parts)
#                     header = String.(parts)
#                     header_found = true
#                 end
#                 continue
#             end

#             length(parts) < length(header) && continue

#             vals = Vector{Float64}(undef, length(header))
#             ok = true
#             for j in eachindex(header)
#                 v = tryparse(Float64, parts[j])
#                 if v === nothing
#                     ok = false
#                     break
#                 end
#                 vals[j] = v
#             end
#             ok && push!(rows, vals)
#         end

#         header_found || error("Could not find HEOM header line with columns `time` and `p1` in $(path)")
#         isempty(rows) && error("No numeric HEOM data rows found in $(path)")

#         col = Dict{String,Int}()
#         for (j, name) in pairs(header)
#             col[name] = j
#         end

#         haskey(col, "time") || error("HEOM file is missing `time` column")
#         times = [row[col["time"]] for row in rows]

#         # Require monotonic time data.
#         for j in 2:length(times)
#             times[j] >= times[j - 1] || error("HEOM time column must be nondecreasing")
#         end

#         function interp_value(name::String, t::Float64)
#             haskey(col, name) || error("HEOM file is missing required column `$(name)`")

#             if t <= times[1] + time_tol
#                 return rows[1][col[name]]
#             elseif t >= times[end] - time_tol
#                 return rows[end][col[name]]
#             end

#             j = searchsortedlast(times, t)
#             j = clamp(j, 1, length(times) - 1)
#             t0 = times[j]
#             t1 = times[j + 1]

#             if abs(t1 - t0) <= eps(Float64)
#                 return rows[j][col[name]]
#             end

#             λ = (t - t0) / (t1 - t0)
#             return (1.0 - λ) * rows[j][col[name]] + λ * rows[j + 1][col[name]]
#         end

#         function interp_optional_value(name::String, t::Float64)
#             haskey(col, name) || return NaN

#             if t <= times[1] + time_tol
#                 return rows[1][col[name]]
#             elseif t >= times[end] - time_tol
#                 return rows[end][col[name]]
#             end

#             j = searchsortedlast(times, t)
#             j = clamp(j, 1, length(times) - 1)
#             t0 = times[j]
#             t1 = times[j + 1]

#             if abs(t1 - t0) <= eps(Float64)
#                 return rows[j][col[name]]
#             end

#             λ = (t - t0) / (t1 - t0)
#             return (1.0 - λ) * rows[j][col[name]] + λ * rows[j + 1][col[name]]
#         end

#         heom_sigma = Array{ComplexF64}(undef, n_sys, n_sys, n_itr)
#         heom_sigmap = fill(ComplexF64(NaN, NaN), n_sys, n_sys, n_itr)

#         for itr in 1:n_itr
#             t = (itr - 1) * Δt

#             for a in 1:n_sys
#                 pa = interp_value("p$(a)", t)
#                 heom_sigma[a, a, itr] = pa + 0.0im

#                 dpa = interp_optional_value("dp$(a)", t)
#                 if isfinite(dpa)
#                     heom_sigmap[a, a, itr] = dpa + 0.0im
#                 end
#             end

#             for a in 1:(n_sys - 1)
#                 for b in (a + 1):n_sys
#                     re = interp_value("c$(a)$(b)_re", t)
#                     imv = interp_value("c$(a)$(b)_im", t)
#                     cab = re + 1.0im * imv
#                     heom_sigma[a, b, itr] = cab
#                     heom_sigma[b, a, itr] = conj(cab)

#                     dre = interp_optional_value("cp$(a)$(b)_re", t)
#                     dimv = interp_optional_value("cp$(a)$(b)_im", t)
#                     if isfinite(dre) && isfinite(dimv)
#                         dcab = dre + 1.0im * dimv
#                         heom_sigmap[a, b, itr] = dcab
#                         heom_sigmap[b, a, itr] = conj(dcab)
#                     end
#                 end
#             end
#         end

#         return heom_sigma, heom_sigmap
#     end

#     heom_sigma_ref = nothing
#     heom_sigmap_ref = nothing

#     teacher_forcing_enabled = heom_teacher_forcing_cutoff !== nothing
#     teacher_cutoff_itr = 0

#     if teacher_forcing_enabled
#         t_cut = Float64(heom_teacher_forcing_cutoff)
#         t_cut >= 0.0 || error("heom_teacher_forcing_cutoff must be >= 0")

#         # 1-based index with t = (itr - 1) * Δt.
#         # Choose the largest index with t <= t_cut, up to tiny roundoff.
#         teacher_cutoff_itr = clamp(
#             floor(Int, t_cut / Δt + 1.0e-9) + 1,
#             1,
#             n_itr,
#         )
#     end

#     if use_heom_input || teacher_forcing_enabled
#         heom_file === nothing && error(
#             "HEOM forcing requires heom_file=\"path/to/heom.txt\""
#         )

#         heom_sigma_ref, heom_sigmap_ref = load__heom_reference_density_and_derivative(
#             String(heom_file),
#             n_sys,
#             n_itr,
#             Δt;
#             time_tol = heom_time_tol,
#         )
#     end

#     @inline function use_heom_for_storage_itr(itr::Int)
#         # Store exact HEOM state up to and including the cutoff point.
#         return use_heom_input ||
#                (teacher_forcing_enabled && itr <= teacher_cutoff_itr)
#     end

#     @inline function use_heom_for_rhs_current_itr(itr::Int)
#         # Before cutoff, RHS diagnostics are evaluated on HEOM current.
#         # At the cutoff point, RK stages use σ_t so that the first self step
#         # starts from the HEOM cutoff state.
#         return use_heom_input ||
#                (teacher_forcing_enabled && itr < teacher_cutoff_itr)
#     end

#     @inline function use_heom_for_history_itr(itr::Int)
#         # Memory history strictly before cutoff is HEOM.  The cutoff point is
#         # already stored in σ as HEOM and then treated as the self initial state.
#         return use_heom_input ||
#                (teacher_forcing_enabled && itr < teacher_cutoff_itr)
#     end

#     method_sym = Symbol(lowercase(String(method)))
#     method_sym in (:euler, :rk2, :rk4) || error(
#         "Unsupported integration method: $(method). " *
#         "Use :euler, :rk2, or :rk4."
#     )
#     needs_half_shifted_grid = method_sym in (:rk2, :rk4)

#     if needs_half_shifted_grid
#         if auto_prepare_half_shifted_grid
#             ensure__half_shifted_grid!(
#                 context;
#                 recompute = recompute_half_shifted_grid,
#                 use_threads = use_threads,
#                 verbose = verbose,
#             )
#         elseif !context.using_half_shifted_grid || !has__half_shifted_grid(context)
#             error(
#                 "method=$(method_sym) requires half-shifted g-series cache. " *
#                 "Call ensure__half_shifted_grid!(context) or " *
#                 "calc__g_g′_g″_half_shifted!(context) before propagation, " *
#                 "or set auto_prepare_half_shifted_grid=true."
#             )
#         end
#     end

#     verify_L0Q_every >= 1 || error("verify_L0Q_every must be >= 1")
#     trace_phase_every >= 1 || error("trace_phase_every must be >= 1")
#     trace_phase_eps > 0.0 || error("trace_phase_eps must be > 0")
#     trace_population_every >= 1 || error("trace_population_every must be >= 1")
#     trace_population_decomp_every >= 1 || error("trace_population_decomp_every must be >= 1")
#     verify_L0P_every >= 1 || error("verify_L0P_every must be >= 1")
#     verify_L0P_s_offset >= 1 || error("verify_L0P_s_offset must be >= 1")

#     start_itr < n_itr || return @view σ[:, :, n_itr]

#     @inline ω(a::Int, b::Int) = ϵ[a] - ϵ[b]

#     # -------------------------------------------------------------------------
#     # Verification for QL0P - dot(P)P = 0 convention
#     #
#     # We check consistency between:
#     #
#     #   D_code[a,b](t)
#     #     = -iω_ab - g′_aaaa + conj(g′_aabb) + g′_bbaa - conj(g′_bbbb)
#     #
#     # and
#     #
#     #   D_N[a,b](t)
#     #     = -iω_ab + d/dt log N_ab(t)
#     #
#     # where
#     #
#     #   log N_ab(t)
#     #     = -g_aaaa(t) - conj(g_bbbb(t))
#     #       + g_bbaa(t) + conj(g_aabb(t)).
#     #
#     # If D_code - D_N is not small, the L0-transport convention,
#     # N_ab convention, or g′ sign/conjugation convention is inconsistent.
#     # -------------------------------------------------------------------------

#     verify_L0P_header_printed = Ref(false)
#     verify_L0P_lock = ReentrantLock()

#     @inline function logN_bath(
#         itr::Int,
#         a::Int,
#         b::Int,
#     )
#         return (
#             -g[itr, a, a, a, a]
#             -conj(g[itr, b, b, b, b])
#             +g[itr, b, b, a, a]
#             +conj(g[itr, a, a, b, b])
#         )
#     end

#     @inline function D_L0P_code(
#         itr::Int,
#         a::Int,
#         b::Int,
#     )
#         return (
#             -1.0im * ω(a, b)
#             -g′[itr, a, a, a, a]
#             +conj(g′[itr, a, a, b, b])
#             +g′[itr, b, b, a, a]
#             -conj(g′[itr, b, b, b, b])
#         )
#     end

#     @inline function dlogN_bath_gprime(
#         itr::Int,
#         a::Int,
#         b::Int,
#     )
#         return (
#             -g′[itr, a, a, a, a]
#             -conj(g′[itr, b, b, b, b])
#             +g′[itr, b, b, a, a]
#             +conj(g′[itr, a, a, b, b])
#         )
#     end

#     @inline function D_L0P_from_N_gprime(
#         itr::Int,
#         a::Int,
#         b::Int,
#     )
#         return -1.0im * ω(a, b) + dlogN_bath_gprime(itr, a, b)
#     end

#     @inline function dlogN_bath_finite_difference(
#         itr::Int,
#         a::Int,
#         b::Int,
#     )
#         if itr <= 1
#             return (logN_bath(2, a, b) - logN_bath(1, a, b)) / Δt
#         elseif itr >= n_itr
#             return (logN_bath(n_itr, a, b) - logN_bath(n_itr - 1, a, b)) / Δt
#         else
#             return (logN_bath(itr + 1, a, b) - logN_bath(itr - 1, a, b)) / (2.0 * Δt)
#         end
#     end

#     @inline function D_L0P_from_N_fd(
#         itr::Int,
#         a::Int,
#         b::Int,
#     )
#         return -1.0im * ω(a, b) + dlogN_bath_finite_difference(itr, a, b)
#     end

#     @inline function should_verify_L0P(
#         curr_itr::Int,
#         a::Int,
#         b::Int,
#     )
#         if !verify_L0P_transport || a == b
#             return false
#         end

#         ((curr_itr - start_itr) % verify_L0P_every == 0) || return false

#         verify_L0P_pair === nothing && return true

#         return verify_L0P_pair == (a, b)
#     end

#     function print_verify_L0P_header_if_needed!()
#         if verify_L0P_header && !verify_L0P_header_printed[]
#             lock(verify_L0P_lock)
#             try
#                 if !verify_L0P_header_printed[]
#                     println(
#                         verify_L0P_io,
#                         "# L0P_VERIFY columns: itr t a b s_itr Δ " *
#                         "D_code D_N_gprime D_N_fd " *
#                         "D_code_minus_D_N_gprime D_code_minus_D_N_fd " *
#                         "log_transport_exact log_transport_quad transport_error"
#                     )
#                     verify_L0P_header_printed[] = true
#                 end
#             finally
#                 unlock(verify_L0P_lock)
#             end
#         end

#         return nothing
#     end

#     @inline function fmt__L0P_real(x)
#         return Printf.@sprintf("%+.12e", Float64(real(x)))
#     end

#     @inline function fmt__L0P_time(x)
#         return Printf.@sprintf("%.10e", Float64(x))
#     end

#     @inline function fmt__L0P_complex(z)
#         return "(" * fmt__L0P_real(real(z)) * "," * fmt__L0P_real(imag(z)) * ")"
#     end

#     @inline function log_transport_exact(
#         s_itr::Int,
#         t_itr::Int,
#         a::Int,
#         b::Int,
#     )
#         Δ = (t_itr - s_itr) * Δt

#         return (
#             -1.0im * ω(a, b) * Δ
#             + logN_bath(t_itr, a, b)
#             - logN_bath(s_itr, a, b)
#         )
#     end

#     function log_transport_quad(
#         s_itr::Int,
#         t_itr::Int,
#         a::Int,
#         b::Int,
#     )
#         if s_itr == t_itr
#             return 0.0 + 0.0im
#         end

#         acc = 0.0 + 0.0im

#         @inbounds for itr in s_itr:t_itr
#             w = (itr == s_itr || itr == t_itr) ? 0.5 * Δt : Δt
#             acc += w * D_L0P_code(itr, a, b)
#         end

#         return acc
#     end

#     function print__L0P_verify_line!(
#         io::IO,
#         curr_itr::Int,
#         time::Float64,
#         a::Int,
#         b::Int,
#         s_itr::Int,
#         Δ::Float64,
#         D_code,
#         D_N_gprime,
#         D_N_fd,
#         D_diff_gprime,
#         D_diff_fd,
#         log_exact,
#         log_quad,
#         log_err,
#     )
#         println(
#             io,
#             "L0P_VERIFY",
#             " itr=", curr_itr,
#             " t=", fmt__L0P_time(time),
#             " a=", a,
#             " b=", b,
#             " s_itr=", s_itr,
#             " Delta=", fmt__L0P_time(Δ),
#             " D_code=", fmt__L0P_complex(D_code),
#             " D_N_gprime=", fmt__L0P_complex(D_N_gprime),
#             " D_N_fd=", fmt__L0P_complex(D_N_fd),
#             " D_code_minus_D_N_gprime=", fmt__L0P_complex(D_diff_gprime),
#             " D_code_minus_D_N_fd=", fmt__L0P_complex(D_diff_fd),
#             " log_transport_exact=", fmt__L0P_complex(log_exact),
#             " log_transport_quad=", fmt__L0P_complex(log_quad),
#             " transport_error=", fmt__L0P_complex(log_err),
#         )

#         return nothing
#     end

#     function verify__L0P_transport_at!(
#         curr_itr::Int,
#     )
#         verify_L0P_transport || return nothing

#         time = (curr_itr - 1) * Δt
#         s_itr = max(1, curr_itr - verify_L0P_s_offset)
#         Δ = (curr_itr - s_itr) * Δt

#         print_verify_L0P_header_if_needed!()

#         @inbounds for b in 1:n_sys
#             for a in 1:n_sys
#                 should_verify_L0P(curr_itr, a, b) || continue

#                 D_code = D_L0P_code(curr_itr, a, b)
#                 D_N_gprime = D_L0P_from_N_gprime(curr_itr, a, b)
#                 D_N_fd = D_L0P_from_N_fd(curr_itr, a, b)

#                 D_diff_gprime = D_code - D_N_gprime
#                 D_diff_fd = D_code - D_N_fd

#                 log_exact = log_transport_exact(s_itr, curr_itr, a, b)
#                 log_quad = log_transport_quad(s_itr, curr_itr, a, b)
#                 log_err = log_exact - log_quad

#                 lock(verify_L0P_lock)
#                 try
#                     print__L0P_verify_line!(
#                         verify_L0P_io,
#                         curr_itr,
#                         time,
#                         a,
#                         b,
#                         s_itr,
#                         Δ,
#                         D_code,
#                         D_N_gprime,
#                         D_N_fd,
#                         D_diff_gprime,
#                         D_diff_fd,
#                         log_exact,
#                         log_quad,
#                         log_err,
#                     )
#                 finally
#                     unlock(verify_L0P_lock)
#                 end
#             end
#         end

#         return nothing
#     end

#     @inline function is_secular_pair(
#         out_a::Int,
#         out_b::Int,
#         in_a::Int,
#         in_b::Int,
#     )
#         if !use_secular
#             return true
#         end

#         return abs(ω(out_a, out_b) - ω(in_a, in_b)) <= secular_tol
#     end

#     @inline function is_local_population_to_coherence(
#         out_a::Int,
#         out_b::Int,
#         in_a::Int,
#         in_b::Int,
#     )
#         return out_a != out_b && in_a == in_b
#     end

#     @inline function is_local_coherence_to_population(
#         out_a::Int,
#         out_b::Int,
#         in_a::Int,
#         in_b::Int,
#     )
#         return out_a == out_b && in_a != in_b
#     end

#     @inline function is_population_input(in_a::Int, in_b::Int)
#         return in_a == in_b
#     end

#     @inline function is_coherence_input(in_a::Int, in_b::Int)
#         return in_a != in_b
#     end

#     @inline function should_skip_population_memory_input(
#         out_a::Int,
#         out_b::Int,
#         in_a::Int,
#         in_b::Int,
#     )
#         out_a == out_b || return false

#         if is_population_input(in_a, in_b) && !use_population_memory_population_input
#             return true
#         end

#         if is_coherence_input(in_a, in_b) && !use_population_memory_coherence_input
#             return true
#         end

#         return false
#     end

#     @inline function ∫weight(s_itr::Int, curr_itr::Int)
#         return (s_itr == 1 || s_itr == curr_itr) ? 0.5 * Δt : Δt
#     end

#     @inline function phase(a::Int, b::Int, Δ_itr::Int)
#         Δ = (Δ_itr - 1) * Δt
#         return exp(-1.0im * ω(a, b) * Δ)
#     end

#     # -------------------------------------------------------------------------
#     # Non-time-localized memory density access
#     # -------------------------------------------------------------------------
#     # Memory integrals must use the density at the integration time s.
#     # For the endpoint s = t during RK stages, use the current trial matrix σ_t;
#     # for earlier times, use the already stored trajectory σ[:, :, s_itr].
#     # This removes the previous time-localization σ(s) -> σ(t) in the memory
#     # integrals while preserving RK stage consistency at the moving endpoint.
#     # -------------------------------------------------------------------------
#     @inline function σ_mem(
#         σ_t::AbstractMatrix,
#         s_itr::Int,
#         curr_itr::Int,
#         a::Int,
#         b::Int,
#     )
#         if use_heom_for_history_itr(s_itr)
#             return heom_sigma_ref[a, b, s_itr]
#         end

#         return s_itr == curr_itr ? σ_t[a, b] : σ[a, b, s_itr]
#     end

#     @inline function σ_now(
#         σ_t::AbstractMatrix,
#         curr_itr::Int,
#         a::Int,
#         b::Int,
#     )
#         if use_heom_for_rhs_current_itr(curr_itr)
#             return heom_sigma_ref[a, b, curr_itr]
#         end

#         return σ_t[a, b]
#     end

#     @inline function σprime_heom_now(
#         curr_itr::Int,
#         a::Int,
#         b::Int,
#     )
#         if use_heom_input || teacher_forcing_enabled
#             return heom_sigmap_ref[a, b, curr_itr]
#         end

#         return ComplexF64(NaN, NaN)
#     end

#     # -------------------------------------------------------------------------
#     # Verification diagnostics for the first-order coherence-source package.
#     # These diagnostics are intentionally separated from the applied RHS.
#     # They report the raw, secular-filtered local PL1P pop -> coh source and
#     # the raw, secular-filtered PL0Q exp(ΔL0) QL1P return source.
#     # If use_local_population_to_coherence = false, the applied RHS may skip
#     # these pop -> coh sources, but the verification values still show what
#     # would have been present before that switch removes them.
#     # -------------------------------------------------------------------------

#     verify_header_printed = Ref(false)
#     verify_lock = ReentrantLock()

#     @inline function should_verify_L0Q(curr_itr::Int, α::Int, β::Int)
#         if !verify_L0Q_terms || α == β
#             return false
#         end
#         ((curr_itr - start_itr) % verify_L0Q_every == 0) || return false
#         verify_L0Q_pair === nothing && return true
#         return verify_L0Q_pair == (α, β)
#     end

#     function print_verify_L0Q_header_if_needed!()
#         if verify_L0Q_header && !verify_header_printed[]
#             lock(verify_lock)
#             try
#                 if !verify_header_printed[]
#                     println(
#                         verify_L0Q_io,
#                         "# L0Q_VERIFY columns: itr t α β " *
#                         "local_A_pop local_B_pop local_pop " *
#                         "L0Q_A_pop L0Q_B_pop L0Q_pop " *
#                         "local_A_plus_L0Q_A_pop local_B_plus_L0Q_B_pop local_plus_L0Q_pop " *
#                         "L0Q_A_all L0Q_B_all L0Q_all rhs"
#                     )
#                     verify_header_printed[] = true
#                 end
#             finally
#                 unlock(verify_lock)
#             end
#         end
#         return nothing
#     end

#     @inline function fmt__real(x)
#         return Printf.@sprintf("%+.12e", Float64(real(x)))
#     end

#     @inline function fmt__time(x)
#         return Printf.@sprintf("%.10e", Float64(x))
#     end

#     @inline function fmt__complex(z)
#         return fmt__real(real(z)) * " " * fmt__real(imag(z))
#     end

#     @inline function print__L0Q_verify_line!(
#         io::IO,
#         curr_itr::Int,
#         t::Float64,
#         α::Int,
#         β::Int,
#         local_A_pop,
#         local_B_pop,
#         local_pop,
#         L0Q_A_pop,
#         L0Q_B_pop,
#         L0Q_pop,
#         local_A_plus_L0Q_A_pop,
#         local_B_plus_L0Q_B_pop,
#         local_plus_L0Q_pop,
#         L0Q_A_all,
#         L0Q_B_all,
#         L0Q_all,
#         rhs,
#     )
#         println(
#             io,
#             "L0Q_VERIFY",
#             " itr=", curr_itr,
#             " t=", fmt__time(t),
#             " a=", α,
#             " b=", β,
#             " local_A_pop=", fmt__complex(local_A_pop),
#             " local_B_pop=", fmt__complex(local_B_pop),
#             " local_pop=", fmt__complex(local_pop),
#             " L0Q_A_pop=", fmt__complex(L0Q_A_pop),
#             " L0Q_B_pop=", fmt__complex(L0Q_B_pop),
#             " L0Q_pop=", fmt__complex(L0Q_pop),
#             " local_A_plus_L0Q_A_pop=", fmt__complex(local_A_plus_L0Q_A_pop),
#             " local_B_plus_L0Q_B_pop=", fmt__complex(local_B_plus_L0Q_B_pop),
#             " local_plus_L0Q_pop=", fmt__complex(local_plus_L0Q_pop),
#             " L0Q_A_all=", fmt__complex(L0Q_A_all),
#             " L0Q_B_all=", fmt__complex(L0Q_B_all),
#             " L0Q_all=", fmt__complex(L0Q_all),
#             " rhs=", fmt__complex(rhs),
#         )
#         return nothing
#     end

#     # -------------------------------------------------------------------------
#     # Phase / instantaneous-frequency diagnostics
#     # -------------------------------------------------------------------------

#     trace_phase_header_printed = Ref(false)
#     trace_phase_lock = ReentrantLock()

#     @inline function should_trace_phase(curr_itr::Int, α::Int, β::Int)
#         if !trace_phase_terms || α == β
#             return false
#         end

#         ((curr_itr - start_itr) % trace_phase_every == 0) || return false

#         trace_phase_pair === nothing && return true

#         return trace_phase_pair == (α, β)
#     end

#     @inline function omega_from_rhs_term(term, c)
#         denom = abs2(c)
#         if denom <= trace_phase_eps^2
#             return NaN
#         end
#         return imag(conj(c) * term) / denom
#     end

#     @inline function growth_from_rhs_term(term, c)
#         denom = abs2(c)
#         if denom <= trace_phase_eps^2
#             return NaN
#         end
#         return real(conj(c) * term) / denom
#     end

#     function print_trace_phase_header_if_needed!()
#         if trace_phase_header && !trace_phase_header_printed[]
#             lock(trace_phase_lock)
#             try
#                 if !trace_phase_header_printed[]
#                     println(
#                         trace_phase_io,
#                         "PHASE_TRACE itr t a b abs_c " *
#                         "REAL_c IMAG_c REAL_rhs_diag IMAG_rhs_diag REAL_rhs_local_raw IMAG_rhs_local_raw REAL_rhs_local_app IMAG_rhs_local_app " *
#                         "REAL_rhs_mem_core_app IMAG_rhs_mem_core_app REAL_rhs_L0Q_raw IMAG_rhs_L0Q_raw REAL_rhs_L0Q_app IMAG_rhs_L0Q_app REAL_rhs_total IMAG_rhs_total REAL_rhs_heom_fd IMAG_rhs_heom_fd " *
#                         "omega_total omega_heom_fd omega_diag omega_local_raw omega_local_app " *
#                         "omega_mem_core_app omega_L0Q_raw omega_L0Q_app " *
#                         "omega_L0Q_A_raw omega_L0Q_B_raw " *
#                         "growth_total growth_heom_fd growth_L0Q_raw growth_L0Q_app"
#                     )
#                     trace_phase_header_printed[] = true
#                 end
#             finally
#                 unlock(trace_phase_lock)
#             end
#         end
#         return nothing
#     end

#     function print__phase_trace_line!(
#         io::IO,
#         curr_itr::Int,
#         t::Float64,
#         α::Int,
#         β::Int,
#         c,
#         rhs_diag,
#         rhs_local_raw,
#         rhs_local_app,
#         rhs_mem_core_app,
#         rhs_L0Q_raw,
#         rhs_L0Q_app,
#         rhs_L0Q_A_raw,
#         rhs_L0Q_B_raw,
#         rhs_total,
#         rhs_heom_fd,
#     )
#         omega_total        = omega_from_rhs_term(rhs_total, c)
#         omega_heom_fd      = omega_from_rhs_term(rhs_heom_fd, c)
#         omega_diag         = omega_from_rhs_term(rhs_diag, c)
#         omega_local_raw    = omega_from_rhs_term(rhs_local_raw, c)
#         omega_local_app    = omega_from_rhs_term(rhs_local_app, c)
#         omega_mem_core_app = omega_from_rhs_term(rhs_mem_core_app, c)
#         omega_L0Q_raw      = omega_from_rhs_term(rhs_L0Q_raw, c)
#         omega_L0Q_app      = omega_from_rhs_term(rhs_L0Q_app, c)
#         omega_L0Q_A_raw    = omega_from_rhs_term(rhs_L0Q_A_raw, c)
#         omega_L0Q_B_raw    = omega_from_rhs_term(rhs_L0Q_B_raw, c)

#         growth_total       = growth_from_rhs_term(rhs_total, c)
#         growth_heom_fd     = growth_from_rhs_term(rhs_heom_fd, c)
#         growth_L0Q_raw     = growth_from_rhs_term(rhs_L0Q_raw, c)
#         growth_L0Q_app     = growth_from_rhs_term(rhs_L0Q_app, c)

#         println(
#             io,
#             "PHASE_TRACE",                          " ",
#             # " itr=",                curr_itr,
#             # " t=",                  fmt__time(t),
#             # " a=",                  α,
#             # " b=",                  β,
#             # " abs_c=",              fmt__real(abs(c)),
#             # " c=",                  fmt__complex(c),
#             # " rhs_diag=",           fmt__complex(rhs_diag),
#             # " rhs_local_raw=",      fmt__complex(rhs_local_raw),
#             # " rhs_local_app=",      fmt__complex(rhs_local_app),
#             # " rhs_mem_core_app=",   fmt__complex(rhs_mem_core_app),
#             # " rhs_L0Q_raw=",        fmt__complex(rhs_L0Q_raw),
#             # " rhs_L0Q_app=",        fmt__complex(rhs_L0Q_app),
#             # " rhs_total=",          fmt__complex(rhs_total),
#             # " omega_total=",        fmt__real(omega_total),
#             # " omega_diag=",         fmt__real(omega_diag),
#             # " omega_local_raw=",    fmt__real(omega_local_raw),
#             # " omega_local_app=",    fmt__real(omega_local_app),
#             # " omega_mem_core_app=", fmt__real(omega_mem_core_app),
#             # " omega_L0Q_raw=",      fmt__real(omega_L0Q_raw),
#             # " omega_L0Q_app=",      fmt__real(omega_L0Q_app),
#             # " omega_L0Q_A_raw=",    fmt__real(omega_L0Q_A_raw),
#             # " omega_L0Q_B_raw=",    fmt__real(omega_L0Q_B_raw),
#             # " growth_total=",       fmt__real(growth_total),
#             # " growth_L0Q_raw=",     fmt__real(growth_L0Q_raw),
#             # " growth_L0Q_app=",     fmt__real(growth_L0Q_app),
#             curr_itr,                           " ",
#             fmt__time(t),                       " ",
#             α,                                  " ",
#             β,                                  " ",
#             fmt__real(abs(c)),                  " ",                
#             fmt__complex(c),                    " ",              
#             fmt__complex(rhs_diag),             " ",                     
#             fmt__complex(rhs_local_raw),        " ",         
#             fmt__complex(rhs_local_app),        " ",         
#             fmt__complex(rhs_mem_core_app),     " ",            
#             fmt__complex(rhs_L0Q_raw),          " ",       
#             fmt__complex(rhs_L0Q_app),          " ",       
#             fmt__complex(rhs_total),            " ",     
#             fmt__complex(rhs_heom_fd),          " ",
#             fmt__real(omega_total),             " ",    
#             fmt__real(omega_heom_fd),           " ",
#             fmt__real(omega_diag),              " ",   
#             fmt__real(omega_local_raw),         " ",        
#             fmt__real(omega_local_app),         " ",        
#             fmt__real(omega_mem_core_app),      " ",           
#             fmt__real(omega_L0Q_raw),           " ",      
#             fmt__real(omega_L0Q_app),           " ",      
#             fmt__real(omega_L0Q_A_raw),         " ",        
#             fmt__real(omega_L0Q_B_raw),         " ",        
#             fmt__real(growth_total),            " ",     
#             fmt__real(growth_heom_fd),          " ",
#             fmt__real(growth_L0Q_raw),          " ",       
#             fmt__real(growth_L0Q_app),          " ",       
#         )
#         return nothing
#     end


#     # -------------------------------------------------------------------------
#     # Population RHS diagnostics
#     # -------------------------------------------------------------------------

#     trace_population_header_printed = Ref(false)
#     trace_population_lock = ReentrantLock()

#     @inline function should_trace_population(curr_itr::Int, a::Int)
#         trace_population_rhs || return false
#         ((curr_itr - start_itr) % trace_population_every == 0) || return false
#         trace_population_indices === nothing && return true
#         return a in trace_population_indices
#     end

#     function print_trace_population_header_if_needed!()
#         if trace_population_header && !trace_population_header_printed[]
#             lock(trace_population_lock)
#             try
#                 if !trace_population_header_printed[]
#                     println(
#                         trace_population_io,
#                         "POP_RHS_TRACE itr t a p_a " *
#                         "REAL_Raa_RMRT IMAG_Raa_RMRT " *
#                         "REAL_dp_a_HEOM IMAG_dp_a_HEOM " *
#                         "REAL_R_minus_dp IMAG_R_minus_dp"
#                     )
#                     trace_population_header_printed[] = true
#                 end
#             finally
#                 unlock(trace_population_lock)
#             end
#         end
#         return nothing
#     end

#     function print__population_rhs_trace_line!(
#         io::IO,
#         curr_itr::Int,
#         t::Float64,
#         a::Int,
#         p_a,
#         rhs_aa,
#         heom_dp_aa,
#     )
#         diff = rhs_aa - heom_dp_aa

#         println(
#             io,
#             "POP_RHS_TRACE", " ",
#             curr_itr, " ",
#             fmt__time(t), " ",
#             a, " ",
#             fmt__real(real(p_a)), " ",
#             fmt__complex(rhs_aa), " ",
#             fmt__complex(heom_dp_aa), " ",
#             fmt__complex(diff),
#         )

#         return nothing
#     end

#     @inline function trace__population_rhs_if_needed!(
#         curr_itr::Int,
#         σ_t::AbstractMatrix,
#         a::Int,
#         rhs_aa,
#     )
#         should_trace_population(curr_itr, a) || return nothing

#         time = (curr_itr - 1) * Δt
#         p_a = real(σ_now(σ_t, curr_itr, a, a))
#         heom_dp_aa = σprime_heom_now(curr_itr, a, a)

#         print_trace_population_header_if_needed!()
#         lock(trace_population_lock)
#         try
#             print__population_rhs_trace_line!(
#                 trace_population_io,
#                 curr_itr,
#                 time,
#                 a,
#                 p_a,
#                 rhs_aa,
#                 heom_dp_aa,
#             )
#         finally
#             unlock(trace_population_lock)
#         end

#         return nothing
#     end


#     # -------------------------------------------------------------------------
#     # Population-output RHS decomposition diagnostics
#     # -------------------------------------------------------------------------

#     trace_population_decomp_header_printed = Ref(false)
#     trace_population_decomp_lock = ReentrantLock()

#     @inline function should_trace_population_decomp(curr_itr::Int, a::Int)
#         trace_population_decomp || return false
#         ((curr_itr - start_itr) % trace_population_decomp_every == 0) || return false
#         trace_population_decomp_indices === nothing && return true
#         return a in trace_population_decomp_indices
#     end

#     function print_trace_population_decomp_header_if_needed!()
#         if trace_population_decomp_header && !trace_population_decomp_header_printed[]
#             lock(trace_population_decomp_lock)
#             try
#                 if !trace_population_decomp_header_printed[]
#                     println(
#                         trace_population_decomp_io,
#                         "POP_DECOMP_TRACE itr t a p_a " *
#                         "REAL_rhs_diag IMAG_rhs_diag " *
#                         "REAL_local_A_raw IMAG_local_A_raw REAL_local_B_raw IMAG_local_B_raw REAL_local_raw IMAG_local_raw " *
#                         "REAL_local_A_app IMAG_local_A_app REAL_local_B_app IMAG_local_B_app REAL_local_app IMAG_local_app " *
#                         "REAL_mem_b1_pop_app IMAG_mem_b1_pop_app REAL_mem_b1_coh_app IMAG_mem_b1_coh_app " *
#                         "REAL_mem_b2_pop_app IMAG_mem_b2_pop_app REAL_mem_b2_coh_app IMAG_mem_b2_coh_app " *
#                         "REAL_mem_b3_pop_app IMAG_mem_b3_pop_app REAL_mem_b3_coh_app IMAG_mem_b3_coh_app " *
#                         "REAL_mem_b4_pop_app IMAG_mem_b4_pop_app REAL_mem_b4_coh_app IMAG_mem_b4_coh_app " *
#                         "REAL_mem_pop_app IMAG_mem_pop_app REAL_mem_coh_app IMAG_mem_coh_app REAL_mem_core_app IMAG_mem_core_app " *
#                         "REAL_rhs_total IMAG_rhs_total REAL_dp_HEOM IMAG_dp_HEOM REAL_diff IMAG_diff " *
#                         "REAL_diff_minus_local_app IMAG_diff_minus_local_app " *
#                         "REAL_diff_minus_mem_coh_app IMAG_diff_minus_mem_coh_app " *
#                         "REAL_diff_minus_local_and_mem_coh_app IMAG_diff_minus_local_and_mem_coh_app"
#                     )
#                     trace_population_decomp_header_printed[] = true
#                 end
#             finally
#                 unlock(trace_population_decomp_lock)
#             end
#         end
#         return nothing
#     end

#     function print__population_decomp_line!(
#         io::IO,
#         curr_itr::Int,
#         t::Float64,
#         a::Int,
#         p_a,
#         rhs_diag,
#         local_A_raw,
#         local_B_raw,
#         local_A_app,
#         local_B_app,
#         mem_b1_pop_app,
#         mem_b1_coh_app,
#         mem_b2_pop_app,
#         mem_b2_coh_app,
#         mem_b3_pop_app,
#         mem_b3_coh_app,
#         mem_b4_pop_app,
#         mem_b4_coh_app,
#         rhs_total,
#         heom_dp_aa,
#     )
#         local_raw = local_A_raw + local_B_raw
#         local_app = local_A_app + local_B_app

#         mem_pop_app = mem_b1_pop_app + mem_b2_pop_app + mem_b3_pop_app + mem_b4_pop_app
#         mem_coh_app = mem_b1_coh_app + mem_b2_coh_app + mem_b3_coh_app + mem_b4_coh_app
#         mem_core_app = mem_pop_app + mem_coh_app

#         diff = rhs_total - heom_dp_aa
#         diff_minus_local_app = diff - local_app
#         diff_minus_mem_coh_app = diff - mem_coh_app
#         diff_minus_local_and_mem_coh_app = diff - local_app - mem_coh_app

#         println(
#             io,
#             "POP_DECOMP_TRACE", " ",
#             curr_itr, " ",
#             fmt__time(t), " ",
#             a, " ",
#             fmt__real(real(p_a)), " ",
#             fmt__complex(rhs_diag), " ",
#             fmt__complex(local_A_raw), " ",
#             fmt__complex(local_B_raw), " ",
#             fmt__complex(local_raw), " ",
#             fmt__complex(local_A_app), " ",
#             fmt__complex(local_B_app), " ",
#             fmt__complex(local_app), " ",
#             fmt__complex(mem_b1_pop_app), " ",
#             fmt__complex(mem_b1_coh_app), " ",
#             fmt__complex(mem_b2_pop_app), " ",
#             fmt__complex(mem_b2_coh_app), " ",
#             fmt__complex(mem_b3_pop_app), " ",
#             fmt__complex(mem_b3_coh_app), " ",
#             fmt__complex(mem_b4_pop_app), " ",
#             fmt__complex(mem_b4_coh_app), " ",
#             fmt__complex(mem_pop_app), " ",
#             fmt__complex(mem_coh_app), " ",
#             fmt__complex(mem_core_app), " ",
#             fmt__complex(rhs_total), " ",
#             fmt__complex(heom_dp_aa), " ",
#             fmt__complex(diff), " ",
#             fmt__complex(diff_minus_local_app), " ",
#             fmt__complex(diff_minus_mem_coh_app), " ",
#             fmt__complex(diff_minus_local_and_mem_coh_app),
#         )

#         return nothing
#     end

#     @inline function trace__population_decomp_if_needed!(
#         curr_itr::Int,
#         σ_t::AbstractMatrix,
#         a::Int,
#         rhs_diag,
#         local_A_raw,
#         local_B_raw,
#         local_A_app,
#         local_B_app,
#         mem_b1_pop_app,
#         mem_b1_coh_app,
#         mem_b2_pop_app,
#         mem_b2_coh_app,
#         mem_b3_pop_app,
#         mem_b3_coh_app,
#         mem_b4_pop_app,
#         mem_b4_coh_app,
#         rhs_total,
#     )
#         should_trace_population_decomp(curr_itr, a) || return nothing

#         time = (curr_itr - 1) * Δt
#         p_a = real(σ_now(σ_t, curr_itr, a, a))
#         heom_dp_aa = σprime_heom_now(curr_itr, a, a)

#         print_trace_population_decomp_header_if_needed!()
#         lock(trace_population_decomp_lock)
#         try
#             print__population_decomp_line!(
#                 trace_population_decomp_io,
#                 curr_itr,
#                 time,
#                 a,
#                 p_a,
#                 rhs_diag,
#                 local_A_raw,
#                 local_B_raw,
#                 local_A_app,
#                 local_B_app,
#                 mem_b1_pop_app,
#                 mem_b1_coh_app,
#                 mem_b2_pop_app,
#                 mem_b2_coh_app,
#                 mem_b3_pop_app,
#                 mem_b3_coh_app,
#                 mem_b4_pop_app,
#                 mem_b4_coh_app,
#                 rhs_total,
#                 heom_dp_aa,
#             )
#         finally
#             unlock(trace_population_decomp_lock)
#         end

#         return nothing
#     end

#     @inline function add_if_finite(z::Complex)
#         return (isfinite(real(z)) && isfinite(imag(z))) ? z : (NaN + NaN * im)
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


#     # -------------------------------------------------------------------------
#     # Non-time-localized exponent helpers
#     # -------------------------------------------------------------------------
#     # The original exponent_type_1/type_2 functions are strict transported
#     # time-localized prefactors.  They include the input-channel backward
#     # transport factor
#     #
#     #     T_in(s,t) = exp(+i omega_in Δ) * N_in^bath(s)/N_in^bath(t).
#     #
#     # For a genuinely non-time-localized kernel multiplying σ(input, s), the
#     # scalar phase and the bath exponent must both be divided by this input
#     # transport factor.  The phase division is handled explicitly in each branch
#     # by changing phase(...).  The bath part is removed here by subtracting
#     # logN_bath(s,input)-logN_bath(t,input).

#     @inline function logT_input_bath(
#         s_itr::Int,
#         t_itr::Int,
#         in_a::Int,
#         in_b::Int,
#     )
#         return logN_bath(s_itr, in_a, in_b) - logN_bath(t_itr, in_a, in_b)
#     end

#     @inline function gen__exponent_type_1_nonlocal(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α⁻::Int,
#         α⁼::Int,
#         β::Int,
#     )
#         # Strict-TL type-1 input channel is (α⁼, β).
#         return gen__exponent_type_1(
#             s_itr,
#             Δ_itr,
#             t_itr,
#             α⁻,
#             α⁼,
#             β,
#         ) - logT_input_bath(s_itr, t_itr, α⁼, β)
#     end

#     @inline function gen__exponent_type_2_nonlocal(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         a::Int,
#         β⁼::Int,
#         β⁻::Int,
#     )
#         # Strict-TL type-2 input channel is (a, β⁼).
#         return gen__exponent_type_2(
#             s_itr,
#             Δ_itr,
#             t_itr,
#             a,
#             β⁼,
#             β⁻,
#         ) - logT_input_bath(s_itr, t_itr, a, β⁼)
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
#     # Missing first-order memory-return term
#     #   P L0 Q exp(Δ L0) Q L1 P
#     # Non-time-localized, g-closed raw-trace form.
#     # External output indices: (α, β)
#     # Branch A: input (f, β) -> output (α, β), coefficient sign = -
#     # Branch B: input (α, f) -> output (α, β), coefficient sign = +
#     # -------------------------------------------------------------------------

#     @inline function gen__L0Q_exponent_A(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         f::Int,
#     )
#         # Non-time-localized Branch A raw trace prefactor.
#         #
#         # This corresponds to the direct raw block
#         #
#         #   exp(-iω[α,β]Δ)
#         #   TrB{ exp(-iH_α Δ) B_{αf} τ_{fβ}(s) exp(+iH_β Δ) }
#         #
#         # plus the Q(s)- and Q(t)-subtracted covariance structure.  No
#         # transported σ[f,β](s) -> σ[f,β](t) replacement is used here.
#         # Therefore this is the old E_A(t,s,Δ), not Λ_A[f,α|β](t,s).
#         return (
#             -conj(g[t_itr, β, β, β, β] - g[s_itr, β, β, β, β])
#             +conj(g[t_itr, α, α, β, β] - g[s_itr, α, α, β, β])

#             +(g[t_itr, β, β, f, f] - g[s_itr, β, β, f, f] - g[Δ_itr, β, β, f, f])
#             -(g[t_itr, α, α, f, f] - g[s_itr, α, α, f, f] - g[Δ_itr, α, α, f, f])

#             -g[Δ_itr, α, α, α, α]
#             +g[Δ_itr, β, β, α, α]
#         )
#     end

#     @inline function gen__L0Q_coef_A(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         f::Int,
#     )
#         left_one_point = (
#              g′[Δ_itr, α, α, f, f]
#             -g′[t_itr, α, α, f, f]
#             -g′[Δ_itr, α, α, α, α]
#             +g′[t_itr, α, α, α, α]
#             -g′[Δ_itr, β, β, f, f]
#             +g′[t_itr, β, β, f, f]
#             +g′[Δ_itr, β, β, α, α]
#             -g′[t_itr, β, β, α, α]
#         )

#         right_one_point = (
#             -g′[s_itr, α, f, f, f]
#             -g′[Δ_itr, α, α, α, f]
#             +conj(g′[s_itr, f, α, β, β])
#             +g′[Δ_itr, β, β, α, f]
#         )

#         return (
#             g″[Δ_itr, α, α, α, f]
#             -g″[Δ_itr, β, β, α, f]
#             -left_one_point * right_one_point
#         )
#     end

#     @inline function gen__L0Q_exponent_B(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         f::Int,
#     )
#         # Non-time-localized Branch B raw trace prefactor.
#         #
#         # This corresponds to the direct raw block
#         #
#         #   exp(-iω[α,β]Δ)
#         #   TrB{ exp(-iH_α Δ) τ_{αf}(s) B_{fβ} exp(+iH_β Δ) }
#         #
#         # plus the Q(s)- and Q(t)-subtracted covariance structure.  No
#         # transported σ[α,f](s) -> σ[α,f](t) replacement is used here.
#         # Therefore this is the old E_B(t,s,Δ), not Λ_B[α|f,β](t,s).
#         return (
#             -(g[t_itr, α, α, α, α] - g[s_itr, α, α, α, α])
#             +(g[t_itr, β, β, α, α] - g[s_itr, β, β, α, α])

#             +conj(g[t_itr, α, α, f, f] - g[s_itr, α, α, f, f] - g[Δ_itr, α, α, f, f])
#             -conj(g[t_itr, β, β, f, f] - g[s_itr, β, β, f, f] - g[Δ_itr, β, β, f, f])

#             -conj(g[Δ_itr, β, β, β, β])
#             +conj(g[Δ_itr, α, α, β, β])
#         )
#     end

#     @inline function gen__L0Q_coef_B(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         f::Int,
#     )
#         left_one_point = (
#             conj(
#                 g′[t_itr, α, α, f, f]
#                 -g′[Δ_itr, α, α, f, f]
#                 +g′[Δ_itr, α, α, β, β]
#                 -g′[t_itr, α, α, β, β]
#             )
#             -conj(
#                 g′[t_itr, β, β, f, f]
#                 -g′[Δ_itr, β, β, f, f]
#                 +g′[Δ_itr, β, β, β, β]
#                 -g′[t_itr, β, β, β, β]
#             )
#         )

#         right_one_point = (
#             -g′[s_itr, f, β, α, α]
#             -conj(g′[Δ_itr, α, α, β, f])
#             +conj(g′[s_itr, β, f, f, f])
#             +conj(g′[Δ_itr, β, β, β, f])
#         )

#         # if g″[Δ_itr, β, f, α, α] == g″[Δ_itr, α, α, β, f]
#         #     @printf("meaningless1! ")
#         # end
#         # if g″[Δ_itr, β, f, β, β] == g″[Δ_itr, β, β, β, f]
#         #     @printf("meaningless2! \n")
#         # end

#         return (
#             conj(g″[Δ_itr, β, f, α, α])
#             -conj(g″[Δ_itr, β, f, β, β])
#             # conj(g″[Δ_itr, α, α, β, f])
#             # -conj(g″[Δ_itr, β, β, β, f])
#             -left_one_point * right_one_point
#         )
#     end

#     # -------------------------------------------------------------------------
#     # Population-closed path kernel
#     # Used only when use_population_closure = true.
#     # -------------------------------------------------------------------------

#     @inline function gen__population_transfer_exponent(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         src::Int,
#         dst::Int,
#     )
#         return (
#             g[t_itr, src, src, src, src]
#             - g[s_itr, src, src, src, src]

#             - g[Δ_itr, dst, dst, dst, dst]

#             - (
#                 g[t_itr, dst, dst, src, src]
#                 - g[s_itr, dst, dst, src, src]
#                 - g[Δ_itr, dst, dst, src, src]
#             )

#             - g[Δ_itr, src, src, src, src]

#             + conj(
#                 g[s_itr, src, src, src, src]
#                 - g[t_itr, src, src, src, src]
#             )

#             + g[Δ_itr, src, src, dst, dst]

#             + conj(
#                 g[t_itr, dst, dst, src, src]
#                 - g[s_itr, dst, dst, src, src]
#             )
#         )
#     end

#     @inline function gen__population_transfer_coef(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         src::Int,
#         dst::Int,
#     )
#         left_one_point = (
#             -1.0im * g′[t_itr, src, dst, src, src]
#             +1.0im * g′[Δ_itr, src, dst, src, src]
#             -1.0im * g′[Δ_itr, src, dst, dst, dst]
#             +1.0im * conj(g′[t_itr, dst, src, src, src])
#         )

#         right_one_point = (
#             -1.0im * g′[s_itr, dst, src, src, src]
#             -1.0im * g′[Δ_itr, dst, dst, dst, src]
#             +1.0im * conj(g′[s_itr, src, dst, src, src])
#             +1.0im * g′[Δ_itr, src, src, dst, src]
#         )

#         return (
#             g″[Δ_itr, src, dst, dst, src]
#             + left_one_point * right_one_point
#         )
#     end

#     @inline function gen__population_transfer_kernel(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         src::Int,
#         dst::Int,
#     )
#         return 2.0 * real(
#             phase(dst, src, Δ_itr)
#             * exp(gen__population_transfer_exponent(
#                 s_itr,
#                 Δ_itr,
#                 t_itr,
#                 src,
#                 dst,
#             ))
#             * gen__population_transfer_coef(
#                 s_itr,
#                 Δ_itr,
#                 t_itr,
#                 src,
#                 dst,
#             )
#         )
#     end

#     @inline function calc__population_closed_rhs(
#         α::Int,
#         curr_itr::Int,
#         σ_t::AbstractMatrix,
#     )
#         curr_itr > 1 || return (
#             0.0 + 0.0im,
#             0.0 + 0.0im,
#             0.0 + 0.0im,
#         )

#         rhs  = 0.0
#         loss = 0.0
#         gain = 0.0

#         for s_itr in 1:curr_itr
#             Δ_itr = curr_itr - s_itr + 1
#             w_int = ∫weight(s_itr, curr_itr)
#             pα = real(σ_mem(σ_t, s_itr, curr_itr, α, α))

#             for f in 1:n_sys
#                 f == α && continue

#                 pf = real(σ_mem(σ_t, s_itr, curr_itr, f, f))

#                 # α -> f loss
#                 k_loss = gen__population_transfer_kernel(
#                     s_itr,
#                     Δ_itr,
#                     curr_itr,
#                     α,
#                     f,
#                 )

#                 # f -> α gain
#                 k_gain = gen__population_transfer_kernel(
#                     s_itr,
#                     Δ_itr,
#                     curr_itr,
#                     f,
#                     α,
#                 )

#                 loss_contrib = -pα * k_loss
#                 gain_contrib =  pf * k_gain

#                 rhs  += w_int * (loss_contrib + gain_contrib)
#                 loss += w_int * loss_contrib
#                 gain += w_int * gain_contrib
#             end
#         end

#         return (
#             rhs + 0.0im,
#             loss + 0.0im,
#             gain + 0.0im,
#         )
#     end

#     # -------------------------------------------------------------------------
#     # RHS evaluator
#     # -------------------------------------------------------------------------
#     # The memory kernel, g/g′/g″, and secular filters are evaluated at curr_itr.
#     # The memory integrals use σ(s_itr) for s < t and σ_t only at the moving
#     # endpoint s = t.  This is the non-time-localized memory form.
#     # This avoids requiring half-step interpolation of precomputed g-arrays.
#     # -------------------------------------------------------------------------

#     @inline function calc__rhs_element!(
#         rhs_mat::AbstractMatrix,
#         curr_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#     )

#         # -----------------------------------------------------------------
#         # Optional population closure.
#         # If disabled, populations are propagated by the secularized
#         # full RMRT memory equation below.
#         # -----------------------------------------------------------------
#         if use_population_closure && α == β
#             rhs_pop, pop_loss, pop_gain = calc__population_closed_rhs(
#                 α,
#                 curr_itr,
#                 σ_t,
#             )

#             rhs_mat[α, α] = rhs_pop
#             trace__population_rhs_if_needed!(curr_itr, σ_t, α, rhs_pop)
#             trace__population_decomp_if_needed!(
#                 curr_itr,
#                 σ_t,
#                 α,
#                 0.0 + 0.0im,
#                 0.0 + 0.0im,
#                 0.0 + 0.0im,
#                 0.0 + 0.0im,
#                 0.0 + 0.0im,
#                 rhs_pop,
#                 0.0 + 0.0im,
#                 0.0 + 0.0im,
#                 0.0 + 0.0im,
#                 0.0 + 0.0im,
#                 0.0 + 0.0im,
#                 0.0 + 0.0im,
#                 0.0 + 0.0im,
#                 rhs_pop,
#             )
#             return nothing
#         end

#         # -----------------------------------------------------------------
#         # t-local diagonal / coherence phase block.
#         # This is self-coupling, so it is always secular.
#         # For α == β this block algebraically vanishes.
#         # -----------------------------------------------------------------
#         rhs_diag = (
#             -1.0im * ω(α, β)
#             -g′[curr_itr, α, α, α, α]
#             +conj(g′[curr_itr, α, α, β, β])
#             +g′[curr_itr, β, β, α, α]
#             -conj(g′[curr_itr, β, β, β, β])
#         ) * σ_now(σ_t, curr_itr, α, β)

#         rhs = rhs_diag

#         verify_this = should_verify_L0Q(curr_itr, α, β)
#         trace_phase_this = should_trace_phase(curr_itr, α, β)
#         trace_pop_decomp_this = (α == β) && should_trace_population_decomp(curr_itr, α)

#         phase_local_A_raw = 0.0 + 0.0im
#         phase_local_B_raw = 0.0 + 0.0im
#         phase_local_A_app = 0.0 + 0.0im
#         phase_local_B_app = 0.0 + 0.0im
#         phase_mem_core_app = 0.0 + 0.0im
#         phase_L0Q_A_raw = 0.0 + 0.0im
#         phase_L0Q_B_raw = 0.0 + 0.0im
#         phase_L0Q_A_app = 0.0 + 0.0im
#         phase_L0Q_B_app = 0.0 + 0.0im
#         verify_local_A_pop_to_coh = 0.0 + 0.0im
#         verify_local_B_pop_to_coh = 0.0 + 0.0im
#         verify_local_pop_to_coh   = 0.0 + 0.0im
#         verify_L0Q_A_pop          = 0.0 + 0.0im
#         verify_L0Q_B_pop          = 0.0 + 0.0im
#         verify_L0Q_A_all          = 0.0 + 0.0im
#         verify_L0Q_B_all          = 0.0 + 0.0im

#         pop_local_A_raw = 0.0 + 0.0im
#         pop_local_B_raw = 0.0 + 0.0im
#         pop_local_A_app = 0.0 + 0.0im
#         pop_local_B_app = 0.0 + 0.0im

#         pop_mem_b1_pop_app = 0.0 + 0.0im
#         pop_mem_b1_coh_app = 0.0 + 0.0im
#         pop_mem_b2_pop_app = 0.0 + 0.0im
#         pop_mem_b2_coh_app = 0.0 + 0.0im
#         pop_mem_b3_pop_app = 0.0 + 0.0im
#         pop_mem_b3_coh_app = 0.0 + 0.0im
#         pop_mem_b4_pop_app = 0.0 + 0.0im
#         pop_mem_b4_coh_app = 0.0 + 0.0im

#         # -----------------------------------------------------------------
#         # t-local left mixing:
#         # input component  (α⁻, β)
#         # output component (α, β)
#         # keep if ω(α,β) ≈ ω(α⁻,β).
#         #
#         # If use_local_population_to_coherence = false, remove only the
#         # local population -> coherence source σ[β,β] -> σ[α,β], i.e.
#         # output α != β and input α⁻ == β.
#         # -----------------------------------------------------------------
#         for α⁻ in 1:n_sys
#             α⁻ == α && continue

#             if is_secular_pair(α, β, α⁻, β)
#                 local_left_term = σ_now(σ_t, curr_itr, α⁻, β) * (
#                     g′[curr_itr, α, α⁻, α⁻, α⁻]
#                     - conj(g′[curr_itr, α⁻, α, β, β])
#                 )

#                 local_left_contrib = -local_left_term
#                 phase_local_A_raw += local_left_contrib

#                 if trace_pop_decomp_this
#                     pop_local_A_raw += local_left_contrib
#                 end

#                 if verify_this && is_local_population_to_coherence(α, β, α⁻, β)
#                     verify_local_A_pop_to_coh += local_left_contrib
#                     verify_local_pop_to_coh   += local_left_contrib
#                 end

#                 if !use_local_population_to_coherence &&
#                    is_local_population_to_coherence(α, β, α⁻, β)
#                     continue
#                 end

#                 if !use_local_coherence_to_population &&
#                    is_local_coherence_to_population(α, β, α⁻, β)
#                     continue
#                 end

#                 phase_local_A_app += local_left_contrib
#                 if trace_pop_decomp_this
#                     pop_local_A_app += local_left_contrib
#                 end
#                 rhs += local_left_contrib
#             end
#         end

#         # -----------------------------------------------------------------
#         # t-local right mixing:
#         # input component  (α, β⁻)
#         # output component (α, β)
#         # keep if ω(α,β) ≈ ω(α,β⁻).
#         #
#         # If use_local_population_to_coherence = false, remove only the
#         # local population -> coherence source σ[α,α] -> σ[α,β], i.e.
#         # output α != β and input β⁻ == α.
#         # -----------------------------------------------------------------
#         for β⁻ in 1:n_sys
#             β⁻ == β && continue

#             if is_secular_pair(α, β, α, β⁻)
#                 local_right_term = σ_now(σ_t, curr_itr, α, β⁻) * (
#                     g′[curr_itr, β⁻, β, α, α]
#                     - conj(g′[curr_itr, β, β⁻, β⁻, β⁻])
#                 )

#                 local_right_contrib = local_right_term
#                 phase_local_B_raw += local_right_contrib

#                 if trace_pop_decomp_this
#                     pop_local_B_raw += local_right_contrib
#                 end

#                 if verify_this && is_local_population_to_coherence(α, β, α, β⁻)
#                     verify_local_B_pop_to_coh += local_right_contrib
#                     verify_local_pop_to_coh   += local_right_contrib
#                 end

#                 if !use_local_population_to_coherence &&
#                    is_local_population_to_coherence(α, β, α, β⁻)
#                     continue
#                 end

#                 if !use_local_coherence_to_population &&
#                    is_local_coherence_to_population(α, β, α, β⁻)
#                     continue
#                 end

#                 phase_local_B_app += local_right_contrib
#                 if trace_pop_decomp_this
#                     pop_local_B_app += local_right_contrib
#                 end
#                 rhs += local_right_contrib
#             end
#         end

#         # -----------------------------------------------------------------
#         # Memory integral
#         # -----------------------------------------------------------------
#         if curr_itr > 1
#             integral = 0.0 + 0.0im

#             for s_itr in 1:curr_itr
#                 Δ_itr = curr_itr - s_itr + 1
#                 w_int = ∫weight(s_itr, curr_itr)
#                 kernel = 0.0 + 0.0im

#                 # ---------------------------------------------------------
#                 # Branch 1
#                 #
#                 # input component  (α⁼, β)
#                 # output component (α, β)
#                 # keep if ω(α,β) ≈ ω(α⁼,β).
#                 # ---------------------------------------------------------
#                 for α⁻ in 1:n_sys
#                     α⁻ == α && continue

#                     for α⁼ in 1:n_sys
#                         α⁼ == α⁻ && continue

#                         if is_secular_pair(α, β, α⁼, β)
#                             branch1_contrib = -(
#                                 σ_mem(σ_t, s_itr, curr_itr, α⁼, β)
#                                 * phase(α⁻, β, Δ_itr)
#                                 * exp(gen__exponent_type_1_nonlocal(
#                                     s_itr,
#                                     Δ_itr,
#                                     curr_itr,
#                                     α⁻,
#                                     α⁼,
#                                     β,
#                                 ))
#                                 * gen_coef_block_type_1(
#                                     s_itr,
#                                     Δ_itr,
#                                     curr_itr,
#                                     α,
#                                     α⁻,
#                                     α⁼,
#                                     β,
#                                 )
#                             )
#                             if should_skip_population_memory_input(α, β, α⁼, β)
#                                 continue
#                             end

#                             kernel += branch1_contrib
#                             phase_mem_core_app += w_int * branch1_contrib

#                             if trace_pop_decomp_this
#                                 if is_population_input(α⁼, β)
#                                     pop_mem_b1_pop_app += w_int * branch1_contrib
#                                 else
#                                     pop_mem_b1_coh_app += w_int * branch1_contrib
#                                 end
#                             end
#                         end
#                     end
#                 end

#                 # ---------------------------------------------------------
#                 # Branch 2
#                 #
#                 # input component  (α, β⁼)
#                 # output component (α, β)
#                 # keep if ω(α,β) ≈ ω(α,β⁼).
#                 # ---------------------------------------------------------
#                 for β⁻ in 1:n_sys
#                     β⁻ == β && continue

#                     for β⁼ in 1:n_sys
#                         β⁼ == β⁻ && continue

#                         if is_secular_pair(α, β, α, β⁼)
#                             branch2_contrib = -(
#                                 σ_mem(σ_t, s_itr, curr_itr, α, β⁼)
#                                 * phase(α, β⁻, Δ_itr)
#                                 * exp(gen__exponent_type_2_nonlocal(
#                                     s_itr,
#                                     Δ_itr,
#                                     curr_itr,
#                                     α,
#                                     β⁼,
#                                     β⁻,
#                                 ))
#                                 * gen_coef_block_type_2(
#                                     s_itr,
#                                     Δ_itr,
#                                     curr_itr,
#                                     β,
#                                     β⁻,
#                                     β⁼,
#                                     α,
#                                 )
#                             )
#                             if should_skip_population_memory_input(α, β, α, β⁼)
#                                 continue
#                             end

#                             kernel += branch2_contrib
#                             phase_mem_core_app += w_int * branch2_contrib

#                             if trace_pop_decomp_this
#                                 if is_population_input(α, β⁼)
#                                     pop_mem_b2_pop_app += w_int * branch2_contrib
#                                 else
#                                     pop_mem_b2_coh_app += w_int * branch2_contrib
#                                 end
#                             end
#                         end
#                     end
#                 end

#                 # ---------------------------------------------------------
#                 # Branch 3
#                 #
#                 # input component  (α⁻, β⁻)
#                 # output component (α, β)
#                 # keep if ω(α,β) ≈ ω(α⁻,β⁻).
#                 # ---------------------------------------------------------
#                 for α⁻ in 1:n_sys
#                     α⁻ == α && continue

#                     for β⁻ in 1:n_sys
#                         β⁻ == β && continue

#                         if is_secular_pair(α, β, α⁻, β⁻)
#                             branch3_contrib = (
#                                 σ_mem(σ_t, s_itr, curr_itr, α⁻, β⁻)
#                                 * phase(α, β⁻, Δ_itr)
#                                 * exp(gen__exponent_type_1_nonlocal(
#                                     s_itr,
#                                     Δ_itr,
#                                     curr_itr,
#                                     α,
#                                     α⁻,
#                                     β⁻,
#                                 ))
#                                 * gen_coef_block_type_3(
#                                     s_itr,
#                                     Δ_itr,
#                                     curr_itr,
#                                     α,
#                                     β,
#                                     α⁻,
#                                     β⁻,
#                                 )
#                             )
#                             if should_skip_population_memory_input(α, β, α⁻, β⁻)
#                                 continue
#                             end

#                             kernel += branch3_contrib
#                             phase_mem_core_app += w_int * branch3_contrib

#                             if trace_pop_decomp_this
#                                 if is_population_input(α⁻, β⁻)
#                                     pop_mem_b3_pop_app += w_int * branch3_contrib
#                                 else
#                                     pop_mem_b3_coh_app += w_int * branch3_contrib
#                                 end
#                             end
#                         end
#                     end
#                 end

#                 # ---------------------------------------------------------
#                 # Branch 4
#                 #
#                 # input component  (α⁻, β⁻)
#                 # output component (α, β)
#                 # keep if ω(α,β) ≈ ω(α⁻,β⁻).
#                 # ---------------------------------------------------------
#                 for α⁻ in 1:n_sys
#                     α⁻ == α && continue

#                     for β⁻ in 1:n_sys
#                         β⁻ == β && continue

#                         if is_secular_pair(α, β, α⁻, β⁻)
#                             branch4_contrib = (
#                                 σ_mem(σ_t, s_itr, curr_itr, α⁻, β⁻)
#                                 * phase(α⁻, β, Δ_itr)
#                                 * exp(gen__exponent_type_2_nonlocal(
#                                     s_itr,
#                                     Δ_itr,
#                                     curr_itr,
#                                     α⁻,
#                                     β⁻,
#                                     β,
#                                 ))
#                                 * gen_coef_block_type_4(
#                                     s_itr,
#                                     Δ_itr,
#                                     curr_itr,
#                                     α,
#                                     β,
#                                     α⁻,
#                                     β⁻,
#                                 )
#                             )
#                             if should_skip_population_memory_input(α, β, α⁻, β⁻)
#                                 continue
#                             end

#                             kernel += branch4_contrib
#                             phase_mem_core_app += w_int * branch4_contrib

#                             if trace_pop_decomp_this
#                                 if is_population_input(α⁻, β⁻)
#                                     pop_mem_b4_pop_app += w_int * branch4_contrib
#                                 else
#                                     pop_mem_b4_coh_app += w_int * branch4_contrib
#                                 end
#                             end
#                         end
#                     end
#                 end

#                 # ---------------------------------------------------------
#                 # Missing first-order memory-return term
#                 #   P L0 Q exp(Δ L0) Q L1 P
#                 # This term vanishes for population output (α == β), but is
#                 # generally needed for coherence output.
#                 #
#                 # Non-time-localized form is used here:
#                 #   Branch A uses σ[f,β](s), not σ[f,β](t).
#                 #   Branch B uses σ[α,f](s), not σ[α,f](t).
#                 # Therefore the raw output-block propagation phase remains
#                 # phase(α,β,Δ), and the raw E_A/E_B prefactors are used rather
#                 # than the transported Λ_A/Λ_B prefactors.
#                 # ---------------------------------------------------------
#                 if (use_L0Q_memory_return || verify_this || trace_phase_this) && α != β
#                     # Branch A: input (f, β) -> output (α, β), sign = -
#                     for f in 1:n_sys
#                         f == α && continue

#                         if is_secular_pair(α, β, f, β)
#                             L0Q_A_term = (
#                                 σ_mem(σ_t, s_itr, curr_itr, f, β)
#                                 * phase(α, β, Δ_itr)
#                                 * exp(gen__L0Q_exponent_A(
#                                     s_itr,
#                                     Δ_itr,
#                                     curr_itr,
#                                     α,
#                                     β,
#                                     f,
#                                 ))
#                                 * gen__L0Q_coef_A(
#                                     s_itr,
#                                     Δ_itr,
#                                     curr_itr,
#                                     α,
#                                     β,
#                                     f,
#                                 )
#                             )
#                             L0Q_A_contrib = -L0Q_A_term

#                             phase_L0Q_A_raw += w_int * L0Q_A_contrib

#                             if verify_this
#                                 verify_L0Q_A_all += w_int * L0Q_A_contrib
#                                 if is_local_population_to_coherence(α, β, f, β)
#                                     verify_L0Q_A_pop += w_int * L0Q_A_contrib
#                                 end
#                             end

#                             if !use_local_population_to_coherence &&
#                                is_local_population_to_coherence(α, β, f, β)
#                                 continue
#                             end

#                             if use_L0Q_memory_return
#                                 kernel += L0Q_A_contrib
#                                 phase_L0Q_A_app += w_int * L0Q_A_contrib
#                             end
#                         end
#                     end

#                     # Branch B: input (α, f) -> output (α, β), sign = +
#                     for f in 1:n_sys
#                         f == β && continue

#                         if is_secular_pair(α, β, α, f)
#                             L0Q_B_contrib = (
#                                 σ_mem(σ_t, s_itr, curr_itr, α, f)
#                                 * phase(α, β, Δ_itr)
#                                 * exp(gen__L0Q_exponent_B(
#                                     s_itr,
#                                     Δ_itr,
#                                     curr_itr,
#                                     α,
#                                     β,
#                                     f,
#                                 ))
#                                 * gen__L0Q_coef_B(
#                                     s_itr,
#                                     Δ_itr,
#                                     curr_itr,
#                                     α,
#                                     β,
#                                     f,
#                                 )
#                             )

#                             phase_L0Q_B_raw += w_int * L0Q_B_contrib

#                             if verify_this
#                                 verify_L0Q_B_all += w_int * L0Q_B_contrib
#                                 if is_local_population_to_coherence(α, β, α, f)
#                                     verify_L0Q_B_pop += w_int * L0Q_B_contrib
#                                 end
#                             end

#                             if !use_local_population_to_coherence &&
#                                is_local_population_to_coherence(α, β, α, f)
#                                 continue
#                             end

#                             if use_L0Q_memory_return
#                                 kernel += L0Q_B_contrib
#                                 phase_L0Q_B_app += w_int * L0Q_B_contrib
#                             end
#                         end
#                     end
#                 end


#                 integral += w_int * kernel
#             end

#             rhs += integral
#         end

#         if verify_this
#             L0Q_pop = verify_L0Q_A_pop + verify_L0Q_B_pop
#             L0Q_all = verify_L0Q_A_all + verify_L0Q_B_all
#             local_A_plus_L0Q_A_pop = verify_local_A_pop_to_coh + verify_L0Q_A_pop
#             local_B_plus_L0Q_B_pop = verify_local_B_pop_to_coh + verify_L0Q_B_pop
#             local_plus_L0Q_pop = verify_local_pop_to_coh + L0Q_pop
#             time = (curr_itr - 1) * Δt

#             print_verify_L0Q_header_if_needed!()
#             lock(verify_lock)
#             try
#                 print__L0Q_verify_line!(
#                     verify_L0Q_io,
#                     curr_itr,
#                     time,
#                     α,
#                     β,
#                     verify_local_A_pop_to_coh,
#                     verify_local_B_pop_to_coh,
#                     verify_local_pop_to_coh,
#                     verify_L0Q_A_pop,
#                     verify_L0Q_B_pop,
#                     L0Q_pop,
#                     local_A_plus_L0Q_A_pop,
#                     local_B_plus_L0Q_B_pop,
#                     local_plus_L0Q_pop,
#                     verify_L0Q_A_all,
#                     verify_L0Q_B_all,
#                     L0Q_all,
#                     rhs,
#                 )
#             finally
#                 unlock(verify_lock)
#             end
#         end

#         if trace_phase_this
#             time = (curr_itr - 1) * Δt
#             c_ab = σ_now(σ_t, curr_itr, α, β)
#             rhs_local_raw = phase_local_A_raw + phase_local_B_raw
#             rhs_local_app = phase_local_A_app + phase_local_B_app
#             rhs_L0Q_raw = phase_L0Q_A_raw + phase_L0Q_B_raw
#             rhs_L0Q_app = phase_L0Q_A_app + phase_L0Q_B_app

#             print_trace_phase_header_if_needed!()
#             lock(trace_phase_lock)
#             try
#                 print__phase_trace_line!(
#                     trace_phase_io,
#                     curr_itr,
#                     time,
#                     α,
#                     β,
#                     c_ab,
#                     rhs_diag,
#                     rhs_local_raw,
#                     rhs_local_app,
#                     phase_mem_core_app,
#                     rhs_L0Q_raw,
#                     rhs_L0Q_app,
#                     phase_L0Q_A_raw,
#                     phase_L0Q_B_raw,
#                     rhs,
#                     σprime_heom_now(curr_itr, α, β),
#                 )
#             finally
#                 unlock(trace_phase_lock)
#             end
#         end

#         if trace_pop_decomp_this
#             trace__population_decomp_if_needed!(
#                 curr_itr,
#                 σ_t,
#                 α,
#                 rhs_diag,
#                 pop_local_A_raw,
#                 pop_local_B_raw,
#                 pop_local_A_app,
#                 pop_local_B_app,
#                 pop_mem_b1_pop_app,
#                 pop_mem_b1_coh_app,
#                 pop_mem_b2_pop_app,
#                 pop_mem_b2_coh_app,
#                 pop_mem_b3_pop_app,
#                 pop_mem_b3_coh_app,
#                 pop_mem_b4_pop_app,
#                 pop_mem_b4_coh_app,
#                 rhs,
#             )
#         end

#         rhs_mat[α, β] = rhs

#         if α == β
#             trace__population_rhs_if_needed!(curr_itr, σ_t, α, rhs)
#         end

#         return nothing
#     end

#     @inline function calc__rhs!(
#         rhs_mat::AbstractMatrix,
#         curr_itr::Int,
#         σ_t::AbstractMatrix,
#     )
#         fill!(rhs_mat, 0.0 + 0.0im)

#         # ---------------------------------------------------------------------
#         # Main loop
#         # ---------------------------------------------------------------------
#         # curr_itr must remain sequential because σ[:,:,curr_itr+1] depends on
#         # previous time slices. The independent output components (α, β) of the
#         # RHS can be computed safely in parallel because each thread writes to a
#         # distinct rhs_mat[α, β] entry and only reads g/g′/g″ and σ_t.
#         # ---------------------------------------------------------------------
#         n_components = n_sys * n_sys

#         if use_threads && Threads.nthreads() > 1 && n_components > 1
#             Threads.@threads for linear_idx in 1:n_components
#                 @inbounds begin
#                     α = ((linear_idx - 1) % n_sys) + 1
#                     β = ((linear_idx - 1) ÷ n_sys) + 1

#                     calc__rhs_element!(
#                         rhs_mat,
#                         curr_itr,
#                         σ_t,
#                         α,
#                         β,
#                     )
#                 end
#             end
#         else
#             @inbounds for β in 1:n_sys, α in 1:n_sys
#                 calc__rhs_element!(
#                     rhs_mat,
#                     curr_itr,
#                     σ_t,
#                     α,
#                     β,
#                 )
#             end
#         end

#         return rhs_mat
#     end

#     @inline function enforce_hermiticity!(
#         σ_next::AbstractMatrix,
#     )
#         for i in 1:n_sys
#             σ_next[i, i] = real(σ_next[i, i]) + 0.0im
#         end

#         for i in 1:n_sys-1
#             for j in i+1:n_sys
#                 c = 0.5 * (
#                     σ_next[i, j]
#                     +
#                     conj(σ_next[j, i])
#                 )

#                 σ_next[i, j] = c
#                 σ_next[j, i] = conj(c)
#             end
#         end

#         return σ_next
#     end

#     # -------------------------------------------------------------------------
#     # RK work buffers
#     # -------------------------------------------------------------------------

#     σ_stage = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k2      = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k3      = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k4      = Matrix{ComplexF64}(undef, n_sys, n_sys)

#     # -------------------------------------------------------------------------
#     # Time propagation loop
#     # -------------------------------------------------------------------------

#     @inbounds for curr_itr in start_itr:(n_itr - 1)

#         if verbose
#             @printf(
#                 stderr,
#                 "Current iteration: %6d / %6d  method=%s  threads=%d  use_threads=%s  local_pop_to_coh=%s  local_coh_to_pop=%s  mem_pop_input=%s  mem_coh_input=%s  L0Q_return=%s  verify_L0Q=%s  heom_input=%s  teacher_cutoff=%s
# ",
#                 curr_itr,
#                 n_itr,
#                 String(method_sym),
#                 Threads.nthreads(),
#                 string(use_threads),
#                 string(use_local_population_to_coherence),
#                 string(use_local_coherence_to_population),
#                 string(use_population_memory_population_input),
#                 string(use_population_memory_coherence_input),
#                 string(use_L0Q_memory_return),
#                 string(verify_L0Q_terms),
#                 string(use_heom_input),
#                 teacher_forcing_enabled ? string(Float64(heom_teacher_forcing_cutoff)) : "nothing",
#             )
#         end

#         verify__L0P_transport_at!(curr_itr)

#         if use_heom_for_storage_itr(curr_itr)
#             @views σ[:, :, curr_itr] .= heom_sigma_ref[:, :, curr_itr]
#         end

#         σ_t    = @view σ[:, :, curr_itr]
#         σ_next = @view σ[:, :, curr_itr + 1]
#         k1     = @view σ′[:, :, curr_itr]

#         if use_heom_input ||
#            (teacher_forcing_enabled && curr_itr < teacher_cutoff_itr)

#             # Before the teacher-forcing cutoff, evaluate diagnostics/RHS on
#             # HEOM and overwrite the next stored state by HEOM.
#             # At curr_itr == teacher_cutoff_itr, this branch is skipped and the
#             # first self-propagated step begins from the HEOM cutoff state.
#             calc__rhs!(k1, curr_itr, σ_t)
#             @views σ_next .= heom_sigma_ref[:, :, curr_itr + 1]

#         elseif method_sym == :euler
#             calc__rhs!(k1, curr_itr, σ_t)

#             @. σ_next = σ_t + Δt * k1

#         elseif method_sym == :rk2
#             calc__rhs!(k1, curr_itr, σ_t)

#             @. σ_stage = σ_t + 0.5 * Δt * k1
#             calc__rhs!(k2, curr_itr, σ_stage)

#             @. σ_next = σ_t + Δt * k2

#         elseif method_sym == :rk4
#             calc__rhs!(k1, curr_itr, σ_t)

#             @. σ_stage = σ_t + 0.5 * Δt * k1
#             calc__rhs!(k2, curr_itr, σ_stage)

#             @. σ_stage = σ_t + 0.5 * Δt * k2
#             calc__rhs!(k3, curr_itr, σ_stage)

#             @. σ_stage = σ_t + Δt * k3
#             calc__rhs!(k4, curr_itr, σ_stage)

#             @. σ_next = σ_t + (Δt / 6.0) * (
#                 k1 + 2.0 * k2 + 2.0 * k3 + k4
#             )
#         end

#         enforce_hermiticity!(σ_next)

#         context.curr_itr = UInt64(curr_itr + 1)
#     end

#     return @view σ[:, :, Int(context.curr_itr)]
# end


# function calc__σ_σ′_secular_core!(
#     context::RmrtContext;
#     hbar::Real = 1.0,
#     use_population_closure::Bool = false,

#     # P L0 P local self block.
#     include_PL0P_local::Bool = true,

#     # P L1 P local left/right mixing block.
#     include_PL1P_local::Bool = true,

#     include_PL0QG0QL1P::Bool = true,
#     include_PL1QG0QL1P::Bool = true,
#     use_secular::Bool = false,
#     secular_tol::Float64 = 1.0e-10,
#     method::Union{Symbol,String} = :rk4,
#     use_threads::Bool = true,

#     # Keep the original half-shifted g-grid preparation hook.
#     # RK2/RK4 stage-time support depends on this cache in the wider code base,
#     # even if this explicit-kernel refactor currently evaluates g-series on the
#     # integer grid inside calc__rhs!.
#     auto_prepare_half_shifted_grid::Bool = true,
#     recompute_half_shifted_grid::Bool = false,

#     enforce_hermitian::Bool = true,
#     verbose::Bool = true,
# )
#     start_itr = Int(context.curr_itr)

#     n_sys = context.system.n_sys
#     n_itr = context.simulation_details.num_of_iteration
#     Δt    = context.simulation_details.Δt
#     ϵ     = context.ϵ_exci

#     σ  = context.σ
#     σ′ = context.σ′
#     g  = context.g
#     g′ = context.g′
#     g″ = context.g″

#     hbar_f = Float64(hbar)
#     hbar_f > 0.0 || error("hbar must be positive")
#     hbar2 = hbar_f^2
#     hbar4 = hbar_f^4

#     method_sym = Symbol(lowercase(String(method)))
#     method_sym in (:euler, :rk2, :rk4) || error(
#         "Unsupported integration method: $(method). Use :euler, :rk2, or :rk4."
#     )

#     needs_half_shifted_grid = method_sym in (:rk2, :rk4)

#     if needs_half_shifted_grid
#         if auto_prepare_half_shifted_grid
#             ensure__half_shifted_grid!(
#                 context;
#                 recompute = recompute_half_shifted_grid,
#                 use_threads = use_threads,
#                 verbose = verbose,
#             )
#         elseif !context.using_half_shifted_grid || !has__half_shifted_grid(context)
#             error(
#                 "method=$(method_sym) requires half-shifted g-series cache. " *
#                 "Call ensure__half_shifted_grid!(context) or " *
#                 "calc__g_g′_g″_half_shifted!(context) before propagation, " *
#                 "or set auto_prepare_half_shifted_grid=true."
#             )
#         end
#     end

#     start_itr < n_itr || return @view σ[:, :, n_itr]

#     @inline ω(a::Int, b::Int) = ϵ[a] - ϵ[b]

#     @inline function is_secular_pair(
#         out_a::Int,
#         out_b::Int,
#         in_a::Int,
#         in_b::Int,
#     )
#         if !use_secular
#             return true
#         end
#         return abs(ω(out_a, out_b) - ω(in_a, in_b)) <= secular_tol
#     end

#     @inline function ∫weight(s_itr::Int, curr_itr::Int)
#         return (s_itr == 1 || s_itr == curr_itr) ? 0.5 * Δt : Δt
#     end

#     @inline function Δtime(Δ_itr::Int)
#         return (Δ_itr - 1) * Δt
#     end

#     @inline function σ_mem(
#         σ_t::AbstractMatrix,
#         s_itr::Int,
#         curr_itr::Int,
#         a::Int,
#         b::Int,
#     )
#         return s_itr == curr_itr ? σ_t[a, b] : σ[a, b, s_itr]
#     end

#     @inline function phase_exp(from_a::Int, from_b::Int, Δ_itr::Int)
#         # exp[-i (epsilon_a - epsilon_b) Δ / hbar]
#         return exp(-1.0im * ω(from_a, from_b) * Δtime(Δ_itr) / hbar_f)
#     end

#     @inline function D_L0P(
#         itr::Int,
#         α::Int,
#         β::Int,
#     )
#         return (
#             -1.0im * ω(α, β) / hbar_f
#             +(
#                 -g′[itr, α, α, α, α]
#                 +conj(g′[itr, α, α, β, β])
#                 +g′[itr, β, β, α, α]
#                 -conj(g′[itr, β, β, β, β])
#             ) / hbar2
#         )
#     end

#     # ---------------------------------------------------------------------
#     # Explicit PL1 Q G0 Q L1 P, g-closed branch prefactors and coefficients.
#     # Branch naming follows the trace side of the two L1 insertions.
#     # All formulas are written with explicit hbar factors.  With hbar=1 this
#     # reduces to the previous internal convention.
#     # ---------------------------------------------------------------------

#     @inline function exponent_LL(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         return (
#             g[Δ_itr, β, β, χ, χ]
#             -g[s_itr, β, β, μ, μ]
#             +g[t_itr, β, β, μ, μ]
#             -g[Δ_itr, β, β, μ, μ]
#             -g[Δ_itr, χ, χ, χ, χ]
#             +g[s_itr, χ, χ, μ, μ]
#             -g[t_itr, χ, χ, μ, μ]
#             +g[Δ_itr, χ, χ, μ, μ]
#             +conj(g[s_itr, β, β, β, β])
#             -conj(g[t_itr, β, β, β, β])
#             -conj(g[s_itr, χ, χ, β, β])
#             +conj(g[t_itr, χ, χ, β, β])
#         )
#     end

#     @inline function coef_LL(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         left_bracket = (
#             g′[t_itr, α, χ, χ, χ]
#             -g′[Δ_itr, α, χ, χ, χ]
#             -g′[t_itr, α, χ, μ, μ]
#             +g′[Δ_itr, α, χ, μ, μ]
#         )

#         right_bracket = (
#             g′[Δ_itr, β, β, χ, μ]
#             -g′[Δ_itr, χ, χ, χ, μ]
#             -g′[s_itr, χ, μ, μ, μ]
#             +conj(g′[s_itr, μ, χ, β, β])
#         )

#         return hbar2 * g″[Δ_itr, α, χ, χ, μ] - left_bracket * right_bracket
#     end

#     @inline function kernel_LL(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         return -(
#             σ_mem(σ_t, s_itr, t_itr, μ, β)
#             * phase_exp(χ, β, Δ_itr)
#             * exp(exponent_LL(s_itr, Δ_itr, t_itr, α, β, χ, μ) / hbar2)
#             * coef_LL(s_itr, Δ_itr, t_itr, α, β, χ, μ) / hbar4
#         )
#     end

#     @inline function exponent_LR(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         return (
#             -g[s_itr, β, β, χ, χ]
#             +g[t_itr, β, β, χ, χ]
#             +g[s_itr, χ, χ, χ, χ]
#             -g[t_itr, χ, χ, χ, χ]
#             -conj(g[Δ_itr, β, β, β, β])
#             +conj(g[s_itr, β, β, ν, ν])
#             -conj(g[t_itr, β, β, ν, ν])
#             +conj(g[Δ_itr, β, β, ν, ν])
#             +conj(g[Δ_itr, χ, χ, β, β])
#             -conj(g[s_itr, χ, χ, ν, ν])
#             +conj(g[t_itr, χ, χ, ν, ν])
#             -conj(g[Δ_itr, χ, χ, ν, ν])
#         )
#     end

#     @inline function coef_LR(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         right_bracket = (
#             g′[s_itr, ν, β, χ, χ]
#             -conj(g′[Δ_itr, β, β, β, ν])
#             -conj(g′[s_itr, β, ν, ν, ν])
#             +conj(g′[Δ_itr, χ, χ, β, ν])
#         )

#         left_bracket = (
#             conj(g′[t_itr, χ, α, β, β])
#             -conj(g′[Δ_itr, χ, α, β, β])
#             -conj(g′[t_itr, χ, α, ν, ν])
#             +conj(g′[Δ_itr, χ, α, ν, ν])
#         )

#         return hbar2 * conj(g″[Δ_itr, χ, α, β, ν]) - right_bracket * left_bracket
#     end

#     @inline function kernel_LR(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         return (
#             σ_mem(σ_t, s_itr, t_itr, χ, ν)
#             * phase_exp(χ, β, Δ_itr)
#             * exp(exponent_LR(s_itr, Δ_itr, t_itr, α, β, χ, ν) / hbar2)
#             * coef_LR(s_itr, Δ_itr, t_itr, α, β, χ, ν) / hbar4
#         )
#     end

#     @inline function exponent_RL(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         return (
#             -g[Δ_itr, α, α, α, α]
#             +g[s_itr, α, α, μ, μ]
#             -g[t_itr, α, α, μ, μ]
#             +g[Δ_itr, α, α, μ, μ]
#             +g[Δ_itr, χ, χ, α, α]
#             -g[s_itr, χ, χ, μ, μ]
#             +g[t_itr, χ, χ, μ, μ]
#             -g[Δ_itr, χ, χ, μ, μ]
#             -conj(g[s_itr, α, α, χ, χ])
#             +conj(g[t_itr, α, α, χ, χ])
#             +conj(g[s_itr, χ, χ, χ, χ])
#             -conj(g[t_itr, χ, χ, χ, χ])
#         )
#     end

#     @inline function coef_RL(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         left_bracket = (
#             g′[Δ_itr, α, α, α, μ]
#             +g′[s_itr, α, μ, μ, μ]
#             -g′[Δ_itr, χ, χ, α, μ]
#             -conj(g′[s_itr, μ, α, χ, χ])
#         )

#         right_bracket = (
#             g′[t_itr, χ, β, α, α]
#             -g′[Δ_itr, χ, β, α, α]
#             -g′[t_itr, χ, β, μ, μ]
#             +g′[Δ_itr, χ, β, μ, μ]
#         )

#         return hbar2 * g″[Δ_itr, χ, β, α, μ] + left_bracket * right_bracket
#     end

#     @inline function kernel_RL(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         return (
#             σ_mem(σ_t, s_itr, t_itr, μ, χ)
#             * phase_exp(α, χ, Δ_itr)
#             * exp(exponent_RL(s_itr, Δ_itr, t_itr, α, β, χ, μ) / hbar2)
#             * coef_RL(s_itr, Δ_itr, t_itr, α, β, χ, μ) / hbar4
#         )
#     end

#     @inline function exponent_RR(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         # This is the g-part after distributing the leading minus sign in
#         # exp(-(i hbar Δε Δ + G_raw) / hbar^2).
#         return (
#             g[s_itr, α, α, α, α]
#             -g[t_itr, α, α, α, α]
#             -g[s_itr, χ, χ, α, α]
#             +g[t_itr, χ, χ, α, α]
#             +conj(g[Δ_itr, α, α, χ, χ])
#             -conj(g[s_itr, α, α, ν, ν])
#             +conj(g[t_itr, α, α, ν, ν])
#             -conj(g[Δ_itr, α, α, ν, ν])
#             -conj(g[Δ_itr, χ, χ, χ, χ])
#             +conj(g[s_itr, χ, χ, ν, ν])
#             -conj(g[t_itr, χ, χ, ν, ν])
#             +conj(g[Δ_itr, χ, χ, ν, ν])
#         )
#     end

#     @inline function coef_RR(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         right_bracket = (
#             g′[s_itr, ν, χ, α, α]
#             +conj(g′[Δ_itr, α, α, χ, ν])
#             -conj(g′[Δ_itr, χ, χ, χ, ν])
#             -conj(g′[s_itr, χ, ν, ν, ν])
#         )

#         left_bracket = (
#             conj(g′[t_itr, β, χ, χ, χ])
#             -conj(g′[Δ_itr, β, χ, χ, χ])
#             -conj(g′[t_itr, β, χ, ν, ν])
#             +conj(g′[Δ_itr, β, χ, ν, ν])
#         )

#         return hbar2 * conj(g″[Δ_itr, β, χ, χ, ν]) - right_bracket * left_bracket
#     end

#     @inline function kernel_RR(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         return -(
#             σ_mem(σ_t, s_itr, t_itr, α, ν)
#             * phase_exp(α, χ, Δ_itr)
#             * exp(exponent_RR(s_itr, Δ_itr, t_itr, α, β, χ, ν) / hbar2)
#             * coef_RR(s_itr, Δ_itr, t_itr, α, β, χ, ν) / hbar4
#         )
#     end


#     # ---------------------------------------------------------------------
#     # Explicit P L0 Q G0 Q L1 P, g-closed branch prefactors and coefficients.
#     # Branch L : input (μ, β) -> output (α, β)
#     # Branch R : input (α, ν) -> output (α, β)
#     # ---------------------------------------------------------------------

#     @inline function exponent_L0_L(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         μ::Int,
#     )
#         # exp[-(i hbar ω_{αβ} Δ + G_L) / hbar^2]
#         # = phase_exp(α, β, Δ) * exp[-G_L / hbar^2]
#         return (
#             -g[Δ_itr, α, α, α, α]
#             +g[s_itr, α, α, μ, μ]
#             -g[t_itr, α, α, μ, μ]
#             +g[Δ_itr, α, α, μ, μ]
#             +g[Δ_itr, β, β, α, α]
#             -g[s_itr, β, β, μ, μ]
#             +g[t_itr, β, β, μ, μ]
#             -g[Δ_itr, β, β, μ, μ]
#             -conj(g[s_itr, α, α, β, β])
#             +conj(g[t_itr, α, α, β, β])
#             +conj(g[s_itr, β, β, β, β])
#             -conj(g[t_itr, β, β, β, β])
#         )
#     end

#     @inline function coef_L0_L(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         μ::Int,
#     )
#         left_bracket = (
#             g′[Δ_itr, α, α, α, μ]
#             +g′[s_itr, α, μ, μ, μ]
#             -g′[Δ_itr, β, β, α, μ]
#             -conj(g′[s_itr, μ, α, β, β])
#         )

#         right_bracket = (
#             g′[t_itr, α, α, α, α]
#             -g′[Δ_itr, α, α, α, α]
#             -g′[t_itr, α, α, μ, μ]
#             +g′[Δ_itr, α, α, μ, μ]
#             -g′[t_itr, β, β, α, α]
#             +g′[Δ_itr, β, β, α, α]
#             +g′[t_itr, β, β, μ, μ]
#             -g′[Δ_itr, β, β, μ, μ]
#         )

#         return (
#             hbar2 * g″[Δ_itr, α, α, α, μ]
#             -hbar2 * g″[Δ_itr, β, β, α, μ]
#             +left_bracket * right_bracket
#         )
#     end

#     @inline function kernel_L0_L(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         μ::Int,
#     )
#         return -(
#             σ_mem(σ_t, s_itr, t_itr, μ, β)
#             * phase_exp(α, β, Δ_itr)
#             * exp(exponent_L0_L(s_itr, Δ_itr, t_itr, α, β, μ) / hbar2)
#             * coef_L0_L(s_itr, Δ_itr, t_itr, α, β, μ) / hbar4
#         )
#     end

#     @inline function exponent_L0_R(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         ν::Int,
#     )
#         # exp[-(i hbar ω_{αβ} Δ + G_R) / hbar^2]
#         # = phase_exp(α, β, Δ) * exp[-G_R / hbar^2]
#         return (
#             +g[s_itr, α, α, α, α]
#             -g[t_itr, α, α, α, α]
#             -g[s_itr, β, β, α, α]
#             +g[t_itr, β, β, α, α]
#             +conj(g[Δ_itr, α, α, β, β])
#             -conj(g[s_itr, α, α, ν, ν])
#             +conj(g[t_itr, α, α, ν, ν])
#             -conj(g[Δ_itr, α, α, ν, ν])
#             -conj(g[Δ_itr, β, β, β, β])
#             +conj(g[s_itr, β, β, ν, ν])
#             -conj(g[t_itr, β, β, ν, ν])
#             +conj(g[Δ_itr, β, β, ν, ν])
#         )
#     end

#     @inline function coef_L0_R(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         ν::Int,
#     )
#         left_bracket = (
#             g′[s_itr, ν, β, α, α]
#             +conj(g′[Δ_itr, α, α, β, ν])
#             -conj(g′[Δ_itr, β, β, β, ν])
#             -conj(g′[s_itr, β, ν, ν, ν])
#         )

#         right_bracket = (
#             conj(g′[t_itr, α, α, β, β])
#             -conj(g′[Δ_itr, α, α, β, β])
#             -conj(g′[t_itr, α, α, ν, ν])
#             +conj(g′[Δ_itr, α, α, ν, ν])
#             -conj(g′[t_itr, β, β, β, β])
#             +conj(g′[Δ_itr, β, β, β, β])
#             +conj(g′[t_itr, β, β, ν, ν])
#             -conj(g′[Δ_itr, β, β, ν, ν])
#         )

#         return (
#             -hbar2 * conj(g″[Δ_itr, α, α, β, ν])
#             +hbar2 * conj(g″[Δ_itr, β, β, β, ν])
#             +left_bracket * right_bracket
#         )
#     end

#     @inline function kernel_L0_R(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         ν::Int,
#     )
#         return -(
#             σ_mem(σ_t, s_itr, t_itr, α, ν)
#             * phase_exp(α, β, Δ_itr)
#             * exp(exponent_L0_R(s_itr, Δ_itr, t_itr, α, β, ν) / hbar2)
#             * coef_L0_R(s_itr, Δ_itr, t_itr, α, β, ν) / hbar4
#         )
#     end

#     @inline function calc__rhs_element!(
#         rhs_mat::AbstractMatrix,
#         curr_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#     )
#         rhs = include_PL0P_local ? D_L0P(curr_itr, α, β) * σ_t[α, β] : (0.0 + 0.0im)

#         # -------------------------------------------------------------
#         # P L1 P local left/right mixing.
#         # Left branch:  input (μ, β) -> output (α, β).
#         # Right branch: input (α, ν) -> output (α, β).
#         # These are time-local, so σ_t is used.
#         # -------------------------------------------------------------
#         if include_PL1P_local
#             for μ in 1:n_sys
#                 μ == α && continue
#                 if is_secular_pair(α, β, μ, β)
#                     rhs += -(
#                         σ_t[μ, β]
#                         * (
#                             g′[curr_itr, α, μ, μ, μ]
#                             -conj(g′[curr_itr, μ, α, β, β])
#                         ) / hbar2
#                     )
#                 end
#             end

#             for ν in 1:n_sys
#                 ν == β && continue
#                 if is_secular_pair(α, β, α, ν)
#                     rhs += (
#                         σ_t[α, ν]
#                         * (
#                             g′[curr_itr, ν, β, α, α]
#                             -conj(g′[curr_itr, β, ν, ν, ν])
#                         ) / hbar2
#                     )
#                 end
#             end
#         end

#         if curr_itr > 1
#             integral = 0.0 + 0.0im

#             for s_itr in 1:curr_itr
#                 Δ_itr = curr_itr - s_itr + 1
#                 w_int = ∫weight(s_itr, curr_itr)
#                 kernel = 0.0 + 0.0im

#                 if include_PL0QG0QL1P
#                     # P L0 Q G0 Q L1 P, L branch: input (μ, β).
#                     for μ in 1:n_sys
#                         μ == α && continue
#                         if is_secular_pair(α, β, μ, β)
#                             kernel += kernel_L0_L(s_itr, Δ_itr, curr_itr, σ_t, α, β, μ)
#                         end
#                     end

#                     # P L0 Q G0 Q L1 P, R branch: input (α, ν).
#                     for ν in 1:n_sys
#                         ν == β && continue
#                         if is_secular_pair(α, β, α, ν)
#                             kernel += kernel_L0_R(s_itr, Δ_itr, curr_itr, σ_t, α, β, ν)
#                         end
#                     end
#                 end

#                 if include_PL1QG0QL1P
#                     # P L1 Q G0 Q L1 P, LL: input (μ, β), intermediate χ.
#                     for χ in 1:n_sys
#                         χ == α && continue
#                         for μ in 1:n_sys
#                             μ == χ && continue
#                             if is_secular_pair(α, β, μ, β)
#                                 kernel += kernel_LL(s_itr, Δ_itr, curr_itr, σ_t, α, β, χ, μ)
#                             end
#                         end
#                     end

#                     # P L1 Q G0 Q L1 P, RR: input (α, ν), intermediate χ.
#                     for χ in 1:n_sys
#                         χ == β && continue
#                         for ν in 1:n_sys
#                             ν == χ && continue
#                             if is_secular_pair(α, β, α, ν)
#                                 kernel += kernel_RR(s_itr, Δ_itr, curr_itr, σ_t, α, β, χ, ν)
#                             end
#                         end
#                     end

#                     # P L1 Q G0 Q L1 P, RL: input (μ, χ), left input μ, right intermediate χ.
#                     for μ in 1:n_sys
#                         μ == α && continue
#                         for χ in 1:n_sys
#                             χ == β && continue
#                             if is_secular_pair(α, β, μ, χ)
#                                 kernel += kernel_RL(s_itr, Δ_itr, curr_itr, σ_t, α, β, χ, μ)
#                             end
#                         end
#                     end

#                     # P L1 Q G0 Q L1 P, LR: input (χ, ν), left intermediate χ, right input ν.
#                     for χ in 1:n_sys
#                         χ == α && continue
#                         for ν in 1:n_sys
#                             ν == β && continue
#                             if is_secular_pair(α, β, χ, ν)
#                                 kernel += kernel_LR(s_itr, Δ_itr, curr_itr, σ_t, α, β, χ, ν)
#                             end
#                         end
#                     end
#                 end

#                 integral += w_int * kernel
#             end

#             rhs += integral
#         end

#         rhs_mat[α, β] = rhs
#         return nothing
#     end

#     @inline function calc__rhs!(
#         rhs_mat::AbstractMatrix,
#         curr_itr::Int,
#         σ_t::AbstractMatrix,
#     )
#         fill!(rhs_mat, 0.0 + 0.0im)

#         n_components = n_sys * n_sys
#         if use_threads && Threads.nthreads() > 1 && n_components > 1
#             Threads.@threads for linear_idx in 1:n_components
#                 @inbounds begin
#                     α = ((linear_idx - 1) % n_sys) + 1
#                     β = ((linear_idx - 1) ÷ n_sys) + 1
#                     calc__rhs_element!(rhs_mat, curr_itr, σ_t, α, β)
#                 end
#             end
#         else
#             @inbounds for β in 1:n_sys, α in 1:n_sys
#                 calc__rhs_element!(rhs_mat, curr_itr, σ_t, α, β)
#             end
#         end

#         return rhs_mat
#     end

#     @inline function enforce_hermiticity!(σ_next::AbstractMatrix)
#         for i in 1:n_sys
#             σ_next[i, i] = real(σ_next[i, i]) + 0.0im
#         end

#         for i in 1:(n_sys - 1)
#             for j in (i + 1):n_sys
#                 c = 0.5 * (σ_next[i, j] + conj(σ_next[j, i]))
#                 σ_next[i, j] = c
#                 σ_next[j, i] = conj(c)
#             end
#         end

#         return σ_next
#     end

#     σ_stage = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k2      = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k3      = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k4      = Matrix{ComplexF64}(undef, n_sys, n_sys)

#     @inbounds for curr_itr in start_itr:(n_itr - 1)
#         if verbose
#             @printf(
#                 stderr,
#                 "Current iteration: %6d / %6d  method=%s  threads=%d  use_threads=%s  secular=%s\n",
#                 curr_itr,
#                 n_itr,
#                 String(method_sym),
#                 Threads.nthreads(),
#                 string(use_threads),
#                 string(use_secular),
#             )
#         end

#         σ_t    = @view σ[:, :, curr_itr]
#         σ_next = @view σ[:, :, curr_itr + 1]
#         k1     = @view σ′[:, :, curr_itr]

#         if method_sym == :euler
#             calc__rhs!(k1, curr_itr, σ_t)
#             @. σ_next = σ_t + Δt * k1

#         elseif method_sym == :rk2
#             calc__rhs!(k1, curr_itr, σ_t)
#             @. σ_stage = σ_t + 0.5 * Δt * k1
#             calc__rhs!(k2, curr_itr, σ_stage)
#             @. σ_next = σ_t + Δt * k2

#         elseif method_sym == :rk4
#             calc__rhs!(k1, curr_itr, σ_t)

#             @. σ_stage = σ_t + 0.5 * Δt * k1
#             calc__rhs!(k2, curr_itr, σ_stage)

#             @. σ_stage = σ_t + 0.5 * Δt * k2
#             calc__rhs!(k3, curr_itr, σ_stage)

#             @. σ_stage = σ_t + Δt * k3
#             calc__rhs!(k4, curr_itr, σ_stage)

#             @. σ_next = σ_t + (Δt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
#         end

#         if enforce_hermitian
#             enforce_hermiticity!(σ_next)
#         end

#         context.curr_itr = UInt64(curr_itr + 1)
#     end

#     return @view σ[:, :, Int(context.curr_itr)]
# end



# function calc__σ_σ′_secular_core!(
#     context::RmrtContext;
#     hbar::Real = 1.0,
#     use_population_closure::Bool = false,
    
#     # P L0 P local self block.
#     include_PL0P_local::Bool = true,

#     # P L1 P local left/right mixing block.
#     include_PL1P_local::Bool = true,

#     include_PL0QG0QL1P::Bool = true,
#     include_PL1QG0QL1P::Bool = true,

#     # P L0 Q G1 Q L1 P correction. This is a nested s-tau memory integral,
#     # so it is intentionally opt-in by default.
#     include_PL0QG1QL1P::Bool = true,

#     use_secular::Bool = false,
#     secular_tol::Float64 = 1.0e-10,
#     method::Union{Symbol,String} = :rk4,
#     use_threads::Bool = true,

#     # Keep the original half-shifted g-grid preparation hook.
#     # RK2/RK4 stage-time support depends on this cache in the wider code base,
#     # even if this explicit-kernel refactor currently evaluates g-series on the
#     # integer grid inside calc__rhs!.
#     auto_prepare_half_shifted_grid::Bool = true,
#     recompute_half_shifted_grid::Bool = false,

#     enforce_hermitian::Bool = true,
#     verbose::Bool = true,
# )
#     start_itr = Int(context.curr_itr)

#     n_sys = context.system.n_sys
#     n_itr = context.simulation_details.num_of_iteration
#     Δt    = context.simulation_details.Δt
#     ϵ     = context.ϵ_exci

#     σ  = context.σ
#     σ′ = context.σ′
#     g  = context.g
#     g′ = context.g′
#     g″ = context.g″

#     hbar_f = Float64(hbar)
#     hbar_f > 0.0 || error("hbar must be positive")
#     hbar2 = hbar_f^2
#     hbar4 = hbar_f^4
#     hbar6 = hbar_f^6

#     method_sym = Symbol(lowercase(String(method)))
#     method_sym in (:euler, :rk2, :rk4) || error(
#         "Unsupported integration method: $(method). Use :euler, :rk2, or :rk4."
#     )

#     needs_half_shifted_grid = method_sym in (:rk2, :rk4)

#     if needs_half_shifted_grid
#         if auto_prepare_half_shifted_grid
#             ensure__half_shifted_grid!(
#                 context;
#                 recompute = recompute_half_shifted_grid,
#                 use_threads = use_threads,
#                 verbose = verbose,
#             )
#         elseif !context.using_half_shifted_grid || !has__half_shifted_grid(context)
#             error(
#                 "method=$(method_sym) requires half-shifted g-series cache. " *
#                 "Call ensure__half_shifted_grid!(context) or " *
#                 "calc__g_g′_g″_half_shifted!(context) before propagation, " *
#                 "or set auto_prepare_half_shifted_grid=true."
#             )
#         end
#     end

#     start_itr < n_itr || return @view σ[:, :, n_itr]

#     @inline ω(a::Int, b::Int) = ϵ[a] - ϵ[b]

#     @inline function is_secular_pair(
#         out_a::Int,
#         out_b::Int,
#         in_a::Int,
#         in_b::Int,
#     )
#         if !use_secular
#             return true
#         end
#         return abs(ω(out_a, out_b) - ω(in_a, in_b)) <= secular_tol
#     end

#     @inline function ∫weight(s_itr::Int, curr_itr::Int)
#         return (s_itr == 1 || s_itr == curr_itr) ? 0.5 * Δt : Δt
#     end

#     @inline function ∫weight_between(τ_itr::Int, s_itr::Int, curr_itr::Int)
#         # Trapezoid rule for the inner integral ∫_s^t dτ.
#         # If s == t, the integration interval has zero length.
#         if s_itr == curr_itr
#             return 0.0
#         end
#         return (τ_itr == s_itr || τ_itr == curr_itr) ? 0.5 * Δt : Δt
#     end

#     @inline function Δtime(Δ_itr::Int)
#         return (Δ_itr - 1) * Δt
#     end

#     @inline function σ_mem(
#         σ_t::AbstractMatrix,
#         s_itr::Int,
#         curr_itr::Int,
#         a::Int,
#         b::Int,
#     )
#         return s_itr == curr_itr ? σ_t[a, b] : σ[a, b, s_itr]
#     end

#     @inline function phase_exp(from_a::Int, from_b::Int, Δ_itr::Int)
#         # exp[-i (epsilon_a - epsilon_b) Δ / hbar]
#         return exp(-1.0im * ω(from_a, from_b) * Δtime(Δ_itr) / hbar_f)
#     end

#     @inline function D_L0P(
#         itr::Int,
#         α::Int,
#         β::Int,
#     )
#         return (
#             -1.0im * ω(α, β) / hbar_f
#             +(
#                 -g′[itr, α, α, α, α]
#                 +conj(g′[itr, α, α, β, β])
#                 +g′[itr, β, β, α, α]
#                 -conj(g′[itr, β, β, β, β])
#             ) / hbar2
#         )
#     end

#     # ---------------------------------------------------------------------
#     # Explicit PL1 Q G0 Q L1 P, g-closed branch prefactors and coefficients.
#     # Branch naming follows the trace side of the two L1 insertions.
#     # All formulas are written with explicit hbar factors.  With hbar=1 this
#     # reduces to the previous internal convention.
#     # ---------------------------------------------------------------------

#     @inline function exponent_LL(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         return (
#             g[Δ_itr, β, β, χ, χ]
#             -g[s_itr, β, β, μ, μ]
#             +g[t_itr, β, β, μ, μ]
#             -g[Δ_itr, β, β, μ, μ]
#             -g[Δ_itr, χ, χ, χ, χ]
#             +g[s_itr, χ, χ, μ, μ]
#             -g[t_itr, χ, χ, μ, μ]
#             +g[Δ_itr, χ, χ, μ, μ]
#             +conj(g[s_itr, β, β, β, β])
#             -conj(g[t_itr, β, β, β, β])
#             -conj(g[s_itr, χ, χ, β, β])
#             +conj(g[t_itr, χ, χ, β, β])
#         )
#     end

#     @inline function coef_LL(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         left_bracket = (
#             g′[t_itr, α, χ, χ, χ]
#             -g′[Δ_itr, α, χ, χ, χ]
#             -g′[t_itr, α, χ, μ, μ]
#             +g′[Δ_itr, α, χ, μ, μ]
#         )

#         right_bracket = (
#             g′[Δ_itr, β, β, χ, μ]
#             -g′[Δ_itr, χ, χ, χ, μ]
#             -g′[s_itr, χ, μ, μ, μ]
#             +conj(g′[s_itr, μ, χ, β, β])
#         )

#         return hbar2 * g″[Δ_itr, α, χ, χ, μ] - left_bracket * right_bracket
#     end

#     @inline function kernel_LL(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         return -(
#             σ_mem(σ_t, s_itr, t_itr, μ, β)
#             * phase_exp(χ, β, Δ_itr)
#             * exp(exponent_LL(s_itr, Δ_itr, t_itr, α, β, χ, μ) / hbar2)
#             * coef_LL(s_itr, Δ_itr, t_itr, α, β, χ, μ) / hbar4
#         )
#     end

#     @inline function exponent_LR(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         return (
#             -g[s_itr, β, β, χ, χ]
#             +g[t_itr, β, β, χ, χ]
#             +g[s_itr, χ, χ, χ, χ]
#             -g[t_itr, χ, χ, χ, χ]
#             -conj(g[Δ_itr, β, β, β, β])
#             +conj(g[s_itr, β, β, ν, ν])
#             -conj(g[t_itr, β, β, ν, ν])
#             +conj(g[Δ_itr, β, β, ν, ν])
#             +conj(g[Δ_itr, χ, χ, β, β])
#             -conj(g[s_itr, χ, χ, ν, ν])
#             +conj(g[t_itr, χ, χ, ν, ν])
#             -conj(g[Δ_itr, χ, χ, ν, ν])
#         )
#     end

#     @inline function coef_LR(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         right_bracket = (
#             g′[s_itr, ν, β, χ, χ]
#             -conj(g′[Δ_itr, β, β, β, ν])
#             -conj(g′[s_itr, β, ν, ν, ν])
#             +conj(g′[Δ_itr, χ, χ, β, ν])
#         )

#         left_bracket = (
#             conj(g′[t_itr, χ, α, β, β])
#             -conj(g′[Δ_itr, χ, α, β, β])
#             -conj(g′[t_itr, χ, α, ν, ν])
#             +conj(g′[Δ_itr, χ, α, ν, ν])
#         )

#         return hbar2 * conj(g″[Δ_itr, χ, α, β, ν]) - right_bracket * left_bracket
#     end

#     @inline function kernel_LR(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         return (
#             σ_mem(σ_t, s_itr, t_itr, χ, ν)
#             * phase_exp(χ, β, Δ_itr)
#             * exp(exponent_LR(s_itr, Δ_itr, t_itr, α, β, χ, ν) / hbar2)
#             * coef_LR(s_itr, Δ_itr, t_itr, α, β, χ, ν) / hbar4
#         )
#     end

#     @inline function exponent_RL(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         return (
#             -g[Δ_itr, α, α, α, α]
#             +g[s_itr, α, α, μ, μ]
#             -g[t_itr, α, α, μ, μ]
#             +g[Δ_itr, α, α, μ, μ]
#             +g[Δ_itr, χ, χ, α, α]
#             -g[s_itr, χ, χ, μ, μ]
#             +g[t_itr, χ, χ, μ, μ]
#             -g[Δ_itr, χ, χ, μ, μ]
#             -conj(g[s_itr, α, α, χ, χ])
#             +conj(g[t_itr, α, α, χ, χ])
#             +conj(g[s_itr, χ, χ, χ, χ])
#             -conj(g[t_itr, χ, χ, χ, χ])
#         )
#     end

#     @inline function coef_RL(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         left_bracket = (
#             g′[Δ_itr, α, α, α, μ]
#             +g′[s_itr, α, μ, μ, μ]
#             -g′[Δ_itr, χ, χ, α, μ]
#             -conj(g′[s_itr, μ, α, χ, χ])
#         )

#         right_bracket = (
#             g′[t_itr, χ, β, α, α]
#             -g′[Δ_itr, χ, β, α, α]
#             -g′[t_itr, χ, β, μ, μ]
#             +g′[Δ_itr, χ, β, μ, μ]
#         )

#         return hbar2 * g″[Δ_itr, χ, β, α, μ] + left_bracket * right_bracket
#     end

#     @inline function kernel_RL(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         return (
#             σ_mem(σ_t, s_itr, t_itr, μ, χ)
#             * phase_exp(α, χ, Δ_itr)
#             * exp(exponent_RL(s_itr, Δ_itr, t_itr, α, β, χ, μ) / hbar2)
#             * coef_RL(s_itr, Δ_itr, t_itr, α, β, χ, μ) / hbar4
#         )
#     end

#     @inline function exponent_RR(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         # This is the g-part after distributing the leading minus sign in
#         # exp(-(i hbar Δε Δ + G_raw) / hbar^2).
#         return (
#             g[s_itr, α, α, α, α]
#             -g[t_itr, α, α, α, α]
#             -g[s_itr, χ, χ, α, α]
#             +g[t_itr, χ, χ, α, α]
#             +conj(g[Δ_itr, α, α, χ, χ])
#             -conj(g[s_itr, α, α, ν, ν])
#             +conj(g[t_itr, α, α, ν, ν])
#             -conj(g[Δ_itr, α, α, ν, ν])
#             -conj(g[Δ_itr, χ, χ, χ, χ])
#             +conj(g[s_itr, χ, χ, ν, ν])
#             -conj(g[t_itr, χ, χ, ν, ν])
#             +conj(g[Δ_itr, χ, χ, ν, ν])
#         )
#     end

#     @inline function coef_RR(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         right_bracket = (
#             g′[s_itr, ν, χ, α, α]
#             +conj(g′[Δ_itr, α, α, χ, ν])
#             -conj(g′[Δ_itr, χ, χ, χ, ν])
#             -conj(g′[s_itr, χ, ν, ν, ν])
#         )

#         left_bracket = (
#             conj(g′[t_itr, β, χ, χ, χ])
#             -conj(g′[Δ_itr, β, χ, χ, χ])
#             -conj(g′[t_itr, β, χ, ν, ν])
#             +conj(g′[Δ_itr, β, χ, ν, ν])
#         )

#         return hbar2 * conj(g″[Δ_itr, β, χ, χ, ν]) - right_bracket * left_bracket
#     end

#     @inline function kernel_RR(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         return -(
#             σ_mem(σ_t, s_itr, t_itr, α, ν)
#             * phase_exp(α, χ, Δ_itr)
#             * exp(exponent_RR(s_itr, Δ_itr, t_itr, α, β, χ, ν) / hbar2)
#             * coef_RR(s_itr, Δ_itr, t_itr, α, β, χ, ν) / hbar4
#         )
#     end


#     # ---------------------------------------------------------------------
#     # Explicit P L0 Q G0 Q L1 P, g-closed branch prefactors and coefficients.
#     # Branch L : input (μ, β) -> output (α, β)
#     # Branch R : input (α, ν) -> output (α, β)
#     # ---------------------------------------------------------------------

#     @inline function exponent_L0_L(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         μ::Int,
#     )
#         # exp[-(i hbar ω_{αβ} Δ + G_L) / hbar^2]
#         # = phase_exp(α, β, Δ) * exp[-G_L / hbar^2]
#         return (
#             -g[Δ_itr, α, α, α, α]
#             +g[s_itr, α, α, μ, μ]
#             -g[t_itr, α, α, μ, μ]
#             +g[Δ_itr, α, α, μ, μ]
#             +g[Δ_itr, β, β, α, α]
#             -g[s_itr, β, β, μ, μ]
#             +g[t_itr, β, β, μ, μ]
#             -g[Δ_itr, β, β, μ, μ]
#             -conj(g[s_itr, α, α, β, β])
#             +conj(g[t_itr, α, α, β, β])
#             +conj(g[s_itr, β, β, β, β])
#             -conj(g[t_itr, β, β, β, β])
#         )
#     end

#     @inline function coef_L0_L(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         μ::Int,
#     )
#         left_bracket = (
#             g′[Δ_itr, α, α, α, μ]
#             +g′[s_itr, α, μ, μ, μ]
#             -g′[Δ_itr, β, β, α, μ]
#             -conj(g′[s_itr, μ, α, β, β])
#         )

#         right_bracket = (
#             g′[t_itr, α, α, α, α]
#             -g′[Δ_itr, α, α, α, α]
#             -g′[t_itr, α, α, μ, μ]
#             +g′[Δ_itr, α, α, μ, μ]
#             -g′[t_itr, β, β, α, α]
#             +g′[Δ_itr, β, β, α, α]
#             +g′[t_itr, β, β, μ, μ]
#             -g′[Δ_itr, β, β, μ, μ]
#         )

#         return (
#             hbar2 * g″[Δ_itr, α, α, α, μ]
#             -hbar2 * g″[Δ_itr, β, β, α, μ]
#             +left_bracket * right_bracket
#         )
#     end

#     @inline function kernel_L0_L(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         μ::Int,
#     )
#         return -(
#             σ_mem(σ_t, s_itr, t_itr, μ, β)
#             * phase_exp(α, β, Δ_itr)
#             * exp(exponent_L0_L(s_itr, Δ_itr, t_itr, α, β, μ) / hbar2)
#             * coef_L0_L(s_itr, Δ_itr, t_itr, α, β, μ) / hbar4
#         )
#     end

#     @inline function exponent_L0_R(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         ν::Int,
#     )
#         # exp[-(i hbar ω_{αβ} Δ + G_R) / hbar^2]
#         # = phase_exp(α, β, Δ) * exp[-G_R / hbar^2]
#         return (
#             +g[s_itr, α, α, α, α]
#             -g[t_itr, α, α, α, α]
#             -g[s_itr, β, β, α, α]
#             +g[t_itr, β, β, α, α]
#             +conj(g[Δ_itr, α, α, β, β])
#             -conj(g[s_itr, α, α, ν, ν])
#             +conj(g[t_itr, α, α, ν, ν])
#             -conj(g[Δ_itr, α, α, ν, ν])
#             -conj(g[Δ_itr, β, β, β, β])
#             +conj(g[s_itr, β, β, ν, ν])
#             -conj(g[t_itr, β, β, ν, ν])
#             +conj(g[Δ_itr, β, β, ν, ν])
#         )
#     end

#     @inline function coef_L0_R(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         ν::Int,
#     )
#         left_bracket = (
#             g′[s_itr, ν, β, α, α]
#             +conj(g′[Δ_itr, α, α, β, ν])
#             -conj(g′[Δ_itr, β, β, β, ν])
#             -conj(g′[s_itr, β, ν, ν, ν])
#         )

#         right_bracket = (
#             conj(g′[t_itr, α, α, β, β])
#             -conj(g′[Δ_itr, α, α, β, β])
#             -conj(g′[t_itr, α, α, ν, ν])
#             +conj(g′[Δ_itr, α, α, ν, ν])
#             -conj(g′[t_itr, β, β, β, β])
#             +conj(g′[Δ_itr, β, β, β, β])
#             +conj(g′[t_itr, β, β, ν, ν])
#             -conj(g′[Δ_itr, β, β, ν, ν])
#         )

#         return (
#             -hbar2 * conj(g″[Δ_itr, α, α, β, ν])
#             +hbar2 * conj(g″[Δ_itr, β, β, β, ν])
#             +left_bracket * right_bracket
#         )
#     end

#     @inline function kernel_L0_R(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         ν::Int,
#     )
#         return -(
#             σ_mem(σ_t, s_itr, t_itr, α, ν)
#             * phase_exp(α, β, Δ_itr)
#             * exp(exponent_L0_R(s_itr, Δ_itr, t_itr, α, β, ν) / hbar2)
#             * coef_L0_R(s_itr, Δ_itr, t_itr, α, β, ν) / hbar4
#         )
#     end



#     # ---------------------------------------------------------------------
#     # Explicit P L0 Q G1 Q L1 P, g-closed 2-time branch kernels.
#     # The extra variable τ satisfies s <= τ <= t.  Each branch is the
#     # direct g-closed expression from I^{*,L0G1L1,g}_{αβ}(t,τ,s).
#     # ---------------------------------------------------------------------

#     @inline function kernel_L0G1_LL(
#         s_itr::Int,
#         τ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((hbar2 * g″[Δtτ_itr, α, α, α, χ] - hbar2 * g″[Δtτ_itr, β, β, α, χ] + (- g′[Δtτ_itr, α, α, α, χ] - g′[τ_itr, α, χ, χ, χ] + g′[Δtτ_itr, β, β, α, χ] + conj(g′[τ_itr, χ, α, β, β])) * (- g′[t_itr, α, α, α, α] + g′[Δtτ_itr, α, α, α, α] + g′[t_itr, α, α, χ, χ] - g′[Δtτ_itr, α, α, χ, χ] + g′[t_itr, β, β, α, α] - g′[Δtτ_itr, β, β, α, α] - g′[t_itr, β, β, χ, χ] + g′[Δtτ_itr, β, β, χ, χ])) * (g′[Δτs_itr, β, β, χ, μ] - g′[Δτs_itr, χ, χ, χ, μ] - g′[s_itr, χ, μ, μ, μ] + conj(g′[s_itr, μ, χ, β, β])) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - g[Δtτ_itr, α, α, α, α] - g[t_itr, α, α, χ, χ] + g[τ_itr, α, α, χ, χ] + g[Δtτ_itr, α, α, χ, χ] + g[Δtτ_itr, β, β, α, α] + g[t_itr, β, β, χ, χ] - g[τ_itr, β, β, χ, χ] - g[Δtτ_itr, β, β, χ, χ] + g[Δτs_itr, β, β, χ, χ] - g[t_itr, β, β, μ, μ] + g[τ_itr, β, β, μ, μ] - g[Δτs_itr, β, β, μ, μ] - g[Δτs_itr, χ, χ, χ, χ] + g[s_itr, χ, χ, μ, μ] - g[τ_itr, χ, χ, μ, μ] + g[Δτs_itr, χ, χ, μ, μ] - g[s_itr, μ, μ, μ, μ] + g[t_itr, μ, μ, μ, μ] + conj(g[t_itr, α, α, β, β]) - conj(g[τ_itr, α, α, β, β]) - conj(g[s_itr, χ, χ, β, β]) + conj(g[τ_itr, χ, χ, β, β]) + conj(g[s_itr, μ, μ, β, β]) - conj(g[t_itr, μ, μ, β, β])) / (hbar2))) + (hbar2 * (- g″[Δtτ_itr, α, α, α, χ] + g″[Δtτ_itr, β, β, α, χ]) * (- g′[Δts_itr, α, α, χ, μ] + g′[Δτs_itr, α, α, χ, μ] + g′[Δts_itr, β, β, χ, μ] - g′[Δτs_itr, χ, χ, χ, μ] - g′[s_itr, χ, μ, μ, μ] + conj(g′[s_itr, μ, χ, β, β])) + hbar2 * (- g′[Δtτ_itr, β, β, α, α] - g′[Δts_itr, β, β, χ, χ] + g′[Δtτ_itr, β, β, χ, χ] - g′[t_itr, β, β, μ, μ] + g′[Δts_itr, β, β, μ, μ] + conj(g′[t_itr, β, β, β, β])) * g″[Δτs_itr, α, χ, χ, μ] + (hbar2 * g″[Δτs_itr, α, χ, χ, μ] + (g′[Δtτ_itr, α, α, α, χ] + g′[Δτs_itr, α, χ, χ, χ] + g′[τ_itr, α, χ, μ, μ] - g′[Δτs_itr, α, χ, μ, μ] - g′[Δtτ_itr, β, β, α, χ] - conj(g′[τ_itr, χ, α, β, β])) * (- g′[Δts_itr, α, α, χ, μ] + g′[Δτs_itr, α, α, χ, μ] + g′[Δts_itr, β, β, χ, μ] - g′[Δτs_itr, χ, χ, χ, μ] - g′[s_itr, χ, μ, μ, μ] + conj(g′[s_itr, μ, χ, β, β]))) * (- g′[t_itr, α, α, α, α] + g′[Δtτ_itr, α, α, α, α] + g′[Δts_itr, α, α, χ, χ] - g′[Δtτ_itr, α, α, χ, χ] + g′[t_itr, α, α, μ, μ] - g′[Δts_itr, α, α, μ, μ] + g′[t_itr, β, β, α, α] - conj(g′[t_itr, β, β, β, β])) + (- hbar2 * g″[Δts_itr, α, α, χ, μ] + hbar2 * g″[Δts_itr, β, β, χ, μ] + (g′[Δts_itr, α, α, χ, μ] - g′[Δτs_itr, α, α, χ, μ] - g′[Δts_itr, β, β, χ, μ] + g′[Δτs_itr, χ, χ, χ, μ] + g′[s_itr, χ, μ, μ, μ] - conj(g′[s_itr, μ, χ, β, β])) * (- g′[Δtτ_itr, β, β, α, α] - g′[Δts_itr, β, β, χ, χ] + g′[Δtτ_itr, β, β, χ, χ] - g′[t_itr, β, β, μ, μ] + g′[Δts_itr, β, β, μ, μ] + conj(g′[t_itr, β, β, β, β]))) * (- g′[Δtτ_itr, α, α, α, χ] - g′[Δτs_itr, α, χ, χ, χ] - g′[τ_itr, α, χ, μ, μ] + g′[Δτs_itr, α, χ, μ, μ] + g′[Δtτ_itr, β, β, α, χ] + conj(g′[τ_itr, χ, α, β, β]))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - g[Δtτ_itr, α, α, α, α] - g[Δts_itr, α, α, χ, χ] + g[Δtτ_itr, α, α, χ, χ] + g[Δτs_itr, α, α, χ, χ] - g[t_itr, α, α, μ, μ] + g[τ_itr, α, α, μ, μ] + g[Δts_itr, α, α, μ, μ] - g[Δτs_itr, α, α, μ, μ] + g[Δtτ_itr, β, β, α, α] + g[Δts_itr, β, β, χ, χ] - g[Δtτ_itr, β, β, χ, χ] - g[Δts_itr, β, β, μ, μ] - g[Δτs_itr, χ, χ, χ, χ] + g[s_itr, χ, χ, μ, μ] - g[τ_itr, χ, χ, μ, μ] + g[Δτs_itr, χ, χ, μ, μ] - g[s_itr, μ, μ, μ, μ] + g[t_itr, μ, μ, μ, μ] + conj(g[t_itr, α, α, β, β]) - conj(g[τ_itr, α, α, β, β]) - conj(g[s_itr, χ, χ, β, β]) + conj(g[τ_itr, χ, χ, β, β]) + conj(g[s_itr, μ, μ, β, β]) - conj(g[t_itr, μ, μ, β, β])) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, μ, β) * exp(((- g[s_itr, β, β, μ, μ] + g[t_itr, β, β, μ, μ] + g[s_itr, μ, μ, μ, μ] - g[t_itr, μ, μ, μ, μ] + conj(g[s_itr, β, β, β, β]) - conj(g[t_itr, β, β, β, β]) - conj(g[s_itr, μ, μ, β, β]) + conj(g[t_itr, μ, μ, β, β])) / (hbar2)))
#         )
#     end

#     @inline function kernel_L0G1_LR(
#         s_itr::Int,
#         τ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((- hbar2 * g″[Δtτ_itr, α, α, α, χ] + hbar2 * g″[Δtτ_itr, β, β, α, χ] + (- g′[Δtτ_itr, α, α, α, χ] - g′[τ_itr, α, χ, χ, χ] + g′[Δtτ_itr, β, β, α, χ] + conj(g′[τ_itr, χ, α, β, β])) * (g′[t_itr, α, α, α, α] - g′[Δtτ_itr, α, α, α, α] - g′[t_itr, α, α, χ, χ] + g′[Δtτ_itr, α, α, χ, χ] - g′[t_itr, β, β, α, α] + g′[Δtτ_itr, β, β, α, α] + g′[t_itr, β, β, χ, χ] - g′[Δtτ_itr, β, β, χ, χ])) * (- g′[s_itr, ν, β, χ, χ] + conj(g′[Δτs_itr, β, β, β, ν]) + conj(g′[s_itr, β, ν, ν, ν]) - conj(g′[Δτs_itr, χ, χ, β, ν])) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - g[Δtτ_itr, α, α, α, α] - g[t_itr, α, α, χ, χ] + g[τ_itr, α, α, χ, χ] + g[Δtτ_itr, α, α, χ, χ] + g[Δtτ_itr, β, β, α, α] - g[s_itr, β, β, χ, χ] + g[t_itr, β, β, χ, χ] - g[Δtτ_itr, β, β, χ, χ] + g[t_itr, χ, χ, χ, χ] - g[τ_itr, χ, χ, χ, χ] + g[s_itr, ν, ν, χ, χ] - g[t_itr, ν, ν, χ, χ] + conj(g[t_itr, α, α, β, β]) - conj(g[τ_itr, α, α, β, β]) - conj(g[t_itr, β, β, β, β]) + conj(g[τ_itr, β, β, β, β]) - conj(g[Δτs_itr, β, β, β, β]) + conj(g[s_itr, β, β, ν, ν]) - conj(g[τ_itr, β, β, ν, ν]) + conj(g[Δτs_itr, β, β, ν, ν]) + conj(g[Δτs_itr, χ, χ, β, β]) - conj(g[t_itr, χ, χ, ν, ν]) + conj(g[τ_itr, χ, χ, ν, ν]) - conj(g[Δτs_itr, χ, χ, ν, ν]) - conj(g[s_itr, ν, ν, ν, ν]) + conj(g[t_itr, ν, ν, ν, ν])) / (hbar2))) + (hbar2 * (conj(g″[Δts_itr, α, α, β, ν]) - conj(g″[Δts_itr, β, β, β, ν])) * (- g′[Δtτ_itr, α, α, α, χ] - g′[τ_itr, α, χ, χ, χ] + g′[Δtτ_itr, β, β, α, χ] + conj(g′[Δτs_itr, χ, α, β, β]) + conj(g′[τ_itr, χ, α, ν, ν]) - conj(g′[Δτs_itr, χ, α, ν, ν])) + hbar2 * (- g′[Δtτ_itr, α, α, α, α] - g′[t_itr, α, α, χ, χ] + g′[Δtτ_itr, α, α, χ, χ] + conj(g′[Δts_itr, α, α, β, β]) + conj(g′[t_itr, α, α, ν, ν]) - conj(g′[Δts_itr, α, α, ν, ν])) * conj(g″[Δτs_itr, χ, α, β, ν]) - hbar2 * (- g′[Δtτ_itr, β, β, α, α] - g′[t_itr, β, β, χ, χ] + g′[Δtτ_itr, β, β, χ, χ] + conj(g′[Δts_itr, β, β, β, β]) + conj(g′[t_itr, β, β, ν, ν]) - conj(g′[Δts_itr, β, β, ν, ν])) * conj(g″[Δτs_itr, χ, α, β, ν]) + (hbar2 * conj(g″[Δτs_itr, χ, α, β, ν]) + (- g′[Δtτ_itr, α, α, α, χ] - g′[τ_itr, α, χ, χ, χ] + g′[Δtτ_itr, β, β, α, χ] + conj(g′[Δτs_itr, χ, α, β, β]) + conj(g′[τ_itr, χ, α, ν, ν]) - conj(g′[Δτs_itr, χ, α, ν, ν])) * (g′[s_itr, ν, β, χ, χ] + conj(g′[Δts_itr, α, α, β, ν]) - conj(g′[Δτs_itr, α, α, β, ν]) - conj(g′[Δts_itr, β, β, β, ν]) - conj(g′[s_itr, β, ν, ν, ν]) + conj(g′[Δτs_itr, χ, χ, β, ν]))) * (g′[t_itr, α, α, α, α] - g′[t_itr, β, β, α, α] - conj(g′[t_itr, α, α, β, β]) + conj(g′[t_itr, β, β, β, β])) + (hbar2 * g″[Δtτ_itr, α, α, α, χ] - hbar2 * g″[Δtτ_itr, β, β, α, χ] + (- g′[Δtτ_itr, α, α, α, χ] - g′[τ_itr, α, χ, χ, χ] + g′[Δtτ_itr, β, β, α, χ] + conj(g′[Δτs_itr, χ, α, β, β]) + conj(g′[τ_itr, χ, α, ν, ν]) - conj(g′[Δτs_itr, χ, α, ν, ν])) * (g′[Δtτ_itr, α, α, α, α] + g′[t_itr, α, α, χ, χ] - g′[Δtτ_itr, α, α, χ, χ] - g′[Δtτ_itr, β, β, α, α] - g′[t_itr, β, β, χ, χ] + g′[Δtτ_itr, β, β, χ, χ] - conj(g′[Δts_itr, α, α, β, β]) - conj(g′[t_itr, α, α, ν, ν]) + conj(g′[Δts_itr, α, α, ν, ν]) + conj(g′[Δts_itr, β, β, β, β]) + conj(g′[t_itr, β, β, ν, ν]) - conj(g′[Δts_itr, β, β, ν, ν]))) * (- g′[s_itr, ν, β, χ, χ] - conj(g′[Δts_itr, α, α, β, ν]) + conj(g′[Δτs_itr, α, α, β, ν]) + conj(g′[Δts_itr, β, β, β, ν]) + conj(g′[s_itr, β, ν, ν, ν]) - conj(g′[Δτs_itr, χ, χ, β, ν]))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - g[Δtτ_itr, α, α, α, α] - g[t_itr, α, α, χ, χ] + g[τ_itr, α, α, χ, χ] + g[Δtτ_itr, α, α, χ, χ] + g[Δtτ_itr, β, β, α, α] - g[s_itr, β, β, χ, χ] + g[t_itr, β, β, χ, χ] - g[Δtτ_itr, β, β, χ, χ] + g[t_itr, χ, χ, χ, χ] - g[τ_itr, χ, χ, χ, χ] + g[s_itr, ν, ν, χ, χ] - g[t_itr, ν, ν, χ, χ] + conj(g[Δts_itr, α, α, β, β]) - conj(g[Δτs_itr, α, α, β, β]) + conj(g[t_itr, α, α, ν, ν]) - conj(g[τ_itr, α, α, ν, ν]) - conj(g[Δts_itr, α, α, ν, ν]) + conj(g[Δτs_itr, α, α, ν, ν]) - conj(g[Δts_itr, β, β, β, β]) + conj(g[s_itr, β, β, ν, ν]) - conj(g[t_itr, β, β, ν, ν]) + conj(g[Δts_itr, β, β, ν, ν]) + conj(g[Δτs_itr, χ, χ, β, β]) - conj(g[t_itr, χ, χ, ν, ν]) + conj(g[τ_itr, χ, χ, ν, ν]) - conj(g[Δτs_itr, χ, χ, ν, ν]) - conj(g[s_itr, ν, ν, ν, ν]) + conj(g[t_itr, ν, ν, ν, ν])) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, χ, ν) * exp(((g[s_itr, χ, χ, χ, χ] - g[t_itr, χ, χ, χ, χ] - g[s_itr, ν, ν, χ, χ] + g[t_itr, ν, ν, χ, χ] - conj(g[s_itr, χ, χ, ν, ν]) + conj(g[t_itr, χ, χ, ν, ν]) + conj(g[s_itr, ν, ν, ν, ν]) - conj(g[t_itr, ν, ν, ν, ν])) / (hbar2)))
#         )
#     end

#     @inline function kernel_L0G1_RL(
#         s_itr::Int,
#         τ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((- hbar2 * conj(g″[Δtτ_itr, α, α, β, χ]) + hbar2 * conj(g″[Δtτ_itr, β, β, β, χ]) + (- g′[τ_itr, χ, β, α, α] - conj(g′[Δtτ_itr, α, α, β, χ]) + conj(g′[Δtτ_itr, β, β, β, χ]) + conj(g′[τ_itr, β, χ, χ, χ])) * (- conj(g′[t_itr, α, α, β, β]) + conj(g′[Δtτ_itr, α, α, β, β]) + conj(g′[t_itr, α, α, χ, χ]) - conj(g′[Δtτ_itr, α, α, χ, χ]) + conj(g′[t_itr, β, β, β, β]) - conj(g′[Δtτ_itr, β, β, β, β]) - conj(g′[t_itr, β, β, χ, χ]) + conj(g′[Δtτ_itr, β, β, χ, χ]))) * (- g′[Δτs_itr, α, α, α, μ] - g′[s_itr, α, μ, μ, μ] + g′[Δτs_itr, χ, χ, α, μ] + conj(g′[s_itr, μ, α, χ, χ])) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - g[t_itr, α, α, α, α] + g[τ_itr, α, α, α, α] - g[Δτs_itr, α, α, α, α] + g[s_itr, α, α, μ, μ] - g[τ_itr, α, α, μ, μ] + g[Δτs_itr, α, α, μ, μ] + g[t_itr, β, β, α, α] - g[τ_itr, β, β, α, α] + g[Δτs_itr, χ, χ, α, α] - g[t_itr, χ, χ, μ, μ] + g[τ_itr, χ, χ, μ, μ] - g[Δτs_itr, χ, χ, μ, μ] - g[s_itr, μ, μ, μ, μ] + g[t_itr, μ, μ, μ, μ] + conj(g[Δtτ_itr, α, α, β, β]) - conj(g[s_itr, α, α, χ, χ]) + conj(g[t_itr, α, α, χ, χ]) - conj(g[Δtτ_itr, α, α, χ, χ]) - conj(g[Δtτ_itr, β, β, β, β]) - conj(g[t_itr, β, β, χ, χ]) + conj(g[τ_itr, β, β, χ, χ]) + conj(g[Δtτ_itr, β, β, χ, χ]) + conj(g[t_itr, χ, χ, χ, χ]) - conj(g[τ_itr, χ, χ, χ, χ]) + conj(g[s_itr, μ, μ, χ, χ]) - conj(g[t_itr, μ, μ, χ, χ])) / (hbar2))) + (hbar2 * (conj(g″[Δtτ_itr, α, α, β, χ]) - conj(g″[Δtτ_itr, β, β, β, χ])) * (- g′[Δts_itr, α, α, α, μ] - g′[s_itr, α, μ, μ, μ] + g′[Δts_itr, β, β, α, μ] - g′[Δτs_itr, β, β, α, μ] + g′[Δτs_itr, χ, χ, α, μ] + conj(g′[s_itr, μ, α, χ, χ])) + hbar2 * (- g′[Δts_itr, α, α, α, α] - g′[t_itr, α, α, μ, μ] + g′[Δts_itr, α, α, μ, μ] + conj(g′[Δtτ_itr, α, α, β, β]) + conj(g′[t_itr, α, α, χ, χ]) - conj(g′[Δtτ_itr, α, α, χ, χ])) * g″[Δτs_itr, χ, β, α, μ] + (hbar2 * g″[Δτs_itr, χ, β, α, μ] + (- g′[Δts_itr, α, α, α, μ] - g′[s_itr, α, μ, μ, μ] + g′[Δts_itr, β, β, α, μ] - g′[Δτs_itr, β, β, α, μ] + g′[Δτs_itr, χ, χ, α, μ] + conj(g′[s_itr, μ, α, χ, χ])) * (g′[Δτs_itr, χ, β, α, α] + g′[τ_itr, χ, β, μ, μ] - g′[Δτs_itr, χ, β, μ, μ] + conj(g′[Δtτ_itr, α, α, β, χ]) - conj(g′[Δtτ_itr, β, β, β, χ]) - conj(g′[τ_itr, β, χ, χ, χ]))) * (g′[t_itr, α, α, α, α] - g′[t_itr, β, β, α, α] + g′[Δts_itr, β, β, α, α] + g′[t_itr, β, β, μ, μ] - g′[Δts_itr, β, β, μ, μ] - conj(g′[t_itr, α, α, β, β]) + conj(g′[t_itr, β, β, β, β]) - conj(g′[Δtτ_itr, β, β, β, β]) - conj(g′[t_itr, β, β, χ, χ]) + conj(g′[Δtτ_itr, β, β, χ, χ])) + (hbar2 * g″[Δts_itr, α, α, α, μ] - hbar2 * g″[Δts_itr, β, β, α, μ] + (g′[Δts_itr, α, α, α, α] + g′[t_itr, α, α, μ, μ] - g′[Δts_itr, α, α, μ, μ] - conj(g′[Δtτ_itr, α, α, β, β]) - conj(g′[t_itr, α, α, χ, χ]) + conj(g′[Δtτ_itr, α, α, χ, χ])) * (- g′[Δts_itr, α, α, α, μ] - g′[s_itr, α, μ, μ, μ] + g′[Δts_itr, β, β, α, μ] - g′[Δτs_itr, β, β, α, μ] + g′[Δτs_itr, χ, χ, α, μ] + conj(g′[s_itr, μ, α, χ, χ]))) * (- g′[Δτs_itr, χ, β, α, α] - g′[τ_itr, χ, β, μ, μ] + g′[Δτs_itr, χ, β, μ, μ] - conj(g′[Δtτ_itr, α, α, β, χ]) + conj(g′[Δtτ_itr, β, β, β, χ]) + conj(g′[τ_itr, β, χ, χ, χ]))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - g[Δts_itr, α, α, α, α] + g[s_itr, α, α, μ, μ] - g[t_itr, α, α, μ, μ] + g[Δts_itr, α, α, μ, μ] + g[Δts_itr, β, β, α, α] - g[Δτs_itr, β, β, α, α] + g[t_itr, β, β, μ, μ] - g[τ_itr, β, β, μ, μ] - g[Δts_itr, β, β, μ, μ] + g[Δτs_itr, β, β, μ, μ] + g[Δτs_itr, χ, χ, α, α] - g[t_itr, χ, χ, μ, μ] + g[τ_itr, χ, χ, μ, μ] - g[Δτs_itr, χ, χ, μ, μ] - g[s_itr, μ, μ, μ, μ] + g[t_itr, μ, μ, μ, μ] + conj(g[Δtτ_itr, α, α, β, β]) - conj(g[s_itr, α, α, χ, χ]) + conj(g[t_itr, α, α, χ, χ]) - conj(g[Δtτ_itr, α, α, χ, χ]) - conj(g[Δtτ_itr, β, β, β, β]) - conj(g[t_itr, β, β, χ, χ]) + conj(g[τ_itr, β, β, χ, χ]) + conj(g[Δtτ_itr, β, β, χ, χ]) + conj(g[t_itr, χ, χ, χ, χ]) - conj(g[τ_itr, χ, χ, χ, χ]) + conj(g[s_itr, μ, μ, χ, χ]) - conj(g[t_itr, μ, μ, χ, χ])) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, μ, χ) * exp(((- g[s_itr, χ, χ, μ, μ] + g[t_itr, χ, χ, μ, μ] + g[s_itr, μ, μ, μ, μ] - g[t_itr, μ, μ, μ, μ] + conj(g[s_itr, χ, χ, χ, χ]) - conj(g[t_itr, χ, χ, χ, χ]) - conj(g[s_itr, μ, μ, χ, χ]) + conj(g[t_itr, μ, μ, χ, χ])) / (hbar2)))
#         )
#     end

#     @inline function kernel_L0G1_RR(
#         s_itr::Int,
#         τ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((hbar2 * conj(g″[Δtτ_itr, α, α, β, χ]) - hbar2 * conj(g″[Δtτ_itr, β, β, β, χ]) + (- g′[τ_itr, χ, β, α, α] - conj(g′[Δtτ_itr, α, α, β, χ]) + conj(g′[Δtτ_itr, β, β, β, χ]) + conj(g′[τ_itr, β, χ, χ, χ])) * (conj(g′[t_itr, α, α, β, β]) - conj(g′[Δtτ_itr, α, α, β, β]) - conj(g′[t_itr, α, α, χ, χ]) + conj(g′[Δtτ_itr, α, α, χ, χ]) - conj(g′[t_itr, β, β, β, β]) + conj(g′[Δtτ_itr, β, β, β, β]) + conj(g′[t_itr, β, β, χ, χ]) - conj(g′[Δtτ_itr, β, β, χ, χ]))) * (- g′[s_itr, ν, χ, α, α] - conj(g′[Δτs_itr, α, α, χ, ν]) + conj(g′[Δτs_itr, χ, χ, χ, ν]) + conj(g′[s_itr, χ, ν, ν, ν])) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) + g[t_itr, β, β, α, α] - g[τ_itr, β, β, α, α] - g[s_itr, χ, χ, α, α] + g[τ_itr, χ, χ, α, α] + g[s_itr, ν, ν, α, α] - g[t_itr, ν, ν, α, α] + conj(g[Δtτ_itr, α, α, β, β]) + conj(g[t_itr, α, α, χ, χ]) - conj(g[τ_itr, α, α, χ, χ]) - conj(g[Δtτ_itr, α, α, χ, χ]) + conj(g[Δτs_itr, α, α, χ, χ]) - conj(g[t_itr, α, α, ν, ν]) + conj(g[τ_itr, α, α, ν, ν]) - conj(g[Δτs_itr, α, α, ν, ν]) - conj(g[Δtτ_itr, β, β, β, β]) - conj(g[t_itr, β, β, χ, χ]) + conj(g[τ_itr, β, β, χ, χ]) + conj(g[Δtτ_itr, β, β, χ, χ]) - conj(g[Δτs_itr, χ, χ, χ, χ]) + conj(g[s_itr, χ, χ, ν, ν]) - conj(g[τ_itr, χ, χ, ν, ν]) + conj(g[Δτs_itr, χ, χ, ν, ν]) - conj(g[s_itr, ν, ν, ν, ν]) + conj(g[t_itr, ν, ν, ν, ν])) / (hbar2))) + (hbar2 * (- conj(g″[Δtτ_itr, α, α, β, χ]) + conj(g″[Δtτ_itr, β, β, β, χ])) * (- g′[s_itr, ν, χ, α, α] - conj(g′[Δts_itr, α, α, χ, ν]) + conj(g′[Δts_itr, β, β, χ, ν]) - conj(g′[Δτs_itr, β, β, χ, ν]) + conj(g′[Δτs_itr, χ, χ, χ, ν]) + conj(g′[s_itr, χ, ν, ν, ν])) + hbar2 * (- g′[t_itr, β, β, α, α] + conj(g′[Δtτ_itr, β, β, β, β]) + conj(g′[Δts_itr, β, β, χ, χ]) - conj(g′[Δtτ_itr, β, β, χ, χ]) + conj(g′[t_itr, β, β, ν, ν]) - conj(g′[Δts_itr, β, β, ν, ν])) * conj(g″[Δτs_itr, β, χ, χ, ν]) + (hbar2 * conj(g″[Δτs_itr, β, χ, χ, ν]) + (- g′[τ_itr, χ, β, α, α] - conj(g′[Δtτ_itr, α, α, β, χ]) + conj(g′[Δtτ_itr, β, β, β, χ]) + conj(g′[Δτs_itr, β, χ, χ, χ]) + conj(g′[τ_itr, β, χ, ν, ν]) - conj(g′[Δτs_itr, β, χ, ν, ν])) * (g′[s_itr, ν, χ, α, α] + conj(g′[Δts_itr, α, α, χ, ν]) - conj(g′[Δts_itr, β, β, χ, ν]) + conj(g′[Δτs_itr, β, β, χ, ν]) - conj(g′[Δτs_itr, χ, χ, χ, ν]) - conj(g′[s_itr, χ, ν, ν, ν]))) * (g′[t_itr, β, β, α, α] + conj(g′[t_itr, α, α, β, β]) - conj(g′[Δtτ_itr, α, α, β, β]) - conj(g′[Δts_itr, α, α, χ, χ]) + conj(g′[Δtτ_itr, α, α, χ, χ]) - conj(g′[t_itr, α, α, ν, ν]) + conj(g′[Δts_itr, α, α, ν, ν]) - conj(g′[t_itr, β, β, β, β])) + (- hbar2 * conj(g″[Δts_itr, α, α, χ, ν]) + hbar2 * conj(g″[Δts_itr, β, β, χ, ν]) + (g′[t_itr, β, β, α, α] - conj(g′[Δtτ_itr, β, β, β, β]) - conj(g′[Δts_itr, β, β, χ, χ]) + conj(g′[Δtτ_itr, β, β, χ, χ]) - conj(g′[t_itr, β, β, ν, ν]) + conj(g′[Δts_itr, β, β, ν, ν])) * (- g′[s_itr, ν, χ, α, α] - conj(g′[Δts_itr, α, α, χ, ν]) + conj(g′[Δts_itr, β, β, χ, ν]) - conj(g′[Δτs_itr, β, β, χ, ν]) + conj(g′[Δτs_itr, χ, χ, χ, ν]) + conj(g′[s_itr, χ, ν, ν, ν]))) * (- g′[τ_itr, χ, β, α, α] - conj(g′[Δtτ_itr, α, α, β, χ]) + conj(g′[Δtτ_itr, β, β, β, χ]) + conj(g′[Δτs_itr, β, χ, χ, χ]) + conj(g′[τ_itr, β, χ, ν, ν]) - conj(g′[Δτs_itr, β, χ, ν, ν]))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) + g[t_itr, β, β, α, α] - g[τ_itr, β, β, α, α] - g[s_itr, χ, χ, α, α] + g[τ_itr, χ, χ, α, α] + g[s_itr, ν, ν, α, α] - g[t_itr, ν, ν, α, α] + conj(g[Δtτ_itr, α, α, β, β]) + conj(g[Δts_itr, α, α, χ, χ]) - conj(g[Δtτ_itr, α, α, χ, χ]) - conj(g[Δts_itr, α, α, ν, ν]) - conj(g[Δtτ_itr, β, β, β, β]) - conj(g[Δts_itr, β, β, χ, χ]) + conj(g[Δtτ_itr, β, β, χ, χ]) + conj(g[Δτs_itr, β, β, χ, χ]) - conj(g[t_itr, β, β, ν, ν]) + conj(g[τ_itr, β, β, ν, ν]) + conj(g[Δts_itr, β, β, ν, ν]) - conj(g[Δτs_itr, β, β, ν, ν]) - conj(g[Δτs_itr, χ, χ, χ, χ]) + conj(g[s_itr, χ, χ, ν, ν]) - conj(g[τ_itr, χ, χ, ν, ν]) + conj(g[Δτs_itr, χ, χ, ν, ν]) - conj(g[s_itr, ν, ν, ν, ν]) + conj(g[t_itr, ν, ν, ν, ν])) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, α, ν) * exp(((g[s_itr, α, α, α, α] - g[t_itr, α, α, α, α] - g[s_itr, ν, ν, α, α] + g[t_itr, ν, ν, α, α] - conj(g[s_itr, α, α, ν, ν]) + conj(g[t_itr, α, α, ν, ν]) + conj(g[s_itr, ν, ν, ν, ν]) - conj(g[t_itr, ν, ν, ν, ν])) / (hbar2)))
#         )
#     end

#     @inline function calc__rhs_element!(
#         rhs_mat::AbstractMatrix,
#         curr_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#     )
#         rhs = include_PL0P_local ? D_L0P(curr_itr, α, β) * σ_t[α, β] : (0.0 + 0.0im)

#         # -------------------------------------------------------------
#         # P L1 P local left/right mixing.
#         # Left branch:  input (μ, β) -> output (α, β).
#         # Right branch: input (α, ν) -> output (α, β).
#         # These are time-local, so σ_t is used.
#         # -------------------------------------------------------------
#         if include_PL1P_local
#             for μ in 1:n_sys
#                 μ == α && continue
#                 if is_secular_pair(α, β, μ, β)
#                     rhs += -(
#                         σ_t[μ, β]
#                         * (
#                             g′[curr_itr, α, μ, μ, μ]
#                             -conj(g′[curr_itr, μ, α, β, β])
#                         ) / hbar2
#                     )
#                 end
#             end

#             for ν in 1:n_sys
#                 ν == β && continue
#                 if is_secular_pair(α, β, α, ν)
#                     rhs += (
#                         σ_t[α, ν]
#                         * (
#                             g′[curr_itr, ν, β, α, α]
#                             -conj(g′[curr_itr, β, ν, ν, ν])
#                         ) / hbar2
#                     )
#                 end
#             end
#         end

#         if curr_itr > 1
#             integral = 0.0 + 0.0im

#             for s_itr in 1:curr_itr
#                 Δ_itr = curr_itr - s_itr + 1
#                 w_int = ∫weight(s_itr, curr_itr)
#                 kernel = 0.0 + 0.0im

#                 if include_PL0QG0QL1P
#                     # P L0 Q G0 Q L1 P, L branch: input (μ, β).
#                     for μ in 1:n_sys
#                         μ == α && continue
#                         if is_secular_pair(α, β, μ, β)
#                             kernel += kernel_L0_L(s_itr, Δ_itr, curr_itr, σ_t, α, β, μ)
#                         end
#                     end

#                     # P L0 Q G0 Q L1 P, R branch: input (α, ν).
#                     for ν in 1:n_sys
#                         ν == β && continue
#                         if is_secular_pair(α, β, α, ν)
#                             kernel += kernel_L0_R(s_itr, Δ_itr, curr_itr, σ_t, α, β, ν)
#                         end
#                     end
#                 end

#                 if include_PL1QG0QL1P
#                     # P L1 Q G0 Q L1 P, LL: input (μ, β), intermediate χ.
#                     for χ in 1:n_sys
#                         χ == α && continue
#                         for μ in 1:n_sys
#                             μ == χ && continue
#                             if is_secular_pair(α, β, μ, β)
#                                 kernel += kernel_LL(s_itr, Δ_itr, curr_itr, σ_t, α, β, χ, μ)
#                             end
#                         end
#                     end

#                     # P L1 Q G0 Q L1 P, RR: input (α, ν), intermediate χ.
#                     for χ in 1:n_sys
#                         χ == β && continue
#                         for ν in 1:n_sys
#                             ν == χ && continue
#                             if is_secular_pair(α, β, α, ν)
#                                 kernel += kernel_RR(s_itr, Δ_itr, curr_itr, σ_t, α, β, χ, ν)
#                             end
#                         end
#                     end

#                     # P L1 Q G0 Q L1 P, RL: input (μ, χ), left input μ, right intermediate χ.
#                     for μ in 1:n_sys
#                         μ == α && continue
#                         for χ in 1:n_sys
#                             χ == β && continue
#                             if is_secular_pair(α, β, μ, χ)
#                                 kernel += kernel_RL(s_itr, Δ_itr, curr_itr, σ_t, α, β, χ, μ)
#                             end
#                         end
#                     end

#                     # P L1 Q G0 Q L1 P, LR: input (χ, ν), left intermediate χ, right input ν.
#                     for χ in 1:n_sys
#                         χ == α && continue
#                         for ν in 1:n_sys
#                             ν == β && continue
#                             if is_secular_pair(α, β, χ, ν)
#                                 kernel += kernel_LR(s_itr, Δ_itr, curr_itr, σ_t, α, β, χ, ν)
#                             end
#                         end
#                     end
#                 end

#                 if include_PL0QG1QL1P
#                     inner_kernel = 0.0 + 0.0im

#                     for τ_itr in s_itr:curr_itr
#                         w_τ = ∫weight_between(τ_itr, s_itr, curr_itr)
#                         w_τ == 0.0 && continue

#                         τ_kernel = 0.0 + 0.0im

#                         # P L0 Q G1 Q L1 P, LL: input (μ, β), intermediate χ.
#                         for χ in 1:n_sys
#                             χ == α && continue
#                             for μ in 1:n_sys
#                                 μ == χ && continue
#                                 if is_secular_pair(α, β, μ, β)
#                                     τ_kernel += kernel_L0G1_LL(s_itr, τ_itr, curr_itr, σ_t, α, β, χ, μ)
#                                 end
#                             end
#                         end

#                         # P L0 Q G1 Q L1 P, RR: input (α, ν), intermediate χ.
#                         for χ in 1:n_sys
#                             χ == β && continue
#                             for ν in 1:n_sys
#                                 ν == χ && continue
#                                 if is_secular_pair(α, β, α, ν)
#                                     τ_kernel += kernel_L0G1_RR(s_itr, τ_itr, curr_itr, σ_t, α, β, χ, ν)
#                                 end
#                             end
#                         end

#                         # P L0 Q G1 Q L1 P, RL: input (μ, χ), left input μ, right intermediate χ.
#                         for μ in 1:n_sys
#                             μ == α && continue
#                             for χ in 1:n_sys
#                                 χ == β && continue
#                                 if is_secular_pair(α, β, μ, χ)
#                                     τ_kernel += kernel_L0G1_RL(s_itr, τ_itr, curr_itr, σ_t, α, β, χ, μ)
#                                 end
#                             end
#                         end

#                         # P L0 Q G1 Q L1 P, LR: input (χ, ν), left intermediate χ, right input ν.
#                         for χ in 1:n_sys
#                             χ == α && continue
#                             for ν in 1:n_sys
#                                 ν == β && continue
#                                 if is_secular_pair(α, β, χ, ν)
#                                     τ_kernel += kernel_L0G1_LR(s_itr, τ_itr, curr_itr, σ_t, α, β, χ, ν)
#                                 end
#                             end
#                         end

#                         inner_kernel += w_τ * τ_kernel
#                     end

#                     kernel += inner_kernel
#                 end

#                 integral += w_int * kernel
#             end

#             rhs += integral
#         end

#         rhs_mat[α, β] = rhs
#         return nothing
#     end

#     @inline function calc__rhs!(
#         rhs_mat::AbstractMatrix,
#         curr_itr::Int,
#         σ_t::AbstractMatrix,
#     )
#         fill!(rhs_mat, 0.0 + 0.0im)

#         n_components = n_sys * n_sys
#         if use_threads && Threads.nthreads() > 1 && n_components > 1
#             Threads.@threads for linear_idx in 1:n_components
#                 @inbounds begin
#                     α = ((linear_idx - 1) % n_sys) + 1
#                     β = ((linear_idx - 1) ÷ n_sys) + 1
#                     calc__rhs_element!(rhs_mat, curr_itr, σ_t, α, β)
#                 end
#             end
#         else
#             @inbounds for β in 1:n_sys, α in 1:n_sys
#                 calc__rhs_element!(rhs_mat, curr_itr, σ_t, α, β)
#             end
#         end

#         return rhs_mat
#     end

#     @inline function enforce_hermiticity!(σ_next::AbstractMatrix)
#         for i in 1:n_sys
#             σ_next[i, i] = real(σ_next[i, i]) + 0.0im
#         end

#         for i in 1:(n_sys - 1)
#             for j in (i + 1):n_sys
#                 c = 0.5 * (σ_next[i, j] + conj(σ_next[j, i]))
#                 σ_next[i, j] = c
#                 σ_next[j, i] = conj(c)
#             end
#         end

#         return σ_next
#     end

#     σ_stage = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k2      = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k3      = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k4      = Matrix{ComplexF64}(undef, n_sys, n_sys)

#     @inbounds for curr_itr in start_itr:(n_itr - 1)
#         if verbose
#             @printf(
#                 stderr,
#                 "Current iteration: %6d / %6d  method=%s  threads=%d  use_threads=%s  secular=%s\n",
#                 curr_itr,
#                 n_itr,
#                 String(method_sym),
#                 Threads.nthreads(),
#                 string(use_threads),
#                 string(use_secular),
#             )
#         end

#         σ_t    = @view σ[:, :, curr_itr]
#         σ_next = @view σ[:, :, curr_itr + 1]
#         k1     = @view σ′[:, :, curr_itr]

#         if method_sym == :euler
#             calc__rhs!(k1, curr_itr, σ_t)
#             @. σ_next = σ_t + Δt * k1

#         elseif method_sym == :rk2
#             calc__rhs!(k1, curr_itr, σ_t)
#             @. σ_stage = σ_t + 0.5 * Δt * k1
#             calc__rhs!(k2, curr_itr, σ_stage)
#             @. σ_next = σ_t + Δt * k2

#         elseif method_sym == :rk4
#             calc__rhs!(k1, curr_itr, σ_t)

#             @. σ_stage = σ_t + 0.5 * Δt * k1
#             calc__rhs!(k2, curr_itr, σ_stage)

#             @. σ_stage = σ_t + 0.5 * Δt * k2
#             calc__rhs!(k3, curr_itr, σ_stage)

#             @. σ_stage = σ_t + Δt * k3
#             calc__rhs!(k4, curr_itr, σ_stage)

#             @. σ_next = σ_t + (Δt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
#         end

#         if enforce_hermitian
#             enforce_hermiticity!(σ_next)
#         end

#         context.curr_itr = UInt64(curr_itr + 1)
#     end

#     return @view σ[:, :, Int(context.curr_itr)]
# end


# function calc__σ_σ′_secular_core!(
#     context::RmrtContext;
#     hbar::Real = 1.0,
#     use_population_closure::Bool = false,
    
#     # P L0 P local self block.
#     include_PL0P_local::Bool = true,

#     # P L1 P local left/right mixing block.
#     include_PL1P_local::Bool = true,

#     include_PL0QG0QL1P::Bool = true,
#     include_PL1QG0QL1P::Bool = true,

#     # P L0 Q G1 Q L1 P correction. This is a nested s-tau memory integral,
#     # so it is intentionally opt-in by default.
#     include_PL0QG1QL1P::Bool = true,

#     use_secular::Bool = false,
#     secular_tol::Float64 = 1.0e-10,
#     method::Union{Symbol,String} = :rk4,
#     use_threads::Bool = true,

#     # Kept for API compatibility.  This corrected version evaluates RK stage
#     # times through local linear interpolation of g/g′/g″, so the half-grid cache
#     # is not required by this function.
#     auto_prepare_half_shifted_grid::Bool = true,
#     recompute_half_shifted_grid::Bool = false,

#     enforce_hermitian::Bool = true,
#     verbose::Bool = true,
# )
#     start_itr = Int(context.curr_itr)

#     n_sys = context.system.n_sys
#     n_itr = context.simulation_details.num_of_iteration
#     Δt    = context.simulation_details.Δt
#     ϵ     = context.ϵ_exci

#     σ  = context.σ
#     σ′ = context.σ′
#     g  = context.g
#     g′ = context.g′
#     g″ = context.g″

#     hbar_f = Float64(hbar)
#     hbar_f > 0.0 || error("hbar must be positive")
#     hbar2 = hbar_f^2
#     hbar4 = hbar_f^4
#     hbar6 = hbar_f^6

#     method_sym = Symbol(lowercase(String(method)))
#     method_sym in (:euler, :rk2, :rk4) || error(
#         "Unsupported integration method: $(method). Use :euler, :rk2, or :rk4."
#     )

#     # RK2/RK4 stage-time support is handled below by real-coordinate accessors
#     # G/Gp/Gpp.  The half-shifted-grid keyword arguments are kept only for API
#     # compatibility with older callers; this function no longer requires the
#     # half-shifted cache to evaluate midpoint stages.
#     _ = auto_prepare_half_shifted_grid
#     _ = recompute_half_shifted_grid

#     start_itr < n_itr || return @view σ[:, :, n_itr]

#     const_COORD_TOL = 1.0e-10

#     @inline ω(a::Int, b::Int) = ϵ[a] - ϵ[b]
#     @inline Ω(a::Int, b::Int) = (ϵ[a] - ϵ[b]) / hbar_f

#     @inline function is_integer_coord(x::Real)
#         xf = Float64(x)
#         return abs(xf - round(xf)) <= const_COORD_TOL
#     end

#     @inline function clamp_coord(x::Real)
#         xf = Float64(x)
#         if xf < 1.0 - const_COORD_TOL || xf > Float64(n_itr) + const_COORD_TOL
#             error("time-grid coordinate out of range: $(xf), allowed [1, $(n_itr)]")
#         end
#         return min(max(xf, 1.0), Float64(n_itr))
#     end

#     @inline function coord_to_index(x::Real)
#         idx = Int(round(clamp_coord(x)))
#         return min(max(idx, 1), n_itr)
#     end

#     @inline function interp_g(A, x::Real, a::Int, b::Int, c::Int, d::Int)
#         xf = clamp_coord(x)
#         if is_integer_coord(xf)
#             return A[coord_to_index(xf), a, b, c, d]
#         end

#         lo = floor(Int, xf)
#         hi = lo + 1
#         lo = min(max(lo, 1), n_itr)
#         hi = min(max(hi, 1), n_itr)
#         θ = xf - Float64(lo)
#         return (1.0 - θ) * A[lo, a, b, c, d] + θ * A[hi, a, b, c, d]
#     end

#     @inline G(x::Real, a::Int, b::Int, c::Int, d::Int)   = interp_g(g,  x, a, b, c, d)
#     @inline Gp(x::Real, a::Int, b::Int, c::Int, d::Int)  = interp_g(g′, x, a, b, c, d)
#     @inline Gpp(x::Real, a::Int, b::Int, c::Int, d::Int) = interp_g(g″, x, a, b, c, d)


#     @inline function is_secular_pair(
#         out_a::Int,
#         out_b::Int,
#         in_a::Int,
#         in_b::Int,
#     )
#         if !use_secular
#             return true
#         end
#         return abs(Ω(out_a, out_b) - Ω(in_a, in_b)) <= secular_tol
#     end

#     @inline function n_outer_nodes(curr_itr::Real, t_coord::Real)
#         # Nodes for ∫_0^t ds: all stored integer history nodes 1:curr_itr,
#         # plus the RK stage endpoint t_coord when it lies beyond curr_itr.
#         return (Float64(t_coord) > Float64(curr_itr) + const_COORD_TOL) ? curr_itr + 1 : curr_itr
#     end

#     @inline function outer_node_coord(node_idx::Int, curr_itr::Real, t_coord::Real)
#         return node_idx <= curr_itr ? Float64(node_idx) : Float64(t_coord)
#     end

#     @inline function ∫weight_to_t(node_idx::Int, curr_itr::Real, t_coord::Real)
#         n_nodes = n_outer_nodes(curr_itr, t_coord)
#         n_nodes <= 1 && return 0.0

#         x_prev = node_idx == 1       ? nothing : outer_node_coord(node_idx - 1, curr_itr, t_coord)
#         x      = outer_node_coord(node_idx, curr_itr, t_coord)
#         x_next = node_idx == n_nodes ? nothing : outer_node_coord(node_idx + 1, curr_itr, t_coord)

#         if node_idx == 1
#             return 0.5 * Δt * (x_next - x)
#         elseif node_idx == n_nodes
#             return 0.5 * Δt * (x - x_prev)
#         else
#             return 0.5 * Δt * (x_next - x_prev)
#         end
#     end

#     @inline function n_inner_nodes(s_coord::Real, t_coord::Real)
#         Float64(t_coord) <= Float64(s_coord) + const_COORD_TOL && return 1

#         s_int = coord_to_index(s_coord)
#         t_floor = floor(Int, clamp_coord(t_coord))
#         n_int = max(t_floor - s_int + 1, 1)
#         has_stage_endpoint = !is_integer_coord(t_coord)
#         return n_int + (has_stage_endpoint ? 1 : 0)
#     end

#     @inline function inner_node_coord(node_idx::Int, s_coord::Real, t_coord::Real)
#         s_int = coord_to_index(s_coord)
#         t_floor = floor(Int, clamp_coord(t_coord))
#         n_int = max(t_floor - s_int + 1, 1)
#         return node_idx <= n_int ? Float64(s_int + node_idx - 1) : Float64(t_coord)
#     end

#     @inline function ∫weight_between_coord(node_idx::Int, s_coord::Real, t_coord::Real)
#         Float64(t_coord) <= Float64(s_coord) + const_COORD_TOL && return 0.0

#         n_nodes = n_inner_nodes(s_coord, t_coord)
#         n_nodes <= 1 && return 0.0

#         x_prev = node_idx == 1       ? nothing : inner_node_coord(node_idx - 1, s_coord, t_coord)
#         x      = inner_node_coord(node_idx, s_coord, t_coord)
#         x_next = node_idx == n_nodes ? nothing : inner_node_coord(node_idx + 1, s_coord, t_coord)

#         if node_idx == 1
#             return 0.5 * Δt * (x_next - x)
#         elseif node_idx == n_nodes
#             return 0.5 * Δt * (x - x_prev)
#         else
#             return 0.5 * Δt * (x_next - x_prev)
#         end
#     end

#     @inline function Δtime(Δ_coord::Real)
#         return (Float64(Δ_coord) - 1.0) * Δt
#     end

#     @inline function σ_local(σ_t::AbstractMatrix, a::Int, b::Int)
#         return (use_population_closure && a != b) ? (0.0 + 0.0im) : σ_t[a, b]
#     end

#     @inline function σ_mem(
#         σ_t::AbstractMatrix,
#         s_coord::Real,
#         t_coord::Real,
#         a::Int,
#         b::Int,
#     )
#         if use_population_closure && a != b
#             return 0.0 + 0.0im
#         end
#         if abs(Float64(s_coord) - Float64(t_coord)) <= const_COORD_TOL
#             return σ_t[a, b]
#         end
#         is_integer_coord(s_coord) || error("non-endpoint memory coordinate is not stored: $(s_coord)")
#         return σ[a, b, coord_to_index(s_coord)]
#     end

#     @inline function phase_exp(from_a::Int, from_b::Int, Δ_coord::Real)
#         # exp[-i (epsilon_a - epsilon_b) Δ / hbar]
#         return exp(-1.0im * ω(from_a, from_b) * Δtime(Δ_coord) / hbar_f)
#     end

#     @inline function D_L0P(
#         itr::Real,
#         α::Int,
#         β::Int,
#     )
#         return (
#             -1.0im * ω(α, β) / hbar_f
#             +(
#                 -Gp(itr, α, α, α, α)
#                 +conj(Gp(itr, α, α, β, β))
#                 +Gp(itr, β, β, α, α)
#                 -conj(Gp(itr, β, β, β, β))
#             ) / hbar2
#         )
#     end

#     # ---------------------------------------------------------------------
#     # Explicit PL1 Q G0 Q L1 P, g-closed branch prefactors and coefficients.
#     # Branch naming follows the trace side of the two L1 insertions.
#     # All formulas are written with explicit hbar factors.  With hbar=1 this
#     # reduces to the previous internal convention.
#     # ---------------------------------------------------------------------

#     @inline function exponent_LL(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         return (
#             G(Δ_itr, β, β, χ, χ)
#             -G(s_itr, β, β, μ, μ)
#             +G(t_itr, β, β, μ, μ)
#             -G(Δ_itr, β, β, μ, μ)
#             -G(Δ_itr, χ, χ, χ, χ)
#             +G(s_itr, χ, χ, μ, μ)
#             -G(t_itr, χ, χ, μ, μ)
#             +G(Δ_itr, χ, χ, μ, μ)
#             +conj(G(s_itr, β, β, β, β))
#             -conj(G(t_itr, β, β, β, β))
#             -conj(G(s_itr, χ, χ, β, β))
#             +conj(G(t_itr, χ, χ, β, β))
#         )
#     end

#     @inline function coef_LL(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         left_bracket = (
#             Gp(t_itr, α, χ, χ, χ)
#             -Gp(Δ_itr, α, χ, χ, χ)
#             -Gp(t_itr, α, χ, μ, μ)
#             +Gp(Δ_itr, α, χ, μ, μ)
#         )

#         right_bracket = (
#             Gp(Δ_itr, β, β, χ, μ)
#             -Gp(Δ_itr, χ, χ, χ, μ)
#             -Gp(s_itr, χ, μ, μ, μ)
#             +conj(Gp(s_itr, μ, χ, β, β))
#         )

#         return hbar2 * Gpp(Δ_itr, α, χ, χ, μ) - left_bracket * right_bracket
#     end

#     @inline function kernel_LL(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         return -(
#             σ_mem(σ_t, s_itr, t_itr, μ, β)
#             * phase_exp(χ, β, Δ_itr)
#             * exp(exponent_LL(s_itr, Δ_itr, t_itr, α, β, χ, μ) / hbar2)
#             * coef_LL(s_itr, Δ_itr, t_itr, α, β, χ, μ) / hbar4
#         )
#     end

#     @inline function exponent_LR(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         return (
#             -G(s_itr, β, β, χ, χ)
#             +G(t_itr, β, β, χ, χ)
#             +G(s_itr, χ, χ, χ, χ)
#             -G(t_itr, χ, χ, χ, χ)
#             -conj(G(Δ_itr, β, β, β, β))
#             +conj(G(s_itr, β, β, ν, ν))
#             -conj(G(t_itr, β, β, ν, ν))
#             +conj(G(Δ_itr, β, β, ν, ν))
#             +conj(G(Δ_itr, χ, χ, β, β))
#             -conj(G(s_itr, χ, χ, ν, ν))
#             +conj(G(t_itr, χ, χ, ν, ν))
#             -conj(G(Δ_itr, χ, χ, ν, ν))
#         )
#     end

#     @inline function coef_LR(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         right_bracket = (
#             Gp(s_itr, ν, β, χ, χ)
#             -conj(Gp(Δ_itr, β, β, β, ν))
#             -conj(Gp(s_itr, β, ν, ν, ν))
#             +conj(Gp(Δ_itr, χ, χ, β, ν))
#         )

#         left_bracket = (
#             conj(Gp(t_itr, χ, α, β, β))
#             -conj(Gp(Δ_itr, χ, α, β, β))
#             -conj(Gp(t_itr, χ, α, ν, ν))
#             +conj(Gp(Δ_itr, χ, α, ν, ν))
#         )

#         return hbar2 * conj(Gpp(Δ_itr, χ, α, β, ν)) - right_bracket * left_bracket
#     end

#     @inline function kernel_LR(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         return (
#             σ_mem(σ_t, s_itr, t_itr, χ, ν)
#             * phase_exp(χ, β, Δ_itr)
#             * exp(exponent_LR(s_itr, Δ_itr, t_itr, α, β, χ, ν) / hbar2)
#             * coef_LR(s_itr, Δ_itr, t_itr, α, β, χ, ν) / hbar4
#         )
#     end

#     @inline function exponent_RL(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         return (
#             -G(Δ_itr, α, α, α, α)
#             +G(s_itr, α, α, μ, μ)
#             -G(t_itr, α, α, μ, μ)
#             +G(Δ_itr, α, α, μ, μ)
#             +G(Δ_itr, χ, χ, α, α)
#             -G(s_itr, χ, χ, μ, μ)
#             +G(t_itr, χ, χ, μ, μ)
#             -G(Δ_itr, χ, χ, μ, μ)
#             -conj(G(s_itr, α, α, χ, χ))
#             +conj(G(t_itr, α, α, χ, χ))
#             +conj(G(s_itr, χ, χ, χ, χ))
#             -conj(G(t_itr, χ, χ, χ, χ))
#         )
#     end

#     @inline function coef_RL(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         left_bracket = (
#             Gp(Δ_itr, α, α, α, μ)
#             +Gp(s_itr, α, μ, μ, μ)
#             -Gp(Δ_itr, χ, χ, α, μ)
#             -conj(Gp(s_itr, μ, α, χ, χ))
#         )

#         right_bracket = (
#             Gp(t_itr, χ, β, α, α)
#             -Gp(Δ_itr, χ, β, α, α)
#             -Gp(t_itr, χ, β, μ, μ)
#             +Gp(Δ_itr, χ, β, μ, μ)
#         )

#         return hbar2 * Gpp(Δ_itr, χ, β, α, μ) + left_bracket * right_bracket
#     end

#     @inline function kernel_RL(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         return (
#             σ_mem(σ_t, s_itr, t_itr, μ, χ)
#             * phase_exp(α, χ, Δ_itr)
#             * exp(exponent_RL(s_itr, Δ_itr, t_itr, α, β, χ, μ) / hbar2)
#             * coef_RL(s_itr, Δ_itr, t_itr, α, β, χ, μ) / hbar4
#         )
#     end

#     @inline function exponent_RR(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         # This is the g-part after distributing the leading minus sign in
#         # exp(-(i hbar Δε Δ + G_raw) / hbar^2).
#         return (
#             G(s_itr, α, α, α, α)
#             -G(t_itr, α, α, α, α)
#             -G(s_itr, χ, χ, α, α)
#             +G(t_itr, χ, χ, α, α)
#             +conj(G(Δ_itr, α, α, χ, χ))
#             -conj(G(s_itr, α, α, ν, ν))
#             +conj(G(t_itr, α, α, ν, ν))
#             -conj(G(Δ_itr, α, α, ν, ν))
#             -conj(G(Δ_itr, χ, χ, χ, χ))
#             +conj(G(s_itr, χ, χ, ν, ν))
#             -conj(G(t_itr, χ, χ, ν, ν))
#             +conj(G(Δ_itr, χ, χ, ν, ν))
#         )
#     end

#     @inline function coef_RR(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         right_bracket = (
#             Gp(s_itr, ν, χ, α, α)
#             +conj(Gp(Δ_itr, α, α, χ, ν))
#             -conj(Gp(Δ_itr, χ, χ, χ, ν))
#             -conj(Gp(s_itr, χ, ν, ν, ν))
#         )

#         left_bracket = (
#             conj(Gp(t_itr, β, χ, χ, χ))
#             -conj(Gp(Δ_itr, β, χ, χ, χ))
#             -conj(Gp(t_itr, β, χ, ν, ν))
#             +conj(Gp(Δ_itr, β, χ, ν, ν))
#         )

#         return hbar2 * conj(Gpp(Δ_itr, β, χ, χ, ν)) - right_bracket * left_bracket
#     end

#     @inline function kernel_RR(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         return -(
#             σ_mem(σ_t, s_itr, t_itr, α, ν)
#             * phase_exp(α, χ, Δ_itr)
#             * exp(exponent_RR(s_itr, Δ_itr, t_itr, α, β, χ, ν) / hbar2)
#             * coef_RR(s_itr, Δ_itr, t_itr, α, β, χ, ν) / hbar4
#         )
#     end


#     # ---------------------------------------------------------------------
#     # Explicit P L0 Q G0 Q L1 P, g-closed branch prefactors and coefficients.
#     # Branch L : input (μ, β) -> output (α, β)
#     # Branch R : input (α, ν) -> output (α, β)
#     # ---------------------------------------------------------------------

#     @inline function exponent_L0_L(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         μ::Int,
#     )
#         # exp[-(i hbar ω_{αβ} Δ + G_L) / hbar^2]
#         # = phase_exp(α, β, Δ) * exp[-G_L / hbar^2]
#         return (
#             -G(Δ_itr, α, α, α, α)
#             +G(s_itr, α, α, μ, μ)
#             -G(t_itr, α, α, μ, μ)
#             +G(Δ_itr, α, α, μ, μ)
#             +G(Δ_itr, β, β, α, α)
#             -G(s_itr, β, β, μ, μ)
#             +G(t_itr, β, β, μ, μ)
#             -G(Δ_itr, β, β, μ, μ)
#             -conj(G(s_itr, α, α, β, β))
#             +conj(G(t_itr, α, α, β, β))
#             +conj(G(s_itr, β, β, β, β))
#             -conj(G(t_itr, β, β, β, β))
#         )
#     end

#     @inline function coef_L0_L(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         μ::Int,
#     )
#         left_bracket = (
#             Gp(Δ_itr, α, α, α, μ)
#             +Gp(s_itr, α, μ, μ, μ)
#             -Gp(Δ_itr, β, β, α, μ)
#             -conj(Gp(s_itr, μ, α, β, β))
#         )

#         right_bracket = (
#             Gp(t_itr, α, α, α, α)
#             -Gp(Δ_itr, α, α, α, α)
#             -Gp(t_itr, α, α, μ, μ)
#             +Gp(Δ_itr, α, α, μ, μ)
#             -Gp(t_itr, β, β, α, α)
#             +Gp(Δ_itr, β, β, α, α)
#             +Gp(t_itr, β, β, μ, μ)
#             -Gp(Δ_itr, β, β, μ, μ)
#         )

#         return (
#             hbar2 * Gpp(Δ_itr, α, α, α, μ)
#             -hbar2 * Gpp(Δ_itr, β, β, α, μ)
#             +left_bracket * right_bracket
#         )
#     end

#     @inline function kernel_L0_L(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         μ::Int,
#     )
#         return -(
#             σ_mem(σ_t, s_itr, t_itr, μ, β)
#             * phase_exp(α, β, Δ_itr)
#             * exp(exponent_L0_L(s_itr, Δ_itr, t_itr, α, β, μ) / hbar2)
#             * coef_L0_L(s_itr, Δ_itr, t_itr, α, β, μ) / hbar4
#         )
#     end

#     @inline function exponent_L0_R(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         ν::Int,
#     )
#         # exp[-(i hbar ω_{αβ} Δ + G_R) / hbar^2]
#         # = phase_exp(α, β, Δ) * exp[-G_R / hbar^2]
#         return (
#             +G(s_itr, α, α, α, α)
#             -G(t_itr, α, α, α, α)
#             -G(s_itr, β, β, α, α)
#             +G(t_itr, β, β, α, α)
#             +conj(G(Δ_itr, α, α, β, β))
#             -conj(G(s_itr, α, α, ν, ν))
#             +conj(G(t_itr, α, α, ν, ν))
#             -conj(G(Δ_itr, α, α, ν, ν))
#             -conj(G(Δ_itr, β, β, β, β))
#             +conj(G(s_itr, β, β, ν, ν))
#             -conj(G(t_itr, β, β, ν, ν))
#             +conj(G(Δ_itr, β, β, ν, ν))
#         )
#     end

#     @inline function coef_L0_R(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         ν::Int,
#     )
#         left_bracket = (
#             Gp(s_itr, ν, β, α, α)
#             +conj(Gp(Δ_itr, α, α, β, ν))
#             -conj(Gp(Δ_itr, β, β, β, ν))
#             -conj(Gp(s_itr, β, ν, ν, ν))
#         )

#         right_bracket = (
#             conj(Gp(t_itr, α, α, β, β))
#             -conj(Gp(Δ_itr, α, α, β, β))
#             -conj(Gp(t_itr, α, α, ν, ν))
#             +conj(Gp(Δ_itr, α, α, ν, ν))
#             -conj(Gp(t_itr, β, β, β, β))
#             +conj(Gp(Δ_itr, β, β, β, β))
#             +conj(Gp(t_itr, β, β, ν, ν))
#             -conj(Gp(Δ_itr, β, β, ν, ν))
#         )

#         return (
#             -hbar2 * conj(Gpp(Δ_itr, α, α, β, ν))
#             +hbar2 * conj(Gpp(Δ_itr, β, β, β, ν))
#             +left_bracket * right_bracket
#         )
#     end

#     @inline function kernel_L0_R(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         ν::Int,
#     )
#         return -(
#             σ_mem(σ_t, s_itr, t_itr, α, ν)
#             * phase_exp(α, β, Δ_itr)
#             * exp(exponent_L0_R(s_itr, Δ_itr, t_itr, α, β, ν) / hbar2)
#             * coef_L0_R(s_itr, Δ_itr, t_itr, α, β, ν) / hbar4
#         )
#     end



#     # ---------------------------------------------------------------------
#     # Explicit P L0 Q G1 Q L1 P, g-closed 2-time branch kernels.
#     # The extra variable τ satisfies s <= τ <= t.  Each branch is the
#     # direct g-closed expression from I^{*,L0G1L1,g}_{αβ}(t,τ,s).
#     # ---------------------------------------------------------------------

#     @inline function kernel_L0G1_LL(
#         s_itr::Real,
#         τ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((hbar2 * Gpp(Δtτ_itr, α, α, α, χ) - hbar2 * Gpp(Δtτ_itr, β, β, α, χ) + (- Gp(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β))) * (- Gp(t_itr, α, α, α, α) + Gp(Δtτ_itr, α, α, α, α) + Gp(t_itr, α, α, χ, χ) - Gp(Δtτ_itr, α, α, χ, χ) + Gp(t_itr, β, β, α, α) - Gp(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp(Δtτ_itr, β, β, χ, χ))) * (Gp(Δτs_itr, β, β, χ, μ) - Gp(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) + G(t_itr, β, β, χ, χ) - G(τ_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(Δτs_itr, β, β, χ, χ) - G(t_itr, β, β, μ, μ) + G(τ_itr, β, β, μ, μ) - G(Δτs_itr, β, β, μ, μ) - G(Δτs_itr, χ, χ, χ, χ) + G(s_itr, χ, χ, μ, μ) - G(τ_itr, χ, χ, μ, μ) + G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(s_itr, χ, χ, β, β)) + conj(G(τ_itr, χ, χ, β, β)) + conj(G(s_itr, μ, μ, β, β)) - conj(G(t_itr, μ, μ, β, β))) / (hbar2))) + (hbar2 * (- Gpp(Δtτ_itr, α, α, α, χ) + Gpp(Δtτ_itr, β, β, α, χ)) * (- Gp(Δts_itr, α, α, χ, μ) + Gp(Δτs_itr, α, α, χ, μ) + Gp(Δts_itr, β, β, χ, μ) - Gp(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β))) + hbar2 * (- Gp(Δtτ_itr, β, β, α, α) - Gp(Δts_itr, β, β, χ, χ) + Gp(Δtτ_itr, β, β, χ, χ) - Gp(t_itr, β, β, μ, μ) + Gp(Δts_itr, β, β, μ, μ) + conj(Gp(t_itr, β, β, β, β))) * Gpp(Δτs_itr, α, χ, χ, μ) + (hbar2 * Gpp(Δτs_itr, α, χ, χ, μ) + (Gp(Δtτ_itr, α, α, α, χ) + Gp(Δτs_itr, α, χ, χ, χ) + Gp(τ_itr, α, χ, μ, μ) - Gp(Δτs_itr, α, χ, μ, μ) - Gp(Δtτ_itr, β, β, α, χ) - conj(Gp(τ_itr, χ, α, β, β))) * (- Gp(Δts_itr, α, α, χ, μ) + Gp(Δτs_itr, α, α, χ, μ) + Gp(Δts_itr, β, β, χ, μ) - Gp(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β)))) * (- Gp(t_itr, α, α, α, α) + Gp(Δtτ_itr, α, α, α, α) + Gp(Δts_itr, α, α, χ, χ) - Gp(Δtτ_itr, α, α, χ, χ) + Gp(t_itr, α, α, μ, μ) - Gp(Δts_itr, α, α, μ, μ) + Gp(t_itr, β, β, α, α) - conj(Gp(t_itr, β, β, β, β))) + (- hbar2 * Gpp(Δts_itr, α, α, χ, μ) + hbar2 * Gpp(Δts_itr, β, β, χ, μ) + (Gp(Δts_itr, α, α, χ, μ) - Gp(Δτs_itr, α, α, χ, μ) - Gp(Δts_itr, β, β, χ, μ) + Gp(Δτs_itr, χ, χ, χ, μ) + Gp(s_itr, χ, μ, μ, μ) - conj(Gp(s_itr, μ, χ, β, β))) * (- Gp(Δtτ_itr, β, β, α, α) - Gp(Δts_itr, β, β, χ, χ) + Gp(Δtτ_itr, β, β, χ, χ) - Gp(t_itr, β, β, μ, μ) + Gp(Δts_itr, β, β, μ, μ) + conj(Gp(t_itr, β, β, β, β)))) * (- Gp(Δtτ_itr, α, α, α, χ) - Gp(Δτs_itr, α, χ, χ, χ) - Gp(τ_itr, α, χ, μ, μ) + Gp(Δτs_itr, α, χ, μ, μ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(Δts_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δτs_itr, α, α, χ, χ) - G(t_itr, α, α, μ, μ) + G(τ_itr, α, α, μ, μ) + G(Δts_itr, α, α, μ, μ) - G(Δτs_itr, α, α, μ, μ) + G(Δtτ_itr, β, β, α, α) + G(Δts_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) - G(Δts_itr, β, β, μ, μ) - G(Δτs_itr, χ, χ, χ, χ) + G(s_itr, χ, χ, μ, μ) - G(τ_itr, χ, χ, μ, μ) + G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(s_itr, χ, χ, β, β)) + conj(G(τ_itr, χ, χ, β, β)) + conj(G(s_itr, μ, μ, β, β)) - conj(G(t_itr, μ, μ, β, β))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, μ, β) * exp(((- G(s_itr, β, β, μ, μ) + G(t_itr, β, β, μ, μ) + G(s_itr, μ, μ, μ, μ) - G(t_itr, μ, μ, μ, μ) + conj(G(s_itr, β, β, β, β)) - conj(G(t_itr, β, β, β, β)) - conj(G(s_itr, μ, μ, β, β)) + conj(G(t_itr, μ, μ, β, β))) / (hbar2)))
#         )
#     end

#     @inline function kernel_L0G1_LR(
#         s_itr::Real,
#         τ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((- hbar2 * Gpp(Δtτ_itr, α, α, α, χ) + hbar2 * Gpp(Δtτ_itr, β, β, α, χ) + (- Gp(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β))) * (Gp(t_itr, α, α, α, α) - Gp(Δtτ_itr, α, α, α, α) - Gp(t_itr, α, α, χ, χ) + Gp(Δtτ_itr, α, α, χ, χ) - Gp(t_itr, β, β, α, α) + Gp(Δtτ_itr, β, β, α, α) + Gp(t_itr, β, β, χ, χ) - Gp(Δtτ_itr, β, β, χ, χ))) * (- Gp(s_itr, ν, β, χ, χ) + conj(Gp(Δτs_itr, β, β, β, ν)) + conj(Gp(s_itr, β, ν, ν, ν)) - conj(Gp(Δτs_itr, χ, χ, β, ν))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) - G(s_itr, β, β, χ, χ) + G(t_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(t_itr, χ, χ, χ, χ) - G(τ_itr, χ, χ, χ, χ) + G(s_itr, ν, ν, χ, χ) - G(t_itr, ν, ν, χ, χ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(t_itr, β, β, β, β)) + conj(G(τ_itr, β, β, β, β)) - conj(G(Δτs_itr, β, β, β, β)) + conj(G(s_itr, β, β, ν, ν)) - conj(G(τ_itr, β, β, ν, ν)) + conj(G(Δτs_itr, β, β, ν, ν)) + conj(G(Δτs_itr, χ, χ, β, β)) - conj(G(t_itr, χ, χ, ν, ν)) + conj(G(τ_itr, χ, χ, ν, ν)) - conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2))) + (hbar2 * (conj(Gpp(Δts_itr, α, α, β, ν)) - conj(Gpp(Δts_itr, β, β, β, ν))) * (- Gp(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp(Δτs_itr, χ, α, ν, ν))) + hbar2 * (- Gp(Δtτ_itr, α, α, α, α) - Gp(t_itr, α, α, χ, χ) + Gp(Δtτ_itr, α, α, χ, χ) + conj(Gp(Δts_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, ν, ν)) - conj(Gp(Δts_itr, α, α, ν, ν))) * conj(Gpp(Δτs_itr, χ, α, β, ν)) - hbar2 * (- Gp(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp(Δtτ_itr, β, β, χ, χ) + conj(Gp(Δts_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp(Δts_itr, β, β, ν, ν))) * conj(Gpp(Δτs_itr, χ, α, β, ν)) + (hbar2 * conj(Gpp(Δτs_itr, χ, α, β, ν)) + (- Gp(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp(Δτs_itr, χ, α, ν, ν))) * (Gp(s_itr, ν, β, χ, χ) + conj(Gp(Δts_itr, α, α, β, ν)) - conj(Gp(Δτs_itr, α, α, β, ν)) - conj(Gp(Δts_itr, β, β, β, ν)) - conj(Gp(s_itr, β, ν, ν, ν)) + conj(Gp(Δτs_itr, χ, χ, β, ν)))) * (Gp(t_itr, α, α, α, α) - Gp(t_itr, β, β, α, α) - conj(Gp(t_itr, α, α, β, β)) + conj(Gp(t_itr, β, β, β, β))) + (hbar2 * Gpp(Δtτ_itr, α, α, α, χ) - hbar2 * Gpp(Δtτ_itr, β, β, α, χ) + (- Gp(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp(Δτs_itr, χ, α, ν, ν))) * (Gp(Δtτ_itr, α, α, α, α) + Gp(t_itr, α, α, χ, χ) - Gp(Δtτ_itr, α, α, χ, χ) - Gp(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp(Δtτ_itr, β, β, χ, χ) - conj(Gp(Δts_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, ν, ν)) + conj(Gp(Δts_itr, α, α, ν, ν)) + conj(Gp(Δts_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp(Δts_itr, β, β, ν, ν)))) * (- Gp(s_itr, ν, β, χ, χ) - conj(Gp(Δts_itr, α, α, β, ν)) + conj(Gp(Δτs_itr, α, α, β, ν)) + conj(Gp(Δts_itr, β, β, β, ν)) + conj(Gp(s_itr, β, ν, ν, ν)) - conj(Gp(Δτs_itr, χ, χ, β, ν)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) - G(s_itr, β, β, χ, χ) + G(t_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(t_itr, χ, χ, χ, χ) - G(τ_itr, χ, χ, χ, χ) + G(s_itr, ν, ν, χ, χ) - G(t_itr, ν, ν, χ, χ) + conj(G(Δts_itr, α, α, β, β)) - conj(G(Δτs_itr, α, α, β, β)) + conj(G(t_itr, α, α, ν, ν)) - conj(G(τ_itr, α, α, ν, ν)) - conj(G(Δts_itr, α, α, ν, ν)) + conj(G(Δτs_itr, α, α, ν, ν)) - conj(G(Δts_itr, β, β, β, β)) + conj(G(s_itr, β, β, ν, ν)) - conj(G(t_itr, β, β, ν, ν)) + conj(G(Δts_itr, β, β, ν, ν)) + conj(G(Δτs_itr, χ, χ, β, β)) - conj(G(t_itr, χ, χ, ν, ν)) + conj(G(τ_itr, χ, χ, ν, ν)) - conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, χ, ν) * exp(((G(s_itr, χ, χ, χ, χ) - G(t_itr, χ, χ, χ, χ) - G(s_itr, ν, ν, χ, χ) + G(t_itr, ν, ν, χ, χ) - conj(G(s_itr, χ, χ, ν, ν)) + conj(G(t_itr, χ, χ, ν, ν)) + conj(G(s_itr, ν, ν, ν, ν)) - conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))
#         )
#     end

#     @inline function kernel_L0G1_RL(
#         s_itr::Real,
#         τ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((- hbar2 * conj(Gpp(Δtτ_itr, α, α, β, χ)) + hbar2 * conj(Gpp(Δtτ_itr, β, β, β, χ)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp(Δtτ_itr, α, α, β, χ)) + conj(Gp(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ))) * (- conj(Gp(t_itr, α, α, β, β)) + conj(Gp(Δtτ_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, χ, χ)) - conj(Gp(Δtτ_itr, α, α, χ, χ)) + conj(Gp(t_itr, β, β, β, β)) - conj(Gp(Δtτ_itr, β, β, β, β)) - conj(Gp(t_itr, β, β, χ, χ)) + conj(Gp(Δtτ_itr, β, β, χ, χ)))) * (- Gp(Δτs_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(t_itr, α, α, α, α) + G(τ_itr, α, α, α, α) - G(Δτs_itr, α, α, α, α) + G(s_itr, α, α, μ, μ) - G(τ_itr, α, α, μ, μ) + G(Δτs_itr, α, α, μ, μ) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) + G(Δτs_itr, χ, χ, α, α) - G(t_itr, χ, χ, μ, μ) + G(τ_itr, χ, χ, μ, μ) - G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(Δtτ_itr, α, α, β, β)) - conj(G(s_itr, α, α, χ, χ)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(t_itr, χ, χ, χ, χ)) - conj(G(τ_itr, χ, χ, χ, χ)) + conj(G(s_itr, μ, μ, χ, χ)) - conj(G(t_itr, μ, μ, χ, χ))) / (hbar2))) + (hbar2 * (conj(Gpp(Δtτ_itr, α, α, β, χ)) - conj(Gpp(Δtτ_itr, β, β, β, χ))) * (- Gp(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp(Δts_itr, β, β, α, μ) - Gp(Δτs_itr, β, β, α, μ) + Gp(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) + hbar2 * (- Gp(Δts_itr, α, α, α, α) - Gp(t_itr, α, α, μ, μ) + Gp(Δts_itr, α, α, μ, μ) + conj(Gp(Δtτ_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, χ, χ)) - conj(Gp(Δtτ_itr, α, α, χ, χ))) * Gpp(Δτs_itr, χ, β, α, μ) + (hbar2 * Gpp(Δτs_itr, χ, β, α, μ) + (- Gp(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp(Δts_itr, β, β, α, μ) - Gp(Δτs_itr, β, β, α, μ) + Gp(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) * (Gp(Δτs_itr, χ, β, α, α) + Gp(τ_itr, χ, β, μ, μ) - Gp(Δτs_itr, χ, β, μ, μ) + conj(Gp(Δtτ_itr, α, α, β, χ)) - conj(Gp(Δtτ_itr, β, β, β, χ)) - conj(Gp(τ_itr, β, χ, χ, χ)))) * (Gp(t_itr, α, α, α, α) - Gp(t_itr, β, β, α, α) + Gp(Δts_itr, β, β, α, α) + Gp(t_itr, β, β, μ, μ) - Gp(Δts_itr, β, β, μ, μ) - conj(Gp(t_itr, α, α, β, β)) + conj(Gp(t_itr, β, β, β, β)) - conj(Gp(Δtτ_itr, β, β, β, β)) - conj(Gp(t_itr, β, β, χ, χ)) + conj(Gp(Δtτ_itr, β, β, χ, χ))) + (hbar2 * Gpp(Δts_itr, α, α, α, μ) - hbar2 * Gpp(Δts_itr, β, β, α, μ) + (Gp(Δts_itr, α, α, α, α) + Gp(t_itr, α, α, μ, μ) - Gp(Δts_itr, α, α, μ, μ) - conj(Gp(Δtτ_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, χ, χ)) + conj(Gp(Δtτ_itr, α, α, χ, χ))) * (- Gp(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp(Δts_itr, β, β, α, μ) - Gp(Δτs_itr, β, β, α, μ) + Gp(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ)))) * (- Gp(Δτs_itr, χ, β, α, α) - Gp(τ_itr, χ, β, μ, μ) + Gp(Δτs_itr, χ, β, μ, μ) - conj(Gp(Δtτ_itr, α, α, β, χ)) + conj(Gp(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δts_itr, α, α, α, α) + G(s_itr, α, α, μ, μ) - G(t_itr, α, α, μ, μ) + G(Δts_itr, α, α, μ, μ) + G(Δts_itr, β, β, α, α) - G(Δτs_itr, β, β, α, α) + G(t_itr, β, β, μ, μ) - G(τ_itr, β, β, μ, μ) - G(Δts_itr, β, β, μ, μ) + G(Δτs_itr, β, β, μ, μ) + G(Δτs_itr, χ, χ, α, α) - G(t_itr, χ, χ, μ, μ) + G(τ_itr, χ, χ, μ, μ) - G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(Δtτ_itr, α, α, β, β)) - conj(G(s_itr, α, α, χ, χ)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(t_itr, χ, χ, χ, χ)) - conj(G(τ_itr, χ, χ, χ, χ)) + conj(G(s_itr, μ, μ, χ, χ)) - conj(G(t_itr, μ, μ, χ, χ))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, μ, χ) * exp(((- G(s_itr, χ, χ, μ, μ) + G(t_itr, χ, χ, μ, μ) + G(s_itr, μ, μ, μ, μ) - G(t_itr, μ, μ, μ, μ) + conj(G(s_itr, χ, χ, χ, χ)) - conj(G(t_itr, χ, χ, χ, χ)) - conj(G(s_itr, μ, μ, χ, χ)) + conj(G(t_itr, μ, μ, χ, χ))) / (hbar2)))
#         )
#     end

#     @inline function kernel_L0G1_RR(
#         s_itr::Real,
#         τ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((hbar2 * conj(Gpp(Δtτ_itr, α, α, β, χ)) - hbar2 * conj(Gpp(Δtτ_itr, β, β, β, χ)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp(Δtτ_itr, α, α, β, χ)) + conj(Gp(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ))) * (conj(Gp(t_itr, α, α, β, β)) - conj(Gp(Δtτ_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, χ, χ)) + conj(Gp(Δtτ_itr, α, α, χ, χ)) - conj(Gp(t_itr, β, β, β, β)) + conj(Gp(Δtτ_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, χ, χ)) - conj(Gp(Δtτ_itr, β, β, χ, χ)))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp(Δτs_itr, α, α, χ, ν)) + conj(Gp(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) - G(s_itr, χ, χ, α, α) + G(τ_itr, χ, χ, α, α) + G(s_itr, ν, ν, α, α) - G(t_itr, ν, ν, α, α) + conj(G(Δtτ_itr, α, α, β, β)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(τ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) + conj(G(Δτs_itr, α, α, χ, χ)) - conj(G(t_itr, α, α, ν, ν)) + conj(G(τ_itr, α, α, ν, ν)) - conj(G(Δτs_itr, α, α, ν, ν)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) - conj(G(Δτs_itr, χ, χ, χ, χ)) + conj(G(s_itr, χ, χ, ν, ν)) - conj(G(τ_itr, χ, χ, ν, ν)) + conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2))) + (hbar2 * (- conj(Gpp(Δtτ_itr, α, α, β, χ)) + conj(Gpp(Δtτ_itr, β, β, β, χ))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp(Δts_itr, α, α, χ, ν)) + conj(Gp(Δts_itr, β, β, χ, ν)) - conj(Gp(Δτs_itr, β, β, χ, ν)) + conj(Gp(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν))) + hbar2 * (- Gp(t_itr, β, β, α, α) + conj(Gp(Δtτ_itr, β, β, β, β)) + conj(Gp(Δts_itr, β, β, χ, χ)) - conj(Gp(Δtτ_itr, β, β, χ, χ)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp(Δts_itr, β, β, ν, ν))) * conj(Gpp(Δτs_itr, β, χ, χ, ν)) + (hbar2 * conj(Gpp(Δτs_itr, β, χ, χ, ν)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp(Δtτ_itr, α, α, β, χ)) + conj(Gp(Δtτ_itr, β, β, β, χ)) + conj(Gp(Δτs_itr, β, χ, χ, χ)) + conj(Gp(τ_itr, β, χ, ν, ν)) - conj(Gp(Δτs_itr, β, χ, ν, ν))) * (Gp(s_itr, ν, χ, α, α) + conj(Gp(Δts_itr, α, α, χ, ν)) - conj(Gp(Δts_itr, β, β, χ, ν)) + conj(Gp(Δτs_itr, β, β, χ, ν)) - conj(Gp(Δτs_itr, χ, χ, χ, ν)) - conj(Gp(s_itr, χ, ν, ν, ν)))) * (Gp(t_itr, β, β, α, α) + conj(Gp(t_itr, α, α, β, β)) - conj(Gp(Δtτ_itr, α, α, β, β)) - conj(Gp(Δts_itr, α, α, χ, χ)) + conj(Gp(Δtτ_itr, α, α, χ, χ)) - conj(Gp(t_itr, α, α, ν, ν)) + conj(Gp(Δts_itr, α, α, ν, ν)) - conj(Gp(t_itr, β, β, β, β))) + (- hbar2 * conj(Gpp(Δts_itr, α, α, χ, ν)) + hbar2 * conj(Gpp(Δts_itr, β, β, χ, ν)) + (Gp(t_itr, β, β, α, α) - conj(Gp(Δtτ_itr, β, β, β, β)) - conj(Gp(Δts_itr, β, β, χ, χ)) + conj(Gp(Δtτ_itr, β, β, χ, χ)) - conj(Gp(t_itr, β, β, ν, ν)) + conj(Gp(Δts_itr, β, β, ν, ν))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp(Δts_itr, α, α, χ, ν)) + conj(Gp(Δts_itr, β, β, χ, ν)) - conj(Gp(Δτs_itr, β, β, χ, ν)) + conj(Gp(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν)))) * (- Gp(τ_itr, χ, β, α, α) - conj(Gp(Δtτ_itr, α, α, β, χ)) + conj(Gp(Δtτ_itr, β, β, β, χ)) + conj(Gp(Δτs_itr, β, χ, χ, χ)) + conj(Gp(τ_itr, β, χ, ν, ν)) - conj(Gp(Δτs_itr, β, χ, ν, ν)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) - G(s_itr, χ, χ, α, α) + G(τ_itr, χ, χ, α, α) + G(s_itr, ν, ν, α, α) - G(t_itr, ν, ν, α, α) + conj(G(Δtτ_itr, α, α, β, β)) + conj(G(Δts_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δts_itr, α, α, ν, ν)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(Δts_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(Δτs_itr, β, β, χ, χ)) - conj(G(t_itr, β, β, ν, ν)) + conj(G(τ_itr, β, β, ν, ν)) + conj(G(Δts_itr, β, β, ν, ν)) - conj(G(Δτs_itr, β, β, ν, ν)) - conj(G(Δτs_itr, χ, χ, χ, χ)) + conj(G(s_itr, χ, χ, ν, ν)) - conj(G(τ_itr, χ, χ, ν, ν)) + conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, α, ν) * exp(((G(s_itr, α, α, α, α) - G(t_itr, α, α, α, α) - G(s_itr, ν, ν, α, α) + G(t_itr, ν, ν, α, α) - conj(G(s_itr, α, α, ν, ν)) + conj(G(t_itr, α, α, ν, ν)) + conj(G(s_itr, ν, ν, ν, ν)) - conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))
#         )
#     end

#     @inline function calc__rhs_element!(
#         rhs_mat::AbstractMatrix,
#         curr_itr::Int,
#         t_eval_coord::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#     )
#         if use_population_closure && α != β
#             rhs_mat[α, β] = 0.0 + 0.0im
#             return nothing
#         end

#         rhs = include_PL0P_local ? D_L0P(t_eval_coord, α, β) * σ_local(σ_t, α, β) : (0.0 + 0.0im)

#         # -------------------------------------------------------------
#         # P L1 P local left/right mixing.
#         # Left branch:  input (μ, β) -> output (α, β).
#         # Right branch: input (α, ν) -> output (α, β).
#         # These are time-local, so σ_t and t_eval_coord are used.
#         # -------------------------------------------------------------
#         if include_PL1P_local
#             for μ in 1:n_sys
#                 μ == α && continue
#                 if is_secular_pair(α, β, μ, β)
#                     rhs += -(
#                         σ_local(σ_t, μ, β)
#                         * (
#                             Gp(t_eval_coord, α, μ, μ, μ)
#                             -conj(Gp(t_eval_coord, μ, α, β, β))
#                         ) / hbar2
#                     )
#                 end
#             end

#             for ν in 1:n_sys
#                 ν == β && continue
#                 if is_secular_pair(α, β, α, ν)
#                     rhs += (
#                         σ_local(σ_t, α, ν)
#                         * (
#                             Gp(t_eval_coord, ν, β, α, α)
#                             -conj(Gp(t_eval_coord, β, ν, ν, ν))
#                         ) / hbar2
#                     )
#                 end
#             end
#         end

#         if Float64(t_eval_coord) > 1.0 + const_COORD_TOL
#             integral = 0.0 + 0.0im
#             n_s_nodes = n_outer_nodes(curr_itr, t_eval_coord)

#             for s_node in 1:n_s_nodes
#                 s_coord = outer_node_coord(s_node, curr_itr, t_eval_coord)
#                 Δ_coord = Float64(t_eval_coord) - s_coord + 1.0
#                 w_int = ∫weight_to_t(s_node, curr_itr, t_eval_coord)
#                 w_int == 0.0 && continue

#                 kernel = 0.0 + 0.0im

#                 if include_PL0QG0QL1P
#                     # P L0 Q G0 Q L1 P, L branch: input (μ, β).
#                     for μ in 1:n_sys
#                         μ == α && continue
#                         if is_secular_pair(α, β, μ, β)
#                             kernel += kernel_L0_L(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, μ)
#                         end
#                     end

#                     # P L0 Q G0 Q L1 P, R branch: input (α, ν).
#                     for ν in 1:n_sys
#                         ν == β && continue
#                         if is_secular_pair(α, β, α, ν)
#                             kernel += kernel_L0_R(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, ν)
#                         end
#                     end
#                 end

#                 if include_PL1QG0QL1P
#                     # P L1 Q G0 Q L1 P, LL: input (μ, β), intermediate χ.
#                     for χ in 1:n_sys
#                         χ == α && continue
#                         for μ in 1:n_sys
#                             μ == χ && continue
#                             if is_secular_pair(α, β, μ, β)
#                                 kernel += kernel_LL(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, χ, μ)
#                             end
#                         end
#                     end

#                     # P L1 Q G0 Q L1 P, RR: input (α, ν), intermediate χ.
#                     for χ in 1:n_sys
#                         χ == β && continue
#                         for ν in 1:n_sys
#                             ν == χ && continue
#                             if is_secular_pair(α, β, α, ν)
#                                 kernel += kernel_RR(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, χ, ν)
#                             end
#                         end
#                     end

#                     # P L1 Q G0 Q L1 P, RL: input (μ, χ), left input μ, right intermediate χ.
#                     for μ in 1:n_sys
#                         μ == α && continue
#                         for χ in 1:n_sys
#                             χ == β && continue
#                             if is_secular_pair(α, β, μ, χ)
#                                 kernel += kernel_RL(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, χ, μ)
#                             end
#                         end
#                     end

#                     # P L1 Q G0 Q L1 P, LR: input (χ, ν), left intermediate χ, right input ν.
#                     for χ in 1:n_sys
#                         χ == α && continue
#                         for ν in 1:n_sys
#                             ν == β && continue
#                             if is_secular_pair(α, β, χ, ν)
#                                 kernel += kernel_LR(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, χ, ν)
#                             end
#                         end
#                     end
#                 end

#                 if include_PL0QG1QL1P
#                     inner_kernel = 0.0 + 0.0im

#                     n_τ_nodes = n_inner_nodes(s_coord, t_eval_coord)
#                     for τ_node in 1:n_τ_nodes
#                         τ_coord = inner_node_coord(τ_node, s_coord, t_eval_coord)
#                         w_τ = ∫weight_between_coord(τ_node, s_coord, t_eval_coord)
#                         w_τ == 0.0 && continue

#                         τ_kernel = 0.0 + 0.0im

#                         # P L0 Q G1 Q L1 P, LL: input (μ, β), intermediate χ.
#                         for χ in 1:n_sys
#                             χ == α && continue
#                             for μ in 1:n_sys
#                                 μ == χ && continue
#                                 if is_secular_pair(α, β, μ, β)
#                                     τ_kernel += kernel_L0G1_LL(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, μ)
#                                 end
#                             end
#                         end

#                         # P L0 Q G1 Q L1 P, RR: input (α, ν), intermediate χ.
#                         for χ in 1:n_sys
#                             χ == β && continue
#                             for ν in 1:n_sys
#                                 ν == χ && continue
#                                 if is_secular_pair(α, β, α, ν)
#                                     τ_kernel += kernel_L0G1_RR(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, ν)
#                                 end
#                             end
#                         end

#                         # P L0 Q G1 Q L1 P, RL: input (μ, χ), left input μ, right intermediate χ.
#                         for μ in 1:n_sys
#                             μ == α && continue
#                             for χ in 1:n_sys
#                                 χ == β && continue
#                                 if is_secular_pair(α, β, μ, χ)
#                                     τ_kernel += kernel_L0G1_RL(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, μ)
#                                 end
#                             end
#                         end

#                         # P L0 Q G1 Q L1 P, LR: input (χ, ν), left intermediate χ, right input ν.
#                         for χ in 1:n_sys
#                             χ == α && continue
#                             for ν in 1:n_sys
#                                 ν == β && continue
#                                 if is_secular_pair(α, β, χ, ν)
#                                     τ_kernel += kernel_L0G1_LR(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, ν)
#                                 end
#                             end
#                         end

#                         inner_kernel += w_τ * τ_kernel
#                     end

#                     kernel += inner_kernel
#                 end

#                 integral += w_int * kernel
#             end

#             rhs += integral
#         end

#         rhs_mat[α, β] = rhs
#         return nothing
#     end

#     @inline function calc__rhs!(
#         rhs_mat::AbstractMatrix,
#         curr_itr::Int,
#         t_eval_coord::Real,
#         σ_t::AbstractMatrix,
#     )
#         fill!(rhs_mat, 0.0 + 0.0im)

#         n_components = n_sys * n_sys
#         if use_threads && Threads.nthreads() > 1 && n_components > 1
#             Threads.@threads for linear_idx in 1:n_components
#                 @inbounds begin
#                     α = ((linear_idx - 1) % n_sys) + 1
#                     β = ((linear_idx - 1) ÷ n_sys) + 1
#                     calc__rhs_element!(rhs_mat, curr_itr, t_eval_coord, σ_t, α, β)
#                 end
#             end
#         else
#             @inbounds for β in 1:n_sys, α in 1:n_sys
#                 calc__rhs_element!(rhs_mat, curr_itr, t_eval_coord, σ_t, α, β)
#             end
#         end

#         return rhs_mat
#     end

#     @inline function enforce_hermiticity!(σ_next::AbstractMatrix)
#         for i in 1:n_sys
#             σ_next[i, i] = real(σ_next[i, i]) + 0.0im
#         end

#         for i in 1:(n_sys - 1)
#             for j in (i + 1):n_sys
#                 c = 0.5 * (σ_next[i, j] + conj(σ_next[j, i]))
#                 σ_next[i, j] = c
#                 σ_next[j, i] = conj(c)
#             end
#         end

#         return σ_next
#     end

#     @inline function enforce_population_closure!(σ_next::AbstractMatrix)
#         for j in 1:n_sys, i in 1:n_sys
#             if i != j
#                 σ_next[i, j] = 0.0 + 0.0im
#             else
#                 σ_next[i, i] = real(σ_next[i, i]) + 0.0im
#             end
#         end
#         return σ_next
#     end

#     σ_stage = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k2      = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k3      = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k4      = Matrix{ComplexF64}(undef, n_sys, n_sys)

#     @inbounds for curr_itr in start_itr:(n_itr - 1)
#         if verbose
#             @printf(
#                 stderr,
#                 "Current iteration: %6d / %6d  method=%s  threads=%d  use_threads=%s  secular=%s\n",
#                 curr_itr,
#                 n_itr,
#                 String(method_sym),
#                 Threads.nthreads(),
#                 string(use_threads),
#                 string(use_secular),
#             )
#         end

#         σ_t    = @view σ[:, :, curr_itr]
#         σ_next = @view σ[:, :, curr_itr + 1]
#         k1     = @view σ′[:, :, curr_itr]

#         if method_sym == :euler
#             calc__rhs!(k1, curr_itr, Float64(curr_itr), σ_t)
#             @. σ_next = σ_t + Δt * k1

#         elseif method_sym == :rk2
#             calc__rhs!(k1, curr_itr, Float64(curr_itr), σ_t)
#             @. σ_stage = σ_t + 0.5 * Δt * k1
#             use_population_closure && enforce_population_closure!(σ_stage)
#             calc__rhs!(k2, curr_itr, Float64(curr_itr) + 0.5, σ_stage)
#             @. σ_next = σ_t + Δt * k2

#         elseif method_sym == :rk4
#             calc__rhs!(k1, curr_itr, Float64(curr_itr), σ_t)

#             @. σ_stage = σ_t + 0.5 * Δt * k1
#             use_population_closure && enforce_population_closure!(σ_stage)
#             calc__rhs!(k2, curr_itr, Float64(curr_itr) + 0.5, σ_stage)

#             @. σ_stage = σ_t + 0.5 * Δt * k2
#             use_population_closure && enforce_population_closure!(σ_stage)
#             calc__rhs!(k3, curr_itr, Float64(curr_itr) + 0.5, σ_stage)

#             @. σ_stage = σ_t + Δt * k3
#             use_population_closure && enforce_population_closure!(σ_stage)
#             calc__rhs!(k4, curr_itr, Float64(curr_itr) + 1.0, σ_stage)

#             @. σ_next = σ_t + (Δt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
#         end

#         if use_population_closure
#             enforce_population_closure!(σ_next)
#         elseif enforce_hermitian
#             enforce_hermiticity!(σ_next)
#         end

#         context.curr_itr = UInt64(curr_itr + 1)
#     end

#     return @view σ[:, :, Int(context.curr_itr)]
# end


# function calc__σ_σ′_secular_core!(
#     context::RmrtContext;
#     hbar::Real = 1.0,
#     use_population_closure::Bool = false,
    
#     # P L0 P local self block.
#     include_PL0P_local::Bool = true,

#     # P L1 P local left/right mixing block.
#     include_PL1P_local::Bool = true,

#     include_PL0QG0QL1P::Bool = true,
#     include_PL1QG0QL1P::Bool = true,

#     # P L0 Q G1 Q L1 P correction. This is a nested s-tau memory integral,
#     # so it is intentionally opt-in by default.
#     include_PL0QG1QL1P::Bool = true,

#     # Markovianize only the relative-time g′/g″ factors that belong to the
#     # inner G1 correction in P L0 Q G1 Q L1 P.  In this mode,
#     #     g′_{abcd}(∞) ≈ -im * Λ_{abcd},      g″_{abcd}(∞) ≈ 0,
#     # while the outer s/t Gaussian dressing factors are still evaluated from g.
#     markovianize_PL0QG1QL1P_G1::Bool = false,

#     use_secular::Bool = false,
#     secular_tol::Float64 = 1.0e-10,
#     method::Union{Symbol,String} = :rk4,
#     use_threads::Bool = true,

#     # Kept for API compatibility.  This corrected version evaluates RK stage
#     # times through local linear interpolation of g/g′/g″, so the half-grid cache
#     # is not required by this function.
#     auto_prepare_half_shifted_grid::Bool = true,
#     recompute_half_shifted_grid::Bool = false,

#     enforce_hermitian::Bool = true,
#     verbose::Bool = true,
# )
#     start_itr = Int(context.curr_itr)

#     n_sys = context.system.n_sys
#     n_itr = context.simulation_details.num_of_iteration
#     Δt    = context.simulation_details.Δt
#     ϵ     = context.ϵ_exci

#     σ  = context.σ
#     σ′ = context.σ′
#     g  = context.g
#     g′ = context.g′
#     g″ = context.g″

#     if include_PL0QG1QL1P && markovianize_PL0QG1QL1P_G1 && !hasproperty(context, :Λ)
#         error("markovianize_PL0QG1QL1P_G1=true requires context.Λ, with Λ[a,b,c,d] ≈ i*g′_{abcd}(∞).")
#     end
#     Λ = (include_PL0QG1QL1P && markovianize_PL0QG1QL1P_G1) ? getfield(context, :Λ) : nothing

#     hbar_f = Float64(hbar)
#     hbar_f > 0.0 || error("hbar must be positive")
#     hbar2 = hbar_f^2
#     hbar4 = hbar_f^4
#     hbar6 = hbar_f^6

#     method_sym = Symbol(lowercase(String(method)))
#     method_sym in (:euler, :rk2, :rk4) || error(
#         "Unsupported integration method: $(method). Use :euler, :rk2, or :rk4."
#     )

#     # RK2/RK4 stage-time support is handled below by real-coordinate accessors
#     # G/Gp/Gpp.  The half-shifted-grid keyword arguments are kept only for API
#     # compatibility with older callers; this function no longer requires the
#     # half-shifted cache to evaluate midpoint stages.
#     _ = auto_prepare_half_shifted_grid
#     _ = recompute_half_shifted_grid

#     start_itr < n_itr || return @view σ[:, :, n_itr]

#     const_COORD_TOL = 1.0e-10

#     @inline ω(a::Int, b::Int) = ϵ[a] - ϵ[b]
#     @inline Ω(a::Int, b::Int) = (ϵ[a] - ϵ[b]) / hbar_f

#     @inline function is_integer_coord(x::Real)
#         xf = Float64(x)
#         return abs(xf - round(xf)) <= const_COORD_TOL
#     end

#     @inline function clamp_coord(x::Real)
#         xf = Float64(x)
#         if xf < 1.0 - const_COORD_TOL || xf > Float64(n_itr) + const_COORD_TOL
#             error("time-grid coordinate out of range: $(xf), allowed [1, $(n_itr)]")
#         end
#         return min(max(xf, 1.0), Float64(n_itr))
#     end

#     @inline function coord_to_index(x::Real)
#         idx = Int(round(clamp_coord(x)))
#         return min(max(idx, 1), n_itr)
#     end

#     @inline function interp_g(A, x::Real, a::Int, b::Int, c::Int, d::Int)
#         xf = clamp_coord(x)
#         if is_integer_coord(xf)
#             return A[coord_to_index(xf), a, b, c, d]
#         end

#         lo = floor(Int, xf)
#         hi = lo + 1
#         lo = min(max(lo, 1), n_itr)
#         hi = min(max(hi, 1), n_itr)
#         θ = xf - Float64(lo)
#         return (1.0 - θ) * A[lo, a, b, c, d] + θ * A[hi, a, b, c, d]
#     end

#     @inline G(x::Real, a::Int, b::Int, c::Int, d::Int)   = interp_g(g,  x, a, b, c, d)
#     @inline Gp(x::Real, a::Int, b::Int, c::Int, d::Int)  = interp_g(g′, x, a, b, c, d)
#     @inline Gpp(x::Real, a::Int, b::Int, c::Int, d::Int) = interp_g(g″, x, a, b, c, d)

#     # Markovian asymptote used only by the optional inner-G1 version of
#     # P L0 Q G1 Q L1 P.  Because Λ stores i*g′(∞), the replacement is
#     # g′(∞) = -im*Λ and g″(∞) = 0.
#     @inline Gp_markovian_G1(Δ_coord::Real, a::Int, b::Int, c::Int, d::Int) =
#         -1.0im * Λ[a, b, c, d]

#     @inline Gpp_markovian_G1(Δ_coord::Real, a::Int, b::Int, c::Int, d::Int) =
#         0.0 + 0.0im


#     @inline function is_secular_pair(
#         out_a::Int,
#         out_b::Int,
#         in_a::Int,
#         in_b::Int,
#     )
#         if !use_secular
#             return true
#         end
#         return abs(Ω(out_a, out_b) - Ω(in_a, in_b)) <= secular_tol
#     end

#     @inline function n_outer_nodes(curr_itr::Real, t_coord::Real)
#         # Nodes for ∫_0^t ds: all stored integer history nodes 1:curr_itr,
#         # plus the RK stage endpoint t_coord when it lies beyond curr_itr.
#         return (Float64(t_coord) > Float64(curr_itr) + const_COORD_TOL) ? curr_itr + 1 : curr_itr
#     end

#     @inline function outer_node_coord(node_idx::Int, curr_itr::Real, t_coord::Real)
#         return node_idx <= curr_itr ? Float64(node_idx) : Float64(t_coord)
#     end

#     @inline function ∫weight_to_t(node_idx::Int, curr_itr::Real, t_coord::Real)
#         n_nodes = n_outer_nodes(curr_itr, t_coord)
#         n_nodes <= 1 && return 0.0

#         x_prev = node_idx == 1       ? nothing : outer_node_coord(node_idx - 1, curr_itr, t_coord)
#         x      = outer_node_coord(node_idx, curr_itr, t_coord)
#         x_next = node_idx == n_nodes ? nothing : outer_node_coord(node_idx + 1, curr_itr, t_coord)

#         if node_idx == 1
#             return 0.5 * Δt * (x_next - x)
#         elseif node_idx == n_nodes
#             return 0.5 * Δt * (x - x_prev)
#         else
#             return 0.5 * Δt * (x_next - x_prev)
#         end
#     end

#     @inline function n_inner_nodes(s_coord::Real, t_coord::Real)
#         Float64(t_coord) <= Float64(s_coord) + const_COORD_TOL && return 1

#         s_int = coord_to_index(s_coord)
#         t_floor = floor(Int, clamp_coord(t_coord))
#         n_int = max(t_floor - s_int + 1, 1)
#         has_stage_endpoint = !is_integer_coord(t_coord)
#         return n_int + (has_stage_endpoint ? 1 : 0)
#     end

#     @inline function inner_node_coord(node_idx::Int, s_coord::Real, t_coord::Real)
#         s_int = coord_to_index(s_coord)
#         t_floor = floor(Int, clamp_coord(t_coord))
#         n_int = max(t_floor - s_int + 1, 1)
#         return node_idx <= n_int ? Float64(s_int + node_idx - 1) : Float64(t_coord)
#     end

#     @inline function ∫weight_between_coord(node_idx::Int, s_coord::Real, t_coord::Real)
#         Float64(t_coord) <= Float64(s_coord) + const_COORD_TOL && return 0.0

#         n_nodes = n_inner_nodes(s_coord, t_coord)
#         n_nodes <= 1 && return 0.0

#         x_prev = node_idx == 1       ? nothing : inner_node_coord(node_idx - 1, s_coord, t_coord)
#         x      = inner_node_coord(node_idx, s_coord, t_coord)
#         x_next = node_idx == n_nodes ? nothing : inner_node_coord(node_idx + 1, s_coord, t_coord)

#         if node_idx == 1
#             return 0.5 * Δt * (x_next - x)
#         elseif node_idx == n_nodes
#             return 0.5 * Δt * (x - x_prev)
#         else
#             return 0.5 * Δt * (x_next - x_prev)
#         end
#     end

#     @inline function Δtime(Δ_coord::Real)
#         return (Float64(Δ_coord) - 1.0) * Δt
#     end

#     @inline function σ_local(σ_t::AbstractMatrix, a::Int, b::Int)
#         return (use_population_closure && a != b) ? (0.0 + 0.0im) : σ_t[a, b]
#     end

#     @inline function σ_mem(
#         σ_t::AbstractMatrix,
#         s_coord::Real,
#         t_coord::Real,
#         a::Int,
#         b::Int,
#     )
#         if use_population_closure && a != b
#             return 0.0 + 0.0im
#         end
#         if abs(Float64(s_coord) - Float64(t_coord)) <= const_COORD_TOL
#             return σ_t[a, b]
#         end
#         is_integer_coord(s_coord) || error("non-endpoint memory coordinate is not stored: $(s_coord)")
#         return σ[a, b, coord_to_index(s_coord)]
#     end

#     @inline function phase_exp(from_a::Int, from_b::Int, Δ_coord::Real)
#         # exp[-i (epsilon_a - epsilon_b) Δ / hbar]
#         return exp(-1.0im * ω(from_a, from_b) * Δtime(Δ_coord) / hbar_f)
#     end

#     @inline function D_L0P(
#         itr::Real,
#         α::Int,
#         β::Int,
#     )
#         return (
#             -1.0im * ω(α, β) / hbar_f
#             +(
#                 -Gp(itr, α, α, α, α)
#                 +conj(Gp(itr, α, α, β, β))
#                 +Gp(itr, β, β, α, α)
#                 -conj(Gp(itr, β, β, β, β))
#             ) / hbar2
#         )
#     end

#     # ---------------------------------------------------------------------
#     # Explicit PL1 Q G0 Q L1 P, g-closed branch prefactors and coefficients.
#     # Branch naming follows the trace side of the two L1 insertions.
#     # All formulas are written with explicit hbar factors.  With hbar=1 this
#     # reduces to the previous internal convention.
#     # ---------------------------------------------------------------------

#     @inline function exponent_LL(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         return (
#             G(Δ_itr, β, β, χ, χ)
#             -G(s_itr, β, β, μ, μ)
#             +G(t_itr, β, β, μ, μ)
#             -G(Δ_itr, β, β, μ, μ)
#             -G(Δ_itr, χ, χ, χ, χ)
#             +G(s_itr, χ, χ, μ, μ)
#             -G(t_itr, χ, χ, μ, μ)
#             +G(Δ_itr, χ, χ, μ, μ)
#             +conj(G(s_itr, β, β, β, β))
#             -conj(G(t_itr, β, β, β, β))
#             -conj(G(s_itr, χ, χ, β, β))
#             +conj(G(t_itr, χ, χ, β, β))
#         )
#     end

#     @inline function coef_LL(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         left_bracket = (
#             Gp(t_itr, α, χ, χ, χ)
#             -Gp(Δ_itr, α, χ, χ, χ)
#             -Gp(t_itr, α, χ, μ, μ)
#             +Gp(Δ_itr, α, χ, μ, μ)
#         )

#         right_bracket = (
#             Gp(Δ_itr, β, β, χ, μ)
#             -Gp(Δ_itr, χ, χ, χ, μ)
#             -Gp(s_itr, χ, μ, μ, μ)
#             +conj(Gp(s_itr, μ, χ, β, β))
#         )

#         return hbar2 * Gpp(Δ_itr, α, χ, χ, μ) - left_bracket * right_bracket
#     end

#     @inline function kernel_LL(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         return -(
#             σ_mem(σ_t, s_itr, t_itr, μ, β)
#             * phase_exp(χ, β, Δ_itr)
#             * exp(exponent_LL(s_itr, Δ_itr, t_itr, α, β, χ, μ) / hbar2)
#             * coef_LL(s_itr, Δ_itr, t_itr, α, β, χ, μ) / hbar4
#         )
#     end

#     @inline function exponent_LR(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         return (
#             -G(s_itr, β, β, χ, χ)
#             +G(t_itr, β, β, χ, χ)
#             +G(s_itr, χ, χ, χ, χ)
#             -G(t_itr, χ, χ, χ, χ)
#             -conj(G(Δ_itr, β, β, β, β))
#             +conj(G(s_itr, β, β, ν, ν))
#             -conj(G(t_itr, β, β, ν, ν))
#             +conj(G(Δ_itr, β, β, ν, ν))
#             +conj(G(Δ_itr, χ, χ, β, β))
#             -conj(G(s_itr, χ, χ, ν, ν))
#             +conj(G(t_itr, χ, χ, ν, ν))
#             -conj(G(Δ_itr, χ, χ, ν, ν))
#         )
#     end

#     @inline function coef_LR(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         right_bracket = (
#             Gp(s_itr, ν, β, χ, χ)
#             -conj(Gp(Δ_itr, β, β, β, ν))
#             -conj(Gp(s_itr, β, ν, ν, ν))
#             +conj(Gp(Δ_itr, χ, χ, β, ν))
#         )

#         left_bracket = (
#             conj(Gp(t_itr, χ, α, β, β))
#             -conj(Gp(Δ_itr, χ, α, β, β))
#             -conj(Gp(t_itr, χ, α, ν, ν))
#             +conj(Gp(Δ_itr, χ, α, ν, ν))
#         )

#         return hbar2 * conj(Gpp(Δ_itr, χ, α, β, ν)) - right_bracket * left_bracket
#     end

#     @inline function kernel_LR(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         return (
#             σ_mem(σ_t, s_itr, t_itr, χ, ν)
#             * phase_exp(χ, β, Δ_itr)
#             * exp(exponent_LR(s_itr, Δ_itr, t_itr, α, β, χ, ν) / hbar2)
#             * coef_LR(s_itr, Δ_itr, t_itr, α, β, χ, ν) / hbar4
#         )
#     end

#     @inline function exponent_RL(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         return (
#             -G(Δ_itr, α, α, α, α)
#             +G(s_itr, α, α, μ, μ)
#             -G(t_itr, α, α, μ, μ)
#             +G(Δ_itr, α, α, μ, μ)
#             +G(Δ_itr, χ, χ, α, α)
#             -G(s_itr, χ, χ, μ, μ)
#             +G(t_itr, χ, χ, μ, μ)
#             -G(Δ_itr, χ, χ, μ, μ)
#             -conj(G(s_itr, α, α, χ, χ))
#             +conj(G(t_itr, α, α, χ, χ))
#             +conj(G(s_itr, χ, χ, χ, χ))
#             -conj(G(t_itr, χ, χ, χ, χ))
#         )
#     end

#     @inline function coef_RL(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         left_bracket = (
#             Gp(Δ_itr, α, α, α, μ)
#             +Gp(s_itr, α, μ, μ, μ)
#             -Gp(Δ_itr, χ, χ, α, μ)
#             -conj(Gp(s_itr, μ, α, χ, χ))
#         )

#         right_bracket = (
#             Gp(t_itr, χ, β, α, α)
#             -Gp(Δ_itr, χ, β, α, α)
#             -Gp(t_itr, χ, β, μ, μ)
#             +Gp(Δ_itr, χ, β, μ, μ)
#         )

#         return hbar2 * Gpp(Δ_itr, χ, β, α, μ) + left_bracket * right_bracket
#     end

#     @inline function kernel_RL(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         return (
#             σ_mem(σ_t, s_itr, t_itr, μ, χ)
#             * phase_exp(α, χ, Δ_itr)
#             * exp(exponent_RL(s_itr, Δ_itr, t_itr, α, β, χ, μ) / hbar2)
#             * coef_RL(s_itr, Δ_itr, t_itr, α, β, χ, μ) / hbar4
#         )
#     end

#     @inline function exponent_RR(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         # This is the g-part after distributing the leading minus sign in
#         # exp(-(i hbar Δε Δ + G_raw) / hbar^2).
#         return (
#             G(s_itr, α, α, α, α)
#             -G(t_itr, α, α, α, α)
#             -G(s_itr, χ, χ, α, α)
#             +G(t_itr, χ, χ, α, α)
#             +conj(G(Δ_itr, α, α, χ, χ))
#             -conj(G(s_itr, α, α, ν, ν))
#             +conj(G(t_itr, α, α, ν, ν))
#             -conj(G(Δ_itr, α, α, ν, ν))
#             -conj(G(Δ_itr, χ, χ, χ, χ))
#             +conj(G(s_itr, χ, χ, ν, ν))
#             -conj(G(t_itr, χ, χ, ν, ν))
#             +conj(G(Δ_itr, χ, χ, ν, ν))
#         )
#     end

#     @inline function coef_RR(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         right_bracket = (
#             Gp(s_itr, ν, χ, α, α)
#             +conj(Gp(Δ_itr, α, α, χ, ν))
#             -conj(Gp(Δ_itr, χ, χ, χ, ν))
#             -conj(Gp(s_itr, χ, ν, ν, ν))
#         )

#         left_bracket = (
#             conj(Gp(t_itr, β, χ, χ, χ))
#             -conj(Gp(Δ_itr, β, χ, χ, χ))
#             -conj(Gp(t_itr, β, χ, ν, ν))
#             +conj(Gp(Δ_itr, β, χ, ν, ν))
#         )

#         return hbar2 * conj(Gpp(Δ_itr, β, χ, χ, ν)) - right_bracket * left_bracket
#     end

#     @inline function kernel_RR(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         return -(
#             σ_mem(σ_t, s_itr, t_itr, α, ν)
#             * phase_exp(α, χ, Δ_itr)
#             * exp(exponent_RR(s_itr, Δ_itr, t_itr, α, β, χ, ν) / hbar2)
#             * coef_RR(s_itr, Δ_itr, t_itr, α, β, χ, ν) / hbar4
#         )
#     end


#     # ---------------------------------------------------------------------
#     # Explicit P L0 Q G0 Q L1 P, g-closed branch prefactors and coefficients.
#     # Branch L : input (μ, β) -> output (α, β)
#     # Branch R : input (α, ν) -> output (α, β)
#     # ---------------------------------------------------------------------

#     @inline function exponent_L0_L(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         μ::Int,
#     )
#         # exp[-(i hbar ω_{αβ} Δ + G_L) / hbar^2]
#         # = phase_exp(α, β, Δ) * exp[-G_L / hbar^2]
#         return (
#             -G(Δ_itr, α, α, α, α)
#             +G(s_itr, α, α, μ, μ)
#             -G(t_itr, α, α, μ, μ)
#             +G(Δ_itr, α, α, μ, μ)
#             +G(Δ_itr, β, β, α, α)
#             -G(s_itr, β, β, μ, μ)
#             +G(t_itr, β, β, μ, μ)
#             -G(Δ_itr, β, β, μ, μ)
#             -conj(G(s_itr, α, α, β, β))
#             +conj(G(t_itr, α, α, β, β))
#             +conj(G(s_itr, β, β, β, β))
#             -conj(G(t_itr, β, β, β, β))
#         )
#     end

#     @inline function coef_L0_L(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         μ::Int,
#     )
#         left_bracket = (
#             Gp(Δ_itr, α, α, α, μ)
#             +Gp(s_itr, α, μ, μ, μ)
#             -Gp(Δ_itr, β, β, α, μ)
#             -conj(Gp(s_itr, μ, α, β, β))
#         )

#         right_bracket = (
#             Gp(t_itr, α, α, α, α)
#             -Gp(Δ_itr, α, α, α, α)
#             -Gp(t_itr, α, α, μ, μ)
#             +Gp(Δ_itr, α, α, μ, μ)
#             -Gp(t_itr, β, β, α, α)
#             +Gp(Δ_itr, β, β, α, α)
#             +Gp(t_itr, β, β, μ, μ)
#             -Gp(Δ_itr, β, β, μ, μ)
#         )

#         return (
#             hbar2 * Gpp(Δ_itr, α, α, α, μ)
#             -hbar2 * Gpp(Δ_itr, β, β, α, μ)
#             +left_bracket * right_bracket
#         )
#     end

#     @inline function kernel_L0_L(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         μ::Int,
#     )
#         return -(
#             σ_mem(σ_t, s_itr, t_itr, μ, β)
#             * phase_exp(α, β, Δ_itr)
#             * exp(exponent_L0_L(s_itr, Δ_itr, t_itr, α, β, μ) / hbar2)
#             * coef_L0_L(s_itr, Δ_itr, t_itr, α, β, μ) / hbar4
#         )
#     end

#     @inline function exponent_L0_R(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         ν::Int,
#     )
#         # exp[-(i hbar ω_{αβ} Δ + G_R) / hbar^2]
#         # = phase_exp(α, β, Δ) * exp[-G_R / hbar^2]
#         return (
#             +G(s_itr, α, α, α, α)
#             -G(t_itr, α, α, α, α)
#             -G(s_itr, β, β, α, α)
#             +G(t_itr, β, β, α, α)
#             +conj(G(Δ_itr, α, α, β, β))
#             -conj(G(s_itr, α, α, ν, ν))
#             +conj(G(t_itr, α, α, ν, ν))
#             -conj(G(Δ_itr, α, α, ν, ν))
#             -conj(G(Δ_itr, β, β, β, β))
#             +conj(G(s_itr, β, β, ν, ν))
#             -conj(G(t_itr, β, β, ν, ν))
#             +conj(G(Δ_itr, β, β, ν, ν))
#         )
#     end

#     @inline function coef_L0_R(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         ν::Int,
#     )
#         left_bracket = (
#             Gp(s_itr, ν, β, α, α)
#             +conj(Gp(Δ_itr, α, α, β, ν))
#             -conj(Gp(Δ_itr, β, β, β, ν))
#             -conj(Gp(s_itr, β, ν, ν, ν))
#         )

#         right_bracket = (
#             conj(Gp(t_itr, α, α, β, β))
#             -conj(Gp(Δ_itr, α, α, β, β))
#             -conj(Gp(t_itr, α, α, ν, ν))
#             +conj(Gp(Δ_itr, α, α, ν, ν))
#             -conj(Gp(t_itr, β, β, β, β))
#             +conj(Gp(Δ_itr, β, β, β, β))
#             +conj(Gp(t_itr, β, β, ν, ν))
#             -conj(Gp(Δ_itr, β, β, ν, ν))
#         )

#         return (
#             -hbar2 * conj(Gpp(Δ_itr, α, α, β, ν))
#             +hbar2 * conj(Gpp(Δ_itr, β, β, β, ν))
#             +left_bracket * right_bracket
#         )
#     end

#     @inline function kernel_L0_R(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         ν::Int,
#     )
#         return -(
#             σ_mem(σ_t, s_itr, t_itr, α, ν)
#             * phase_exp(α, β, Δ_itr)
#             * exp(exponent_L0_R(s_itr, Δ_itr, t_itr, α, β, ν) / hbar2)
#             * coef_L0_R(s_itr, Δ_itr, t_itr, α, β, ν) / hbar4
#         )
#     end



#     # ---------------------------------------------------------------------
#     # Explicit P L0 Q G1 Q L1 P, g-closed 2-time branch kernels.
#     # The extra variable τ satisfies s <= τ <= t.  Each branch is the
#     # direct g-closed expression from I^{*,L0G1L1,g}_{αβ}(t,τ,s).
#     # ---------------------------------------------------------------------

#     @inline function kernel_L0G1_LL(
#         s_itr::Real,
#         τ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((hbar2 * Gpp(Δtτ_itr, α, α, α, χ) - hbar2 * Gpp(Δtτ_itr, β, β, α, χ) + (- Gp(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β))) * (- Gp(t_itr, α, α, α, α) + Gp(Δtτ_itr, α, α, α, α) + Gp(t_itr, α, α, χ, χ) - Gp(Δtτ_itr, α, α, χ, χ) + Gp(t_itr, β, β, α, α) - Gp(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp(Δtτ_itr, β, β, χ, χ))) * (Gp(Δτs_itr, β, β, χ, μ) - Gp(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) + G(t_itr, β, β, χ, χ) - G(τ_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(Δτs_itr, β, β, χ, χ) - G(t_itr, β, β, μ, μ) + G(τ_itr, β, β, μ, μ) - G(Δτs_itr, β, β, μ, μ) - G(Δτs_itr, χ, χ, χ, χ) + G(s_itr, χ, χ, μ, μ) - G(τ_itr, χ, χ, μ, μ) + G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(s_itr, χ, χ, β, β)) + conj(G(τ_itr, χ, χ, β, β)) + conj(G(s_itr, μ, μ, β, β)) - conj(G(t_itr, μ, μ, β, β))) / (hbar2))) + (hbar2 * (- Gpp(Δtτ_itr, α, α, α, χ) + Gpp(Δtτ_itr, β, β, α, χ)) * (- Gp(Δts_itr, α, α, χ, μ) + Gp(Δτs_itr, α, α, χ, μ) + Gp(Δts_itr, β, β, χ, μ) - Gp(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β))) + hbar2 * (- Gp(Δtτ_itr, β, β, α, α) - Gp(Δts_itr, β, β, χ, χ) + Gp(Δtτ_itr, β, β, χ, χ) - Gp(t_itr, β, β, μ, μ) + Gp(Δts_itr, β, β, μ, μ) + conj(Gp(t_itr, β, β, β, β))) * Gpp(Δτs_itr, α, χ, χ, μ) + (hbar2 * Gpp(Δτs_itr, α, χ, χ, μ) + (Gp(Δtτ_itr, α, α, α, χ) + Gp(Δτs_itr, α, χ, χ, χ) + Gp(τ_itr, α, χ, μ, μ) - Gp(Δτs_itr, α, χ, μ, μ) - Gp(Δtτ_itr, β, β, α, χ) - conj(Gp(τ_itr, χ, α, β, β))) * (- Gp(Δts_itr, α, α, χ, μ) + Gp(Δτs_itr, α, α, χ, μ) + Gp(Δts_itr, β, β, χ, μ) - Gp(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β)))) * (- Gp(t_itr, α, α, α, α) + Gp(Δtτ_itr, α, α, α, α) + Gp(Δts_itr, α, α, χ, χ) - Gp(Δtτ_itr, α, α, χ, χ) + Gp(t_itr, α, α, μ, μ) - Gp(Δts_itr, α, α, μ, μ) + Gp(t_itr, β, β, α, α) - conj(Gp(t_itr, β, β, β, β))) + (- hbar2 * Gpp(Δts_itr, α, α, χ, μ) + hbar2 * Gpp(Δts_itr, β, β, χ, μ) + (Gp(Δts_itr, α, α, χ, μ) - Gp(Δτs_itr, α, α, χ, μ) - Gp(Δts_itr, β, β, χ, μ) + Gp(Δτs_itr, χ, χ, χ, μ) + Gp(s_itr, χ, μ, μ, μ) - conj(Gp(s_itr, μ, χ, β, β))) * (- Gp(Δtτ_itr, β, β, α, α) - Gp(Δts_itr, β, β, χ, χ) + Gp(Δtτ_itr, β, β, χ, χ) - Gp(t_itr, β, β, μ, μ) + Gp(Δts_itr, β, β, μ, μ) + conj(Gp(t_itr, β, β, β, β)))) * (- Gp(Δtτ_itr, α, α, α, χ) - Gp(Δτs_itr, α, χ, χ, χ) - Gp(τ_itr, α, χ, μ, μ) + Gp(Δτs_itr, α, χ, μ, μ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(Δts_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δτs_itr, α, α, χ, χ) - G(t_itr, α, α, μ, μ) + G(τ_itr, α, α, μ, μ) + G(Δts_itr, α, α, μ, μ) - G(Δτs_itr, α, α, μ, μ) + G(Δtτ_itr, β, β, α, α) + G(Δts_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) - G(Δts_itr, β, β, μ, μ) - G(Δτs_itr, χ, χ, χ, χ) + G(s_itr, χ, χ, μ, μ) - G(τ_itr, χ, χ, μ, μ) + G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(s_itr, χ, χ, β, β)) + conj(G(τ_itr, χ, χ, β, β)) + conj(G(s_itr, μ, μ, β, β)) - conj(G(t_itr, μ, μ, β, β))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, μ, β) * exp(((- G(s_itr, β, β, μ, μ) + G(t_itr, β, β, μ, μ) + G(s_itr, μ, μ, μ, μ) - G(t_itr, μ, μ, μ, μ) + conj(G(s_itr, β, β, β, β)) - conj(G(t_itr, β, β, β, β)) - conj(G(s_itr, μ, μ, β, β)) + conj(G(t_itr, μ, μ, β, β))) / (hbar2)))
#         )
#     end

#     @inline function kernel_L0G1_LR(
#         s_itr::Real,
#         τ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((- hbar2 * Gpp(Δtτ_itr, α, α, α, χ) + hbar2 * Gpp(Δtτ_itr, β, β, α, χ) + (- Gp(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β))) * (Gp(t_itr, α, α, α, α) - Gp(Δtτ_itr, α, α, α, α) - Gp(t_itr, α, α, χ, χ) + Gp(Δtτ_itr, α, α, χ, χ) - Gp(t_itr, β, β, α, α) + Gp(Δtτ_itr, β, β, α, α) + Gp(t_itr, β, β, χ, χ) - Gp(Δtτ_itr, β, β, χ, χ))) * (- Gp(s_itr, ν, β, χ, χ) + conj(Gp(Δτs_itr, β, β, β, ν)) + conj(Gp(s_itr, β, ν, ν, ν)) - conj(Gp(Δτs_itr, χ, χ, β, ν))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) - G(s_itr, β, β, χ, χ) + G(t_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(t_itr, χ, χ, χ, χ) - G(τ_itr, χ, χ, χ, χ) + G(s_itr, ν, ν, χ, χ) - G(t_itr, ν, ν, χ, χ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(t_itr, β, β, β, β)) + conj(G(τ_itr, β, β, β, β)) - conj(G(Δτs_itr, β, β, β, β)) + conj(G(s_itr, β, β, ν, ν)) - conj(G(τ_itr, β, β, ν, ν)) + conj(G(Δτs_itr, β, β, ν, ν)) + conj(G(Δτs_itr, χ, χ, β, β)) - conj(G(t_itr, χ, χ, ν, ν)) + conj(G(τ_itr, χ, χ, ν, ν)) - conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2))) + (hbar2 * (conj(Gpp(Δts_itr, α, α, β, ν)) - conj(Gpp(Δts_itr, β, β, β, ν))) * (- Gp(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp(Δτs_itr, χ, α, ν, ν))) + hbar2 * (- Gp(Δtτ_itr, α, α, α, α) - Gp(t_itr, α, α, χ, χ) + Gp(Δtτ_itr, α, α, χ, χ) + conj(Gp(Δts_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, ν, ν)) - conj(Gp(Δts_itr, α, α, ν, ν))) * conj(Gpp(Δτs_itr, χ, α, β, ν)) - hbar2 * (- Gp(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp(Δtτ_itr, β, β, χ, χ) + conj(Gp(Δts_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp(Δts_itr, β, β, ν, ν))) * conj(Gpp(Δτs_itr, χ, α, β, ν)) + (hbar2 * conj(Gpp(Δτs_itr, χ, α, β, ν)) + (- Gp(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp(Δτs_itr, χ, α, ν, ν))) * (Gp(s_itr, ν, β, χ, χ) + conj(Gp(Δts_itr, α, α, β, ν)) - conj(Gp(Δτs_itr, α, α, β, ν)) - conj(Gp(Δts_itr, β, β, β, ν)) - conj(Gp(s_itr, β, ν, ν, ν)) + conj(Gp(Δτs_itr, χ, χ, β, ν)))) * (Gp(t_itr, α, α, α, α) - Gp(t_itr, β, β, α, α) - conj(Gp(t_itr, α, α, β, β)) + conj(Gp(t_itr, β, β, β, β))) + (hbar2 * Gpp(Δtτ_itr, α, α, α, χ) - hbar2 * Gpp(Δtτ_itr, β, β, α, χ) + (- Gp(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp(Δτs_itr, χ, α, ν, ν))) * (Gp(Δtτ_itr, α, α, α, α) + Gp(t_itr, α, α, χ, χ) - Gp(Δtτ_itr, α, α, χ, χ) - Gp(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp(Δtτ_itr, β, β, χ, χ) - conj(Gp(Δts_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, ν, ν)) + conj(Gp(Δts_itr, α, α, ν, ν)) + conj(Gp(Δts_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp(Δts_itr, β, β, ν, ν)))) * (- Gp(s_itr, ν, β, χ, χ) - conj(Gp(Δts_itr, α, α, β, ν)) + conj(Gp(Δτs_itr, α, α, β, ν)) + conj(Gp(Δts_itr, β, β, β, ν)) + conj(Gp(s_itr, β, ν, ν, ν)) - conj(Gp(Δτs_itr, χ, χ, β, ν)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) - G(s_itr, β, β, χ, χ) + G(t_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(t_itr, χ, χ, χ, χ) - G(τ_itr, χ, χ, χ, χ) + G(s_itr, ν, ν, χ, χ) - G(t_itr, ν, ν, χ, χ) + conj(G(Δts_itr, α, α, β, β)) - conj(G(Δτs_itr, α, α, β, β)) + conj(G(t_itr, α, α, ν, ν)) - conj(G(τ_itr, α, α, ν, ν)) - conj(G(Δts_itr, α, α, ν, ν)) + conj(G(Δτs_itr, α, α, ν, ν)) - conj(G(Δts_itr, β, β, β, β)) + conj(G(s_itr, β, β, ν, ν)) - conj(G(t_itr, β, β, ν, ν)) + conj(G(Δts_itr, β, β, ν, ν)) + conj(G(Δτs_itr, χ, χ, β, β)) - conj(G(t_itr, χ, χ, ν, ν)) + conj(G(τ_itr, χ, χ, ν, ν)) - conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, χ, ν) * exp(((G(s_itr, χ, χ, χ, χ) - G(t_itr, χ, χ, χ, χ) - G(s_itr, ν, ν, χ, χ) + G(t_itr, ν, ν, χ, χ) - conj(G(s_itr, χ, χ, ν, ν)) + conj(G(t_itr, χ, χ, ν, ν)) + conj(G(s_itr, ν, ν, ν, ν)) - conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))
#         )
#     end

#     @inline function kernel_L0G1_RL(
#         s_itr::Real,
#         τ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((- hbar2 * conj(Gpp(Δtτ_itr, α, α, β, χ)) + hbar2 * conj(Gpp(Δtτ_itr, β, β, β, χ)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp(Δtτ_itr, α, α, β, χ)) + conj(Gp(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ))) * (- conj(Gp(t_itr, α, α, β, β)) + conj(Gp(Δtτ_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, χ, χ)) - conj(Gp(Δtτ_itr, α, α, χ, χ)) + conj(Gp(t_itr, β, β, β, β)) - conj(Gp(Δtτ_itr, β, β, β, β)) - conj(Gp(t_itr, β, β, χ, χ)) + conj(Gp(Δtτ_itr, β, β, χ, χ)))) * (- Gp(Δτs_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(t_itr, α, α, α, α) + G(τ_itr, α, α, α, α) - G(Δτs_itr, α, α, α, α) + G(s_itr, α, α, μ, μ) - G(τ_itr, α, α, μ, μ) + G(Δτs_itr, α, α, μ, μ) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) + G(Δτs_itr, χ, χ, α, α) - G(t_itr, χ, χ, μ, μ) + G(τ_itr, χ, χ, μ, μ) - G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(Δtτ_itr, α, α, β, β)) - conj(G(s_itr, α, α, χ, χ)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(t_itr, χ, χ, χ, χ)) - conj(G(τ_itr, χ, χ, χ, χ)) + conj(G(s_itr, μ, μ, χ, χ)) - conj(G(t_itr, μ, μ, χ, χ))) / (hbar2))) + (hbar2 * (conj(Gpp(Δtτ_itr, α, α, β, χ)) - conj(Gpp(Δtτ_itr, β, β, β, χ))) * (- Gp(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp(Δts_itr, β, β, α, μ) - Gp(Δτs_itr, β, β, α, μ) + Gp(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) + hbar2 * (- Gp(Δts_itr, α, α, α, α) - Gp(t_itr, α, α, μ, μ) + Gp(Δts_itr, α, α, μ, μ) + conj(Gp(Δtτ_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, χ, χ)) - conj(Gp(Δtτ_itr, α, α, χ, χ))) * Gpp(Δτs_itr, χ, β, α, μ) + (hbar2 * Gpp(Δτs_itr, χ, β, α, μ) + (- Gp(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp(Δts_itr, β, β, α, μ) - Gp(Δτs_itr, β, β, α, μ) + Gp(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) * (Gp(Δτs_itr, χ, β, α, α) + Gp(τ_itr, χ, β, μ, μ) - Gp(Δτs_itr, χ, β, μ, μ) + conj(Gp(Δtτ_itr, α, α, β, χ)) - conj(Gp(Δtτ_itr, β, β, β, χ)) - conj(Gp(τ_itr, β, χ, χ, χ)))) * (Gp(t_itr, α, α, α, α) - Gp(t_itr, β, β, α, α) + Gp(Δts_itr, β, β, α, α) + Gp(t_itr, β, β, μ, μ) - Gp(Δts_itr, β, β, μ, μ) - conj(Gp(t_itr, α, α, β, β)) + conj(Gp(t_itr, β, β, β, β)) - conj(Gp(Δtτ_itr, β, β, β, β)) - conj(Gp(t_itr, β, β, χ, χ)) + conj(Gp(Δtτ_itr, β, β, χ, χ))) + (hbar2 * Gpp(Δts_itr, α, α, α, μ) - hbar2 * Gpp(Δts_itr, β, β, α, μ) + (Gp(Δts_itr, α, α, α, α) + Gp(t_itr, α, α, μ, μ) - Gp(Δts_itr, α, α, μ, μ) - conj(Gp(Δtτ_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, χ, χ)) + conj(Gp(Δtτ_itr, α, α, χ, χ))) * (- Gp(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp(Δts_itr, β, β, α, μ) - Gp(Δτs_itr, β, β, α, μ) + Gp(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ)))) * (- Gp(Δτs_itr, χ, β, α, α) - Gp(τ_itr, χ, β, μ, μ) + Gp(Δτs_itr, χ, β, μ, μ) - conj(Gp(Δtτ_itr, α, α, β, χ)) + conj(Gp(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δts_itr, α, α, α, α) + G(s_itr, α, α, μ, μ) - G(t_itr, α, α, μ, μ) + G(Δts_itr, α, α, μ, μ) + G(Δts_itr, β, β, α, α) - G(Δτs_itr, β, β, α, α) + G(t_itr, β, β, μ, μ) - G(τ_itr, β, β, μ, μ) - G(Δts_itr, β, β, μ, μ) + G(Δτs_itr, β, β, μ, μ) + G(Δτs_itr, χ, χ, α, α) - G(t_itr, χ, χ, μ, μ) + G(τ_itr, χ, χ, μ, μ) - G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(Δtτ_itr, α, α, β, β)) - conj(G(s_itr, α, α, χ, χ)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(t_itr, χ, χ, χ, χ)) - conj(G(τ_itr, χ, χ, χ, χ)) + conj(G(s_itr, μ, μ, χ, χ)) - conj(G(t_itr, μ, μ, χ, χ))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, μ, χ) * exp(((- G(s_itr, χ, χ, μ, μ) + G(t_itr, χ, χ, μ, μ) + G(s_itr, μ, μ, μ, μ) - G(t_itr, μ, μ, μ, μ) + conj(G(s_itr, χ, χ, χ, χ)) - conj(G(t_itr, χ, χ, χ, χ)) - conj(G(s_itr, μ, μ, χ, χ)) + conj(G(t_itr, μ, μ, χ, χ))) / (hbar2)))
#         )
#     end

#     @inline function kernel_L0G1_RR(
#         s_itr::Real,
#         τ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((hbar2 * conj(Gpp(Δtτ_itr, α, α, β, χ)) - hbar2 * conj(Gpp(Δtτ_itr, β, β, β, χ)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp(Δtτ_itr, α, α, β, χ)) + conj(Gp(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ))) * (conj(Gp(t_itr, α, α, β, β)) - conj(Gp(Δtτ_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, χ, χ)) + conj(Gp(Δtτ_itr, α, α, χ, χ)) - conj(Gp(t_itr, β, β, β, β)) + conj(Gp(Δtτ_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, χ, χ)) - conj(Gp(Δtτ_itr, β, β, χ, χ)))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp(Δτs_itr, α, α, χ, ν)) + conj(Gp(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) - G(s_itr, χ, χ, α, α) + G(τ_itr, χ, χ, α, α) + G(s_itr, ν, ν, α, α) - G(t_itr, ν, ν, α, α) + conj(G(Δtτ_itr, α, α, β, β)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(τ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) + conj(G(Δτs_itr, α, α, χ, χ)) - conj(G(t_itr, α, α, ν, ν)) + conj(G(τ_itr, α, α, ν, ν)) - conj(G(Δτs_itr, α, α, ν, ν)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) - conj(G(Δτs_itr, χ, χ, χ, χ)) + conj(G(s_itr, χ, χ, ν, ν)) - conj(G(τ_itr, χ, χ, ν, ν)) + conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2))) + (hbar2 * (- conj(Gpp(Δtτ_itr, α, α, β, χ)) + conj(Gpp(Δtτ_itr, β, β, β, χ))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp(Δts_itr, α, α, χ, ν)) + conj(Gp(Δts_itr, β, β, χ, ν)) - conj(Gp(Δτs_itr, β, β, χ, ν)) + conj(Gp(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν))) + hbar2 * (- Gp(t_itr, β, β, α, α) + conj(Gp(Δtτ_itr, β, β, β, β)) + conj(Gp(Δts_itr, β, β, χ, χ)) - conj(Gp(Δtτ_itr, β, β, χ, χ)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp(Δts_itr, β, β, ν, ν))) * conj(Gpp(Δτs_itr, β, χ, χ, ν)) + (hbar2 * conj(Gpp(Δτs_itr, β, χ, χ, ν)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp(Δtτ_itr, α, α, β, χ)) + conj(Gp(Δtτ_itr, β, β, β, χ)) + conj(Gp(Δτs_itr, β, χ, χ, χ)) + conj(Gp(τ_itr, β, χ, ν, ν)) - conj(Gp(Δτs_itr, β, χ, ν, ν))) * (Gp(s_itr, ν, χ, α, α) + conj(Gp(Δts_itr, α, α, χ, ν)) - conj(Gp(Δts_itr, β, β, χ, ν)) + conj(Gp(Δτs_itr, β, β, χ, ν)) - conj(Gp(Δτs_itr, χ, χ, χ, ν)) - conj(Gp(s_itr, χ, ν, ν, ν)))) * (Gp(t_itr, β, β, α, α) + conj(Gp(t_itr, α, α, β, β)) - conj(Gp(Δtτ_itr, α, α, β, β)) - conj(Gp(Δts_itr, α, α, χ, χ)) + conj(Gp(Δtτ_itr, α, α, χ, χ)) - conj(Gp(t_itr, α, α, ν, ν)) + conj(Gp(Δts_itr, α, α, ν, ν)) - conj(Gp(t_itr, β, β, β, β))) + (- hbar2 * conj(Gpp(Δts_itr, α, α, χ, ν)) + hbar2 * conj(Gpp(Δts_itr, β, β, χ, ν)) + (Gp(t_itr, β, β, α, α) - conj(Gp(Δtτ_itr, β, β, β, β)) - conj(Gp(Δts_itr, β, β, χ, χ)) + conj(Gp(Δtτ_itr, β, β, χ, χ)) - conj(Gp(t_itr, β, β, ν, ν)) + conj(Gp(Δts_itr, β, β, ν, ν))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp(Δts_itr, α, α, χ, ν)) + conj(Gp(Δts_itr, β, β, χ, ν)) - conj(Gp(Δτs_itr, β, β, χ, ν)) + conj(Gp(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν)))) * (- Gp(τ_itr, χ, β, α, α) - conj(Gp(Δtτ_itr, α, α, β, χ)) + conj(Gp(Δtτ_itr, β, β, β, χ)) + conj(Gp(Δτs_itr, β, χ, χ, χ)) + conj(Gp(τ_itr, β, χ, ν, ν)) - conj(Gp(Δτs_itr, β, χ, ν, ν)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) - G(s_itr, χ, χ, α, α) + G(τ_itr, χ, χ, α, α) + G(s_itr, ν, ν, α, α) - G(t_itr, ν, ν, α, α) + conj(G(Δtτ_itr, α, α, β, β)) + conj(G(Δts_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δts_itr, α, α, ν, ν)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(Δts_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(Δτs_itr, β, β, χ, χ)) - conj(G(t_itr, β, β, ν, ν)) + conj(G(τ_itr, β, β, ν, ν)) + conj(G(Δts_itr, β, β, ν, ν)) - conj(G(Δτs_itr, β, β, ν, ν)) - conj(G(Δτs_itr, χ, χ, χ, χ)) + conj(G(s_itr, χ, χ, ν, ν)) - conj(G(τ_itr, χ, χ, ν, ν)) + conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, α, ν) * exp(((G(s_itr, α, α, α, α) - G(t_itr, α, α, α, α) - G(s_itr, ν, ν, α, α) + G(t_itr, ν, ν, α, α) - conj(G(s_itr, α, α, ν, ν)) + conj(G(t_itr, α, α, ν, ν)) + conj(G(s_itr, ν, ν, ν, ν)) - conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))
#         )
#     end


#     # ---------------------------------------------------------------------
#     # Markovianized inner-G1 variants of P L0 Q G1 Q L1 P.
#     #
#     # These are mechanically identical to the exact 2-time kernels above,
#     # except that relative-time derivative factors with arguments
#     # Δts_itr, Δtτ_itr, and Δτs_itr use
#     #
#     #     Gp  -> Gp_markovian_G1  = -im * Λ
#     #     Gpp -> Gpp_markovian_G1 = 0
#     #
#     # Absolute-time g/g′ factors and all g-exponential dressing factors are
#     # intentionally left unchanged.  Therefore this switch only Markovianizes
#     # the internal G1 derivative/correlation content, not the full outer
#     # P L0 Q ... Q L1 P memory kernel.
#     # ---------------------------------------------------------------------

#     @inline function kernel_L0G1_LL_markovian_G1(
#         s_itr::Real,
#         τ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((hbar2 * Gpp_markovian_G1(Δtτ_itr, α, α, α, χ) - hbar2 * Gpp_markovian_G1(Δtτ_itr, β, β, α, χ) + (- Gp_markovian_G1(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β))) * (- Gp(t_itr, α, α, α, α) + Gp_markovian_G1(Δtτ_itr, α, α, α, α) + Gp(t_itr, α, α, χ, χ) - Gp_markovian_G1(Δtτ_itr, α, α, χ, χ) + Gp(t_itr, β, β, α, α) - Gp_markovian_G1(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, χ, χ))) * (Gp_markovian_G1(Δτs_itr, β, β, χ, μ) - Gp_markovian_G1(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) + G(t_itr, β, β, χ, χ) - G(τ_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(Δτs_itr, β, β, χ, χ) - G(t_itr, β, β, μ, μ) + G(τ_itr, β, β, μ, μ) - G(Δτs_itr, β, β, μ, μ) - G(Δτs_itr, χ, χ, χ, χ) + G(s_itr, χ, χ, μ, μ) - G(τ_itr, χ, χ, μ, μ) + G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(s_itr, χ, χ, β, β)) + conj(G(τ_itr, χ, χ, β, β)) + conj(G(s_itr, μ, μ, β, β)) - conj(G(t_itr, μ, μ, β, β))) / (hbar2))) + (hbar2 * (- Gpp_markovian_G1(Δtτ_itr, α, α, α, χ) + Gpp_markovian_G1(Δtτ_itr, β, β, α, χ)) * (- Gp_markovian_G1(Δts_itr, α, α, χ, μ) + Gp_markovian_G1(Δτs_itr, α, α, χ, μ) + Gp_markovian_G1(Δts_itr, β, β, χ, μ) - Gp_markovian_G1(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β))) + hbar2 * (- Gp_markovian_G1(Δtτ_itr, β, β, α, α) - Gp_markovian_G1(Δts_itr, β, β, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, χ, χ) - Gp(t_itr, β, β, μ, μ) + Gp_markovian_G1(Δts_itr, β, β, μ, μ) + conj(Gp(t_itr, β, β, β, β))) * Gpp_markovian_G1(Δτs_itr, α, χ, χ, μ) + (hbar2 * Gpp_markovian_G1(Δτs_itr, α, χ, χ, μ) + (Gp_markovian_G1(Δtτ_itr, α, α, α, χ) + Gp_markovian_G1(Δτs_itr, α, χ, χ, χ) + Gp(τ_itr, α, χ, μ, μ) - Gp_markovian_G1(Δτs_itr, α, χ, μ, μ) - Gp_markovian_G1(Δtτ_itr, β, β, α, χ) - conj(Gp(τ_itr, χ, α, β, β))) * (- Gp_markovian_G1(Δts_itr, α, α, χ, μ) + Gp_markovian_G1(Δτs_itr, α, α, χ, μ) + Gp_markovian_G1(Δts_itr, β, β, χ, μ) - Gp_markovian_G1(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β)))) * (- Gp(t_itr, α, α, α, α) + Gp_markovian_G1(Δtτ_itr, α, α, α, α) + Gp_markovian_G1(Δts_itr, α, α, χ, χ) - Gp_markovian_G1(Δtτ_itr, α, α, χ, χ) + Gp(t_itr, α, α, μ, μ) - Gp_markovian_G1(Δts_itr, α, α, μ, μ) + Gp(t_itr, β, β, α, α) - conj(Gp(t_itr, β, β, β, β))) + (- hbar2 * Gpp_markovian_G1(Δts_itr, α, α, χ, μ) + hbar2 * Gpp_markovian_G1(Δts_itr, β, β, χ, μ) + (Gp_markovian_G1(Δts_itr, α, α, χ, μ) - Gp_markovian_G1(Δτs_itr, α, α, χ, μ) - Gp_markovian_G1(Δts_itr, β, β, χ, μ) + Gp_markovian_G1(Δτs_itr, χ, χ, χ, μ) + Gp(s_itr, χ, μ, μ, μ) - conj(Gp(s_itr, μ, χ, β, β))) * (- Gp_markovian_G1(Δtτ_itr, β, β, α, α) - Gp_markovian_G1(Δts_itr, β, β, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, χ, χ) - Gp(t_itr, β, β, μ, μ) + Gp_markovian_G1(Δts_itr, β, β, μ, μ) + conj(Gp(t_itr, β, β, β, β)))) * (- Gp_markovian_G1(Δtτ_itr, α, α, α, χ) - Gp_markovian_G1(Δτs_itr, α, χ, χ, χ) - Gp(τ_itr, α, χ, μ, μ) + Gp_markovian_G1(Δτs_itr, α, χ, μ, μ) + Gp_markovian_G1(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(Δts_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δτs_itr, α, α, χ, χ) - G(t_itr, α, α, μ, μ) + G(τ_itr, α, α, μ, μ) + G(Δts_itr, α, α, μ, μ) - G(Δτs_itr, α, α, μ, μ) + G(Δtτ_itr, β, β, α, α) + G(Δts_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) - G(Δts_itr, β, β, μ, μ) - G(Δτs_itr, χ, χ, χ, χ) + G(s_itr, χ, χ, μ, μ) - G(τ_itr, χ, χ, μ, μ) + G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(s_itr, χ, χ, β, β)) + conj(G(τ_itr, χ, χ, β, β)) + conj(G(s_itr, μ, μ, β, β)) - conj(G(t_itr, μ, μ, β, β))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, μ, β) * exp(((- G(s_itr, β, β, μ, μ) + G(t_itr, β, β, μ, μ) + G(s_itr, μ, μ, μ, μ) - G(t_itr, μ, μ, μ, μ) + conj(G(s_itr, β, β, β, β)) - conj(G(t_itr, β, β, β, β)) - conj(G(s_itr, μ, μ, β, β)) + conj(G(t_itr, μ, μ, β, β))) / (hbar2)))
#         )
#     end


#     @inline function kernel_L0G1_LR_markovian_G1(
#         s_itr::Real,
#         τ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((- hbar2 * Gpp_markovian_G1(Δtτ_itr, α, α, α, χ) + hbar2 * Gpp_markovian_G1(Δtτ_itr, β, β, α, χ) + (- Gp_markovian_G1(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β))) * (Gp(t_itr, α, α, α, α) - Gp_markovian_G1(Δtτ_itr, α, α, α, α) - Gp(t_itr, α, α, χ, χ) + Gp_markovian_G1(Δtτ_itr, α, α, χ, χ) - Gp(t_itr, β, β, α, α) + Gp_markovian_G1(Δtτ_itr, β, β, α, α) + Gp(t_itr, β, β, χ, χ) - Gp_markovian_G1(Δtτ_itr, β, β, χ, χ))) * (- Gp(s_itr, ν, β, χ, χ) + conj(Gp_markovian_G1(Δτs_itr, β, β, β, ν)) + conj(Gp(s_itr, β, ν, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, χ, χ, β, ν))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) - G(s_itr, β, β, χ, χ) + G(t_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(t_itr, χ, χ, χ, χ) - G(τ_itr, χ, χ, χ, χ) + G(s_itr, ν, ν, χ, χ) - G(t_itr, ν, ν, χ, χ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(t_itr, β, β, β, β)) + conj(G(τ_itr, β, β, β, β)) - conj(G(Δτs_itr, β, β, β, β)) + conj(G(s_itr, β, β, ν, ν)) - conj(G(τ_itr, β, β, ν, ν)) + conj(G(Δτs_itr, β, β, ν, ν)) + conj(G(Δτs_itr, χ, χ, β, β)) - conj(G(t_itr, χ, χ, ν, ν)) + conj(G(τ_itr, χ, χ, ν, ν)) - conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2))) + (hbar2 * (conj(Gpp_markovian_G1(Δts_itr, α, α, β, ν)) - conj(Gpp_markovian_G1(Δts_itr, β, β, β, ν))) * (- Gp_markovian_G1(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, α, χ) + conj(Gp_markovian_G1(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, χ, α, ν, ν))) + hbar2 * (- Gp_markovian_G1(Δtτ_itr, α, α, α, α) - Gp(t_itr, α, α, χ, χ) + Gp_markovian_G1(Δtτ_itr, α, α, χ, χ) + conj(Gp_markovian_G1(Δts_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, ν, ν)) - conj(Gp_markovian_G1(Δts_itr, α, α, ν, ν))) * conj(Gpp_markovian_G1(Δτs_itr, χ, α, β, ν)) - hbar2 * (- Gp_markovian_G1(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, χ, χ) + conj(Gp_markovian_G1(Δts_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp_markovian_G1(Δts_itr, β, β, ν, ν))) * conj(Gpp_markovian_G1(Δτs_itr, χ, α, β, ν)) + (hbar2 * conj(Gpp_markovian_G1(Δτs_itr, χ, α, β, ν)) + (- Gp_markovian_G1(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, α, χ) + conj(Gp_markovian_G1(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, χ, α, ν, ν))) * (Gp(s_itr, ν, β, χ, χ) + conj(Gp_markovian_G1(Δts_itr, α, α, β, ν)) - conj(Gp_markovian_G1(Δτs_itr, α, α, β, ν)) - conj(Gp_markovian_G1(Δts_itr, β, β, β, ν)) - conj(Gp(s_itr, β, ν, ν, ν)) + conj(Gp_markovian_G1(Δτs_itr, χ, χ, β, ν)))) * (Gp(t_itr, α, α, α, α) - Gp(t_itr, β, β, α, α) - conj(Gp(t_itr, α, α, β, β)) + conj(Gp(t_itr, β, β, β, β))) + (hbar2 * Gpp_markovian_G1(Δtτ_itr, α, α, α, χ) - hbar2 * Gpp_markovian_G1(Δtτ_itr, β, β, α, χ) + (- Gp_markovian_G1(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, α, χ) + conj(Gp_markovian_G1(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, χ, α, ν, ν))) * (Gp_markovian_G1(Δtτ_itr, α, α, α, α) + Gp(t_itr, α, α, χ, χ) - Gp_markovian_G1(Δtτ_itr, α, α, χ, χ) - Gp_markovian_G1(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, χ, χ) - conj(Gp_markovian_G1(Δts_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, ν, ν)) + conj(Gp_markovian_G1(Δts_itr, α, α, ν, ν)) + conj(Gp_markovian_G1(Δts_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp_markovian_G1(Δts_itr, β, β, ν, ν)))) * (- Gp(s_itr, ν, β, χ, χ) - conj(Gp_markovian_G1(Δts_itr, α, α, β, ν)) + conj(Gp_markovian_G1(Δτs_itr, α, α, β, ν)) + conj(Gp_markovian_G1(Δts_itr, β, β, β, ν)) + conj(Gp(s_itr, β, ν, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, χ, χ, β, ν)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) - G(s_itr, β, β, χ, χ) + G(t_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(t_itr, χ, χ, χ, χ) - G(τ_itr, χ, χ, χ, χ) + G(s_itr, ν, ν, χ, χ) - G(t_itr, ν, ν, χ, χ) + conj(G(Δts_itr, α, α, β, β)) - conj(G(Δτs_itr, α, α, β, β)) + conj(G(t_itr, α, α, ν, ν)) - conj(G(τ_itr, α, α, ν, ν)) - conj(G(Δts_itr, α, α, ν, ν)) + conj(G(Δτs_itr, α, α, ν, ν)) - conj(G(Δts_itr, β, β, β, β)) + conj(G(s_itr, β, β, ν, ν)) - conj(G(t_itr, β, β, ν, ν)) + conj(G(Δts_itr, β, β, ν, ν)) + conj(G(Δτs_itr, χ, χ, β, β)) - conj(G(t_itr, χ, χ, ν, ν)) + conj(G(τ_itr, χ, χ, ν, ν)) - conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, χ, ν) * exp(((G(s_itr, χ, χ, χ, χ) - G(t_itr, χ, χ, χ, χ) - G(s_itr, ν, ν, χ, χ) + G(t_itr, ν, ν, χ, χ) - conj(G(s_itr, χ, χ, ν, ν)) + conj(G(t_itr, χ, χ, ν, ν)) + conj(G(s_itr, ν, ν, ν, ν)) - conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))
#         )
#     end


#     @inline function kernel_L0G1_RL_markovian_G1(
#         s_itr::Real,
#         τ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((- hbar2 * conj(Gpp_markovian_G1(Δtτ_itr, α, α, β, χ)) + hbar2 * conj(Gpp_markovian_G1(Δtτ_itr, β, β, β, χ)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ))) * (- conj(Gp(t_itr, α, α, β, β)) + conj(Gp_markovian_G1(Δtτ_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, χ, χ)) - conj(Gp_markovian_G1(Δtτ_itr, α, α, χ, χ)) + conj(Gp(t_itr, β, β, β, β)) - conj(Gp_markovian_G1(Δtτ_itr, β, β, β, β)) - conj(Gp(t_itr, β, β, χ, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, χ, χ)))) * (- Gp_markovian_G1(Δτs_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp_markovian_G1(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(t_itr, α, α, α, α) + G(τ_itr, α, α, α, α) - G(Δτs_itr, α, α, α, α) + G(s_itr, α, α, μ, μ) - G(τ_itr, α, α, μ, μ) + G(Δτs_itr, α, α, μ, μ) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) + G(Δτs_itr, χ, χ, α, α) - G(t_itr, χ, χ, μ, μ) + G(τ_itr, χ, χ, μ, μ) - G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(Δtτ_itr, α, α, β, β)) - conj(G(s_itr, α, α, χ, χ)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(t_itr, χ, χ, χ, χ)) - conj(G(τ_itr, χ, χ, χ, χ)) + conj(G(s_itr, μ, μ, χ, χ)) - conj(G(t_itr, μ, μ, χ, χ))) / (hbar2))) + (hbar2 * (conj(Gpp_markovian_G1(Δtτ_itr, α, α, β, χ)) - conj(Gpp_markovian_G1(Δtτ_itr, β, β, β, χ))) * (- Gp_markovian_G1(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp_markovian_G1(Δts_itr, β, β, α, μ) - Gp_markovian_G1(Δτs_itr, β, β, α, μ) + Gp_markovian_G1(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) + hbar2 * (- Gp_markovian_G1(Δts_itr, α, α, α, α) - Gp(t_itr, α, α, μ, μ) + Gp_markovian_G1(Δts_itr, α, α, μ, μ) + conj(Gp_markovian_G1(Δtτ_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, χ, χ)) - conj(Gp_markovian_G1(Δtτ_itr, α, α, χ, χ))) * Gpp_markovian_G1(Δτs_itr, χ, β, α, μ) + (hbar2 * Gpp_markovian_G1(Δτs_itr, χ, β, α, μ) + (- Gp_markovian_G1(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp_markovian_G1(Δts_itr, β, β, α, μ) - Gp_markovian_G1(Δτs_itr, β, β, α, μ) + Gp_markovian_G1(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) * (Gp_markovian_G1(Δτs_itr, χ, β, α, α) + Gp(τ_itr, χ, β, μ, μ) - Gp_markovian_G1(Δτs_itr, χ, β, μ, μ) + conj(Gp_markovian_G1(Δtτ_itr, α, α, β, χ)) - conj(Gp_markovian_G1(Δtτ_itr, β, β, β, χ)) - conj(Gp(τ_itr, β, χ, χ, χ)))) * (Gp(t_itr, α, α, α, α) - Gp(t_itr, β, β, α, α) + Gp_markovian_G1(Δts_itr, β, β, α, α) + Gp(t_itr, β, β, μ, μ) - Gp_markovian_G1(Δts_itr, β, β, μ, μ) - conj(Gp(t_itr, α, α, β, β)) + conj(Gp(t_itr, β, β, β, β)) - conj(Gp_markovian_G1(Δtτ_itr, β, β, β, β)) - conj(Gp(t_itr, β, β, χ, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, χ, χ))) + (hbar2 * Gpp_markovian_G1(Δts_itr, α, α, α, μ) - hbar2 * Gpp_markovian_G1(Δts_itr, β, β, α, μ) + (Gp_markovian_G1(Δts_itr, α, α, α, α) + Gp(t_itr, α, α, μ, μ) - Gp_markovian_G1(Δts_itr, α, α, μ, μ) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, χ, χ)) + conj(Gp_markovian_G1(Δtτ_itr, α, α, χ, χ))) * (- Gp_markovian_G1(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp_markovian_G1(Δts_itr, β, β, α, μ) - Gp_markovian_G1(Δτs_itr, β, β, α, μ) + Gp_markovian_G1(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ)))) * (- Gp_markovian_G1(Δτs_itr, χ, β, α, α) - Gp(τ_itr, χ, β, μ, μ) + Gp_markovian_G1(Δτs_itr, χ, β, μ, μ) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δts_itr, α, α, α, α) + G(s_itr, α, α, μ, μ) - G(t_itr, α, α, μ, μ) + G(Δts_itr, α, α, μ, μ) + G(Δts_itr, β, β, α, α) - G(Δτs_itr, β, β, α, α) + G(t_itr, β, β, μ, μ) - G(τ_itr, β, β, μ, μ) - G(Δts_itr, β, β, μ, μ) + G(Δτs_itr, β, β, μ, μ) + G(Δτs_itr, χ, χ, α, α) - G(t_itr, χ, χ, μ, μ) + G(τ_itr, χ, χ, μ, μ) - G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(Δtτ_itr, α, α, β, β)) - conj(G(s_itr, α, α, χ, χ)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(t_itr, χ, χ, χ, χ)) - conj(G(τ_itr, χ, χ, χ, χ)) + conj(G(s_itr, μ, μ, χ, χ)) - conj(G(t_itr, μ, μ, χ, χ))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, μ, χ) * exp(((- G(s_itr, χ, χ, μ, μ) + G(t_itr, χ, χ, μ, μ) + G(s_itr, μ, μ, μ, μ) - G(t_itr, μ, μ, μ, μ) + conj(G(s_itr, χ, χ, χ, χ)) - conj(G(t_itr, χ, χ, χ, χ)) - conj(G(s_itr, μ, μ, χ, χ)) + conj(G(t_itr, μ, μ, χ, χ))) / (hbar2)))
#         )
#     end


#     @inline function kernel_L0G1_RR_markovian_G1(
#         s_itr::Real,
#         τ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((hbar2 * conj(Gpp_markovian_G1(Δtτ_itr, α, α, β, χ)) - hbar2 * conj(Gpp_markovian_G1(Δtτ_itr, β, β, β, χ)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ))) * (conj(Gp(t_itr, α, α, β, β)) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, χ, χ)) + conj(Gp_markovian_G1(Δtτ_itr, α, α, χ, χ)) - conj(Gp(t_itr, β, β, β, β)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, χ, χ)) - conj(Gp_markovian_G1(Δtτ_itr, β, β, χ, χ)))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp_markovian_G1(Δτs_itr, α, α, χ, ν)) + conj(Gp_markovian_G1(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) - G(s_itr, χ, χ, α, α) + G(τ_itr, χ, χ, α, α) + G(s_itr, ν, ν, α, α) - G(t_itr, ν, ν, α, α) + conj(G(Δtτ_itr, α, α, β, β)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(τ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) + conj(G(Δτs_itr, α, α, χ, χ)) - conj(G(t_itr, α, α, ν, ν)) + conj(G(τ_itr, α, α, ν, ν)) - conj(G(Δτs_itr, α, α, ν, ν)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) - conj(G(Δτs_itr, χ, χ, χ, χ)) + conj(G(s_itr, χ, χ, ν, ν)) - conj(G(τ_itr, χ, χ, ν, ν)) + conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2))) + (hbar2 * (- conj(Gpp_markovian_G1(Δtτ_itr, α, α, β, χ)) + conj(Gpp_markovian_G1(Δtτ_itr, β, β, β, χ))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp_markovian_G1(Δts_itr, α, α, χ, ν)) + conj(Gp_markovian_G1(Δts_itr, β, β, χ, ν)) - conj(Gp_markovian_G1(Δτs_itr, β, β, χ, ν)) + conj(Gp_markovian_G1(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν))) + hbar2 * (- Gp(t_itr, β, β, α, α) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, β)) + conj(Gp_markovian_G1(Δts_itr, β, β, χ, χ)) - conj(Gp_markovian_G1(Δtτ_itr, β, β, χ, χ)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp_markovian_G1(Δts_itr, β, β, ν, ν))) * conj(Gpp_markovian_G1(Δτs_itr, β, χ, χ, ν)) + (hbar2 * conj(Gpp_markovian_G1(Δτs_itr, β, χ, χ, ν)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, χ)) + conj(Gp_markovian_G1(Δτs_itr, β, χ, χ, χ)) + conj(Gp(τ_itr, β, χ, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, β, χ, ν, ν))) * (Gp(s_itr, ν, χ, α, α) + conj(Gp_markovian_G1(Δts_itr, α, α, χ, ν)) - conj(Gp_markovian_G1(Δts_itr, β, β, χ, ν)) + conj(Gp_markovian_G1(Δτs_itr, β, β, χ, ν)) - conj(Gp_markovian_G1(Δτs_itr, χ, χ, χ, ν)) - conj(Gp(s_itr, χ, ν, ν, ν)))) * (Gp(t_itr, β, β, α, α) + conj(Gp(t_itr, α, α, β, β)) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, β)) - conj(Gp_markovian_G1(Δts_itr, α, α, χ, χ)) + conj(Gp_markovian_G1(Δtτ_itr, α, α, χ, χ)) - conj(Gp(t_itr, α, α, ν, ν)) + conj(Gp_markovian_G1(Δts_itr, α, α, ν, ν)) - conj(Gp(t_itr, β, β, β, β))) + (- hbar2 * conj(Gpp_markovian_G1(Δts_itr, α, α, χ, ν)) + hbar2 * conj(Gpp_markovian_G1(Δts_itr, β, β, χ, ν)) + (Gp(t_itr, β, β, α, α) - conj(Gp_markovian_G1(Δtτ_itr, β, β, β, β)) - conj(Gp_markovian_G1(Δts_itr, β, β, χ, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, χ, χ)) - conj(Gp(t_itr, β, β, ν, ν)) + conj(Gp_markovian_G1(Δts_itr, β, β, ν, ν))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp_markovian_G1(Δts_itr, α, α, χ, ν)) + conj(Gp_markovian_G1(Δts_itr, β, β, χ, ν)) - conj(Gp_markovian_G1(Δτs_itr, β, β, χ, ν)) + conj(Gp_markovian_G1(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν)))) * (- Gp(τ_itr, χ, β, α, α) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, χ)) + conj(Gp_markovian_G1(Δτs_itr, β, χ, χ, χ)) + conj(Gp(τ_itr, β, χ, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, β, χ, ν, ν)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) - G(s_itr, χ, χ, α, α) + G(τ_itr, χ, χ, α, α) + G(s_itr, ν, ν, α, α) - G(t_itr, ν, ν, α, α) + conj(G(Δtτ_itr, α, α, β, β)) + conj(G(Δts_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δts_itr, α, α, ν, ν)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(Δts_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(Δτs_itr, β, β, χ, χ)) - conj(G(t_itr, β, β, ν, ν)) + conj(G(τ_itr, β, β, ν, ν)) + conj(G(Δts_itr, β, β, ν, ν)) - conj(G(Δτs_itr, β, β, ν, ν)) - conj(G(Δτs_itr, χ, χ, χ, χ)) + conj(G(s_itr, χ, χ, ν, ν)) - conj(G(τ_itr, χ, χ, ν, ν)) + conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, α, ν) * exp(((G(s_itr, α, α, α, α) - G(t_itr, α, α, α, α) - G(s_itr, ν, ν, α, α) + G(t_itr, ν, ν, α, α) - conj(G(s_itr, α, α, ν, ν)) + conj(G(t_itr, α, α, ν, ν)) + conj(G(s_itr, ν, ν, ν, ν)) - conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))
#         )
#     end


#     @inline function calc__rhs_element!(
#         rhs_mat::AbstractMatrix,
#         curr_itr::Int,
#         t_eval_coord::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#     )
#         if use_population_closure && α != β
#             rhs_mat[α, β] = 0.0 + 0.0im
#             return nothing
#         end

#         rhs = include_PL0P_local ? D_L0P(t_eval_coord, α, β) * σ_local(σ_t, α, β) : (0.0 + 0.0im)

#         # -------------------------------------------------------------
#         # P L1 P local left/right mixing.
#         # Left branch:  input (μ, β) -> output (α, β).
#         # Right branch: input (α, ν) -> output (α, β).
#         # These are time-local, so σ_t and t_eval_coord are used.
#         # -------------------------------------------------------------
#         if include_PL1P_local
#             for μ in 1:n_sys
#                 μ == α && continue
#                 if is_secular_pair(α, β, μ, β)
#                     rhs += -(
#                         σ_local(σ_t, μ, β)
#                         * (
#                             Gp(t_eval_coord, α, μ, μ, μ)
#                             -conj(Gp(t_eval_coord, μ, α, β, β))
#                         ) / hbar2
#                     )
#                 end
#             end

#             for ν in 1:n_sys
#                 ν == β && continue
#                 if is_secular_pair(α, β, α, ν)
#                     rhs += (
#                         σ_local(σ_t, α, ν)
#                         * (
#                             Gp(t_eval_coord, ν, β, α, α)
#                             -conj(Gp(t_eval_coord, β, ν, ν, ν))
#                         ) / hbar2
#                     )
#                 end
#             end
#         end

#         if Float64(t_eval_coord) > 1.0 + const_COORD_TOL
#             integral = 0.0 + 0.0im
#             n_s_nodes = n_outer_nodes(curr_itr, t_eval_coord)

#             for s_node in 1:n_s_nodes
#                 s_coord = outer_node_coord(s_node, curr_itr, t_eval_coord)
#                 Δ_coord = Float64(t_eval_coord) - s_coord + 1.0
#                 w_int = ∫weight_to_t(s_node, curr_itr, t_eval_coord)
#                 w_int == 0.0 && continue

#                 kernel = 0.0 + 0.0im

#                 if include_PL0QG0QL1P
#                     # P L0 Q G0 Q L1 P, L branch: input (μ, β).
#                     for μ in 1:n_sys
#                         μ == α && continue
#                         if is_secular_pair(α, β, μ, β)
#                             kernel += kernel_L0_L(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, μ)
#                         end
#                     end

#                     # P L0 Q G0 Q L1 P, R branch: input (α, ν).
#                     for ν in 1:n_sys
#                         ν == β && continue
#                         if is_secular_pair(α, β, α, ν)
#                             kernel += kernel_L0_R(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, ν)
#                         end
#                     end
#                 end

#                 if include_PL1QG0QL1P
#                     # P L1 Q G0 Q L1 P, LL: input (μ, β), intermediate χ.
#                     for χ in 1:n_sys
#                         χ == α && continue
#                         for μ in 1:n_sys
#                             μ == χ && continue
#                             if is_secular_pair(α, β, μ, β)
#                                 kernel += kernel_LL(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, χ, μ)
#                             end
#                         end
#                     end

#                     # P L1 Q G0 Q L1 P, RR: input (α, ν), intermediate χ.
#                     for χ in 1:n_sys
#                         χ == β && continue
#                         for ν in 1:n_sys
#                             ν == χ && continue
#                             if is_secular_pair(α, β, α, ν)
#                                 kernel += kernel_RR(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, χ, ν)
#                             end
#                         end
#                     end

#                     # P L1 Q G0 Q L1 P, RL: input (μ, χ), left input μ, right intermediate χ.
#                     for μ in 1:n_sys
#                         μ == α && continue
#                         for χ in 1:n_sys
#                             χ == β && continue
#                             if is_secular_pair(α, β, μ, χ)
#                                 kernel += kernel_RL(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, χ, μ)
#                             end
#                         end
#                     end

#                     # P L1 Q G0 Q L1 P, LR: input (χ, ν), left intermediate χ, right input ν.
#                     for χ in 1:n_sys
#                         χ == α && continue
#                         for ν in 1:n_sys
#                             ν == β && continue
#                             if is_secular_pair(α, β, χ, ν)
#                                 kernel += kernel_LR(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, χ, ν)
#                             end
#                         end
#                     end
#                 end

#                 if include_PL0QG1QL1P
#                     inner_kernel = 0.0 + 0.0im

#                     n_τ_nodes = n_inner_nodes(s_coord, t_eval_coord)
#                     for τ_node in 1:n_τ_nodes
#                         τ_coord = inner_node_coord(τ_node, s_coord, t_eval_coord)
#                         w_τ = ∫weight_between_coord(τ_node, s_coord, t_eval_coord)
#                         w_τ == 0.0 && continue

#                         τ_kernel = 0.0 + 0.0im

#                         # P L0 Q G1 Q L1 P, LL: input (μ, β), intermediate χ.
#                         for χ in 1:n_sys
#                             χ == α && continue
#                             for μ in 1:n_sys
#                                 μ == χ && continue
#                                 if is_secular_pair(α, β, μ, β)
#                                     τ_kernel += (markovianize_PL0QG1QL1P_G1 ? kernel_L0G1_LL_markovian_G1(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, μ) : kernel_L0G1_LL(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, μ))
#                                 end
#                             end
#                         end

#                         # P L0 Q G1 Q L1 P, RR: input (α, ν), intermediate χ.
#                         for χ in 1:n_sys
#                             χ == β && continue
#                             for ν in 1:n_sys
#                                 ν == χ && continue
#                                 if is_secular_pair(α, β, α, ν)
#                                     τ_kernel += (markovianize_PL0QG1QL1P_G1 ? kernel_L0G1_RR_markovian_G1(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, ν) : kernel_L0G1_RR(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, ν))
#                                 end
#                             end
#                         end

#                         # P L0 Q G1 Q L1 P, RL: input (μ, χ), left input μ, right intermediate χ.
#                         for μ in 1:n_sys
#                             μ == α && continue
#                             for χ in 1:n_sys
#                                 χ == β && continue
#                                 if is_secular_pair(α, β, μ, χ)
#                                     τ_kernel += (markovianize_PL0QG1QL1P_G1 ? kernel_L0G1_RL_markovian_G1(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, μ) : kernel_L0G1_RL(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, μ))
#                                 end
#                             end
#                         end

#                         # P L0 Q G1 Q L1 P, LR: input (χ, ν), left intermediate χ, right input ν.
#                         for χ in 1:n_sys
#                             χ == α && continue
#                             for ν in 1:n_sys
#                                 ν == β && continue
#                                 if is_secular_pair(α, β, χ, ν)
#                                     τ_kernel += (markovianize_PL0QG1QL1P_G1 ? kernel_L0G1_LR_markovian_G1(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, ν) : kernel_L0G1_LR(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, ν))
#                                 end
#                             end
#                         end

#                         inner_kernel += w_τ * τ_kernel
#                     end

#                     kernel += inner_kernel
#                 end

#                 integral += w_int * kernel
#             end

#             rhs += integral
#         end

#         rhs_mat[α, β] = rhs
#         return nothing
#     end

#     @inline function calc__rhs!(
#         rhs_mat::AbstractMatrix,
#         curr_itr::Int,
#         t_eval_coord::Real,
#         σ_t::AbstractMatrix,
#     )
#         fill!(rhs_mat, 0.0 + 0.0im)

#         n_components = n_sys * n_sys
#         if use_threads && Threads.nthreads() > 1 && n_components > 1
#             Threads.@threads for linear_idx in 1:n_components
#                 @inbounds begin
#                     α = ((linear_idx - 1) % n_sys) + 1
#                     β = ((linear_idx - 1) ÷ n_sys) + 1
#                     calc__rhs_element!(rhs_mat, curr_itr, t_eval_coord, σ_t, α, β)
#                 end
#             end
#         else
#             @inbounds for β in 1:n_sys, α in 1:n_sys
#                 calc__rhs_element!(rhs_mat, curr_itr, t_eval_coord, σ_t, α, β)
#             end
#         end

#         return rhs_mat
#     end

#     @inline function enforce_hermiticity!(σ_next::AbstractMatrix)
#         for i in 1:n_sys
#             σ_next[i, i] = real(σ_next[i, i]) + 0.0im
#         end

#         for i in 1:(n_sys - 1)
#             for j in (i + 1):n_sys
#                 c = 0.5 * (σ_next[i, j] + conj(σ_next[j, i]))
#                 σ_next[i, j] = c
#                 σ_next[j, i] = conj(c)
#             end
#         end

#         return σ_next
#     end

#     @inline function enforce_population_closure!(σ_next::AbstractMatrix)
#         for j in 1:n_sys, i in 1:n_sys
#             if i != j
#                 σ_next[i, j] = 0.0 + 0.0im
#             else
#                 σ_next[i, i] = real(σ_next[i, i]) + 0.0im
#             end
#         end
#         return σ_next
#     end

#     σ_stage = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k2      = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k3      = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k4      = Matrix{ComplexF64}(undef, n_sys, n_sys)

#     @inbounds for curr_itr in start_itr:(n_itr - 1)
#         if verbose
#             @printf(
#                 stderr,
#                 "Current iteration: %6d / %6d  method=%s  threads=%d  use_threads=%s  secular=%s  PL0QG1_markovian_G1=%s\n",
#                 curr_itr,
#                 n_itr,
#                 String(method_sym),
#                 Threads.nthreads(),
#                 string(use_threads),
#                 string(use_secular),
#                 string(markovianize_PL0QG1QL1P_G1),
#             )
#         end

#         σ_t    = @view σ[:, :, curr_itr]
#         σ_next = @view σ[:, :, curr_itr + 1]
#         k1     = @view σ′[:, :, curr_itr]

#         if method_sym == :euler
#             calc__rhs!(k1, curr_itr, Float64(curr_itr), σ_t)
#             @. σ_next = σ_t + Δt * k1

#         elseif method_sym == :rk2
#             calc__rhs!(k1, curr_itr, Float64(curr_itr), σ_t)
#             @. σ_stage = σ_t + 0.5 * Δt * k1
#             use_population_closure && enforce_population_closure!(σ_stage)
#             calc__rhs!(k2, curr_itr, Float64(curr_itr) + 0.5, σ_stage)
#             @. σ_next = σ_t + Δt * k2

#         elseif method_sym == :rk4
#             # print("시작0\n")
#             calc__rhs!(k1, curr_itr, Float64(curr_itr), σ_t)

#             # print("시작1\n")
#             @. σ_stage = σ_t + 0.5 * Δt * k1
#             use_population_closure && enforce_population_closure!(σ_stage)
#             calc__rhs!(k2, curr_itr, Float64(curr_itr) + 0.5, σ_stage)

#             # print("시작2\n")
#             @. σ_stage = σ_t + 0.5 * Δt * k2
#             use_population_closure && enforce_population_closure!(σ_stage)
#             calc__rhs!(k3, curr_itr, Float64(curr_itr) + 0.5, σ_stage)

#             # print("시작3\n")
#             @. σ_stage = σ_t + Δt * k3
#             use_population_closure && enforce_population_closure!(σ_stage)
#             calc__rhs!(k4, curr_itr, Float64(curr_itr) + 1.0, σ_stage)

#             @. σ_next = σ_t + (Δt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
#             # print("끝\n")
#         end

#         if use_population_closure
#             enforce_population_closure!(σ_next)
#         elseif enforce_hermitian
#             enforce_hermiticity!(σ_next)
#         end

#         context.curr_itr = UInt64(curr_itr + 1)
#     end

#     return @view σ[:, :, Int(context.curr_itr)]
# end



# OPT 버전

# function calc__σ_σ′_secular_core!(
#     context::RmrtContext;
#     hbar::Real = 1.0,
#     use_population_closure::Bool = false,
    
#     # P L0 P local self block.
#     include_PL0P_local::Bool = true,

#     # P L1 P local left/right mixing block.
#     include_PL1P_local::Bool = true,

#     include_PL0QG0QL1P::Bool = true,
#     include_PL1QG0QL1P::Bool = true,

#     # P L0 Q G1 Q L1 P correction. This is a nested s-tau memory integral,
#     # so it is opt-in in this optimized version.
#     include_PL0QG1QL1P::Bool = true,

#     # Markovianize only the relative-time g′/g″ factors that belong to the
#     # inner G1 correction in P L0 Q G1 Q L1 P.  In this mode,
#     #     g′_{abcd}(∞) ≈ -im * Λ_{abcd},      g″_{abcd}(∞) ≈ 0,
#     # while the outer s/t Gaussian dressing factors are still evaluated from g.
#     markovianize_PL0QG1QL1P_G1::Bool = false,

#     use_secular::Bool = false,
#     secular_tol::Float64 = 1.0e-10,
#     method::Union{Symbol,String} = :rk4,
#     use_threads::Bool = true,

#     # Kept for API compatibility.  This corrected version evaluates RK stage
#     # times through local linear interpolation of g/g′/g″, so the half-grid cache
#     # is not required by this function.
#     auto_prepare_half_shifted_grid::Bool = true,
#     recompute_half_shifted_grid::Bool = false,

#     enforce_hermitian::Bool = true,
#     verbose::Bool = true,
#     verbose_every::Int = 1,
# )
#     start_itr = Int(context.curr_itr)

#     n_sys = context.system.n_sys
#     n_itr = context.simulation_details.num_of_iteration
#     Δt    = context.simulation_details.Δt
#     ϵ     = context.ϵ_exci

#     σ  = context.σ
#     σ′ = context.σ′
#     g  = context.g
#     g′ = context.g′
#     g″ = context.g″

#     if include_PL0QG1QL1P && markovianize_PL0QG1QL1P_G1 && !hasproperty(context, :Λ)
#         error("markovianize_PL0QG1QL1P_G1=true requires context.Λ, with Λ[a,b,c,d] ≈ i*g′_{abcd}(∞).")
#     end
#     Λ = (include_PL0QG1QL1P && markovianize_PL0QG1QL1P_G1) ? getfield(context, :Λ) : nothing

#     hbar_f = Float64(hbar)
#     hbar_f > 0.0 || error("hbar must be positive")
#     hbar2 = hbar_f^2
#     hbar4 = hbar_f^4
#     hbar6 = hbar_f^6

#     method_sym = Symbol(lowercase(String(method)))
#     method_sym in (:euler, :rk2, :rk4) || error(
#         "Unsupported integration method: $(method). Use :euler, :rk2, or :rk4."
#     )

#     # Avoid pathological logging settings; printing to stderr can dominate
#     # short runs, so the loop below prints only every verbose_every steps.
#     verbose_every_f = max(1, Int(verbose_every))

#     # RK2/RK4 stage-time support is handled below by real-coordinate accessors
#     # G/Gp/Gpp.  The half-shifted-grid keyword arguments are kept only for API
#     # compatibility with older callers; this function no longer requires the
#     # half-shifted cache to evaluate midpoint stages.
#     _ = auto_prepare_half_shifted_grid
#     _ = recompute_half_shifted_grid

#     start_itr < n_itr || return @view σ[:, :, n_itr]

#     const_COORD_TOL = 1.0e-10
#     const_N_ITR_FLOAT = Float64(n_itr)
#     const_ZERO_C = 0.0 + 0.0im

#     @inline function ω(a::Int, b::Int)
#         @inbounds return ϵ[a] - ϵ[b]
#     end
#     @inline function Ω(a::Int, b::Int)
#         @inbounds return (ϵ[a] - ϵ[b]) / hbar_f
#     end

#     @inline is_integer_coord(x::Int) = true
#     @inline function is_integer_coord(x::Float64)
#         return abs(x - round(x)) <= const_COORD_TOL
#     end
#     @inline is_integer_coord(x::Real) = is_integer_coord(Float64(x))

#     @inline function clamp_coord(x::Float64)
#         if x < 1.0 - const_COORD_TOL || x > const_N_ITR_FLOAT + const_COORD_TOL
#             error("time-grid coordinate out of range: $(x), allowed [1, $(n_itr)]")
#         end
#         return ifelse(x < 1.0, 1.0, ifelse(x > const_N_ITR_FLOAT, const_N_ITR_FLOAT, x))
#     end
#     @inline clamp_coord(x::Int) = min(max(x, 1), n_itr)
#     @inline clamp_coord(x::Real) = clamp_coord(Float64(x))

#     @inline function coord_to_index(x::Int)
#         return min(max(x, 1), n_itr)
#     end
#     @inline function coord_to_index(x::Float64)
#         idx = round(Int, clamp_coord(x))
#         return min(max(idx, 1), n_itr)
#     end
#     @inline coord_to_index(x::Real) = coord_to_index(Float64(x))

#     @inline function interp_g(A, x::Int, a::Int, b::Int, c::Int, d::Int)
#         idx = min(max(x, 1), n_itr)
#         @inbounds return A[idx, a, b, c, d]
#     end

#     @inline function interp_g(A, x::Float64, a::Int, b::Int, c::Int, d::Int)
#         xf = clamp_coord(x)
#         idx = round(Int, xf)
#         if abs(xf - Float64(idx)) <= const_COORD_TOL
#             idx = min(max(idx, 1), n_itr)
#             @inbounds return A[idx, a, b, c, d]
#         end

#         lo = floor(Int, xf)
#         hi = lo + 1
#         lo = min(max(lo, 1), n_itr)
#         hi = min(max(hi, 1), n_itr)
#         θ = xf - Float64(lo)
#         @inbounds return (1.0 - θ) * A[lo, a, b, c, d] + θ * A[hi, a, b, c, d]
#     end
#     @inline interp_g(A, x::Real, a::Int, b::Int, c::Int, d::Int) = interp_g(A, Float64(x), a, b, c, d)

#     @inline G(x::Int, a::Int, b::Int, c::Int, d::Int) = interp_g(g, x, a, b, c, d)
#     @inline Gp(x::Int, a::Int, b::Int, c::Int, d::Int) = interp_g(g′, x, a, b, c, d)
#     @inline Gpp(x::Int, a::Int, b::Int, c::Int, d::Int) = interp_g(g″, x, a, b, c, d)

#     @inline G(x::Float64, a::Int, b::Int, c::Int, d::Int) = interp_g(g, x, a, b, c, d)
#     @inline Gp(x::Float64, a::Int, b::Int, c::Int, d::Int) = interp_g(g′, x, a, b, c, d)
#     @inline Gpp(x::Float64, a::Int, b::Int, c::Int, d::Int) = interp_g(g″, x, a, b, c, d)

#     @inline G(x::Real, a::Int, b::Int, c::Int, d::Int) = interp_g(g, Float64(x), a, b, c, d)
#     @inline Gp(x::Real, a::Int, b::Int, c::Int, d::Int) = interp_g(g′, Float64(x), a, b, c, d)
#     @inline Gpp(x::Real, a::Int, b::Int, c::Int, d::Int) = interp_g(g″, Float64(x), a, b, c, d)

#     # Integer-grid phase cache.  The full Gaussian dressing still uses exp(...),
#     # but this removes many repeated pure Hamiltonian phase exponentials.
#     phase_cache = Array{ComplexF64}(undef, n_itr, n_sys, n_sys)
#     @inbounds for Δ_idx in 1:n_itr, a in 1:n_sys, b in 1:n_sys
#         phase_cache[Δ_idx, a, b] = exp(-1.0im * (ϵ[a] - ϵ[b]) * ((Float64(Δ_idx) - 1.0) * Δt) / hbar_f)
#     end

#     # Markovian asymptote used only by the optional inner-G1 version of
#     # P L0 Q G1 Q L1 P.  Because Λ stores i*g′(∞), the replacement is
#     # g′(∞) = -im*Λ and g″(∞) = 0.
#     @inline Gp_markovian_G1(Δ_coord::Real, a::Int, b::Int, c::Int, d::Int) =
#         -1.0im * Λ[a, b, c, d]

#     @inline Gpp_markovian_G1(Δ_coord::Real, a::Int, b::Int, c::Int, d::Int) =
#         const_ZERO_C


#     @inline function is_secular_pair(
#         out_a::Int,
#         out_b::Int,
#         in_a::Int,
#         in_b::Int,
#     )
#         if !use_secular
#             return true
#         end
#         return abs(Ω(out_a, out_b) - Ω(in_a, in_b)) <= secular_tol
#     end

#     @inline function n_outer_nodes(curr_itr::Real, t_coord::Real)
#         # Nodes for ∫_0^t ds: all stored integer history nodes 1:curr_itr,
#         # plus the RK stage endpoint t_coord when it lies beyond curr_itr.
#         return (Float64(t_coord) > Float64(curr_itr) + const_COORD_TOL) ? curr_itr + 1 : curr_itr
#     end

#     @inline function outer_node_coord(node_idx::Int, curr_itr::Real, t_coord::Real)
#         return node_idx <= curr_itr ? Float64(node_idx) : Float64(t_coord)
#     end

#     @inline function ∫weight_to_t(node_idx::Int, curr_itr::Real, t_coord::Real)
#         n_nodes = n_outer_nodes(curr_itr, t_coord)
#         n_nodes <= 1 && return 0.0

#         x_prev = node_idx == 1       ? nothing : outer_node_coord(node_idx - 1, curr_itr, t_coord)
#         x      = outer_node_coord(node_idx, curr_itr, t_coord)
#         x_next = node_idx == n_nodes ? nothing : outer_node_coord(node_idx + 1, curr_itr, t_coord)

#         if node_idx == 1
#             return 0.5 * Δt * (x_next - x)
#         elseif node_idx == n_nodes
#             return 0.5 * Δt * (x - x_prev)
#         else
#             return 0.5 * Δt * (x_next - x_prev)
#         end
#     end

#     @inline function n_inner_nodes(s_coord::Real, t_coord::Real)
#         Float64(t_coord) <= Float64(s_coord) + const_COORD_TOL && return 1

#         s_int = coord_to_index(s_coord)
#         t_floor = floor(Int, clamp_coord(t_coord))
#         n_int = max(t_floor - s_int + 1, 1)
#         has_stage_endpoint = !is_integer_coord(t_coord)
#         return n_int + (has_stage_endpoint ? 1 : 0)
#     end

#     @inline function inner_node_coord(node_idx::Int, s_coord::Real, t_coord::Real)
#         s_int = coord_to_index(s_coord)
#         t_floor = floor(Int, clamp_coord(t_coord))
#         n_int = max(t_floor - s_int + 1, 1)
#         return node_idx <= n_int ? Float64(s_int + node_idx - 1) : Float64(t_coord)
#     end

#     @inline function ∫weight_between_coord(node_idx::Int, s_coord::Real, t_coord::Real)
#         Float64(t_coord) <= Float64(s_coord) + const_COORD_TOL && return 0.0

#         n_nodes = n_inner_nodes(s_coord, t_coord)
#         n_nodes <= 1 && return 0.0

#         x_prev = node_idx == 1       ? nothing : inner_node_coord(node_idx - 1, s_coord, t_coord)
#         x      = inner_node_coord(node_idx, s_coord, t_coord)
#         x_next = node_idx == n_nodes ? nothing : inner_node_coord(node_idx + 1, s_coord, t_coord)

#         if node_idx == 1
#             return 0.5 * Δt * (x_next - x)
#         elseif node_idx == n_nodes
#             return 0.5 * Δt * (x - x_prev)
#         else
#             return 0.5 * Δt * (x_next - x_prev)
#         end
#     end

#     @inline function Δtime(Δ_coord::Real)
#         return (Float64(Δ_coord) - 1.0) * Δt
#     end

#     @inline function σ_local(σ_t::AbstractMatrix, a::Int, b::Int)
#         return (use_population_closure && a != b) ? const_ZERO_C : σ_t[a, b]
#     end

#     @inline function σ_mem(
#         σ_t::AbstractMatrix,
#         s_coord::Real,
#         t_coord::Real,
#         a::Int,
#         b::Int,
#     )
#         if use_population_closure && a != b
#             return const_ZERO_C
#         end
#         if abs(Float64(s_coord) - Float64(t_coord)) <= const_COORD_TOL
#             return σ_t[a, b]
#         end
#         is_integer_coord(s_coord) || error("non-endpoint memory coordinate is not stored: $(s_coord)")
#         return σ[a, b, coord_to_index(s_coord)]
#     end

#     @inline function phase_exp(from_a::Int, from_b::Int, Δ_coord::Int)
#         idx = min(max(Δ_coord, 1), n_itr)
#         @inbounds return phase_cache[idx, from_a, from_b]
#     end

#     @inline function phase_exp(from_a::Int, from_b::Int, Δ_coord::Float64)
#         Δ_clamped = clamp_coord(Δ_coord)
#         idx = round(Int, Δ_clamped)
#         if abs(Δ_clamped - Float64(idx)) <= const_COORD_TOL
#             idx = min(max(idx, 1), n_itr)
#             @inbounds return phase_cache[idx, from_a, from_b]
#         end
#         return exp(-1.0im * ω(from_a, from_b) * Δtime(Δ_clamped) / hbar_f)
#     end

#     @inline phase_exp(from_a::Int, from_b::Int, Δ_coord::Real) =
#         phase_exp(from_a, from_b, Float64(Δ_coord))

#     @inline function D_L0P(
#         itr::Real,
#         α::Int,
#         β::Int,
#     )
#         return (
#             -1.0im * ω(α, β) / hbar_f
#             +(
#                 -Gp(itr, α, α, α, α)
#                 +conj(Gp(itr, α, α, β, β))
#                 +Gp(itr, β, β, α, α)
#                 -conj(Gp(itr, β, β, β, β))
#             ) / hbar2
#         )
#     end

#     # ---------------------------------------------------------------------
#     # Explicit PL1 Q G0 Q L1 P, g-closed branch prefactors and coefficients.
#     # Branch naming follows the trace side of the two L1 insertions.
#     # All formulas are written with explicit hbar factors.  With hbar=1 this
#     # reduces to the previous internal convention.
#     # ---------------------------------------------------------------------

#     @inline function exponent_LL(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         return (
#             G(Δ_itr, β, β, χ, χ)
#             -G(s_itr, β, β, μ, μ)
#             +G(t_itr, β, β, μ, μ)
#             -G(Δ_itr, β, β, μ, μ)
#             -G(Δ_itr, χ, χ, χ, χ)
#             +G(s_itr, χ, χ, μ, μ)
#             -G(t_itr, χ, χ, μ, μ)
#             +G(Δ_itr, χ, χ, μ, μ)
#             +conj(G(s_itr, β, β, β, β))
#             -conj(G(t_itr, β, β, β, β))
#             -conj(G(s_itr, χ, χ, β, β))
#             +conj(G(t_itr, χ, χ, β, β))
#         )
#     end

#     @inline function coef_LL(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         left_bracket = (
#             Gp(t_itr, α, χ, χ, χ)
#             -Gp(Δ_itr, α, χ, χ, χ)
#             -Gp(t_itr, α, χ, μ, μ)
#             +Gp(Δ_itr, α, χ, μ, μ)
#         )

#         right_bracket = (
#             Gp(Δ_itr, β, β, χ, μ)
#             -Gp(Δ_itr, χ, χ, χ, μ)
#             -Gp(s_itr, χ, μ, μ, μ)
#             +conj(Gp(s_itr, μ, χ, β, β))
#         )

#         return hbar2 * Gpp(Δ_itr, α, χ, χ, μ) - left_bracket * right_bracket
#     end

#     @inline function kernel_LL(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         return -(
#             σ_mem(σ_t, s_itr, t_itr, μ, β)
#             * phase_exp(χ, β, Δ_itr)
#             * exp(exponent_LL(s_itr, Δ_itr, t_itr, α, β, χ, μ) / hbar2)
#             * coef_LL(s_itr, Δ_itr, t_itr, α, β, χ, μ) / hbar4
#         )
#     end

#     @inline function exponent_LR(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         return (
#             -G(s_itr, β, β, χ, χ)
#             +G(t_itr, β, β, χ, χ)
#             +G(s_itr, χ, χ, χ, χ)
#             -G(t_itr, χ, χ, χ, χ)
#             -conj(G(Δ_itr, β, β, β, β))
#             +conj(G(s_itr, β, β, ν, ν))
#             -conj(G(t_itr, β, β, ν, ν))
#             +conj(G(Δ_itr, β, β, ν, ν))
#             +conj(G(Δ_itr, χ, χ, β, β))
#             -conj(G(s_itr, χ, χ, ν, ν))
#             +conj(G(t_itr, χ, χ, ν, ν))
#             -conj(G(Δ_itr, χ, χ, ν, ν))
#         )
#     end

#     @inline function coef_LR(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         right_bracket = (
#             Gp(s_itr, ν, β, χ, χ)
#             -conj(Gp(Δ_itr, β, β, β, ν))
#             -conj(Gp(s_itr, β, ν, ν, ν))
#             +conj(Gp(Δ_itr, χ, χ, β, ν))
#         )

#         left_bracket = (
#             conj(Gp(t_itr, χ, α, β, β))
#             -conj(Gp(Δ_itr, χ, α, β, β))
#             -conj(Gp(t_itr, χ, α, ν, ν))
#             +conj(Gp(Δ_itr, χ, α, ν, ν))
#         )

#         return hbar2 * conj(Gpp(Δ_itr, χ, α, β, ν)) - right_bracket * left_bracket
#     end

#     @inline function kernel_LR(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         return (
#             σ_mem(σ_t, s_itr, t_itr, χ, ν)
#             * phase_exp(χ, β, Δ_itr)
#             * exp(exponent_LR(s_itr, Δ_itr, t_itr, α, β, χ, ν) / hbar2)
#             * coef_LR(s_itr, Δ_itr, t_itr, α, β, χ, ν) / hbar4
#         )
#     end

#     @inline function exponent_RL(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         return (
#             -G(Δ_itr, α, α, α, α)
#             +G(s_itr, α, α, μ, μ)
#             -G(t_itr, α, α, μ, μ)
#             +G(Δ_itr, α, α, μ, μ)
#             +G(Δ_itr, χ, χ, α, α)
#             -G(s_itr, χ, χ, μ, μ)
#             +G(t_itr, χ, χ, μ, μ)
#             -G(Δ_itr, χ, χ, μ, μ)
#             -conj(G(s_itr, α, α, χ, χ))
#             +conj(G(t_itr, α, α, χ, χ))
#             +conj(G(s_itr, χ, χ, χ, χ))
#             -conj(G(t_itr, χ, χ, χ, χ))
#         )
#     end

#     @inline function coef_RL(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         left_bracket = (
#             Gp(Δ_itr, α, α, α, μ)
#             +Gp(s_itr, α, μ, μ, μ)
#             -Gp(Δ_itr, χ, χ, α, μ)
#             -conj(Gp(s_itr, μ, α, χ, χ))
#         )

#         right_bracket = (
#             Gp(t_itr, χ, β, α, α)
#             -Gp(Δ_itr, χ, β, α, α)
#             -Gp(t_itr, χ, β, μ, μ)
#             +Gp(Δ_itr, χ, β, μ, μ)
#         )

#         return hbar2 * Gpp(Δ_itr, χ, β, α, μ) + left_bracket * right_bracket
#     end

#     @inline function kernel_RL(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         return (
#             σ_mem(σ_t, s_itr, t_itr, μ, χ)
#             * phase_exp(α, χ, Δ_itr)
#             * exp(exponent_RL(s_itr, Δ_itr, t_itr, α, β, χ, μ) / hbar2)
#             * coef_RL(s_itr, Δ_itr, t_itr, α, β, χ, μ) / hbar4
#         )
#     end

#     @inline function exponent_RR(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         # This is the g-part after distributing the leading minus sign in
#         # exp(-(i hbar Δε Δ + G_raw) / hbar^2).
#         return (
#             G(s_itr, α, α, α, α)
#             -G(t_itr, α, α, α, α)
#             -G(s_itr, χ, χ, α, α)
#             +G(t_itr, χ, χ, α, α)
#             +conj(G(Δ_itr, α, α, χ, χ))
#             -conj(G(s_itr, α, α, ν, ν))
#             +conj(G(t_itr, α, α, ν, ν))
#             -conj(G(Δ_itr, α, α, ν, ν))
#             -conj(G(Δ_itr, χ, χ, χ, χ))
#             +conj(G(s_itr, χ, χ, ν, ν))
#             -conj(G(t_itr, χ, χ, ν, ν))
#             +conj(G(Δ_itr, χ, χ, ν, ν))
#         )
#     end

#     @inline function coef_RR(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         right_bracket = (
#             Gp(s_itr, ν, χ, α, α)
#             +conj(Gp(Δ_itr, α, α, χ, ν))
#             -conj(Gp(Δ_itr, χ, χ, χ, ν))
#             -conj(Gp(s_itr, χ, ν, ν, ν))
#         )

#         left_bracket = (
#             conj(Gp(t_itr, β, χ, χ, χ))
#             -conj(Gp(Δ_itr, β, χ, χ, χ))
#             -conj(Gp(t_itr, β, χ, ν, ν))
#             +conj(Gp(Δ_itr, β, χ, ν, ν))
#         )

#         return hbar2 * conj(Gpp(Δ_itr, β, χ, χ, ν)) - right_bracket * left_bracket
#     end

#     @inline function kernel_RR(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         return -(
#             σ_mem(σ_t, s_itr, t_itr, α, ν)
#             * phase_exp(α, χ, Δ_itr)
#             * exp(exponent_RR(s_itr, Δ_itr, t_itr, α, β, χ, ν) / hbar2)
#             * coef_RR(s_itr, Δ_itr, t_itr, α, β, χ, ν) / hbar4
#         )
#     end


#     # ---------------------------------------------------------------------
#     # Explicit P L0 Q G0 Q L1 P, g-closed branch prefactors and coefficients.
#     # Branch L : input (μ, β) -> output (α, β)
#     # Branch R : input (α, ν) -> output (α, β)
#     # ---------------------------------------------------------------------

#     @inline function exponent_L0_L(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         μ::Int,
#     )
#         # exp[-(i hbar ω_{αβ} Δ + G_L) / hbar^2]
#         # = phase_exp(α, β, Δ) * exp[-G_L / hbar^2]
#         return (
#             -G(Δ_itr, α, α, α, α)
#             +G(s_itr, α, α, μ, μ)
#             -G(t_itr, α, α, μ, μ)
#             +G(Δ_itr, α, α, μ, μ)
#             +G(Δ_itr, β, β, α, α)
#             -G(s_itr, β, β, μ, μ)
#             +G(t_itr, β, β, μ, μ)
#             -G(Δ_itr, β, β, μ, μ)
#             -conj(G(s_itr, α, α, β, β))
#             +conj(G(t_itr, α, α, β, β))
#             +conj(G(s_itr, β, β, β, β))
#             -conj(G(t_itr, β, β, β, β))
#         )
#     end

#     @inline function coef_L0_L(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         μ::Int,
#     )
#         left_bracket = (
#             Gp(Δ_itr, α, α, α, μ)
#             +Gp(s_itr, α, μ, μ, μ)
#             -Gp(Δ_itr, β, β, α, μ)
#             -conj(Gp(s_itr, μ, α, β, β))
#         )

#         right_bracket = (
#             Gp(t_itr, α, α, α, α)
#             -Gp(Δ_itr, α, α, α, α)
#             -Gp(t_itr, α, α, μ, μ)
#             +Gp(Δ_itr, α, α, μ, μ)
#             -Gp(t_itr, β, β, α, α)
#             +Gp(Δ_itr, β, β, α, α)
#             +Gp(t_itr, β, β, μ, μ)
#             -Gp(Δ_itr, β, β, μ, μ)
#         )

#         return (
#             hbar2 * Gpp(Δ_itr, α, α, α, μ)
#             -hbar2 * Gpp(Δ_itr, β, β, α, μ)
#             +left_bracket * right_bracket
#         )
#     end

#     @inline function kernel_L0_L(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         μ::Int,
#     )
#         return -(
#             σ_mem(σ_t, s_itr, t_itr, μ, β)
#             * phase_exp(α, β, Δ_itr)
#             * exp(exponent_L0_L(s_itr, Δ_itr, t_itr, α, β, μ) / hbar2)
#             * coef_L0_L(s_itr, Δ_itr, t_itr, α, β, μ) / hbar4
#         )
#     end

#     @inline function exponent_L0_R(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         ν::Int,
#     )
#         # exp[-(i hbar ω_{αβ} Δ + G_R) / hbar^2]
#         # = phase_exp(α, β, Δ) * exp[-G_R / hbar^2]
#         return (
#             +G(s_itr, α, α, α, α)
#             -G(t_itr, α, α, α, α)
#             -G(s_itr, β, β, α, α)
#             +G(t_itr, β, β, α, α)
#             +conj(G(Δ_itr, α, α, β, β))
#             -conj(G(s_itr, α, α, ν, ν))
#             +conj(G(t_itr, α, α, ν, ν))
#             -conj(G(Δ_itr, α, α, ν, ν))
#             -conj(G(Δ_itr, β, β, β, β))
#             +conj(G(s_itr, β, β, ν, ν))
#             -conj(G(t_itr, β, β, ν, ν))
#             +conj(G(Δ_itr, β, β, ν, ν))
#         )
#     end

#     @inline function coef_L0_R(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         α::Int,
#         β::Int,
#         ν::Int,
#     )
#         left_bracket = (
#             Gp(s_itr, ν, β, α, α)
#             +conj(Gp(Δ_itr, α, α, β, ν))
#             -conj(Gp(Δ_itr, β, β, β, ν))
#             -conj(Gp(s_itr, β, ν, ν, ν))
#         )

#         right_bracket = (
#             conj(Gp(t_itr, α, α, β, β))
#             -conj(Gp(Δ_itr, α, α, β, β))
#             -conj(Gp(t_itr, α, α, ν, ν))
#             +conj(Gp(Δ_itr, α, α, ν, ν))
#             -conj(Gp(t_itr, β, β, β, β))
#             +conj(Gp(Δ_itr, β, β, β, β))
#             +conj(Gp(t_itr, β, β, ν, ν))
#             -conj(Gp(Δ_itr, β, β, ν, ν))
#         )

#         return (
#             -hbar2 * conj(Gpp(Δ_itr, α, α, β, ν))
#             +hbar2 * conj(Gpp(Δ_itr, β, β, β, ν))
#             +left_bracket * right_bracket
#         )
#     end

#     @inline function kernel_L0_R(
#         s_itr::Real,
#         Δ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         ν::Int,
#     )
#         return -(
#             σ_mem(σ_t, s_itr, t_itr, α, ν)
#             * phase_exp(α, β, Δ_itr)
#             * exp(exponent_L0_R(s_itr, Δ_itr, t_itr, α, β, ν) / hbar2)
#             * coef_L0_R(s_itr, Δ_itr, t_itr, α, β, ν) / hbar4
#         )
#     end



#     # ---------------------------------------------------------------------
#     # Explicit P L0 Q G1 Q L1 P, g-closed 2-time branch kernels.
#     # The extra variable τ satisfies s <= τ <= t.  Each branch is the
#     # direct g-closed expression from I^{*,L0G1L1,g}_{αβ}(t,τ,s).
#     # ---------------------------------------------------------------------

#     Base.@noinline function kernel_L0G1_LL(
#         s_itr::Real,
#         τ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((hbar2 * Gpp(Δtτ_itr, α, α, α, χ) - hbar2 * Gpp(Δtτ_itr, β, β, α, χ) + (- Gp(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β))) * (- Gp(t_itr, α, α, α, α) + Gp(Δtτ_itr, α, α, α, α) + Gp(t_itr, α, α, χ, χ) - Gp(Δtτ_itr, α, α, χ, χ) + Gp(t_itr, β, β, α, α) - Gp(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp(Δtτ_itr, β, β, χ, χ))) * (Gp(Δτs_itr, β, β, χ, μ) - Gp(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) + G(t_itr, β, β, χ, χ) - G(τ_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(Δτs_itr, β, β, χ, χ) - G(t_itr, β, β, μ, μ) + G(τ_itr, β, β, μ, μ) - G(Δτs_itr, β, β, μ, μ) - G(Δτs_itr, χ, χ, χ, χ) + G(s_itr, χ, χ, μ, μ) - G(τ_itr, χ, χ, μ, μ) + G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(s_itr, χ, χ, β, β)) + conj(G(τ_itr, χ, χ, β, β)) + conj(G(s_itr, μ, μ, β, β)) - conj(G(t_itr, μ, μ, β, β))) / (hbar2))) + (hbar2 * (- Gpp(Δtτ_itr, α, α, α, χ) + Gpp(Δtτ_itr, β, β, α, χ)) * (- Gp(Δts_itr, α, α, χ, μ) + Gp(Δτs_itr, α, α, χ, μ) + Gp(Δts_itr, β, β, χ, μ) - Gp(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β))) + hbar2 * (- Gp(Δtτ_itr, β, β, α, α) - Gp(Δts_itr, β, β, χ, χ) + Gp(Δtτ_itr, β, β, χ, χ) - Gp(t_itr, β, β, μ, μ) + Gp(Δts_itr, β, β, μ, μ) + conj(Gp(t_itr, β, β, β, β))) * Gpp(Δτs_itr, α, χ, χ, μ) + (hbar2 * Gpp(Δτs_itr, α, χ, χ, μ) + (Gp(Δtτ_itr, α, α, α, χ) + Gp(Δτs_itr, α, χ, χ, χ) + Gp(τ_itr, α, χ, μ, μ) - Gp(Δτs_itr, α, χ, μ, μ) - Gp(Δtτ_itr, β, β, α, χ) - conj(Gp(τ_itr, χ, α, β, β))) * (- Gp(Δts_itr, α, α, χ, μ) + Gp(Δτs_itr, α, α, χ, μ) + Gp(Δts_itr, β, β, χ, μ) - Gp(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β)))) * (- Gp(t_itr, α, α, α, α) + Gp(Δtτ_itr, α, α, α, α) + Gp(Δts_itr, α, α, χ, χ) - Gp(Δtτ_itr, α, α, χ, χ) + Gp(t_itr, α, α, μ, μ) - Gp(Δts_itr, α, α, μ, μ) + Gp(t_itr, β, β, α, α) - conj(Gp(t_itr, β, β, β, β))) + (- hbar2 * Gpp(Δts_itr, α, α, χ, μ) + hbar2 * Gpp(Δts_itr, β, β, χ, μ) + (Gp(Δts_itr, α, α, χ, μ) - Gp(Δτs_itr, α, α, χ, μ) - Gp(Δts_itr, β, β, χ, μ) + Gp(Δτs_itr, χ, χ, χ, μ) + Gp(s_itr, χ, μ, μ, μ) - conj(Gp(s_itr, μ, χ, β, β))) * (- Gp(Δtτ_itr, β, β, α, α) - Gp(Δts_itr, β, β, χ, χ) + Gp(Δtτ_itr, β, β, χ, χ) - Gp(t_itr, β, β, μ, μ) + Gp(Δts_itr, β, β, μ, μ) + conj(Gp(t_itr, β, β, β, β)))) * (- Gp(Δtτ_itr, α, α, α, χ) - Gp(Δτs_itr, α, χ, χ, χ) - Gp(τ_itr, α, χ, μ, μ) + Gp(Δτs_itr, α, χ, μ, μ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(Δts_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δτs_itr, α, α, χ, χ) - G(t_itr, α, α, μ, μ) + G(τ_itr, α, α, μ, μ) + G(Δts_itr, α, α, μ, μ) - G(Δτs_itr, α, α, μ, μ) + G(Δtτ_itr, β, β, α, α) + G(Δts_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) - G(Δts_itr, β, β, μ, μ) - G(Δτs_itr, χ, χ, χ, χ) + G(s_itr, χ, χ, μ, μ) - G(τ_itr, χ, χ, μ, μ) + G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(s_itr, χ, χ, β, β)) + conj(G(τ_itr, χ, χ, β, β)) + conj(G(s_itr, μ, μ, β, β)) - conj(G(t_itr, μ, μ, β, β))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, μ, β) * exp(((- G(s_itr, β, β, μ, μ) + G(t_itr, β, β, μ, μ) + G(s_itr, μ, μ, μ, μ) - G(t_itr, μ, μ, μ, μ) + conj(G(s_itr, β, β, β, β)) - conj(G(t_itr, β, β, β, β)) - conj(G(s_itr, μ, μ, β, β)) + conj(G(t_itr, μ, μ, β, β))) / (hbar2)))
#         )
#     end

#     Base.@noinline function kernel_L0G1_LR(
#         s_itr::Real,
#         τ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((- hbar2 * Gpp(Δtτ_itr, α, α, α, χ) + hbar2 * Gpp(Δtτ_itr, β, β, α, χ) + (- Gp(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β))) * (Gp(t_itr, α, α, α, α) - Gp(Δtτ_itr, α, α, α, α) - Gp(t_itr, α, α, χ, χ) + Gp(Δtτ_itr, α, α, χ, χ) - Gp(t_itr, β, β, α, α) + Gp(Δtτ_itr, β, β, α, α) + Gp(t_itr, β, β, χ, χ) - Gp(Δtτ_itr, β, β, χ, χ))) * (- Gp(s_itr, ν, β, χ, χ) + conj(Gp(Δτs_itr, β, β, β, ν)) + conj(Gp(s_itr, β, ν, ν, ν)) - conj(Gp(Δτs_itr, χ, χ, β, ν))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) - G(s_itr, β, β, χ, χ) + G(t_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(t_itr, χ, χ, χ, χ) - G(τ_itr, χ, χ, χ, χ) + G(s_itr, ν, ν, χ, χ) - G(t_itr, ν, ν, χ, χ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(t_itr, β, β, β, β)) + conj(G(τ_itr, β, β, β, β)) - conj(G(Δτs_itr, β, β, β, β)) + conj(G(s_itr, β, β, ν, ν)) - conj(G(τ_itr, β, β, ν, ν)) + conj(G(Δτs_itr, β, β, ν, ν)) + conj(G(Δτs_itr, χ, χ, β, β)) - conj(G(t_itr, χ, χ, ν, ν)) + conj(G(τ_itr, χ, χ, ν, ν)) - conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2))) + (hbar2 * (conj(Gpp(Δts_itr, α, α, β, ν)) - conj(Gpp(Δts_itr, β, β, β, ν))) * (- Gp(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp(Δτs_itr, χ, α, ν, ν))) + hbar2 * (- Gp(Δtτ_itr, α, α, α, α) - Gp(t_itr, α, α, χ, χ) + Gp(Δtτ_itr, α, α, χ, χ) + conj(Gp(Δts_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, ν, ν)) - conj(Gp(Δts_itr, α, α, ν, ν))) * conj(Gpp(Δτs_itr, χ, α, β, ν)) - hbar2 * (- Gp(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp(Δtτ_itr, β, β, χ, χ) + conj(Gp(Δts_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp(Δts_itr, β, β, ν, ν))) * conj(Gpp(Δτs_itr, χ, α, β, ν)) + (hbar2 * conj(Gpp(Δτs_itr, χ, α, β, ν)) + (- Gp(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp(Δτs_itr, χ, α, ν, ν))) * (Gp(s_itr, ν, β, χ, χ) + conj(Gp(Δts_itr, α, α, β, ν)) - conj(Gp(Δτs_itr, α, α, β, ν)) - conj(Gp(Δts_itr, β, β, β, ν)) - conj(Gp(s_itr, β, ν, ν, ν)) + conj(Gp(Δτs_itr, χ, χ, β, ν)))) * (Gp(t_itr, α, α, α, α) - Gp(t_itr, β, β, α, α) - conj(Gp(t_itr, α, α, β, β)) + conj(Gp(t_itr, β, β, β, β))) + (hbar2 * Gpp(Δtτ_itr, α, α, α, χ) - hbar2 * Gpp(Δtτ_itr, β, β, α, χ) + (- Gp(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp(Δτs_itr, χ, α, ν, ν))) * (Gp(Δtτ_itr, α, α, α, α) + Gp(t_itr, α, α, χ, χ) - Gp(Δtτ_itr, α, α, χ, χ) - Gp(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp(Δtτ_itr, β, β, χ, χ) - conj(Gp(Δts_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, ν, ν)) + conj(Gp(Δts_itr, α, α, ν, ν)) + conj(Gp(Δts_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp(Δts_itr, β, β, ν, ν)))) * (- Gp(s_itr, ν, β, χ, χ) - conj(Gp(Δts_itr, α, α, β, ν)) + conj(Gp(Δτs_itr, α, α, β, ν)) + conj(Gp(Δts_itr, β, β, β, ν)) + conj(Gp(s_itr, β, ν, ν, ν)) - conj(Gp(Δτs_itr, χ, χ, β, ν)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) - G(s_itr, β, β, χ, χ) + G(t_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(t_itr, χ, χ, χ, χ) - G(τ_itr, χ, χ, χ, χ) + G(s_itr, ν, ν, χ, χ) - G(t_itr, ν, ν, χ, χ) + conj(G(Δts_itr, α, α, β, β)) - conj(G(Δτs_itr, α, α, β, β)) + conj(G(t_itr, α, α, ν, ν)) - conj(G(τ_itr, α, α, ν, ν)) - conj(G(Δts_itr, α, α, ν, ν)) + conj(G(Δτs_itr, α, α, ν, ν)) - conj(G(Δts_itr, β, β, β, β)) + conj(G(s_itr, β, β, ν, ν)) - conj(G(t_itr, β, β, ν, ν)) + conj(G(Δts_itr, β, β, ν, ν)) + conj(G(Δτs_itr, χ, χ, β, β)) - conj(G(t_itr, χ, χ, ν, ν)) + conj(G(τ_itr, χ, χ, ν, ν)) - conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, χ, ν) * exp(((G(s_itr, χ, χ, χ, χ) - G(t_itr, χ, χ, χ, χ) - G(s_itr, ν, ν, χ, χ) + G(t_itr, ν, ν, χ, χ) - conj(G(s_itr, χ, χ, ν, ν)) + conj(G(t_itr, χ, χ, ν, ν)) + conj(G(s_itr, ν, ν, ν, ν)) - conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))
#         )
#     end

#     Base.@noinline function kernel_L0G1_RL(
#         s_itr::Real,
#         τ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((- hbar2 * conj(Gpp(Δtτ_itr, α, α, β, χ)) + hbar2 * conj(Gpp(Δtτ_itr, β, β, β, χ)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp(Δtτ_itr, α, α, β, χ)) + conj(Gp(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ))) * (- conj(Gp(t_itr, α, α, β, β)) + conj(Gp(Δtτ_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, χ, χ)) - conj(Gp(Δtτ_itr, α, α, χ, χ)) + conj(Gp(t_itr, β, β, β, β)) - conj(Gp(Δtτ_itr, β, β, β, β)) - conj(Gp(t_itr, β, β, χ, χ)) + conj(Gp(Δtτ_itr, β, β, χ, χ)))) * (- Gp(Δτs_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(t_itr, α, α, α, α) + G(τ_itr, α, α, α, α) - G(Δτs_itr, α, α, α, α) + G(s_itr, α, α, μ, μ) - G(τ_itr, α, α, μ, μ) + G(Δτs_itr, α, α, μ, μ) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) + G(Δτs_itr, χ, χ, α, α) - G(t_itr, χ, χ, μ, μ) + G(τ_itr, χ, χ, μ, μ) - G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(Δtτ_itr, α, α, β, β)) - conj(G(s_itr, α, α, χ, χ)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(t_itr, χ, χ, χ, χ)) - conj(G(τ_itr, χ, χ, χ, χ)) + conj(G(s_itr, μ, μ, χ, χ)) - conj(G(t_itr, μ, μ, χ, χ))) / (hbar2))) + (hbar2 * (conj(Gpp(Δtτ_itr, α, α, β, χ)) - conj(Gpp(Δtτ_itr, β, β, β, χ))) * (- Gp(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp(Δts_itr, β, β, α, μ) - Gp(Δτs_itr, β, β, α, μ) + Gp(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) + hbar2 * (- Gp(Δts_itr, α, α, α, α) - Gp(t_itr, α, α, μ, μ) + Gp(Δts_itr, α, α, μ, μ) + conj(Gp(Δtτ_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, χ, χ)) - conj(Gp(Δtτ_itr, α, α, χ, χ))) * Gpp(Δτs_itr, χ, β, α, μ) + (hbar2 * Gpp(Δτs_itr, χ, β, α, μ) + (- Gp(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp(Δts_itr, β, β, α, μ) - Gp(Δτs_itr, β, β, α, μ) + Gp(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) * (Gp(Δτs_itr, χ, β, α, α) + Gp(τ_itr, χ, β, μ, μ) - Gp(Δτs_itr, χ, β, μ, μ) + conj(Gp(Δtτ_itr, α, α, β, χ)) - conj(Gp(Δtτ_itr, β, β, β, χ)) - conj(Gp(τ_itr, β, χ, χ, χ)))) * (Gp(t_itr, α, α, α, α) - Gp(t_itr, β, β, α, α) + Gp(Δts_itr, β, β, α, α) + Gp(t_itr, β, β, μ, μ) - Gp(Δts_itr, β, β, μ, μ) - conj(Gp(t_itr, α, α, β, β)) + conj(Gp(t_itr, β, β, β, β)) - conj(Gp(Δtτ_itr, β, β, β, β)) - conj(Gp(t_itr, β, β, χ, χ)) + conj(Gp(Δtτ_itr, β, β, χ, χ))) + (hbar2 * Gpp(Δts_itr, α, α, α, μ) - hbar2 * Gpp(Δts_itr, β, β, α, μ) + (Gp(Δts_itr, α, α, α, α) + Gp(t_itr, α, α, μ, μ) - Gp(Δts_itr, α, α, μ, μ) - conj(Gp(Δtτ_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, χ, χ)) + conj(Gp(Δtτ_itr, α, α, χ, χ))) * (- Gp(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp(Δts_itr, β, β, α, μ) - Gp(Δτs_itr, β, β, α, μ) + Gp(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ)))) * (- Gp(Δτs_itr, χ, β, α, α) - Gp(τ_itr, χ, β, μ, μ) + Gp(Δτs_itr, χ, β, μ, μ) - conj(Gp(Δtτ_itr, α, α, β, χ)) + conj(Gp(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δts_itr, α, α, α, α) + G(s_itr, α, α, μ, μ) - G(t_itr, α, α, μ, μ) + G(Δts_itr, α, α, μ, μ) + G(Δts_itr, β, β, α, α) - G(Δτs_itr, β, β, α, α) + G(t_itr, β, β, μ, μ) - G(τ_itr, β, β, μ, μ) - G(Δts_itr, β, β, μ, μ) + G(Δτs_itr, β, β, μ, μ) + G(Δτs_itr, χ, χ, α, α) - G(t_itr, χ, χ, μ, μ) + G(τ_itr, χ, χ, μ, μ) - G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(Δtτ_itr, α, α, β, β)) - conj(G(s_itr, α, α, χ, χ)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(t_itr, χ, χ, χ, χ)) - conj(G(τ_itr, χ, χ, χ, χ)) + conj(G(s_itr, μ, μ, χ, χ)) - conj(G(t_itr, μ, μ, χ, χ))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, μ, χ) * exp(((- G(s_itr, χ, χ, μ, μ) + G(t_itr, χ, χ, μ, μ) + G(s_itr, μ, μ, μ, μ) - G(t_itr, μ, μ, μ, μ) + conj(G(s_itr, χ, χ, χ, χ)) - conj(G(t_itr, χ, χ, χ, χ)) - conj(G(s_itr, μ, μ, χ, χ)) + conj(G(t_itr, μ, μ, χ, χ))) / (hbar2)))
#         )
#     end

#     Base.@noinline function kernel_L0G1_RR(
#         s_itr::Real,
#         τ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((hbar2 * conj(Gpp(Δtτ_itr, α, α, β, χ)) - hbar2 * conj(Gpp(Δtτ_itr, β, β, β, χ)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp(Δtτ_itr, α, α, β, χ)) + conj(Gp(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ))) * (conj(Gp(t_itr, α, α, β, β)) - conj(Gp(Δtτ_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, χ, χ)) + conj(Gp(Δtτ_itr, α, α, χ, χ)) - conj(Gp(t_itr, β, β, β, β)) + conj(Gp(Δtτ_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, χ, χ)) - conj(Gp(Δtτ_itr, β, β, χ, χ)))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp(Δτs_itr, α, α, χ, ν)) + conj(Gp(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) - G(s_itr, χ, χ, α, α) + G(τ_itr, χ, χ, α, α) + G(s_itr, ν, ν, α, α) - G(t_itr, ν, ν, α, α) + conj(G(Δtτ_itr, α, α, β, β)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(τ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) + conj(G(Δτs_itr, α, α, χ, χ)) - conj(G(t_itr, α, α, ν, ν)) + conj(G(τ_itr, α, α, ν, ν)) - conj(G(Δτs_itr, α, α, ν, ν)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) - conj(G(Δτs_itr, χ, χ, χ, χ)) + conj(G(s_itr, χ, χ, ν, ν)) - conj(G(τ_itr, χ, χ, ν, ν)) + conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2))) + (hbar2 * (- conj(Gpp(Δtτ_itr, α, α, β, χ)) + conj(Gpp(Δtτ_itr, β, β, β, χ))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp(Δts_itr, α, α, χ, ν)) + conj(Gp(Δts_itr, β, β, χ, ν)) - conj(Gp(Δτs_itr, β, β, χ, ν)) + conj(Gp(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν))) + hbar2 * (- Gp(t_itr, β, β, α, α) + conj(Gp(Δtτ_itr, β, β, β, β)) + conj(Gp(Δts_itr, β, β, χ, χ)) - conj(Gp(Δtτ_itr, β, β, χ, χ)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp(Δts_itr, β, β, ν, ν))) * conj(Gpp(Δτs_itr, β, χ, χ, ν)) + (hbar2 * conj(Gpp(Δτs_itr, β, χ, χ, ν)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp(Δtτ_itr, α, α, β, χ)) + conj(Gp(Δtτ_itr, β, β, β, χ)) + conj(Gp(Δτs_itr, β, χ, χ, χ)) + conj(Gp(τ_itr, β, χ, ν, ν)) - conj(Gp(Δτs_itr, β, χ, ν, ν))) * (Gp(s_itr, ν, χ, α, α) + conj(Gp(Δts_itr, α, α, χ, ν)) - conj(Gp(Δts_itr, β, β, χ, ν)) + conj(Gp(Δτs_itr, β, β, χ, ν)) - conj(Gp(Δτs_itr, χ, χ, χ, ν)) - conj(Gp(s_itr, χ, ν, ν, ν)))) * (Gp(t_itr, β, β, α, α) + conj(Gp(t_itr, α, α, β, β)) - conj(Gp(Δtτ_itr, α, α, β, β)) - conj(Gp(Δts_itr, α, α, χ, χ)) + conj(Gp(Δtτ_itr, α, α, χ, χ)) - conj(Gp(t_itr, α, α, ν, ν)) + conj(Gp(Δts_itr, α, α, ν, ν)) - conj(Gp(t_itr, β, β, β, β))) + (- hbar2 * conj(Gpp(Δts_itr, α, α, χ, ν)) + hbar2 * conj(Gpp(Δts_itr, β, β, χ, ν)) + (Gp(t_itr, β, β, α, α) - conj(Gp(Δtτ_itr, β, β, β, β)) - conj(Gp(Δts_itr, β, β, χ, χ)) + conj(Gp(Δtτ_itr, β, β, χ, χ)) - conj(Gp(t_itr, β, β, ν, ν)) + conj(Gp(Δts_itr, β, β, ν, ν))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp(Δts_itr, α, α, χ, ν)) + conj(Gp(Δts_itr, β, β, χ, ν)) - conj(Gp(Δτs_itr, β, β, χ, ν)) + conj(Gp(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν)))) * (- Gp(τ_itr, χ, β, α, α) - conj(Gp(Δtτ_itr, α, α, β, χ)) + conj(Gp(Δtτ_itr, β, β, β, χ)) + conj(Gp(Δτs_itr, β, χ, χ, χ)) + conj(Gp(τ_itr, β, χ, ν, ν)) - conj(Gp(Δτs_itr, β, χ, ν, ν)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) - G(s_itr, χ, χ, α, α) + G(τ_itr, χ, χ, α, α) + G(s_itr, ν, ν, α, α) - G(t_itr, ν, ν, α, α) + conj(G(Δtτ_itr, α, α, β, β)) + conj(G(Δts_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δts_itr, α, α, ν, ν)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(Δts_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(Δτs_itr, β, β, χ, χ)) - conj(G(t_itr, β, β, ν, ν)) + conj(G(τ_itr, β, β, ν, ν)) + conj(G(Δts_itr, β, β, ν, ν)) - conj(G(Δτs_itr, β, β, ν, ν)) - conj(G(Δτs_itr, χ, χ, χ, χ)) + conj(G(s_itr, χ, χ, ν, ν)) - conj(G(τ_itr, χ, χ, ν, ν)) + conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, α, ν) * exp(((G(s_itr, α, α, α, α) - G(t_itr, α, α, α, α) - G(s_itr, ν, ν, α, α) + G(t_itr, ν, ν, α, α) - conj(G(s_itr, α, α, ν, ν)) + conj(G(t_itr, α, α, ν, ν)) + conj(G(s_itr, ν, ν, ν, ν)) - conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))
#         )
#     end


#     # ---------------------------------------------------------------------
#     # Markovianized inner-G1 variants of P L0 Q G1 Q L1 P.
#     #
#     # These are mechanically identical to the exact 2-time kernels above,
#     # except that relative-time derivative factors with arguments
#     # Δts_itr, Δtτ_itr, and Δτs_itr use
#     #
#     #     Gp  -> Gp_markovian_G1  = -im * Λ
#     #     Gpp -> Gpp_markovian_G1 = 0
#     #
#     # Absolute-time g/g′ factors and all g-exponential dressing factors are
#     # intentionally left unchanged.  Therefore this switch only Markovianizes
#     # the internal G1 derivative/correlation content, not the full outer
#     # P L0 Q ... Q L1 P memory kernel.
#     # ---------------------------------------------------------------------

#     Base.@noinline function kernel_L0G1_LL_markovian_G1(
#         s_itr::Real,
#         τ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((hbar2 * Gpp_markovian_G1(Δtτ_itr, α, α, α, χ) - hbar2 * Gpp_markovian_G1(Δtτ_itr, β, β, α, χ) + (- Gp_markovian_G1(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β))) * (- Gp(t_itr, α, α, α, α) + Gp_markovian_G1(Δtτ_itr, α, α, α, α) + Gp(t_itr, α, α, χ, χ) - Gp_markovian_G1(Δtτ_itr, α, α, χ, χ) + Gp(t_itr, β, β, α, α) - Gp_markovian_G1(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, χ, χ))) * (Gp_markovian_G1(Δτs_itr, β, β, χ, μ) - Gp_markovian_G1(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) + G(t_itr, β, β, χ, χ) - G(τ_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(Δτs_itr, β, β, χ, χ) - G(t_itr, β, β, μ, μ) + G(τ_itr, β, β, μ, μ) - G(Δτs_itr, β, β, μ, μ) - G(Δτs_itr, χ, χ, χ, χ) + G(s_itr, χ, χ, μ, μ) - G(τ_itr, χ, χ, μ, μ) + G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(s_itr, χ, χ, β, β)) + conj(G(τ_itr, χ, χ, β, β)) + conj(G(s_itr, μ, μ, β, β)) - conj(G(t_itr, μ, μ, β, β))) / (hbar2))) + (hbar2 * (- Gpp_markovian_G1(Δtτ_itr, α, α, α, χ) + Gpp_markovian_G1(Δtτ_itr, β, β, α, χ)) * (- Gp_markovian_G1(Δts_itr, α, α, χ, μ) + Gp_markovian_G1(Δτs_itr, α, α, χ, μ) + Gp_markovian_G1(Δts_itr, β, β, χ, μ) - Gp_markovian_G1(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β))) + hbar2 * (- Gp_markovian_G1(Δtτ_itr, β, β, α, α) - Gp_markovian_G1(Δts_itr, β, β, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, χ, χ) - Gp(t_itr, β, β, μ, μ) + Gp_markovian_G1(Δts_itr, β, β, μ, μ) + conj(Gp(t_itr, β, β, β, β))) * Gpp_markovian_G1(Δτs_itr, α, χ, χ, μ) + (hbar2 * Gpp_markovian_G1(Δτs_itr, α, χ, χ, μ) + (Gp_markovian_G1(Δtτ_itr, α, α, α, χ) + Gp_markovian_G1(Δτs_itr, α, χ, χ, χ) + Gp(τ_itr, α, χ, μ, μ) - Gp_markovian_G1(Δτs_itr, α, χ, μ, μ) - Gp_markovian_G1(Δtτ_itr, β, β, α, χ) - conj(Gp(τ_itr, χ, α, β, β))) * (- Gp_markovian_G1(Δts_itr, α, α, χ, μ) + Gp_markovian_G1(Δτs_itr, α, α, χ, μ) + Gp_markovian_G1(Δts_itr, β, β, χ, μ) - Gp_markovian_G1(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β)))) * (- Gp(t_itr, α, α, α, α) + Gp_markovian_G1(Δtτ_itr, α, α, α, α) + Gp_markovian_G1(Δts_itr, α, α, χ, χ) - Gp_markovian_G1(Δtτ_itr, α, α, χ, χ) + Gp(t_itr, α, α, μ, μ) - Gp_markovian_G1(Δts_itr, α, α, μ, μ) + Gp(t_itr, β, β, α, α) - conj(Gp(t_itr, β, β, β, β))) + (- hbar2 * Gpp_markovian_G1(Δts_itr, α, α, χ, μ) + hbar2 * Gpp_markovian_G1(Δts_itr, β, β, χ, μ) + (Gp_markovian_G1(Δts_itr, α, α, χ, μ) - Gp_markovian_G1(Δτs_itr, α, α, χ, μ) - Gp_markovian_G1(Δts_itr, β, β, χ, μ) + Gp_markovian_G1(Δτs_itr, χ, χ, χ, μ) + Gp(s_itr, χ, μ, μ, μ) - conj(Gp(s_itr, μ, χ, β, β))) * (- Gp_markovian_G1(Δtτ_itr, β, β, α, α) - Gp_markovian_G1(Δts_itr, β, β, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, χ, χ) - Gp(t_itr, β, β, μ, μ) + Gp_markovian_G1(Δts_itr, β, β, μ, μ) + conj(Gp(t_itr, β, β, β, β)))) * (- Gp_markovian_G1(Δtτ_itr, α, α, α, χ) - Gp_markovian_G1(Δτs_itr, α, χ, χ, χ) - Gp(τ_itr, α, χ, μ, μ) + Gp_markovian_G1(Δτs_itr, α, χ, μ, μ) + Gp_markovian_G1(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(Δts_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δτs_itr, α, α, χ, χ) - G(t_itr, α, α, μ, μ) + G(τ_itr, α, α, μ, μ) + G(Δts_itr, α, α, μ, μ) - G(Δτs_itr, α, α, μ, μ) + G(Δtτ_itr, β, β, α, α) + G(Δts_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) - G(Δts_itr, β, β, μ, μ) - G(Δτs_itr, χ, χ, χ, χ) + G(s_itr, χ, χ, μ, μ) - G(τ_itr, χ, χ, μ, μ) + G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(s_itr, χ, χ, β, β)) + conj(G(τ_itr, χ, χ, β, β)) + conj(G(s_itr, μ, μ, β, β)) - conj(G(t_itr, μ, μ, β, β))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, μ, β) * exp(((- G(s_itr, β, β, μ, μ) + G(t_itr, β, β, μ, μ) + G(s_itr, μ, μ, μ, μ) - G(t_itr, μ, μ, μ, μ) + conj(G(s_itr, β, β, β, β)) - conj(G(t_itr, β, β, β, β)) - conj(G(s_itr, μ, μ, β, β)) + conj(G(t_itr, μ, μ, β, β))) / (hbar2)))
#         )
#     end


#     Base.@noinline function kernel_L0G1_LR_markovian_G1(
#         s_itr::Real,
#         τ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((- hbar2 * Gpp_markovian_G1(Δtτ_itr, α, α, α, χ) + hbar2 * Gpp_markovian_G1(Δtτ_itr, β, β, α, χ) + (- Gp_markovian_G1(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β))) * (Gp(t_itr, α, α, α, α) - Gp_markovian_G1(Δtτ_itr, α, α, α, α) - Gp(t_itr, α, α, χ, χ) + Gp_markovian_G1(Δtτ_itr, α, α, χ, χ) - Gp(t_itr, β, β, α, α) + Gp_markovian_G1(Δtτ_itr, β, β, α, α) + Gp(t_itr, β, β, χ, χ) - Gp_markovian_G1(Δtτ_itr, β, β, χ, χ))) * (- Gp(s_itr, ν, β, χ, χ) + conj(Gp_markovian_G1(Δτs_itr, β, β, β, ν)) + conj(Gp(s_itr, β, ν, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, χ, χ, β, ν))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) - G(s_itr, β, β, χ, χ) + G(t_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(t_itr, χ, χ, χ, χ) - G(τ_itr, χ, χ, χ, χ) + G(s_itr, ν, ν, χ, χ) - G(t_itr, ν, ν, χ, χ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(t_itr, β, β, β, β)) + conj(G(τ_itr, β, β, β, β)) - conj(G(Δτs_itr, β, β, β, β)) + conj(G(s_itr, β, β, ν, ν)) - conj(G(τ_itr, β, β, ν, ν)) + conj(G(Δτs_itr, β, β, ν, ν)) + conj(G(Δτs_itr, χ, χ, β, β)) - conj(G(t_itr, χ, χ, ν, ν)) + conj(G(τ_itr, χ, χ, ν, ν)) - conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2))) + (hbar2 * (conj(Gpp_markovian_G1(Δts_itr, α, α, β, ν)) - conj(Gpp_markovian_G1(Δts_itr, β, β, β, ν))) * (- Gp_markovian_G1(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, α, χ) + conj(Gp_markovian_G1(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, χ, α, ν, ν))) + hbar2 * (- Gp_markovian_G1(Δtτ_itr, α, α, α, α) - Gp(t_itr, α, α, χ, χ) + Gp_markovian_G1(Δtτ_itr, α, α, χ, χ) + conj(Gp_markovian_G1(Δts_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, ν, ν)) - conj(Gp_markovian_G1(Δts_itr, α, α, ν, ν))) * conj(Gpp_markovian_G1(Δτs_itr, χ, α, β, ν)) - hbar2 * (- Gp_markovian_G1(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, χ, χ) + conj(Gp_markovian_G1(Δts_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp_markovian_G1(Δts_itr, β, β, ν, ν))) * conj(Gpp_markovian_G1(Δτs_itr, χ, α, β, ν)) + (hbar2 * conj(Gpp_markovian_G1(Δτs_itr, χ, α, β, ν)) + (- Gp_markovian_G1(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, α, χ) + conj(Gp_markovian_G1(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, χ, α, ν, ν))) * (Gp(s_itr, ν, β, χ, χ) + conj(Gp_markovian_G1(Δts_itr, α, α, β, ν)) - conj(Gp_markovian_G1(Δτs_itr, α, α, β, ν)) - conj(Gp_markovian_G1(Δts_itr, β, β, β, ν)) - conj(Gp(s_itr, β, ν, ν, ν)) + conj(Gp_markovian_G1(Δτs_itr, χ, χ, β, ν)))) * (Gp(t_itr, α, α, α, α) - Gp(t_itr, β, β, α, α) - conj(Gp(t_itr, α, α, β, β)) + conj(Gp(t_itr, β, β, β, β))) + (hbar2 * Gpp_markovian_G1(Δtτ_itr, α, α, α, χ) - hbar2 * Gpp_markovian_G1(Δtτ_itr, β, β, α, χ) + (- Gp_markovian_G1(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, α, χ) + conj(Gp_markovian_G1(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, χ, α, ν, ν))) * (Gp_markovian_G1(Δtτ_itr, α, α, α, α) + Gp(t_itr, α, α, χ, χ) - Gp_markovian_G1(Δtτ_itr, α, α, χ, χ) - Gp_markovian_G1(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, χ, χ) - conj(Gp_markovian_G1(Δts_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, ν, ν)) + conj(Gp_markovian_G1(Δts_itr, α, α, ν, ν)) + conj(Gp_markovian_G1(Δts_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp_markovian_G1(Δts_itr, β, β, ν, ν)))) * (- Gp(s_itr, ν, β, χ, χ) - conj(Gp_markovian_G1(Δts_itr, α, α, β, ν)) + conj(Gp_markovian_G1(Δτs_itr, α, α, β, ν)) + conj(Gp_markovian_G1(Δts_itr, β, β, β, ν)) + conj(Gp(s_itr, β, ν, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, χ, χ, β, ν)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) - G(s_itr, β, β, χ, χ) + G(t_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(t_itr, χ, χ, χ, χ) - G(τ_itr, χ, χ, χ, χ) + G(s_itr, ν, ν, χ, χ) - G(t_itr, ν, ν, χ, χ) + conj(G(Δts_itr, α, α, β, β)) - conj(G(Δτs_itr, α, α, β, β)) + conj(G(t_itr, α, α, ν, ν)) - conj(G(τ_itr, α, α, ν, ν)) - conj(G(Δts_itr, α, α, ν, ν)) + conj(G(Δτs_itr, α, α, ν, ν)) - conj(G(Δts_itr, β, β, β, β)) + conj(G(s_itr, β, β, ν, ν)) - conj(G(t_itr, β, β, ν, ν)) + conj(G(Δts_itr, β, β, ν, ν)) + conj(G(Δτs_itr, χ, χ, β, β)) - conj(G(t_itr, χ, χ, ν, ν)) + conj(G(τ_itr, χ, χ, ν, ν)) - conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, χ, ν) * exp(((G(s_itr, χ, χ, χ, χ) - G(t_itr, χ, χ, χ, χ) - G(s_itr, ν, ν, χ, χ) + G(t_itr, ν, ν, χ, χ) - conj(G(s_itr, χ, χ, ν, ν)) + conj(G(t_itr, χ, χ, ν, ν)) + conj(G(s_itr, ν, ν, ν, ν)) - conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))
#         )
#     end


#     Base.@noinline function kernel_L0G1_RL_markovian_G1(
#         s_itr::Real,
#         τ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((- hbar2 * conj(Gpp_markovian_G1(Δtτ_itr, α, α, β, χ)) + hbar2 * conj(Gpp_markovian_G1(Δtτ_itr, β, β, β, χ)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ))) * (- conj(Gp(t_itr, α, α, β, β)) + conj(Gp_markovian_G1(Δtτ_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, χ, χ)) - conj(Gp_markovian_G1(Δtτ_itr, α, α, χ, χ)) + conj(Gp(t_itr, β, β, β, β)) - conj(Gp_markovian_G1(Δtτ_itr, β, β, β, β)) - conj(Gp(t_itr, β, β, χ, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, χ, χ)))) * (- Gp_markovian_G1(Δτs_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp_markovian_G1(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(t_itr, α, α, α, α) + G(τ_itr, α, α, α, α) - G(Δτs_itr, α, α, α, α) + G(s_itr, α, α, μ, μ) - G(τ_itr, α, α, μ, μ) + G(Δτs_itr, α, α, μ, μ) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) + G(Δτs_itr, χ, χ, α, α) - G(t_itr, χ, χ, μ, μ) + G(τ_itr, χ, χ, μ, μ) - G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(Δtτ_itr, α, α, β, β)) - conj(G(s_itr, α, α, χ, χ)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(t_itr, χ, χ, χ, χ)) - conj(G(τ_itr, χ, χ, χ, χ)) + conj(G(s_itr, μ, μ, χ, χ)) - conj(G(t_itr, μ, μ, χ, χ))) / (hbar2))) + (hbar2 * (conj(Gpp_markovian_G1(Δtτ_itr, α, α, β, χ)) - conj(Gpp_markovian_G1(Δtτ_itr, β, β, β, χ))) * (- Gp_markovian_G1(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp_markovian_G1(Δts_itr, β, β, α, μ) - Gp_markovian_G1(Δτs_itr, β, β, α, μ) + Gp_markovian_G1(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) + hbar2 * (- Gp_markovian_G1(Δts_itr, α, α, α, α) - Gp(t_itr, α, α, μ, μ) + Gp_markovian_G1(Δts_itr, α, α, μ, μ) + conj(Gp_markovian_G1(Δtτ_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, χ, χ)) - conj(Gp_markovian_G1(Δtτ_itr, α, α, χ, χ))) * Gpp_markovian_G1(Δτs_itr, χ, β, α, μ) + (hbar2 * Gpp_markovian_G1(Δτs_itr, χ, β, α, μ) + (- Gp_markovian_G1(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp_markovian_G1(Δts_itr, β, β, α, μ) - Gp_markovian_G1(Δτs_itr, β, β, α, μ) + Gp_markovian_G1(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) * (Gp_markovian_G1(Δτs_itr, χ, β, α, α) + Gp(τ_itr, χ, β, μ, μ) - Gp_markovian_G1(Δτs_itr, χ, β, μ, μ) + conj(Gp_markovian_G1(Δtτ_itr, α, α, β, χ)) - conj(Gp_markovian_G1(Δtτ_itr, β, β, β, χ)) - conj(Gp(τ_itr, β, χ, χ, χ)))) * (Gp(t_itr, α, α, α, α) - Gp(t_itr, β, β, α, α) + Gp_markovian_G1(Δts_itr, β, β, α, α) + Gp(t_itr, β, β, μ, μ) - Gp_markovian_G1(Δts_itr, β, β, μ, μ) - conj(Gp(t_itr, α, α, β, β)) + conj(Gp(t_itr, β, β, β, β)) - conj(Gp_markovian_G1(Δtτ_itr, β, β, β, β)) - conj(Gp(t_itr, β, β, χ, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, χ, χ))) + (hbar2 * Gpp_markovian_G1(Δts_itr, α, α, α, μ) - hbar2 * Gpp_markovian_G1(Δts_itr, β, β, α, μ) + (Gp_markovian_G1(Δts_itr, α, α, α, α) + Gp(t_itr, α, α, μ, μ) - Gp_markovian_G1(Δts_itr, α, α, μ, μ) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, χ, χ)) + conj(Gp_markovian_G1(Δtτ_itr, α, α, χ, χ))) * (- Gp_markovian_G1(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp_markovian_G1(Δts_itr, β, β, α, μ) - Gp_markovian_G1(Δτs_itr, β, β, α, μ) + Gp_markovian_G1(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ)))) * (- Gp_markovian_G1(Δτs_itr, χ, β, α, α) - Gp(τ_itr, χ, β, μ, μ) + Gp_markovian_G1(Δτs_itr, χ, β, μ, μ) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δts_itr, α, α, α, α) + G(s_itr, α, α, μ, μ) - G(t_itr, α, α, μ, μ) + G(Δts_itr, α, α, μ, μ) + G(Δts_itr, β, β, α, α) - G(Δτs_itr, β, β, α, α) + G(t_itr, β, β, μ, μ) - G(τ_itr, β, β, μ, μ) - G(Δts_itr, β, β, μ, μ) + G(Δτs_itr, β, β, μ, μ) + G(Δτs_itr, χ, χ, α, α) - G(t_itr, χ, χ, μ, μ) + G(τ_itr, χ, χ, μ, μ) - G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(Δtτ_itr, α, α, β, β)) - conj(G(s_itr, α, α, χ, χ)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(t_itr, χ, χ, χ, χ)) - conj(G(τ_itr, χ, χ, χ, χ)) + conj(G(s_itr, μ, μ, χ, χ)) - conj(G(t_itr, μ, μ, χ, χ))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, μ, χ) * exp(((- G(s_itr, χ, χ, μ, μ) + G(t_itr, χ, χ, μ, μ) + G(s_itr, μ, μ, μ, μ) - G(t_itr, μ, μ, μ, μ) + conj(G(s_itr, χ, χ, χ, χ)) - conj(G(t_itr, χ, χ, χ, χ)) - conj(G(s_itr, μ, μ, χ, χ)) + conj(G(t_itr, μ, μ, χ, χ))) / (hbar2)))
#         )
#     end


#     Base.@noinline function kernel_L0G1_RR_markovian_G1(
#         s_itr::Real,
#         τ_itr::Real,
#         t_itr::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((hbar2 * conj(Gpp_markovian_G1(Δtτ_itr, α, α, β, χ)) - hbar2 * conj(Gpp_markovian_G1(Δtτ_itr, β, β, β, χ)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ))) * (conj(Gp(t_itr, α, α, β, β)) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, χ, χ)) + conj(Gp_markovian_G1(Δtτ_itr, α, α, χ, χ)) - conj(Gp(t_itr, β, β, β, β)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, χ, χ)) - conj(Gp_markovian_G1(Δtτ_itr, β, β, χ, χ)))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp_markovian_G1(Δτs_itr, α, α, χ, ν)) + conj(Gp_markovian_G1(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) - G(s_itr, χ, χ, α, α) + G(τ_itr, χ, χ, α, α) + G(s_itr, ν, ν, α, α) - G(t_itr, ν, ν, α, α) + conj(G(Δtτ_itr, α, α, β, β)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(τ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) + conj(G(Δτs_itr, α, α, χ, χ)) - conj(G(t_itr, α, α, ν, ν)) + conj(G(τ_itr, α, α, ν, ν)) - conj(G(Δτs_itr, α, α, ν, ν)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) - conj(G(Δτs_itr, χ, χ, χ, χ)) + conj(G(s_itr, χ, χ, ν, ν)) - conj(G(τ_itr, χ, χ, ν, ν)) + conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2))) + (hbar2 * (- conj(Gpp_markovian_G1(Δtτ_itr, α, α, β, χ)) + conj(Gpp_markovian_G1(Δtτ_itr, β, β, β, χ))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp_markovian_G1(Δts_itr, α, α, χ, ν)) + conj(Gp_markovian_G1(Δts_itr, β, β, χ, ν)) - conj(Gp_markovian_G1(Δτs_itr, β, β, χ, ν)) + conj(Gp_markovian_G1(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν))) + hbar2 * (- Gp(t_itr, β, β, α, α) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, β)) + conj(Gp_markovian_G1(Δts_itr, β, β, χ, χ)) - conj(Gp_markovian_G1(Δtτ_itr, β, β, χ, χ)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp_markovian_G1(Δts_itr, β, β, ν, ν))) * conj(Gpp_markovian_G1(Δτs_itr, β, χ, χ, ν)) + (hbar2 * conj(Gpp_markovian_G1(Δτs_itr, β, χ, χ, ν)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, χ)) + conj(Gp_markovian_G1(Δτs_itr, β, χ, χ, χ)) + conj(Gp(τ_itr, β, χ, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, β, χ, ν, ν))) * (Gp(s_itr, ν, χ, α, α) + conj(Gp_markovian_G1(Δts_itr, α, α, χ, ν)) - conj(Gp_markovian_G1(Δts_itr, β, β, χ, ν)) + conj(Gp_markovian_G1(Δτs_itr, β, β, χ, ν)) - conj(Gp_markovian_G1(Δτs_itr, χ, χ, χ, ν)) - conj(Gp(s_itr, χ, ν, ν, ν)))) * (Gp(t_itr, β, β, α, α) + conj(Gp(t_itr, α, α, β, β)) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, β)) - conj(Gp_markovian_G1(Δts_itr, α, α, χ, χ)) + conj(Gp_markovian_G1(Δtτ_itr, α, α, χ, χ)) - conj(Gp(t_itr, α, α, ν, ν)) + conj(Gp_markovian_G1(Δts_itr, α, α, ν, ν)) - conj(Gp(t_itr, β, β, β, β))) + (- hbar2 * conj(Gpp_markovian_G1(Δts_itr, α, α, χ, ν)) + hbar2 * conj(Gpp_markovian_G1(Δts_itr, β, β, χ, ν)) + (Gp(t_itr, β, β, α, α) - conj(Gp_markovian_G1(Δtτ_itr, β, β, β, β)) - conj(Gp_markovian_G1(Δts_itr, β, β, χ, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, χ, χ)) - conj(Gp(t_itr, β, β, ν, ν)) + conj(Gp_markovian_G1(Δts_itr, β, β, ν, ν))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp_markovian_G1(Δts_itr, α, α, χ, ν)) + conj(Gp_markovian_G1(Δts_itr, β, β, χ, ν)) - conj(Gp_markovian_G1(Δτs_itr, β, β, χ, ν)) + conj(Gp_markovian_G1(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν)))) * (- Gp(τ_itr, χ, β, α, α) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, χ)) + conj(Gp_markovian_G1(Δτs_itr, β, χ, χ, χ)) + conj(Gp(τ_itr, β, χ, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, β, χ, ν, ν)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) - G(s_itr, χ, χ, α, α) + G(τ_itr, χ, χ, α, α) + G(s_itr, ν, ν, α, α) - G(t_itr, ν, ν, α, α) + conj(G(Δtτ_itr, α, α, β, β)) + conj(G(Δts_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δts_itr, α, α, ν, ν)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(Δts_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(Δτs_itr, β, β, χ, χ)) - conj(G(t_itr, β, β, ν, ν)) + conj(G(τ_itr, β, β, ν, ν)) + conj(G(Δts_itr, β, β, ν, ν)) - conj(G(Δτs_itr, β, β, ν, ν)) - conj(G(Δτs_itr, χ, χ, χ, χ)) + conj(G(s_itr, χ, χ, ν, ν)) - conj(G(τ_itr, χ, χ, ν, ν)) + conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, α, ν) * exp(((G(s_itr, α, α, α, α) - G(t_itr, α, α, α, α) - G(s_itr, ν, ν, α, α) + G(t_itr, ν, ν, α, α) - conj(G(s_itr, α, α, ν, ν)) + conj(G(t_itr, α, α, ν, ν)) + conj(G(s_itr, ν, ν, ν, ν)) - conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))
#         )
#     end


#     Base.@noinline function calc__rhs_element!(
#         rhs_mat::AbstractMatrix,
#         curr_itr::Int,
#         t_eval_coord::Real,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#     )
#         if use_population_closure && α != β
#             rhs_mat[α, β] = 0.0 + 0.0im
#             return nothing
#         end

#         rhs = include_PL0P_local ? D_L0P(t_eval_coord, α, β) * σ_local(σ_t, α, β) : (0.0 + 0.0im)

#         # -------------------------------------------------------------
#         # P L1 P local left/right mixing.
#         # Left branch:  input (μ, β) -> output (α, β).
#         # Right branch: input (α, ν) -> output (α, β).
#         # These are time-local, so σ_t and t_eval_coord are used.
#         # -------------------------------------------------------------
#         if include_PL1P_local
#             for μ in 1:n_sys
#                 μ == α && continue
#                 if is_secular_pair(α, β, μ, β)
#                     rhs += -(
#                         σ_local(σ_t, μ, β)
#                         * (
#                             Gp(t_eval_coord, α, μ, μ, μ)
#                             -conj(Gp(t_eval_coord, μ, α, β, β))
#                         ) / hbar2
#                     )
#                 end
#             end

#             for ν in 1:n_sys
#                 ν == β && continue
#                 if is_secular_pair(α, β, α, ν)
#                     rhs += (
#                         σ_local(σ_t, α, ν)
#                         * (
#                             Gp(t_eval_coord, ν, β, α, α)
#                             -conj(Gp(t_eval_coord, β, ν, ν, ν))
#                         ) / hbar2
#                     )
#                 end
#             end
#         end

#         if Float64(t_eval_coord) > 1.0 + const_COORD_TOL
#             integral = 0.0 + 0.0im
#             n_s_nodes = n_outer_nodes(curr_itr, t_eval_coord)

#             for s_node in 1:n_s_nodes
#                 s_coord = outer_node_coord(s_node, curr_itr, t_eval_coord)
#                 Δ_coord = Float64(t_eval_coord) - s_coord + 1.0
#                 w_int = ∫weight_to_t(s_node, curr_itr, t_eval_coord)
#                 w_int == 0.0 && continue

#                 kernel = 0.0 + 0.0im

#                 if include_PL0QG0QL1P
#                     # P L0 Q G0 Q L1 P, L branch: input (μ, β).
#                     for μ in 1:n_sys
#                         μ == α && continue
#                         if is_secular_pair(α, β, μ, β)
#                             kernel += kernel_L0_L(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, μ)
#                         end
#                     end

#                     # P L0 Q G0 Q L1 P, R branch: input (α, ν).
#                     for ν in 1:n_sys
#                         ν == β && continue
#                         if is_secular_pair(α, β, α, ν)
#                             kernel += kernel_L0_R(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, ν)
#                         end
#                     end
#                 end

#                 if include_PL1QG0QL1P
#                     # P L1 Q G0 Q L1 P, LL: input (μ, β), intermediate χ.
#                     for χ in 1:n_sys
#                         χ == α && continue
#                         for μ in 1:n_sys
#                             μ == χ && continue
#                             if is_secular_pair(α, β, μ, β)
#                                 kernel += kernel_LL(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, χ, μ)
#                             end
#                         end
#                     end

#                     # P L1 Q G0 Q L1 P, RR: input (α, ν), intermediate χ.
#                     for χ in 1:n_sys
#                         χ == β && continue
#                         for ν in 1:n_sys
#                             ν == χ && continue
#                             if is_secular_pair(α, β, α, ν)
#                                 kernel += kernel_RR(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, χ, ν)
#                             end
#                         end
#                     end

#                     # P L1 Q G0 Q L1 P, RL: input (μ, χ), left input μ, right intermediate χ.
#                     for μ in 1:n_sys
#                         μ == α && continue
#                         for χ in 1:n_sys
#                             χ == β && continue
#                             if is_secular_pair(α, β, μ, χ)
#                                 kernel += kernel_RL(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, χ, μ)
#                             end
#                         end
#                     end

#                     # P L1 Q G0 Q L1 P, LR: input (χ, ν), left intermediate χ, right input ν.
#                     for χ in 1:n_sys
#                         χ == α && continue
#                         for ν in 1:n_sys
#                             ν == β && continue
#                             if is_secular_pair(α, β, χ, ν)
#                                 kernel += kernel_LR(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, χ, ν)
#                             end
#                         end
#                     end
#                 end

#                 if include_PL0QG1QL1P
#                     inner_kernel = 0.0 + 0.0im

#                     n_τ_nodes = n_inner_nodes(s_coord, t_eval_coord)
#                     for τ_node in 1:n_τ_nodes
#                         τ_coord = inner_node_coord(τ_node, s_coord, t_eval_coord)
#                         w_τ = ∫weight_between_coord(τ_node, s_coord, t_eval_coord)
#                         w_τ == 0.0 && continue

#                         τ_kernel = 0.0 + 0.0im

#                         # P L0 Q G1 Q L1 P, LL: input (μ, β), intermediate χ.
#                         for χ in 1:n_sys
#                             χ == α && continue
#                             for μ in 1:n_sys
#                                 μ == χ && continue
#                                 if is_secular_pair(α, β, μ, β)
#                                     τ_kernel += (markovianize_PL0QG1QL1P_G1 ? kernel_L0G1_LL_markovian_G1(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, μ) : kernel_L0G1_LL(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, μ))
#                                 end
#                             end
#                         end

#                         # P L0 Q G1 Q L1 P, RR: input (α, ν), intermediate χ.
#                         for χ in 1:n_sys
#                             χ == β && continue
#                             for ν in 1:n_sys
#                                 ν == χ && continue
#                                 if is_secular_pair(α, β, α, ν)
#                                     τ_kernel += (markovianize_PL0QG1QL1P_G1 ? kernel_L0G1_RR_markovian_G1(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, ν) : kernel_L0G1_RR(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, ν))
#                                 end
#                             end
#                         end

#                         # P L0 Q G1 Q L1 P, RL: input (μ, χ), left input μ, right intermediate χ.
#                         for μ in 1:n_sys
#                             μ == α && continue
#                             for χ in 1:n_sys
#                                 χ == β && continue
#                                 if is_secular_pair(α, β, μ, χ)
#                                     τ_kernel += (markovianize_PL0QG1QL1P_G1 ? kernel_L0G1_RL_markovian_G1(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, μ) : kernel_L0G1_RL(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, μ))
#                                 end
#                             end
#                         end

#                         # P L0 Q G1 Q L1 P, LR: input (χ, ν), left intermediate χ, right input ν.
#                         for χ in 1:n_sys
#                             χ == α && continue
#                             for ν in 1:n_sys
#                                 ν == β && continue
#                                 if is_secular_pair(α, β, χ, ν)
#                                     τ_kernel += (markovianize_PL0QG1QL1P_G1 ? kernel_L0G1_LR_markovian_G1(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, ν) : kernel_L0G1_LR(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, ν))
#                                 end
#                             end
#                         end

#                         inner_kernel += w_τ * τ_kernel
#                     end

#                     kernel += inner_kernel
#                 end

#                 integral += w_int * kernel
#             end

#             rhs += integral
#         end

#         rhs_mat[α, β] = rhs
#         return nothing
#     end

#     @inline function calc__rhs!(
#         rhs_mat::AbstractMatrix,
#         curr_itr::Int,
#         t_eval_coord::Real,
#         σ_t::AbstractMatrix,
#     )
#         fill!(rhs_mat, 0.0 + 0.0im)

#         n_components = n_sys * n_sys
#         # For small systems, component-level threading creates more overhead than work.
#         min_thread_components = max(16, 4 * Threads.nthreads())
#         if use_threads && Threads.nthreads() > 1 && n_components >= min_thread_components
#             Threads.@threads for linear_idx in 1:n_components
#                 @inbounds begin
#                     α = ((linear_idx - 1) % n_sys) + 1
#                     β = ((linear_idx - 1) ÷ n_sys) + 1
#                     calc__rhs_element!(rhs_mat, curr_itr, t_eval_coord, σ_t, α, β)
#                 end
#             end
#         else
#             @inbounds for β in 1:n_sys, α in 1:n_sys
#                 calc__rhs_element!(rhs_mat, curr_itr, t_eval_coord, σ_t, α, β)
#             end
#         end

#         return rhs_mat
#     end

#     @inline function enforce_hermiticity!(σ_next::AbstractMatrix)
#         for i in 1:n_sys
#             σ_next[i, i] = real(σ_next[i, i]) + 0.0im
#         end

#         for i in 1:(n_sys - 1)
#             for j in (i + 1):n_sys
#                 c = 0.5 * (σ_next[i, j] + conj(σ_next[j, i]))
#                 σ_next[i, j] = c
#                 σ_next[j, i] = conj(c)
#             end
#         end

#         return σ_next
#     end

#     @inline function enforce_population_closure!(σ_next::AbstractMatrix)
#         for j in 1:n_sys, i in 1:n_sys
#             if i != j
#                 σ_next[i, j] = 0.0 + 0.0im
#             else
#                 σ_next[i, i] = real(σ_next[i, i]) + 0.0im
#             end
#         end
#         return σ_next
#     end

#     if verbose && include_PL0QG1QL1P
#         @printf(stderr, "Warning: include_PL0QG1QL1P=true enables the nested s-tau memory term; runtime scales roughly as O(N_t^3).\n")
#     end

#     σ_stage = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k2      = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k3      = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k4      = Matrix{ComplexF64}(undef, n_sys, n_sys)

#     @inbounds for curr_itr in start_itr:(n_itr - 1)
#         if verbose && (curr_itr == start_itr || curr_itr == n_itr - 1 || ((curr_itr - start_itr) % verbose_every_f == 0))
#             @printf(
#                 stderr,
#                 "Current iteration: %6d / %6d  method=%s  threads=%d  use_threads=%s  secular=%s  PL0QG1_markovian_G1=%s\n",
#                 curr_itr,
#                 n_itr,
#                 String(method_sym),
#                 Threads.nthreads(),
#                 string(use_threads),
#                 string(use_secular),
#                 string(markovianize_PL0QG1QL1P_G1),
#             )
#         end

#         σ_t    = @view σ[:, :, curr_itr]
#         σ_next = @view σ[:, :, curr_itr + 1]
#         k1     = @view σ′[:, :, curr_itr]

#         if method_sym == :euler
#             calc__rhs!(k1, curr_itr, Float64(curr_itr), σ_t)
#             @. σ_next = σ_t + Δt * k1

#         elseif method_sym == :rk2
#             calc__rhs!(k1, curr_itr, Float64(curr_itr), σ_t)
#             @. σ_stage = σ_t + 0.5 * Δt * k1
#             use_population_closure && enforce_population_closure!(σ_stage)
#             calc__rhs!(k2, curr_itr, Float64(curr_itr) + 0.5, σ_stage)
#             @. σ_next = σ_t + Δt * k2

#         elseif method_sym == :rk4
#             # print("시작0\n")
#             calc__rhs!(k1, curr_itr, Float64(curr_itr), σ_t)

#             # print("시작1\n")
#             @. σ_stage = σ_t + 0.5 * Δt * k1
#             use_population_closure && enforce_population_closure!(σ_stage)
#             calc__rhs!(k2, curr_itr, Float64(curr_itr) + 0.5, σ_stage)

#             # print("시작2\n")
#             @. σ_stage = σ_t + 0.5 * Δt * k2
#             use_population_closure && enforce_population_closure!(σ_stage)
#             calc__rhs!(k3, curr_itr, Float64(curr_itr) + 0.5, σ_stage)

#             # print("시작3\n")
#             @. σ_stage = σ_t + Δt * k3
#             use_population_closure && enforce_population_closure!(σ_stage)
#             calc__rhs!(k4, curr_itr, Float64(curr_itr) + 1.0, σ_stage)

#             @. σ_next = σ_t + (Δt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
#             # print("끝\n")
#         end

#         if use_population_closure
#             enforce_population_closure!(σ_next)
#         elseif enforce_hermitian
#             enforce_hermiticity!(σ_next)
#         end

#         context.curr_itr = UInt64(curr_itr + 1)
#     end

#     return @view σ[:, :, Int(context.curr_itr)]
# end

# Integer-index optimized version generated from calc__σ_σ′_secular_core!.
# Main changes:
#   - PL0QG1QL1P is disabled by default because it is the O(N_t^3) nested-memory term.
#   - Option-B collapse for PL0QG1QL1P_G1 can approximate the tau integral by
#     a midpoint effective kernel, reducing that term from O(N_t^3) to O(N_t^2).
#   - Large L0G1 kernels are marked @noinline to reduce compile-time IR explosion.
#   - g/g′/g″ accessors use doubled integer q-grid full/half direct access.
#   - Phase factors are cached for doubled integer relative-time grid points.
#   - Verbose logging is throttled through verbose_every.
#   - Heavy memory terms can use component-level threading even for small n_sys.

# function calc__σ_σ′_secular_core!(
#     context::RmrtContext;
#     hbar::Real = 1.0,
#     use_population_closure::Bool = false,
    
#     # P L0 P local self block.
#     include_PL0P_local::Bool = true,

#     # P L1 P local left/right mixing block.
#     include_PL1P_local::Bool = true,

#     include_PL0QG0QL1P::Bool = true,
#     include_PL1QG0QL1P::Bool = true,

#     # P L0 Q G1 Q L1 P correction. This is a nested s-tau memory integral,
#     # so it is opt-in in this optimized version.
#     include_PL0QG1QL1P::Bool = false,

#     # Markovianize only the relative-time g′/g″ factors that belong to the
#     # inner G1 correction in P L0 Q G1 Q L1 P.  In this mode,
#     #     g′_{abcd}(∞) ≈ -im * Λ_{abcd},      g″_{abcd}(∞) ≈ 0,
#     # while the outer s/t Gaussian dressing factors are still evaluated from g.
#     markovianize_PL0QG1QL1P_G1::Bool = false,

#     # Stronger Option-B approximation for P L0 Q G1 Q L1 P.
#     # When true, the inner tau integral is collapsed by a midpoint rule:
#     #     ∫_s^t dτ K(t,τ,s) ≈ (t-s) K(t,nearest_q((t+s)/2),s).
#     # This keeps the outer s-memory integral, but removes the O(N_t) inner tau loop.
#     # It also implies markovianize_PL0QG1QL1P_G1=true for the G1-relative
#     # g′/g″ factors.
#     collapse_tau_PL0QG1QL1P_G1::Bool = false,

#     use_secular::Bool = false,
#     secular_tol::Float64 = 1.0e-10,
#     method::Union{Symbol,String} = :rk4,
#     use_threads::Bool = true,

#     # Half-grid handling.  This version uses doubled integer q-grid indexing:
#     #   full grid i      -> q = 2i - 1
#     #   half grid i+1/2  -> q = 2i
#     # IMPORTANT: for Patternized_g containers, we must NOT materialize a dense
#     # half-grid by looping over all (a,b,c,d), because unsupported patterns may
#     # be intentionally absent.  RK2/RK4 and collapsed-tau G1 therefore require
#     # precomputed patternized half-grid containers on context.
#     auto_prepare_half_shifted_grid::Bool = true,
#     recompute_half_shifted_grid::Bool = false,

#     enforce_hermitian::Bool = true,
#     verbose::Bool = true,
#     verbose_every::Int = 1,
# )
#     start_itr = Int(context.curr_itr)

#     n_sys = context.system.n_sys
#     n_itr = context.simulation_details.num_of_iteration
#     Δt    = context.simulation_details.Δt
#     ϵ     = context.ϵ_exci

#     σ  = context.σ
#     σ′ = context.σ′
#     g  = context.g
#     g′ = context.g′
#     g″ = context.g″

#     effective_markovianize_PL0QG1QL1P_G1 =
#         markovianize_PL0QG1QL1P_G1 || collapse_tau_PL0QG1QL1P_G1

#     if include_PL0QG1QL1P && effective_markovianize_PL0QG1QL1P_G1 && !hasproperty(context, :Λ)
#         error("markovianize/collapse PL0QG1QL1P_G1 requires context.Λ, with Λ[a,b,c,d] ≈ i*g′_{abcd}(∞).")
#     end
#     Λ = (include_PL0QG1QL1P && effective_markovianize_PL0QG1QL1P_G1) ? getfield(context, :Λ) : nothing

#     hbar_f = Float64(hbar)
#     hbar_f > 0.0 || error("hbar must be positive")
#     hbar2 = hbar_f^2
#     hbar4 = hbar_f^4
#     hbar6 = hbar_f^6

#     method_sym = Symbol(lowercase(String(method)))
#     method_sym in (:euler, :rk2, :rk4) || error(
#         "Unsupported integration method: $(method). Use :euler, :rk2, or :rk4."
#     )

#     # Avoid pathological logging settings; printing to stderr can dominate
#     # short runs, so the loop below prints only every verbose_every steps.
#     verbose_every_f = max(1, Int(verbose_every))

#     # Half-grid keyword arguments are kept for API compatibility.  This version
#     # does not materialize dense fallback half-grids, because g/g′/g″ are often
#     # Patternized containers with intentionally unsupported index patterns.
#     _ = auto_prepare_half_shifted_grid
#     _ = recompute_half_shifted_grid

#     start_itr < n_itr || return @view σ[:, :, n_itr]

#     const_ZERO_C = 0.0 + 0.0im
#     const_Q_MAX = 2 * n_itr - 1

#     @inline function ω(a::Int, b::Int)
#         @inbounds return ϵ[a] - ϵ[b]
#     end
#     @inline function Ω(a::Int, b::Int)
#         @inbounds return (ϵ[a] - ϵ[b]) / hbar_f
#     end

#     # ------------------------------------------------------------------
#     # Doubled integer q-grid utilities.
#     #   full grid i      -> q = 2i - 1
#     #   half grid i+1/2  -> q = 2i
#     # Relative times are also q-indices:
#     #   qΔ = q_t - q_s + 1
#     # so full/half relative times close without Float64 coordinates.
#     # ------------------------------------------------------------------

#     @inline q_full(i::Int) = 2 * i - 1
#     @inline q_half_after(i::Int) = 2 * i
#     @inline is_full_q(q::Int) = isodd(q)
#     @inline full_index_from_q(q::Int) = (q + 1) >>> 1
#     @inline half_index_from_q(q::Int) = q >>> 1

#     @inline function check_q(q::Int)
#         (1 <= q <= const_Q_MAX) || error("q-grid index out of range: $(q), allowed [1, $(const_Q_MAX)]")
#         return q
#     end

#     function _first_existing_property(obj, names::Tuple)
#         for name in names
#             if hasproperty(obj, name)
#                 return getfield(obj, name)
#             end
#         end
#         return nothing
#     end

#     # Try to use the user's precomputed half-grid containers if they exist.
#     # The symbol list intentionally accepts several common naming conventions.
#     g_half = _first_existing_property(context, (
#         :g_half,
#         :g_half_shifted,
#         :g_shifted_half,
#         :g_mid,
#         :g_midpoint,
#         Symbol("g__half"),
#         Symbol("g_half_grid"),
#         Symbol("g_half_shifted_grid"),
#     ))
#     gp_half = _first_existing_property(context, (
#         Symbol("g′_half"),
#         Symbol("g′_half_shifted"),
#         Symbol("g′_shifted_half"),
#         Symbol("g′_mid"),
#         Symbol("g′_midpoint"),
#         Symbol("gp_half"),
#         Symbol("gp_half_shifted"),
#         Symbol("g_prime_half"),
#         Symbol("g_prime_half_shifted"),
#     ))
#     gpp_half = _first_existing_property(context, (
#         Symbol("g″_half"),
#         Symbol("g″_half_shifted"),
#         Symbol("g″_shifted_half"),
#         Symbol("g″_mid"),
#         Symbol("g″_midpoint"),
#         Symbol("gpp_half"),
#         Symbol("gpp_half_shifted"),
#         Symbol("g_doubleprime_half"),
#         Symbol("g_doubleprime_half_shifted"),
#     ))

#     needs_half_grid = (method_sym in (:rk2, :rk4)) || (include_PL0QG1QL1P && collapse_tau_PL0QG1QL1P_G1)

#     if needs_half_grid && (g_half === nothing || gp_half === nothing || gpp_half === nothing)
#         missing = String[]
#         g_half === nothing && push!(missing, "g half-grid")
#         gp_half === nothing && push!(missing, "g′ half-grid")
#         gpp_half === nothing && push!(missing, "g″ half-grid")
#         error(
#             "q-grid mode requires precomputed patternized half-grid containers for " *
#             join(missing, ", ") *
#             ". Do not dense-materialize Patternized_g; prepare and store half-grid " *
#             "containers on context, or use method=:euler with collapse_tau_PL0QG1QL1P_G1=false."
#         )
#     end

#     @inline function _missing_half_grid_error(name::String)
#         error(
#             "Attempted to access " * name * " at a half-grid q index, but the corresponding " *
#             "patternized half-grid container was not found on context."
#         )
#     end

#     @inline function G(q::Int, a::Int, b::Int, c::Int, d::Int)
#         check_q(q)
#         if isodd(q)
#             idx = full_index_from_q(q)
#             @inbounds return g[idx, a, b, c, d]
#         else
#             g_half === nothing && _missing_half_grid_error("g")
#             idx = half_index_from_q(q)
#             @inbounds return g_half[idx, a, b, c, d]
#         end
#     end

#     @inline function Gp(q::Int, a::Int, b::Int, c::Int, d::Int)
#         check_q(q)
#         if isodd(q)
#             idx = full_index_from_q(q)
#             @inbounds return g′[idx, a, b, c, d]
#         else
#             gp_half === nothing && _missing_half_grid_error("g′")
#             idx = half_index_from_q(q)
#             @inbounds return gp_half[idx, a, b, c, d]
#         end
#     end

#     @inline function Gpp(q::Int, a::Int, b::Int, c::Int, d::Int)
#         check_q(q)
#         if isodd(q)
#             idx = full_index_from_q(q)
#             @inbounds return g″[idx, a, b, c, d]
#         else
#             gpp_half === nothing && _missing_half_grid_error("g″")
#             idx = half_index_from_q(q)
#             @inbounds return gpp_half[idx, a, b, c, d]
#         end
#     end

#     # Pure Hamiltonian phase cache over doubled relative q-grid.
#     phase_cache = Array{ComplexF64}(undef, const_Q_MAX, n_sys, n_sys)
#     @inbounds for qΔ in 1:const_Q_MAX, a in 1:n_sys, b in 1:n_sys
#         phase_cache[qΔ, a, b] = exp(-1.0im * (ϵ[a] - ϵ[b]) * (0.5 * (Float64(qΔ) - 1.0) * Δt) / hbar_f)
#     end

#     @inline function n_outer_nodes(curr_itr::Int, q_t::Int)
#         q_now = q_full(curr_itr)
#         return q_t > q_now ? curr_itr + 1 : curr_itr
#     end

#     @inline function outer_node_coord(node_idx::Int, curr_itr::Int, q_t::Int)
#         return node_idx <= curr_itr ? q_full(node_idx) : q_t
#     end

#     @inline function ∫weight_to_t(node_idx::Int, curr_itr::Int, q_t::Int)
#         n_nodes = n_outer_nodes(curr_itr, q_t)
#         n_nodes <= 1 && return 0.0

#         q = outer_node_coord(node_idx, curr_itr, q_t)
#         if node_idx == 1
#             q_next = outer_node_coord(node_idx + 1, curr_itr, q_t)
#             return 0.25 * Δt * Float64(q_next - q)
#         elseif node_idx == n_nodes
#             q_prev = outer_node_coord(node_idx - 1, curr_itr, q_t)
#             return 0.25 * Δt * Float64(q - q_prev)
#         else
#             q_prev = outer_node_coord(node_idx - 1, curr_itr, q_t)
#             q_next = outer_node_coord(node_idx + 1, curr_itr, q_t)
#             return 0.25 * Δt * Float64(q_next - q_prev)
#         end
#     end

#     @inline function n_inner_nodes(q_s::Int, q_t::Int)
#         q_t <= q_s && return 1
#         q_floor_full = isodd(q_t) ? q_t : q_t - 1
#         n_full = ((q_floor_full - q_s) >>> 1) + 1
#         return n_full + (iseven(q_t) ? 1 : 0)
#     end

#     @inline function inner_node_coord(node_idx::Int, q_s::Int, q_t::Int)
#         q_floor_full = isodd(q_t) ? q_t : q_t - 1
#         n_full = ((q_floor_full - q_s) >>> 1) + 1
#         return node_idx <= n_full ? q_s + 2 * (node_idx - 1) : q_t
#     end

#     @inline function ∫weight_between_coord(node_idx::Int, q_s::Int, q_t::Int)
#         q_t <= q_s && return 0.0
#         n_nodes = n_inner_nodes(q_s, q_t)
#         n_nodes <= 1 && return 0.0

#         q = inner_node_coord(node_idx, q_s, q_t)
#         if node_idx == 1
#             q_next = inner_node_coord(node_idx + 1, q_s, q_t)
#             return 0.25 * Δt * Float64(q_next - q)
#         elseif node_idx == n_nodes
#             q_prev = inner_node_coord(node_idx - 1, q_s, q_t)
#             return 0.25 * Δt * Float64(q - q_prev)
#         else
#             q_prev = inner_node_coord(node_idx - 1, q_s, q_t)
#             q_next = inner_node_coord(node_idx + 1, q_s, q_t)
#             return 0.25 * Δt * Float64(q_next - q_prev)
#         end
#     end

#     @inline function Δtime(qΔ::Int)
#         return 0.5 * (Float64(qΔ) - 1.0) * Δt
#     end

#     @inline function σ_local(σ_t::AbstractMatrix, a::Int, b::Int)
#         return (use_population_closure && a != b) ? const_ZERO_C : σ_t[a, b]
#     end

#     @inline function σ_mem(
#         σ_t::AbstractMatrix,
#         q_s::Int,
#         q_t::Int,
#         a::Int,
#         b::Int,
#     )
#         if use_population_closure && a != b
#             return const_ZERO_C
#         end
#         if q_s == q_t
#             return σ_t[a, b]
#         end
#         isodd(q_s) || error("non-endpoint memory q index must be a stored full-grid point, got q=$(q_s)")
#         s_idx = full_index_from_q(q_s)
#         @inbounds return σ[a, b, s_idx]
#     end

#     @inline function phase_exp(from_a::Int, from_b::Int, qΔ::Int)
#         check_q(qΔ)
#         @inbounds return phase_cache[qΔ, from_a, from_b]
#     end

#     # Markovian asymptote used only by the optional inner-G1 version of
#     # P L0 Q G1 Q L1 P.  Because Λ stores i*g′(∞), the replacement is
#     # g′(∞) = -im*Λ and g″(∞) = 0.
#     @inline Gp_markovian_G1(Δ_coord::Int, a::Int, b::Int, c::Int, d::Int) =
#         -1.0im * Λ[a, b, c, d]

#     @inline Gpp_markovian_G1(Δ_coord::Int, a::Int, b::Int, c::Int, d::Int) =
#         const_ZERO_C


#     @inline function is_secular_pair(
#         out_a::Int,
#         out_b::Int,
#         in_a::Int,
#         in_b::Int,
#     )
#         if !use_secular
#             return true
#         end
#         return abs(Ω(out_a, out_b) - Ω(in_a, in_b)) <= secular_tol
#     end

#     @inline function D_L0P(
#         itr::Int,
#         α::Int,
#         β::Int,
#     )
#         return (
#             -1.0im * ω(α, β) / hbar_f
#             +(
#                 -Gp(itr, α, α, α, α)
#                 +conj(Gp(itr, α, α, β, β))
#                 +Gp(itr, β, β, α, α)
#                 -conj(Gp(itr, β, β, β, β))
#             ) / hbar2
#         )
#     end

#     # ---------------------------------------------------------------------
#     # Explicit PL1 Q G0 Q L1 P, g-closed branch prefactors and coefficients.
#     # Branch naming follows the trace side of the two L1 insertions.
#     # All formulas are written with explicit hbar factors.  With hbar=1 this
#     # reduces to the previous internal convention.
#     # ---------------------------------------------------------------------

#     @inline function exponent_LL(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         return (
#             G(Δ_itr, β, β, χ, χ)
#             -G(s_itr, β, β, μ, μ)
#             +G(t_itr, β, β, μ, μ)
#             -G(Δ_itr, β, β, μ, μ)
#             -G(Δ_itr, χ, χ, χ, χ)
#             +G(s_itr, χ, χ, μ, μ)
#             -G(t_itr, χ, χ, μ, μ)
#             +G(Δ_itr, χ, χ, μ, μ)
#             +conj(G(s_itr, β, β, β, β))
#             -conj(G(t_itr, β, β, β, β))
#             -conj(G(s_itr, χ, χ, β, β))
#             +conj(G(t_itr, χ, χ, β, β))
#         )
#     end

#     @inline function coef_LL(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         left_bracket = (
#             Gp(t_itr, α, χ, χ, χ)
#             -Gp(Δ_itr, α, χ, χ, χ)
#             -Gp(t_itr, α, χ, μ, μ)
#             +Gp(Δ_itr, α, χ, μ, μ)
#         )

#         right_bracket = (
#             Gp(Δ_itr, β, β, χ, μ)
#             -Gp(Δ_itr, χ, χ, χ, μ)
#             -Gp(s_itr, χ, μ, μ, μ)
#             +conj(Gp(s_itr, μ, χ, β, β))
#         )

#         return hbar2 * Gpp(Δ_itr, α, χ, χ, μ) - left_bracket * right_bracket
#     end

#     @inline function kernel_LL(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         return -(
#             σ_mem(σ_t, s_itr, t_itr, μ, β)
#             * phase_exp(χ, β, Δ_itr)
#             * exp(exponent_LL(s_itr, Δ_itr, t_itr, α, β, χ, μ) / hbar2)
#             * coef_LL(s_itr, Δ_itr, t_itr, α, β, χ, μ) / hbar4
#         )
#     end

#     @inline function exponent_LR(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         return (
#             -G(s_itr, β, β, χ, χ)
#             +G(t_itr, β, β, χ, χ)
#             +G(s_itr, χ, χ, χ, χ)
#             -G(t_itr, χ, χ, χ, χ)
#             -conj(G(Δ_itr, β, β, β, β))
#             +conj(G(s_itr, β, β, ν, ν))
#             -conj(G(t_itr, β, β, ν, ν))
#             +conj(G(Δ_itr, β, β, ν, ν))
#             +conj(G(Δ_itr, χ, χ, β, β))
#             -conj(G(s_itr, χ, χ, ν, ν))
#             +conj(G(t_itr, χ, χ, ν, ν))
#             -conj(G(Δ_itr, χ, χ, ν, ν))
#         )
#     end

#     @inline function coef_LR(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         right_bracket = (
#             Gp(s_itr, ν, β, χ, χ)
#             -conj(Gp(Δ_itr, β, β, β, ν))
#             -conj(Gp(s_itr, β, ν, ν, ν))
#             +conj(Gp(Δ_itr, χ, χ, β, ν))
#         )

#         left_bracket = (
#             conj(Gp(t_itr, χ, α, β, β))
#             -conj(Gp(Δ_itr, χ, α, β, β))
#             -conj(Gp(t_itr, χ, α, ν, ν))
#             +conj(Gp(Δ_itr, χ, α, ν, ν))
#         )

#         return hbar2 * conj(Gpp(Δ_itr, χ, α, β, ν)) - right_bracket * left_bracket
#     end

#     @inline function kernel_LR(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         return (
#             σ_mem(σ_t, s_itr, t_itr, χ, ν)
#             * phase_exp(χ, β, Δ_itr)
#             * exp(exponent_LR(s_itr, Δ_itr, t_itr, α, β, χ, ν) / hbar2)
#             * coef_LR(s_itr, Δ_itr, t_itr, α, β, χ, ν) / hbar4
#         )
#     end

#     @inline function exponent_RL(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         return (
#             -G(Δ_itr, α, α, α, α)
#             +G(s_itr, α, α, μ, μ)
#             -G(t_itr, α, α, μ, μ)
#             +G(Δ_itr, α, α, μ, μ)
#             +G(Δ_itr, χ, χ, α, α)
#             -G(s_itr, χ, χ, μ, μ)
#             +G(t_itr, χ, χ, μ, μ)
#             -G(Δ_itr, χ, χ, μ, μ)
#             -conj(G(s_itr, α, α, χ, χ))
#             +conj(G(t_itr, α, α, χ, χ))
#             +conj(G(s_itr, χ, χ, χ, χ))
#             -conj(G(t_itr, χ, χ, χ, χ))
#         )
#     end

#     @inline function coef_RL(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         left_bracket = (
#             Gp(Δ_itr, α, α, α, μ)
#             +Gp(s_itr, α, μ, μ, μ)
#             -Gp(Δ_itr, χ, χ, α, μ)
#             -conj(Gp(s_itr, μ, α, χ, χ))
#         )

#         right_bracket = (
#             Gp(t_itr, χ, β, α, α)
#             -Gp(Δ_itr, χ, β, α, α)
#             -Gp(t_itr, χ, β, μ, μ)
#             +Gp(Δ_itr, χ, β, μ, μ)
#         )

#         return hbar2 * Gpp(Δ_itr, χ, β, α, μ) + left_bracket * right_bracket
#     end

#     @inline function kernel_RL(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         return (
#             σ_mem(σ_t, s_itr, t_itr, μ, χ)
#             * phase_exp(α, χ, Δ_itr)
#             * exp(exponent_RL(s_itr, Δ_itr, t_itr, α, β, χ, μ) / hbar2)
#             * coef_RL(s_itr, Δ_itr, t_itr, α, β, χ, μ) / hbar4
#         )
#     end

#     @inline function exponent_RR(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         # This is the g-part after distributing the leading minus sign in
#         # exp(-(i hbar Δε Δ + G_raw) / hbar^2).
#         return (
#             G(s_itr, α, α, α, α)
#             -G(t_itr, α, α, α, α)
#             -G(s_itr, χ, χ, α, α)
#             +G(t_itr, χ, χ, α, α)
#             +conj(G(Δ_itr, α, α, χ, χ))
#             -conj(G(s_itr, α, α, ν, ν))
#             +conj(G(t_itr, α, α, ν, ν))
#             -conj(G(Δ_itr, α, α, ν, ν))
#             -conj(G(Δ_itr, χ, χ, χ, χ))
#             +conj(G(s_itr, χ, χ, ν, ν))
#             -conj(G(t_itr, χ, χ, ν, ν))
#             +conj(G(Δ_itr, χ, χ, ν, ν))
#         )
#     end

#     @inline function coef_RR(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         right_bracket = (
#             Gp(s_itr, ν, χ, α, α)
#             +conj(Gp(Δ_itr, α, α, χ, ν))
#             -conj(Gp(Δ_itr, χ, χ, χ, ν))
#             -conj(Gp(s_itr, χ, ν, ν, ν))
#         )

#         left_bracket = (
#             conj(Gp(t_itr, β, χ, χ, χ))
#             -conj(Gp(Δ_itr, β, χ, χ, χ))
#             -conj(Gp(t_itr, β, χ, ν, ν))
#             +conj(Gp(Δ_itr, β, χ, ν, ν))
#         )

#         return hbar2 * conj(Gpp(Δ_itr, β, χ, χ, ν)) - right_bracket * left_bracket
#     end

#     @inline function kernel_RR(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         return -(
#             σ_mem(σ_t, s_itr, t_itr, α, ν)
#             * phase_exp(α, χ, Δ_itr)
#             * exp(exponent_RR(s_itr, Δ_itr, t_itr, α, β, χ, ν) / hbar2)
#             * coef_RR(s_itr, Δ_itr, t_itr, α, β, χ, ν) / hbar4
#         )
#     end


#     # ---------------------------------------------------------------------
#     # Explicit P L0 Q G0 Q L1 P, g-closed branch prefactors and coefficients.
#     # Branch L : input (μ, β) -> output (α, β)
#     # Branch R : input (α, ν) -> output (α, β)
#     # ---------------------------------------------------------------------

#     @inline function exponent_L0_L(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         μ::Int,
#     )
#         # exp[-(i hbar ω_{αβ} Δ + G_L) / hbar^2]
#         # = phase_exp(α, β, Δ) * exp[-G_L / hbar^2]
#         return (
#             -G(Δ_itr, α, α, α, α)
#             +G(s_itr, α, α, μ, μ)
#             -G(t_itr, α, α, μ, μ)
#             +G(Δ_itr, α, α, μ, μ)
#             +G(Δ_itr, β, β, α, α)
#             -G(s_itr, β, β, μ, μ)
#             +G(t_itr, β, β, μ, μ)
#             -G(Δ_itr, β, β, μ, μ)
#             -conj(G(s_itr, α, α, β, β))
#             +conj(G(t_itr, α, α, β, β))
#             +conj(G(s_itr, β, β, β, β))
#             -conj(G(t_itr, β, β, β, β))
#         )
#     end

#     @inline function coef_L0_L(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         μ::Int,
#     )
#         left_bracket = (
#             Gp(Δ_itr, α, α, α, μ)
#             +Gp(s_itr, α, μ, μ, μ)
#             -Gp(Δ_itr, β, β, α, μ)
#             -conj(Gp(s_itr, μ, α, β, β))
#         )

#         right_bracket = (
#             Gp(t_itr, α, α, α, α)
#             -Gp(Δ_itr, α, α, α, α)
#             -Gp(t_itr, α, α, μ, μ)
#             +Gp(Δ_itr, α, α, μ, μ)
#             -Gp(t_itr, β, β, α, α)
#             +Gp(Δ_itr, β, β, α, α)
#             +Gp(t_itr, β, β, μ, μ)
#             -Gp(Δ_itr, β, β, μ, μ)
#         )

#         return (
#             hbar2 * Gpp(Δ_itr, α, α, α, μ)
#             -hbar2 * Gpp(Δ_itr, β, β, α, μ)
#             +left_bracket * right_bracket
#         )
#     end

#     @inline function kernel_L0_L(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         μ::Int,
#     )
#         return -(
#             σ_mem(σ_t, s_itr, t_itr, μ, β)
#             * phase_exp(α, β, Δ_itr)
#             * exp(exponent_L0_L(s_itr, Δ_itr, t_itr, α, β, μ) / hbar2)
#             * coef_L0_L(s_itr, Δ_itr, t_itr, α, β, μ) / hbar4
#         )
#     end

#     @inline function exponent_L0_R(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         ν::Int,
#     )
#         # exp[-(i hbar ω_{αβ} Δ + G_R) / hbar^2]
#         # = phase_exp(α, β, Δ) * exp[-G_R / hbar^2]
#         return (
#             +G(s_itr, α, α, α, α)
#             -G(t_itr, α, α, α, α)
#             -G(s_itr, β, β, α, α)
#             +G(t_itr, β, β, α, α)
#             +conj(G(Δ_itr, α, α, β, β))
#             -conj(G(s_itr, α, α, ν, ν))
#             +conj(G(t_itr, α, α, ν, ν))
#             -conj(G(Δ_itr, α, α, ν, ν))
#             -conj(G(Δ_itr, β, β, β, β))
#             +conj(G(s_itr, β, β, ν, ν))
#             -conj(G(t_itr, β, β, ν, ν))
#             +conj(G(Δ_itr, β, β, ν, ν))
#         )
#     end

#     @inline function coef_L0_R(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         ν::Int,
#     )
#         left_bracket = (
#             Gp(s_itr, ν, β, α, α)
#             +conj(Gp(Δ_itr, α, α, β, ν))
#             -conj(Gp(Δ_itr, β, β, β, ν))
#             -conj(Gp(s_itr, β, ν, ν, ν))
#         )

#         right_bracket = (
#             conj(Gp(t_itr, α, α, β, β))
#             -conj(Gp(Δ_itr, α, α, β, β))
#             -conj(Gp(t_itr, α, α, ν, ν))
#             +conj(Gp(Δ_itr, α, α, ν, ν))
#             -conj(Gp(t_itr, β, β, β, β))
#             +conj(Gp(Δ_itr, β, β, β, β))
#             +conj(Gp(t_itr, β, β, ν, ν))
#             -conj(Gp(Δ_itr, β, β, ν, ν))
#         )

#         return (
#             -hbar2 * conj(Gpp(Δ_itr, α, α, β, ν))
#             +hbar2 * conj(Gpp(Δ_itr, β, β, β, ν))
#             +left_bracket * right_bracket
#         )
#     end

#     @inline function kernel_L0_R(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         ν::Int,
#     )
#         return -(
#             σ_mem(σ_t, s_itr, t_itr, α, ν)
#             * phase_exp(α, β, Δ_itr)
#             * exp(exponent_L0_R(s_itr, Δ_itr, t_itr, α, β, ν) / hbar2)
#             * coef_L0_R(s_itr, Δ_itr, t_itr, α, β, ν) / hbar4
#         )
#     end



#     # ---------------------------------------------------------------------
#     # Explicit P L0 Q G1 Q L1 P, g-closed 2-time branch kernels.
#     # The extra variable τ satisfies s <= τ <= t.  Each branch is the
#     # direct g-closed expression from I^{*,L0G1L1,g}_{αβ}(t,τ,s).
#     # ---------------------------------------------------------------------

#     Base.@noinline function kernel_L0G1_LL(
#         s_itr::Int,
#         τ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((hbar2 * Gpp(Δtτ_itr, α, α, α, χ) - hbar2 * Gpp(Δtτ_itr, β, β, α, χ) + (- Gp(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β))) * (- Gp(t_itr, α, α, α, α) + Gp(Δtτ_itr, α, α, α, α) + Gp(t_itr, α, α, χ, χ) - Gp(Δtτ_itr, α, α, χ, χ) + Gp(t_itr, β, β, α, α) - Gp(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp(Δtτ_itr, β, β, χ, χ))) * (Gp(Δτs_itr, β, β, χ, μ) - Gp(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) + G(t_itr, β, β, χ, χ) - G(τ_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(Δτs_itr, β, β, χ, χ) - G(t_itr, β, β, μ, μ) + G(τ_itr, β, β, μ, μ) - G(Δτs_itr, β, β, μ, μ) - G(Δτs_itr, χ, χ, χ, χ) + G(s_itr, χ, χ, μ, μ) - G(τ_itr, χ, χ, μ, μ) + G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(s_itr, χ, χ, β, β)) + conj(G(τ_itr, χ, χ, β, β)) + conj(G(s_itr, μ, μ, β, β)) - conj(G(t_itr, μ, μ, β, β))) / (hbar2))) + (hbar2 * (- Gpp(Δtτ_itr, α, α, α, χ) + Gpp(Δtτ_itr, β, β, α, χ)) * (- Gp(Δts_itr, α, α, χ, μ) + Gp(Δτs_itr, α, α, χ, μ) + Gp(Δts_itr, β, β, χ, μ) - Gp(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β))) + hbar2 * (- Gp(Δtτ_itr, β, β, α, α) - Gp(Δts_itr, β, β, χ, χ) + Gp(Δtτ_itr, β, β, χ, χ) - Gp(t_itr, β, β, μ, μ) + Gp(Δts_itr, β, β, μ, μ) + conj(Gp(t_itr, β, β, β, β))) * Gpp(Δτs_itr, α, χ, χ, μ) + (hbar2 * Gpp(Δτs_itr, α, χ, χ, μ) + (Gp(Δtτ_itr, α, α, α, χ) + Gp(Δτs_itr, α, χ, χ, χ) + Gp(τ_itr, α, χ, μ, μ) - Gp(Δτs_itr, α, χ, μ, μ) - Gp(Δtτ_itr, β, β, α, χ) - conj(Gp(τ_itr, χ, α, β, β))) * (- Gp(Δts_itr, α, α, χ, μ) + Gp(Δτs_itr, α, α, χ, μ) + Gp(Δts_itr, β, β, χ, μ) - Gp(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β)))) * (- Gp(t_itr, α, α, α, α) + Gp(Δtτ_itr, α, α, α, α) + Gp(Δts_itr, α, α, χ, χ) - Gp(Δtτ_itr, α, α, χ, χ) + Gp(t_itr, α, α, μ, μ) - Gp(Δts_itr, α, α, μ, μ) + Gp(t_itr, β, β, α, α) - conj(Gp(t_itr, β, β, β, β))) + (- hbar2 * Gpp(Δts_itr, α, α, χ, μ) + hbar2 * Gpp(Δts_itr, β, β, χ, μ) + (Gp(Δts_itr, α, α, χ, μ) - Gp(Δτs_itr, α, α, χ, μ) - Gp(Δts_itr, β, β, χ, μ) + Gp(Δτs_itr, χ, χ, χ, μ) + Gp(s_itr, χ, μ, μ, μ) - conj(Gp(s_itr, μ, χ, β, β))) * (- Gp(Δtτ_itr, β, β, α, α) - Gp(Δts_itr, β, β, χ, χ) + Gp(Δtτ_itr, β, β, χ, χ) - Gp(t_itr, β, β, μ, μ) + Gp(Δts_itr, β, β, μ, μ) + conj(Gp(t_itr, β, β, β, β)))) * (- Gp(Δtτ_itr, α, α, α, χ) - Gp(Δτs_itr, α, χ, χ, χ) - Gp(τ_itr, α, χ, μ, μ) + Gp(Δτs_itr, α, χ, μ, μ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(Δts_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δτs_itr, α, α, χ, χ) - G(t_itr, α, α, μ, μ) + G(τ_itr, α, α, μ, μ) + G(Δts_itr, α, α, μ, μ) - G(Δτs_itr, α, α, μ, μ) + G(Δtτ_itr, β, β, α, α) + G(Δts_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) - G(Δts_itr, β, β, μ, μ) - G(Δτs_itr, χ, χ, χ, χ) + G(s_itr, χ, χ, μ, μ) - G(τ_itr, χ, χ, μ, μ) + G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(s_itr, χ, χ, β, β)) + conj(G(τ_itr, χ, χ, β, β)) + conj(G(s_itr, μ, μ, β, β)) - conj(G(t_itr, μ, μ, β, β))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, μ, β) * exp(((- G(s_itr, β, β, μ, μ) + G(t_itr, β, β, μ, μ) + G(s_itr, μ, μ, μ, μ) - G(t_itr, μ, μ, μ, μ) + conj(G(s_itr, β, β, β, β)) - conj(G(t_itr, β, β, β, β)) - conj(G(s_itr, μ, μ, β, β)) + conj(G(t_itr, μ, μ, β, β))) / (hbar2)))
#         )
#     end

#     Base.@noinline function kernel_L0G1_LR(
#         s_itr::Int,
#         τ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((- hbar2 * Gpp(Δtτ_itr, α, α, α, χ) + hbar2 * Gpp(Δtτ_itr, β, β, α, χ) + (- Gp(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β))) * (Gp(t_itr, α, α, α, α) - Gp(Δtτ_itr, α, α, α, α) - Gp(t_itr, α, α, χ, χ) + Gp(Δtτ_itr, α, α, χ, χ) - Gp(t_itr, β, β, α, α) + Gp(Δtτ_itr, β, β, α, α) + Gp(t_itr, β, β, χ, χ) - Gp(Δtτ_itr, β, β, χ, χ))) * (- Gp(s_itr, ν, β, χ, χ) + conj(Gp(Δτs_itr, β, β, β, ν)) + conj(Gp(s_itr, β, ν, ν, ν)) - conj(Gp(Δτs_itr, χ, χ, β, ν))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) - G(s_itr, β, β, χ, χ) + G(t_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(t_itr, χ, χ, χ, χ) - G(τ_itr, χ, χ, χ, χ) + G(s_itr, ν, ν, χ, χ) - G(t_itr, ν, ν, χ, χ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(t_itr, β, β, β, β)) + conj(G(τ_itr, β, β, β, β)) - conj(G(Δτs_itr, β, β, β, β)) + conj(G(s_itr, β, β, ν, ν)) - conj(G(τ_itr, β, β, ν, ν)) + conj(G(Δτs_itr, β, β, ν, ν)) + conj(G(Δτs_itr, χ, χ, β, β)) - conj(G(t_itr, χ, χ, ν, ν)) + conj(G(τ_itr, χ, χ, ν, ν)) - conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2))) + (hbar2 * (conj(Gpp(Δts_itr, α, α, β, ν)) - conj(Gpp(Δts_itr, β, β, β, ν))) * (- Gp(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp(Δτs_itr, χ, α, ν, ν))) + hbar2 * (- Gp(Δtτ_itr, α, α, α, α) - Gp(t_itr, α, α, χ, χ) + Gp(Δtτ_itr, α, α, χ, χ) + conj(Gp(Δts_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, ν, ν)) - conj(Gp(Δts_itr, α, α, ν, ν))) * conj(Gpp(Δτs_itr, χ, α, β, ν)) - hbar2 * (- Gp(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp(Δtτ_itr, β, β, χ, χ) + conj(Gp(Δts_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp(Δts_itr, β, β, ν, ν))) * conj(Gpp(Δτs_itr, χ, α, β, ν)) + (hbar2 * conj(Gpp(Δτs_itr, χ, α, β, ν)) + (- Gp(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp(Δτs_itr, χ, α, ν, ν))) * (Gp(s_itr, ν, β, χ, χ) + conj(Gp(Δts_itr, α, α, β, ν)) - conj(Gp(Δτs_itr, α, α, β, ν)) - conj(Gp(Δts_itr, β, β, β, ν)) - conj(Gp(s_itr, β, ν, ν, ν)) + conj(Gp(Δτs_itr, χ, χ, β, ν)))) * (Gp(t_itr, α, α, α, α) - Gp(t_itr, β, β, α, α) - conj(Gp(t_itr, α, α, β, β)) + conj(Gp(t_itr, β, β, β, β))) + (hbar2 * Gpp(Δtτ_itr, α, α, α, χ) - hbar2 * Gpp(Δtτ_itr, β, β, α, χ) + (- Gp(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp(Δτs_itr, χ, α, ν, ν))) * (Gp(Δtτ_itr, α, α, α, α) + Gp(t_itr, α, α, χ, χ) - Gp(Δtτ_itr, α, α, χ, χ) - Gp(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp(Δtτ_itr, β, β, χ, χ) - conj(Gp(Δts_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, ν, ν)) + conj(Gp(Δts_itr, α, α, ν, ν)) + conj(Gp(Δts_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp(Δts_itr, β, β, ν, ν)))) * (- Gp(s_itr, ν, β, χ, χ) - conj(Gp(Δts_itr, α, α, β, ν)) + conj(Gp(Δτs_itr, α, α, β, ν)) + conj(Gp(Δts_itr, β, β, β, ν)) + conj(Gp(s_itr, β, ν, ν, ν)) - conj(Gp(Δτs_itr, χ, χ, β, ν)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) - G(s_itr, β, β, χ, χ) + G(t_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(t_itr, χ, χ, χ, χ) - G(τ_itr, χ, χ, χ, χ) + G(s_itr, ν, ν, χ, χ) - G(t_itr, ν, ν, χ, χ) + conj(G(Δts_itr, α, α, β, β)) - conj(G(Δτs_itr, α, α, β, β)) + conj(G(t_itr, α, α, ν, ν)) - conj(G(τ_itr, α, α, ν, ν)) - conj(G(Δts_itr, α, α, ν, ν)) + conj(G(Δτs_itr, α, α, ν, ν)) - conj(G(Δts_itr, β, β, β, β)) + conj(G(s_itr, β, β, ν, ν)) - conj(G(t_itr, β, β, ν, ν)) + conj(G(Δts_itr, β, β, ν, ν)) + conj(G(Δτs_itr, χ, χ, β, β)) - conj(G(t_itr, χ, χ, ν, ν)) + conj(G(τ_itr, χ, χ, ν, ν)) - conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, χ, ν) * exp(((G(s_itr, χ, χ, χ, χ) - G(t_itr, χ, χ, χ, χ) - G(s_itr, ν, ν, χ, χ) + G(t_itr, ν, ν, χ, χ) - conj(G(s_itr, χ, χ, ν, ν)) + conj(G(t_itr, χ, χ, ν, ν)) + conj(G(s_itr, ν, ν, ν, ν)) - conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))
#         )
#     end

#     Base.@noinline function kernel_L0G1_RL(
#         s_itr::Int,
#         τ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((- hbar2 * conj(Gpp(Δtτ_itr, α, α, β, χ)) + hbar2 * conj(Gpp(Δtτ_itr, β, β, β, χ)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp(Δtτ_itr, α, α, β, χ)) + conj(Gp(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ))) * (- conj(Gp(t_itr, α, α, β, β)) + conj(Gp(Δtτ_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, χ, χ)) - conj(Gp(Δtτ_itr, α, α, χ, χ)) + conj(Gp(t_itr, β, β, β, β)) - conj(Gp(Δtτ_itr, β, β, β, β)) - conj(Gp(t_itr, β, β, χ, χ)) + conj(Gp(Δtτ_itr, β, β, χ, χ)))) * (- Gp(Δτs_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(t_itr, α, α, α, α) + G(τ_itr, α, α, α, α) - G(Δτs_itr, α, α, α, α) + G(s_itr, α, α, μ, μ) - G(τ_itr, α, α, μ, μ) + G(Δτs_itr, α, α, μ, μ) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) + G(Δτs_itr, χ, χ, α, α) - G(t_itr, χ, χ, μ, μ) + G(τ_itr, χ, χ, μ, μ) - G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(Δtτ_itr, α, α, β, β)) - conj(G(s_itr, α, α, χ, χ)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(t_itr, χ, χ, χ, χ)) - conj(G(τ_itr, χ, χ, χ, χ)) + conj(G(s_itr, μ, μ, χ, χ)) - conj(G(t_itr, μ, μ, χ, χ))) / (hbar2))) + (hbar2 * (conj(Gpp(Δtτ_itr, α, α, β, χ)) - conj(Gpp(Δtτ_itr, β, β, β, χ))) * (- Gp(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp(Δts_itr, β, β, α, μ) - Gp(Δτs_itr, β, β, α, μ) + Gp(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) + hbar2 * (- Gp(Δts_itr, α, α, α, α) - Gp(t_itr, α, α, μ, μ) + Gp(Δts_itr, α, α, μ, μ) + conj(Gp(Δtτ_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, χ, χ)) - conj(Gp(Δtτ_itr, α, α, χ, χ))) * Gpp(Δτs_itr, χ, β, α, μ) + (hbar2 * Gpp(Δτs_itr, χ, β, α, μ) + (- Gp(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp(Δts_itr, β, β, α, μ) - Gp(Δτs_itr, β, β, α, μ) + Gp(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) * (Gp(Δτs_itr, χ, β, α, α) + Gp(τ_itr, χ, β, μ, μ) - Gp(Δτs_itr, χ, β, μ, μ) + conj(Gp(Δtτ_itr, α, α, β, χ)) - conj(Gp(Δtτ_itr, β, β, β, χ)) - conj(Gp(τ_itr, β, χ, χ, χ)))) * (Gp(t_itr, α, α, α, α) - Gp(t_itr, β, β, α, α) + Gp(Δts_itr, β, β, α, α) + Gp(t_itr, β, β, μ, μ) - Gp(Δts_itr, β, β, μ, μ) - conj(Gp(t_itr, α, α, β, β)) + conj(Gp(t_itr, β, β, β, β)) - conj(Gp(Δtτ_itr, β, β, β, β)) - conj(Gp(t_itr, β, β, χ, χ)) + conj(Gp(Δtτ_itr, β, β, χ, χ))) + (hbar2 * Gpp(Δts_itr, α, α, α, μ) - hbar2 * Gpp(Δts_itr, β, β, α, μ) + (Gp(Δts_itr, α, α, α, α) + Gp(t_itr, α, α, μ, μ) - Gp(Δts_itr, α, α, μ, μ) - conj(Gp(Δtτ_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, χ, χ)) + conj(Gp(Δtτ_itr, α, α, χ, χ))) * (- Gp(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp(Δts_itr, β, β, α, μ) - Gp(Δτs_itr, β, β, α, μ) + Gp(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ)))) * (- Gp(Δτs_itr, χ, β, α, α) - Gp(τ_itr, χ, β, μ, μ) + Gp(Δτs_itr, χ, β, μ, μ) - conj(Gp(Δtτ_itr, α, α, β, χ)) + conj(Gp(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δts_itr, α, α, α, α) + G(s_itr, α, α, μ, μ) - G(t_itr, α, α, μ, μ) + G(Δts_itr, α, α, μ, μ) + G(Δts_itr, β, β, α, α) - G(Δτs_itr, β, β, α, α) + G(t_itr, β, β, μ, μ) - G(τ_itr, β, β, μ, μ) - G(Δts_itr, β, β, μ, μ) + G(Δτs_itr, β, β, μ, μ) + G(Δτs_itr, χ, χ, α, α) - G(t_itr, χ, χ, μ, μ) + G(τ_itr, χ, χ, μ, μ) - G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(Δtτ_itr, α, α, β, β)) - conj(G(s_itr, α, α, χ, χ)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(t_itr, χ, χ, χ, χ)) - conj(G(τ_itr, χ, χ, χ, χ)) + conj(G(s_itr, μ, μ, χ, χ)) - conj(G(t_itr, μ, μ, χ, χ))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, μ, χ) * exp(((- G(s_itr, χ, χ, μ, μ) + G(t_itr, χ, χ, μ, μ) + G(s_itr, μ, μ, μ, μ) - G(t_itr, μ, μ, μ, μ) + conj(G(s_itr, χ, χ, χ, χ)) - conj(G(t_itr, χ, χ, χ, χ)) - conj(G(s_itr, μ, μ, χ, χ)) + conj(G(t_itr, μ, μ, χ, χ))) / (hbar2)))
#         )
#     end

#     Base.@noinline function kernel_L0G1_RR(
#         s_itr::Int,
#         τ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((hbar2 * conj(Gpp(Δtτ_itr, α, α, β, χ)) - hbar2 * conj(Gpp(Δtτ_itr, β, β, β, χ)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp(Δtτ_itr, α, α, β, χ)) + conj(Gp(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ))) * (conj(Gp(t_itr, α, α, β, β)) - conj(Gp(Δtτ_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, χ, χ)) + conj(Gp(Δtτ_itr, α, α, χ, χ)) - conj(Gp(t_itr, β, β, β, β)) + conj(Gp(Δtτ_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, χ, χ)) - conj(Gp(Δtτ_itr, β, β, χ, χ)))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp(Δτs_itr, α, α, χ, ν)) + conj(Gp(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) - G(s_itr, χ, χ, α, α) + G(τ_itr, χ, χ, α, α) + G(s_itr, ν, ν, α, α) - G(t_itr, ν, ν, α, α) + conj(G(Δtτ_itr, α, α, β, β)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(τ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) + conj(G(Δτs_itr, α, α, χ, χ)) - conj(G(t_itr, α, α, ν, ν)) + conj(G(τ_itr, α, α, ν, ν)) - conj(G(Δτs_itr, α, α, ν, ν)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) - conj(G(Δτs_itr, χ, χ, χ, χ)) + conj(G(s_itr, χ, χ, ν, ν)) - conj(G(τ_itr, χ, χ, ν, ν)) + conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2))) + (hbar2 * (- conj(Gpp(Δtτ_itr, α, α, β, χ)) + conj(Gpp(Δtτ_itr, β, β, β, χ))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp(Δts_itr, α, α, χ, ν)) + conj(Gp(Δts_itr, β, β, χ, ν)) - conj(Gp(Δτs_itr, β, β, χ, ν)) + conj(Gp(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν))) + hbar2 * (- Gp(t_itr, β, β, α, α) + conj(Gp(Δtτ_itr, β, β, β, β)) + conj(Gp(Δts_itr, β, β, χ, χ)) - conj(Gp(Δtτ_itr, β, β, χ, χ)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp(Δts_itr, β, β, ν, ν))) * conj(Gpp(Δτs_itr, β, χ, χ, ν)) + (hbar2 * conj(Gpp(Δτs_itr, β, χ, χ, ν)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp(Δtτ_itr, α, α, β, χ)) + conj(Gp(Δtτ_itr, β, β, β, χ)) + conj(Gp(Δτs_itr, β, χ, χ, χ)) + conj(Gp(τ_itr, β, χ, ν, ν)) - conj(Gp(Δτs_itr, β, χ, ν, ν))) * (Gp(s_itr, ν, χ, α, α) + conj(Gp(Δts_itr, α, α, χ, ν)) - conj(Gp(Δts_itr, β, β, χ, ν)) + conj(Gp(Δτs_itr, β, β, χ, ν)) - conj(Gp(Δτs_itr, χ, χ, χ, ν)) - conj(Gp(s_itr, χ, ν, ν, ν)))) * (Gp(t_itr, β, β, α, α) + conj(Gp(t_itr, α, α, β, β)) - conj(Gp(Δtτ_itr, α, α, β, β)) - conj(Gp(Δts_itr, α, α, χ, χ)) + conj(Gp(Δtτ_itr, α, α, χ, χ)) - conj(Gp(t_itr, α, α, ν, ν)) + conj(Gp(Δts_itr, α, α, ν, ν)) - conj(Gp(t_itr, β, β, β, β))) + (- hbar2 * conj(Gpp(Δts_itr, α, α, χ, ν)) + hbar2 * conj(Gpp(Δts_itr, β, β, χ, ν)) + (Gp(t_itr, β, β, α, α) - conj(Gp(Δtτ_itr, β, β, β, β)) - conj(Gp(Δts_itr, β, β, χ, χ)) + conj(Gp(Δtτ_itr, β, β, χ, χ)) - conj(Gp(t_itr, β, β, ν, ν)) + conj(Gp(Δts_itr, β, β, ν, ν))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp(Δts_itr, α, α, χ, ν)) + conj(Gp(Δts_itr, β, β, χ, ν)) - conj(Gp(Δτs_itr, β, β, χ, ν)) + conj(Gp(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν)))) * (- Gp(τ_itr, χ, β, α, α) - conj(Gp(Δtτ_itr, α, α, β, χ)) + conj(Gp(Δtτ_itr, β, β, β, χ)) + conj(Gp(Δτs_itr, β, χ, χ, χ)) + conj(Gp(τ_itr, β, χ, ν, ν)) - conj(Gp(Δτs_itr, β, χ, ν, ν)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) - G(s_itr, χ, χ, α, α) + G(τ_itr, χ, χ, α, α) + G(s_itr, ν, ν, α, α) - G(t_itr, ν, ν, α, α) + conj(G(Δtτ_itr, α, α, β, β)) + conj(G(Δts_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δts_itr, α, α, ν, ν)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(Δts_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(Δτs_itr, β, β, χ, χ)) - conj(G(t_itr, β, β, ν, ν)) + conj(G(τ_itr, β, β, ν, ν)) + conj(G(Δts_itr, β, β, ν, ν)) - conj(G(Δτs_itr, β, β, ν, ν)) - conj(G(Δτs_itr, χ, χ, χ, χ)) + conj(G(s_itr, χ, χ, ν, ν)) - conj(G(τ_itr, χ, χ, ν, ν)) + conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, α, ν) * exp(((G(s_itr, α, α, α, α) - G(t_itr, α, α, α, α) - G(s_itr, ν, ν, α, α) + G(t_itr, ν, ν, α, α) - conj(G(s_itr, α, α, ν, ν)) + conj(G(t_itr, α, α, ν, ν)) + conj(G(s_itr, ν, ν, ν, ν)) - conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))
#         )
#     end


#     # ---------------------------------------------------------------------
#     # Markovianized inner-G1 variants of P L0 Q G1 Q L1 P.
#     #
#     # These are mechanically identical to the exact 2-time kernels above,
#     # except that relative-time derivative factors with arguments
#     # Δts_itr, Δtτ_itr, and Δτs_itr use
#     #
#     #     Gp  -> Gp_markovian_G1  = -im * Λ
#     #     Gpp -> Gpp_markovian_G1 = 0
#     #
#     # Absolute-time g/g′ factors and all g-exponential dressing factors are
#     # intentionally left unchanged.  Therefore this switch only Markovianizes
#     # the internal G1 derivative/correlation content, not the full outer
#     # P L0 Q ... Q L1 P memory kernel.
#     # ---------------------------------------------------------------------

#     Base.@noinline function kernel_L0G1_LL_markovian_G1(
#         s_itr::Int,
#         τ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((hbar2 * Gpp_markovian_G1(Δtτ_itr, α, α, α, χ) - hbar2 * Gpp_markovian_G1(Δtτ_itr, β, β, α, χ) + (- Gp_markovian_G1(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β))) * (- Gp(t_itr, α, α, α, α) + Gp_markovian_G1(Δtτ_itr, α, α, α, α) + Gp(t_itr, α, α, χ, χ) - Gp_markovian_G1(Δtτ_itr, α, α, χ, χ) + Gp(t_itr, β, β, α, α) - Gp_markovian_G1(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, χ, χ))) * (Gp_markovian_G1(Δτs_itr, β, β, χ, μ) - Gp_markovian_G1(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) + G(t_itr, β, β, χ, χ) - G(τ_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(Δτs_itr, β, β, χ, χ) - G(t_itr, β, β, μ, μ) + G(τ_itr, β, β, μ, μ) - G(Δτs_itr, β, β, μ, μ) - G(Δτs_itr, χ, χ, χ, χ) + G(s_itr, χ, χ, μ, μ) - G(τ_itr, χ, χ, μ, μ) + G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(s_itr, χ, χ, β, β)) + conj(G(τ_itr, χ, χ, β, β)) + conj(G(s_itr, μ, μ, β, β)) - conj(G(t_itr, μ, μ, β, β))) / (hbar2))) + (hbar2 * (- Gpp_markovian_G1(Δtτ_itr, α, α, α, χ) + Gpp_markovian_G1(Δtτ_itr, β, β, α, χ)) * (- Gp_markovian_G1(Δts_itr, α, α, χ, μ) + Gp_markovian_G1(Δτs_itr, α, α, χ, μ) + Gp_markovian_G1(Δts_itr, β, β, χ, μ) - Gp_markovian_G1(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β))) + hbar2 * (- Gp_markovian_G1(Δtτ_itr, β, β, α, α) - Gp_markovian_G1(Δts_itr, β, β, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, χ, χ) - Gp(t_itr, β, β, μ, μ) + Gp_markovian_G1(Δts_itr, β, β, μ, μ) + conj(Gp(t_itr, β, β, β, β))) * Gpp_markovian_G1(Δτs_itr, α, χ, χ, μ) + (hbar2 * Gpp_markovian_G1(Δτs_itr, α, χ, χ, μ) + (Gp_markovian_G1(Δtτ_itr, α, α, α, χ) + Gp_markovian_G1(Δτs_itr, α, χ, χ, χ) + Gp(τ_itr, α, χ, μ, μ) - Gp_markovian_G1(Δτs_itr, α, χ, μ, μ) - Gp_markovian_G1(Δtτ_itr, β, β, α, χ) - conj(Gp(τ_itr, χ, α, β, β))) * (- Gp_markovian_G1(Δts_itr, α, α, χ, μ) + Gp_markovian_G1(Δτs_itr, α, α, χ, μ) + Gp_markovian_G1(Δts_itr, β, β, χ, μ) - Gp_markovian_G1(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β)))) * (- Gp(t_itr, α, α, α, α) + Gp_markovian_G1(Δtτ_itr, α, α, α, α) + Gp_markovian_G1(Δts_itr, α, α, χ, χ) - Gp_markovian_G1(Δtτ_itr, α, α, χ, χ) + Gp(t_itr, α, α, μ, μ) - Gp_markovian_G1(Δts_itr, α, α, μ, μ) + Gp(t_itr, β, β, α, α) - conj(Gp(t_itr, β, β, β, β))) + (- hbar2 * Gpp_markovian_G1(Δts_itr, α, α, χ, μ) + hbar2 * Gpp_markovian_G1(Δts_itr, β, β, χ, μ) + (Gp_markovian_G1(Δts_itr, α, α, χ, μ) - Gp_markovian_G1(Δτs_itr, α, α, χ, μ) - Gp_markovian_G1(Δts_itr, β, β, χ, μ) + Gp_markovian_G1(Δτs_itr, χ, χ, χ, μ) + Gp(s_itr, χ, μ, μ, μ) - conj(Gp(s_itr, μ, χ, β, β))) * (- Gp_markovian_G1(Δtτ_itr, β, β, α, α) - Gp_markovian_G1(Δts_itr, β, β, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, χ, χ) - Gp(t_itr, β, β, μ, μ) + Gp_markovian_G1(Δts_itr, β, β, μ, μ) + conj(Gp(t_itr, β, β, β, β)))) * (- Gp_markovian_G1(Δtτ_itr, α, α, α, χ) - Gp_markovian_G1(Δτs_itr, α, χ, χ, χ) - Gp(τ_itr, α, χ, μ, μ) + Gp_markovian_G1(Δτs_itr, α, χ, μ, μ) + Gp_markovian_G1(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(Δts_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δτs_itr, α, α, χ, χ) - G(t_itr, α, α, μ, μ) + G(τ_itr, α, α, μ, μ) + G(Δts_itr, α, α, μ, μ) - G(Δτs_itr, α, α, μ, μ) + G(Δtτ_itr, β, β, α, α) + G(Δts_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) - G(Δts_itr, β, β, μ, μ) - G(Δτs_itr, χ, χ, χ, χ) + G(s_itr, χ, χ, μ, μ) - G(τ_itr, χ, χ, μ, μ) + G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(s_itr, χ, χ, β, β)) + conj(G(τ_itr, χ, χ, β, β)) + conj(G(s_itr, μ, μ, β, β)) - conj(G(t_itr, μ, μ, β, β))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, μ, β) * exp(((- G(s_itr, β, β, μ, μ) + G(t_itr, β, β, μ, μ) + G(s_itr, μ, μ, μ, μ) - G(t_itr, μ, μ, μ, μ) + conj(G(s_itr, β, β, β, β)) - conj(G(t_itr, β, β, β, β)) - conj(G(s_itr, μ, μ, β, β)) + conj(G(t_itr, μ, μ, β, β))) / (hbar2)))
#         )
#     end


#     Base.@noinline function kernel_L0G1_LR_markovian_G1(
#         s_itr::Int,
#         τ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((- hbar2 * Gpp_markovian_G1(Δtτ_itr, α, α, α, χ) + hbar2 * Gpp_markovian_G1(Δtτ_itr, β, β, α, χ) + (- Gp_markovian_G1(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β))) * (Gp(t_itr, α, α, α, α) - Gp_markovian_G1(Δtτ_itr, α, α, α, α) - Gp(t_itr, α, α, χ, χ) + Gp_markovian_G1(Δtτ_itr, α, α, χ, χ) - Gp(t_itr, β, β, α, α) + Gp_markovian_G1(Δtτ_itr, β, β, α, α) + Gp(t_itr, β, β, χ, χ) - Gp_markovian_G1(Δtτ_itr, β, β, χ, χ))) * (- Gp(s_itr, ν, β, χ, χ) + conj(Gp_markovian_G1(Δτs_itr, β, β, β, ν)) + conj(Gp(s_itr, β, ν, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, χ, χ, β, ν))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) - G(s_itr, β, β, χ, χ) + G(t_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(t_itr, χ, χ, χ, χ) - G(τ_itr, χ, χ, χ, χ) + G(s_itr, ν, ν, χ, χ) - G(t_itr, ν, ν, χ, χ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(t_itr, β, β, β, β)) + conj(G(τ_itr, β, β, β, β)) - conj(G(Δτs_itr, β, β, β, β)) + conj(G(s_itr, β, β, ν, ν)) - conj(G(τ_itr, β, β, ν, ν)) + conj(G(Δτs_itr, β, β, ν, ν)) + conj(G(Δτs_itr, χ, χ, β, β)) - conj(G(t_itr, χ, χ, ν, ν)) + conj(G(τ_itr, χ, χ, ν, ν)) - conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2))) + (hbar2 * (conj(Gpp_markovian_G1(Δts_itr, α, α, β, ν)) - conj(Gpp_markovian_G1(Δts_itr, β, β, β, ν))) * (- Gp_markovian_G1(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, α, χ) + conj(Gp_markovian_G1(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, χ, α, ν, ν))) + hbar2 * (- Gp_markovian_G1(Δtτ_itr, α, α, α, α) - Gp(t_itr, α, α, χ, χ) + Gp_markovian_G1(Δtτ_itr, α, α, χ, χ) + conj(Gp_markovian_G1(Δts_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, ν, ν)) - conj(Gp_markovian_G1(Δts_itr, α, α, ν, ν))) * conj(Gpp_markovian_G1(Δτs_itr, χ, α, β, ν)) - hbar2 * (- Gp_markovian_G1(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, χ, χ) + conj(Gp_markovian_G1(Δts_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp_markovian_G1(Δts_itr, β, β, ν, ν))) * conj(Gpp_markovian_G1(Δτs_itr, χ, α, β, ν)) + (hbar2 * conj(Gpp_markovian_G1(Δτs_itr, χ, α, β, ν)) + (- Gp_markovian_G1(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, α, χ) + conj(Gp_markovian_G1(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, χ, α, ν, ν))) * (Gp(s_itr, ν, β, χ, χ) + conj(Gp_markovian_G1(Δts_itr, α, α, β, ν)) - conj(Gp_markovian_G1(Δτs_itr, α, α, β, ν)) - conj(Gp_markovian_G1(Δts_itr, β, β, β, ν)) - conj(Gp(s_itr, β, ν, ν, ν)) + conj(Gp_markovian_G1(Δτs_itr, χ, χ, β, ν)))) * (Gp(t_itr, α, α, α, α) - Gp(t_itr, β, β, α, α) - conj(Gp(t_itr, α, α, β, β)) + conj(Gp(t_itr, β, β, β, β))) + (hbar2 * Gpp_markovian_G1(Δtτ_itr, α, α, α, χ) - hbar2 * Gpp_markovian_G1(Δtτ_itr, β, β, α, χ) + (- Gp_markovian_G1(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, α, χ) + conj(Gp_markovian_G1(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, χ, α, ν, ν))) * (Gp_markovian_G1(Δtτ_itr, α, α, α, α) + Gp(t_itr, α, α, χ, χ) - Gp_markovian_G1(Δtτ_itr, α, α, χ, χ) - Gp_markovian_G1(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, χ, χ) - conj(Gp_markovian_G1(Δts_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, ν, ν)) + conj(Gp_markovian_G1(Δts_itr, α, α, ν, ν)) + conj(Gp_markovian_G1(Δts_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp_markovian_G1(Δts_itr, β, β, ν, ν)))) * (- Gp(s_itr, ν, β, χ, χ) - conj(Gp_markovian_G1(Δts_itr, α, α, β, ν)) + conj(Gp_markovian_G1(Δτs_itr, α, α, β, ν)) + conj(Gp_markovian_G1(Δts_itr, β, β, β, ν)) + conj(Gp(s_itr, β, ν, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, χ, χ, β, ν)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) - G(s_itr, β, β, χ, χ) + G(t_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(t_itr, χ, χ, χ, χ) - G(τ_itr, χ, χ, χ, χ) + G(s_itr, ν, ν, χ, χ) - G(t_itr, ν, ν, χ, χ) + conj(G(Δts_itr, α, α, β, β)) - conj(G(Δτs_itr, α, α, β, β)) + conj(G(t_itr, α, α, ν, ν)) - conj(G(τ_itr, α, α, ν, ν)) - conj(G(Δts_itr, α, α, ν, ν)) + conj(G(Δτs_itr, α, α, ν, ν)) - conj(G(Δts_itr, β, β, β, β)) + conj(G(s_itr, β, β, ν, ν)) - conj(G(t_itr, β, β, ν, ν)) + conj(G(Δts_itr, β, β, ν, ν)) + conj(G(Δτs_itr, χ, χ, β, β)) - conj(G(t_itr, χ, χ, ν, ν)) + conj(G(τ_itr, χ, χ, ν, ν)) - conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, χ, ν) * exp(((G(s_itr, χ, χ, χ, χ) - G(t_itr, χ, χ, χ, χ) - G(s_itr, ν, ν, χ, χ) + G(t_itr, ν, ν, χ, χ) - conj(G(s_itr, χ, χ, ν, ν)) + conj(G(t_itr, χ, χ, ν, ν)) + conj(G(s_itr, ν, ν, ν, ν)) - conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))
#         )
#     end


#     Base.@noinline function kernel_L0G1_RL_markovian_G1(
#         s_itr::Int,
#         τ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((- hbar2 * conj(Gpp_markovian_G1(Δtτ_itr, α, α, β, χ)) + hbar2 * conj(Gpp_markovian_G1(Δtτ_itr, β, β, β, χ)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ))) * (- conj(Gp(t_itr, α, α, β, β)) + conj(Gp_markovian_G1(Δtτ_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, χ, χ)) - conj(Gp_markovian_G1(Δtτ_itr, α, α, χ, χ)) + conj(Gp(t_itr, β, β, β, β)) - conj(Gp_markovian_G1(Δtτ_itr, β, β, β, β)) - conj(Gp(t_itr, β, β, χ, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, χ, χ)))) * (- Gp_markovian_G1(Δτs_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp_markovian_G1(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(t_itr, α, α, α, α) + G(τ_itr, α, α, α, α) - G(Δτs_itr, α, α, α, α) + G(s_itr, α, α, μ, μ) - G(τ_itr, α, α, μ, μ) + G(Δτs_itr, α, α, μ, μ) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) + G(Δτs_itr, χ, χ, α, α) - G(t_itr, χ, χ, μ, μ) + G(τ_itr, χ, χ, μ, μ) - G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(Δtτ_itr, α, α, β, β)) - conj(G(s_itr, α, α, χ, χ)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(t_itr, χ, χ, χ, χ)) - conj(G(τ_itr, χ, χ, χ, χ)) + conj(G(s_itr, μ, μ, χ, χ)) - conj(G(t_itr, μ, μ, χ, χ))) / (hbar2))) + (hbar2 * (conj(Gpp_markovian_G1(Δtτ_itr, α, α, β, χ)) - conj(Gpp_markovian_G1(Δtτ_itr, β, β, β, χ))) * (- Gp_markovian_G1(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp_markovian_G1(Δts_itr, β, β, α, μ) - Gp_markovian_G1(Δτs_itr, β, β, α, μ) + Gp_markovian_G1(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) + hbar2 * (- Gp_markovian_G1(Δts_itr, α, α, α, α) - Gp(t_itr, α, α, μ, μ) + Gp_markovian_G1(Δts_itr, α, α, μ, μ) + conj(Gp_markovian_G1(Δtτ_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, χ, χ)) - conj(Gp_markovian_G1(Δtτ_itr, α, α, χ, χ))) * Gpp_markovian_G1(Δτs_itr, χ, β, α, μ) + (hbar2 * Gpp_markovian_G1(Δτs_itr, χ, β, α, μ) + (- Gp_markovian_G1(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp_markovian_G1(Δts_itr, β, β, α, μ) - Gp_markovian_G1(Δτs_itr, β, β, α, μ) + Gp_markovian_G1(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) * (Gp_markovian_G1(Δτs_itr, χ, β, α, α) + Gp(τ_itr, χ, β, μ, μ) - Gp_markovian_G1(Δτs_itr, χ, β, μ, μ) + conj(Gp_markovian_G1(Δtτ_itr, α, α, β, χ)) - conj(Gp_markovian_G1(Δtτ_itr, β, β, β, χ)) - conj(Gp(τ_itr, β, χ, χ, χ)))) * (Gp(t_itr, α, α, α, α) - Gp(t_itr, β, β, α, α) + Gp_markovian_G1(Δts_itr, β, β, α, α) + Gp(t_itr, β, β, μ, μ) - Gp_markovian_G1(Δts_itr, β, β, μ, μ) - conj(Gp(t_itr, α, α, β, β)) + conj(Gp(t_itr, β, β, β, β)) - conj(Gp_markovian_G1(Δtτ_itr, β, β, β, β)) - conj(Gp(t_itr, β, β, χ, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, χ, χ))) + (hbar2 * Gpp_markovian_G1(Δts_itr, α, α, α, μ) - hbar2 * Gpp_markovian_G1(Δts_itr, β, β, α, μ) + (Gp_markovian_G1(Δts_itr, α, α, α, α) + Gp(t_itr, α, α, μ, μ) - Gp_markovian_G1(Δts_itr, α, α, μ, μ) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, χ, χ)) + conj(Gp_markovian_G1(Δtτ_itr, α, α, χ, χ))) * (- Gp_markovian_G1(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp_markovian_G1(Δts_itr, β, β, α, μ) - Gp_markovian_G1(Δτs_itr, β, β, α, μ) + Gp_markovian_G1(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ)))) * (- Gp_markovian_G1(Δτs_itr, χ, β, α, α) - Gp(τ_itr, χ, β, μ, μ) + Gp_markovian_G1(Δτs_itr, χ, β, μ, μ) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δts_itr, α, α, α, α) + G(s_itr, α, α, μ, μ) - G(t_itr, α, α, μ, μ) + G(Δts_itr, α, α, μ, μ) + G(Δts_itr, β, β, α, α) - G(Δτs_itr, β, β, α, α) + G(t_itr, β, β, μ, μ) - G(τ_itr, β, β, μ, μ) - G(Δts_itr, β, β, μ, μ) + G(Δτs_itr, β, β, μ, μ) + G(Δτs_itr, χ, χ, α, α) - G(t_itr, χ, χ, μ, μ) + G(τ_itr, χ, χ, μ, μ) - G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(Δtτ_itr, α, α, β, β)) - conj(G(s_itr, α, α, χ, χ)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(t_itr, χ, χ, χ, χ)) - conj(G(τ_itr, χ, χ, χ, χ)) + conj(G(s_itr, μ, μ, χ, χ)) - conj(G(t_itr, μ, μ, χ, χ))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, μ, χ) * exp(((- G(s_itr, χ, χ, μ, μ) + G(t_itr, χ, χ, μ, μ) + G(s_itr, μ, μ, μ, μ) - G(t_itr, μ, μ, μ, μ) + conj(G(s_itr, χ, χ, χ, χ)) - conj(G(t_itr, χ, χ, χ, χ)) - conj(G(s_itr, μ, μ, χ, χ)) + conj(G(t_itr, μ, μ, χ, χ))) / (hbar2)))
#         )
#     end


#     Base.@noinline function kernel_L0G1_RR_markovian_G1(
#         s_itr::Int,
#         τ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((hbar2 * conj(Gpp_markovian_G1(Δtτ_itr, α, α, β, χ)) - hbar2 * conj(Gpp_markovian_G1(Δtτ_itr, β, β, β, χ)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ))) * (conj(Gp(t_itr, α, α, β, β)) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, χ, χ)) + conj(Gp_markovian_G1(Δtτ_itr, α, α, χ, χ)) - conj(Gp(t_itr, β, β, β, β)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, χ, χ)) - conj(Gp_markovian_G1(Δtτ_itr, β, β, χ, χ)))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp_markovian_G1(Δτs_itr, α, α, χ, ν)) + conj(Gp_markovian_G1(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) - G(s_itr, χ, χ, α, α) + G(τ_itr, χ, χ, α, α) + G(s_itr, ν, ν, α, α) - G(t_itr, ν, ν, α, α) + conj(G(Δtτ_itr, α, α, β, β)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(τ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) + conj(G(Δτs_itr, α, α, χ, χ)) - conj(G(t_itr, α, α, ν, ν)) + conj(G(τ_itr, α, α, ν, ν)) - conj(G(Δτs_itr, α, α, ν, ν)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) - conj(G(Δτs_itr, χ, χ, χ, χ)) + conj(G(s_itr, χ, χ, ν, ν)) - conj(G(τ_itr, χ, χ, ν, ν)) + conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2))) + (hbar2 * (- conj(Gpp_markovian_G1(Δtτ_itr, α, α, β, χ)) + conj(Gpp_markovian_G1(Δtτ_itr, β, β, β, χ))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp_markovian_G1(Δts_itr, α, α, χ, ν)) + conj(Gp_markovian_G1(Δts_itr, β, β, χ, ν)) - conj(Gp_markovian_G1(Δτs_itr, β, β, χ, ν)) + conj(Gp_markovian_G1(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν))) + hbar2 * (- Gp(t_itr, β, β, α, α) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, β)) + conj(Gp_markovian_G1(Δts_itr, β, β, χ, χ)) - conj(Gp_markovian_G1(Δtτ_itr, β, β, χ, χ)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp_markovian_G1(Δts_itr, β, β, ν, ν))) * conj(Gpp_markovian_G1(Δτs_itr, β, χ, χ, ν)) + (hbar2 * conj(Gpp_markovian_G1(Δτs_itr, β, χ, χ, ν)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, χ)) + conj(Gp_markovian_G1(Δτs_itr, β, χ, χ, χ)) + conj(Gp(τ_itr, β, χ, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, β, χ, ν, ν))) * (Gp(s_itr, ν, χ, α, α) + conj(Gp_markovian_G1(Δts_itr, α, α, χ, ν)) - conj(Gp_markovian_G1(Δts_itr, β, β, χ, ν)) + conj(Gp_markovian_G1(Δτs_itr, β, β, χ, ν)) - conj(Gp_markovian_G1(Δτs_itr, χ, χ, χ, ν)) - conj(Gp(s_itr, χ, ν, ν, ν)))) * (Gp(t_itr, β, β, α, α) + conj(Gp(t_itr, α, α, β, β)) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, β)) - conj(Gp_markovian_G1(Δts_itr, α, α, χ, χ)) + conj(Gp_markovian_G1(Δtτ_itr, α, α, χ, χ)) - conj(Gp(t_itr, α, α, ν, ν)) + conj(Gp_markovian_G1(Δts_itr, α, α, ν, ν)) - conj(Gp(t_itr, β, β, β, β))) + (- hbar2 * conj(Gpp_markovian_G1(Δts_itr, α, α, χ, ν)) + hbar2 * conj(Gpp_markovian_G1(Δts_itr, β, β, χ, ν)) + (Gp(t_itr, β, β, α, α) - conj(Gp_markovian_G1(Δtτ_itr, β, β, β, β)) - conj(Gp_markovian_G1(Δts_itr, β, β, χ, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, χ, χ)) - conj(Gp(t_itr, β, β, ν, ν)) + conj(Gp_markovian_G1(Δts_itr, β, β, ν, ν))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp_markovian_G1(Δts_itr, α, α, χ, ν)) + conj(Gp_markovian_G1(Δts_itr, β, β, χ, ν)) - conj(Gp_markovian_G1(Δτs_itr, β, β, χ, ν)) + conj(Gp_markovian_G1(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν)))) * (- Gp(τ_itr, χ, β, α, α) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, χ)) + conj(Gp_markovian_G1(Δτs_itr, β, χ, χ, χ)) + conj(Gp(τ_itr, β, χ, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, β, χ, ν, ν)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) - G(s_itr, χ, χ, α, α) + G(τ_itr, χ, χ, α, α) + G(s_itr, ν, ν, α, α) - G(t_itr, ν, ν, α, α) + conj(G(Δtτ_itr, α, α, β, β)) + conj(G(Δts_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δts_itr, α, α, ν, ν)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(Δts_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(Δτs_itr, β, β, χ, χ)) - conj(G(t_itr, β, β, ν, ν)) + conj(G(τ_itr, β, β, ν, ν)) + conj(G(Δts_itr, β, β, ν, ν)) - conj(G(Δτs_itr, β, β, ν, ν)) - conj(G(Δτs_itr, χ, χ, χ, χ)) + conj(G(s_itr, χ, χ, ν, ν)) - conj(G(τ_itr, χ, χ, ν, ν)) + conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, α, ν) * exp(((G(s_itr, α, α, α, α) - G(t_itr, α, α, α, α) - G(s_itr, ν, ν, α, α) + G(t_itr, ν, ν, α, α) - conj(G(s_itr, α, α, ν, ν)) + conj(G(t_itr, α, α, ν, ν)) + conj(G(s_itr, ν, ν, ν, ν)) - conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))
#         )
#     end


#     # Evaluate the four PL0QG1QL1P branch kernels at a single tau point.
#     # This helper is used by both the original nested quadrature and the
#     # Option-B collapsed-tau approximation below.
#     Base.@noinline function eval_L0G1_tau_kernel(
#         s_coord::Int,
#         τ_coord::Int,
#         t_eval_coord::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#     )
#         τ_kernel = 0.0 + 0.0im

#         # P L0 Q G1 Q L1 P, LL: input (μ, β), intermediate χ.
#         for χ in 1:n_sys
#             χ == α && continue
#             for μ in 1:n_sys
#                 μ == χ && continue
#                 if is_secular_pair(α, β, μ, β)
#                     τ_kernel += (
#                         effective_markovianize_PL0QG1QL1P_G1 ?
#                         kernel_L0G1_LL_markovian_G1(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, μ) :
#                         kernel_L0G1_LL(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, μ)
#                     )
#                 end
#             end
#         end

#         # P L0 Q G1 Q L1 P, RR: input (α, ν), intermediate χ.
#         for χ in 1:n_sys
#             χ == β && continue
#             for ν in 1:n_sys
#                 ν == χ && continue
#                 if is_secular_pair(α, β, α, ν)
#                     τ_kernel += (
#                         effective_markovianize_PL0QG1QL1P_G1 ?
#                         kernel_L0G1_RR_markovian_G1(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, ν) :
#                         kernel_L0G1_RR(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, ν)
#                     )
#                 end
#             end
#         end

#         # P L0 Q G1 Q L1 P, RL: input (μ, χ), left input μ, right intermediate χ.
#         for μ in 1:n_sys
#             μ == α && continue
#             for χ in 1:n_sys
#                 χ == β && continue
#                 if is_secular_pair(α, β, μ, χ)
#                     τ_kernel += (
#                         effective_markovianize_PL0QG1QL1P_G1 ?
#                         kernel_L0G1_RL_markovian_G1(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, μ) :
#                         kernel_L0G1_RL(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, μ)
#                     )
#                 end
#             end
#         end

#         # P L0 Q G1 Q L1 P, LR: input (χ, ν), left intermediate χ, right input ν.
#         for χ in 1:n_sys
#             χ == α && continue
#             for ν in 1:n_sys
#                 ν == β && continue
#                 if is_secular_pair(α, β, χ, ν)
#                     τ_kernel += (
#                         effective_markovianize_PL0QG1QL1P_G1 ?
#                         kernel_L0G1_LR_markovian_G1(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, ν) :
#                         kernel_L0G1_LR(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, ν)
#                     )
#                 end
#             end
#         end

#         return τ_kernel
#     end


#     Base.@noinline function calc__rhs_element!(
#         rhs_mat::AbstractMatrix,
#         curr_itr::Int,
#         t_eval_coord::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#     )
#         if use_population_closure && α != β
#             rhs_mat[α, β] = 0.0 + 0.0im
#             return nothing
#         end

#         rhs = include_PL0P_local ? D_L0P(t_eval_coord, α, β) * σ_local(σ_t, α, β) : (0.0 + 0.0im)

#         # -------------------------------------------------------------
#         # P L1 P local left/right mixing.
#         # Left branch:  input (μ, β) -> output (α, β).
#         # Right branch: input (α, ν) -> output (α, β).
#         # These are time-local, so σ_t and t_eval_coord are used.
#         # -------------------------------------------------------------
#         if include_PL1P_local
#             for μ in 1:n_sys
#                 μ == α && continue
#                 if is_secular_pair(α, β, μ, β)
#                     rhs += -(
#                         σ_local(σ_t, μ, β)
#                         * (
#                             Gp(t_eval_coord, α, μ, μ, μ)
#                             -conj(Gp(t_eval_coord, μ, α, β, β))
#                         ) / hbar2
#                     )
#                 end
#             end

#             for ν in 1:n_sys
#                 ν == β && continue
#                 if is_secular_pair(α, β, α, ν)
#                     rhs += (
#                         σ_local(σ_t, α, ν)
#                         * (
#                             Gp(t_eval_coord, ν, β, α, α)
#                             -conj(Gp(t_eval_coord, β, ν, ν, ν))
#                         ) / hbar2
#                     )
#                 end
#             end
#         end

#         if t_eval_coord > 1
#             integral = 0.0 + 0.0im
#             n_s_nodes = n_outer_nodes(curr_itr, t_eval_coord)

#             for s_node in 1:n_s_nodes
#                 s_coord = outer_node_coord(s_node, curr_itr, t_eval_coord)
#                 Δ_coord = t_eval_coord - s_coord + 1
#                 w_int = ∫weight_to_t(s_node, curr_itr, t_eval_coord)
#                 w_int == 0.0 && continue

#                 kernel = 0.0 + 0.0im

#                 if include_PL0QG0QL1P
#                     # P L0 Q G0 Q L1 P, L branch: input (μ, β).
#                     for μ in 1:n_sys
#                         μ == α && continue
#                         if is_secular_pair(α, β, μ, β)
#                             kernel += kernel_L0_L(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, μ)
#                         end
#                     end

#                     # P L0 Q G0 Q L1 P, R branch: input (α, ν).
#                     for ν in 1:n_sys
#                         ν == β && continue
#                         if is_secular_pair(α, β, α, ν)
#                             kernel += kernel_L0_R(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, ν)
#                         end
#                     end
#                 end

#                 if include_PL1QG0QL1P
#                     # P L1 Q G0 Q L1 P, LL: input (μ, β), intermediate χ.
#                     for χ in 1:n_sys
#                         χ == α && continue
#                         for μ in 1:n_sys
#                             μ == χ && continue
#                             if is_secular_pair(α, β, μ, β)
#                                 kernel += kernel_LL(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, χ, μ)
#                             end
#                         end
#                     end

#                     # P L1 Q G0 Q L1 P, RR: input (α, ν), intermediate χ.
#                     for χ in 1:n_sys
#                         χ == β && continue
#                         for ν in 1:n_sys
#                             ν == χ && continue
#                             if is_secular_pair(α, β, α, ν)
#                                 kernel += kernel_RR(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, χ, ν)
#                             end
#                         end
#                     end

#                     # P L1 Q G0 Q L1 P, RL: input (μ, χ), left input μ, right intermediate χ.
#                     for μ in 1:n_sys
#                         μ == α && continue
#                         for χ in 1:n_sys
#                             χ == β && continue
#                             if is_secular_pair(α, β, μ, χ)
#                                 kernel += kernel_RL(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, χ, μ)
#                             end
#                         end
#                     end

#                     # P L1 Q G0 Q L1 P, LR: input (χ, ν), left intermediate χ, right input ν.
#                     for χ in 1:n_sys
#                         χ == α && continue
#                         for ν in 1:n_sys
#                             ν == β && continue
#                             if is_secular_pair(α, β, χ, ν)
#                                 kernel += kernel_LR(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, χ, ν)
#                             end
#                         end
#                     end
#                 end

#                 if include_PL0QG1QL1P
#                     inner_kernel = 0.0 + 0.0im

#                     if collapse_tau_PL0QG1QL1P_G1
#                         # Option B: collapse the inner tau integral while keeping
#                         # the outer s-memory integral.  In grid coordinates,
#                         # physical interval length is 0.5*(q_t - q_s)*Δt in doubled q-grid.
#                         # This is a midpoint effective-kernel approximation:
#                         #   ∫_s^t dτ K(t,τ,s) ≈ (t-s) K(t,nearest_q((t+s)/2),s).
#                         τ_interval = 0.5 * Float64(t_eval_coord - s_coord) * Δt
#                         if τ_interval > 0.0
#                             τ_mid = min(max(div(t_eval_coord + s_coord + 1, 2), 1), const_Q_MAX)
#                             inner_kernel += τ_interval * eval_L0G1_tau_kernel(
#                                 s_coord,
#                                 τ_mid,
#                                 t_eval_coord,
#                                 σ_t,
#                                 α,
#                                 β,
#                             )
#                         end
#                     else
#                         n_τ_nodes = n_inner_nodes(s_coord, t_eval_coord)
#                         for τ_node in 1:n_τ_nodes
#                             τ_coord = inner_node_coord(τ_node, s_coord, t_eval_coord)
#                             w_τ = ∫weight_between_coord(τ_node, s_coord, t_eval_coord)
#                             w_τ == 0.0 && continue

#                             inner_kernel += w_τ * eval_L0G1_tau_kernel(
#                                 s_coord,
#                                 τ_coord,
#                                 t_eval_coord,
#                                 σ_t,
#                                 α,
#                                 β,
#                             )
#                         end
#                     end

#                     kernel += inner_kernel
#                 end

#                 integral += w_int * kernel
#             end

#             rhs += integral
#         end

#         rhs_mat[α, β] = rhs
#         return nothing
#     end

#     @inline function calc__rhs!(
#         rhs_mat::AbstractMatrix,
#         curr_itr::Int,
#         t_eval_coord::Int,
#         σ_t::AbstractMatrix,
#     )
#         fill!(rhs_mat, 0.0 + 0.0im)

#         n_components = n_sys * n_sys
#         heavy_memory = include_PL0QG0QL1P || include_PL1QG0QL1P || include_PL0QG1QL1P
#         min_thread_components = max(16, 4 * Threads.nthreads())
#         if use_threads && Threads.nthreads() > 1 && (heavy_memory || n_components >= min_thread_components)
#             Threads.@threads for linear_idx in 1:n_components
#                 @inbounds begin
#                     α = ((linear_idx - 1) % n_sys) + 1
#                     β = ((linear_idx - 1) ÷ n_sys) + 1
#                     calc__rhs_element!(rhs_mat, curr_itr, t_eval_coord, σ_t, α, β)
#                 end
#             end
#         else
#             @inbounds for β in 1:n_sys, α in 1:n_sys
#                 calc__rhs_element!(rhs_mat, curr_itr, t_eval_coord, σ_t, α, β)
#             end
#         end

#         return rhs_mat
#     end

#     @inline function enforce_hermiticity!(σ_next::AbstractMatrix)
#         for i in 1:n_sys
#             σ_next[i, i] = real(σ_next[i, i]) + 0.0im
#         end

#         for i in 1:(n_sys - 1)
#             for j in (i + 1):n_sys
#                 c = 0.5 * (σ_next[i, j] + conj(σ_next[j, i]))
#                 σ_next[i, j] = c
#                 σ_next[j, i] = conj(c)
#             end
#         end

#         return σ_next
#     end

#     @inline function enforce_population_closure!(σ_next::AbstractMatrix)
#         for j in 1:n_sys, i in 1:n_sys
#             if i != j
#                 σ_next[i, j] = 0.0 + 0.0im
#             else
#                 σ_next[i, i] = real(σ_next[i, i]) + 0.0im
#             end
#         end
#         return σ_next
#     end

#     if verbose && include_PL0QG1QL1P
#         if collapse_tau_PL0QG1QL1P_G1
#             @printf(stderr, "Warning: include_PL0QG1QL1P=true with collapse_tau_PL0QG1QL1P_G1=true uses midpoint tau-collapse; this reduces the G1 term to roughly O(N_t^2).\n")
#         else
#             @printf(stderr, "Warning: include_PL0QG1QL1P=true enables the nested s-tau memory term; runtime scales roughly as O(N_t^3).\n")
#         end
#     end

#     σ_stage = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k2      = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k3      = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k4      = Matrix{ComplexF64}(undef, n_sys, n_sys)

#     @inbounds for curr_itr in start_itr:(n_itr - 1)
#         if verbose && (curr_itr == start_itr || curr_itr == n_itr - 1 || ((curr_itr - start_itr) % verbose_every_f == 0))
#             @printf(
#                 stderr,
#                 "Current iteration: %6d / %6d  method=%s  threads=%d  use_threads=%s  secular=%s  PL0QG1_markovian_G1=%s  PL0QG1_collapse_tau=%s\n",
#                 curr_itr,
#                 n_itr,
#                 String(method_sym),
#                 Threads.nthreads(),
#                 string(use_threads),
#                 string(use_secular),
#                 string(effective_markovianize_PL0QG1QL1P_G1),
#                 string(collapse_tau_PL0QG1QL1P_G1),
#             )
#         end

#         σ_t    = @view σ[:, :, curr_itr]
#         σ_next = @view σ[:, :, curr_itr + 1]
#         k1     = @view σ′[:, :, curr_itr]

#         if method_sym == :euler
#             calc__rhs!(k1, curr_itr, q_full(curr_itr), σ_t)
#             @. σ_next = σ_t + Δt * k1

#         elseif method_sym == :rk2
#             calc__rhs!(k1, curr_itr, q_full(curr_itr), σ_t)
#             @. σ_stage = σ_t + 0.5 * Δt * k1
#             use_population_closure && enforce_population_closure!(σ_stage)
#             calc__rhs!(k2, curr_itr, q_half_after(curr_itr), σ_stage)
#             @. σ_next = σ_t + Δt * k2

#         elseif method_sym == :rk4
#             # print("시작0\n")
#             calc__rhs!(k1, curr_itr, q_full(curr_itr), σ_t)

#             # print("시작1\n")
#             @. σ_stage = σ_t + 0.5 * Δt * k1
#             use_population_closure && enforce_population_closure!(σ_stage)
#             calc__rhs!(k2, curr_itr, q_half_after(curr_itr), σ_stage)

#             # print("시작2\n")
#             @. σ_stage = σ_t + 0.5 * Δt * k2
#             use_population_closure && enforce_population_closure!(σ_stage)
#             calc__rhs!(k3, curr_itr, q_half_after(curr_itr), σ_stage)

#             # print("시작3\n")
#             @. σ_stage = σ_t + Δt * k3
#             use_population_closure && enforce_population_closure!(σ_stage)
#             calc__rhs!(k4, curr_itr, q_full(curr_itr + 1), σ_stage)

#             @. σ_next = σ_t + (Δt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
#             # print("끝\n")
#         end

#         if use_population_closure
#             enforce_population_closure!(σ_next)
#         elseif enforce_hermitian
#             enforce_hermiticity!(σ_next)
#         end

#         context.curr_itr = UInt64(curr_itr + 1)
#     end

#     return @view σ[:, :, Int(context.curr_itr)]
# end



# 현재 완성본!!!!
# function calc__σ_σ′_secular_core!(
#     context::RmrtContext;
#     hbar::Real = 1.0,
#     use_population_closure::Bool = false,
    
#     # P L0 P local self block.
#     include_PL0P_local::Bool = true,

#     # P L1 P local left/right mixing block.
#     include_PL1P_local::Bool = true,

#     # Combined P (L0 + L1) Q G0 Q L1 P memory block.
#     # This evaluates the previous PL0QG0QL1P boundary contribution and the
#     # previous PL1QG0QL1P off-diagonal interior contribution in one set of
#     # widened branch loops.  The first intermediate index is intentionally
#     # unrestricted; the second L1 insertion keeps its off-diagonal condition.
#     include_P_L0plusL1_QG0QL1P::Bool = true,

#     # P L0 Q G1 Q L1 P correction. This is a nested s-tau memory integral,
#     # so it is opt-in in this optimized version.
#     include_PL0QG1QL1P::Bool = false,

#     # Markovianize only the relative-time g′/g″ factors that belong to the
#     # inner G1 correction in P L0 Q G1 Q L1 P.  In this mode,
#     #     g′_{abcd}(∞) ≈ -im * Λ_{abcd},      g″_{abcd}(∞) ≈ 0,
#     # while the outer s/t Gaussian dressing factors are still evaluated from g.
#     markovianize_PL0QG1QL1P_G1::Bool = false,

#     # Stronger Option-B approximation for P L0 Q G1 Q L1 P.
#     # When true, the inner tau integral is collapsed by a midpoint rule:
#     #     ∫_s^t dτ K(t,τ,s) ≈ (t-s) K(t,nearest_q((t+s)/2),s).
#     # This keeps the outer s-memory integral, but removes the O(N_t) inner tau loop.
#     # It also implies markovianize_PL0QG1QL1P_G1=true for the G1-relative
#     # g′/g″ factors.
#     collapse_tau_PL0QG1QL1P_G1::Bool = false,

#     use_secular::Bool = false,
#     secular_tol::Float64 = 1.0e-10,
#     method::Union{Symbol,String} = :rk4,
#     use_threads::Bool = true,

#     # Half-grid handling.  This version uses doubled integer q-grid indexing:
#     #   full grid i      -> q = 2i - 1
#     #   half grid i+1/2  -> q = 2i
#     # IMPORTANT: for Patternized_g containers, we must NOT materialize a dense
#     # half-grid by looping over all (a,b,c,d), because unsupported patterns may
#     # be intentionally absent.  RK2/RK4 and collapsed-tau G1 therefore require
#     # precomputed patternized half-grid containers on context.
#     auto_prepare_half_shifted_grid::Bool = true,
#     recompute_half_shifted_grid::Bool = false,

#     enforce_hermitian::Bool = true,
#     verbose::Bool = true,
#     verbose_every::Int = 1,
# )
#     start_itr = Int(context.curr_itr)

#     n_sys = context.system.n_sys
#     n_itr = context.simulation_details.num_of_iteration
#     Δt    = context.simulation_details.Δt
#     ϵ     = context.ϵ_exci

#     σ  = context.σ
#     σ′ = context.σ′
#     g  = context.g
#     g′ = context.g′
#     g″ = context.g″

#     effective_markovianize_PL0QG1QL1P_G1 =
#         markovianize_PL0QG1QL1P_G1 || collapse_tau_PL0QG1QL1P_G1

#     if include_PL0QG1QL1P && effective_markovianize_PL0QG1QL1P_G1 && !hasproperty(context, :Λ)
#         error("markovianize/collapse PL0QG1QL1P_G1 requires context.Λ, with Λ[a,b,c,d] ≈ i*g′_{abcd}(∞).")
#     end
#     Λ = (include_PL0QG1QL1P && effective_markovianize_PL0QG1QL1P_G1) ? getfield(context, :Λ) : nothing

#     hbar_f = Float64(hbar)
#     hbar_f > 0.0 || error("hbar must be positive")
#     hbar2 = hbar_f^2
#     hbar4 = hbar_f^4
#     hbar6 = hbar_f^6

#     method_sym = Symbol(lowercase(String(method)))
#     method_sym in (:euler, :rk2, :rk4) || error(
#         "Unsupported integration method: $(method). Use :euler, :rk2, or :rk4."
#     )

#     # Avoid pathological logging settings; printing to stderr can dominate
#     # short runs, so the loop below prints only every verbose_every steps.
#     verbose_every_f = max(1, Int(verbose_every))

#     # Half-grid keyword arguments are kept for API compatibility.  This version
#     # does not materialize dense fallback half-grids, because g/g′/g″ are often
#     # Patternized containers with intentionally unsupported index patterns.
#     _ = auto_prepare_half_shifted_grid
#     _ = recompute_half_shifted_grid

#     start_itr < n_itr || return @view σ[:, :, n_itr]

#     const_ZERO_C = 0.0 + 0.0im
#     const_Q_MAX = 2 * n_itr - 1

#     @inline function ω(a::Int, b::Int)
#         @inbounds return ϵ[a] - ϵ[b]
#     end
#     @inline function Ω(a::Int, b::Int)
#         @inbounds return (ϵ[a] - ϵ[b]) / hbar_f
#     end

#     # ------------------------------------------------------------------
#     # Doubled integer q-grid utilities.
#     #   full grid i      -> q = 2i - 1
#     #   half grid i+1/2  -> q = 2i
#     # Relative times are also q-indices:
#     #   qΔ = q_t - q_s + 1
#     # so full/half relative times close without Float64 coordinates.
#     # ------------------------------------------------------------------

#     @inline q_full(i::Int) = 2 * i - 1
#     @inline q_half_after(i::Int) = 2 * i
#     @inline is_full_q(q::Int) = isodd(q)
#     @inline full_index_from_q(q::Int) = (q + 1) >>> 1
#     @inline half_index_from_q(q::Int) = q >>> 1

#     @inline function check_q(q::Int)
#         (1 <= q <= const_Q_MAX) || error("q-grid index out of range: $(q), allowed [1, $(const_Q_MAX)]")
#         return q
#     end

#     function _first_existing_property(obj, names::Tuple)
#         for name in names
#             if hasproperty(obj, name)
#                 return getfield(obj, name)
#             end
#         end
#         return nothing
#     end

#     # Try to use the user's precomputed half-grid containers if they exist.
#     # The symbol list intentionally accepts several common naming conventions.
#     g_half = _first_existing_property(context, (
#         :g_half,
#         :g_half_shifted,
#         :g_shifted_half,
#         :g_mid,
#         :g_midpoint,
#         Symbol("g__half"),
#         Symbol("g_half_grid"),
#         Symbol("g_half_shifted_grid"),
#     ))
#     gp_half = _first_existing_property(context, (
#         Symbol("g′_half"),
#         Symbol("g′_half_shifted"),
#         Symbol("g′_shifted_half"),
#         Symbol("g′_mid"),
#         Symbol("g′_midpoint"),
#         Symbol("gp_half"),
#         Symbol("gp_half_shifted"),
#         Symbol("g_prime_half"),
#         Symbol("g_prime_half_shifted"),
#     ))
#     gpp_half = _first_existing_property(context, (
#         Symbol("g″_half"),
#         Symbol("g″_half_shifted"),
#         Symbol("g″_shifted_half"),
#         Symbol("g″_mid"),
#         Symbol("g″_midpoint"),
#         Symbol("gpp_half"),
#         Symbol("gpp_half_shifted"),
#         Symbol("g_doubleprime_half"),
#         Symbol("g_doubleprime_half_shifted"),
#     ))

#     needs_half_grid = (method_sym in (:rk2, :rk4)) || (include_PL0QG1QL1P && collapse_tau_PL0QG1QL1P_G1)

#     if needs_half_grid && (g_half === nothing || gp_half === nothing || gpp_half === nothing)
#         missing = String[]
#         g_half === nothing && push!(missing, "g half-grid")
#         gp_half === nothing && push!(missing, "g′ half-grid")
#         gpp_half === nothing && push!(missing, "g″ half-grid")
#         error(
#             "q-grid mode requires precomputed patternized half-grid containers for " *
#             join(missing, ", ") *
#             ". Do not dense-materialize Patternized_g; prepare and store half-grid " *
#             "containers on context, or use method=:euler with collapse_tau_PL0QG1QL1P_G1=false."
#         )
#     end

#     @inline function _missing_half_grid_error(name::String)
#         error(
#             "Attempted to access " * name * " at a half-grid q index, but the corresponding " *
#             "patternized half-grid container was not found on context."
#         )
#     end

#     @inline function G(q::Int, a::Int, b::Int, c::Int, d::Int)
#         check_q(q)
#         if isodd(q)
#             idx = full_index_from_q(q)
#             @inbounds return g[idx, a, b, c, d]
#         else
#             g_half === nothing && _missing_half_grid_error("g")
#             idx = half_index_from_q(q)
#             @inbounds return g_half[idx, a, b, c, d]
#         end
#     end

#     @inline function Gp(q::Int, a::Int, b::Int, c::Int, d::Int)
#         check_q(q)
#         if isodd(q)
#             idx = full_index_from_q(q)
#             @inbounds return g′[idx, a, b, c, d]
#         else
#             gp_half === nothing && _missing_half_grid_error("g′")
#             idx = half_index_from_q(q)
#             @inbounds return gp_half[idx, a, b, c, d]
#         end
#     end

#     @inline function Gpp(q::Int, a::Int, b::Int, c::Int, d::Int)
#         check_q(q)
#         if isodd(q)
#             idx = full_index_from_q(q)
#             @inbounds return g″[idx, a, b, c, d]
#         else
#             gpp_half === nothing && _missing_half_grid_error("g″")
#             idx = half_index_from_q(q)
#             @inbounds return gpp_half[idx, a, b, c, d]
#         end
#     end

#     # Pure Hamiltonian phase cache over doubled relative q-grid.
#     phase_cache = Array{ComplexF64}(undef, const_Q_MAX, n_sys, n_sys)
#     @inbounds for qΔ in 1:const_Q_MAX, a in 1:n_sys, b in 1:n_sys
#         phase_cache[qΔ, a, b] = exp(-1.0im * (ϵ[a] - ϵ[b]) * (0.5 * (Float64(qΔ) - 1.0) * Δt) / hbar_f)
#     end

#     @inline function n_outer_nodes(curr_itr::Int, q_t::Int)
#         q_now = q_full(curr_itr)
#         return q_t > q_now ? curr_itr + 1 : curr_itr
#     end

#     @inline function outer_node_coord(node_idx::Int, curr_itr::Int, q_t::Int)
#         return node_idx <= curr_itr ? q_full(node_idx) : q_t
#     end

#     @inline function ∫weight_to_t(node_idx::Int, curr_itr::Int, q_t::Int)
#         n_nodes = n_outer_nodes(curr_itr, q_t)
#         n_nodes <= 1 && return 0.0

#         q = outer_node_coord(node_idx, curr_itr, q_t)
#         if node_idx == 1
#             q_next = outer_node_coord(node_idx + 1, curr_itr, q_t)
#             return 0.25 * Δt * Float64(q_next - q)
#         elseif node_idx == n_nodes
#             q_prev = outer_node_coord(node_idx - 1, curr_itr, q_t)
#             return 0.25 * Δt * Float64(q - q_prev)
#         else
#             q_prev = outer_node_coord(node_idx - 1, curr_itr, q_t)
#             q_next = outer_node_coord(node_idx + 1, curr_itr, q_t)
#             return 0.25 * Δt * Float64(q_next - q_prev)
#         end
#     end

#     @inline function n_inner_nodes(q_s::Int, q_t::Int)
#         q_t <= q_s && return 1
#         q_floor_full = isodd(q_t) ? q_t : q_t - 1
#         n_full = ((q_floor_full - q_s) >>> 1) + 1
#         return n_full + (iseven(q_t) ? 1 : 0)
#     end

#     @inline function inner_node_coord(node_idx::Int, q_s::Int, q_t::Int)
#         q_floor_full = isodd(q_t) ? q_t : q_t - 1
#         n_full = ((q_floor_full - q_s) >>> 1) + 1
#         return node_idx <= n_full ? q_s + 2 * (node_idx - 1) : q_t
#     end

#     @inline function ∫weight_between_coord(node_idx::Int, q_s::Int, q_t::Int)
#         q_t <= q_s && return 0.0
#         n_nodes = n_inner_nodes(q_s, q_t)
#         n_nodes <= 1 && return 0.0

#         q = inner_node_coord(node_idx, q_s, q_t)
#         if node_idx == 1
#             q_next = inner_node_coord(node_idx + 1, q_s, q_t)
#             return 0.25 * Δt * Float64(q_next - q)
#         elseif node_idx == n_nodes
#             q_prev = inner_node_coord(node_idx - 1, q_s, q_t)
#             return 0.25 * Δt * Float64(q - q_prev)
#         else
#             q_prev = inner_node_coord(node_idx - 1, q_s, q_t)
#             q_next = inner_node_coord(node_idx + 1, q_s, q_t)
#             return 0.25 * Δt * Float64(q_next - q_prev)
#         end
#     end

#     @inline function Δtime(qΔ::Int)
#         return 0.5 * (Float64(qΔ) - 1.0) * Δt
#     end

#     @inline function σ_local(σ_t::AbstractMatrix, a::Int, b::Int)
#         return (use_population_closure && a != b) ? const_ZERO_C : σ_t[a, b]
#     end

#     @inline function σ_mem(
#         σ_t::AbstractMatrix,
#         q_s::Int,
#         q_t::Int,
#         a::Int,
#         b::Int,
#     )
#         if use_population_closure && a != b
#             return const_ZERO_C
#         end
#         if q_s == q_t
#             return σ_t[a, b]
#         end
#         isodd(q_s) || error("non-endpoint memory q index must be a stored full-grid point, got q=$(q_s)")
#         s_idx = full_index_from_q(q_s)
#         @inbounds return σ[a, b, s_idx]
#     end

#     @inline function phase_exp(from_a::Int, from_b::Int, qΔ::Int)
#         check_q(qΔ)
#         @inbounds return phase_cache[qΔ, from_a, from_b]
#     end

#     # Markovian asymptote used only by the optional inner-G1 version of
#     # P L0 Q G1 Q L1 P.  Because Λ stores i*g′(∞), the replacement is
#     # g′(∞) = -im*Λ and g″(∞) = 0.
#     @inline Gp_markovian_G1(Δ_coord::Int, a::Int, b::Int, c::Int, d::Int) =
#         -1.0im * Λ[a, b, c, d]

#     @inline Gpp_markovian_G1(Δ_coord::Int, a::Int, b::Int, c::Int, d::Int) =
#         const_ZERO_C


#     @inline function is_secular_pair(
#         out_a::Int,
#         out_b::Int,
#         in_a::Int,
#         in_b::Int,
#     )
#         if !use_secular
#             return true
#         end
#         return abs(Ω(out_a, out_b) - Ω(in_a, in_b)) <= secular_tol
#     end

#     @inline function D_L0P(
#         itr::Int,
#         α::Int,
#         β::Int,
#     )
#         return (
#             -1.0im * ω(α, β) / hbar_f
#             +(
#                 -Gp(itr, α, α, α, α)
#                 +conj(Gp(itr, α, α, β, β))
#                 +Gp(itr, β, β, α, α)
#                 -conj(Gp(itr, β, β, β, β))
#             ) / hbar2
#         )
#     end

#     # ---------------------------------------------------------------------
#     # Explicit PL1 Q G0 Q L1 P, g-closed branch prefactors and coefficients.
#     # Branch naming follows the trace side of the two L1 insertions.
#     # All formulas are written with explicit hbar factors.  With hbar=1 this
#     # reduces to the previous internal convention.
#     # ---------------------------------------------------------------------

#     @inline function exponent_LL(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         return (
#             G(Δ_itr, β, β, χ, χ)
#             -G(s_itr, β, β, μ, μ)
#             +G(t_itr, β, β, μ, μ)
#             -G(Δ_itr, β, β, μ, μ)
#             -G(Δ_itr, χ, χ, χ, χ)
#             +G(s_itr, χ, χ, μ, μ)
#             -G(t_itr, χ, χ, μ, μ)
#             +G(Δ_itr, χ, χ, μ, μ)
#             +conj(G(s_itr, β, β, β, β))
#             -conj(G(t_itr, β, β, β, β))
#             -conj(G(s_itr, χ, χ, β, β))
#             +conj(G(t_itr, χ, χ, β, β))
#         )
#     end

#     @inline function coef_LL(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         left_bracket = (
#             Gp(t_itr, α, χ, χ, χ)
#             -Gp(Δ_itr, α, χ, χ, χ)
#             -Gp(t_itr, α, χ, μ, μ)
#             +Gp(Δ_itr, α, χ, μ, μ)
#         )

#         right_bracket = (
#             Gp(Δ_itr, β, β, χ, μ)
#             -Gp(Δ_itr, χ, χ, χ, μ)
#             -Gp(s_itr, χ, μ, μ, μ)
#             +conj(Gp(s_itr, μ, χ, β, β))
#         )

#         return hbar2 * Gpp(Δ_itr, α, χ, χ, μ) - left_bracket * right_bracket
#     end

#     @inline function kernel_LL(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         return -(
#             σ_mem(σ_t, s_itr, t_itr, μ, β)
#             * phase_exp(χ, β, Δ_itr)
#             * exp(exponent_LL(s_itr, Δ_itr, t_itr, α, β, χ, μ) / hbar2)
#             * coef_LL(s_itr, Δ_itr, t_itr, α, β, χ, μ) / hbar4
#         )
#     end

#     @inline function exponent_LR(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         return (
#             -G(s_itr, β, β, χ, χ)
#             +G(t_itr, β, β, χ, χ)
#             +G(s_itr, χ, χ, χ, χ)
#             -G(t_itr, χ, χ, χ, χ)
#             -conj(G(Δ_itr, β, β, β, β))
#             +conj(G(s_itr, β, β, ν, ν))
#             -conj(G(t_itr, β, β, ν, ν))
#             +conj(G(Δ_itr, β, β, ν, ν))
#             +conj(G(Δ_itr, χ, χ, β, β))
#             -conj(G(s_itr, χ, χ, ν, ν))
#             +conj(G(t_itr, χ, χ, ν, ν))
#             -conj(G(Δ_itr, χ, χ, ν, ν))
#         )
#     end

#     @inline function coef_LR(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         right_bracket = (
#             Gp(s_itr, ν, β, χ, χ)
#             -conj(Gp(Δ_itr, β, β, β, ν))
#             -conj(Gp(s_itr, β, ν, ν, ν))
#             +conj(Gp(Δ_itr, χ, χ, β, ν))
#         )

#         left_bracket = (
#             conj(Gp(t_itr, χ, α, β, β))
#             -conj(Gp(Δ_itr, χ, α, β, β))
#             -conj(Gp(t_itr, χ, α, ν, ν))
#             +conj(Gp(Δ_itr, χ, α, ν, ν))
#         )

#         return hbar2 * conj(Gpp(Δ_itr, χ, α, β, ν)) - right_bracket * left_bracket
#     end

#     @inline function kernel_LR(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         return (
#             σ_mem(σ_t, s_itr, t_itr, χ, ν)
#             * phase_exp(χ, β, Δ_itr)
#             * exp(exponent_LR(s_itr, Δ_itr, t_itr, α, β, χ, ν) / hbar2)
#             * coef_LR(s_itr, Δ_itr, t_itr, α, β, χ, ν) / hbar4
#         )
#     end

#     @inline function exponent_RL(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         return (
#             -G(Δ_itr, α, α, α, α)
#             +G(s_itr, α, α, μ, μ)
#             -G(t_itr, α, α, μ, μ)
#             +G(Δ_itr, α, α, μ, μ)
#             +G(Δ_itr, χ, χ, α, α)
#             -G(s_itr, χ, χ, μ, μ)
#             +G(t_itr, χ, χ, μ, μ)
#             -G(Δ_itr, χ, χ, μ, μ)
#             -conj(G(s_itr, α, α, χ, χ))
#             +conj(G(t_itr, α, α, χ, χ))
#             +conj(G(s_itr, χ, χ, χ, χ))
#             -conj(G(t_itr, χ, χ, χ, χ))
#         )
#     end

#     @inline function coef_RL(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         left_bracket = (
#             Gp(Δ_itr, α, α, α, μ)
#             +Gp(s_itr, α, μ, μ, μ)
#             -Gp(Δ_itr, χ, χ, α, μ)
#             -conj(Gp(s_itr, μ, α, χ, χ))
#         )

#         right_bracket = (
#             Gp(t_itr, χ, β, α, α)
#             -Gp(Δ_itr, χ, β, α, α)
#             -Gp(t_itr, χ, β, μ, μ)
#             +Gp(Δ_itr, χ, β, μ, μ)
#         )

#         return hbar2 * Gpp(Δ_itr, χ, β, α, μ) + left_bracket * right_bracket
#     end

#     @inline function kernel_RL(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         return (
#             σ_mem(σ_t, s_itr, t_itr, μ, χ)
#             * phase_exp(α, χ, Δ_itr)
#             * exp(exponent_RL(s_itr, Δ_itr, t_itr, α, β, χ, μ) / hbar2)
#             * coef_RL(s_itr, Δ_itr, t_itr, α, β, χ, μ) / hbar4
#         )
#     end

#     @inline function exponent_RR(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         # This is the g-part after distributing the leading minus sign in
#         # exp(-(i hbar Δε Δ + G_raw) / hbar^2).
#         return (
#             G(s_itr, α, α, α, α)
#             -G(t_itr, α, α, α, α)
#             -G(s_itr, χ, χ, α, α)
#             +G(t_itr, χ, χ, α, α)
#             +conj(G(Δ_itr, α, α, χ, χ))
#             -conj(G(s_itr, α, α, ν, ν))
#             +conj(G(t_itr, α, α, ν, ν))
#             -conj(G(Δ_itr, α, α, ν, ν))
#             -conj(G(Δ_itr, χ, χ, χ, χ))
#             +conj(G(s_itr, χ, χ, ν, ν))
#             -conj(G(t_itr, χ, χ, ν, ν))
#             +conj(G(Δ_itr, χ, χ, ν, ν))
#         )
#     end

#     @inline function coef_RR(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         right_bracket = (
#             Gp(s_itr, ν, χ, α, α)
#             +conj(Gp(Δ_itr, α, α, χ, ν))
#             -conj(Gp(Δ_itr, χ, χ, χ, ν))
#             -conj(Gp(s_itr, χ, ν, ν, ν))
#         )

#         left_bracket = (
#             conj(Gp(t_itr, β, χ, χ, χ))
#             -conj(Gp(Δ_itr, β, χ, χ, χ))
#             -conj(Gp(t_itr, β, χ, ν, ν))
#             +conj(Gp(Δ_itr, β, χ, ν, ν))
#         )

#         return hbar2 * conj(Gpp(Δ_itr, β, χ, χ, ν)) - right_bracket * left_bracket
#     end

#     @inline function kernel_RR(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         return -(
#             σ_mem(σ_t, s_itr, t_itr, α, ν)
#             * phase_exp(α, χ, Δ_itr)
#             * exp(exponent_RR(s_itr, Δ_itr, t_itr, α, β, χ, ν) / hbar2)
#             * coef_RR(s_itr, Δ_itr, t_itr, α, β, χ, ν) / hbar4
#         )
#     end


#     # ---------------------------------------------------------------------
#     # Explicit P L0 Q G0 Q L1 P, g-closed branch prefactors and coefficients.
#     # Branch L : input (μ, β) -> output (α, β)
#     # Branch R : input (α, ν) -> output (α, β)
#     # ---------------------------------------------------------------------

#     @inline function exponent_L0_L(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         μ::Int,
#     )
#         # exp[-(i hbar ω_{αβ} Δ + G_L) / hbar^2]
#         # = phase_exp(α, β, Δ) * exp[-G_L / hbar^2]
#         return (
#             -G(Δ_itr, α, α, α, α)
#             +G(s_itr, α, α, μ, μ)
#             -G(t_itr, α, α, μ, μ)
#             +G(Δ_itr, α, α, μ, μ)
#             +G(Δ_itr, β, β, α, α)
#             -G(s_itr, β, β, μ, μ)
#             +G(t_itr, β, β, μ, μ)
#             -G(Δ_itr, β, β, μ, μ)
#             -conj(G(s_itr, α, α, β, β))
#             +conj(G(t_itr, α, α, β, β))
#             +conj(G(s_itr, β, β, β, β))
#             -conj(G(t_itr, β, β, β, β))
#         )
#     end

#     @inline function coef_L0_L(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         μ::Int,
#     )
#         left_bracket = (
#             Gp(Δ_itr, α, α, α, μ)
#             +Gp(s_itr, α, μ, μ, μ)
#             -Gp(Δ_itr, β, β, α, μ)
#             -conj(Gp(s_itr, μ, α, β, β))
#         )

#         right_bracket = (
#             Gp(t_itr, α, α, α, α)
#             -Gp(Δ_itr, α, α, α, α)
#             -Gp(t_itr, α, α, μ, μ)
#             +Gp(Δ_itr, α, α, μ, μ)
#             -Gp(t_itr, β, β, α, α)
#             +Gp(Δ_itr, β, β, α, α)
#             +Gp(t_itr, β, β, μ, μ)
#             -Gp(Δ_itr, β, β, μ, μ)
#         )

#         return (
#             hbar2 * Gpp(Δ_itr, α, α, α, μ)
#             -hbar2 * Gpp(Δ_itr, β, β, α, μ)
#             +left_bracket * right_bracket
#         )
#     end

#     @inline function kernel_L0_L(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         μ::Int,
#     )
#         return -(
#             σ_mem(σ_t, s_itr, t_itr, μ, β)
#             * phase_exp(α, β, Δ_itr)
#             * exp(exponent_L0_L(s_itr, Δ_itr, t_itr, α, β, μ) / hbar2)
#             * coef_L0_L(s_itr, Δ_itr, t_itr, α, β, μ) / hbar4
#         )
#     end

#     @inline function exponent_L0_R(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         ν::Int,
#     )
#         # exp[-(i hbar ω_{αβ} Δ + G_R) / hbar^2]
#         # = phase_exp(α, β, Δ) * exp[-G_R / hbar^2]
#         return (
#             +G(s_itr, α, α, α, α)
#             -G(t_itr, α, α, α, α)
#             -G(s_itr, β, β, α, α)
#             +G(t_itr, β, β, α, α)
#             +conj(G(Δ_itr, α, α, β, β))
#             -conj(G(s_itr, α, α, ν, ν))
#             +conj(G(t_itr, α, α, ν, ν))
#             -conj(G(Δ_itr, α, α, ν, ν))
#             -conj(G(Δ_itr, β, β, β, β))
#             +conj(G(s_itr, β, β, ν, ν))
#             -conj(G(t_itr, β, β, ν, ν))
#             +conj(G(Δ_itr, β, β, ν, ν))
#         )
#     end

#     @inline function coef_L0_R(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         ν::Int,
#     )
#         left_bracket = (
#             Gp(s_itr, ν, β, α, α)
#             +conj(Gp(Δ_itr, α, α, β, ν))
#             -conj(Gp(Δ_itr, β, β, β, ν))
#             -conj(Gp(s_itr, β, ν, ν, ν))
#         )

#         right_bracket = (
#             conj(Gp(t_itr, α, α, β, β))
#             -conj(Gp(Δ_itr, α, α, β, β))
#             -conj(Gp(t_itr, α, α, ν, ν))
#             +conj(Gp(Δ_itr, α, α, ν, ν))
#             -conj(Gp(t_itr, β, β, β, β))
#             +conj(Gp(Δ_itr, β, β, β, β))
#             +conj(Gp(t_itr, β, β, ν, ν))
#             -conj(Gp(Δ_itr, β, β, ν, ν))
#         )

#         return (
#             -hbar2 * conj(Gpp(Δ_itr, α, α, β, ν))
#             +hbar2 * conj(Gpp(Δ_itr, β, β, β, ν))
#             +left_bracket * right_bracket
#         )
#     end

#     @inline function kernel_L0_R(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         ν::Int,
#     )
#         return -(
#             σ_mem(σ_t, s_itr, t_itr, α, ν)
#             * phase_exp(α, β, Δ_itr)
#             * exp(exponent_L0_R(s_itr, Δ_itr, t_itr, α, β, ν) / hbar2)
#             * coef_L0_R(s_itr, Δ_itr, t_itr, α, β, ν) / hbar4
#         )
#     end



#     # ---------------------------------------------------------------------
#     # Explicit P L0 Q G1 Q L1 P, g-closed 2-time branch kernels.
#     # The extra variable τ satisfies s <= τ <= t.  Each branch is the
#     # direct g-closed expression from I^{*,L0G1L1,g}_{αβ}(t,τ,s).
#     # ---------------------------------------------------------------------

#     Base.@noinline function kernel_L0G1_LL(
#         s_itr::Int,
#         τ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((hbar2 * Gpp(Δtτ_itr, α, α, α, χ) - hbar2 * Gpp(Δtτ_itr, β, β, α, χ) + (- Gp(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β))) * (- Gp(t_itr, α, α, α, α) + Gp(Δtτ_itr, α, α, α, α) + Gp(t_itr, α, α, χ, χ) - Gp(Δtτ_itr, α, α, χ, χ) + Gp(t_itr, β, β, α, α) - Gp(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp(Δtτ_itr, β, β, χ, χ))) * (Gp(Δτs_itr, β, β, χ, μ) - Gp(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) + G(t_itr, β, β, χ, χ) - G(τ_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(Δτs_itr, β, β, χ, χ) - G(t_itr, β, β, μ, μ) + G(τ_itr, β, β, μ, μ) - G(Δτs_itr, β, β, μ, μ) - G(Δτs_itr, χ, χ, χ, χ) + G(s_itr, χ, χ, μ, μ) - G(τ_itr, χ, χ, μ, μ) + G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(s_itr, χ, χ, β, β)) + conj(G(τ_itr, χ, χ, β, β)) + conj(G(s_itr, μ, μ, β, β)) - conj(G(t_itr, μ, μ, β, β))) / (hbar2))) + (hbar2 * (- Gpp(Δtτ_itr, α, α, α, χ) + Gpp(Δtτ_itr, β, β, α, χ)) * (- Gp(Δts_itr, α, α, χ, μ) + Gp(Δτs_itr, α, α, χ, μ) + Gp(Δts_itr, β, β, χ, μ) - Gp(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β))) + hbar2 * (- Gp(Δtτ_itr, β, β, α, α) - Gp(Δts_itr, β, β, χ, χ) + Gp(Δtτ_itr, β, β, χ, χ) - Gp(t_itr, β, β, μ, μ) + Gp(Δts_itr, β, β, μ, μ) + conj(Gp(t_itr, β, β, β, β))) * Gpp(Δτs_itr, α, χ, χ, μ) + (hbar2 * Gpp(Δτs_itr, α, χ, χ, μ) + (Gp(Δtτ_itr, α, α, α, χ) + Gp(Δτs_itr, α, χ, χ, χ) + Gp(τ_itr, α, χ, μ, μ) - Gp(Δτs_itr, α, χ, μ, μ) - Gp(Δtτ_itr, β, β, α, χ) - conj(Gp(τ_itr, χ, α, β, β))) * (- Gp(Δts_itr, α, α, χ, μ) + Gp(Δτs_itr, α, α, χ, μ) + Gp(Δts_itr, β, β, χ, μ) - Gp(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β)))) * (- Gp(t_itr, α, α, α, α) + Gp(Δtτ_itr, α, α, α, α) + Gp(Δts_itr, α, α, χ, χ) - Gp(Δtτ_itr, α, α, χ, χ) + Gp(t_itr, α, α, μ, μ) - Gp(Δts_itr, α, α, μ, μ) + Gp(t_itr, β, β, α, α) - conj(Gp(t_itr, β, β, β, β))) + (- hbar2 * Gpp(Δts_itr, α, α, χ, μ) + hbar2 * Gpp(Δts_itr, β, β, χ, μ) + (Gp(Δts_itr, α, α, χ, μ) - Gp(Δτs_itr, α, α, χ, μ) - Gp(Δts_itr, β, β, χ, μ) + Gp(Δτs_itr, χ, χ, χ, μ) + Gp(s_itr, χ, μ, μ, μ) - conj(Gp(s_itr, μ, χ, β, β))) * (- Gp(Δtτ_itr, β, β, α, α) - Gp(Δts_itr, β, β, χ, χ) + Gp(Δtτ_itr, β, β, χ, χ) - Gp(t_itr, β, β, μ, μ) + Gp(Δts_itr, β, β, μ, μ) + conj(Gp(t_itr, β, β, β, β)))) * (- Gp(Δtτ_itr, α, α, α, χ) - Gp(Δτs_itr, α, χ, χ, χ) - Gp(τ_itr, α, χ, μ, μ) + Gp(Δτs_itr, α, χ, μ, μ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(Δts_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δτs_itr, α, α, χ, χ) - G(t_itr, α, α, μ, μ) + G(τ_itr, α, α, μ, μ) + G(Δts_itr, α, α, μ, μ) - G(Δτs_itr, α, α, μ, μ) + G(Δtτ_itr, β, β, α, α) + G(Δts_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) - G(Δts_itr, β, β, μ, μ) - G(Δτs_itr, χ, χ, χ, χ) + G(s_itr, χ, χ, μ, μ) - G(τ_itr, χ, χ, μ, μ) + G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(s_itr, χ, χ, β, β)) + conj(G(τ_itr, χ, χ, β, β)) + conj(G(s_itr, μ, μ, β, β)) - conj(G(t_itr, μ, μ, β, β))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, μ, β) * exp(((- G(s_itr, β, β, μ, μ) + G(t_itr, β, β, μ, μ) + G(s_itr, μ, μ, μ, μ) - G(t_itr, μ, μ, μ, μ) + conj(G(s_itr, β, β, β, β)) - conj(G(t_itr, β, β, β, β)) - conj(G(s_itr, μ, μ, β, β)) + conj(G(t_itr, μ, μ, β, β))) / (hbar2)))
#         )
#     end

#     Base.@noinline function kernel_L0G1_LR(
#         s_itr::Int,
#         τ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((- hbar2 * Gpp(Δtτ_itr, α, α, α, χ) + hbar2 * Gpp(Δtτ_itr, β, β, α, χ) + (- Gp(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β))) * (Gp(t_itr, α, α, α, α) - Gp(Δtτ_itr, α, α, α, α) - Gp(t_itr, α, α, χ, χ) + Gp(Δtτ_itr, α, α, χ, χ) - Gp(t_itr, β, β, α, α) + Gp(Δtτ_itr, β, β, α, α) + Gp(t_itr, β, β, χ, χ) - Gp(Δtτ_itr, β, β, χ, χ))) * (- Gp(s_itr, ν, β, χ, χ) + conj(Gp(Δτs_itr, β, β, β, ν)) + conj(Gp(s_itr, β, ν, ν, ν)) - conj(Gp(Δτs_itr, χ, χ, β, ν))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) - G(s_itr, β, β, χ, χ) + G(t_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(t_itr, χ, χ, χ, χ) - G(τ_itr, χ, χ, χ, χ) + G(s_itr, ν, ν, χ, χ) - G(t_itr, ν, ν, χ, χ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(t_itr, β, β, β, β)) + conj(G(τ_itr, β, β, β, β)) - conj(G(Δτs_itr, β, β, β, β)) + conj(G(s_itr, β, β, ν, ν)) - conj(G(τ_itr, β, β, ν, ν)) + conj(G(Δτs_itr, β, β, ν, ν)) + conj(G(Δτs_itr, χ, χ, β, β)) - conj(G(t_itr, χ, χ, ν, ν)) + conj(G(τ_itr, χ, χ, ν, ν)) - conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2))) + (hbar2 * (conj(Gpp(Δts_itr, α, α, β, ν)) - conj(Gpp(Δts_itr, β, β, β, ν))) * (- Gp(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp(Δτs_itr, χ, α, ν, ν))) + hbar2 * (- Gp(Δtτ_itr, α, α, α, α) - Gp(t_itr, α, α, χ, χ) + Gp(Δtτ_itr, α, α, χ, χ) + conj(Gp(Δts_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, ν, ν)) - conj(Gp(Δts_itr, α, α, ν, ν))) * conj(Gpp(Δτs_itr, χ, α, β, ν)) - hbar2 * (- Gp(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp(Δtτ_itr, β, β, χ, χ) + conj(Gp(Δts_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp(Δts_itr, β, β, ν, ν))) * conj(Gpp(Δτs_itr, χ, α, β, ν)) + (hbar2 * conj(Gpp(Δτs_itr, χ, α, β, ν)) + (- Gp(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp(Δτs_itr, χ, α, ν, ν))) * (Gp(s_itr, ν, β, χ, χ) + conj(Gp(Δts_itr, α, α, β, ν)) - conj(Gp(Δτs_itr, α, α, β, ν)) - conj(Gp(Δts_itr, β, β, β, ν)) - conj(Gp(s_itr, β, ν, ν, ν)) + conj(Gp(Δτs_itr, χ, χ, β, ν)))) * (Gp(t_itr, α, α, α, α) - Gp(t_itr, β, β, α, α) - conj(Gp(t_itr, α, α, β, β)) + conj(Gp(t_itr, β, β, β, β))) + (hbar2 * Gpp(Δtτ_itr, α, α, α, χ) - hbar2 * Gpp(Δtτ_itr, β, β, α, χ) + (- Gp(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp(Δτs_itr, χ, α, ν, ν))) * (Gp(Δtτ_itr, α, α, α, α) + Gp(t_itr, α, α, χ, χ) - Gp(Δtτ_itr, α, α, χ, χ) - Gp(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp(Δtτ_itr, β, β, χ, χ) - conj(Gp(Δts_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, ν, ν)) + conj(Gp(Δts_itr, α, α, ν, ν)) + conj(Gp(Δts_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp(Δts_itr, β, β, ν, ν)))) * (- Gp(s_itr, ν, β, χ, χ) - conj(Gp(Δts_itr, α, α, β, ν)) + conj(Gp(Δτs_itr, α, α, β, ν)) + conj(Gp(Δts_itr, β, β, β, ν)) + conj(Gp(s_itr, β, ν, ν, ν)) - conj(Gp(Δτs_itr, χ, χ, β, ν)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) - G(s_itr, β, β, χ, χ) + G(t_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(t_itr, χ, χ, χ, χ) - G(τ_itr, χ, χ, χ, χ) + G(s_itr, ν, ν, χ, χ) - G(t_itr, ν, ν, χ, χ) + conj(G(Δts_itr, α, α, β, β)) - conj(G(Δτs_itr, α, α, β, β)) + conj(G(t_itr, α, α, ν, ν)) - conj(G(τ_itr, α, α, ν, ν)) - conj(G(Δts_itr, α, α, ν, ν)) + conj(G(Δτs_itr, α, α, ν, ν)) - conj(G(Δts_itr, β, β, β, β)) + conj(G(s_itr, β, β, ν, ν)) - conj(G(t_itr, β, β, ν, ν)) + conj(G(Δts_itr, β, β, ν, ν)) + conj(G(Δτs_itr, χ, χ, β, β)) - conj(G(t_itr, χ, χ, ν, ν)) + conj(G(τ_itr, χ, χ, ν, ν)) - conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, χ, ν) * exp(((G(s_itr, χ, χ, χ, χ) - G(t_itr, χ, χ, χ, χ) - G(s_itr, ν, ν, χ, χ) + G(t_itr, ν, ν, χ, χ) - conj(G(s_itr, χ, χ, ν, ν)) + conj(G(t_itr, χ, χ, ν, ν)) + conj(G(s_itr, ν, ν, ν, ν)) - conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))
#         )
#     end

#     Base.@noinline function kernel_L0G1_RL(
#         s_itr::Int,
#         τ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((- hbar2 * conj(Gpp(Δtτ_itr, α, α, β, χ)) + hbar2 * conj(Gpp(Δtτ_itr, β, β, β, χ)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp(Δtτ_itr, α, α, β, χ)) + conj(Gp(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ))) * (- conj(Gp(t_itr, α, α, β, β)) + conj(Gp(Δtτ_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, χ, χ)) - conj(Gp(Δtτ_itr, α, α, χ, χ)) + conj(Gp(t_itr, β, β, β, β)) - conj(Gp(Δtτ_itr, β, β, β, β)) - conj(Gp(t_itr, β, β, χ, χ)) + conj(Gp(Δtτ_itr, β, β, χ, χ)))) * (- Gp(Δτs_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(t_itr, α, α, α, α) + G(τ_itr, α, α, α, α) - G(Δτs_itr, α, α, α, α) + G(s_itr, α, α, μ, μ) - G(τ_itr, α, α, μ, μ) + G(Δτs_itr, α, α, μ, μ) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) + G(Δτs_itr, χ, χ, α, α) - G(t_itr, χ, χ, μ, μ) + G(τ_itr, χ, χ, μ, μ) - G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(Δtτ_itr, α, α, β, β)) - conj(G(s_itr, α, α, χ, χ)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(t_itr, χ, χ, χ, χ)) - conj(G(τ_itr, χ, χ, χ, χ)) + conj(G(s_itr, μ, μ, χ, χ)) - conj(G(t_itr, μ, μ, χ, χ))) / (hbar2))) + (hbar2 * (conj(Gpp(Δtτ_itr, α, α, β, χ)) - conj(Gpp(Δtτ_itr, β, β, β, χ))) * (- Gp(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp(Δts_itr, β, β, α, μ) - Gp(Δτs_itr, β, β, α, μ) + Gp(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) + hbar2 * (- Gp(Δts_itr, α, α, α, α) - Gp(t_itr, α, α, μ, μ) + Gp(Δts_itr, α, α, μ, μ) + conj(Gp(Δtτ_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, χ, χ)) - conj(Gp(Δtτ_itr, α, α, χ, χ))) * Gpp(Δτs_itr, χ, β, α, μ) + (hbar2 * Gpp(Δτs_itr, χ, β, α, μ) + (- Gp(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp(Δts_itr, β, β, α, μ) - Gp(Δτs_itr, β, β, α, μ) + Gp(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) * (Gp(Δτs_itr, χ, β, α, α) + Gp(τ_itr, χ, β, μ, μ) - Gp(Δτs_itr, χ, β, μ, μ) + conj(Gp(Δtτ_itr, α, α, β, χ)) - conj(Gp(Δtτ_itr, β, β, β, χ)) - conj(Gp(τ_itr, β, χ, χ, χ)))) * (Gp(t_itr, α, α, α, α) - Gp(t_itr, β, β, α, α) + Gp(Δts_itr, β, β, α, α) + Gp(t_itr, β, β, μ, μ) - Gp(Δts_itr, β, β, μ, μ) - conj(Gp(t_itr, α, α, β, β)) + conj(Gp(t_itr, β, β, β, β)) - conj(Gp(Δtτ_itr, β, β, β, β)) - conj(Gp(t_itr, β, β, χ, χ)) + conj(Gp(Δtτ_itr, β, β, χ, χ))) + (hbar2 * Gpp(Δts_itr, α, α, α, μ) - hbar2 * Gpp(Δts_itr, β, β, α, μ) + (Gp(Δts_itr, α, α, α, α) + Gp(t_itr, α, α, μ, μ) - Gp(Δts_itr, α, α, μ, μ) - conj(Gp(Δtτ_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, χ, χ)) + conj(Gp(Δtτ_itr, α, α, χ, χ))) * (- Gp(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp(Δts_itr, β, β, α, μ) - Gp(Δτs_itr, β, β, α, μ) + Gp(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ)))) * (- Gp(Δτs_itr, χ, β, α, α) - Gp(τ_itr, χ, β, μ, μ) + Gp(Δτs_itr, χ, β, μ, μ) - conj(Gp(Δtτ_itr, α, α, β, χ)) + conj(Gp(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δts_itr, α, α, α, α) + G(s_itr, α, α, μ, μ) - G(t_itr, α, α, μ, μ) + G(Δts_itr, α, α, μ, μ) + G(Δts_itr, β, β, α, α) - G(Δτs_itr, β, β, α, α) + G(t_itr, β, β, μ, μ) - G(τ_itr, β, β, μ, μ) - G(Δts_itr, β, β, μ, μ) + G(Δτs_itr, β, β, μ, μ) + G(Δτs_itr, χ, χ, α, α) - G(t_itr, χ, χ, μ, μ) + G(τ_itr, χ, χ, μ, μ) - G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(Δtτ_itr, α, α, β, β)) - conj(G(s_itr, α, α, χ, χ)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(t_itr, χ, χ, χ, χ)) - conj(G(τ_itr, χ, χ, χ, χ)) + conj(G(s_itr, μ, μ, χ, χ)) - conj(G(t_itr, μ, μ, χ, χ))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, μ, χ) * exp(((- G(s_itr, χ, χ, μ, μ) + G(t_itr, χ, χ, μ, μ) + G(s_itr, μ, μ, μ, μ) - G(t_itr, μ, μ, μ, μ) + conj(G(s_itr, χ, χ, χ, χ)) - conj(G(t_itr, χ, χ, χ, χ)) - conj(G(s_itr, μ, μ, χ, χ)) + conj(G(t_itr, μ, μ, χ, χ))) / (hbar2)))
#         )
#     end

#     Base.@noinline function kernel_L0G1_RR(
#         s_itr::Int,
#         τ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((hbar2 * conj(Gpp(Δtτ_itr, α, α, β, χ)) - hbar2 * conj(Gpp(Δtτ_itr, β, β, β, χ)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp(Δtτ_itr, α, α, β, χ)) + conj(Gp(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ))) * (conj(Gp(t_itr, α, α, β, β)) - conj(Gp(Δtτ_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, χ, χ)) + conj(Gp(Δtτ_itr, α, α, χ, χ)) - conj(Gp(t_itr, β, β, β, β)) + conj(Gp(Δtτ_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, χ, χ)) - conj(Gp(Δtτ_itr, β, β, χ, χ)))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp(Δτs_itr, α, α, χ, ν)) + conj(Gp(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) - G(s_itr, χ, χ, α, α) + G(τ_itr, χ, χ, α, α) + G(s_itr, ν, ν, α, α) - G(t_itr, ν, ν, α, α) + conj(G(Δtτ_itr, α, α, β, β)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(τ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) + conj(G(Δτs_itr, α, α, χ, χ)) - conj(G(t_itr, α, α, ν, ν)) + conj(G(τ_itr, α, α, ν, ν)) - conj(G(Δτs_itr, α, α, ν, ν)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) - conj(G(Δτs_itr, χ, χ, χ, χ)) + conj(G(s_itr, χ, χ, ν, ν)) - conj(G(τ_itr, χ, χ, ν, ν)) + conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2))) + (hbar2 * (- conj(Gpp(Δtτ_itr, α, α, β, χ)) + conj(Gpp(Δtτ_itr, β, β, β, χ))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp(Δts_itr, α, α, χ, ν)) + conj(Gp(Δts_itr, β, β, χ, ν)) - conj(Gp(Δτs_itr, β, β, χ, ν)) + conj(Gp(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν))) + hbar2 * (- Gp(t_itr, β, β, α, α) + conj(Gp(Δtτ_itr, β, β, β, β)) + conj(Gp(Δts_itr, β, β, χ, χ)) - conj(Gp(Δtτ_itr, β, β, χ, χ)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp(Δts_itr, β, β, ν, ν))) * conj(Gpp(Δτs_itr, β, χ, χ, ν)) + (hbar2 * conj(Gpp(Δτs_itr, β, χ, χ, ν)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp(Δtτ_itr, α, α, β, χ)) + conj(Gp(Δtτ_itr, β, β, β, χ)) + conj(Gp(Δτs_itr, β, χ, χ, χ)) + conj(Gp(τ_itr, β, χ, ν, ν)) - conj(Gp(Δτs_itr, β, χ, ν, ν))) * (Gp(s_itr, ν, χ, α, α) + conj(Gp(Δts_itr, α, α, χ, ν)) - conj(Gp(Δts_itr, β, β, χ, ν)) + conj(Gp(Δτs_itr, β, β, χ, ν)) - conj(Gp(Δτs_itr, χ, χ, χ, ν)) - conj(Gp(s_itr, χ, ν, ν, ν)))) * (Gp(t_itr, β, β, α, α) + conj(Gp(t_itr, α, α, β, β)) - conj(Gp(Δtτ_itr, α, α, β, β)) - conj(Gp(Δts_itr, α, α, χ, χ)) + conj(Gp(Δtτ_itr, α, α, χ, χ)) - conj(Gp(t_itr, α, α, ν, ν)) + conj(Gp(Δts_itr, α, α, ν, ν)) - conj(Gp(t_itr, β, β, β, β))) + (- hbar2 * conj(Gpp(Δts_itr, α, α, χ, ν)) + hbar2 * conj(Gpp(Δts_itr, β, β, χ, ν)) + (Gp(t_itr, β, β, α, α) - conj(Gp(Δtτ_itr, β, β, β, β)) - conj(Gp(Δts_itr, β, β, χ, χ)) + conj(Gp(Δtτ_itr, β, β, χ, χ)) - conj(Gp(t_itr, β, β, ν, ν)) + conj(Gp(Δts_itr, β, β, ν, ν))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp(Δts_itr, α, α, χ, ν)) + conj(Gp(Δts_itr, β, β, χ, ν)) - conj(Gp(Δτs_itr, β, β, χ, ν)) + conj(Gp(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν)))) * (- Gp(τ_itr, χ, β, α, α) - conj(Gp(Δtτ_itr, α, α, β, χ)) + conj(Gp(Δtτ_itr, β, β, β, χ)) + conj(Gp(Δτs_itr, β, χ, χ, χ)) + conj(Gp(τ_itr, β, χ, ν, ν)) - conj(Gp(Δτs_itr, β, χ, ν, ν)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) - G(s_itr, χ, χ, α, α) + G(τ_itr, χ, χ, α, α) + G(s_itr, ν, ν, α, α) - G(t_itr, ν, ν, α, α) + conj(G(Δtτ_itr, α, α, β, β)) + conj(G(Δts_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δts_itr, α, α, ν, ν)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(Δts_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(Δτs_itr, β, β, χ, χ)) - conj(G(t_itr, β, β, ν, ν)) + conj(G(τ_itr, β, β, ν, ν)) + conj(G(Δts_itr, β, β, ν, ν)) - conj(G(Δτs_itr, β, β, ν, ν)) - conj(G(Δτs_itr, χ, χ, χ, χ)) + conj(G(s_itr, χ, χ, ν, ν)) - conj(G(τ_itr, χ, χ, ν, ν)) + conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, α, ν) * exp(((G(s_itr, α, α, α, α) - G(t_itr, α, α, α, α) - G(s_itr, ν, ν, α, α) + G(t_itr, ν, ν, α, α) - conj(G(s_itr, α, α, ν, ν)) + conj(G(t_itr, α, α, ν, ν)) + conj(G(s_itr, ν, ν, ν, ν)) - conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))
#         )
#     end


#     # ---------------------------------------------------------------------
#     # Markovianized inner-G1 variants of P L0 Q G1 Q L1 P.
#     #
#     # These are mechanically identical to the exact 2-time kernels above,
#     # except that relative-time derivative factors with arguments
#     # Δts_itr, Δtτ_itr, and Δτs_itr use
#     #
#     #     Gp  -> Gp_markovian_G1  = -im * Λ
#     #     Gpp -> Gpp_markovian_G1 = 0
#     #
#     # Absolute-time g/g′ factors and all g-exponential dressing factors are
#     # intentionally left unchanged.  Therefore this switch only Markovianizes
#     # the internal G1 derivative/correlation content, not the full outer
#     # P L0 Q ... Q L1 P memory kernel.
#     # ---------------------------------------------------------------------

#     Base.@noinline function kernel_L0G1_LL_markovian_G1(
#         s_itr::Int,
#         τ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((hbar2 * Gpp_markovian_G1(Δtτ_itr, α, α, α, χ) - hbar2 * Gpp_markovian_G1(Δtτ_itr, β, β, α, χ) + (- Gp_markovian_G1(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β))) * (- Gp(t_itr, α, α, α, α) + Gp_markovian_G1(Δtτ_itr, α, α, α, α) + Gp(t_itr, α, α, χ, χ) - Gp_markovian_G1(Δtτ_itr, α, α, χ, χ) + Gp(t_itr, β, β, α, α) - Gp_markovian_G1(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, χ, χ))) * (Gp_markovian_G1(Δτs_itr, β, β, χ, μ) - Gp_markovian_G1(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) + G(t_itr, β, β, χ, χ) - G(τ_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(Δτs_itr, β, β, χ, χ) - G(t_itr, β, β, μ, μ) + G(τ_itr, β, β, μ, μ) - G(Δτs_itr, β, β, μ, μ) - G(Δτs_itr, χ, χ, χ, χ) + G(s_itr, χ, χ, μ, μ) - G(τ_itr, χ, χ, μ, μ) + G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(s_itr, χ, χ, β, β)) + conj(G(τ_itr, χ, χ, β, β)) + conj(G(s_itr, μ, μ, β, β)) - conj(G(t_itr, μ, μ, β, β))) / (hbar2))) + (hbar2 * (- Gpp_markovian_G1(Δtτ_itr, α, α, α, χ) + Gpp_markovian_G1(Δtτ_itr, β, β, α, χ)) * (- Gp_markovian_G1(Δts_itr, α, α, χ, μ) + Gp_markovian_G1(Δτs_itr, α, α, χ, μ) + Gp_markovian_G1(Δts_itr, β, β, χ, μ) - Gp_markovian_G1(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β))) + hbar2 * (- Gp_markovian_G1(Δtτ_itr, β, β, α, α) - Gp_markovian_G1(Δts_itr, β, β, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, χ, χ) - Gp(t_itr, β, β, μ, μ) + Gp_markovian_G1(Δts_itr, β, β, μ, μ) + conj(Gp(t_itr, β, β, β, β))) * Gpp_markovian_G1(Δτs_itr, α, χ, χ, μ) + (hbar2 * Gpp_markovian_G1(Δτs_itr, α, χ, χ, μ) + (Gp_markovian_G1(Δtτ_itr, α, α, α, χ) + Gp_markovian_G1(Δτs_itr, α, χ, χ, χ) + Gp(τ_itr, α, χ, μ, μ) - Gp_markovian_G1(Δτs_itr, α, χ, μ, μ) - Gp_markovian_G1(Δtτ_itr, β, β, α, χ) - conj(Gp(τ_itr, χ, α, β, β))) * (- Gp_markovian_G1(Δts_itr, α, α, χ, μ) + Gp_markovian_G1(Δτs_itr, α, α, χ, μ) + Gp_markovian_G1(Δts_itr, β, β, χ, μ) - Gp_markovian_G1(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β)))) * (- Gp(t_itr, α, α, α, α) + Gp_markovian_G1(Δtτ_itr, α, α, α, α) + Gp_markovian_G1(Δts_itr, α, α, χ, χ) - Gp_markovian_G1(Δtτ_itr, α, α, χ, χ) + Gp(t_itr, α, α, μ, μ) - Gp_markovian_G1(Δts_itr, α, α, μ, μ) + Gp(t_itr, β, β, α, α) - conj(Gp(t_itr, β, β, β, β))) + (- hbar2 * Gpp_markovian_G1(Δts_itr, α, α, χ, μ) + hbar2 * Gpp_markovian_G1(Δts_itr, β, β, χ, μ) + (Gp_markovian_G1(Δts_itr, α, α, χ, μ) - Gp_markovian_G1(Δτs_itr, α, α, χ, μ) - Gp_markovian_G1(Δts_itr, β, β, χ, μ) + Gp_markovian_G1(Δτs_itr, χ, χ, χ, μ) + Gp(s_itr, χ, μ, μ, μ) - conj(Gp(s_itr, μ, χ, β, β))) * (- Gp_markovian_G1(Δtτ_itr, β, β, α, α) - Gp_markovian_G1(Δts_itr, β, β, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, χ, χ) - Gp(t_itr, β, β, μ, μ) + Gp_markovian_G1(Δts_itr, β, β, μ, μ) + conj(Gp(t_itr, β, β, β, β)))) * (- Gp_markovian_G1(Δtτ_itr, α, α, α, χ) - Gp_markovian_G1(Δτs_itr, α, χ, χ, χ) - Gp(τ_itr, α, χ, μ, μ) + Gp_markovian_G1(Δτs_itr, α, χ, μ, μ) + Gp_markovian_G1(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(Δts_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δτs_itr, α, α, χ, χ) - G(t_itr, α, α, μ, μ) + G(τ_itr, α, α, μ, μ) + G(Δts_itr, α, α, μ, μ) - G(Δτs_itr, α, α, μ, μ) + G(Δtτ_itr, β, β, α, α) + G(Δts_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) - G(Δts_itr, β, β, μ, μ) - G(Δτs_itr, χ, χ, χ, χ) + G(s_itr, χ, χ, μ, μ) - G(τ_itr, χ, χ, μ, μ) + G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(s_itr, χ, χ, β, β)) + conj(G(τ_itr, χ, χ, β, β)) + conj(G(s_itr, μ, μ, β, β)) - conj(G(t_itr, μ, μ, β, β))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, μ, β) * exp(((- G(s_itr, β, β, μ, μ) + G(t_itr, β, β, μ, μ) + G(s_itr, μ, μ, μ, μ) - G(t_itr, μ, μ, μ, μ) + conj(G(s_itr, β, β, β, β)) - conj(G(t_itr, β, β, β, β)) - conj(G(s_itr, μ, μ, β, β)) + conj(G(t_itr, μ, μ, β, β))) / (hbar2)))
#         )
#     end


#     Base.@noinline function kernel_L0G1_LR_markovian_G1(
#         s_itr::Int,
#         τ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((- hbar2 * Gpp_markovian_G1(Δtτ_itr, α, α, α, χ) + hbar2 * Gpp_markovian_G1(Δtτ_itr, β, β, α, χ) + (- Gp_markovian_G1(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β))) * (Gp(t_itr, α, α, α, α) - Gp_markovian_G1(Δtτ_itr, α, α, α, α) - Gp(t_itr, α, α, χ, χ) + Gp_markovian_G1(Δtτ_itr, α, α, χ, χ) - Gp(t_itr, β, β, α, α) + Gp_markovian_G1(Δtτ_itr, β, β, α, α) + Gp(t_itr, β, β, χ, χ) - Gp_markovian_G1(Δtτ_itr, β, β, χ, χ))) * (- Gp(s_itr, ν, β, χ, χ) + conj(Gp_markovian_G1(Δτs_itr, β, β, β, ν)) + conj(Gp(s_itr, β, ν, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, χ, χ, β, ν))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) - G(s_itr, β, β, χ, χ) + G(t_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(t_itr, χ, χ, χ, χ) - G(τ_itr, χ, χ, χ, χ) + G(s_itr, ν, ν, χ, χ) - G(t_itr, ν, ν, χ, χ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(t_itr, β, β, β, β)) + conj(G(τ_itr, β, β, β, β)) - conj(G(Δτs_itr, β, β, β, β)) + conj(G(s_itr, β, β, ν, ν)) - conj(G(τ_itr, β, β, ν, ν)) + conj(G(Δτs_itr, β, β, ν, ν)) + conj(G(Δτs_itr, χ, χ, β, β)) - conj(G(t_itr, χ, χ, ν, ν)) + conj(G(τ_itr, χ, χ, ν, ν)) - conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2))) + (hbar2 * (conj(Gpp_markovian_G1(Δts_itr, α, α, β, ν)) - conj(Gpp_markovian_G1(Δts_itr, β, β, β, ν))) * (- Gp_markovian_G1(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, α, χ) + conj(Gp_markovian_G1(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, χ, α, ν, ν))) + hbar2 * (- Gp_markovian_G1(Δtτ_itr, α, α, α, α) - Gp(t_itr, α, α, χ, χ) + Gp_markovian_G1(Δtτ_itr, α, α, χ, χ) + conj(Gp_markovian_G1(Δts_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, ν, ν)) - conj(Gp_markovian_G1(Δts_itr, α, α, ν, ν))) * conj(Gpp_markovian_G1(Δτs_itr, χ, α, β, ν)) - hbar2 * (- Gp_markovian_G1(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, χ, χ) + conj(Gp_markovian_G1(Δts_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp_markovian_G1(Δts_itr, β, β, ν, ν))) * conj(Gpp_markovian_G1(Δτs_itr, χ, α, β, ν)) + (hbar2 * conj(Gpp_markovian_G1(Δτs_itr, χ, α, β, ν)) + (- Gp_markovian_G1(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, α, χ) + conj(Gp_markovian_G1(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, χ, α, ν, ν))) * (Gp(s_itr, ν, β, χ, χ) + conj(Gp_markovian_G1(Δts_itr, α, α, β, ν)) - conj(Gp_markovian_G1(Δτs_itr, α, α, β, ν)) - conj(Gp_markovian_G1(Δts_itr, β, β, β, ν)) - conj(Gp(s_itr, β, ν, ν, ν)) + conj(Gp_markovian_G1(Δτs_itr, χ, χ, β, ν)))) * (Gp(t_itr, α, α, α, α) - Gp(t_itr, β, β, α, α) - conj(Gp(t_itr, α, α, β, β)) + conj(Gp(t_itr, β, β, β, β))) + (hbar2 * Gpp_markovian_G1(Δtτ_itr, α, α, α, χ) - hbar2 * Gpp_markovian_G1(Δtτ_itr, β, β, α, χ) + (- Gp_markovian_G1(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, α, χ) + conj(Gp_markovian_G1(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, χ, α, ν, ν))) * (Gp_markovian_G1(Δtτ_itr, α, α, α, α) + Gp(t_itr, α, α, χ, χ) - Gp_markovian_G1(Δtτ_itr, α, α, χ, χ) - Gp_markovian_G1(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, χ, χ) - conj(Gp_markovian_G1(Δts_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, ν, ν)) + conj(Gp_markovian_G1(Δts_itr, α, α, ν, ν)) + conj(Gp_markovian_G1(Δts_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp_markovian_G1(Δts_itr, β, β, ν, ν)))) * (- Gp(s_itr, ν, β, χ, χ) - conj(Gp_markovian_G1(Δts_itr, α, α, β, ν)) + conj(Gp_markovian_G1(Δτs_itr, α, α, β, ν)) + conj(Gp_markovian_G1(Δts_itr, β, β, β, ν)) + conj(Gp(s_itr, β, ν, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, χ, χ, β, ν)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) - G(s_itr, β, β, χ, χ) + G(t_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(t_itr, χ, χ, χ, χ) - G(τ_itr, χ, χ, χ, χ) + G(s_itr, ν, ν, χ, χ) - G(t_itr, ν, ν, χ, χ) + conj(G(Δts_itr, α, α, β, β)) - conj(G(Δτs_itr, α, α, β, β)) + conj(G(t_itr, α, α, ν, ν)) - conj(G(τ_itr, α, α, ν, ν)) - conj(G(Δts_itr, α, α, ν, ν)) + conj(G(Δτs_itr, α, α, ν, ν)) - conj(G(Δts_itr, β, β, β, β)) + conj(G(s_itr, β, β, ν, ν)) - conj(G(t_itr, β, β, ν, ν)) + conj(G(Δts_itr, β, β, ν, ν)) + conj(G(Δτs_itr, χ, χ, β, β)) - conj(G(t_itr, χ, χ, ν, ν)) + conj(G(τ_itr, χ, χ, ν, ν)) - conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, χ, ν) * exp(((G(s_itr, χ, χ, χ, χ) - G(t_itr, χ, χ, χ, χ) - G(s_itr, ν, ν, χ, χ) + G(t_itr, ν, ν, χ, χ) - conj(G(s_itr, χ, χ, ν, ν)) + conj(G(t_itr, χ, χ, ν, ν)) + conj(G(s_itr, ν, ν, ν, ν)) - conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))
#         )
#     end


#     Base.@noinline function kernel_L0G1_RL_markovian_G1(
#         s_itr::Int,
#         τ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((- hbar2 * conj(Gpp_markovian_G1(Δtτ_itr, α, α, β, χ)) + hbar2 * conj(Gpp_markovian_G1(Δtτ_itr, β, β, β, χ)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ))) * (- conj(Gp(t_itr, α, α, β, β)) + conj(Gp_markovian_G1(Δtτ_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, χ, χ)) - conj(Gp_markovian_G1(Δtτ_itr, α, α, χ, χ)) + conj(Gp(t_itr, β, β, β, β)) - conj(Gp_markovian_G1(Δtτ_itr, β, β, β, β)) - conj(Gp(t_itr, β, β, χ, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, χ, χ)))) * (- Gp_markovian_G1(Δτs_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp_markovian_G1(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(t_itr, α, α, α, α) + G(τ_itr, α, α, α, α) - G(Δτs_itr, α, α, α, α) + G(s_itr, α, α, μ, μ) - G(τ_itr, α, α, μ, μ) + G(Δτs_itr, α, α, μ, μ) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) + G(Δτs_itr, χ, χ, α, α) - G(t_itr, χ, χ, μ, μ) + G(τ_itr, χ, χ, μ, μ) - G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(Δtτ_itr, α, α, β, β)) - conj(G(s_itr, α, α, χ, χ)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(t_itr, χ, χ, χ, χ)) - conj(G(τ_itr, χ, χ, χ, χ)) + conj(G(s_itr, μ, μ, χ, χ)) - conj(G(t_itr, μ, μ, χ, χ))) / (hbar2))) + (hbar2 * (conj(Gpp_markovian_G1(Δtτ_itr, α, α, β, χ)) - conj(Gpp_markovian_G1(Δtτ_itr, β, β, β, χ))) * (- Gp_markovian_G1(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp_markovian_G1(Δts_itr, β, β, α, μ) - Gp_markovian_G1(Δτs_itr, β, β, α, μ) + Gp_markovian_G1(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) + hbar2 * (- Gp_markovian_G1(Δts_itr, α, α, α, α) - Gp(t_itr, α, α, μ, μ) + Gp_markovian_G1(Δts_itr, α, α, μ, μ) + conj(Gp_markovian_G1(Δtτ_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, χ, χ)) - conj(Gp_markovian_G1(Δtτ_itr, α, α, χ, χ))) * Gpp_markovian_G1(Δτs_itr, χ, β, α, μ) + (hbar2 * Gpp_markovian_G1(Δτs_itr, χ, β, α, μ) + (- Gp_markovian_G1(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp_markovian_G1(Δts_itr, β, β, α, μ) - Gp_markovian_G1(Δτs_itr, β, β, α, μ) + Gp_markovian_G1(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) * (Gp_markovian_G1(Δτs_itr, χ, β, α, α) + Gp(τ_itr, χ, β, μ, μ) - Gp_markovian_G1(Δτs_itr, χ, β, μ, μ) + conj(Gp_markovian_G1(Δtτ_itr, α, α, β, χ)) - conj(Gp_markovian_G1(Δtτ_itr, β, β, β, χ)) - conj(Gp(τ_itr, β, χ, χ, χ)))) * (Gp(t_itr, α, α, α, α) - Gp(t_itr, β, β, α, α) + Gp_markovian_G1(Δts_itr, β, β, α, α) + Gp(t_itr, β, β, μ, μ) - Gp_markovian_G1(Δts_itr, β, β, μ, μ) - conj(Gp(t_itr, α, α, β, β)) + conj(Gp(t_itr, β, β, β, β)) - conj(Gp_markovian_G1(Δtτ_itr, β, β, β, β)) - conj(Gp(t_itr, β, β, χ, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, χ, χ))) + (hbar2 * Gpp_markovian_G1(Δts_itr, α, α, α, μ) - hbar2 * Gpp_markovian_G1(Δts_itr, β, β, α, μ) + (Gp_markovian_G1(Δts_itr, α, α, α, α) + Gp(t_itr, α, α, μ, μ) - Gp_markovian_G1(Δts_itr, α, α, μ, μ) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, χ, χ)) + conj(Gp_markovian_G1(Δtτ_itr, α, α, χ, χ))) * (- Gp_markovian_G1(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp_markovian_G1(Δts_itr, β, β, α, μ) - Gp_markovian_G1(Δτs_itr, β, β, α, μ) + Gp_markovian_G1(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ)))) * (- Gp_markovian_G1(Δτs_itr, χ, β, α, α) - Gp(τ_itr, χ, β, μ, μ) + Gp_markovian_G1(Δτs_itr, χ, β, μ, μ) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δts_itr, α, α, α, α) + G(s_itr, α, α, μ, μ) - G(t_itr, α, α, μ, μ) + G(Δts_itr, α, α, μ, μ) + G(Δts_itr, β, β, α, α) - G(Δτs_itr, β, β, α, α) + G(t_itr, β, β, μ, μ) - G(τ_itr, β, β, μ, μ) - G(Δts_itr, β, β, μ, μ) + G(Δτs_itr, β, β, μ, μ) + G(Δτs_itr, χ, χ, α, α) - G(t_itr, χ, χ, μ, μ) + G(τ_itr, χ, χ, μ, μ) - G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(Δtτ_itr, α, α, β, β)) - conj(G(s_itr, α, α, χ, χ)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(t_itr, χ, χ, χ, χ)) - conj(G(τ_itr, χ, χ, χ, χ)) + conj(G(s_itr, μ, μ, χ, χ)) - conj(G(t_itr, μ, μ, χ, χ))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, μ, χ) * exp(((- G(s_itr, χ, χ, μ, μ) + G(t_itr, χ, χ, μ, μ) + G(s_itr, μ, μ, μ, μ) - G(t_itr, μ, μ, μ, μ) + conj(G(s_itr, χ, χ, χ, χ)) - conj(G(t_itr, χ, χ, χ, χ)) - conj(G(s_itr, μ, μ, χ, χ)) + conj(G(t_itr, μ, μ, χ, χ))) / (hbar2)))
#         )
#     end


#     Base.@noinline function kernel_L0G1_RR_markovian_G1(
#         s_itr::Int,
#         τ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((hbar2 * conj(Gpp_markovian_G1(Δtτ_itr, α, α, β, χ)) - hbar2 * conj(Gpp_markovian_G1(Δtτ_itr, β, β, β, χ)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ))) * (conj(Gp(t_itr, α, α, β, β)) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, χ, χ)) + conj(Gp_markovian_G1(Δtτ_itr, α, α, χ, χ)) - conj(Gp(t_itr, β, β, β, β)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, χ, χ)) - conj(Gp_markovian_G1(Δtτ_itr, β, β, χ, χ)))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp_markovian_G1(Δτs_itr, α, α, χ, ν)) + conj(Gp_markovian_G1(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) - G(s_itr, χ, χ, α, α) + G(τ_itr, χ, χ, α, α) + G(s_itr, ν, ν, α, α) - G(t_itr, ν, ν, α, α) + conj(G(Δtτ_itr, α, α, β, β)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(τ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) + conj(G(Δτs_itr, α, α, χ, χ)) - conj(G(t_itr, α, α, ν, ν)) + conj(G(τ_itr, α, α, ν, ν)) - conj(G(Δτs_itr, α, α, ν, ν)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) - conj(G(Δτs_itr, χ, χ, χ, χ)) + conj(G(s_itr, χ, χ, ν, ν)) - conj(G(τ_itr, χ, χ, ν, ν)) + conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2))) + (hbar2 * (- conj(Gpp_markovian_G1(Δtτ_itr, α, α, β, χ)) + conj(Gpp_markovian_G1(Δtτ_itr, β, β, β, χ))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp_markovian_G1(Δts_itr, α, α, χ, ν)) + conj(Gp_markovian_G1(Δts_itr, β, β, χ, ν)) - conj(Gp_markovian_G1(Δτs_itr, β, β, χ, ν)) + conj(Gp_markovian_G1(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν))) + hbar2 * (- Gp(t_itr, β, β, α, α) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, β)) + conj(Gp_markovian_G1(Δts_itr, β, β, χ, χ)) - conj(Gp_markovian_G1(Δtτ_itr, β, β, χ, χ)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp_markovian_G1(Δts_itr, β, β, ν, ν))) * conj(Gpp_markovian_G1(Δτs_itr, β, χ, χ, ν)) + (hbar2 * conj(Gpp_markovian_G1(Δτs_itr, β, χ, χ, ν)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, χ)) + conj(Gp_markovian_G1(Δτs_itr, β, χ, χ, χ)) + conj(Gp(τ_itr, β, χ, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, β, χ, ν, ν))) * (Gp(s_itr, ν, χ, α, α) + conj(Gp_markovian_G1(Δts_itr, α, α, χ, ν)) - conj(Gp_markovian_G1(Δts_itr, β, β, χ, ν)) + conj(Gp_markovian_G1(Δτs_itr, β, β, χ, ν)) - conj(Gp_markovian_G1(Δτs_itr, χ, χ, χ, ν)) - conj(Gp(s_itr, χ, ν, ν, ν)))) * (Gp(t_itr, β, β, α, α) + conj(Gp(t_itr, α, α, β, β)) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, β)) - conj(Gp_markovian_G1(Δts_itr, α, α, χ, χ)) + conj(Gp_markovian_G1(Δtτ_itr, α, α, χ, χ)) - conj(Gp(t_itr, α, α, ν, ν)) + conj(Gp_markovian_G1(Δts_itr, α, α, ν, ν)) - conj(Gp(t_itr, β, β, β, β))) + (- hbar2 * conj(Gpp_markovian_G1(Δts_itr, α, α, χ, ν)) + hbar2 * conj(Gpp_markovian_G1(Δts_itr, β, β, χ, ν)) + (Gp(t_itr, β, β, α, α) - conj(Gp_markovian_G1(Δtτ_itr, β, β, β, β)) - conj(Gp_markovian_G1(Δts_itr, β, β, χ, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, χ, χ)) - conj(Gp(t_itr, β, β, ν, ν)) + conj(Gp_markovian_G1(Δts_itr, β, β, ν, ν))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp_markovian_G1(Δts_itr, α, α, χ, ν)) + conj(Gp_markovian_G1(Δts_itr, β, β, χ, ν)) - conj(Gp_markovian_G1(Δτs_itr, β, β, χ, ν)) + conj(Gp_markovian_G1(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν)))) * (- Gp(τ_itr, χ, β, α, α) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, χ)) + conj(Gp_markovian_G1(Δτs_itr, β, χ, χ, χ)) + conj(Gp(τ_itr, β, χ, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, β, χ, ν, ν)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) - G(s_itr, χ, χ, α, α) + G(τ_itr, χ, χ, α, α) + G(s_itr, ν, ν, α, α) - G(t_itr, ν, ν, α, α) + conj(G(Δtτ_itr, α, α, β, β)) + conj(G(Δts_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δts_itr, α, α, ν, ν)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(Δts_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(Δτs_itr, β, β, χ, χ)) - conj(G(t_itr, β, β, ν, ν)) + conj(G(τ_itr, β, β, ν, ν)) + conj(G(Δts_itr, β, β, ν, ν)) - conj(G(Δτs_itr, β, β, ν, ν)) - conj(G(Δτs_itr, χ, χ, χ, χ)) + conj(G(s_itr, χ, χ, ν, ν)) - conj(G(τ_itr, χ, χ, ν, ν)) + conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, α, ν) * exp(((G(s_itr, α, α, α, α) - G(t_itr, α, α, α, α) - G(s_itr, ν, ν, α, α) + G(t_itr, ν, ν, α, α) - conj(G(s_itr, α, α, ν, ν)) + conj(G(t_itr, α, α, ν, ν)) + conj(G(s_itr, ν, ν, ν, ν)) - conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))
#         )
#     end


#     # Evaluate the four PL0QG1QL1P branch kernels at a single tau point.
#     # This helper is used by both the original nested quadrature and the
#     # Option-B collapsed-tau approximation below.
#     Base.@noinline function eval_L0G1_tau_kernel(
#         s_coord::Int,
#         τ_coord::Int,
#         t_eval_coord::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#     )
#         τ_kernel = 0.0 + 0.0im

#         # P L0 Q G1 Q L1 P, LL: input (μ, β), intermediate χ.
#         for χ in 1:n_sys
#             χ == α && continue
#             for μ in 1:n_sys
#                 μ == χ && continue
#                 if is_secular_pair(α, β, μ, β)
#                     τ_kernel += (
#                         effective_markovianize_PL0QG1QL1P_G1 ?
#                         kernel_L0G1_LL_markovian_G1(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, μ) :
#                         kernel_L0G1_LL(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, μ)
#                     )
#                 end
#             end
#         end

#         # P L0 Q G1 Q L1 P, RR: input (α, ν), intermediate χ.
#         for χ in 1:n_sys
#             χ == β && continue
#             for ν in 1:n_sys
#                 ν == χ && continue
#                 if is_secular_pair(α, β, α, ν)
#                     τ_kernel += (
#                         effective_markovianize_PL0QG1QL1P_G1 ?
#                         kernel_L0G1_RR_markovian_G1(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, ν) :
#                         kernel_L0G1_RR(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, ν)
#                     )
#                 end
#             end
#         end

#         # P L0 Q G1 Q L1 P, RL: input (μ, χ), left input μ, right intermediate χ.
#         for μ in 1:n_sys
#             μ == α && continue
#             for χ in 1:n_sys
#                 χ == β && continue
#                 if is_secular_pair(α, β, μ, χ)
#                     τ_kernel += (
#                         effective_markovianize_PL0QG1QL1P_G1 ?
#                         kernel_L0G1_RL_markovian_G1(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, μ) :
#                         kernel_L0G1_RL(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, μ)
#                     )
#                 end
#             end
#         end

#         # P L0 Q G1 Q L1 P, LR: input (χ, ν), left intermediate χ, right input ν.
#         for χ in 1:n_sys
#             χ == α && continue
#             for ν in 1:n_sys
#                 ν == β && continue
#                 if is_secular_pair(α, β, χ, ν)
#                     τ_kernel += (
#                         effective_markovianize_PL0QG1QL1P_G1 ?
#                         kernel_L0G1_LR_markovian_G1(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, ν) :
#                         kernel_L0G1_LR(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, ν)
#                     )
#                 end
#             end
#         end

#         return τ_kernel
#     end


#     Base.@noinline function calc__rhs_element!(
#         rhs_mat::AbstractMatrix,
#         curr_itr::Int,
#         t_eval_coord::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#     )
#         if use_population_closure && α != β
#             rhs_mat[α, β] = 0.0 + 0.0im
#             return nothing
#         end

#         rhs = include_PL0P_local ? D_L0P(t_eval_coord, α, β) * σ_local(σ_t, α, β) : (0.0 + 0.0im)

#         # -------------------------------------------------------------
#         # P L1 P local left/right mixing.
#         # Left branch:  input (μ, β) -> output (α, β).
#         # Right branch: input (α, ν) -> output (α, β).
#         # These are time-local, so σ_t and t_eval_coord are used.
#         # -------------------------------------------------------------
#         if include_PL1P_local
#             for μ in 1:n_sys
#                 μ == α && continue
#                 if is_secular_pair(α, β, μ, β)
#                     rhs += -(
#                         σ_local(σ_t, μ, β)
#                         * (
#                             Gp(t_eval_coord, α, μ, μ, μ)
#                             -conj(Gp(t_eval_coord, μ, α, β, β))
#                         ) / hbar2
#                     )
#                 end
#             end

#             for ν in 1:n_sys
#                 ν == β && continue
#                 if is_secular_pair(α, β, α, ν)
#                     rhs += (
#                         σ_local(σ_t, α, ν)
#                         * (
#                             Gp(t_eval_coord, ν, β, α, α)
#                             -conj(Gp(t_eval_coord, β, ν, ν, ν))
#                         ) / hbar2
#                     )
#                 end
#             end
#         end

#         if t_eval_coord > 1
#             integral = 0.0 + 0.0im
#             n_s_nodes = n_outer_nodes(curr_itr, t_eval_coord)

#             for s_node in 1:n_s_nodes
#                 s_coord = outer_node_coord(s_node, curr_itr, t_eval_coord)
#                 Δ_coord = t_eval_coord - s_coord + 1
#                 w_int = ∫weight_to_t(s_node, curr_itr, t_eval_coord)
#                 w_int == 0.0 && continue

#                 kernel = 0.0 + 0.0im

#                 if include_P_L0plusL1_QG0QL1P
#                     # Combined P (L0 + L1) Q G0 Q L1 P.
#                     #
#                     # Previous implementation evaluated
#                     #   P L0 Q G0 Q L1 P
#                     # and
#                     #   P L1 Q G0 Q L1 P
#                     # in separate loop nests.  Algebraically, the L0 term is
#                     # the diagonal-boundary extension of the same LL/LR/RL/RR
#                     # trace primitive used by the L1 term:
#                     #   L0-left  = LL|χ=α + RL|χ=β,
#                     #   L0-right = LR|χ=α + RR|χ=β.
#                     #
#                     # Therefore the first intermediate index is unrestricted
#                     # below.  Only the second L1 insertion keeps its
#                     # off-diagonal condition.

#                     # LL-like branch:
#                     #   first insertion α -> χ, second insertion χ -> μ,
#                     #   input (μ, β) -> output (α, β).
#                     for χ in 1:n_sys
#                         for μ in 1:n_sys
#                             μ == χ && continue
#                             if is_secular_pair(α, β, μ, β)
#                                 kernel += kernel_LL(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, χ, μ)
#                             end
#                         end
#                     end

#                     # LR-like branch:
#                     #   first insertion α -> χ, second insertion ν -> β,
#                     #   input (χ, ν) -> output (α, β).
#                     for χ in 1:n_sys
#                         for ν in 1:n_sys
#                             ν == β && continue
#                             if is_secular_pair(α, β, χ, ν)
#                                 kernel += kernel_LR(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, χ, ν)
#                             end
#                         end
#                     end

#                     # RL-like branch:
#                     #   first insertion χ -> β, second insertion α -> μ,
#                     #   input (μ, χ) -> output (α, β).
#                     for χ in 1:n_sys
#                         for μ in 1:n_sys
#                             μ == α && continue
#                             if is_secular_pair(α, β, μ, χ)
#                                 kernel += kernel_RL(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, χ, μ)
#                             end
#                         end
#                     end

#                     # RR-like branch:
#                     #   first insertion χ -> β, second insertion ν -> χ,
#                     #   input (α, ν) -> output (α, β).
#                     for χ in 1:n_sys
#                         for ν in 1:n_sys
#                             ν == χ && continue
#                             if is_secular_pair(α, β, α, ν)
#                                 kernel += kernel_RR(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, χ, ν)
#                             end
#                         end
#                     end
#                 end

#                 if include_PL0QG1QL1P
#                     inner_kernel = 0.0 + 0.0im

#                     if collapse_tau_PL0QG1QL1P_G1
#                         # Option B: collapse the inner tau integral while keeping
#                         # the outer s-memory integral.  In grid coordinates,
#                         # physical interval length is 0.5*(q_t - q_s)*Δt in doubled q-grid.
#                         # This is a midpoint effective-kernel approximation:
#                         #   ∫_s^t dτ K(t,τ,s) ≈ (t-s) K(t,nearest_q((t+s)/2),s).
#                         τ_interval = 0.5 * Float64(t_eval_coord - s_coord) * Δt
#                         if τ_interval > 0.0
#                             τ_mid = min(max(div(t_eval_coord + s_coord + 1, 2), 1), const_Q_MAX)
#                             inner_kernel += τ_interval * eval_L0G1_tau_kernel(
#                                 s_coord,
#                                 τ_mid,
#                                 t_eval_coord,
#                                 σ_t,
#                                 α,
#                                 β,
#                             )
#                         end
#                     else
#                         n_τ_nodes = n_inner_nodes(s_coord, t_eval_coord)
#                         for τ_node in 1:n_τ_nodes
#                             τ_coord = inner_node_coord(τ_node, s_coord, t_eval_coord)
#                             w_τ = ∫weight_between_coord(τ_node, s_coord, t_eval_coord)
#                             w_τ == 0.0 && continue

#                             inner_kernel += w_τ * eval_L0G1_tau_kernel(
#                                 s_coord,
#                                 τ_coord,
#                                 t_eval_coord,
#                                 σ_t,
#                                 α,
#                                 β,
#                             )
#                         end
#                     end

#                     kernel += inner_kernel
#                 end

#                 integral += w_int * kernel
#             end

#             rhs += integral
#         end

#         rhs_mat[α, β] = rhs
#         return nothing
#     end

#     @inline function calc__rhs!(
#         rhs_mat::AbstractMatrix,
#         curr_itr::Int,
#         t_eval_coord::Int,
#         σ_t::AbstractMatrix,
#     )
#         fill!(rhs_mat, 0.0 + 0.0im)

#         n_components = n_sys * n_sys
#         heavy_memory = include_P_L0plusL1_QG0QL1P || include_PL0QG1QL1P
#         min_thread_components = max(16, 4 * Threads.nthreads())
#         if use_threads && Threads.nthreads() > 1 && (heavy_memory || n_components >= min_thread_components)
#             Threads.@threads for linear_idx in 1:n_components
#                 @inbounds begin
#                     α = ((linear_idx - 1) % n_sys) + 1
#                     β = ((linear_idx - 1) ÷ n_sys) + 1
#                     calc__rhs_element!(rhs_mat, curr_itr, t_eval_coord, σ_t, α, β)
#                 end
#             end
#         else
#             @inbounds for β in 1:n_sys, α in 1:n_sys
#                 calc__rhs_element!(rhs_mat, curr_itr, t_eval_coord, σ_t, α, β)
#             end
#         end

#         return rhs_mat
#     end

#     @inline function enforce_hermiticity!(σ_next::AbstractMatrix)
#         for i in 1:n_sys
#             σ_next[i, i] = real(σ_next[i, i]) + 0.0im
#         end

#         for i in 1:(n_sys - 1)
#             for j in (i + 1):n_sys
#                 c = 0.5 * (σ_next[i, j] + conj(σ_next[j, i]))
#                 σ_next[i, j] = c
#                 σ_next[j, i] = conj(c)
#             end
#         end

#         return σ_next
#     end

#     @inline function enforce_population_closure!(σ_next::AbstractMatrix)
#         for j in 1:n_sys, i in 1:n_sys
#             if i != j
#                 σ_next[i, j] = 0.0 + 0.0im
#             else
#                 σ_next[i, i] = real(σ_next[i, i]) + 0.0im
#             end
#         end
#         return σ_next
#     end

#     if verbose && include_PL0QG1QL1P
#         if collapse_tau_PL0QG1QL1P_G1
#             @printf(stderr, "Warning: include_PL0QG1QL1P=true with collapse_tau_PL0QG1QL1P_G1=true uses midpoint tau-collapse; this reduces the G1 term to roughly O(N_t^2).\n")
#         else
#             @printf(stderr, "Warning: include_PL0QG1QL1P=true enables the nested s-tau memory term; runtime scales roughly as O(N_t^3).\n")
#         end
#     end

#     σ_stage = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k2      = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k3      = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k4      = Matrix{ComplexF64}(undef, n_sys, n_sys)

#     @inbounds for curr_itr in start_itr:(n_itr - 1)
#         if verbose && (curr_itr == start_itr || curr_itr == n_itr - 1 || ((curr_itr - start_itr) % verbose_every_f == 0))
#             @printf(
#                 stderr,
#                 "Current iteration: %6d / %6d  method=%s  threads=%d  use_threads=%s  secular=%s  PL0QG1_markovian_G1=%s  PL0QG1_collapse_tau=%s\n",
#                 curr_itr,
#                 n_itr,
#                 String(method_sym),
#                 Threads.nthreads(),
#                 string(use_threads),
#                 string(use_secular),
#                 string(effective_markovianize_PL0QG1QL1P_G1),
#                 string(collapse_tau_PL0QG1QL1P_G1),
#             )
#         end

#         σ_t    = @view σ[:, :, curr_itr]
#         σ_next = @view σ[:, :, curr_itr + 1]
#         k1     = @view σ′[:, :, curr_itr]

#         if method_sym == :euler
#             calc__rhs!(k1, curr_itr, q_full(curr_itr), σ_t)
#             @. σ_next = σ_t + Δt * k1

#         elseif method_sym == :rk2
#             calc__rhs!(k1, curr_itr, q_full(curr_itr), σ_t)
#             @. σ_stage = σ_t + 0.5 * Δt * k1
#             use_population_closure && enforce_population_closure!(σ_stage)
#             calc__rhs!(k2, curr_itr, q_half_after(curr_itr), σ_stage)
#             @. σ_next = σ_t + Δt * k2

#         elseif method_sym == :rk4
#             # print("시작0\n")
#             calc__rhs!(k1, curr_itr, q_full(curr_itr), σ_t)

#             # print("시작1\n")
#             @. σ_stage = σ_t + 0.5 * Δt * k1
#             use_population_closure && enforce_population_closure!(σ_stage)
#             calc__rhs!(k2, curr_itr, q_half_after(curr_itr), σ_stage)

#             # print("시작2\n")
#             @. σ_stage = σ_t + 0.5 * Δt * k2
#             use_population_closure && enforce_population_closure!(σ_stage)
#             calc__rhs!(k3, curr_itr, q_half_after(curr_itr), σ_stage)

#             # print("시작3\n")
#             @. σ_stage = σ_t + Δt * k3
#             use_population_closure && enforce_population_closure!(σ_stage)
#             calc__rhs!(k4, curr_itr, q_full(curr_itr + 1), σ_stage)

#             @. σ_next = σ_t + (Δt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
#             # print("끝\n")
#         end

#         if use_population_closure
#             enforce_population_closure!(σ_next)
#         elseif enforce_hermitian
#             enforce_hermiticity!(σ_next)
#         end

#         context.curr_itr = UInt64(curr_itr + 1)
#     end

#     return @view σ[:, :, Int(context.curr_itr)]
# end


# # Formula-based transported-projector c-dynamics + physical sigma readout.
# # V12 patch: cyclic-canonical population-direct c-memory replacement.
# function calc__σ_σ′_secular_core!(
#     context::RmrtContext;
#     hbar::Real = 1.0,
#     use_population_closure::Bool = false,

#     include_c_L0P_local::Bool = true,
#     include_c_PL1P_local::Bool = true,
#     include_c_memory_G0::Bool = true,
#     include_sigma_G0_readout::Bool = true,

#     # Temporary diagnostic option.  Default false because the v5 readout formula
#     # is defined componentwise; set true only to test whether population readout
#     # corrections are causing trace drift in a particular benchmark.
#     readout_population_c_only::Bool = true,

#     # V12 population-direct memory patch.
#     # The coefficient functions already contain the Liouville L/R signs from
#     # cL=-i/hbar and cR=+i/hbar.  Therefore the default diagnostic multipliers
#     # are neutral.  Change them only for explicit sign-audit experiments.
#     use_v12_canonical_population_direct::Bool = true,
#     c_memory_global_sign::Real = -1.0,
#     c_memory_sign_LL::Real = -1.0,
#     c_memory_sign_LR::Real = 1.0,
#     c_memory_sign_RL::Real = 1.0,
#     c_memory_sign_RR::Real = -1.0,

#     include_c_memory_LL::Bool = true,
#     include_c_memory_LR::Bool = true,
#     include_c_memory_RL::Bool = true,
#     include_c_memory_RR::Bool = true,

#     # Store the full-grid k1 memory contribution per branch.  This is useful
#     # for locating which branch breaks population direction or Hermiticity.
#     return_branch_memory_audit::Bool = false,

#     # Kernel row-sum audit independent of the current c-history.
#     # For each full-grid time, stores the memory-kernel row sum
#     #   sum_r K_{rr;ab}^{branch}(t,s integrated)
#     # for each input pair (a,b) and branch.  This is the direct test for
#     # trace preservation of the c-memory block.
#     return_kernel_rowsum_audit::Bool = false,
#     # Output-resolved version of the same audit. Dimensions:
#     #   kernel_output[r_out, a_in, b_in, itr, branch].
#     # This is needed because branch row sums alone can hide whether the
#     # leak is a loss-side or gain-side error.
#     return_kernel_output_audit::Bool = true,

#     sigma_readout_global_sign::Real = 1.0,

#     # Do not silently hide algebraic problems.  These audits print when
#     # physical sigma develops an imaginary trace or loses Hermiticity.
#     audit_trace_hermiticity::Bool = true,
#     trace_imag_tol::Float64 = 1.0e-8,
#     hermiticity_tol::Float64 = 1.0e-8,

#     use_secular::Bool = false,
#     secular_tol::Float64 = 1.0e-10,
#     method::Union{Symbol,String} = :rk4,
#     use_threads::Bool = true,

#     # Validation defaults: do not hide algebraic errors unless explicitly asked.
#     enforce_c_hermitian::Bool = false,
#     enforce_sigma_hermitian::Bool = false,
#     trace_normalize_sigma::Bool = false,

#     # c is local to this function.  Therefore resume is not supported unless c
#     # is later stored externally.
#     allow_resume_without_c_history::Bool = false,

#     verbose::Bool = true,
#     verbose_every::Int = 1,
#     return_internal_c::Bool = false,
# )
#     start_itr = Int(context.curr_itr)

#     n_sys = context.system.n_sys
#     n_itr = context.simulation_details.num_of_iteration
#     Δt    = context.simulation_details.Δt
#     ϵ     = context.ϵ_exci

#     σ   = context.σ              # physical readout output only
#     g   = context.g
#     gp  = context.g′
#     gpp = context.g″

#     if start_itr != 1 && !allow_resume_without_c_history
#         error("This v5 local-c implementation keeps c-history inside the function. Set context.curr_itr=1, or implement persistent context.c before resuming.")
#     end

#     hbar_f = Float64(hbar)
#     hbar_f > 0.0 || error("hbar must be positive")

#     method_sym = Symbol(lowercase(String(method)))
#     method_sym in (:euler, :rk2, :rk4) || error("Unsupported method $(method). Use :euler, :rk2, or :rk4.")

#     verbose_every_f = max(1, Int(verbose_every))
#     const_ZERO_C = 0.0 + 0.0im
#     const_Q_MAX = 2 * n_itr - 1

#     # ------------------------------------------------------------------
#     # Doubled q-grid:
#     #   full grid i      -> q = 2i - 1
#     #   half grid i+1/2  -> q = 2i
#     #   relative q       -> qΔ = q_t - q_s + 1
#     # ------------------------------------------------------------------
#     @inline q_full(i::Int) = 2 * i - 1
#     @inline q_half_after(i::Int) = 2 * i
#     @inline full_index_from_q(q::Int) = (q + 1) >>> 1
#     @inline half_index_from_q(q::Int) = q >>> 1

#     @inline function check_q(q::Int)
#         (1 <= q <= const_Q_MAX) || error("q-grid index out of range: $(q), allowed [1,$(const_Q_MAX)]")
#         return q
#     end

#     function _first_existing_property(obj, names::Tuple)
#         for name in names
#             if hasproperty(obj, name)
#                 return getfield(obj, name)
#             end
#         end
#         return nothing
#     end

#     g_half = _first_existing_property(context, (
#         :g_half, :g_half_shifted, :g_shifted_half, :g_mid, :g_midpoint,
#         Symbol("g__half"), Symbol("g_half_grid"), Symbol("g_half_shifted_grid"),
#     ))
#     gp_half = _first_existing_property(context, (
#         Symbol("g′_half"), Symbol("g′_half_shifted"), Symbol("g′_shifted_half"),
#         Symbol("g′_mid"), Symbol("g′_midpoint"), Symbol("gp_half"),
#         Symbol("gp_half_shifted"), Symbol("g_prime_half"), Symbol("g_prime_half_shifted"),
#     ))
#     gpp_half = _first_existing_property(context, (
#         Symbol("g″_half"), Symbol("g″_half_shifted"), Symbol("g″_shifted_half"),
#         Symbol("g″_mid"), Symbol("g″_midpoint"), Symbol("gpp_half"),
#         Symbol("gpp_half_shifted"), Symbol("g_doubleprime_half"), Symbol("g_doubleprime_half_shifted"),
#     ))

#     needs_half_grid = method_sym in (:rk2, :rk4)
#     if needs_half_grid && (g_half === nothing || gp_half === nothing || gpp_half === nothing)
#         missing = String[]
#         g_half === nothing && push!(missing, "g half-grid")
#         gp_half === nothing && push!(missing, "g′ half-grid")
#         gpp_half === nothing && push!(missing, "g″ half-grid")
#         error("$(method_sym) requires precomputed half-grid containers for " * join(missing, ", "))
#     end

#     @inline function _missing_half_grid_error(name::String)
#         error("Attempted to access $(name) at a half-grid q index, but the half-grid container is missing.")
#     end

#     @inline function G(q::Int, a::Int, b::Int, cidx::Int, d::Int)
#         check_q(q)
#         if isodd(q)
#             @inbounds return g[full_index_from_q(q), a, b, cidx, d]
#         else
#             g_half === nothing && _missing_half_grid_error("g")
#             @inbounds return g_half[half_index_from_q(q), a, b, cidx, d]
#         end
#     end

#     @inline function Gp(q::Int, a::Int, b::Int, cidx::Int, d::Int)
#         check_q(q)
#         if isodd(q)
#             @inbounds return gp[full_index_from_q(q), a, b, cidx, d]
#         else
#             gp_half === nothing && _missing_half_grid_error("g′")
#             @inbounds return gp_half[half_index_from_q(q), a, b, cidx, d]
#         end
#     end

#     @inline function Gpp(q::Int, a::Int, b::Int, cidx::Int, d::Int)
#         check_q(q)
#         if isodd(q)
#             @inbounds return gpp[full_index_from_q(q), a, b, cidx, d]
#         else
#             gpp_half === nothing && _missing_half_grid_error("g″")
#             @inbounds return gpp_half[half_index_from_q(q), a, b, cidx, d]
#         end
#     end

#     # q-time and signed-time g-family helpers.
#     # sign=+1: f_{ab,cd}(+τ)
#     # sign=-1: use the v5 negative-time identities:
#     #   g_abcd(-τ)    = conj(g_dcba(+τ))
#     #   g′_abcd(-τ)   = -conj(g′_dcba(+τ))
#     #   g″_abcd(-τ)   = conj(g″_dcba(+τ))
#     @inline function qtime(q::Int)
#         return 0.5 * (Float64(q) - 1.0) * Δt
#     end

#     @inline function Gv(q::Int, a::Int, b::Int, cidx::Int, d::Int, sign::Int)
#         return sign == 1 ? G(q, a, b, cidx, d) : conj(G(q, d, cidx, b, a))
#     end

#     @inline function Gpv(q::Int, a::Int, b::Int, cidx::Int, d::Int, sign::Int)
#         return sign == 1 ? Gp(q, a, b, cidx, d) : -conj(Gp(q, d, cidx, b, a))
#     end

#     @inline function Gppv(q::Int, a::Int, b::Int, cidx::Int, d::Int, sign::Int)
#         return sign == 1 ? Gpp(q, a, b, cidx, d) : conj(Gpp(q, d, cidx, b, a))
#     end

#     @inline function is_secular_pair(out_a::Int, out_b::Int, in_a::Int, in_b::Int)
#         if !use_secular
#             return true
#         end
#         return abs(((ϵ[out_a] - ϵ[out_b]) - (ϵ[in_a] - ϵ[in_b])) / hbar_f) <= secular_tol
#     end

#     @inline function c_local(c_t::AbstractMatrix, a::Int, b::Int)
#         return (use_population_closure && a != b) ? const_ZERO_C : c_t[a, b]
#     end

#     @inline function c_mem(c_t::AbstractMatrix, qs::Int, qt::Int, a::Int, b::Int, c_hist::Array{ComplexF64,3})
#         if use_population_closure && a != b
#             return const_ZERO_C
#         end
#         if qs == qt
#             return c_t[a, b]
#         end
#         isodd(qs) || error("non-endpoint memory q index must be a stored full-grid point, got q=$(qs)")
#         return c_hist[a, b, full_index_from_q(qs)]
#     end

#     @inline function n_outer_nodes(curr_itr::Int, qt::Int)
#         q_now = q_full(curr_itr)
#         return qt > q_now ? curr_itr + 1 : curr_itr
#     end

#     @inline function outer_node_coord(node_idx::Int, curr_itr::Int, qt::Int)
#         return node_idx <= curr_itr ? q_full(node_idx) : qt
#     end

#     @inline function ∫weight_to_t(node_idx::Int, curr_itr::Int, qt::Int)
#         n_nodes = n_outer_nodes(curr_itr, qt)
#         n_nodes <= 1 && return 0.0
#         q = outer_node_coord(node_idx, curr_itr, qt)
#         if node_idx == 1
#             q_next = outer_node_coord(node_idx + 1, curr_itr, qt)
#             return 0.25 * Δt * Float64(q_next - q)
#         elseif node_idx == n_nodes
#             q_prev = outer_node_coord(node_idx - 1, curr_itr, qt)
#             return 0.25 * Δt * Float64(q - q_prev)
#         else
#             q_prev = outer_node_coord(node_idx - 1, curr_itr, qt)
#             q_next = outer_node_coord(node_idx + 1, curr_itr, qt)
#             return 0.25 * Δt * Float64(q_next - q_prev)
#         end
#     end

#     # ------------------------------------------------------------------
#     # V5 branch-labeled coefficient functions.
#     # Generated from the v5 notebook expression trees, not hand-reduced.
#     # ------------------------------------------------------------------
#     @inline function coef_M_P_0(qt::Int, α::Int, β::Int)
#         return (((1.0/(hbar_f^2)) * ((-1.0 * Gpv(qt, α, α, α, α, 1)) + conj(Gpv(qt, α, α, β, β, 1)))) + (-1.0 * (1.0/(hbar_f^2)) * (conj(Gpv(qt, β, β, β, β, 1)) + (-1.0 * Gpv(qt, β, β, α, α, 1)))) + (1.0im * (1.0/hbar_f) * ϵ[β]) + (-1.0 * 1.0im * (1.0/hbar_f) * ϵ[α]))
#     end

#     @inline function coef_M_P_L(qt::Int, α::Int, β::Int, μ::Int)
#         return (exp(((((1.0/(hbar_f^2)) * Gv(qt, β, β, α, α, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, α, α, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1)))) + (((1.0/(hbar_f^2)) * Gv(qt, α, α, μ, μ, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, μ, μ, α, α, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, β, β, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, α, α, α, α, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, μ, μ, β, β, 1)))))) * ((1.0/(hbar_f^2)) * ((-1.0 * Gpv(qt, α, μ, μ, μ, 1)) + conj(Gpv(qt, μ, α, α, α, 1)))))
#     end

#     @inline function coef_M_P_R(qt::Int, α::Int, β::Int, ν::Int)
#         return (exp(((((1.0/(hbar_f^2)) * Gv(qt, β, β, α, α, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, α, α, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1)))) + (((1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, ν, ν, β, β, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, β, β, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, ν, ν, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, β, β, β, β, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, α, α, ν, ν, 1)))))) * (-1.0 * (1.0/(hbar_f^2)) * (conj(Gpv(qt, β, ν, ν, ν, 1)) + (-1.0 * Gpv(qt, ν, β, β, β, 1)))))
#     end

#     @inline function coef_K_LL_v5_raw(qs::Int, qΔ::Int, qt::Int, α::Int, β::Int, χ::Int, μ::Int)
#         return ((exp(((((1.0/(hbar_f^2)) * Gv(qt, β, β, α, α, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, α, α, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1)))) + (((1.0/(hbar_f^2)) * Gv(qs, μ, μ, μ, μ, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, β, β, μ, μ, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, μ, μ, β, β, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, β, β, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, μ, μ, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, β, β, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, μ, μ, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1)))) + (((1.0/(hbar_f^2)) * Gv(qΔ, α, α, χ, χ, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, α, α, μ, μ, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, μ, μ, μ, μ, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, χ, χ, α, α, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, χ, χ, μ, μ, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qΔ, χ, χ, μ, μ, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, μ, μ, α, α, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qΔ, α, α, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qΔ, χ, χ, χ, χ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, μ, μ, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, β, β, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, α, α, α, α, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, χ, χ, α, α, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, χ, χ, μ, μ, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, μ, μ, β, β, 1)))))) * ((exp((1.0im * (1.0/hbar_f) * ϵ[β] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/hbar_f) * ϵ[χ] * (qtime(qt) - qtime(qs)))) * -1.0 * (1.0/(hbar_f^2)) * Gppv(qΔ, α, χ, χ, μ, 1)) + (exp((1.0im * (1.0/hbar_f) * ϵ[β] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/hbar_f) * ϵ[χ] * (qtime(qt) - qtime(qs)))) * (1.0/(hbar_f^4)) * ((-1.0 * Gpv(qΔ, α, χ, χ, χ, 1)) + (-1.0 * Gpv(qt, α, χ, μ, μ, 1)) + Gpv(qΔ, α, χ, μ, μ, 1) + conj(Gpv(qt, χ, α, α, α, 1))) * ((-1.0 * Gpv(qs, χ, μ, μ, μ, 1)) + (-1.0 * Gpv(qΔ, χ, χ, χ, μ, 1)) + Gpv(qΔ, α, α, χ, μ, 1) + conj(Gpv(qs, μ, χ, α, α, 1)))))) + (exp(((((1.0/(hbar_f^2)) * Gv(qt, β, β, α, α, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, α, α, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1)))) + (((1.0/(hbar_f^2)) * Gv(qt, β, β, χ, χ, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, χ, χ, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, χ, χ, χ, χ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1)))) + (((1.0/(hbar_f^2)) * Gv(qt, α, α, χ, χ, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, χ, χ, α, α, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, β, β, χ, χ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, α, α, α, α, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, χ, χ, β, β, 1)))) + (((1.0/(hbar_f^2)) * Gv(qs, χ, χ, μ, μ, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, μ, μ, μ, μ, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qs, μ, μ, χ, χ, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, μ, μ, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, β, β, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, χ, χ, χ, χ, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, μ, μ, β, β, 1)))) + (((1.0/(hbar_f^2)) * Gv(qs, μ, μ, μ, μ, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, β, β, μ, μ, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, μ, μ, β, β, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, β, β, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, μ, μ, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, β, β, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, μ, μ, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1)))))) * (exp((1.0im * (1.0/hbar_f) * ϵ[β] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/hbar_f) * ϵ[χ] * (qtime(qt) - qtime(qs)))) * -1.0 * (1.0/(hbar_f^4)) * ((-1.0 * Gpv(qt, α, χ, χ, χ, 1)) + conj(Gpv(qt, χ, α, α, α, 1))) * ((-1.0 * Gpv(qs, χ, μ, μ, μ, 1)) + conj(Gpv(qs, μ, χ, χ, χ, 1))))))
#     end

#     @inline function coef_K_LR_v5_raw(qs::Int, qΔ::Int, qt::Int, α::Int, β::Int, χ::Int, ν::Int)
#         return ((exp(((((1.0/(hbar_f^2)) * Gv(qt, β, β, α, α, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, α, α, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1)))) + (((1.0/(hbar_f^2)) * Gv(qs, χ, χ, χ, χ, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, ν, ν, χ, χ, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, χ, χ, ν, ν, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, ν, ν, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, χ, χ, χ, χ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, ν, ν, χ, χ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, χ, χ, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, ν, ν, ν, ν, 1)))) + (((1.0/(hbar_f^2)) * Gv(qt, α, α, χ, χ, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, α, α, β, β, 1)) + ((1.0/(hbar_f^2)) * Gv(qΔ, χ, χ, β, β, 1)) + ((1.0/(hbar_f^2)) * Gv(qs, ν, ν, χ, χ, 1)) + ((1.0/(hbar_f^2)) * Gv(qs, ν, ν, β, β, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qΔ, α, α, ν, ν, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, χ, χ, α, α, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, ν, ν, ν, ν, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, β, β, α, α, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, β, β, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qΔ, α, α, β, β, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, χ, χ, β, β, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, ν, ν, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, ν, ν, χ, χ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, β, β, β, β, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, α, α, α, α, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, α, α, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qΔ, χ, χ, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, ν, ν, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, β, β, χ, χ, 1)))))) * ((exp((1.0im * (1.0/hbar_f) * ϵ[β] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/hbar_f) * ϵ[χ] * (qtime(qt) - qtime(qs)))) * -1.0 * (1.0/(hbar_f^2)) * conj(Gppv(qΔ, α, χ, ν, β, 1))) + (exp((1.0im * (1.0/hbar_f) * ϵ[β] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/hbar_f) * ϵ[χ] * (qtime(qt) - qtime(qs)))) * -1.0 * (1.0/(hbar_f^4)) * ((-1.0 * Gpv(qt, α, χ, β, β, 1)) + Gpv(qΔ, α, χ, β, β, 1) + (-1.0 * Gpv(qt, α, χ, χ, χ, 1)) + (-1.0 * conj(Gpv(qΔ, χ, α, ν, ν, 1))) + conj(Gpv(qt, χ, α, ν, ν, 1)) + conj(Gpv(qt, χ, α, α, α, 1))) * (conj(Gpv(qs, β, ν, ν, ν, 1)) + conj(Gpv(qs, β, ν, α, α, 1)) + conj(Gpv(qΔ, α, α, β, ν, 1)) + (-1.0 * Gpv(qs, ν, β, β, β, 1)) + (-1.0 * Gpv(qs, ν, β, χ, χ, 1)) + (-1.0 * conj(Gpv(qΔ, χ, χ, β, ν, 1))))))) + (exp(((((1.0/(hbar_f^2)) * Gv(qt, β, β, α, α, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, α, α, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1)))) + (((1.0/(hbar_f^2)) * Gv(qt, β, β, χ, χ, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, χ, χ, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, χ, χ, χ, χ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1)))) + (((1.0/(hbar_f^2)) * Gv(qt, α, α, χ, χ, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, χ, χ, α, α, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, β, β, χ, χ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, α, α, α, α, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, χ, χ, β, β, 1)))) + (((1.0/(hbar_f^2)) * Gv(qs, χ, χ, χ, χ, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, ν, ν, χ, χ, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, χ, χ, ν, ν, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, ν, ν, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, χ, χ, χ, χ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, ν, ν, χ, χ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, χ, χ, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, ν, ν, ν, ν, 1)))) + (((1.0/(hbar_f^2)) * Gv(qt, χ, χ, χ, χ, 1)) + ((1.0/(hbar_f^2)) * Gv(qs, ν, ν, β, β, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, ν, ν, ν, ν, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, β, β, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, ν, ν, χ, χ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, β, β, β, β, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, χ, χ, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, ν, ν, ν, ν, 1)))))) * (exp((1.0im * (1.0/hbar_f) * ϵ[β] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/hbar_f) * ϵ[χ] * (qtime(qt) - qtime(qs)))) * (1.0/(hbar_f^4)) * ((-1.0 * Gpv(qt, α, χ, χ, χ, 1)) + conj(Gpv(qt, χ, α, α, α, 1))) * (conj(Gpv(qs, β, ν, ν, ν, 1)) + (-1.0 * Gpv(qs, ν, β, β, β, 1))))))
#     end

#     @inline function coef_K_RL_v5_raw(qs::Int, qΔ::Int, qt::Int, α::Int, β::Int, μ::Int, ν::Int)
#         return ((exp(((((1.0/(hbar_f^2)) * Gv(qt, β, β, α, α, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, α, α, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1)))) + (((1.0/(hbar_f^2)) * Gv(qs, μ, μ, μ, μ, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, ν, ν, μ, μ, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, μ, μ, ν, ν, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, ν, ν, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, μ, μ, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, ν, ν, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, μ, μ, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, ν, ν, ν, ν, 1)))) + (((1.0/(hbar_f^2)) * Gv(qs, α, α, μ, μ, 1)) + ((1.0/(hbar_f^2)) * Gv(qs, α, α, β, β, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, μ, μ, μ, μ, 1)) + ((1.0/(hbar_f^2)) * Gv(qΔ, ν, ν, α, α, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, ν, ν, β, β, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qs, μ, μ, α, α, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, μ, μ, ν, ν, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, β, β, α, α, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qΔ, β, β, μ, μ, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, β, β, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, μ, μ, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, μ, μ, β, β, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, ν, ν, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qΔ, ν, ν, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, β, β, β, β, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, α, α, α, α, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, α, α, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, μ, μ, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qΔ, β, β, α, α, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, β, β, μ, μ, 1)))))) * ((exp((1.0im * (1.0/hbar_f) * ϵ[ν] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/hbar_f) * ϵ[α] * (qtime(qt) - qtime(qs)))) * -1.0 * (1.0/(hbar_f^2)) * conj(Gppv(qΔ, α, μ, ν, β, -1))) + (exp((1.0im * (1.0/hbar_f) * ϵ[ν] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/hbar_f) * ϵ[α] * (qtime(qt) - qtime(qs)))) * -1.0 * (1.0/(hbar_f^4)) * ((-1.0 * Gpv(qs, α, μ, β, β, 1)) + (-1.0 * Gpv(qs, α, μ, μ, μ, 1)) + (-1.0 * Gpv(qΔ, β, β, α, μ, 1)) + Gpv(qΔ, ν, ν, α, μ, 1) + conj(Gpv(qs, μ, α, ν, ν, 1)) + conj(Gpv(qs, μ, α, α, α, 1))) * (conj(Gpv(qt, β, ν, ν, ν, 1)) + (-1.0 * conj(Gpv(qΔ, β, ν, α, α, 1))) + conj(Gpv(qt, β, ν, α, α, 1)) + (-1.0 * Gpv(qt, ν, β, β, β, 1)) + (-1.0 * Gpv(qt, ν, β, μ, μ, 1)) + Gpv(qΔ, ν, β, μ, μ, 1))))) + (exp(((((1.0/(hbar_f^2)) * Gv(qt, ν, ν, α, α, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, α, α, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, ν, ν, ν, ν, 1)))) + (((1.0/(hbar_f^2)) * Gv(qt, β, β, α, α, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, α, α, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1)))) + (((1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, ν, ν, β, β, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, β, β, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, ν, ν, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, β, β, β, β, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, α, α, ν, ν, 1)))) + (((1.0/(hbar_f^2)) * Gv(qs, α, α, μ, μ, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, μ, μ, μ, μ, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qs, μ, μ, α, α, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, ν, ν, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, μ, μ, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, ν, ν, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, α, α, α, α, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, μ, μ, ν, ν, 1)))) + (((1.0/(hbar_f^2)) * Gv(qs, μ, μ, μ, μ, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, ν, ν, μ, μ, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, μ, μ, ν, ν, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, ν, ν, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, μ, μ, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, ν, ν, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, μ, μ, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, ν, ν, ν, ν, 1)))))) * (exp((1.0im * (1.0/hbar_f) * ϵ[ν] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/hbar_f) * ϵ[α] * (qtime(qt) - qtime(qs)))) * (1.0/(hbar_f^4)) * ((-1.0 * Gpv(qs, α, μ, μ, μ, 1)) + conj(Gpv(qs, μ, α, α, α, 1))) * (conj(Gpv(qt, β, ν, ν, ν, 1)) + (-1.0 * Gpv(qt, ν, β, β, β, 1))))))
#     end

#     @inline function coef_K_RR_v5_raw(qs::Int, qΔ::Int, qt::Int, α::Int, β::Int, χ::Int, ν::Int)
#         return ((exp(((((1.0/(hbar_f^2)) * Gv(qt, β, β, α, α, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, α, α, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1)))) + (((1.0/(hbar_f^2)) * Gv(qs, α, α, α, α, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, ν, ν, α, α, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, α, α, ν, ν, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, ν, ν, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, ν, ν, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, α, α, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, ν, ν, ν, ν, 1)))) + (((1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + ((1.0/(hbar_f^2)) * Gv(qs, ν, ν, β, β, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, χ, χ, β, β, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, ν, ν, ν, ν, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, χ, χ, ν, ν, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qΔ, χ, χ, ν, ν, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, β, β, ν, ν, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qΔ, β, β, χ, χ, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, ν, ν, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, χ, χ, β, β, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, β, β, β, β, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, α, α, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, ν, ν, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, χ, χ, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qΔ, χ, χ, χ, χ, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qΔ, β, β, ν, ν, 1)))))) * ((exp((1.0im * (1.0/hbar_f) * ϵ[χ] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/hbar_f) * ϵ[α] * (qtime(qt) - qtime(qs)))) * -1.0 * (1.0/(hbar_f^2)) * conj(Gppv(qΔ, ν, χ, χ, β, -1))) + (exp((1.0im * (1.0/hbar_f) * ϵ[χ] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/hbar_f) * ϵ[α] * (qtime(qt) - qtime(qs)))) * (1.0/(hbar_f^4)) * ((-1.0 * conj(Gpv(qΔ, β, χ, ν, ν, 1))) + conj(Gpv(qt, β, χ, ν, ν, 1)) + conj(Gpv(qΔ, β, χ, χ, χ, 1)) + (-1.0 * Gpv(qt, χ, β, β, β, 1))) * (conj(Gpv(qs, χ, ν, ν, ν, 1)) + conj(Gpv(qΔ, χ, χ, χ, ν, 1)) + (-1.0 * Gpv(qs, ν, χ, β, β, 1)) + (-1.0 * conj(Gpv(qΔ, β, β, χ, ν, 1))))))) + (exp(((((1.0/(hbar_f^2)) * Gv(qt, χ, χ, α, α, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, α, α, χ, χ, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, χ, χ, χ, χ, 1)))) + (((1.0/(hbar_f^2)) * Gv(qt, β, β, α, α, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, α, α, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1)))) + (((1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, χ, χ, β, β, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, β, β, χ, χ, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, χ, χ, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, β, β, β, β, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, α, α, χ, χ, 1)))) + (((1.0/(hbar_f^2)) * Gv(qs, α, α, α, α, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, ν, ν, α, α, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, α, α, ν, ν, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, ν, ν, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, ν, ν, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, α, α, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, ν, ν, ν, ν, 1)))) + (((1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + ((1.0/(hbar_f^2)) * Gv(qs, ν, ν, χ, χ, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, ν, ν, ν, ν, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, χ, χ, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, ν, ν, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, χ, χ, χ, χ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, α, α, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, ν, ν, ν, ν, 1)))))) * (exp((1.0im * (1.0/hbar_f) * ϵ[χ] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/hbar_f) * ϵ[α] * (qtime(qt) - qtime(qs)))) * -1.0 * (1.0/(hbar_f^4)) * (conj(Gpv(qs, χ, ν, ν, ν, 1)) + (-1.0 * Gpv(qs, ν, χ, χ, χ, 1))) * (conj(Gpv(qt, β, χ, χ, χ, 1)) + (-1.0 * Gpv(qt, χ, β, β, β, 1))))))
#     end

#     # ------------------------------------------------------------------
#     # V12 cyclic-canonical population direct patch.
#     # ------------------------------------------------------------------
#     @inline function coef_A12L_v12(qs::Int, qΔ::Int, qt::Int)
#         return ((exp((((1.0/(hbar_f^2)) * Gv(qt, 1, 1, 1, 1, 1)) + ((1.0/(hbar_f^2)) * Gv(qΔ, 1, 1, 2, 2, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qs, 1, 1, 1, 1, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qΔ, 2, 2, 1, 1, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, 1, 1, 1, 1, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qΔ, 1, 1, 1, 1, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qΔ, 2, 2, 2, 2, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, 1, 1, 1, 1, 1))))) * exp((1.0im * (1.0/(hbar_f)) * ϵ[1] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/(hbar_f)) * ϵ[2] * (qtime(qt) - qtime(qs)))) * (1.0/(hbar_f^2)) * Gppv(qΔ, 1, 2, 2, 1, 1)) + (exp((((1.0/(hbar_f^2)) * Gv(qt, 1, 1, 1, 1, 1)) + ((1.0/(hbar_f^2)) * Gv(qΔ, 1, 1, 2, 2, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qs, 1, 1, 1, 1, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qΔ, 2, 2, 1, 1, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, 1, 1, 1, 1, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qΔ, 1, 1, 1, 1, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qΔ, 2, 2, 2, 2, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, 1, 1, 1, 1, 1))))) * exp((1.0im * (1.0/(hbar_f)) * ϵ[1] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/(hbar_f)) * ϵ[2] * (qtime(qt) - qtime(qs)))) * -1.0 * (1.0/(hbar_f^4)) * ((-1.0 * Gpv(qΔ, 1, 2, 2, 2, 1)) + (-1.0 * Gpv(qt, 1, 2, 1, 1, 1)) + Gpv(qΔ, 1, 2, 1, 1, 1) + conj(Gpv(qt, 2, 1, 1, 1, 1))) * ((-1.0 * Gpv(qs, 2, 1, 1, 1, 1)) + (-1.0 * Gpv(qΔ, 2, 2, 2, 1, 1)) + Gpv(qΔ, 1, 1, 2, 1, 1) + conj(Gpv(qs, 1, 2, 1, 1, 1)))))
#     end

#     @inline function coef_A12R_v12(qs::Int, qΔ::Int, qt::Int)
#         return ((exp((((1.0/(hbar_f^2)) * Gv(qs, 1, 1, 1, 1, 1)) + ((1.0/(hbar_f^2)) * Gv(qΔ, 1, 1, 2, 2, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, 2, 2, 1, 1, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, 2, 2, 2, 2, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, 1, 1, 1, 1, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, 1, 1, 2, 2, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qΔ, 2, 2, 1, 1, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, 2, 2, 2, 2, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, 1, 1, 1, 1, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, 1, 1, 2, 2, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, 2, 2, 2, 2, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qΔ, 2, 2, 2, 2, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, 1, 1, 1, 1, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qΔ, 1, 1, 1, 1, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, 2, 2, 1, 1, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, 2, 2, 2, 2, 1))))) * exp((1.0im * (1.0/(hbar_f)) * ϵ[2] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/(hbar_f)) * ϵ[1] * (qtime(qt) - qtime(qs)))) * -1.0 * (1.0/(hbar_f^2)) * conj(Gppv(qΔ, 2, 1, 1, 2, 1))) + (exp((((1.0/(hbar_f^2)) * Gv(qs, 1, 1, 1, 1, 1)) + ((1.0/(hbar_f^2)) * Gv(qΔ, 1, 1, 2, 2, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, 2, 2, 1, 1, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, 2, 2, 2, 2, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, 1, 1, 1, 1, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, 1, 1, 2, 2, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qΔ, 2, 2, 1, 1, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, 2, 2, 2, 2, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, 1, 1, 1, 1, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, 1, 1, 2, 2, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, 2, 2, 2, 2, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qΔ, 2, 2, 2, 2, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, 1, 1, 1, 1, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qΔ, 1, 1, 1, 1, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, 2, 2, 1, 1, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, 2, 2, 2, 2, 1))))) * exp((1.0im * (1.0/(hbar_f)) * ϵ[2] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/(hbar_f)) * ϵ[1] * (qtime(qt) - qtime(qs)))) * -1.0 * (1.0/(hbar_f^4)) * ((-1.0 * Gpv(qt, 2, 1, 2, 2, 1)) + Gpv(qΔ, 2, 1, 2, 2, 1) + (-1.0 * Gpv(qt, 2, 1, 1, 1, 1)) + (-1.0 * conj(Gpv(qΔ, 1, 2, 1, 1, 1))) + conj(Gpv(qt, 1, 2, 1, 1, 1)) + conj(Gpv(qt, 1, 2, 2, 2, 1))) * (conj(Gpv(qs, 2, 1, 1, 1, 1)) + conj(Gpv(qs, 2, 1, 2, 2, 1)) + conj(Gpv(qΔ, 2, 2, 2, 1, 1)) + (-1.0 * Gpv(qs, 1, 2, 2, 2, 1)) + (-1.0 * Gpv(qs, 1, 2, 1, 1, 1)) + (-1.0 * conj(Gpv(qΔ, 1, 1, 2, 1, 1))))))
#     end

#     @inline function coef_A21L_v12(qs::Int, qΔ::Int, qt::Int)
#         return ((exp((((1.0/(hbar_f^2)) * Gv(qΔ, 2, 2, 1, 1, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, 2, 2, 2, 2, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qΔ, 1, 1, 2, 2, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, 2, 2, 2, 2, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qΔ, 1, 1, 1, 1, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, 2, 2, 2, 2, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qΔ, 2, 2, 2, 2, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, 2, 2, 2, 2, 1))))) * exp((1.0im * (1.0/(hbar_f)) * ϵ[2] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/(hbar_f)) * ϵ[1] * (qtime(qt) - qtime(qs)))) * (1.0/(hbar_f^2)) * Gppv(qΔ, 2, 1, 1, 2, 1)) + (exp((((1.0/(hbar_f^2)) * Gv(qΔ, 2, 2, 1, 1, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, 2, 2, 2, 2, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qΔ, 1, 1, 2, 2, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, 2, 2, 2, 2, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qΔ, 1, 1, 1, 1, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, 2, 2, 2, 2, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qΔ, 2, 2, 2, 2, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, 2, 2, 2, 2, 1))))) * exp((1.0im * (1.0/(hbar_f)) * ϵ[2] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/(hbar_f)) * ϵ[1] * (qtime(qt) - qtime(qs)))) * -1.0 * (1.0/(hbar_f^4)) * ((-1.0 * Gpv(qs, 1, 2, 2, 2, 1)) + (-1.0 * Gpv(qΔ, 1, 1, 1, 2, 1)) + Gpv(qΔ, 2, 2, 1, 2, 1) + conj(Gpv(qs, 2, 1, 2, 2, 1))) * ((-1.0 * Gpv(qΔ, 2, 1, 1, 1, 1)) + (-1.0 * Gpv(qt, 2, 1, 2, 2, 1)) + Gpv(qΔ, 2, 1, 2, 2, 1) + conj(Gpv(qt, 1, 2, 2, 2, 1)))))
#     end

#     @inline function coef_A21R_v12(qs::Int, qΔ::Int, qt::Int)
#         return ((exp((((1.0/(hbar_f^2)) * Gv(qt, 1, 1, 1, 1, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, 1, 1, 2, 2, 1)) + ((1.0/(hbar_f^2)) * Gv(qΔ, 2, 2, 1, 1, 1)) + ((1.0/(hbar_f^2)) * Gv(qs, 2, 2, 2, 2, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qs, 1, 1, 1, 1, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qΔ, 1, 1, 2, 2, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, 2, 2, 1, 1, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, 2, 2, 2, 2, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, 1, 1, 1, 1, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qΔ, 1, 1, 1, 1, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, 2, 2, 1, 1, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, 2, 2, 2, 2, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, 1, 1, 1, 1, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, 1, 1, 2, 2, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, 2, 2, 2, 2, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qΔ, 2, 2, 2, 2, 1))))) * exp((1.0im * (1.0/(hbar_f)) * ϵ[1] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/(hbar_f)) * ϵ[2] * (qtime(qt) - qtime(qs)))) * -1.0 * (1.0/(hbar_f^2)) * conj(Gppv(qΔ, 1, 2, 2, 1, 1))) + (exp((((1.0/(hbar_f^2)) * Gv(qt, 1, 1, 1, 1, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, 1, 1, 2, 2, 1)) + ((1.0/(hbar_f^2)) * Gv(qΔ, 2, 2, 1, 1, 1)) + ((1.0/(hbar_f^2)) * Gv(qs, 2, 2, 2, 2, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qs, 1, 1, 1, 1, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qΔ, 1, 1, 2, 2, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, 2, 2, 1, 1, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, 2, 2, 2, 2, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, 1, 1, 1, 1, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qΔ, 1, 1, 1, 1, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, 2, 2, 1, 1, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, 2, 2, 2, 2, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, 1, 1, 1, 1, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, 1, 1, 2, 2, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, 2, 2, 2, 2, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qΔ, 2, 2, 2, 2, 1))))) * exp((1.0im * (1.0/(hbar_f)) * ϵ[1] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/(hbar_f)) * ϵ[2] * (qtime(qt) - qtime(qs)))) * -1.0 * (1.0/(hbar_f^4)) * ((-1.0 * Gpv(qt, 1, 2, 1, 1, 1)) + Gpv(qΔ, 1, 2, 1, 1, 1) + (-1.0 * Gpv(qt, 1, 2, 2, 2, 1)) + (-1.0 * conj(Gpv(qΔ, 2, 1, 2, 2, 1))) + conj(Gpv(qt, 2, 1, 2, 2, 1)) + conj(Gpv(qt, 2, 1, 1, 1, 1))) * (conj(Gpv(qs, 1, 2, 2, 2, 1)) + conj(Gpv(qs, 1, 2, 1, 1, 1)) + conj(Gpv(qΔ, 1, 1, 1, 2, 1)) + (-1.0 * Gpv(qs, 2, 1, 1, 1, 1)) + (-1.0 * Gpv(qs, 2, 1, 2, 2, 1)) + (-1.0 * conj(Gpv(qΔ, 2, 2, 1, 2, 1))))))
#     end

#     @inline function coef_K_LL_dir_v5_raw(qs::Int, qΔ::Int, qt::Int, α::Int, β::Int, χ::Int, μ::Int)
#         return ((-1.0 * (1.0/(hbar_f^2)) * Gppv(qΔ, α, χ, χ, μ, 1) * exp((1.0im * (1.0/(hbar_f)) * ϵ[β] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/(hbar_f)) * ϵ[χ] * (qtime(qt) - qtime(qs)))) * exp((((1.0/(hbar_f^2)) * Gv(qΔ, α, α, χ, χ, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, α, α, μ, μ, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, β, β, α, α, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, α, α, β, β, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, β, β, β, β, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, χ, χ, α, α, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, χ, χ, μ, μ, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qΔ, χ, χ, μ, μ, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, μ, μ, α, α, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qΔ, α, α, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, β, β, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qΔ, χ, χ, χ, χ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, α, α, α, α, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, χ, χ, α, α, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, χ, χ, μ, μ, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, μ, μ, β, β, 1)))))) + ((1.0/(hbar_f^4)) * ((-1.0 * Gpv(qΔ, α, χ, χ, χ, 1)) + (-1.0 * Gpv(qt, α, χ, μ, μ, 1)) + Gpv(qΔ, α, χ, μ, μ, 1) + conj(Gpv(qt, χ, α, α, α, 1))) * ((-1.0 * Gpv(qΔ, χ, χ, χ, μ, 1)) + (-1.0 * Gpv(qs, χ, μ, μ, μ, 1)) + Gpv(qΔ, α, α, χ, μ, 1) + conj(Gpv(qs, μ, χ, α, α, 1))) * exp((1.0im * (1.0/(hbar_f)) * ϵ[β] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/(hbar_f)) * ϵ[χ] * (qtime(qt) - qtime(qs)))) * exp((((1.0/(hbar_f^2)) * Gv(qΔ, α, α, χ, χ, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, α, α, μ, μ, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, β, β, α, α, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, α, α, β, β, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, β, β, β, β, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, χ, χ, α, α, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, χ, χ, μ, μ, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qΔ, χ, χ, μ, μ, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, μ, μ, α, α, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qΔ, α, α, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, β, β, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qΔ, χ, χ, χ, χ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, α, α, α, α, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, χ, χ, α, α, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, χ, χ, μ, μ, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, μ, μ, β, β, 1)))))))
#     end

#     @inline function coef_K_LR_dir_v5_raw(qs::Int, qΔ::Int, qt::Int, α::Int, β::Int, χ::Int, ν::Int)
#         return ((-1.0 * (1.0/(hbar_f^2)) * conj(Gppv(qΔ, α, χ, ν, β, 1)) * exp((1.0im * (1.0/(hbar_f)) * ϵ[β] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/(hbar_f)) * ϵ[χ] * (qtime(qt) - qtime(qs)))) * exp((((1.0/(hbar_f^2)) * Gv(qt, α, α, β, β, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, α, α, χ, χ, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, β, β, α, α, 1)) + ((1.0/(hbar_f^2)) * Gv(qΔ, χ, χ, β, β, 1)) + ((1.0/(hbar_f^2)) * Gv(qs, χ, χ, χ, χ, 1)) + ((1.0/(hbar_f^2)) * Gv(qs, ν, ν, β, β, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, α, α, β, β, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qΔ, α, α, ν, ν, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, β, β, α, α, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, β, β, ν, ν, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, χ, χ, α, α, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, χ, χ, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qΔ, α, α, β, β, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, β, β, β, β, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, χ, χ, β, β, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, χ, χ, χ, χ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, ν, ν, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, α, α, α, α, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, α, α, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, β, β, χ, χ, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, χ, χ, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qΔ, χ, χ, ν, ν, 1)))))) + (-1.0 * (1.0/(hbar_f^4)) * ((-1.0 * Gpv(qt, α, χ, β, β, 1)) + (-1.0 * Gpv(qt, α, χ, χ, χ, 1)) + (-1.0 * conj(Gpv(qΔ, χ, α, ν, ν, 1))) + Gpv(qΔ, α, χ, β, β, 1) + conj(Gpv(qt, χ, α, α, α, 1)) + conj(Gpv(qt, χ, α, ν, ν, 1))) * ((-1.0 * Gpv(qs, ν, β, β, β, 1)) + (-1.0 * Gpv(qs, ν, β, χ, χ, 1)) + (-1.0 * conj(Gpv(qΔ, χ, χ, β, ν, 1))) + conj(Gpv(qΔ, α, α, β, ν, 1)) + conj(Gpv(qs, β, ν, α, α, 1)) + conj(Gpv(qs, β, ν, ν, ν, 1))) * exp((1.0im * (1.0/(hbar_f)) * ϵ[β] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/(hbar_f)) * ϵ[χ] * (qtime(qt) - qtime(qs)))) * exp((((1.0/(hbar_f^2)) * Gv(qt, α, α, β, β, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, α, α, χ, χ, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, β, β, α, α, 1)) + ((1.0/(hbar_f^2)) * Gv(qΔ, χ, χ, β, β, 1)) + ((1.0/(hbar_f^2)) * Gv(qs, χ, χ, χ, χ, 1)) + ((1.0/(hbar_f^2)) * Gv(qs, ν, ν, β, β, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, α, α, β, β, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qΔ, α, α, ν, ν, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, β, β, α, α, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, β, β, ν, ν, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, χ, χ, α, α, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, χ, χ, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qΔ, α, α, β, β, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, β, β, β, β, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, χ, χ, β, β, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, χ, χ, χ, χ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, ν, ν, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, α, α, α, α, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, α, α, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, β, β, χ, χ, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, χ, χ, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qΔ, χ, χ, ν, ν, 1)))))))
#     end

#     @inline function coef_K_RL_dir_v5_raw(qs::Int, qΔ::Int, qt::Int, α::Int, β::Int, μ::Int, ν::Int)
#         return ((-1.0 * (1.0/(hbar_f^2)) * conj(Gppv(qΔ, α, μ, ν, β, -1)) * exp((1.0im * (1.0/(hbar_f)) * ϵ[ν] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/(hbar_f)) * ϵ[α] * (qtime(qt) - qtime(qs)))) * exp((((1.0/(hbar_f^2)) * Gv(qs, α, α, β, β, 1)) + ((1.0/(hbar_f^2)) * Gv(qs, α, α, μ, μ, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, β, β, α, α, 1)) + ((1.0/(hbar_f^2)) * Gv(qΔ, ν, ν, α, α, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, ν, ν, β, β, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, ν, ν, μ, μ, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, α, α, β, β, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, β, β, α, α, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qΔ, β, β, μ, μ, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, β, β, ν, ν, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, μ, μ, α, α, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, ν, ν, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, β, β, β, β, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, μ, μ, β, β, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, ν, ν, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, ν, ν, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qΔ, ν, ν, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, α, α, α, α, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, α, α, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qΔ, β, β, α, α, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, β, β, μ, μ, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, ν, ν, ν, ν, 1)))))) + (-1.0 * (1.0/(hbar_f^4)) * ((-1.0 * Gpv(qs, α, μ, β, β, 1)) + (-1.0 * Gpv(qs, α, μ, μ, μ, 1)) + (-1.0 * Gpv(qΔ, β, β, α, μ, 1)) + Gpv(qΔ, ν, ν, α, μ, 1) + conj(Gpv(qs, μ, α, α, α, 1)) + conj(Gpv(qs, μ, α, ν, ν, 1))) * ((-1.0 * Gpv(qt, ν, β, β, β, 1)) + (-1.0 * Gpv(qt, ν, β, μ, μ, 1)) + (-1.0 * conj(Gpv(qΔ, β, ν, α, α, 1))) + Gpv(qΔ, ν, β, μ, μ, 1) + conj(Gpv(qt, β, ν, α, α, 1)) + conj(Gpv(qt, β, ν, ν, ν, 1))) * exp((1.0im * (1.0/(hbar_f)) * ϵ[ν] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/(hbar_f)) * ϵ[α] * (qtime(qt) - qtime(qs)))) * exp((((1.0/(hbar_f^2)) * Gv(qs, α, α, β, β, 1)) + ((1.0/(hbar_f^2)) * Gv(qs, α, α, μ, μ, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, β, β, α, α, 1)) + ((1.0/(hbar_f^2)) * Gv(qΔ, ν, ν, α, α, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, ν, ν, β, β, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, ν, ν, μ, μ, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, α, α, β, β, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, β, β, α, α, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qΔ, β, β, μ, μ, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, β, β, ν, ν, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, μ, μ, α, α, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, ν, ν, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, β, β, β, β, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, μ, μ, β, β, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, ν, ν, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, ν, ν, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qΔ, ν, ν, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, α, α, α, α, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, α, α, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qΔ, β, β, α, α, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, β, β, μ, μ, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, ν, ν, ν, ν, 1)))))))
#     end

#     @inline function coef_K_RR_dir_v5_raw(qs::Int, qΔ::Int, qt::Int, α::Int, β::Int, χ::Int, ν::Int)
#         return ((-1.0 * (1.0/(hbar_f^2)) * conj(Gppv(qΔ, ν, χ, χ, β, -1)) * exp((1.0im * (1.0/(hbar_f)) * ϵ[χ] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/(hbar_f)) * ϵ[α] * (qtime(qt) - qtime(qs)))) * exp((((1.0/(hbar_f^2)) * Gv(qs, α, α, α, α, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, β, β, α, α, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, χ, χ, β, β, 1)) + ((1.0/(hbar_f^2)) * Gv(qs, ν, ν, β, β, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, α, α, β, β, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qΔ, β, β, χ, χ, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, β, β, ν, ν, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, χ, χ, ν, ν, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qΔ, χ, χ, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, β, β, β, β, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, χ, χ, β, β, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, ν, ν, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, α, α, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qΔ, β, β, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qΔ, χ, χ, χ, χ, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, χ, χ, ν, ν, 1)))))) + ((1.0/(hbar_f^4)) * ((-1.0 * Gpv(qt, χ, β, β, β, 1)) + (-1.0 * conj(Gpv(qΔ, β, χ, ν, ν, 1))) + conj(Gpv(qΔ, β, χ, χ, χ, 1)) + conj(Gpv(qt, β, χ, ν, ν, 1))) * ((-1.0 * Gpv(qs, ν, χ, β, β, 1)) + (-1.0 * conj(Gpv(qΔ, β, β, χ, ν, 1))) + conj(Gpv(qΔ, χ, χ, χ, ν, 1)) + conj(Gpv(qs, χ, ν, ν, ν, 1))) * exp((1.0im * (1.0/(hbar_f)) * ϵ[χ] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/(hbar_f)) * ϵ[α] * (qtime(qt) - qtime(qs)))) * exp((((1.0/(hbar_f^2)) * Gv(qs, α, α, α, α, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, β, β, α, α, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, χ, χ, β, β, 1)) + ((1.0/(hbar_f^2)) * Gv(qs, ν, ν, β, β, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, α, α, β, β, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qΔ, β, β, χ, χ, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, β, β, ν, ν, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, χ, χ, ν, ν, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qΔ, χ, χ, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, β, β, β, β, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, χ, χ, β, β, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, ν, ν, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, α, α, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qΔ, β, β, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qΔ, χ, χ, χ, χ, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, χ, χ, ν, ν, 1)))))))
#     end

#     @inline function coef_A_L_v12(qs::Int, qΔ::Int, qt::Int, from::Int, to::Int)
#         if from == 1 && to == 2
#             return coef_A12L_v12(qs, qΔ, qt)
#         elseif from == 2 && to == 1
#             return coef_A21L_v12(qs, qΔ, qt)
#         end
#         return const_ZERO_C
#     end

#     @inline function coef_A_R_v12(qs::Int, qΔ::Int, qt::Int, from::Int, to::Int)
#         if from == 1 && to == 2
#             return coef_A12R_v12(qs, qΔ, qt)
#         elseif from == 2 && to == 1
#             return coef_A21R_v12(qs, qΔ, qt)
#         end
#         return const_ZERO_C
#     end

#     @inline function coef_K_LL(qs::Int, qΔ::Int, qt::Int, α::Int, β::Int, χ::Int, μ::Int)
#         val = coef_K_LL_v5_raw(qs, qΔ, qt, α, β, χ, μ)
#         # v12: for two-level population-output/direct-population-input channels,
#         # replace the independently closed direct primitive by the cyclic-canonical
#         # source-rate component. Q(t), Q(s) subtraction pieces are left unchanged:
#         #     K_v12 = K_v5_raw - K_v5_direct + K_canonical_direct.
#         if use_v12_canonical_population_direct && n_sys == 2 && α == β && μ == α && χ != α
#             val += (-coef_A_L_v12(qs, qΔ, qt, α, χ)) - coef_K_LL_dir_v5_raw(qs, qΔ, qt, α, β, χ, μ)
#         end
#         return val
#     end

#     @inline function coef_K_LR(qs::Int, qΔ::Int, qt::Int, α::Int, β::Int, χ::Int, ν::Int)
#         val = coef_K_LR_v5_raw(qs, qΔ, qt, α, β, χ, ν)
#         if use_v12_canonical_population_direct && n_sys == 2 && α == β && χ == ν && χ != α
#             val += (+coef_A_R_v12(qs, qΔ, qt, χ, α)) - coef_K_LR_dir_v5_raw(qs, qΔ, qt, α, β, χ, ν)
#         end
#         return val
#     end

#     @inline function coef_K_RL(qs::Int, qΔ::Int, qt::Int, α::Int, β::Int, μ::Int, ν::Int)
#         val = coef_K_RL_v5_raw(qs, qΔ, qt, α, β, μ, ν)
#         if use_v12_canonical_population_direct && n_sys == 2 && α == β && μ == ν && μ != α
#             val += (+coef_A_L_v12(qs, qΔ, qt, μ, α)) - coef_K_RL_dir_v5_raw(qs, qΔ, qt, α, β, μ, ν)
#         end
#         return val
#     end

#     @inline function coef_K_RR(qs::Int, qΔ::Int, qt::Int, α::Int, β::Int, χ::Int, ν::Int)
#         val = coef_K_RR_v5_raw(qs, qΔ, qt, α, β, χ, ν)
#         if use_v12_canonical_population_direct && n_sys == 2 && α == β && ν == α && χ != α
#             val += (-coef_A_R_v12(qs, qΔ, qt, α, χ)) - coef_K_RR_dir_v5_raw(qs, qΔ, qt, α, β, χ, ν)
#         end
#         return val
#     end

#     @inline function coef_Rσ_L(qs::Int, qΔ::Int, qt::Int, α::Int, β::Int, μ::Int)
#         return ((exp(((((1.0/(hbar_f^2)) * Gv(qs, μ, μ, μ, μ, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, β, β, μ, μ, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, μ, μ, β, β, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, β, β, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, μ, μ, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, β, β, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, μ, μ, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1)))) + (((1.0/(hbar_f^2)) * Gv(qt, μ, μ, μ, μ, 1)) + ((1.0/(hbar_f^2)) * Gv(qΔ, β, β, α, α, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qs, α, α, μ, μ, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qΔ, α, α, μ, μ, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, α, α, β, β, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, μ, μ, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qΔ, α, α, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, μ, μ, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qΔ, β, β, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, α, α, μ, μ, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, α, α, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, μ, μ, β, β, 1)))))) * (exp((1.0im * (1.0/hbar_f) * ϵ[β] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/hbar_f) * ϵ[α] * (qtime(qt) - qtime(qs)))) * (1.0/(hbar_f^2)) * ((-1.0 * Gpv(qs, α, μ, μ, μ, 1)) + (-1.0 * Gpv(qΔ, α, α, α, μ, 1)) + Gpv(qΔ, β, β, α, μ, 1) + conj(Gpv(qs, μ, α, β, β, 1))))) + (exp(((((1.0/(hbar_f^2)) * Gv(qt, β, β, α, α, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, α, α, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1)))) + (((1.0/(hbar_f^2)) * Gv(qs, α, α, μ, μ, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, μ, μ, μ, μ, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qs, μ, μ, α, α, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, μ, μ, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, β, β, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, α, α, α, α, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, μ, μ, β, β, 1)))) + (((1.0/(hbar_f^2)) * Gv(qs, μ, μ, μ, μ, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, β, β, μ, μ, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, μ, μ, β, β, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, β, β, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, μ, μ, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, β, β, μ, μ, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, μ, μ, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1)))))) * (exp((1.0im * (1.0/hbar_f) * ϵ[β] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/hbar_f) * ϵ[α] * (qtime(qt) - qtime(qs)))) * -1.0 * (1.0/(hbar_f^2)) * ((-1.0 * Gpv(qs, α, μ, μ, μ, 1)) + conj(Gpv(qs, μ, α, α, α, 1))))))
#     end

#     @inline function coef_Rσ_R(qs::Int, qΔ::Int, qt::Int, α::Int, β::Int, ν::Int)
#         return ((exp(((((1.0/(hbar_f^2)) * Gv(qt, β, β, α, α, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, α, α, β, β, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, β, β, β, β, 1)))) + (((1.0/(hbar_f^2)) * Gv(qs, α, α, α, α, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, ν, ν, α, α, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, α, α, ν, ν, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, ν, ν, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, ν, ν, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, α, α, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, ν, ν, ν, ν, 1)))) + (((1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + ((1.0/(hbar_f^2)) * Gv(qs, ν, ν, β, β, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, ν, ν, ν, ν, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, β, β, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, ν, ν, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, β, β, β, β, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, α, α, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, ν, ν, ν, ν, 1)))))) * (exp((1.0im * (1.0/hbar_f) * ϵ[β] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/hbar_f) * ϵ[α] * (qtime(qt) - qtime(qs)))) * (1.0/(hbar_f^2)) * (conj(Gpv(qs, β, ν, ν, ν, 1)) + (-1.0 * Gpv(qs, ν, β, β, β, 1))))) + (exp(((((1.0/(hbar_f^2)) * Gv(qs, α, α, α, α, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, ν, ν, α, α, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qt, α, α, ν, ν, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, ν, ν, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, α, α, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, ν, ν, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, α, α, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, ν, ν, ν, ν, 1)))) + (((1.0/(hbar_f^2)) * Gv(qs, ν, ν, α, α, 1)) + ((1.0/(hbar_f^2)) * Gv(qt, β, β, α, α, 1)) + ((1.0/(hbar_f^2)) * conj(Gv(qΔ, α, α, β, β, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qt, ν, ν, ν, ν, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qs, β, β, ν, ν, 1))) + ((1.0/(hbar_f^2)) * conj(Gv(qΔ, β, β, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qt, ν, ν, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * Gv(qs, β, β, α, α, 1)) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qΔ, α, α, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qs, ν, ν, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qt, β, β, ν, ν, 1))) + (-1.0 * (1.0/(hbar_f^2)) * conj(Gv(qΔ, β, β, β, β, 1)))))) * (exp((1.0im * (1.0/hbar_f) * ϵ[β] * (qtime(qt) - qtime(qs)))) * exp((-1.0 * 1.0im * (1.0/hbar_f) * ϵ[α] * (qtime(qt) - qtime(qs)))) * -1.0 * (1.0/(hbar_f^2)) * (conj(Gpv(qs, β, ν, ν, ν, 1)) + conj(Gpv(qΔ, β, β, β, ν, 1)) + (-1.0 * Gpv(qs, ν, β, α, α, 1)) + (-1.0 * conj(Gpv(qΔ, α, α, β, ν, 1)))))))
#     end


#     @inline function enforce_hermiticity!(mat::AbstractMatrix)
#         for i in 1:n_sys
#             mat[i, i] = real(mat[i, i]) + 0.0im
#         end
#         for i in 1:(n_sys - 1), j in (i + 1):n_sys
#             z = 0.5 * (mat[i, j] + conj(mat[j, i]))
#             mat[i, j] = z
#             mat[j, i] = conj(z)
#         end
#         return mat
#     end

#     @inline function enforce_population_closure!(mat::AbstractMatrix)
#         for j in 1:n_sys, i in 1:n_sys
#             mat[i, j] = (i == j) ? (real(mat[i, i]) + 0.0im) : const_ZERO_C
#         end
#         return mat
#     end

#     @inline function trace_normalize!(mat::AbstractMatrix)
#         tr = zero(ComplexF64)
#         for i in 1:n_sys
#             tr += mat[i, i]
#         end
#         if abs(tr) > 0
#             mat ./= tr
#         end
#         return mat
#     end

#     Base.@noinline function calc_c_rhs_element!(rhs_mat::AbstractMatrix, curr_itr::Int, qt::Int, c_t::AbstractMatrix, c_hist::Array{ComplexF64,3}, α::Int, β::Int)
#         if use_population_closure && α != β
#             rhs_mat[α, β] = const_ZERO_C
#             return nothing
#         end

#         rhs = const_ZERO_C

#         if include_c_L0P_local
#             rhs += coef_M_P_0(qt, α, β) * c_local(c_t, α, β)
#         end

#         if include_c_PL1P_local
#             for μ in 1:n_sys
#                 μ == α && continue
#                 if is_secular_pair(α, β, μ, β)
#                     rhs += coef_M_P_L(qt, α, β, μ) * c_local(c_t, μ, β)
#                 end
#             end
#             for ν in 1:n_sys
#                 ν == β && continue
#                 if is_secular_pair(α, β, α, ν)
#                     rhs += coef_M_P_R(qt, α, β, ν) * c_local(c_t, α, ν)
#                 end
#             end
#         end

#         if include_c_memory_G0 && qt > 1
#             integral_LL = const_ZERO_C
#             integral_LR = const_ZERO_C
#             integral_RL = const_ZERO_C
#             integral_RR = const_ZERO_C
#             n_s = n_outer_nodes(curr_itr, qt)
#             for s_node in 1:n_s
#                 qs = outer_node_coord(s_node, curr_itr, qt)
#                 qΔ = qt - qs + 1
#                 w = ∫weight_to_t(s_node, curr_itr, qt)
#                 w == 0.0 && continue

#                 branch_LL = const_ZERO_C
#                 branch_LR = const_ZERO_C
#                 branch_RL = const_ZERO_C
#                 branch_RR = const_ZERO_C

#                 # LL: input (μ, β), first intermediate χ, second insertion χ -> μ.
#                 if include_c_memory_LL
#                     for χ in 1:n_sys
#                         χ == α && continue
#                         for μ in 1:n_sys
#                             μ == χ && continue
#                             if is_secular_pair(α, β, μ, β)
#                                 branch_LL += coef_K_LL(qs, qΔ, qt, α, β, χ, μ) * c_mem(c_t, qs, qt, μ, β, c_hist)
#                             end
#                         end
#                     end
#                 end

#                 # LR: input (χ, ν).
#                 if include_c_memory_LR
#                     for χ in 1:n_sys
#                         χ == α && continue
#                         for ν in 1:n_sys
#                             ν == β && continue
#                             if is_secular_pair(α, β, χ, ν)
#                                 branch_LR += coef_K_LR(qs, qΔ, qt, α, β, χ, ν) * c_mem(c_t, qs, qt, χ, ν, c_hist)
#                             end
#                         end
#                     end
#                 end

#                 # RL: input (μ, ν).
#                 if include_c_memory_RL
#                     for μ in 1:n_sys
#                         μ == α && continue
#                         for ν in 1:n_sys
#                             ν == β && continue
#                             if is_secular_pair(α, β, μ, ν)
#                                 branch_RL += coef_K_RL(qs, qΔ, qt, α, β, μ, ν) * c_mem(c_t, qs, qt, μ, ν, c_hist)
#                             end
#                         end
#                     end
#                 end

#                 # RR: input (α, ν), first intermediate χ, second insertion ν -> χ.
#                 if include_c_memory_RR
#                     for χ in 1:n_sys
#                         χ == β && continue
#                         for ν in 1:n_sys
#                             ν == χ && continue
#                             if is_secular_pair(α, β, α, ν)
#                                 branch_RR += coef_K_RR(qs, qΔ, qt, α, β, χ, ν) * c_mem(c_t, qs, qt, α, ν, c_hist)
#                             end
#                         end
#                     end
#                 end

#                 integral_LL += w * branch_LL
#                 integral_LR += w * branch_LR
#                 integral_RL += w * branch_RL
#                 integral_RR += w * branch_RR
#             end

#             signed_LL = c_memory_global_sign * c_memory_sign_LL * integral_LL
#             signed_LR = c_memory_global_sign * c_memory_sign_LR * integral_LR
#             signed_RL = c_memory_global_sign * c_memory_sign_RL * integral_RL
#             signed_RR = c_memory_global_sign * c_memory_sign_RR * integral_RR
#             rhs += signed_LL + signed_LR + signed_RL + signed_RR

#             if return_branch_memory_audit && qt == q_full(curr_itr)
#                 branch_memory_audit[α, β, curr_itr, 1] = signed_LL
#                 branch_memory_audit[α, β, curr_itr, 2] = signed_LR
#                 branch_memory_audit[α, β, curr_itr, 3] = signed_RL
#                 branch_memory_audit[α, β, curr_itr, 4] = signed_RR
#             end
#         end

#         rhs_mat[α, β] = rhs
#         return nothing
#     end

#     Base.@noinline function calc_c_rhs!(rhs_mat::AbstractMatrix, curr_itr::Int, qt::Int, c_t::AbstractMatrix, c_hist::Array{ComplexF64,3})
#         fill!(rhs_mat, const_ZERO_C)
#         n_components = n_sys * n_sys
#         heavy = include_c_memory_G0
#         if use_threads && Threads.nthreads() > 1 && (heavy || n_components >= max(16, 4 * Threads.nthreads()))
#             Threads.@threads for linear_idx in 1:n_components
#                 α = ((linear_idx - 1) % n_sys) + 1
#                 β = ((linear_idx - 1) ÷ n_sys) + 1
#                 calc_c_rhs_element!(rhs_mat, curr_itr, qt, c_t, c_hist, α, β)
#             end
#         else
#             for β in 1:n_sys, α in 1:n_sys
#                 calc_c_rhs_element!(rhs_mat, curr_itr, qt, c_t, c_hist, α, β)
#             end
#         end
#         return rhs_mat
#     end


#     Base.@noinline function compute_memory_kernel_rowsum_audit!(kernel_rowsum_audit::Array{ComplexF64,4}, kernel_output_audit::Array{ComplexF64,5}, curr_itr::Int, qt::Int)
#         # Stores, for this full-grid time t, the integrated row sum
#         #   S_{ab}^{B}(t) = ∫_0^t ds sum_r K^{B}_{rr;ab}(t,s)
#         # independently of c_ab(s).  If the memory kernel is trace-preserving,
#         # the signed total over branches should be approximately zero for every
#         # input pair (a,b).
#         if !(return_kernel_rowsum_audit || return_kernel_output_audit) || !include_c_memory_G0 || qt <= 1
#             return nothing
#         end

#         if return_kernel_rowsum_audit
#             @inbounds for b_in in 1:n_sys, a_in in 1:n_sys, br in 1:4
#                 kernel_rowsum_audit[a_in, b_in, curr_itr, br] = const_ZERO_C
#             end
#         end
#         if return_kernel_output_audit
#             @inbounds for r_out in 1:n_sys, b_in in 1:n_sys, a_in in 1:n_sys, br in 1:4
#                 kernel_output_audit[r_out, a_in, b_in, curr_itr, br] = const_ZERO_C
#             end
#         end

#         n_s = n_outer_nodes(curr_itr, qt)
#         for s_node in 1:n_s
#             qs = outer_node_coord(s_node, curr_itr, qt)
#             qΔ = qt - qs + 1
#             w = ∫weight_to_t(s_node, curr_itr, qt)
#             w == 0.0 && continue

#             # Trace output means output pair is (r,r), summed over r.
#             for r in 1:n_sys
#                 α = r
#                 β = r

#                 # LL: input (μ, r).
#                 if include_c_memory_LL
#                     for χ in 1:n_sys
#                         χ == α && continue
#                         for μ in 1:n_sys
#                             μ == χ && continue
#                             if is_secular_pair(α, β, μ, β)
#                                 begin
#                                     val = w * c_memory_global_sign * c_memory_sign_LL * coef_K_LL(qs, qΔ, qt, α, β, χ, μ)
#                                     return_kernel_rowsum_audit && (kernel_rowsum_audit[μ, β, curr_itr, 1] += val)
#                                     return_kernel_output_audit && (kernel_output_audit[r, μ, β, curr_itr, 1] += val)
#                                 end
#                             end
#                         end
#                     end
#                 end

#                 # LR: input (χ, ν).
#                 if include_c_memory_LR
#                     for χ in 1:n_sys
#                         χ == α && continue
#                         for ν in 1:n_sys
#                             ν == β && continue
#                             if is_secular_pair(α, β, χ, ν)
#                                 begin
#                                     val = w * c_memory_global_sign * c_memory_sign_LR * coef_K_LR(qs, qΔ, qt, α, β, χ, ν)
#                                     return_kernel_rowsum_audit && (kernel_rowsum_audit[χ, ν, curr_itr, 2] += val)
#                                     return_kernel_output_audit && (kernel_output_audit[r, χ, ν, curr_itr, 2] += val)
#                                 end
#                             end
#                         end
#                     end
#                 end

#                 # RL: input (μ, ν).  Here ν is the right-side intermediate/input index.
#                 if include_c_memory_RL
#                     for μ in 1:n_sys
#                         μ == α && continue
#                         for ν in 1:n_sys
#                             ν == β && continue
#                             if is_secular_pair(α, β, μ, ν)
#                                 begin
#                                     val = w * c_memory_global_sign * c_memory_sign_RL * coef_K_RL(qs, qΔ, qt, α, β, μ, ν)
#                                     return_kernel_rowsum_audit && (kernel_rowsum_audit[μ, ν, curr_itr, 3] += val)
#                                     return_kernel_output_audit && (kernel_output_audit[r, μ, ν, curr_itr, 3] += val)
#                                 end
#                             end
#                         end
#                     end
#                 end

#                 # RR: input (r, ν).
#                 if include_c_memory_RR
#                     for χ in 1:n_sys
#                         χ == β && continue
#                         for ν in 1:n_sys
#                             ν == χ && continue
#                             if is_secular_pair(α, β, α, ν)
#                                 begin
#                                     val = w * c_memory_global_sign * c_memory_sign_RR * coef_K_RR(qs, qΔ, qt, α, β, χ, ν)
#                                     return_kernel_rowsum_audit && (kernel_rowsum_audit[α, ν, curr_itr, 4] += val)
#                                     return_kernel_output_audit && (kernel_output_audit[r, α, ν, curr_itr, 4] += val)
#                                 end
#                             end
#                         end
#                     end
#                 end
#             end
#         end
#         return nothing
#     end

#     Base.@noinline function readout_sigma_element(c_hist::Array{ComplexF64,3}, c_t::AbstractMatrix, curr_itr::Int, qt::Int, α::Int, β::Int)
#         val = c_local(c_t, α, β)

#         if readout_population_c_only && α == β
#             return val
#         end

#         if include_sigma_G0_readout && qt > 1
#             n_s = n_outer_nodes(curr_itr, qt)
#             corr = const_ZERO_C
#             for s_node in 1:n_s
#                 qs = outer_node_coord(s_node, curr_itr, qt)
#                 qΔ = qt - qs + 1
#                 w = ∫weight_to_t(s_node, curr_itr, qt)
#                 w == 0.0 && continue

#                 # Equation 1 readout sums are not secular-projected by default.
#                 for μ in 1:n_sys
#                     μ == α && continue
#                     corr += w * coef_Rσ_L(qs, qΔ, qt, α, β, μ) * c_mem(c_t, qs, qt, μ, β, c_hist)
#                 end
#                 for ν in 1:n_sys
#                     ν == β && continue
#                     corr += w * coef_Rσ_R(qs, qΔ, qt, α, β, ν) * c_mem(c_t, qs, qt, α, ν, c_hist)
#                 end
#             end
#             val += sigma_readout_global_sign * corr
#         end
#         return val
#     end

#     Base.@noinline function readout_sigma_full!(σ_target::AbstractMatrix, c_hist::Array{ComplexF64,3}, curr_itr::Int)
#         qt = q_full(curr_itr)
#         c_t = @view c_hist[:, :, curr_itr]
#         if use_threads && Threads.nthreads() > 1 && n_sys * n_sys >= max(16, 4 * Threads.nthreads())
#             Threads.@threads for linear_idx in 1:(n_sys * n_sys)
#                 α = ((linear_idx - 1) % n_sys) + 1
#                 β = ((linear_idx - 1) ÷ n_sys) + 1
#                 σ_target[α, β] = readout_sigma_element(c_hist, c_t, curr_itr, qt, α, β)
#             end
#         else
#             for β in 1:n_sys, α in 1:n_sys
#                 σ_target[α, β] = readout_sigma_element(c_hist, c_t, curr_itr, qt, α, β)
#             end
#         end
#         use_population_closure && enforce_population_closure!(σ_target)
#         enforce_sigma_hermitian && enforce_hermiticity!(σ_target)
#         trace_normalize_sigma && trace_normalize!(σ_target)

#         if audit_trace_hermiticity
#             trσ = const_ZERO_C
#             for i in 1:n_sys
#                 trσ += σ_target[i, i]
#             end
#             herm_res = 0.0
#             for j in 1:n_sys, i in 1:n_sys
#                 herm_res = max(herm_res, abs(σ_target[i, j] - conj(σ_target[j, i])))
#             end
#             if abs(imag(trσ)) > trace_imag_tol || herm_res > hermiticity_tol
#                 @printf(
#                     stderr,
#                     "V5 sign-audit warning at itr=%d: trace=%+.8e%+.8ei  Im(trace)=%.3e  herm_res=%.3e  memory_sign=%+.1f  pop_readout_c_only=%s\n",
#                     curr_itr,
#                     real(trσ),
#                     imag(trσ),
#                     abs(imag(trσ)),
#                     herm_res,
#                     Float64(c_memory_global_sign),
#                     string(readout_population_c_only),
#                 )
#             end
#         end

#         return σ_target
#     end

#     # ------------------------------------------------------------------
#     # Local internal state c and derivative c′.
#     # ------------------------------------------------------------------
#     c_hist = Array{ComplexF64}(undef, n_sys, n_sys, n_itr)
#     cprime_hist = Array{ComplexF64}(undef, n_sys, n_sys, n_itr)
#     branch_memory_audit = return_branch_memory_audit ? Array{ComplexF64}(undef, n_sys, n_sys, n_itr, 4) : Array{ComplexF64}(undef, 0, 0, 0, 0)
#     kernel_rowsum_audit = return_kernel_rowsum_audit ? Array{ComplexF64}(undef, n_sys, n_sys, n_itr, 4) : Array{ComplexF64}(undef, 0, 0, 0, 0)
#     kernel_output_audit = return_kernel_output_audit ? Array{ComplexF64}(undef, n_sys, n_sys, n_sys, n_itr, 4) : Array{ComplexF64}(undef, 0, 0, 0, 0, 0)
#     fill!(c_hist, const_ZERO_C)
#     fill!(cprime_hist, const_ZERO_C)
#     return_branch_memory_audit && fill!(branch_memory_audit, const_ZERO_C)
#     return_kernel_rowsum_audit && fill!(kernel_rowsum_audit, const_ZERO_C)
#     return_kernel_output_audit && fill!(kernel_output_audit, const_ZERO_C)

#     @inbounds for β in 1:n_sys, α in 1:n_sys
#         c_hist[α, β, 1] = σ[α, β, 1]
#     end
#     use_population_closure && enforce_population_closure!(@view(c_hist[:, :, 1]))
#     enforce_c_hermitian && enforce_hermiticity!(@view(c_hist[:, :, 1]))
#     readout_sigma_full!(@view(σ[:, :, 1]), c_hist, 1)

#     c_stage = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k1      = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k2      = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k3      = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k4      = Matrix{ComplexF64}(undef, n_sys, n_sys)

#     start_loop = max(1, start_itr)
#     @inbounds for curr_itr in start_loop:(n_itr - 1)
#         if verbose && (curr_itr == start_loop || curr_itr == n_itr - 1 || ((curr_itr - start_loop) % verbose_every_f == 0))
#             @printf(
#                 stderr,
#                 "Current iteration: %6d / %6d  method=%s  threads=%d  use_threads=%s  secular=%s  state=c  sigma=v5-readout  mem_signs=(G=%+.1f,LL=%+.1f,LR=%+.1f,RL=%+.1f,RR=%+.1f)\n",
#                 curr_itr, n_itr, String(method_sym), Threads.nthreads(), string(use_threads), string(use_secular),
#                 Float64(c_memory_global_sign), Float64(c_memory_sign_LL), Float64(c_memory_sign_LR), Float64(c_memory_sign_RL), Float64(c_memory_sign_RR),
#             )
#         end

#         c_t = @view c_hist[:, :, curr_itr]
#         c_next = @view c_hist[:, :, curr_itr + 1]

#         if method_sym == :euler
#             calc_c_rhs!(k1, curr_itr, q_full(curr_itr), c_t, c_hist)
#             cprime_hist[:, :, curr_itr] .= k1
#             compute_memory_kernel_rowsum_audit!(kernel_rowsum_audit, kernel_output_audit, curr_itr, q_full(curr_itr))
#             @. c_next = c_t + Δt * k1

#         elseif method_sym == :rk2
#             calc_c_rhs!(k1, curr_itr, q_full(curr_itr), c_t, c_hist)
#             cprime_hist[:, :, curr_itr] .= k1
#             compute_memory_kernel_rowsum_audit!(kernel_rowsum_audit, kernel_output_audit, curr_itr, q_full(curr_itr))
#             @. c_stage = c_t + 0.5 * Δt * k1
#             use_population_closure && enforce_population_closure!(c_stage)
#             enforce_c_hermitian && enforce_hermiticity!(c_stage)
#             calc_c_rhs!(k2, curr_itr, q_half_after(curr_itr), c_stage, c_hist)
#             @. c_next = c_t + Δt * k2

#         elseif method_sym == :rk4
#             calc_c_rhs!(k1, curr_itr, q_full(curr_itr), c_t, c_hist)
#             cprime_hist[:, :, curr_itr] .= k1
#             compute_memory_kernel_rowsum_audit!(kernel_rowsum_audit, kernel_output_audit, curr_itr, q_full(curr_itr))

#             @. c_stage = c_t + 0.5 * Δt * k1
#             use_population_closure && enforce_population_closure!(c_stage)
#             enforce_c_hermitian && enforce_hermiticity!(c_stage)
#             calc_c_rhs!(k2, curr_itr, q_half_after(curr_itr), c_stage, c_hist)

#             @. c_stage = c_t + 0.5 * Δt * k2
#             use_population_closure && enforce_population_closure!(c_stage)
#             enforce_c_hermitian && enforce_hermiticity!(c_stage)
#             calc_c_rhs!(k3, curr_itr, q_half_after(curr_itr), c_stage, c_hist)

#             @. c_stage = c_t + Δt * k3
#             use_population_closure && enforce_population_closure!(c_stage)
#             enforce_c_hermitian && enforce_hermiticity!(c_stage)
#             calc_c_rhs!(k4, curr_itr, q_full(curr_itr + 1), c_stage, c_hist)

#             @. c_next = c_t + (Δt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
#         end

#         use_population_closure && enforce_population_closure!(c_next)
#         enforce_c_hermitian && enforce_hermiticity!(c_next)

#         # Store physical sigma only after c has been advanced.
#         readout_sigma_full!(@view(σ[:, :, curr_itr + 1]), c_hist, curr_itr + 1)
#         context.curr_itr = UInt64(curr_itr + 1)
#     end

#     if return_internal_c || return_branch_memory_audit || return_kernel_rowsum_audit || return_kernel_output_audit
#         fields = (
#             σ = @view(σ[:, :, Int(context.curr_itr)]),
#             c = c_hist,
#             c′ = cprime_hist,
#             branch_memory = return_branch_memory_audit ? branch_memory_audit : nothing,
#             kernel_rowsum = return_kernel_rowsum_audit ? kernel_rowsum_audit : nothing,
#             kernel_output = return_kernel_output_audit ? kernel_output_audit : nothing,
#         )
#         return fields
#     end
#     return @view(σ[:, :, Int(context.curr_itr)])
# end


# # ver30: cyclic- and Hermiticity-consistent population source pairs
# #
# # Symbolically audited against TrB_12_primitives_branch_labeled_gclosure_ver3(1).ipynb.
# # F01-F07 and F11-F12 matched exactly; F08-F10 were replaced from the catalog expression trees.
# # F08 and F09 each retain two six-term g′ brackets generated by endpoint-aware before/after rules.
# # No global/per-branch/readout sign multiplier is exposed or applied.
# # Primitive connected signs are F07:+, F08:-, F09:-, F10:+.
# # Q=1-P assembly is LL/RR: projected-direct, LR/RL: direct-projected.
# # Population readout uses the exact identity σ_aa(t)=c_aa(t).
# # Every memory branch includes the outer normalization N_{αβ}(t).
# # Full-grid population RHS dp_a=dot(c_aa) and sum_a dp_a are stored explicitly.

# function calc__σ_σ′_secular_core!(
#     context::RmrtContext;
#     hbar::Real = 1.0,
#     use_population_closure::Bool = false,

#     include_c_L0P_local::Bool = true,
#     include_c_PL1P_local::Bool = true,
#     include_c_memory_G0::Bool = true,
#     include_sigma_G0_readout::Bool = true,


#     include_c_memory_LL::Bool = true,
#     include_c_memory_LR::Bool = true,
#     include_c_memory_RL::Bool = true,
#     include_c_memory_RR::Bool = true,

#     # Store the full-grid k1 memory contribution per branch.  This is useful
#     # for locating which branch breaks population direction or Hermiticity.
#     return_branch_memory_audit::Bool = false,

#     # Store the actual full-grid population RHS
#     #   population_rhs[a, itr] = dot(c_aa)(t_itr)
#     # and its trace sum.  This is the direct dp1, dp2, ... diagnostic.
#     return_population_rhs_audit::Bool = false,

#     # Kernel row-sum audit independent of the current c-history.
#     # For each full-grid time, stores the memory-kernel row sum
#     #   sum_r K_{rr;ab}^{branch}(t,s integrated)
#     # for each input pair (a,b) and branch.  This is the direct test for
#     # trace preservation of the c-memory block.
#     return_kernel_rowsum_audit::Bool = false,
#     # Output-resolved version of the same audit. Dimensions:
#     #   kernel_output[r_out, a_in, b_in, itr, branch].
#     # This is needed because branch row sums alone can hide whether the
#     # leak is a loss-side or gain-side error.
#     return_kernel_output_audit::Bool = true,
#     # Diagnose the trace-free population-difference mode.
#     return_population_difference_audit::Bool = true,
#     difference_damping_warn_tol::Float64 = 0.0,
#     population_rate_imag_tol::Float64 = 1.0e-8,


#     # Do not silently hide algebraic problems.  These audits print when
#     # physical sigma develops an imaginary trace or loses Hermiticity.
#     audit_trace_hermiticity::Bool = true,
#     audit_gclosure_structure::Bool = true,
#     audit_initial_population_direction::Bool = true,
#     initial_direction_steps::Int = 5,
#     trace_imag_tol::Float64 = 1.0e-8,
#     trace_real_tol::Float64 = 1.0e-8,
#     trace_rhs_tol::Float64 = 1.0e-8,
#     abort_on_c_trace_rhs::Bool = true,
#     hermiticity_tol::Float64 = 1.0e-8,

#     # Exact population-sector coefficient identities implied by cyclicity of
#     # the raw bath traces.  These are checked before using a memory kernel.
#     audit_population_cyclic_identity::Bool = true,
#     population_cyclic_identity_tol::Float64 = 1.0e-9,
#     abort_on_population_cyclic_identity::Bool = true,

#     use_secular::Bool = false,
#     secular_tol::Float64 = 1.0e-10,
#     method::Union{Symbol,String} = :rk4,
#     use_threads::Bool = true,

#     # Validation defaults: do not hide algebraic errors unless explicitly asked.
#     enforce_c_hermitian::Bool = false,
#     enforce_sigma_hermitian::Bool = false,
#     trace_normalize_sigma::Bool = false,

#     # c is local to this function.  Therefore resume is not supported unless c
#     # is later stored externally.
#     allow_resume_without_c_history::Bool = false,

#     verbose::Bool = true,
#     verbose_every::Int = 1,
#     return_internal_c::Bool = false,
# )
#     start_itr = Int(context.curr_itr)

#     n_sys = context.system.n_sys
#     n_itr = context.simulation_details.num_of_iteration
#     Δt    = context.simulation_details.Δt
#     ϵ     = context.ϵ_exci

#     σ   = context.σ              # physical readout output only
#     g   = context.g
#     gp  = context.g′
#     gpp = context.g″

#     if start_itr != 1 && !allow_resume_without_c_history
#         error("This ver30 cyclic-Hermitian local-c implementation keeps c-history inside the function. Set context.curr_itr=1, or implement persistent context.c before resuming.")
#     end

#     hbar_f = Float64(hbar)
#     hbar_f > 0.0 || error("hbar must be positive")

#     method_sym = Symbol(lowercase(String(method)))
#     method_sym in (:euler, :rk2, :rk4) || error("Unsupported method $(method). Use :euler, :rk2, or :rk4.")


#     verbose_every_f = max(1, Int(verbose_every))
#     const_ZERO_C = 0.0 + 0.0im
#     const_Q_MAX = 2 * n_itr - 1

#     # ------------------------------------------------------------------
#     # Doubled q-grid:
#     #   full grid i      -> q = 2i - 1
#     #   half grid i+1/2  -> q = 2i
#     #   relative q       -> qΔ = q_t - q_s + 1
#     # ------------------------------------------------------------------
#     @inline q_full(i::Int) = 2 * i - 1
#     @inline q_half_after(i::Int) = 2 * i
#     @inline full_index_from_q(q::Int) = (q + 1) >>> 1
#     @inline half_index_from_q(q::Int) = q >>> 1

#     @inline function check_q(q::Int)
#         (1 <= q <= const_Q_MAX) || error("q-grid index out of range: $(q), allowed [1,$(const_Q_MAX)]")
#         return q
#     end

#     function _first_existing_property(obj, names::Tuple)
#         for name in names
#             if hasproperty(obj, name)
#                 return getfield(obj, name)
#             end
#         end
#         return nothing
#     end

#     g_half = _first_existing_property(context, (
#         :g_half, :g_half_shifted, :g_shifted_half, :g_mid, :g_midpoint,
#         Symbol("g__half"), Symbol("g_half_grid"), Symbol("g_half_shifted_grid"),
#     ))
#     gp_half = _first_existing_property(context, (
#         Symbol("g′_half"), Symbol("g′_half_shifted"), Symbol("g′_shifted_half"),
#         Symbol("g′_mid"), Symbol("g′_midpoint"), Symbol("gp_half"),
#         Symbol("gp_half_shifted"), Symbol("g_prime_half"), Symbol("g_prime_half_shifted"),
#     ))
#     gpp_half = _first_existing_property(context, (
#         Symbol("g″_half"), Symbol("g″_half_shifted"), Symbol("g″_shifted_half"),
#         Symbol("g″_mid"), Symbol("g″_midpoint"), Symbol("gpp_half"),
#         Symbol("gpp_half_shifted"), Symbol("g_doubleprime_half"), Symbol("g_doubleprime_half_shifted"),
#     ))

#     needs_half_grid = method_sym in (:rk2, :rk4)
#     if needs_half_grid && (g_half === nothing || gp_half === nothing || gpp_half === nothing)
#         missing = String[]
#         g_half === nothing && push!(missing, "g half-grid")
#         gp_half === nothing && push!(missing, "g′ half-grid")
#         gpp_half === nothing && push!(missing, "g″ half-grid")
#         error("$(method_sym) requires precomputed half-grid containers for " * join(missing, ", "))
#     end

#     @inline function _missing_half_grid_error(name::String)
#         error("Attempted to access $(name) at a half-grid q index, but the half-grid container is missing.")
#     end

#     @inline function G(q::Int, a::Int, b::Int, cidx::Int, d::Int)
#         check_q(q)
#         if isodd(q)
#             @inbounds return g[full_index_from_q(q), a, b, cidx, d]
#         else
#             g_half === nothing && _missing_half_grid_error("g")
#             @inbounds return g_half[half_index_from_q(q), a, b, cidx, d]
#         end
#     end

#     @inline function Gp(q::Int, a::Int, b::Int, cidx::Int, d::Int)
#         check_q(q)
#         if isodd(q)
#             @inbounds return gp[full_index_from_q(q), a, b, cidx, d]
#         else
#             gp_half === nothing && _missing_half_grid_error("g′")
#             @inbounds return gp_half[half_index_from_q(q), a, b, cidx, d]
#         end
#     end

#     @inline function Gpp(q::Int, a::Int, b::Int, cidx::Int, d::Int)
#         check_q(q)
#         if isodd(q)
#             @inbounds return gpp[full_index_from_q(q), a, b, cidx, d]
#         else
#             gpp_half === nothing && _missing_half_grid_error("g″")
#             @inbounds return gpp_half[half_index_from_q(q), a, b, cidx, d]
#         end
#     end

#     # q-time and signed-time g-family helpers.
#     # sign=+1: f_{ab,cd}(+τ)
#     # sign=-1: use the v5 negative-time identities:
#     #   g_abcd(-τ)    = conj(g_dcba(+τ))
#     #   g′_abcd(-τ)   = -conj(g′_dcba(+τ))
#     #   g″_abcd(-τ)   = conj(g″_dcba(+τ))
#     @inline function qtime(q::Int)
#         return 0.5 * (Float64(q) - 1.0) * Δt
#     end

#     @inline function Gv(q::Int, a::Int, b::Int, cidx::Int, d::Int, sign::Int)
#         return sign == 1 ? G(q, a, b, cidx, d) : conj(G(q, d, cidx, b, a))
#     end

#     @inline function Gpv(q::Int, a::Int, b::Int, cidx::Int, d::Int, sign::Int)
#         return sign == 1 ? Gp(q, a, b, cidx, d) : -conj(Gp(q, d, cidx, b, a))
#     end

#     @inline function Gppv(q::Int, a::Int, b::Int, cidx::Int, d::Int, sign::Int)
#         return sign == 1 ? Gpp(q, a, b, cidx, d) : conj(Gpp(q, d, cidx, b, a))
#     end

#     @inline function is_secular_pair(out_a::Int, out_b::Int, in_a::Int, in_b::Int)
#         if !use_secular
#             return true
#         end
#         return abs(((ϵ[out_a] - ϵ[out_b]) - (ϵ[in_a] - ϵ[in_b])) / hbar_f) <= secular_tol
#     end

#     @inline function c_local(c_t::AbstractMatrix, a::Int, b::Int)
#         return (use_population_closure && a != b) ? const_ZERO_C : c_t[a, b]
#     end

#     @inline function c_mem(c_t::AbstractMatrix, qs::Int, qt::Int, a::Int, b::Int, c_hist::Array{ComplexF64,3})
#         if use_population_closure && a != b
#             return const_ZERO_C
#         end
#         if qs == qt
#             return c_t[a, b]
#         end
#         isodd(qs) || error("non-endpoint memory q index must be a stored full-grid point, got q=$(qs)")
#         return c_hist[a, b, full_index_from_q(qs)]
#     end

#     @inline function n_outer_nodes(curr_itr::Int, qt::Int)
#         q_now = q_full(curr_itr)
#         return qt > q_now ? curr_itr + 1 : curr_itr
#     end

#     @inline function outer_node_coord(node_idx::Int, curr_itr::Int, qt::Int)
#         return node_idx <= curr_itr ? q_full(node_idx) : qt
#     end

#     @inline function ∫weight_to_t(node_idx::Int, curr_itr::Int, qt::Int)
#         n_nodes = n_outer_nodes(curr_itr, qt)
#         n_nodes <= 1 && return 0.0
#         q = outer_node_coord(node_idx, curr_itr, qt)
#         if node_idx == 1
#             q_next = outer_node_coord(node_idx + 1, curr_itr, qt)
#             return 0.25 * Δt * Float64(q_next - q)
#         elseif node_idx == n_nodes
#             q_prev = outer_node_coord(node_idx - 1, curr_itr, qt)
#             return 0.25 * Δt * Float64(q - q_prev)
#         else
#             q_prev = outer_node_coord(node_idx - 1, curr_itr, qt)
#             q_next = outer_node_coord(node_idx + 1, curr_itr, qt)
#             return 0.25 * Δt * Float64(q_next - q_prev)
#         end
#     end

#     # ------------------------------------------------------------------
#     # ver24 direct notebook port.
#     # Catalog source: TrB_12_primitives_complete_gclosure_ver4.ipynb
#     # Coefficient source: L0_transported_normalized_complete_gclosed_dynamics_readout_ver4.ipynb
#     # Generated coefficient block SHA256: 42dc92912043cd811eb378a0756580e99bfda89053738dc964a4000e9fd9152d
#     # --------------------------------------------------------------
#     # ver24: direct notebook port.
#     # g, g′, g″ in context are dimensional correlation integrals; the
#     # notebook catalog uses dimensionless g-family values, hence /hbar².
#     # No contour or primitive is re-derived in Julia.
#     # --------------------------------------------------------------
#     const_inv_hbar2 = 1.0 / (hbar_f * hbar_f)
#     @inline _gd(q::Int,a::Int,b::Int,c::Int,d::Int) = const_inv_hbar2 * G(q,a,b,c,d)
#     @inline _gpd(q::Int,a::Int,b::Int,c::Int,d::Int) = const_inv_hbar2 * Gp(q,a,b,c,d)
#     @inline _gppd(q::Int,a::Int,b::Int,c::Int,d::Int) = const_inv_hbar2 * Gpp(q,a,b,c,d)

#     @inline function _N_g(qT::Int, a::Int, b::Int)
#         return exp(- _gd(qT, a, a, a, a) + _gd(qT, b, b, a, a) + conj(_gd(qT, a, a, b, b)) - conj(_gd(qT, b, b, b, b)))
#     end

#     @inline function _T_F01(qT::Int, a::Int, b::Int)
#         return -1 * 1.0im * hbar_f * (_gpd(qT, a, a, a, a) - conj(_gpd(qT, a, a, b, b)))
#     end

#     @inline function _T_F02(qT::Int, a::Int, b::Int)
#         return -1 * 1.0im * hbar_f * (_gpd(qT, b, b, a, a) - conj(_gpd(qT, b, b, b, b)))
#     end

#     @inline function _T_F03(qT::Int, a::Int, p::Int, b::Int)
#         return -1 * 1.0im * hbar_f * (_gpd(qT, a, p, p, p) - conj(_gpd(qT, p, a, a, a))) * exp(_gd(qT, a, a, p, p) - _gd(qT, b, b, p, p) - conj(_gd(qT, a, a, a, a)) + conj(_gd(qT, b, b, b, b)) + conj(_gd(qT, p, p, a, a)) - conj(_gd(qT, p, p, b, b)))
#     end

#     @inline function _T_F04(qT::Int, a::Int, q::Int, b::Int)
#         return -1 * 1.0im * hbar_f * (_gpd(qT, q, b, b, b) - conj(_gpd(qT, b, q, q, q))) * exp(_gd(qT, a, a, a, a) - _gd(qT, b, b, b, b) - _gd(qT, q, q, a, a) + _gd(qT, q, q, b, b) - conj(_gd(qT, a, a, q, q)) + conj(_gd(qT, b, b, q, q)))
#     end

#     @inline function _T_F05(qS::Int, qD::Int, qT::Int, a::Int, p::Int, b::Int)
#         return -1 * 1.0im * hbar_f * (_gpd(qT, a, p, p, p) - conj(_gpd(qT, p, a, a, a))) * exp(_gd(qT, a, a, p, p) - _gd(qS, b, b, p, p) + _gd(qS, p, p, p, p) - _gd(qT, p, p, p, p) - conj(_gd(qT, a, a, a, a)) + conj(_gd(qS, b, b, b, b)) + conj(_gd(qT, p, p, a, a)) - conj(_gd(qS, p, p, b, b)))
#     end

#     @inline function _T_F06(qS::Int, qD::Int, qT::Int, a::Int, q::Int, b::Int)
#         return -1 * 1.0im * hbar_f * (_gpd(qT, q, b, b, b) - conj(_gpd(qT, b, q, q, q))) * exp(_gd(qS, a, a, a, a) - _gd(qT, b, b, b, b) - _gd(qS, q, q, a, a) + _gd(qT, q, q, b, b) - conj(_gd(qS, a, a, q, q)) + conj(_gd(qT, b, b, q, q)) + conj(_gd(qS, q, q, q, q)) - conj(_gd(qT, q, q, q, q)))
#     end

#     @inline function _T_F07(qS::Int, qD::Int, qT::Int, a::Int, p::Int, r::Int, b::Int)
#         return (- (hbar_f)^(2) * (- _gpd(qD, a, a, p, r) + _gpd(qD, p, p, p, r) + _gpd(qS, p, r, r, r) - conj(_gpd(qS, r, p, a, a))) * (_gpd(qD, a, p, p, p) + _gpd(qT, a, p, r, r) - _gpd(qD, a, p, r, r) - conj(_gpd(qT, p, a, a, a))) + (hbar_f)^(2) * _gppd(qD, a, p, p, r)) * exp(_gd(qD, a, a, p, p) + _gd(qT, a, a, r, r) - _gd(qD, a, a, r, r) - _gd(qS, b, b, r, r) - _gd(qD, p, p, p, p) + _gd(qS, p, p, r, r) - _gd(qT, p, p, r, r) + _gd(qD, p, p, r, r) - conj(_gd(qT, a, a, a, a)) + conj(_gd(qS, b, b, b, b)) - conj(_gd(qS, p, p, a, a)) + conj(_gd(qT, p, p, a, a)) + conj(_gd(qS, r, r, a, a)) - conj(_gd(qS, r, r, b, b)))
#     end

#     @inline function _T_F08(qS::Int, qD::Int, qT::Int, a::Int, p::Int, q::Int, b::Int)
#         return (- (hbar_f)^(2) * (- _gpd(qS, q, b, a, a) + _gpd(qS, q, b, b, b) + _gpd(qS, q, b, p, p) - conj(_gpd(qD, a, a, b, q)) - conj(_gpd(qS, b, q, q, q)) + conj(_gpd(qD, p, p, b, q))) * (_gpd(qT, a, p, p, p) - conj(_gpd(qT, p, a, a, a)) + conj(_gpd(qT, p, a, b, b)) - conj(_gpd(qD, p, a, b, b)) - conj(_gpd(qT, p, a, q, q)) + conj(_gpd(qD, p, a, q, q))) + (hbar_f)^(2) * conj(_gppd(qD, p, a, b, q))) * exp(_gd(qT, a, a, p, p) + _gd(qS, b, b, a, a) - _gd(qS, b, b, b, b) - _gd(qS, b, b, p, p) + _gd(qS, p, p, p, p) - _gd(qT, p, p, p, p) - _gd(qS, q, q, a, a) + _gd(qS, q, q, b, b) - conj(_gd(qT, a, a, a, a)) + conj(_gd(qT, a, a, b, b)) - conj(_gd(qD, a, a, b, b)) - conj(_gd(qT, a, a, q, q)) + conj(_gd(qD, a, a, q, q)) + conj(_gd(qS, b, b, q, q)) + conj(_gd(qT, p, p, a, a)) - conj(_gd(qT, p, p, b, b)) + conj(_gd(qD, p, p, b, b)) - conj(_gd(qS, p, p, q, q)) + conj(_gd(qT, p, p, q, q)) - conj(_gd(qD, p, p, q, q)))
#     end

#     @inline function _T_F09(qS::Int, qD::Int, qT::Int, a::Int, p::Int, q::Int, b::Int)
#         return (- (hbar_f)^(2) * (- _gpd(qT, q, b, a, a) + _gpd(qD, q, b, a, a) + _gpd(qT, q, b, b, b) + _gpd(qT, q, b, p, p) - _gpd(qD, q, b, p, p) - conj(_gpd(qT, b, q, q, q))) * (_gpd(qS, a, p, p, p) + _gpd(qD, b, b, a, p) - _gpd(qD, q, q, a, p) - conj(_gpd(qS, p, a, a, a)) + conj(_gpd(qS, p, a, b, b)) - conj(_gpd(qS, p, a, q, q))) + (hbar_f)^(2) * _gppd(qD, q, b, a, p)) * exp(_gd(qS, a, a, p, p) + _gd(qT, b, b, a, a) - _gd(qD, b, b, a, a) - _gd(qT, b, b, b, b) - _gd(qT, b, b, p, p) + _gd(qD, b, b, p, p) - _gd(qT, q, q, a, a) + _gd(qD, q, q, a, a) + _gd(qT, q, q, b, b) - _gd(qS, q, q, p, p) + _gd(qT, q, q, p, p) - _gd(qD, q, q, p, p) - conj(_gd(qS, a, a, a, a)) + conj(_gd(qS, a, a, b, b)) - conj(_gd(qS, a, a, q, q)) + conj(_gd(qT, b, b, q, q)) + conj(_gd(qS, p, p, a, a)) - conj(_gd(qS, p, p, b, b)) + conj(_gd(qS, q, q, q, q)) - conj(_gd(qT, q, q, q, q)))
#     end

#     @inline function _T_F10(qS::Int, qD::Int, qT::Int, a::Int, u::Int, q::Int, b::Int)
#         return (- (hbar_f)^(2) * (_gpd(qT, q, b, b, b) - conj(_gpd(qD, b, q, q, q)) - conj(_gpd(qT, b, q, u, u)) + conj(_gpd(qD, b, q, u, u))) * (_gpd(qS, u, q, b, b) + conj(_gpd(qD, b, b, q, u)) - conj(_gpd(qD, q, q, q, u)) - conj(_gpd(qS, q, u, u, u))) + (hbar_f)^(2) * conj(_gppd(qD, b, q, q, u))) * exp(_gd(qS, a, a, a, a) - _gd(qT, b, b, b, b) - _gd(qS, q, q, b, b) + _gd(qT, q, q, b, b) - _gd(qS, u, u, a, a) + _gd(qS, u, u, b, b) - conj(_gd(qS, a, a, u, u)) + conj(_gd(qD, b, b, q, q)) + conj(_gd(qT, b, b, u, u)) - conj(_gd(qD, b, b, u, u)) - conj(_gd(qD, q, q, q, q)) + conj(_gd(qS, q, q, u, u)) - conj(_gd(qT, q, q, u, u)) + conj(_gd(qD, q, q, u, u)))
#     end

#     @inline function _T_F11(qS::Int, qD::Int, qT::Int, a::Int, p::Int, b::Int)
#         return -1 * 1.0im * hbar_f * (_gpd(qD, a, a, a, p) + _gpd(qS, a, p, p, p) - _gpd(qD, b, b, a, p) - conj(_gpd(qS, p, a, b, b))) * exp(- _gd(qD, a, a, a, a) + _gd(qS, a, a, p, p) - _gd(qT, a, a, p, p) + _gd(qD, a, a, p, p) + _gd(qD, b, b, a, a) - _gd(qS, b, b, p, p) + _gd(qT, b, b, p, p) - _gd(qD, b, b, p, p) - conj(_gd(qS, a, a, b, b)) + conj(_gd(qT, a, a, b, b)) + conj(_gd(qS, b, b, b, b)) - conj(_gd(qT, b, b, b, b)))
#     end

#     @inline function _T_F12(qS::Int, qD::Int, qT::Int, a::Int, q::Int, b::Int)
#         return -1 * 1.0im * hbar_f * (_gpd(qS, q, b, a, a) + conj(_gpd(qD, a, a, b, q)) - conj(_gpd(qD, b, b, b, q)) - conj(_gpd(qS, b, q, q, q))) * exp(_gd(qS, a, a, a, a) - _gd(qT, a, a, a, a) - _gd(qS, b, b, a, a) + _gd(qT, b, b, a, a) + conj(_gd(qD, a, a, b, b)) - conj(_gd(qS, a, a, q, q)) + conj(_gd(qT, a, a, q, q)) - conj(_gd(qD, a, a, q, q)) - conj(_gd(qD, b, b, b, b)) + conj(_gd(qS, b, b, q, q)) - conj(_gd(qT, b, b, q, q)) + conj(_gd(qD, b, b, q, q)))
#     end

#     @inline function _phase(qT::Int, qS::Int, left::Int, right::Int)
#         delta_t = qtime(qT) - qtime(qS)
#         return exp(-1.0im * ((ϵ[left] - ϵ[right]) / hbar_f) * delta_t)
#     end

#     @inline function coef_M_P_0(qt::Int, a::Int, b::Int)
#         return (
#             -1.0im * (ϵ[a] - ϵ[b]) / hbar_f
#             -1.0im / hbar_f * (_T_F01(qt,a,b) - _T_F02(qt,a,b))
#         )
#     end

#     @inline function coef_M_P_L(qt::Int, a::Int, b::Int, p::Int)
#         return -1.0im / hbar_f * _N_g(qt,a,b) * _T_F03(qt,a,p,b)
#     end

#     @inline function coef_M_P_R(qt::Int, a::Int, b::Int, q::Int)
#         return +1.0im / hbar_f * _N_g(qt,a,b) * _T_F04(qt,a,q,b)
#     end

#     @inline function coef_K_LL_generic(qs::Int, qD::Int, qt::Int,
#                                a::Int, b::Int, p::Int, r::Int)
#         return (
#             _N_g(qt,a,b) / hbar_f^2
#             * _phase(qt,qs,p,b)
#             * (
#                 _N_g(qs,p,b) * _T_F05(qs,qD,qt,a,p,b) * _T_F03(qs,p,r,b)
#                 - _T_F07(qs,qD,qt,a,p,r,b)
#             )
#         )
#     end

#     @inline function coef_K_LR_generic(qs::Int, qD::Int, qt::Int,
#                                a::Int, b::Int, p::Int, q::Int)
#         return (
#             -_N_g(qt,a,b) / hbar_f^2
#             * _phase(qt,qs,p,b)
#             * (
#                 _N_g(qs,p,b) * _T_F05(qs,qD,qt,a,p,b) * _T_F04(qs,p,q,b)
#                 - _T_F08(qs,qD,qt,a,p,q,b)
#             )
#         )
#     end

#     @inline function coef_K_RL_generic(qs::Int, qD::Int, qt::Int,
#                                a::Int, b::Int, p::Int, q::Int)
#         return (
#             -_N_g(qt,a,b) / hbar_f^2
#             * _phase(qt,qs,a,q)
#             * (
#                 _N_g(qs,a,q) * _T_F06(qs,qD,qt,a,q,b) * _T_F03(qs,a,p,q)
#                 - _T_F09(qs,qD,qt,a,p,q,b)
#             )
#         )
#     end

#     @inline function coef_K_RR_generic(qs::Int, qD::Int, qt::Int,
#                                a::Int, b::Int, q::Int, u::Int)
#         return (
#             _N_g(qt,a,b) / hbar_f^2
#             * _phase(qt,qs,a,q)
#             * (
#                 _N_g(qs,a,q) * _T_F06(qs,qD,qt,a,q,b) * _T_F04(qs,a,u,q)
#                 - _T_F10(qs,qD,qt,a,u,q,b)
#             )
#         )
#     end

#     @inline function coef_Rσ_L(qs::Int, qD::Int, qt::Int,
#                                a::Int, b::Int, p::Int)
#         return (
#             +1.0im / hbar_f
#             * _phase(qt,qs,a,b)
#             * (_N_g(qt,a,b) * _T_F03(qs,a,p,b) - _T_F11(qs,qD,qt,a,p,b))
#         )
#     end

#     @inline function coef_Rσ_R(qs::Int, qD::Int, qt::Int,
#                                a::Int, b::Int, q::Int)
#         return (
#             -1.0im / hbar_f
#             * _phase(qt,qs,a,b)
#             * (_N_g(qt,a,b) * _T_F04(qs,a,q,b) - _T_F12(qs,qD,qt,a,q,b))
#         )
#     end

#     # --------------------------------------------------------------
#     # ver30 population direct source-pair closure.
#     #
#     # Close one source-left direct trace.  Its source-right partner is the
#     # Hermitian conjugate, not an independently closed F10/F08 expression.
#     #
#     # The projected Q-subtraction pieces remain exactly those of the generic
#     # notebook port.  Only the direct population specializations are replaced.
#     # --------------------------------------------------------------
#     @inline function coef_population_A_L(
#         qs::Int,
#         qD::Int,
#         qt::Int,
#         from::Int,
#         to::Int,
#     )
#         from == to && return const_ZERO_C

#         return (
#             _N_g(qt,from,from) / hbar_f^2
#             * _phase(qt,qs,to,from)
#             * _T_F07(qs,qD,qt,from,to,from,from)
#         )
#     end

#     @inline function coef_population_A_R(
#         qs::Int,
#         qD::Int,
#         qt::Int,
#         from::Int,
#         to::Int,
#     )
#         return conj(coef_population_A_L(qs,qD,qt,from,to))
#     end

#     @inline function coef_K_LL(
#         qs::Int, qD::Int, qt::Int,
#         a::Int, b::Int, p::Int, r::Int,
#     )
#         value = coef_K_LL_generic(qs,qD,qt,a,b,p,r)

#         # Population loss: input (a,a), output (a,a), intermediate p.
#         if a == b && r == a && p != a
#             old_direct = (
#                 _N_g(qt,a,a) / hbar_f^2
#                 * _phase(qt,qs,p,a)
#                 * _T_F07(qs,qD,qt,a,p,a,a)
#             )
#             new_direct = coef_population_A_L(qs,qD,qt,a,p)
#             value += old_direct - new_direct
#         end

#         return value
#     end

#     @inline function coef_K_RL(
#         qs::Int, qD::Int, qt::Int,
#         a::Int, b::Int, p::Int, q::Int,
#     )
#         value = coef_K_RL_generic(qs,qD,qt,a,b,p,q)

#         # Population gain: input (p,p), output (a,a).
#         if a == b && p == q && p != a
#             old_direct = (
#                 _N_g(qt,a,a) / hbar_f^2
#                 * _phase(qt,qs,a,p)
#                 * _T_F09(qs,qD,qt,a,p,p,a)
#             )
#             new_direct = coef_population_A_L(qs,qD,qt,p,a)
#             value += new_direct - old_direct
#         end

#         return value
#     end

#     @inline function coef_K_RR(
#         qs::Int, qD::Int, qt::Int,
#         a::Int, b::Int, q::Int, u::Int,
#     )
#         value = coef_K_RR_generic(qs,qD,qt,a,b,q,u)

#         # Population loss: input (a,a), output (a,a), intermediate q.
#         if a == b && u == a && q != a
#             old_direct = (
#                 _N_g(qt,a,a) / hbar_f^2
#                 * _phase(qt,qs,a,q)
#                 * _T_F10(qs,qD,qt,a,a,q,a)
#             )
#             new_direct = coef_population_A_R(qs,qD,qt,a,q)
#             value += old_direct - new_direct
#         end

#         return value
#     end

#     @inline function coef_K_LR(
#         qs::Int, qD::Int, qt::Int,
#         a::Int, b::Int, p::Int, q::Int,
#     )
#         value = coef_K_LR_generic(qs,qD,qt,a,b,p,q)

#         # Population gain: input (p,p), output (a,a).
#         if a == b && p == q && p != a
#             old_direct = (
#                 _N_g(qt,a,a) / hbar_f^2
#                 * _phase(qt,qs,p,a)
#                 * _T_F08(qs,qD,qt,a,p,p,a)
#             )
#             new_direct = coef_population_A_R(qs,qD,qt,p,a)
#             value += new_direct - old_direct
#         end

#         return value
#     end

#     @inline function enforce_hermiticity!(mat::AbstractMatrix)
#         for i in 1:n_sys
#             mat[i, i] = real(mat[i, i]) + 0.0im
#         end
#         for i in 1:(n_sys - 1), j in (i + 1):n_sys
#             z = 0.5 * (mat[i, j] + conj(mat[j, i]))
#             mat[i, j] = z
#             mat[j, i] = conj(z)
#         end
#         return mat
#     end

#     @inline function enforce_population_closure!(mat::AbstractMatrix)
#         for j in 1:n_sys, i in 1:n_sys
#             mat[i, j] = (i == j) ? (real(mat[i, i]) + 0.0im) : const_ZERO_C
#         end
#         return mat
#     end

#     @inline function trace_normalize!(mat::AbstractMatrix)
#         tr = zero(ComplexF64)
#         for i in 1:n_sys
#             tr += mat[i, i]
#         end
#         if abs(tr) > 0
#             mat ./= tr
#         end
#         return mat
#     end

#     # --------------------------------------------------------------
#     # Exact two-level population cyclicity audit.
#     #
#     # For x != y and every (t,s), the flat raw traces satisfy
#     #
#     #   F07(x,y,x,x) = F09(y,x,x,y),
#     #   F10(x,x,y,x) = F08(y,x,x,y),
#     #
#     # by cyclicity of Tr_B.  Including the projected Q=1-P pieces and phases,
#     # this gives the coefficient-level row-sum identities
#     #
#     #   K_LL[x<-x] + K_RL[y<-x] = 0,
#     #   K_RR[x<-x] + K_LR[y<-x] = 0.
#     #
#     # A violation is an implementation error, not a physical approximation.
#     # --------------------------------------------------------------
#     Base.@noinline function audit_population_cyclic_identity_at!(
#         curr_itr::Int,
#         qt::Int,
#     )
#         if !audit_population_cyclic_identity || n_sys != 2 || qt <= 1
#             return nothing
#         end

#         max_residual = 0.0
#         worst = nothing
#         n_s = n_outer_nodes(curr_itr, qt)

#         for x in 1:2
#             y = 3 - x

#             for s_node in 1:n_s
#                 qs = outer_node_coord(s_node, curr_itr, qt)
#                 qD = qt - qs + 1

#                 left_pair = (
#                     coef_K_LL(qs, qD, qt, x, x, y, x)
#                     + coef_K_RL(qs, qD, qt, y, y, x, x)
#                 )

#                 right_pair = (
#                     coef_K_RR(qs, qD, qt, x, x, y, x)
#                     + coef_K_LR(qs, qD, qt, y, y, x, x)
#                 )

#                 left_residual = abs(left_pair)
#                 right_residual = abs(right_pair)

#                 if left_residual > max_residual
#                     max_residual = left_residual
#                     worst = (:left_pair, x, y, qs, qD, left_pair)
#                 end

#                 if right_residual > max_residual
#                     max_residual = right_residual
#                     worst = (:right_pair, x, y, qs, qD, right_pair)
#                 end
#             end
#         end

#         # The total direct transition rate must be real:
#         # A_L + A_R = 2 Re(A_L).
#         max_hermitian_residual = 0.0
#         worst_hermitian = nothing

#         for from in 1:2
#             to = 3 - from
#             for s_node in 1:n_s
#                 qs = outer_node_coord(s_node, curr_itr, qt)
#                 qD = qt - qs + 1
#                 AL = coef_population_A_L(qs,qD,qt,from,to)
#                 AR = coef_population_A_R(qs,qD,qt,from,to)

#                 pair_residual = abs(AR - conj(AL))
#                 rate_imaginary = abs(imag(AL + AR))
#                 residual = max(pair_residual, rate_imaginary)

#                 if residual > max_hermitian_residual
#                     max_hermitian_residual = residual
#                     worst_hermitian = (
#                         from,
#                         to,
#                         qs,
#                         qD,
#                         AL,
#                         AR,
#                         AL + AR,
#                     )
#                 end
#             end
#         end

#         if max_hermitian_residual > population_cyclic_identity_tol
#             error(
#                 "Population Hermitian source-pair identity failed: " *
#                 "itr=$(curr_itr), q=$(qt), " *
#                 "max_residual=$(max_hermitian_residual), " *
#                 "worst=$(repr(worst_hermitian))"
#             )
#         end

#         if max_residual > population_cyclic_identity_tol
#             @printf(
#                 stderr,
#                 "ver30 population cyclic-identity failure: itr=%d q=%d max_residual=%.6e worst=%s\n",
#                 curr_itr,
#                 qt,
#                 max_residual,
#                 repr(worst),
#             )

#             if abort_on_population_cyclic_identity
#                 error(
#                     "Population cyclic trace identity failed before propagation: " *
#                     "itr=$(curr_itr), q=$(qt), max_residual=$(max_residual)"
#                 )
#             end
#         end

#         return nothing
#     end

#     Base.@noinline function calc_c_rhs_element!(rhs_mat::AbstractMatrix, curr_itr::Int, qt::Int, c_t::AbstractMatrix, c_hist::Array{ComplexF64,3}, α::Int, β::Int)
#         if use_population_closure && α != β
#             rhs_mat[α, β] = const_ZERO_C
#             return nothing
#         end

#         rhs = const_ZERO_C

#         if include_c_L0P_local
#             rhs += coef_M_P_0(qt, α, β) * c_local(c_t, α, β)
#         end

#         if include_c_PL1P_local
#             for μ in 1:n_sys
#                 μ == α && continue
#                 if is_secular_pair(α, β, μ, β)
#                     rhs += coef_M_P_L(qt, α, β, μ) * c_local(c_t, μ, β)
#                 end
#             end
#             for ν in 1:n_sys
#                 ν == β && continue
#                 if is_secular_pair(α, β, α, ν)
#                     rhs += coef_M_P_R(qt, α, β, ν) * c_local(c_t, α, ν)
#                 end
#             end
#         end

#         if include_c_memory_G0 && qt > 1
#             integral_LL = const_ZERO_C
#             integral_LR = const_ZERO_C
#             integral_RL = const_ZERO_C
#             integral_RR = const_ZERO_C
#             n_s = n_outer_nodes(curr_itr, qt)
#             for s_node in 1:n_s
#                 qs = outer_node_coord(s_node, curr_itr, qt)
#                 qΔ = qt - qs + 1
#                 w = ∫weight_to_t(s_node, curr_itr, qt)
#                 w == 0.0 && continue

#                 branch_LL = const_ZERO_C
#                 branch_LR = const_ZERO_C
#                 branch_RL = const_ZERO_C
#                 branch_RR = const_ZERO_C

#                 # LL: input (μ, β), first intermediate χ, second insertion χ -> μ.
#                 if include_c_memory_LL
#                     for χ in 1:n_sys
#                         χ == α && continue
#                         for μ in 1:n_sys
#                             μ == χ && continue
#                             if is_secular_pair(α, β, μ, β)
#                                 branch_LL += coef_K_LL(qs, qΔ, qt, α, β, χ, μ) * c_mem(c_t, qs, qt, μ, β, c_hist)
#                             end
#                         end
#                     end
#                 end

#                 # LR: input (χ, ν).
#                 if include_c_memory_LR
#                     for χ in 1:n_sys
#                         χ == α && continue
#                         for ν in 1:n_sys
#                             ν == β && continue
#                             if is_secular_pair(α, β, χ, ν)
#                                 branch_LR += coef_K_LR(qs, qΔ, qt, α, β, χ, ν) * c_mem(c_t, qs, qt, χ, ν, c_hist)
#                             end
#                         end
#                     end
#                 end

#                 # RL: input (μ, ν).
#                 if include_c_memory_RL
#                     for μ in 1:n_sys
#                         μ == α && continue
#                         for ν in 1:n_sys
#                             ν == β && continue
#                             if is_secular_pair(α, β, μ, ν)
#                                 branch_RL += coef_K_RL(qs, qΔ, qt, α, β, μ, ν) * c_mem(c_t, qs, qt, μ, ν, c_hist)
#                             end
#                         end
#                     end
#                 end

#                 # RR: input (α, ν), first intermediate χ, second insertion ν -> χ.
#                 if include_c_memory_RR
#                     for χ in 1:n_sys
#                         χ == β && continue
#                         for ν in 1:n_sys
#                             ν == χ && continue
#                             if is_secular_pair(α, β, α, ν)
#                                 branch_RR += coef_K_RR(qs, qΔ, qt, α, β, χ, ν) * c_mem(c_t, qs, qt, α, ν, c_hist)
#                             end
#                         end
#                     end
#                 end

#                 integral_LL += w * branch_LL
#                 integral_LR += w * branch_LR
#                 integral_RL += w * branch_RL
#                 integral_RR += w * branch_RR
#             end

#             # Exact result-based assembly.  The coefficient functions already
#             # contain every Liouville-branch sign and Q=1-P subtraction.
#             rhs += integral_LL + integral_LR + integral_RL + integral_RR

#             if need_branch_memory_storage && qt == q_full(curr_itr)
#                 branch_memory_audit[α, β, curr_itr, 1] = integral_LL
#                 branch_memory_audit[α, β, curr_itr, 2] = integral_LR
#                 branch_memory_audit[α, β, curr_itr, 3] = integral_RL
#                 branch_memory_audit[α, β, curr_itr, 4] = integral_RR
#             end
#         end

#         rhs_mat[α, β] = rhs
#         return nothing
#     end

#     Base.@noinline function calc_c_rhs!(rhs_mat::AbstractMatrix, curr_itr::Int, qt::Int, c_t::AbstractMatrix, c_hist::Array{ComplexF64,3})
#         audit_population_cyclic_identity_at!(curr_itr, qt)
#         fill!(rhs_mat, const_ZERO_C)
#         n_components = n_sys * n_sys
#         heavy = include_c_memory_G0
#         if use_threads && Threads.nthreads() > 1 && (heavy || n_components >= max(16, 4 * Threads.nthreads()))
#             Threads.@threads for linear_idx in 1:n_components
#                 α = ((linear_idx - 1) % n_sys) + 1
#                 β = ((linear_idx - 1) ÷ n_sys) + 1
#                 calc_c_rhs_element!(rhs_mat, curr_itr, qt, c_t, c_hist, α, β)
#             end
#         else
#             for β in 1:n_sys, α in 1:n_sys
#                 calc_c_rhs_element!(rhs_mat, curr_itr, qt, c_t, c_hist, α, β)
#             end
#         end

#         # For the normalized projector, diagonal c elements are physical
#         # populations.  Therefore sum_a dot(c_aa) must vanish.
#         if audit_trace_hermiticity
#             trace_rhs = const_ZERO_C
#             @inbounds for aidx in 1:n_sys
#                 trace_rhs += rhs_mat[aidx, aidx]
#             end
#             if abs(trace_rhs) > trace_rhs_tol
#                 stage_label = isodd(qt) ? "full" : "half"
#                 @printf(
#                     stderr,
#                     "ver30 c-trace RHS warning at itr=%d q=%d stage=%s: sum_a dot(c_aa)=%+.8e%+.8ei  abs=%.3e",
#                     curr_itr,
#                     qt,
#                     stage_label,
#                     real(trace_rhs),
#                     imag(trace_rhs),
#                     abs(trace_rhs),
#                 )
#                 @inbounds for aidx in 1:n_sys
#                     dpa = rhs_mat[aidx, aidx]
#                     @printf(
#                         stderr,
#                         "  dp%d=%+.8e%+.8ei",
#                         aidx,
#                         real(dpa),
#                         imag(dpa),
#                     )
#                 end
#                 @printf(stderr, "\n")
#                 if abort_on_c_trace_rhs
#                     error(
#                         "ver30 trace-RHS identity failed before propagation could be trusted: " *
#                         "itr=$(curr_itr), q=$(qt), sum_a dot(c_aa)=$(trace_rhs)"
#                     )
#                 end
#             end
#         end
#         return rhs_mat
#     end


#     Base.@noinline function compute_memory_kernel_rowsum_audit!(kernel_rowsum_audit::Array{ComplexF64,4}, kernel_output_audit::Array{ComplexF64,5}, curr_itr::Int, qt::Int)
#         # Stores, for this full-grid time t, the integrated row sum
#         #   S_{ab}^{B}(t) = ∫_0^t ds sum_r K^{B}_{rr;ab}(t,s)
#         # independently of c_ab(s).  If the memory kernel is trace-preserving,
#         # the signed total over branches should be approximately zero for every
#         # input pair (a,b).
#         if !(return_kernel_rowsum_audit || return_kernel_output_audit) || !include_c_memory_G0 || qt <= 1
#             return nothing
#         end

#         if return_kernel_rowsum_audit
#             @inbounds for b_in in 1:n_sys, a_in in 1:n_sys, br in 1:4
#                 kernel_rowsum_audit[a_in, b_in, curr_itr, br] = const_ZERO_C
#             end
#         end
#         if return_kernel_output_audit
#             @inbounds for r_out in 1:n_sys, b_in in 1:n_sys, a_in in 1:n_sys, br in 1:4
#                 kernel_output_audit[r_out, a_in, b_in, curr_itr, br] = const_ZERO_C
#             end
#         end

#         n_s = n_outer_nodes(curr_itr, qt)
#         for s_node in 1:n_s
#             qs = outer_node_coord(s_node, curr_itr, qt)
#             qΔ = qt - qs + 1
#             w = ∫weight_to_t(s_node, curr_itr, qt)
#             w == 0.0 && continue

#             # Trace output means output pair is (r,r), summed over r.
#             for r in 1:n_sys
#                 α = r
#                 β = r

#                 # LL: input (μ, r).
#                 if include_c_memory_LL
#                     for χ in 1:n_sys
#                         χ == α && continue
#                         for μ in 1:n_sys
#                             μ == χ && continue
#                             if is_secular_pair(α, β, μ, β)
#                                 begin
#                                     val = w * coef_K_LL(qs, qΔ, qt, α, β, χ, μ)
#                                     return_kernel_rowsum_audit && (kernel_rowsum_audit[μ, β, curr_itr, 1] += val)
#                                     return_kernel_output_audit && (kernel_output_audit[r, μ, β, curr_itr, 1] += val)
#                                 end
#                             end
#                         end
#                     end
#                 end

#                 # LR: input (χ, ν).
#                 if include_c_memory_LR
#                     for χ in 1:n_sys
#                         χ == α && continue
#                         for ν in 1:n_sys
#                             ν == β && continue
#                             if is_secular_pair(α, β, χ, ν)
#                                 begin
#                                     val = w * coef_K_LR(qs, qΔ, qt, α, β, χ, ν)
#                                     return_kernel_rowsum_audit && (kernel_rowsum_audit[χ, ν, curr_itr, 2] += val)
#                                     return_kernel_output_audit && (kernel_output_audit[r, χ, ν, curr_itr, 2] += val)
#                                 end
#                             end
#                         end
#                     end
#                 end

#                 # RL: input (μ, ν).  Here ν is the right-side intermediate/input index.
#                 if include_c_memory_RL
#                     for μ in 1:n_sys
#                         μ == α && continue
#                         for ν in 1:n_sys
#                             ν == β && continue
#                             if is_secular_pair(α, β, μ, ν)
#                                 begin
#                                     val = w * coef_K_RL(qs, qΔ, qt, α, β, μ, ν)
#                                     return_kernel_rowsum_audit && (kernel_rowsum_audit[μ, ν, curr_itr, 3] += val)
#                                     return_kernel_output_audit && (kernel_output_audit[r, μ, ν, curr_itr, 3] += val)
#                                 end
#                             end
#                         end
#                     end
#                 end

#                 # RR: input (r, ν).
#                 if include_c_memory_RR
#                     for χ in 1:n_sys
#                         χ == β && continue
#                         for ν in 1:n_sys
#                             ν == χ && continue
#                             if is_secular_pair(α, β, α, ν)
#                                 begin
#                                     val = w * coef_K_RR(qs, qΔ, qt, α, β, χ, ν)
#                                     return_kernel_rowsum_audit && (kernel_rowsum_audit[α, ν, curr_itr, 4] += val)
#                                     return_kernel_output_audit && (kernel_output_audit[r, α, ν, curr_itr, 4] += val)
#                                 end
#                             end
#                         end
#                     end
#                 end
#             end
#         end
#         return nothing
#     end

#     Base.@noinline function readout_sigma_element(c_hist::Array{ComplexF64,3}, c_t::AbstractMatrix, curr_itr::Int, qt::Int, α::Int, β::Int)
#         val = c_local(c_t, α, β)

#         # Exact identity of the normalized transported projector:
#         #   σ_aa(t) = Tr_B ρ_aa(t) = c_aa(t).
#         # The approximate Q-space readout must never be applied to populations.
#         if α == β
#             return val
#         end

#         if include_sigma_G0_readout && qt > 1
#             n_s = n_outer_nodes(curr_itr, qt)
#             corr = const_ZERO_C
#             for s_node in 1:n_s
#                 qs = outer_node_coord(s_node, curr_itr, qt)
#                 qΔ = qt - qs + 1
#                 w = ∫weight_to_t(s_node, curr_itr, qt)
#                 w == 0.0 && continue

#                 # Equation 1 readout sums are not secular-projected by default.
#                 for μ in 1:n_sys
#                     μ == α && continue
#                     corr += w * coef_Rσ_L(qs, qΔ, qt, α, β, μ) * c_mem(c_t, qs, qt, μ, β, c_hist)
#                 end
#                 for ν in 1:n_sys
#                     ν == β && continue
#                     corr += w * coef_Rσ_R(qs, qΔ, qt, α, β, ν) * c_mem(c_t, qs, qt, α, ν, c_hist)
#                 end
#             end
#             val += corr
#         end
#         return val
#     end

#     Base.@noinline function readout_sigma_full!(σ_target::AbstractMatrix, c_hist::Array{ComplexF64,3}, curr_itr::Int)
#         qt = q_full(curr_itr)
#         c_t = @view c_hist[:, :, curr_itr]
#         if use_threads && Threads.nthreads() > 1 && n_sys * n_sys >= max(16, 4 * Threads.nthreads())
#             Threads.@threads for linear_idx in 1:(n_sys * n_sys)
#                 α = ((linear_idx - 1) % n_sys) + 1
#                 β = ((linear_idx - 1) ÷ n_sys) + 1
#                 σ_target[α, β] = readout_sigma_element(c_hist, c_t, curr_itr, qt, α, β)
#             end
#         else
#             for β in 1:n_sys, α in 1:n_sys
#                 σ_target[α, β] = readout_sigma_element(c_hist, c_t, curr_itr, qt, α, β)
#             end
#         end
#         use_population_closure && enforce_population_closure!(σ_target)
#         enforce_sigma_hermitian && enforce_hermiticity!(σ_target)
#         trace_normalize_sigma && trace_normalize!(σ_target)

#         if audit_trace_hermiticity
#             trσ = const_ZERO_C
#             for i in 1:n_sys
#                 trσ += σ_target[i, i]
#             end
#             sigma_herm_res = 0.0
#             c_herm_res = 0.0
#             for j in 1:n_sys, i in 1:n_sys
#                 sigma_herm_res = max(
#                     sigma_herm_res,
#                     abs(σ_target[i, j] - conj(σ_target[j, i])),
#                 )
#                 c_herm_res = max(
#                     c_herm_res,
#                     abs(c_t[i, j] - conj(c_t[j, i])),
#                 )
#             end
#             trace_real_res = abs(real(trσ) - 1.0)
#             if (
#                 trace_real_res > trace_real_tol
#                 || abs(imag(trσ)) > trace_imag_tol
#                 || sigma_herm_res > hermiticity_tol
#                 || c_herm_res > hermiticity_tol
#             )
#                 @printf(
#                     stderr,
#                     "ver30 exact-result audit warning at itr=%d: trace=%+.8e%+.8ei  Re(trace)-1=%.3e  Im(trace)=%.3e  c_herm_res=%.3e  sigma_herm_res=%.3e\n",
#                     curr_itr,
#                     real(trσ),
#                     imag(trσ),
#                     trace_real_res,
#                     abs(imag(trσ)),
#                     c_herm_res,
#                     sigma_herm_res,
#                 )
#             end
#         end

#         return σ_target
#     end

#     # ------------------------------------------------------------------
#     # Local internal state c and derivative c′.
#     # ------------------------------------------------------------------
#     c_hist = Array{ComplexF64}(undef, n_sys, n_sys, n_itr)
#     cprime_hist = Array{ComplexF64}(undef, n_sys, n_sys, n_itr)
#     population_rhs_hist = Array{ComplexF64}(undef, n_sys, n_itr)
#     population_trace_rhs_hist = Array{ComplexF64}(undef, n_itr)
#     population_rhs_components_hist = Array{ComplexF64}(undef, n_sys, n_itr, 6)
#     population_trace_rhs_components_hist = Array{ComplexF64}(undef, n_itr, 6)
#     population_rhs_component_residual_hist = Array{ComplexF64}(undef, n_sys, n_itr)
#     population_rhs_component_labels = (:local_L0, :local_L1, :LL, :LR, :RL, :RR)
#     need_branch_memory_storage = return_branch_memory_audit || return_population_rhs_audit || return_internal_c
#     branch_memory_audit = need_branch_memory_storage ? Array{ComplexF64}(undef, n_sys, n_sys, n_itr, 4) : Array{ComplexF64}(undef, 0, 0, 0, 0)
#     kernel_rowsum_audit = return_kernel_rowsum_audit ? Array{ComplexF64}(undef, n_sys, n_sys, n_itr, 4) : Array{ComplexF64}(undef, 0, 0, 0, 0)
#     kernel_output_audit = return_kernel_output_audit ? Array{ComplexF64}(undef, n_sys, n_sys, n_sys, n_itr, 4) : Array{ComplexF64}(undef, 0, 0, 0, 0, 0)
#     population_difference_hist = Array{ComplexF64}(undef, n_itr)
#     population_difference_rhs_hist = Array{ComplexF64}(undef, n_itr)
#     population_difference_rhs_components_hist = Array{ComplexF64}(undef, n_itr, 6)
#     population_rate_1_to_2_hist = Array{ComplexF64}(undef, n_itr)
#     population_rate_2_to_1_hist = Array{ComplexF64}(undef, n_itr)
#     population_difference_damping_hist = Array{ComplexF64}(undef, n_itr)
#     fill!(c_hist, const_ZERO_C)
#     fill!(cprime_hist, const_ZERO_C)
#     fill!(population_rhs_hist, const_ZERO_C)
#     fill!(population_trace_rhs_hist, const_ZERO_C)
#     fill!(population_rhs_components_hist, const_ZERO_C)
#     fill!(population_trace_rhs_components_hist, const_ZERO_C)
#     fill!(population_rhs_component_residual_hist, const_ZERO_C)
#     need_branch_memory_storage && fill!(branch_memory_audit, const_ZERO_C)
#     return_kernel_rowsum_audit && fill!(kernel_rowsum_audit, const_ZERO_C)
#     return_kernel_output_audit && fill!(kernel_output_audit, const_ZERO_C)
#     fill!(population_difference_hist, const_ZERO_C)
#     fill!(population_difference_rhs_hist, const_ZERO_C)
#     fill!(population_difference_rhs_components_hist, const_ZERO_C)
#     fill!(population_rate_1_to_2_hist, const_ZERO_C)
#     fill!(population_rate_2_to_1_hist, const_ZERO_C)
#     fill!(population_difference_damping_hist, const_ZERO_C)

#     @inbounds for β in 1:n_sys, α in 1:n_sys
#         c_hist[α, β, 1] = σ[α, β, 1]
#     end
#     use_population_closure && enforce_population_closure!(@view(c_hist[:, :, 1]))
#     enforce_c_hermitian && enforce_hermiticity!(@view(c_hist[:, :, 1]))
#     readout_sigma_full!(@view(σ[:, :, 1]), c_hist, 1)

#     c_stage = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k1      = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k2      = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k3      = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k4      = Matrix{ComplexF64}(undef, n_sys, n_sys)

#     @inline function store_population_difference_audit!(itr::Int, c_t::AbstractMatrix)
#         if n_sys != 2
#             return nothing
#         end

#         population_difference_hist[itr] = c_t[1,1] - c_t[2,2]
#         population_difference_rhs_hist[itr] = population_rhs_hist[1,itr] - population_rhs_hist[2,itr]

#         for component in 1:6
#             population_difference_rhs_components_hist[itr,component] =
#                 population_rhs_components_hist[1,itr,component] -
#                 population_rhs_components_hist[2,itr,component]
#         end

#         if return_kernel_output_audit
#             k12 = const_ZERO_C  # input population 1 -> output population 2
#             k21 = const_ZERO_C  # input population 2 -> output population 1
#             for branch in 1:4
#                 k12 += kernel_output_audit[2,1,1,itr,branch]
#                 k21 += kernel_output_audit[1,2,2,itr,branch]
#             end
#             population_rate_1_to_2_hist[itr] = k12
#             population_rate_2_to_1_hist[itr] = k21
#             population_difference_damping_hist[itr] = k12 + k21

#             if real(k12 + k21) < difference_damping_warn_tol
#                 @printf(
#                     stderr,
#                     "ver30 anti-damping audit at itr=%d: k12=%+.8e%+.8ei  k21=%+.8e%+.8ei  Re(k12+k21)=%.8e\n",
#                     itr,
#                     real(k12), imag(k12),
#                     real(k21), imag(k21),
#                     real(k12+k21),
#                 )
#             end

#             max_rate_imag = max(abs(imag(k12)), abs(imag(k21)))
#             if max_rate_imag > population_rate_imag_tol
#                 @printf(
#                     stderr,
#                     "ver30 population-rate Hermiticity audit at itr=%d: Im(k12)=%.8e  Im(k21)=%.8e  max=%.8e\n",
#                     itr,
#                     imag(k12),
#                     imag(k21),
#                     max_rate_imag,
#                 )
#             end
#         end
#         return nothing
#     end

#     @inline function store_population_rhs!(itr::Int, rhs_full::AbstractMatrix, c_t::AbstractMatrix)
#         sum_dp = const_ZERO_C
#         trace_components = ntuple(_ -> const_ZERO_C, 6)
#         trace_components_mut = collect(trace_components)

#         @inbounds for state in 1:n_sys
#             dp = rhs_full[state, state]
#             population_rhs_hist[state, itr] = dp
#             sum_dp += dp

#             local_L0 = include_c_L0P_local ? coef_M_P_0(q_full(itr), state, state) * c_local(c_t, state, state) : const_ZERO_C
#             local_L1 = const_ZERO_C
#             if include_c_PL1P_local
#                 for μ in 1:n_sys
#                     μ == state && continue
#                     if is_secular_pair(state, state, μ, state)
#                         local_L1 += coef_M_P_L(q_full(itr), state, state, μ) * c_local(c_t, μ, state)
#                     end
#                 end
#                 for ν in 1:n_sys
#                     ν == state && continue
#                     if is_secular_pair(state, state, state, ν)
#                         local_L1 += coef_M_P_R(q_full(itr), state, state, ν) * c_local(c_t, state, ν)
#                     end
#                 end
#             end

#             LL = need_branch_memory_storage ? branch_memory_audit[state, state, itr, 1] : const_ZERO_C
#             LR = need_branch_memory_storage ? branch_memory_audit[state, state, itr, 2] : const_ZERO_C
#             RL = need_branch_memory_storage ? branch_memory_audit[state, state, itr, 3] : const_ZERO_C
#             RR = need_branch_memory_storage ? branch_memory_audit[state, state, itr, 4] : const_ZERO_C
#             components = (local_L0, local_L1, LL, LR, RL, RR)

#             component_total = const_ZERO_C
#             for component in 1:6
#                 value = components[component]
#                 population_rhs_components_hist[state, itr, component] = value
#                 trace_components_mut[component] += value
#                 component_total += value
#             end
#             population_rhs_component_residual_hist[state, itr] = dp - component_total
#         end

#         population_trace_rhs_hist[itr] = sum_dp
#         for component in 1:6
#             population_trace_rhs_components_hist[itr, component] = trace_components_mut[component]
#         end

#         if return_population_difference_audit
#             store_population_difference_audit!(itr, c_t)
#         end

#         if audit_initial_population_direction && itr <= max(1, initial_direction_steps)
#             @printf(stderr, "ver30 population RHS itr=%d", itr)
#             @inbounds for state in 1:n_sys
#                 dp = population_rhs_hist[state, itr]
#                 @printf(stderr, "  dp%d=%+.8e%+.8ei", state, real(dp), imag(dp))
#             end
#             @printf(stderr, "  sum_dp=%+.8e%+.8ei", real(sum_dp), imag(sum_dp))
#             for component in 1:6
#                 value = trace_components_mut[component]
#                 @printf(stderr, "  sum_%s=%+.4e%+.4ei", String(population_rhs_component_labels[component]), real(value), imag(value))
#             end
#             @printf(stderr, "\n")
#         end
#         return sum_dp
#     end

#     start_loop = max(1, start_itr)
#     @inbounds for curr_itr in start_loop:(n_itr - 1)
#         if verbose && (curr_itr == start_loop || curr_itr == n_itr - 1 || ((curr_itr - start_loop) % verbose_every_f == 0))
#             @printf(
#                 stderr,
#                 "Current iteration: %6d / %6d  method=%s  threads=%d  use_threads=%s  secular=%s  state=c  sigma=cyclic-hermitian-population-ver30\n",
#                 curr_itr, n_itr, String(method_sym), Threads.nthreads(), string(use_threads), string(use_secular),
#             )
#         end

#         c_t = @view c_hist[:, :, curr_itr]
#         c_next = @view c_hist[:, :, curr_itr + 1]

#         if method_sym == :euler
#             calc_c_rhs!(k1, curr_itr, q_full(curr_itr), c_t, c_hist)
#             cprime_hist[:, :, curr_itr] .= k1
#             compute_memory_kernel_rowsum_audit!(kernel_rowsum_audit, kernel_output_audit, curr_itr, q_full(curr_itr))
#             store_population_rhs!(curr_itr, k1, c_t)
#             @. c_next = c_t + Δt * k1

#         elseif method_sym == :rk2
#             calc_c_rhs!(k1, curr_itr, q_full(curr_itr), c_t, c_hist)
#             cprime_hist[:, :, curr_itr] .= k1
#             compute_memory_kernel_rowsum_audit!(kernel_rowsum_audit, kernel_output_audit, curr_itr, q_full(curr_itr))
#             store_population_rhs!(curr_itr, k1, c_t)
#             @. c_stage = c_t + 0.5 * Δt * k1
#             use_population_closure && enforce_population_closure!(c_stage)
#             enforce_c_hermitian && enforce_hermiticity!(c_stage)
#             calc_c_rhs!(k2, curr_itr, q_half_after(curr_itr), c_stage, c_hist)
#             @. c_next = c_t + Δt * k2

#         elseif method_sym == :rk4
#             calc_c_rhs!(k1, curr_itr, q_full(curr_itr), c_t, c_hist)
#             cprime_hist[:, :, curr_itr] .= k1
#             compute_memory_kernel_rowsum_audit!(kernel_rowsum_audit, kernel_output_audit, curr_itr, q_full(curr_itr))

#             store_population_rhs!(curr_itr, k1, c_t)

#             @. c_stage = c_t + 0.5 * Δt * k1
#             use_population_closure && enforce_population_closure!(c_stage)
#             enforce_c_hermitian && enforce_hermiticity!(c_stage)
#             calc_c_rhs!(k2, curr_itr, q_half_after(curr_itr), c_stage, c_hist)

#             @. c_stage = c_t + 0.5 * Δt * k2
#             use_population_closure && enforce_population_closure!(c_stage)
#             enforce_c_hermitian && enforce_hermiticity!(c_stage)
#             calc_c_rhs!(k3, curr_itr, q_half_after(curr_itr), c_stage, c_hist)

#             @. c_stage = c_t + Δt * k3
#             use_population_closure && enforce_population_closure!(c_stage)
#             enforce_c_hermitian && enforce_hermiticity!(c_stage)
#             calc_c_rhs!(k4, curr_itr, q_full(curr_itr + 1), c_stage, c_hist)

#             @. c_next = c_t + (Δt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
#         end

#         use_population_closure && enforce_population_closure!(c_next)
#         enforce_c_hermitian && enforce_hermiticity!(c_next)

#         # Store physical sigma only after c has been advanced.
#         readout_sigma_full!(@view(σ[:, :, curr_itr + 1]), c_hist, curr_itr + 1)
#         context.curr_itr = UInt64(curr_itr + 1)
#     end

#     # The integration loop stores k1 only through n_itr-1.  Evaluate the final
#     # full-grid RHS as well so dp1, dp2, ... are available at every saved time.
#     final_itr = Int(context.curr_itr)
#     if 1 <= final_itr <= n_itr
#         c_final = @view c_hist[:, :, final_itr]
#         calc_c_rhs!(k1, final_itr, q_full(final_itr), c_final, c_hist)
#         cprime_hist[:, :, final_itr] .= k1
#         store_population_rhs!(final_itr, k1, c_final)
#     end

#     if return_internal_c || return_branch_memory_audit || return_population_rhs_audit || return_kernel_rowsum_audit || return_kernel_output_audit || return_population_difference_audit
#         fields = (
#             σ = @view(σ[:, :, Int(context.curr_itr)]),
#             c = c_hist,
#             c′ = cprime_hist,
#             population_rhs = return_population_rhs_audit || return_internal_c ? population_rhs_hist : nothing,
#             population_trace_rhs = return_population_rhs_audit || return_internal_c ? population_trace_rhs_hist : nothing,
#             population_rhs_components = return_population_rhs_audit || return_internal_c ? population_rhs_components_hist : nothing,
#             population_trace_rhs_components = return_population_rhs_audit || return_internal_c ? population_trace_rhs_components_hist : nothing,
#             population_rhs_component_labels = return_population_rhs_audit || return_internal_c ? population_rhs_component_labels : nothing,
#             population_rhs_component_residual = return_population_rhs_audit || return_internal_c ? population_rhs_component_residual_hist : nothing,
#             dp1 = (return_population_rhs_audit || return_internal_c) && n_sys >= 1 ? @view(population_rhs_hist[1, :]) : nothing,
#             dp2 = (return_population_rhs_audit || return_internal_c) && n_sys >= 2 ? @view(population_rhs_hist[2, :]) : nothing,
#             sum_dp = return_population_rhs_audit || return_internal_c ? population_trace_rhs_hist : nothing,
#             branch_memory = return_branch_memory_audit ? branch_memory_audit : nothing,
#             kernel_rowsum = return_kernel_rowsum_audit ? kernel_rowsum_audit : nothing,
#             kernel_output = return_kernel_output_audit ? kernel_output_audit : nothing,
#             population_difference = return_population_difference_audit && n_sys == 2 ? population_difference_hist : nothing,
#             population_difference_rhs = return_population_difference_audit && n_sys == 2 ? population_difference_rhs_hist : nothing,
#             population_difference_rhs_components = return_population_difference_audit && n_sys == 2 ? population_difference_rhs_components_hist : nothing,
#             population_rate_1_to_2 = return_population_difference_audit && n_sys == 2 ? population_rate_1_to_2_hist : nothing,
#             population_rate_2_to_1 = return_population_difference_audit && n_sys == 2 ? population_rate_2_to_1_hist : nothing,
#             population_difference_damping = return_population_difference_audit && n_sys == 2 ? population_difference_damping_hist : nothing,
#         )
#         return fields
#     end
#     return @view(σ[:, :, Int(context.curr_itr)])
# end



using LinearAlgebra

# ver31: ver29 dynamics with stationary-state and zero-frequency diagnostics
#
# Symbolically audited against TrB_12_primitives_branch_labeled_gclosure_ver3(1).ipynb.
# F01-F07 and F11-F12 matched exactly; F08-F10 were replaced from the catalog expression trees.
# F08 and F09 each retain two six-term g′ brackets generated by endpoint-aware before/after rules.
# No global/per-branch/readout sign multiplier is exposed or applied.
# Primitive connected signs are F07:+, F08:-, F09:-, F10:+.
# Q=1-P assembly is LL/RR: projected-direct, LR/RL: direct-projected.
# Population readout uses the exact identity σ_aa(t)=c_aa(t).
# Every memory branch includes the outer normalization N_{αβ}(t).
# Full-grid population RHS dp_a=dot(c_aa) and sum_a dp_a are stored explicitly.

function calc__σ_σ′_secular_core!(
    context::RmrtContext;
    hbar::Real = 1.0,
    use_population_closure::Bool = false,

    include_c_L0P_local::Bool = true,
    include_c_PL1P_local::Bool = true,
    include_c_memory_G0::Bool = true,
    include_sigma_G0_readout::Bool = true,


    include_c_memory_LL::Bool = true,
    include_c_memory_LR::Bool = true,
    include_c_memory_RL::Bool = true,
    include_c_memory_RR::Bool = true,

    # Store the full-grid k1 memory contribution per branch.  This is useful
    # for locating which branch breaks population direction or Hermiticity.
    return_branch_memory_audit::Bool = false,

    # Store the actual full-grid population RHS
    #   population_rhs[a, itr] = dot(c_aa)(t_itr)
    # and its trace sum.  This is the direct dp1, dp2, ... diagnostic.
    return_population_rhs_audit::Bool = false,

    # Kernel row-sum audit independent of the current c-history.
    # For each full-grid time, stores the memory-kernel row sum
    #   sum_r K_{rr;ab}^{branch}(t,s integrated)
    # for each input pair (a,b) and branch.  This is the direct test for
    # trace preservation of the c-memory block.
    return_kernel_rowsum_audit::Bool = false,
    # Output-resolved version of the same audit. Dimensions:
    #   kernel_output[r_out, a_in, b_in, itr, branch].
    # This is needed because branch row sums alone can hide whether the
    # leak is a loss-side or gain-side error.
    return_kernel_output_audit::Bool = true,
    # Diagnose the trace-free population-difference mode.
    return_population_difference_audit::Bool = true,
    difference_damping_warn_tol::Float64 = 0.0,
    population_rate_imag_tol::Float64 = 1.0e-8,


    # Do not silently hide algebraic problems.  These audits print when
    # physical sigma develops an imaginary trace or loses Hermiticity.
    audit_trace_hermiticity::Bool = true,
    audit_gclosure_structure::Bool = true,
    audit_initial_population_direction::Bool = true,
    initial_direction_steps::Int = 5,
    trace_imag_tol::Float64 = 1.0e-8,
    trace_real_tol::Float64 = 1.0e-8,
    trace_rhs_tol::Float64 = 1.0e-8,
    abort_on_c_trace_rhs::Bool = true,
    hermiticity_tol::Float64 = 1.0e-8,

    # Exact population-sector coefficient identities implied by cyclicity of
    # the raw bath traces.  These are checked before using a memory kernel.
    audit_population_cyclic_identity::Bool = true,
    population_cyclic_identity_tol::Float64 = 1.0e-9,
    abort_on_population_cyclic_identity::Bool = true,

    use_secular::Bool = false,
    secular_tol::Float64 = 1.0e-10,
    method::Union{Symbol,String} = :rk4,
    use_threads::Bool = true,

    # Validation defaults: do not hide algebraic errors unless explicitly asked.
    enforce_c_hermitian::Bool = false,
    enforce_sigma_hermitian::Bool = false,
    trace_normalize_sigma::Bool = false,

    # c is local to this function.  Therefore resume is not supported unless c
    # is later stored externally.
    allow_resume_without_c_history::Bool = false,

    verbose::Bool = true,
    verbose_every::Int = 1,

    # Stationary/fixed-point diagnostics.  These do not modify propagation.
    print_stationary_diagnostics::Bool = true,
    return_stationary_diagnostics::Bool = false,
    stationary_tail_window::Int = 50,

    # β is needed only for the Gibbs-ratio comparison.  When omitted, the
    # code searches context, context.simulation_details, and context.system
    # for β/beta/inverse_temperature.  It never infers β from temperature.
    stationary_beta::Union{Nothing,Real} = nothing,

    # Optional explicit shifted exciton energies.  When omitted, the code
    # searches for context.ϵ_exci_0 / epsilon_exci_0 / eps_exci_0.
    stationary_shifted_energies::Union{Nothing,AbstractVector} = nothing,

    return_internal_c::Bool = false,
)
    start_itr = Int(context.curr_itr)

    n_sys = context.system.n_sys
    n_itr = context.simulation_details.num_of_iteration
    Δt    = context.simulation_details.Δt
    ϵ     = context.ϵ_exci

    σ   = context.σ              # physical readout output only
    g   = context.g
    gp  = context.g′
    gpp = context.g″

    if start_itr != 1 && !allow_resume_without_c_history
        error("This ver31 population-cyclic-consistent diagnostic implementation keeps c-history inside the function. Set context.curr_itr=1, or implement persistent context.c before resuming.")
    end

    hbar_f = Float64(hbar)
    hbar_f > 0.0 || error("hbar must be positive")

    method_sym = Symbol(lowercase(String(method)))
    method_sym in (:euler, :rk2, :rk4) || error("Unsupported method $(method). Use :euler, :rk2, or :rk4.")


    verbose_every_f = max(1, Int(verbose_every))
    const_ZERO_C = 0.0 + 0.0im
    const_Q_MAX = 2 * n_itr - 1

    # ------------------------------------------------------------------
    # Doubled q-grid:
    #   full grid i      -> q = 2i - 1
    #   half grid i+1/2  -> q = 2i
    #   relative q       -> qΔ = q_t - q_s + 1
    # ------------------------------------------------------------------
    @inline q_full(i::Int) = 2 * i - 1
    @inline q_half_after(i::Int) = 2 * i
    @inline full_index_from_q(q::Int) = (q + 1) >>> 1
    @inline half_index_from_q(q::Int) = q >>> 1

    @inline function check_q(q::Int)
        (1 <= q <= const_Q_MAX) || error("q-grid index out of range: $(q), allowed [1,$(const_Q_MAX)]")
        return q
    end

    function _first_existing_property(obj, names::Tuple)
        for name in names
            if hasproperty(obj, name)
                return getfield(obj, name)
            end
        end
        return nothing
    end

    function _first_existing_property_many(objects::Tuple, names::Tuple)
        for object in objects
            object === nothing && continue
            value = _first_existing_property(object, names)
            value === nothing || return value
        end
        return nothing
    end

    diagnostic_objects = (
        context,
        hasproperty(context, :simulation_details) ? context.simulation_details : nothing,
        hasproperty(context, :system) ? context.system : nothing,
    )

    beta_diagnostic = stationary_beta === nothing ?
        _first_existing_property_many(
            diagnostic_objects,
            (:β, :beta, :inverse_temperature, :inverse_temp),
        ) :
        stationary_beta

    shifted_energies_diagnostic = stationary_shifted_energies === nothing ?
        _first_existing_property_many(
            diagnostic_objects,
            (
                Symbol("ϵ_exci_0"),
                :epsilon_exci_0,
                :eps_exci_0,
                :shifted_exciton_energies,
            ),
        ) :
        stationary_shifted_energies

    stationary_tail_window >= 1 ||
        error("stationary_tail_window must be at least 1")

    g_half = _first_existing_property(context, (
        :g_half, :g_half_shifted, :g_shifted_half, :g_mid, :g_midpoint,
        Symbol("g__half"), Symbol("g_half_grid"), Symbol("g_half_shifted_grid"),
    ))
    gp_half = _first_existing_property(context, (
        Symbol("g′_half"), Symbol("g′_half_shifted"), Symbol("g′_shifted_half"),
        Symbol("g′_mid"), Symbol("g′_midpoint"), Symbol("gp_half"),
        Symbol("gp_half_shifted"), Symbol("g_prime_half"), Symbol("g_prime_half_shifted"),
    ))
    gpp_half = _first_existing_property(context, (
        Symbol("g″_half"), Symbol("g″_half_shifted"), Symbol("g″_shifted_half"),
        Symbol("g″_mid"), Symbol("g″_midpoint"), Symbol("gpp_half"),
        Symbol("gpp_half_shifted"), Symbol("g_doubleprime_half"), Symbol("g_doubleprime_half_shifted"),
    ))

    needs_half_grid = method_sym in (:rk2, :rk4)
    if needs_half_grid && (g_half === nothing || gp_half === nothing || gpp_half === nothing)
        missing = String[]
        g_half === nothing && push!(missing, "g half-grid")
        gp_half === nothing && push!(missing, "g′ half-grid")
        gpp_half === nothing && push!(missing, "g″ half-grid")
        error("$(method_sym) requires precomputed half-grid containers for " * join(missing, ", "))
    end

    @inline function _missing_half_grid_error(name::String)
        error("Attempted to access $(name) at a half-grid q index, but the half-grid container is missing.")
    end

    @inline function G(q::Int, a::Int, b::Int, cidx::Int, d::Int)
        check_q(q)
        if isodd(q)
            @inbounds return g[full_index_from_q(q), a, b, cidx, d]
        else
            g_half === nothing && _missing_half_grid_error("g")
            @inbounds return g_half[half_index_from_q(q), a, b, cidx, d]
        end
    end

    @inline function Gp(q::Int, a::Int, b::Int, cidx::Int, d::Int)
        check_q(q)
        if isodd(q)
            @inbounds return gp[full_index_from_q(q), a, b, cidx, d]
        else
            gp_half === nothing && _missing_half_grid_error("g′")
            @inbounds return gp_half[half_index_from_q(q), a, b, cidx, d]
        end
    end

    @inline function Gpp(q::Int, a::Int, b::Int, cidx::Int, d::Int)
        check_q(q)
        if isodd(q)
            @inbounds return gpp[full_index_from_q(q), a, b, cidx, d]
        else
            gpp_half === nothing && _missing_half_grid_error("g″")
            @inbounds return gpp_half[half_index_from_q(q), a, b, cidx, d]
        end
    end

    # q-time and signed-time g-family helpers.
    # sign=+1: f_{ab,cd}(+τ)
    # sign=-1: use the v5 negative-time identities:
    #   g_abcd(-τ)    = conj(g_dcba(+τ))
    #   g′_abcd(-τ)   = -conj(g′_dcba(+τ))
    #   g″_abcd(-τ)   = conj(g″_dcba(+τ))
    @inline function qtime(q::Int)
        return 0.5 * (Float64(q) - 1.0) * Δt
    end

    @inline function Gv(q::Int, a::Int, b::Int, cidx::Int, d::Int, sign::Int)
        return sign == 1 ? G(q, a, b, cidx, d) : conj(G(q, d, cidx, b, a))
    end

    @inline function Gpv(q::Int, a::Int, b::Int, cidx::Int, d::Int, sign::Int)
        return sign == 1 ? Gp(q, a, b, cidx, d) : -conj(Gp(q, d, cidx, b, a))
    end

    @inline function Gppv(q::Int, a::Int, b::Int, cidx::Int, d::Int, sign::Int)
        return sign == 1 ? Gpp(q, a, b, cidx, d) : conj(Gpp(q, d, cidx, b, a))
    end

    @inline function is_secular_pair(out_a::Int, out_b::Int, in_a::Int, in_b::Int)
        if !use_secular
            return true
        end
        return abs(((ϵ[out_a] - ϵ[out_b]) - (ϵ[in_a] - ϵ[in_b])) / hbar_f) <= secular_tol
    end

    @inline function c_local(c_t::AbstractMatrix, a::Int, b::Int)
        return (use_population_closure && a != b) ? const_ZERO_C : c_t[a, b]
    end

    @inline function c_mem(c_t::AbstractMatrix, qs::Int, qt::Int, a::Int, b::Int, c_hist::Array{ComplexF64,3})
        if use_population_closure && a != b
            return const_ZERO_C
        end
        if qs == qt
            return c_t[a, b]
        end
        isodd(qs) || error("non-endpoint memory q index must be a stored full-grid point, got q=$(qs)")
        return c_hist[a, b, full_index_from_q(qs)]
    end

    @inline function n_outer_nodes(curr_itr::Int, qt::Int)
        q_now = q_full(curr_itr)
        return qt > q_now ? curr_itr + 1 : curr_itr
    end

    @inline function outer_node_coord(node_idx::Int, curr_itr::Int, qt::Int)
        return node_idx <= curr_itr ? q_full(node_idx) : qt
    end

    @inline function ∫weight_to_t(node_idx::Int, curr_itr::Int, qt::Int)
        n_nodes = n_outer_nodes(curr_itr, qt)
        n_nodes <= 1 && return 0.0
        q = outer_node_coord(node_idx, curr_itr, qt)
        if node_idx == 1
            q_next = outer_node_coord(node_idx + 1, curr_itr, qt)
            return 0.25 * Δt * Float64(q_next - q)
        elseif node_idx == n_nodes
            q_prev = outer_node_coord(node_idx - 1, curr_itr, qt)
            return 0.25 * Δt * Float64(q - q_prev)
        else
            q_prev = outer_node_coord(node_idx - 1, curr_itr, qt)
            q_next = outer_node_coord(node_idx + 1, curr_itr, qt)
            return 0.25 * Δt * Float64(q_next - q_prev)
        end
    end

    # ------------------------------------------------------------------
    # ver24 direct notebook port.
    # Catalog source: TrB_12_primitives_complete_gclosure_ver4.ipynb
    # Coefficient source: L0_transported_normalized_complete_gclosed_dynamics_readout_ver4.ipynb
    # Generated coefficient block SHA256: 42dc92912043cd811eb378a0756580e99bfda89053738dc964a4000e9fd9152d
    # --------------------------------------------------------------
    # ver24: direct notebook port.
    # g, g′, g″ in context are dimensional correlation integrals; the
    # notebook catalog uses dimensionless g-family values, hence /hbar².
    # No contour or primitive is re-derived in Julia.
    # --------------------------------------------------------------
    const_inv_hbar2 = 1.0 / (hbar_f * hbar_f)
    @inline _gd(q::Int,a::Int,b::Int,c::Int,d::Int) = const_inv_hbar2 * G(q,a,b,c,d)
    @inline _gpd(q::Int,a::Int,b::Int,c::Int,d::Int) = const_inv_hbar2 * Gp(q,a,b,c,d)
    @inline _gppd(q::Int,a::Int,b::Int,c::Int,d::Int) = const_inv_hbar2 * Gpp(q,a,b,c,d)

    @inline function _N_g(qT::Int, a::Int, b::Int)
        return exp(- _gd(qT, a, a, a, a) + _gd(qT, b, b, a, a) + conj(_gd(qT, a, a, b, b)) - conj(_gd(qT, b, b, b, b)))
    end

    @inline function _T_F01(qT::Int, a::Int, b::Int)
        return -1 * 1.0im * hbar_f * (_gpd(qT, a, a, a, a) - conj(_gpd(qT, a, a, b, b)))
    end

    @inline function _T_F02(qT::Int, a::Int, b::Int)
        return -1 * 1.0im * hbar_f * (_gpd(qT, b, b, a, a) - conj(_gpd(qT, b, b, b, b)))
    end

    @inline function _T_F03(qT::Int, a::Int, p::Int, b::Int)
        return -1 * 1.0im * hbar_f * (_gpd(qT, a, p, p, p) - conj(_gpd(qT, p, a, a, a))) * exp(_gd(qT, a, a, p, p) - _gd(qT, b, b, p, p) - conj(_gd(qT, a, a, a, a)) + conj(_gd(qT, b, b, b, b)) + conj(_gd(qT, p, p, a, a)) - conj(_gd(qT, p, p, b, b)))
    end

    @inline function _T_F04(qT::Int, a::Int, q::Int, b::Int)
        return -1 * 1.0im * hbar_f * (_gpd(qT, q, b, b, b) - conj(_gpd(qT, b, q, q, q))) * exp(_gd(qT, a, a, a, a) - _gd(qT, b, b, b, b) - _gd(qT, q, q, a, a) + _gd(qT, q, q, b, b) - conj(_gd(qT, a, a, q, q)) + conj(_gd(qT, b, b, q, q)))
    end

    @inline function _T_F05(qS::Int, qD::Int, qT::Int, a::Int, p::Int, b::Int)
        return -1 * 1.0im * hbar_f * (_gpd(qT, a, p, p, p) - conj(_gpd(qT, p, a, a, a))) * exp(_gd(qT, a, a, p, p) - _gd(qS, b, b, p, p) + _gd(qS, p, p, p, p) - _gd(qT, p, p, p, p) - conj(_gd(qT, a, a, a, a)) + conj(_gd(qS, b, b, b, b)) + conj(_gd(qT, p, p, a, a)) - conj(_gd(qS, p, p, b, b)))
    end

    @inline function _T_F06(qS::Int, qD::Int, qT::Int, a::Int, q::Int, b::Int)
        return -1 * 1.0im * hbar_f * (_gpd(qT, q, b, b, b) - conj(_gpd(qT, b, q, q, q))) * exp(_gd(qS, a, a, a, a) - _gd(qT, b, b, b, b) - _gd(qS, q, q, a, a) + _gd(qT, q, q, b, b) - conj(_gd(qS, a, a, q, q)) + conj(_gd(qT, b, b, q, q)) + conj(_gd(qS, q, q, q, q)) - conj(_gd(qT, q, q, q, q)))
    end

    @inline function _T_F07(qS::Int, qD::Int, qT::Int, a::Int, p::Int, r::Int, b::Int)
        return (- (hbar_f)^(2) * (- _gpd(qD, a, a, p, r) + _gpd(qD, p, p, p, r) + _gpd(qS, p, r, r, r) - conj(_gpd(qS, r, p, a, a))) * (_gpd(qD, a, p, p, p) + _gpd(qT, a, p, r, r) - _gpd(qD, a, p, r, r) - conj(_gpd(qT, p, a, a, a))) + (hbar_f)^(2) * _gppd(qD, a, p, p, r)) * exp(_gd(qD, a, a, p, p) + _gd(qT, a, a, r, r) - _gd(qD, a, a, r, r) - _gd(qS, b, b, r, r) - _gd(qD, p, p, p, p) + _gd(qS, p, p, r, r) - _gd(qT, p, p, r, r) + _gd(qD, p, p, r, r) - conj(_gd(qT, a, a, a, a)) + conj(_gd(qS, b, b, b, b)) - conj(_gd(qS, p, p, a, a)) + conj(_gd(qT, p, p, a, a)) + conj(_gd(qS, r, r, a, a)) - conj(_gd(qS, r, r, b, b)))
    end

    @inline function _T_F08(qS::Int, qD::Int, qT::Int, a::Int, p::Int, q::Int, b::Int)
        return (- (hbar_f)^(2) * (- _gpd(qS, q, b, a, a) + _gpd(qS, q, b, b, b) + _gpd(qS, q, b, p, p) - conj(_gpd(qD, a, a, b, q)) - conj(_gpd(qS, b, q, q, q)) + conj(_gpd(qD, p, p, b, q))) * (_gpd(qT, a, p, p, p) - conj(_gpd(qT, p, a, a, a)) + conj(_gpd(qT, p, a, b, b)) - conj(_gpd(qD, p, a, b, b)) - conj(_gpd(qT, p, a, q, q)) + conj(_gpd(qD, p, a, q, q))) + (hbar_f)^(2) * conj(_gppd(qD, p, a, b, q))) * exp(_gd(qT, a, a, p, p) + _gd(qS, b, b, a, a) - _gd(qS, b, b, b, b) - _gd(qS, b, b, p, p) + _gd(qS, p, p, p, p) - _gd(qT, p, p, p, p) - _gd(qS, q, q, a, a) + _gd(qS, q, q, b, b) - conj(_gd(qT, a, a, a, a)) + conj(_gd(qT, a, a, b, b)) - conj(_gd(qD, a, a, b, b)) - conj(_gd(qT, a, a, q, q)) + conj(_gd(qD, a, a, q, q)) + conj(_gd(qS, b, b, q, q)) + conj(_gd(qT, p, p, a, a)) - conj(_gd(qT, p, p, b, b)) + conj(_gd(qD, p, p, b, b)) - conj(_gd(qS, p, p, q, q)) + conj(_gd(qT, p, p, q, q)) - conj(_gd(qD, p, p, q, q)))
    end

    @inline function _T_F09(qS::Int, qD::Int, qT::Int, a::Int, p::Int, q::Int, b::Int)
        return (- (hbar_f)^(2) * (- _gpd(qT, q, b, a, a) + _gpd(qD, q, b, a, a) + _gpd(qT, q, b, b, b) + _gpd(qT, q, b, p, p) - _gpd(qD, q, b, p, p) - conj(_gpd(qT, b, q, q, q))) * (_gpd(qS, a, p, p, p) + _gpd(qD, b, b, a, p) - _gpd(qD, q, q, a, p) - conj(_gpd(qS, p, a, a, a)) + conj(_gpd(qS, p, a, b, b)) - conj(_gpd(qS, p, a, q, q))) + (hbar_f)^(2) * _gppd(qD, q, b, a, p)) * exp(_gd(qS, a, a, p, p) + _gd(qT, b, b, a, a) - _gd(qD, b, b, a, a) - _gd(qT, b, b, b, b) - _gd(qT, b, b, p, p) + _gd(qD, b, b, p, p) - _gd(qT, q, q, a, a) + _gd(qD, q, q, a, a) + _gd(qT, q, q, b, b) - _gd(qS, q, q, p, p) + _gd(qT, q, q, p, p) - _gd(qD, q, q, p, p) - conj(_gd(qS, a, a, a, a)) + conj(_gd(qS, a, a, b, b)) - conj(_gd(qS, a, a, q, q)) + conj(_gd(qT, b, b, q, q)) + conj(_gd(qS, p, p, a, a)) - conj(_gd(qS, p, p, b, b)) + conj(_gd(qS, q, q, q, q)) - conj(_gd(qT, q, q, q, q)))
    end

    @inline function _T_F10(qS::Int, qD::Int, qT::Int, a::Int, u::Int, q::Int, b::Int)
        return (- (hbar_f)^(2) * (_gpd(qT, q, b, b, b) - conj(_gpd(qD, b, q, q, q)) - conj(_gpd(qT, b, q, u, u)) + conj(_gpd(qD, b, q, u, u))) * (_gpd(qS, u, q, b, b) + conj(_gpd(qD, b, b, q, u)) - conj(_gpd(qD, q, q, q, u)) - conj(_gpd(qS, q, u, u, u))) + (hbar_f)^(2) * conj(_gppd(qD, b, q, q, u))) * exp(_gd(qS, a, a, a, a) - _gd(qT, b, b, b, b) - _gd(qS, q, q, b, b) + _gd(qT, q, q, b, b) - _gd(qS, u, u, a, a) + _gd(qS, u, u, b, b) - conj(_gd(qS, a, a, u, u)) + conj(_gd(qD, b, b, q, q)) + conj(_gd(qT, b, b, u, u)) - conj(_gd(qD, b, b, u, u)) - conj(_gd(qD, q, q, q, q)) + conj(_gd(qS, q, q, u, u)) - conj(_gd(qT, q, q, u, u)) + conj(_gd(qD, q, q, u, u)))
    end

    @inline function _T_F11(qS::Int, qD::Int, qT::Int, a::Int, p::Int, b::Int)
        return -1 * 1.0im * hbar_f * (_gpd(qD, a, a, a, p) + _gpd(qS, a, p, p, p) - _gpd(qD, b, b, a, p) - conj(_gpd(qS, p, a, b, b))) * exp(- _gd(qD, a, a, a, a) + _gd(qS, a, a, p, p) - _gd(qT, a, a, p, p) + _gd(qD, a, a, p, p) + _gd(qD, b, b, a, a) - _gd(qS, b, b, p, p) + _gd(qT, b, b, p, p) - _gd(qD, b, b, p, p) - conj(_gd(qS, a, a, b, b)) + conj(_gd(qT, a, a, b, b)) + conj(_gd(qS, b, b, b, b)) - conj(_gd(qT, b, b, b, b)))
    end

    @inline function _T_F12(qS::Int, qD::Int, qT::Int, a::Int, q::Int, b::Int)
        return -1 * 1.0im * hbar_f * (_gpd(qS, q, b, a, a) + conj(_gpd(qD, a, a, b, q)) - conj(_gpd(qD, b, b, b, q)) - conj(_gpd(qS, b, q, q, q))) * exp(_gd(qS, a, a, a, a) - _gd(qT, a, a, a, a) - _gd(qS, b, b, a, a) + _gd(qT, b, b, a, a) + conj(_gd(qD, a, a, b, b)) - conj(_gd(qS, a, a, q, q)) + conj(_gd(qT, a, a, q, q)) - conj(_gd(qD, a, a, q, q)) - conj(_gd(qD, b, b, b, b)) + conj(_gd(qS, b, b, q, q)) - conj(_gd(qT, b, b, q, q)) + conj(_gd(qD, b, b, q, q)))
    end

    @inline function _phase(qT::Int, qS::Int, left::Int, right::Int)
        delta_t = qtime(qT) - qtime(qS)
        return exp(-1.0im * ((ϵ[left] - ϵ[right]) / hbar_f) * delta_t)
    end

    @inline function coef_M_P_0(qt::Int, a::Int, b::Int)
        return (
            -1.0im * (ϵ[a] - ϵ[b]) / hbar_f
            -1.0im / hbar_f * (_T_F01(qt,a,b) - _T_F02(qt,a,b))
        )
    end

    @inline function coef_M_P_L(qt::Int, a::Int, b::Int, p::Int)
        return -1.0im / hbar_f * _N_g(qt,a,b) * _T_F03(qt,a,p,b)
    end

    @inline function coef_M_P_R(qt::Int, a::Int, b::Int, q::Int)
        return +1.0im / hbar_f * _N_g(qt,a,b) * _T_F04(qt,a,q,b)
    end

    @inline function coef_K_LL(qs::Int, qD::Int, qt::Int,
                               a::Int, b::Int, p::Int, r::Int)
        return (
            _N_g(qt,a,b) / hbar_f^2
            * _phase(qt,qs,p,b)
            * (
                _N_g(qs,p,b) * _T_F05(qs,qD,qt,a,p,b) * _T_F03(qs,p,r,b)
                - _T_F07(qs,qD,qt,a,p,r,b)
            )
        )
    end

    @inline function coef_K_LR(qs::Int, qD::Int, qt::Int,
                               a::Int, b::Int, p::Int, q::Int)
        return (
            -_N_g(qt,a,b) / hbar_f^2
            * _phase(qt,qs,p,b)
            * (
                _N_g(qs,p,b) * _T_F05(qs,qD,qt,a,p,b) * _T_F04(qs,p,q,b)
                - _T_F08(qs,qD,qt,a,p,q,b)
            )
        )
    end

    @inline function coef_K_RL(qs::Int, qD::Int, qt::Int,
                               a::Int, b::Int, p::Int, q::Int)
        return (
            -_N_g(qt,a,b) / hbar_f^2
            * _phase(qt,qs,a,q)
            * (
                _N_g(qs,a,q) * _T_F06(qs,qD,qt,a,q,b) * _T_F03(qs,a,p,q)
                - _T_F09(qs,qD,qt,a,p,q,b)
            )
        )
    end

    @inline function coef_K_RR(qs::Int, qD::Int, qt::Int,
                               a::Int, b::Int, q::Int, u::Int)
        return (
            _N_g(qt,a,b) / hbar_f^2
            * _phase(qt,qs,a,q)
            * (
                _N_g(qs,a,q) * _T_F06(qs,qD,qt,a,q,b) * _T_F04(qs,a,u,q)
                - _T_F10(qs,qD,qt,a,u,q,b)
            )
        )
    end

    @inline function coef_Rσ_L(qs::Int, qD::Int, qt::Int,
                               a::Int, b::Int, p::Int)
        return (
            +1.0im / hbar_f
            * _phase(qt,qs,a,b)
            * (_N_g(qt,a,b) * _T_F03(qs,a,p,b) - _T_F11(qs,qD,qt,a,p,b))
        )
    end

    @inline function coef_Rσ_R(qs::Int, qD::Int, qt::Int,
                               a::Int, b::Int, q::Int)
        return (
            -1.0im / hbar_f
            * _phase(qt,qs,a,b)
            * (_N_g(qt,a,b) * _T_F04(qs,a,q,b) - _T_F12(qs,qD,qt,a,q,b))
        )
    end

    @inline function enforce_hermiticity!(mat::AbstractMatrix)
        for i in 1:n_sys
            mat[i, i] = real(mat[i, i]) + 0.0im
        end
        for i in 1:(n_sys - 1), j in (i + 1):n_sys
            z = 0.5 * (mat[i, j] + conj(mat[j, i]))
            mat[i, j] = z
            mat[j, i] = conj(z)
        end
        return mat
    end

    @inline function enforce_population_closure!(mat::AbstractMatrix)
        for j in 1:n_sys, i in 1:n_sys
            mat[i, j] = (i == j) ? (real(mat[i, i]) + 0.0im) : const_ZERO_C
        end
        return mat
    end

    @inline function trace_normalize!(mat::AbstractMatrix)
        tr = zero(ComplexF64)
        for i in 1:n_sys
            tr += mat[i, i]
        end
        if abs(tr) > 0
            mat ./= tr
        end
        return mat
    end

    # --------------------------------------------------------------
    # Exact two-level population cyclicity audit.
    #
    # For x != y and every (t,s), the flat raw traces satisfy
    #
    #   F07(x,y,x,x) = F09(y,x,x,y),
    #   F10(x,x,y,x) = F08(y,x,x,y),
    #
    # by cyclicity of Tr_B.  Including the projected Q=1-P pieces and phases,
    # this gives the coefficient-level row-sum identities
    #
    #   K_LL[x<-x] + K_RL[y<-x] = 0,
    #   K_RR[x<-x] + K_LR[y<-x] = 0.
    #
    # A violation is an implementation error, not a physical approximation.
    # --------------------------------------------------------------
    Base.@noinline function audit_population_cyclic_identity_at!(
        curr_itr::Int,
        qt::Int,
    )
        if !audit_population_cyclic_identity || n_sys != 2 || qt <= 1
            return nothing
        end

        max_residual = 0.0
        worst = nothing
        n_s = n_outer_nodes(curr_itr, qt)

        for x in 1:2
            y = 3 - x

            for s_node in 1:n_s
                qs = outer_node_coord(s_node, curr_itr, qt)
                qD = qt - qs + 1

                left_pair = (
                    coef_K_LL(qs, qD, qt, x, x, y, x)
                    + coef_K_RL(qs, qD, qt, y, y, x, x)
                )

                right_pair = (
                    coef_K_RR(qs, qD, qt, x, x, y, x)
                    + coef_K_LR(qs, qD, qt, y, y, x, x)
                )

                left_residual = abs(left_pair)
                right_residual = abs(right_pair)

                if left_residual > max_residual
                    max_residual = left_residual
                    worst = (:left_pair, x, y, qs, qD, left_pair)
                end

                if right_residual > max_residual
                    max_residual = right_residual
                    worst = (:right_pair, x, y, qs, qD, right_pair)
                end
            end
        end

        if max_residual > population_cyclic_identity_tol
            @printf(
                stderr,
                "ver31 population cyclic-identity failure: itr=%d q=%d max_residual=%.6e worst=%s\n",
                curr_itr,
                qt,
                max_residual,
                repr(worst),
            )

            if abort_on_population_cyclic_identity
                error(
                    "Population cyclic trace identity failed before propagation: " *
                    "itr=$(curr_itr), q=$(qt), max_residual=$(max_residual)"
                )
            end
        end

        return nothing
    end

    Base.@noinline function calc_c_rhs_element!(rhs_mat::AbstractMatrix, curr_itr::Int, qt::Int, c_t::AbstractMatrix, c_hist::Array{ComplexF64,3}, α::Int, β::Int)
        if use_population_closure && α != β
            rhs_mat[α, β] = const_ZERO_C
            return nothing
        end

        rhs = const_ZERO_C

        if include_c_L0P_local
            rhs += coef_M_P_0(qt, α, β) * c_local(c_t, α, β)
        end

        if include_c_PL1P_local
            for μ in 1:n_sys
                μ == α && continue
                if is_secular_pair(α, β, μ, β)
                    rhs += coef_M_P_L(qt, α, β, μ) * c_local(c_t, μ, β)
                end
            end
            for ν in 1:n_sys
                ν == β && continue
                if is_secular_pair(α, β, α, ν)
                    rhs += coef_M_P_R(qt, α, β, ν) * c_local(c_t, α, ν)
                end
            end
        end

        if include_c_memory_G0 && qt > 1
            integral_LL = const_ZERO_C
            integral_LR = const_ZERO_C
            integral_RL = const_ZERO_C
            integral_RR = const_ZERO_C
            n_s = n_outer_nodes(curr_itr, qt)
            for s_node in 1:n_s
                qs = outer_node_coord(s_node, curr_itr, qt)
                qΔ = qt - qs + 1
                w = ∫weight_to_t(s_node, curr_itr, qt)
                w == 0.0 && continue

                branch_LL = const_ZERO_C
                branch_LR = const_ZERO_C
                branch_RL = const_ZERO_C
                branch_RR = const_ZERO_C

                # LL: input (μ, β), first intermediate χ, second insertion χ -> μ.
                if include_c_memory_LL
                    for χ in 1:n_sys
                        χ == α && continue
                        for μ in 1:n_sys
                            μ == χ && continue
                            if is_secular_pair(α, β, μ, β)
                                branch_LL += coef_K_LL(qs, qΔ, qt, α, β, χ, μ) * c_mem(c_t, qs, qt, μ, β, c_hist)
                            end
                        end
                    end
                end

                # LR: input (χ, ν).
                if include_c_memory_LR
                    for χ in 1:n_sys
                        χ == α && continue
                        for ν in 1:n_sys
                            ν == β && continue
                            if is_secular_pair(α, β, χ, ν)
                                branch_LR += coef_K_LR(qs, qΔ, qt, α, β, χ, ν) * c_mem(c_t, qs, qt, χ, ν, c_hist)
                            end
                        end
                    end
                end

                # RL: input (μ, ν).
                if include_c_memory_RL
                    for μ in 1:n_sys
                        μ == α && continue
                        for ν in 1:n_sys
                            ν == β && continue
                            if is_secular_pair(α, β, μ, ν)
                                branch_RL += coef_K_RL(qs, qΔ, qt, α, β, μ, ν) * c_mem(c_t, qs, qt, μ, ν, c_hist)
                            end
                        end
                    end
                end

                # RR: input (α, ν), first intermediate χ, second insertion ν -> χ.
                if include_c_memory_RR
                    for χ in 1:n_sys
                        χ == β && continue
                        for ν in 1:n_sys
                            ν == χ && continue
                            if is_secular_pair(α, β, α, ν)
                                branch_RR += coef_K_RR(qs, qΔ, qt, α, β, χ, ν) * c_mem(c_t, qs, qt, α, ν, c_hist)
                            end
                        end
                    end
                end

                integral_LL += w * branch_LL
                integral_LR += w * branch_LR
                integral_RL += w * branch_RL
                integral_RR += w * branch_RR
            end

            # Exact result-based assembly.  The coefficient functions already
            # contain every Liouville-branch sign and Q=1-P subtraction.
            rhs += integral_LL + integral_LR + integral_RL + integral_RR

            if need_branch_memory_storage && qt == q_full(curr_itr)
                branch_memory_audit[α, β, curr_itr, 1] = integral_LL
                branch_memory_audit[α, β, curr_itr, 2] = integral_LR
                branch_memory_audit[α, β, curr_itr, 3] = integral_RL
                branch_memory_audit[α, β, curr_itr, 4] = integral_RR
            end
        end

        rhs_mat[α, β] = rhs
        return nothing
    end

    Base.@noinline function calc_c_rhs!(rhs_mat::AbstractMatrix, curr_itr::Int, qt::Int, c_t::AbstractMatrix, c_hist::Array{ComplexF64,3})
        audit_population_cyclic_identity_at!(curr_itr, qt)
        fill!(rhs_mat, const_ZERO_C)
        n_components = n_sys * n_sys
        heavy = include_c_memory_G0
        if use_threads && Threads.nthreads() > 1 && (heavy || n_components >= max(16, 4 * Threads.nthreads()))
            Threads.@threads for linear_idx in 1:n_components
                α = ((linear_idx - 1) % n_sys) + 1
                β = ((linear_idx - 1) ÷ n_sys) + 1
                calc_c_rhs_element!(rhs_mat, curr_itr, qt, c_t, c_hist, α, β)
            end
        else
            for β in 1:n_sys, α in 1:n_sys
                calc_c_rhs_element!(rhs_mat, curr_itr, qt, c_t, c_hist, α, β)
            end
        end

        # For the normalized projector, diagonal c elements are physical
        # populations.  Therefore sum_a dot(c_aa) must vanish.
        if audit_trace_hermiticity
            trace_rhs = const_ZERO_C
            @inbounds for aidx in 1:n_sys
                trace_rhs += rhs_mat[aidx, aidx]
            end
            if abs(trace_rhs) > trace_rhs_tol
                stage_label = isodd(qt) ? "full" : "half"
                @printf(
                    stderr,
                    "ver31 c-trace RHS warning at itr=%d q=%d stage=%s: sum_a dot(c_aa)=%+.8e%+.8ei  abs=%.3e",
                    curr_itr,
                    qt,
                    stage_label,
                    real(trace_rhs),
                    imag(trace_rhs),
                    abs(trace_rhs),
                )
                @inbounds for aidx in 1:n_sys
                    dpa = rhs_mat[aidx, aidx]
                    @printf(
                        stderr,
                        "  dp%d=%+.8e%+.8ei",
                        aidx,
                        real(dpa),
                        imag(dpa),
                    )
                end
                @printf(stderr, "\n")
                if abort_on_c_trace_rhs
                    error(
                        "ver31 trace-RHS identity failed before propagation could be trusted: " *
                        "itr=$(curr_itr), q=$(qt), sum_a dot(c_aa)=$(trace_rhs)"
                    )
                end
            end
        end
        return rhs_mat
    end


    Base.@noinline function compute_memory_kernel_rowsum_audit!(kernel_rowsum_audit::Array{ComplexF64,4}, kernel_output_audit::Array{ComplexF64,5}, curr_itr::Int, qt::Int)
        # Stores, for this full-grid time t, the integrated row sum
        #   S_{ab}^{B}(t) = ∫_0^t ds sum_r K^{B}_{rr;ab}(t,s)
        # independently of c_ab(s).  If the memory kernel is trace-preserving,
        # the signed total over branches should be approximately zero for every
        # input pair (a,b).
        if !(return_kernel_rowsum_audit || return_kernel_output_audit) || !include_c_memory_G0 || qt <= 1
            return nothing
        end

        if return_kernel_rowsum_audit
            @inbounds for b_in in 1:n_sys, a_in in 1:n_sys, br in 1:4
                kernel_rowsum_audit[a_in, b_in, curr_itr, br] = const_ZERO_C
            end
        end
        if return_kernel_output_audit
            @inbounds for r_out in 1:n_sys, b_in in 1:n_sys, a_in in 1:n_sys, br in 1:4
                kernel_output_audit[r_out, a_in, b_in, curr_itr, br] = const_ZERO_C
            end
        end

        n_s = n_outer_nodes(curr_itr, qt)
        for s_node in 1:n_s
            qs = outer_node_coord(s_node, curr_itr, qt)
            qΔ = qt - qs + 1
            w = ∫weight_to_t(s_node, curr_itr, qt)
            w == 0.0 && continue

            # Trace output means output pair is (r,r), summed over r.
            for r in 1:n_sys
                α = r
                β = r

                # LL: input (μ, r).
                if include_c_memory_LL
                    for χ in 1:n_sys
                        χ == α && continue
                        for μ in 1:n_sys
                            μ == χ && continue
                            if is_secular_pair(α, β, μ, β)
                                begin
                                    val = w * coef_K_LL(qs, qΔ, qt, α, β, χ, μ)
                                    return_kernel_rowsum_audit && (kernel_rowsum_audit[μ, β, curr_itr, 1] += val)
                                    return_kernel_output_audit && (kernel_output_audit[r, μ, β, curr_itr, 1] += val)
                                end
                            end
                        end
                    end
                end

                # LR: input (χ, ν).
                if include_c_memory_LR
                    for χ in 1:n_sys
                        χ == α && continue
                        for ν in 1:n_sys
                            ν == β && continue
                            if is_secular_pair(α, β, χ, ν)
                                begin
                                    val = w * coef_K_LR(qs, qΔ, qt, α, β, χ, ν)
                                    return_kernel_rowsum_audit && (kernel_rowsum_audit[χ, ν, curr_itr, 2] += val)
                                    return_kernel_output_audit && (kernel_output_audit[r, χ, ν, curr_itr, 2] += val)
                                end
                            end
                        end
                    end
                end

                # RL: input (μ, ν).  Here ν is the right-side intermediate/input index.
                if include_c_memory_RL
                    for μ in 1:n_sys
                        μ == α && continue
                        for ν in 1:n_sys
                            ν == β && continue
                            if is_secular_pair(α, β, μ, ν)
                                begin
                                    val = w * coef_K_RL(qs, qΔ, qt, α, β, μ, ν)
                                    return_kernel_rowsum_audit && (kernel_rowsum_audit[μ, ν, curr_itr, 3] += val)
                                    return_kernel_output_audit && (kernel_output_audit[r, μ, ν, curr_itr, 3] += val)
                                end
                            end
                        end
                    end
                end

                # RR: input (r, ν).
                if include_c_memory_RR
                    for χ in 1:n_sys
                        χ == β && continue
                        for ν in 1:n_sys
                            ν == χ && continue
                            if is_secular_pair(α, β, α, ν)
                                begin
                                    val = w * coef_K_RR(qs, qΔ, qt, α, β, χ, ν)
                                    return_kernel_rowsum_audit && (kernel_rowsum_audit[α, ν, curr_itr, 4] += val)
                                    return_kernel_output_audit && (kernel_output_audit[r, α, ν, curr_itr, 4] += val)
                                end
                            end
                        end
                    end
                end
            end
        end
        return nothing
    end

    Base.@noinline function readout_sigma_element(c_hist::Array{ComplexF64,3}, c_t::AbstractMatrix, curr_itr::Int, qt::Int, α::Int, β::Int)
        val = c_local(c_t, α, β)

        # Exact identity of the normalized transported projector:
        #   σ_aa(t) = Tr_B ρ_aa(t) = c_aa(t).
        # The approximate Q-space readout must never be applied to populations.
        if α == β
            return val
        end

        if include_sigma_G0_readout && qt > 1
            n_s = n_outer_nodes(curr_itr, qt)
            corr = const_ZERO_C
            for s_node in 1:n_s
                qs = outer_node_coord(s_node, curr_itr, qt)
                qΔ = qt - qs + 1
                w = ∫weight_to_t(s_node, curr_itr, qt)
                w == 0.0 && continue

                # Equation 1 readout sums are not secular-projected by default.
                for μ in 1:n_sys
                    μ == α && continue
                    corr += w * coef_Rσ_L(qs, qΔ, qt, α, β, μ) * c_mem(c_t, qs, qt, μ, β, c_hist)
                end
                for ν in 1:n_sys
                    ν == β && continue
                    corr += w * coef_Rσ_R(qs, qΔ, qt, α, β, ν) * c_mem(c_t, qs, qt, α, ν, c_hist)
                end
            end
            val += corr
        end
        return val
    end

    Base.@noinline function readout_sigma_full!(σ_target::AbstractMatrix, c_hist::Array{ComplexF64,3}, curr_itr::Int)
        qt = q_full(curr_itr)
        c_t = @view c_hist[:, :, curr_itr]
        if use_threads && Threads.nthreads() > 1 && n_sys * n_sys >= max(16, 4 * Threads.nthreads())
            Threads.@threads for linear_idx in 1:(n_sys * n_sys)
                α = ((linear_idx - 1) % n_sys) + 1
                β = ((linear_idx - 1) ÷ n_sys) + 1
                σ_target[α, β] = readout_sigma_element(c_hist, c_t, curr_itr, qt, α, β)
            end
        else
            for β in 1:n_sys, α in 1:n_sys
                σ_target[α, β] = readout_sigma_element(c_hist, c_t, curr_itr, qt, α, β)
            end
        end
        use_population_closure && enforce_population_closure!(σ_target)
        enforce_sigma_hermitian && enforce_hermiticity!(σ_target)
        trace_normalize_sigma && trace_normalize!(σ_target)

        if audit_trace_hermiticity
            trσ = const_ZERO_C
            for i in 1:n_sys
                trσ += σ_target[i, i]
            end
            sigma_herm_res = 0.0
            c_herm_res = 0.0
            for j in 1:n_sys, i in 1:n_sys
                sigma_herm_res = max(
                    sigma_herm_res,
                    abs(σ_target[i, j] - conj(σ_target[j, i])),
                )
                c_herm_res = max(
                    c_herm_res,
                    abs(c_t[i, j] - conj(c_t[j, i])),
                )
            end
            trace_real_res = abs(real(trσ) - 1.0)
            if (
                trace_real_res > trace_real_tol
                || abs(imag(trσ)) > trace_imag_tol
                || sigma_herm_res > hermiticity_tol
                || c_herm_res > hermiticity_tol
            )
                @printf(
                    stderr,
                    "ver29 exact-result audit warning at itr=%d: trace=%+.8e%+.8ei  Re(trace)-1=%.3e  Im(trace)=%.3e  c_herm_res=%.3e  sigma_herm_res=%.3e\n",
                    curr_itr,
                    real(trσ),
                    imag(trσ),
                    trace_real_res,
                    abs(imag(trσ)),
                    c_herm_res,
                    sigma_herm_res,
                )
            end
        end

        return σ_target
    end

    # ------------------------------------------------------------------
    # Local internal state c and derivative c′.
    # ------------------------------------------------------------------
    c_hist = Array{ComplexF64}(undef, n_sys, n_sys, n_itr)
    cprime_hist = Array{ComplexF64}(undef, n_sys, n_sys, n_itr)
    population_rhs_hist = Array{ComplexF64}(undef, n_sys, n_itr)
    population_trace_rhs_hist = Array{ComplexF64}(undef, n_itr)
    population_rhs_components_hist = Array{ComplexF64}(undef, n_sys, n_itr, 6)
    population_trace_rhs_components_hist = Array{ComplexF64}(undef, n_itr, 6)
    population_rhs_component_residual_hist = Array{ComplexF64}(undef, n_sys, n_itr)
    population_rhs_component_labels = (:local_L0, :local_L1, :LL, :LR, :RL, :RR)
    need_branch_memory_storage = return_branch_memory_audit || return_population_rhs_audit || return_internal_c
    branch_memory_audit = need_branch_memory_storage ? Array{ComplexF64}(undef, n_sys, n_sys, n_itr, 4) : Array{ComplexF64}(undef, 0, 0, 0, 0)
    kernel_rowsum_audit = return_kernel_rowsum_audit ? Array{ComplexF64}(undef, n_sys, n_sys, n_itr, 4) : Array{ComplexF64}(undef, 0, 0, 0, 0)
    kernel_output_audit = return_kernel_output_audit ? Array{ComplexF64}(undef, n_sys, n_sys, n_sys, n_itr, 4) : Array{ComplexF64}(undef, 0, 0, 0, 0, 0)
    population_difference_hist = Array{ComplexF64}(undef, n_itr)
    population_difference_rhs_hist = Array{ComplexF64}(undef, n_itr)
    population_difference_rhs_components_hist = Array{ComplexF64}(undef, n_itr, 6)
    population_rate_1_to_2_hist = Array{ComplexF64}(undef, n_itr)
    population_rate_2_to_1_hist = Array{ComplexF64}(undef, n_itr)
    population_difference_damping_hist = Array{ComplexF64}(undef, n_itr)
    fill!(c_hist, const_ZERO_C)
    fill!(cprime_hist, const_ZERO_C)
    fill!(population_rhs_hist, const_ZERO_C)
    fill!(population_trace_rhs_hist, const_ZERO_C)
    fill!(population_rhs_components_hist, const_ZERO_C)
    fill!(population_trace_rhs_components_hist, const_ZERO_C)
    fill!(population_rhs_component_residual_hist, const_ZERO_C)
    need_branch_memory_storage && fill!(branch_memory_audit, const_ZERO_C)
    return_kernel_rowsum_audit && fill!(kernel_rowsum_audit, const_ZERO_C)
    return_kernel_output_audit && fill!(kernel_output_audit, const_ZERO_C)
    fill!(population_difference_hist, const_ZERO_C)
    fill!(population_difference_rhs_hist, const_ZERO_C)
    fill!(population_difference_rhs_components_hist, const_ZERO_C)
    fill!(population_rate_1_to_2_hist, const_ZERO_C)
    fill!(population_rate_2_to_1_hist, const_ZERO_C)
    fill!(population_difference_damping_hist, const_ZERO_C)

    @inbounds for β in 1:n_sys, α in 1:n_sys
        c_hist[α, β, 1] = σ[α, β, 1]
    end
    use_population_closure && enforce_population_closure!(@view(c_hist[:, :, 1]))
    enforce_c_hermitian && enforce_hermiticity!(@view(c_hist[:, :, 1]))
    readout_sigma_full!(@view(σ[:, :, 1]), c_hist, 1)

    c_stage = Matrix{ComplexF64}(undef, n_sys, n_sys)
    k1      = Matrix{ComplexF64}(undef, n_sys, n_sys)
    k2      = Matrix{ComplexF64}(undef, n_sys, n_sys)
    k3      = Matrix{ComplexF64}(undef, n_sys, n_sys)
    k4      = Matrix{ComplexF64}(undef, n_sys, n_sys)

    @inline function store_population_difference_audit!(itr::Int, c_t::AbstractMatrix)
        if n_sys != 2
            return nothing
        end

        population_difference_hist[itr] = c_t[1,1] - c_t[2,2]
        population_difference_rhs_hist[itr] = population_rhs_hist[1,itr] - population_rhs_hist[2,itr]

        for component in 1:6
            population_difference_rhs_components_hist[itr,component] =
                population_rhs_components_hist[1,itr,component] -
                population_rhs_components_hist[2,itr,component]
        end

        if return_kernel_output_audit
            k12 = const_ZERO_C  # input population 1 -> output population 2
            k21 = const_ZERO_C  # input population 2 -> output population 1
            for branch in 1:4
                k12 += kernel_output_audit[2,1,1,itr,branch]
                k21 += kernel_output_audit[1,2,2,itr,branch]
            end
            population_rate_1_to_2_hist[itr] = k12
            population_rate_2_to_1_hist[itr] = k21
            population_difference_damping_hist[itr] = k12 + k21

            if real(k12 + k21) < difference_damping_warn_tol
                @printf(
                    stderr,
                    "ver31 anti-damping audit at itr=%d: k12=%+.8e%+.8ei  k21=%+.8e%+.8ei  Re(k12+k21)=%.8e\n",
                    itr,
                    real(k12), imag(k12),
                    real(k21), imag(k21),
                    real(k12+k21),
                )
            end

            max_rate_imag = max(abs(imag(k12)), abs(imag(k21)))
            if max_rate_imag > population_rate_imag_tol
                @printf(
                    stderr,
                    "ver31 population-rate Hermiticity audit at itr=%d: Im(k12)=%.8e  Im(k21)=%.8e  max=%.8e\n",
                    itr,
                    imag(k12),
                    imag(k21),
                    max_rate_imag,
                )
            end
        end
        return nothing
    end

    @inline function store_population_rhs!(itr::Int, rhs_full::AbstractMatrix, c_t::AbstractMatrix)
        sum_dp = const_ZERO_C
        trace_components = ntuple(_ -> const_ZERO_C, 6)
        trace_components_mut = collect(trace_components)

        @inbounds for state in 1:n_sys
            dp = rhs_full[state, state]
            population_rhs_hist[state, itr] = dp
            sum_dp += dp

            local_L0 = include_c_L0P_local ? coef_M_P_0(q_full(itr), state, state) * c_local(c_t, state, state) : const_ZERO_C
            local_L1 = const_ZERO_C
            if include_c_PL1P_local
                for μ in 1:n_sys
                    μ == state && continue
                    if is_secular_pair(state, state, μ, state)
                        local_L1 += coef_M_P_L(q_full(itr), state, state, μ) * c_local(c_t, μ, state)
                    end
                end
                for ν in 1:n_sys
                    ν == state && continue
                    if is_secular_pair(state, state, state, ν)
                        local_L1 += coef_M_P_R(q_full(itr), state, state, ν) * c_local(c_t, state, ν)
                    end
                end
            end

            LL = need_branch_memory_storage ? branch_memory_audit[state, state, itr, 1] : const_ZERO_C
            LR = need_branch_memory_storage ? branch_memory_audit[state, state, itr, 2] : const_ZERO_C
            RL = need_branch_memory_storage ? branch_memory_audit[state, state, itr, 3] : const_ZERO_C
            RR = need_branch_memory_storage ? branch_memory_audit[state, state, itr, 4] : const_ZERO_C
            components = (local_L0, local_L1, LL, LR, RL, RR)

            component_total = const_ZERO_C
            for component in 1:6
                value = components[component]
                population_rhs_components_hist[state, itr, component] = value
                trace_components_mut[component] += value
                component_total += value
            end
            population_rhs_component_residual_hist[state, itr] = dp - component_total
        end

        population_trace_rhs_hist[itr] = sum_dp
        for component in 1:6
            population_trace_rhs_components_hist[itr, component] = trace_components_mut[component]
        end

        if return_population_difference_audit
            store_population_difference_audit!(itr, c_t)
        end

        if audit_initial_population_direction && itr <= max(1, initial_direction_steps)
            @printf(stderr, "ver31 population RHS itr=%d", itr)
            @inbounds for state in 1:n_sys
                dp = population_rhs_hist[state, itr]
                @printf(stderr, "  dp%d=%+.8e%+.8ei", state, real(dp), imag(dp))
            end
            @printf(stderr, "  sum_dp=%+.8e%+.8ei", real(sum_dp), imag(sum_dp))
            for component in 1:6
                value = trace_components_mut[component]
                @printf(stderr, "  sum_%s=%+.4e%+.4ei", String(population_rhs_component_labels[component]), real(value), imag(value))
            end
            @printf(stderr, "\n")
        end
        return sum_dp
    end

    start_loop = max(1, start_itr)
    @inbounds for curr_itr in start_loop:(n_itr - 1)
        if verbose && (curr_itr == start_loop || curr_itr == n_itr - 1 || ((curr_itr - start_loop) % verbose_every_f == 0))
            @printf(
                stderr,
                "Current iteration: %6d / %6d  method=%s  threads=%d  use_threads=%s  secular=%s  state=c  sigma=population-cyclic-consistent-ver31-diagnostic\n",
                curr_itr, n_itr, String(method_sym), Threads.nthreads(), string(use_threads), string(use_secular),
            )
        end

        c_t = @view c_hist[:, :, curr_itr]
        c_next = @view c_hist[:, :, curr_itr + 1]

        if method_sym == :euler
            calc_c_rhs!(k1, curr_itr, q_full(curr_itr), c_t, c_hist)
            cprime_hist[:, :, curr_itr] .= k1
            compute_memory_kernel_rowsum_audit!(kernel_rowsum_audit, kernel_output_audit, curr_itr, q_full(curr_itr))
            store_population_rhs!(curr_itr, k1, c_t)
            @. c_next = c_t + Δt * k1

        elseif method_sym == :rk2
            calc_c_rhs!(k1, curr_itr, q_full(curr_itr), c_t, c_hist)
            cprime_hist[:, :, curr_itr] .= k1
            compute_memory_kernel_rowsum_audit!(kernel_rowsum_audit, kernel_output_audit, curr_itr, q_full(curr_itr))
            store_population_rhs!(curr_itr, k1, c_t)
            @. c_stage = c_t + 0.5 * Δt * k1
            use_population_closure && enforce_population_closure!(c_stage)
            enforce_c_hermitian && enforce_hermiticity!(c_stage)
            calc_c_rhs!(k2, curr_itr, q_half_after(curr_itr), c_stage, c_hist)
            @. c_next = c_t + Δt * k2

        elseif method_sym == :rk4
            calc_c_rhs!(k1, curr_itr, q_full(curr_itr), c_t, c_hist)
            cprime_hist[:, :, curr_itr] .= k1
            compute_memory_kernel_rowsum_audit!(kernel_rowsum_audit, kernel_output_audit, curr_itr, q_full(curr_itr))

            store_population_rhs!(curr_itr, k1, c_t)

            @. c_stage = c_t + 0.5 * Δt * k1
            use_population_closure && enforce_population_closure!(c_stage)
            enforce_c_hermitian && enforce_hermiticity!(c_stage)
            calc_c_rhs!(k2, curr_itr, q_half_after(curr_itr), c_stage, c_hist)

            @. c_stage = c_t + 0.5 * Δt * k2
            use_population_closure && enforce_population_closure!(c_stage)
            enforce_c_hermitian && enforce_hermiticity!(c_stage)
            calc_c_rhs!(k3, curr_itr, q_half_after(curr_itr), c_stage, c_hist)

            @. c_stage = c_t + Δt * k3
            use_population_closure && enforce_population_closure!(c_stage)
            enforce_c_hermitian && enforce_hermiticity!(c_stage)
            calc_c_rhs!(k4, curr_itr, q_full(curr_itr + 1), c_stage, c_hist)

            @. c_next = c_t + (Δt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        end

        use_population_closure && enforce_population_closure!(c_next)
        enforce_c_hermitian && enforce_hermiticity!(c_next)

        # Store physical sigma only after c has been advanced.
        readout_sigma_full!(@view(σ[:, :, curr_itr + 1]), c_hist, curr_itr + 1)
        context.curr_itr = UInt64(curr_itr + 1)
    end

    # The integration loop stores k1 only through n_itr-1.  Evaluate the final
    # full-grid RHS as well so dp1, dp2, ... are available at every saved time.
    final_itr = Int(context.curr_itr)
    if 1 <= final_itr <= n_itr
        c_final = @view c_hist[:, :, final_itr]
        calc_c_rhs!(k1, final_itr, q_full(final_itr), c_final, c_hist)
        cprime_hist[:, :, final_itr] .= k1
        store_population_rhs!(final_itr, k1, c_final)
    end

    # --------------------------------------------------------------
    # ver31 finite-time stationary / zero-frequency diagnostics.
    #
    # At the final saved time t_f, build
    #
    #   L_eff(t_f; z=0) = M(t_f) + ∫_0^{t_f} ds K(t_f,s)
    #
    # by applying the existing linear RHS to every constant-history basis
    # matrix.  This uses exactly the implemented generator; no Markovian
    # limit, rate fitting, or energy shift is introduced.
    # --------------------------------------------------------------
    stationary_diagnostics = nothing

    if print_stationary_diagnostics || return_stationary_diagnostics
        final_index = Int(context.curr_itr)
        final_q = q_full(final_index)
        tail_start = max(1, final_index - stationary_tail_window + 1)
        tail_range = tail_start:final_index

        # Last-point and tail-averaged populations.
        p_final = ComplexF64[
            c_hist[state,state,final_index] for state in 1:n_sys
        ]
        p_tail_mean = ComplexF64[
            sum(c_hist[state,state,k] for k in tail_range) / length(tail_range)
            for state in 1:n_sys
        ]

        plateau_ratio_final = n_sys == 2 && abs(p_final[2]) > eps(Float64) ?
            p_final[1] / p_final[2] :
            ComplexF64(NaN,NaN)

        plateau_ratio_tail = n_sys == 2 && abs(p_tail_mean[2]) > eps(Float64) ?
            p_tail_mean[1] / p_tail_mean[2] :
            ComplexF64(NaN,NaN)

        # Gibbs ratios are comparisons only.  Dynamics continues to use ϵ_exci.
        beta_value = beta_diagnostic === nothing ?
            nothing :
            Float64(beta_diagnostic)

        bare_gibbs_ratio = (
            n_sys == 2 && beta_value !== nothing
        ) ? exp(-beta_value * (real(ϵ[1]) - real(ϵ[2]))) : nothing

        shifted_gibbs_ratio = nothing
        if (
            n_sys == 2
            && beta_value !== nothing
            && shifted_energies_diagnostic !== nothing
            && length(shifted_energies_diagnostic) >= 2
        )
            shifted_gibbs_ratio = exp(
                -beta_value * (
                    real(shifted_energies_diagnostic[1])
                    - real(shifted_energies_diagnostic[2])
                )
            )
        end

        # Build the finite-time zero-frequency generator on the complete
        # n_sys^2 operator basis using constant histories.
        operator_dimension = n_sys * n_sys
        zero_frequency_generator = zeros(
            ComplexF64,
            operator_dimension,
            operator_dimension,
        )
        basis_state = zeros(ComplexF64,n_sys,n_sys)
        basis_rhs = zeros(ComplexF64,n_sys,n_sys)
        basis_history = zeros(ComplexF64,n_sys,n_sys,n_itr)

        # calc_c_rhs! writes branch_memory_audit at final_index when that
        # storage is active.  Preserve the physical-trajectory entry.
        branch_final_snapshot = need_branch_memory_storage ?
            copy(@view(branch_memory_audit[:,:,final_index,:])) :
            nothing

        @inline operator_index(a::Int,b::Int) = a + (b-1) * n_sys

        for b_in in 1:n_sys, a_in in 1:n_sys
            fill!(basis_state,const_ZERO_C)
            fill!(basis_history,const_ZERO_C)
            basis_state[a_in,b_in] = 1.0 + 0.0im

            for history_index in 1:final_index
                basis_history[a_in,b_in,history_index] = 1.0 + 0.0im
            end

            calc_c_rhs!(
                basis_rhs,
                final_index,
                final_q,
                basis_state,
                basis_history,
            )

            column = operator_index(a_in,b_in)
            for b_out in 1:n_sys, a_out in 1:n_sys
                row = operator_index(a_out,b_out)
                zero_frequency_generator[row,column] =
                    basis_rhs[a_out,b_out]
            end
        end

        if need_branch_memory_storage
            @view(branch_memory_audit[:,:,final_index,:]) .=
                branch_final_snapshot
        end

        eig = eigen(zero_frequency_generator)
        zero_mode_index = argmin(abs.(eig.values))
        zero_mode_eigenvalue = eig.values[zero_mode_index]
        zero_mode_vector = copy(eig.vectors[:,zero_mode_index])
        zero_mode_matrix = reshape(zero_mode_vector,n_sys,n_sys)
        zero_mode_trace = sum(
            zero_mode_matrix[state,state] for state in 1:n_sys
        )

        if abs(zero_mode_trace) > eps(Float64)
            zero_mode_matrix ./= zero_mode_trace
        end

        zero_mode_populations = ComplexF64[
            zero_mode_matrix[state,state] for state in 1:n_sys
        ]
        zero_mode_ratio = (
            n_sys == 2
            && abs(zero_mode_populations[2]) > eps(Float64)
        ) ? zero_mode_populations[1] / zero_mode_populations[2] :
            ComplexF64(NaN,NaN)

        zero_mode_hermiticity_residual = 0.0
        for b in 1:n_sys, a in 1:n_sys
            zero_mode_hermiticity_residual = max(
                zero_mode_hermiticity_residual,
                abs(
                    zero_mode_matrix[a,b]
                    - conj(zero_mode_matrix[b,a])
                ),
            )
        end

        # Integrated population-only transfer coefficients are off-diagonal
        # population entries of the same finite-time z=0 generator.
        kernel_rate_1_to_2 = n_sys == 2 ?
            zero_frequency_generator[
                operator_index(2,2),
                operator_index(1,1),
            ] :
            ComplexF64(NaN,NaN)

        kernel_rate_2_to_1 = n_sys == 2 ?
            zero_frequency_generator[
                operator_index(1,1),
                operator_index(2,2),
            ] :
            ComplexF64(NaN,NaN)

        kernel_ratio = (
            n_sys == 2
            && abs(kernel_rate_1_to_2) > eps(Float64)
        ) ? kernel_rate_2_to_1 / kernel_rate_1_to_2 :
            ComplexF64(NaN,NaN)

        max_tail_population_rhs = maximum(
            abs(population_rhs_hist[state,k])
            for state in 1:n_sys
            for k in tail_range
        )
        final_population_rhs = ComplexF64[
            population_rhs_hist[state,final_index] for state in 1:n_sys
        ]

        final_memory_population_rhs = ComplexF64[
            sum(
                population_rhs_components_hist[
                    state,
                    final_index,
                    component,
                ]
                for component in 3:6
            )
            for state in 1:n_sys
        ]

        internal_coherence_12 = n_sys == 2 ?
            c_hist[1,2,final_index] :
            ComplexF64(NaN,NaN)

        physical_coherence_12 = n_sys == 2 ?
            σ[1,2,final_index] :
            ComplexF64(NaN,NaN)

        stationary_diagnostics = (
            final_index = final_index,
            final_time = (final_index-1) * Δt,
            tail_start_index = tail_start,
            tail_window = length(tail_range),
            population_final = p_final,
            population_tail_mean = p_tail_mean,
            plateau_ratio_final = plateau_ratio_final,
            plateau_ratio_tail = plateau_ratio_tail,
            beta = beta_value,
            bare_gibbs_ratio = bare_gibbs_ratio,
            shifted_energies = shifted_energies_diagnostic,
            shifted_gibbs_ratio = shifted_gibbs_ratio,
            kernel_rate_1_to_2 = kernel_rate_1_to_2,
            kernel_rate_2_to_1 = kernel_rate_2_to_1,
            kernel_ratio = kernel_ratio,
            zero_frequency_generator = zero_frequency_generator,
            zero_mode_eigenvalue = zero_mode_eigenvalue,
            zero_mode_matrix = zero_mode_matrix,
            zero_mode_populations = zero_mode_populations,
            zero_mode_ratio = zero_mode_ratio,
            zero_mode_hermiticity_residual =
                zero_mode_hermiticity_residual,
            final_population_rhs = final_population_rhs,
            max_tail_population_rhs = max_tail_population_rhs,
            final_memory_population_rhs = final_memory_population_rhs,
            internal_coherence_12 = internal_coherence_12,
            physical_coherence_12 = physical_coherence_12,
        )

        if print_stationary_diagnostics
            @printf(
                stderr,
                "\n=== ver31 stationary / zero-frequency diagnostics ===\n",
            )
            @printf(
                stderr,
                "final_index=%d  final_time=%.8e  tail=[%d,%d]\n",
                final_index,
                (final_index-1) * Δt,
                tail_start,
                final_index,
            )

            for state in 1:n_sys
                @printf(
                    stderr,
                    "p%d(final)=%+.10e%+.10ei  p%d(tail_mean)=%+.10e%+.10ei\n",
                    state,
                    real(p_final[state]),
                    imag(p_final[state]),
                    state,
                    real(p_tail_mean[state]),
                    imag(p_tail_mean[state]),
                )
            end

            if n_sys == 2
                @printf(
                    stderr,
                    "R_plateau_final=p1/p2=%+.10e%+.10ei\n",
                    real(plateau_ratio_final),
                    imag(plateau_ratio_final),
                )
                @printf(
                    stderr,
                    "R_plateau_tail =p1/p2=%+.10e%+.10ei\n",
                    real(plateau_ratio_tail),
                    imag(plateau_ratio_tail),
                )
            end

            if beta_value === nothing
                @printf(
                    stderr,
                    "beta: unavailable; pass stationary_beta=... to print Gibbs ratios\n",
                )
            else
                @printf(stderr,"beta=%.10e\n",beta_value)
                @printf(
                    stderr,
                    "R_bare=exp[-beta*(epsilon1-epsilon2)]=%.10e\n",
                    bare_gibbs_ratio,
                )

                if shifted_gibbs_ratio === nothing
                    @printf(
                        stderr,
                        "R_shifted: unavailable; pass stationary_shifted_energies or provide context.ϵ_exci_0\n",
                    )
                else
                    @printf(
                        stderr,
                        "R_shifted=exp[-beta*(epsilon1_0-epsilon2_0)]=%.10e\n",
                        shifted_gibbs_ratio,
                    )
                end
            end

            if n_sys == 2
                @printf(
                    stderr,
                    "K12(z=0,t_f)=%+.10e%+.10ei\n",
                    real(kernel_rate_1_to_2),
                    imag(kernel_rate_1_to_2),
                )
                @printf(
                    stderr,
                    "K21(z=0,t_f)=%+.10e%+.10ei\n",
                    real(kernel_rate_2_to_1),
                    imag(kernel_rate_2_to_1),
                )
                @printf(
                    stderr,
                    "R_kernel=K21/K12=%+.10e%+.10ei\n",
                    real(kernel_ratio),
                    imag(kernel_ratio),
                )
            end

            @printf(
                stderr,
                "zero_mode_eigenvalue=%+.10e%+.10ei\n",
                real(zero_mode_eigenvalue),
                imag(zero_mode_eigenvalue),
            )
            for state in 1:n_sys
                @printf(
                    stderr,
                    "zero_mode_p%d=%+.10e%+.10ei\n",
                    state,
                    real(zero_mode_populations[state]),
                    imag(zero_mode_populations[state]),
                )
            end
            if n_sys == 2
                @printf(
                    stderr,
                    "R_zero_mode=p1/p2=%+.10e%+.10ei\n",
                    real(zero_mode_ratio),
                    imag(zero_mode_ratio),
                )
            end
            @printf(
                stderr,
                "zero_mode_hermiticity_residual=%.10e\n",
                zero_mode_hermiticity_residual,
            )

            for state in 1:n_sys
                @printf(
                    stderr,
                    "dp%d(final)=%+.10e%+.10ei  memory_dp%d(final)=%+.10e%+.10ei\n",
                    state,
                    real(final_population_rhs[state]),
                    imag(final_population_rhs[state]),
                    state,
                    real(final_memory_population_rhs[state]),
                    imag(final_memory_population_rhs[state]),
                )
            end

            @printf(
                stderr,
                "max_tail_abs_dp=%.10e\n",
                max_tail_population_rhs,
            )

            if n_sys == 2
                @printf(
                    stderr,
                    "c_internal_12(final)=%+.10e%+.10ei  abs=%.10e\n",
                    real(internal_coherence_12),
                    imag(internal_coherence_12),
                    abs(internal_coherence_12),
                )
                @printf(
                    stderr,
                    "sigma_physical_12(final)=%+.10e%+.10ei  abs=%.10e\n",
                    real(physical_coherence_12),
                    imag(physical_coherence_12),
                    abs(physical_coherence_12),
                )
            end

            @printf(
                stderr,
                "====================================================\n\n",
            )
        end
    end

    if return_internal_c || return_branch_memory_audit || return_population_rhs_audit || return_kernel_rowsum_audit || return_kernel_output_audit || return_population_difference_audit || return_stationary_diagnostics
        fields = (
            σ = @view(σ[:, :, Int(context.curr_itr)]),
            c = c_hist,
            c′ = cprime_hist,
            population_rhs = return_population_rhs_audit || return_internal_c ? population_rhs_hist : nothing,
            population_trace_rhs = return_population_rhs_audit || return_internal_c ? population_trace_rhs_hist : nothing,
            population_rhs_components = return_population_rhs_audit || return_internal_c ? population_rhs_components_hist : nothing,
            population_trace_rhs_components = return_population_rhs_audit || return_internal_c ? population_trace_rhs_components_hist : nothing,
            population_rhs_component_labels = return_population_rhs_audit || return_internal_c ? population_rhs_component_labels : nothing,
            population_rhs_component_residual = return_population_rhs_audit || return_internal_c ? population_rhs_component_residual_hist : nothing,
            dp1 = (return_population_rhs_audit || return_internal_c) && n_sys >= 1 ? @view(population_rhs_hist[1, :]) : nothing,
            dp2 = (return_population_rhs_audit || return_internal_c) && n_sys >= 2 ? @view(population_rhs_hist[2, :]) : nothing,
            sum_dp = return_population_rhs_audit || return_internal_c ? population_trace_rhs_hist : nothing,
            branch_memory = return_branch_memory_audit ? branch_memory_audit : nothing,
            kernel_rowsum = return_kernel_rowsum_audit ? kernel_rowsum_audit : nothing,
            kernel_output = return_kernel_output_audit ? kernel_output_audit : nothing,
            population_difference = return_population_difference_audit && n_sys == 2 ? population_difference_hist : nothing,
            population_difference_rhs = return_population_difference_audit && n_sys == 2 ? population_difference_rhs_hist : nothing,
            population_difference_rhs_components = return_population_difference_audit && n_sys == 2 ? population_difference_rhs_components_hist : nothing,
            population_rate_1_to_2 = return_population_difference_audit && n_sys == 2 ? population_rate_1_to_2_hist : nothing,
            population_rate_2_to_1 = return_population_difference_audit && n_sys == 2 ? population_rate_2_to_1_hist : nothing,
            population_difference_damping = return_population_difference_audit && n_sys == 2 ? population_difference_damping_hist : nothing,
            stationary_diagnostics = return_stationary_diagnostics ? stationary_diagnostics : nothing,
        )
        return fields
    end
    return @view(σ[:, :, Int(context.curr_itr)])
end


















# function calc__σ_σ′_secular_core!(
#     context::RmrtContext;
#     hbar::Real = 1.0,
#     use_population_closure::Bool = false,
    
#     # Normalized transported P_T local L0/line-shape block.
#     # For normalized tau^T this generally remains in the c-equation as the
#     # scalar local line-shape / electronic phase generator.
#     include_PL0P_local::Bool = true,

#     # P_T L1 P_T local left/right mixing block.
#     include_PL1P_local::Bool = true,

#     # P_T L1 Q_T G0 Q_T L1 P_T memory block for internal c.
#     # This is NOT the old widened (L0+L1) block; both L1 insertions keep
#     # their off-diagonal constraints.
#     include_P_L0plusL1_QG0QL1P::Bool = true,

#     # Physical readout mode for context.σ.  When true, after each c-step store
#     # σ(t)=c(t)+Tr_B{Q_T(t)rho(t)} using the G0 formal solution for Q_T rho.
#     # When false, store the projected-only diagnostic σ≈c.
#     include_QT_G0_readout::Bool = true,

#     # P L0 Q G1 Q L1 P correction. This is a nested s-tau memory integral,
#     # so it is opt-in in this optimized version.
#     include_PL0QG1QL1P::Bool = false,

#     # Markovianize only the relative-time g′/g″ factors that belong to the
#     # inner G1 correction in P L0 Q G1 Q L1 P.  In this mode,
#     #     g′_{abcd}(∞) ≈ -im * Λ_{abcd},      g″_{abcd}(∞) ≈ 0,
#     # while the outer s/t Gaussian dressing factors are still evaluated from g.
#     markovianize_PL0QG1QL1P_G1::Bool = false,

#     # Stronger Option-B approximation for P L0 Q G1 Q L1 P.
#     # When true, the inner tau integral is collapsed by a midpoint rule:
#     #     ∫_s^t dτ K(t,τ,s) ≈ (t-s) K(t,nearest_q((t+s)/2),s).
#     # This keeps the outer s-memory integral, but removes the O(N_t) inner tau loop.
#     # It also implies markovianize_PL0QG1QL1P_G1=true for the G1-relative
#     # g′/g″ factors.
#     collapse_tau_PL0QG1QL1P_G1::Bool = false,

#     use_secular::Bool = false,
#     secular_tol::Float64 = 1.0e-10,
#     method::Union{Symbol,String} = :rk4,
#     use_threads::Bool = true,

#     # Half-grid handling.  This version uses doubled integer q-grid indexing:
#     #   full grid i      -> q = 2i - 1
#     #   half grid i+1/2  -> q = 2i
#     # IMPORTANT: for Patternized_g containers, we must NOT materialize a dense
#     # half-grid by looping over all (a,b,c,d), because unsupported patterns may
#     # be intentionally absent.  RK2/RK4 and collapsed-tau G1 therefore require
#     # precomputed patternized half-grid containers on context.
#     auto_prepare_half_shifted_grid::Bool = true,
#     recompute_half_shifted_grid::Bool = false,

#     enforce_hermitian::Bool = true,
#     verbose::Bool = true,
#     verbose_every::Int = 1,
# )
#     start_itr = Int(context.curr_itr)

#     n_sys = context.system.n_sys
#     n_itr = context.simulation_details.num_of_iteration
#     Δt    = context.simulation_details.Δt
#     ϵ     = context.ϵ_exci

#     # context.σ stores the physical reduced density matrix readout.
#     # The propagated variable for the transported PT scheme is the internal
#     # coordinate c.  If context.c/context.c′ exist, use them as persistent
#     # internal-coordinate buffers.  Otherwise, fall back to a local copy of
#     # context.σ as the initial c-history and use context.σ′ as the derivative
#     # workspace for c′.
#     σ  = context.σ
#     σ′ = context.σ′
#     has_c_buffer = hasproperty(context, :c)
#     if !has_c_buffer && start_itr > 1
#         error("PT c-readout mode requires context.c when restarting from curr_itr>1; context.σ is physical readout and cannot reconstruct the internal c history.")
#     end
#     c  = has_c_buffer ? getfield(context, :c) : copy(σ)
#     c′ = hasproperty(context, Symbol("c′")) ? getfield(context, Symbol("c′")) : σ′
#     g  = context.g
#     g′ = context.g′
#     g″ = context.g″

#     effective_markovianize_PL0QG1QL1P_G1 =
#         markovianize_PL0QG1QL1P_G1 || collapse_tau_PL0QG1QL1P_G1

#     if include_PL0QG1QL1P
#         error("This PT c-update core is G0-only. Disable include_PL0QG1QL1P, or derive a separate transported P_T G1 kernel.")
#     end

#     if include_PL0QG1QL1P && effective_markovianize_PL0QG1QL1P_G1 && !hasproperty(context, :Λ)
#         error("markovianize/collapse PL0QG1QL1P_G1 requires context.Λ, with Λ[a,b,c,d] ≈ i*g′_{abcd}(∞).")
#     end
#     Λ = (include_PL0QG1QL1P && effective_markovianize_PL0QG1QL1P_G1) ? getfield(context, :Λ) : nothing

#     hbar_f = Float64(hbar)
#     hbar_f > 0.0 || error("hbar must be positive")
#     hbar2 = hbar_f^2
#     hbar4 = hbar_f^4
#     hbar6 = hbar_f^6

#     method_sym = Symbol(lowercase(String(method)))
#     method_sym in (:euler, :rk2, :rk4) || error(
#         "Unsupported integration method: $(method). Use :euler, :rk2, or :rk4."
#     )

#     # Avoid pathological logging settings; printing to stderr can dominate
#     # short runs, so the loop below prints only every verbose_every steps.
#     verbose_every_f = max(1, Int(verbose_every))

#     # Half-grid keyword arguments are kept for API compatibility.  This version
#     # does not materialize dense fallback half-grids, because g/g′/g″ are often
#     # Patternized containers with intentionally unsupported index patterns.
#     _ = auto_prepare_half_shifted_grid
#     _ = recompute_half_shifted_grid

#     start_itr < n_itr || return @view σ[:, :, n_itr]

#     const_ZERO_C = 0.0 + 0.0im
#     const_Q_MAX = 2 * n_itr - 1

#     @inline function ω(a::Int, b::Int)
#         @inbounds return ϵ[a] - ϵ[b]
#     end
#     @inline function Ω(a::Int, b::Int)
#         @inbounds return (ϵ[a] - ϵ[b]) / hbar_f
#     end

#     # ------------------------------------------------------------------
#     # Doubled integer q-grid utilities.
#     #   full grid i      -> q = 2i - 1
#     #   half grid i+1/2  -> q = 2i
#     # Relative times are also q-indices:
#     #   qΔ = q_t - q_s + 1
#     # so full/half relative times close without Float64 coordinates.
#     # ------------------------------------------------------------------

#     @inline q_full(i::Int) = 2 * i - 1
#     @inline q_half_after(i::Int) = 2 * i
#     @inline is_full_q(q::Int) = isodd(q)
#     @inline full_index_from_q(q::Int) = (q + 1) >>> 1
#     @inline half_index_from_q(q::Int) = q >>> 1

#     @inline function check_q(q::Int)
#         (1 <= q <= const_Q_MAX) || error("q-grid index out of range: $(q), allowed [1, $(const_Q_MAX)]")
#         return q
#     end

#     function _first_existing_property(obj, names::Tuple)
#         for name in names
#             if hasproperty(obj, name)
#                 return getfield(obj, name)
#             end
#         end
#         return nothing
#     end

#     # Try to use the user's precomputed half-grid containers if they exist.
#     # The symbol list intentionally accepts several common naming conventions.
#     g_half = _first_existing_property(context, (
#         :g_half,
#         :g_half_shifted,
#         :g_shifted_half,
#         :g_mid,
#         :g_midpoint,
#         Symbol("g__half"),
#         Symbol("g_half_grid"),
#         Symbol("g_half_shifted_grid"),
#     ))
#     gp_half = _first_existing_property(context, (
#         Symbol("g′_half"),
#         Symbol("g′_half_shifted"),
#         Symbol("g′_shifted_half"),
#         Symbol("g′_mid"),
#         Symbol("g′_midpoint"),
#         Symbol("gp_half"),
#         Symbol("gp_half_shifted"),
#         Symbol("g_prime_half"),
#         Symbol("g_prime_half_shifted"),
#     ))
#     gpp_half = _first_existing_property(context, (
#         Symbol("g″_half"),
#         Symbol("g″_half_shifted"),
#         Symbol("g″_shifted_half"),
#         Symbol("g″_mid"),
#         Symbol("g″_midpoint"),
#         Symbol("gpp_half"),
#         Symbol("gpp_half_shifted"),
#         Symbol("g_doubleprime_half"),
#         Symbol("g_doubleprime_half_shifted"),
#     ))

#     needs_half_grid = (method_sym in (:rk2, :rk4)) || (include_PL0QG1QL1P && collapse_tau_PL0QG1QL1P_G1)

#     if needs_half_grid && (g_half === nothing || gp_half === nothing || gpp_half === nothing)
#         missing = String[]
#         g_half === nothing && push!(missing, "g half-grid")
#         gp_half === nothing && push!(missing, "g′ half-grid")
#         gpp_half === nothing && push!(missing, "g″ half-grid")
#         error(
#             "q-grid mode requires precomputed patternized half-grid containers for " *
#             join(missing, ", ") *
#             ". Do not dense-materialize Patternized_g; prepare and store half-grid " *
#             "containers on context, or use method=:euler with collapse_tau_PL0QG1QL1P_G1=false."
#         )
#     end

#     @inline function _missing_half_grid_error(name::String)
#         error(
#             "Attempted to access " * name * " at a half-grid q index, but the corresponding " *
#             "patternized half-grid container was not found on context."
#         )
#     end

#     @inline function G(q::Int, a::Int, b::Int, c::Int, d::Int)
#         check_q(q)
#         if isodd(q)
#             idx = full_index_from_q(q)
#             @inbounds return g[idx, a, b, c, d]
#         else
#             g_half === nothing && _missing_half_grid_error("g")
#             idx = half_index_from_q(q)
#             @inbounds return g_half[idx, a, b, c, d]
#         end
#     end

#     @inline function Gp(q::Int, a::Int, b::Int, c::Int, d::Int)
#         check_q(q)
#         if isodd(q)
#             idx = full_index_from_q(q)
#             @inbounds return g′[idx, a, b, c, d]
#         else
#             gp_half === nothing && _missing_half_grid_error("g′")
#             idx = half_index_from_q(q)
#             @inbounds return gp_half[idx, a, b, c, d]
#         end
#     end

#     @inline function Gpp(q::Int, a::Int, b::Int, c::Int, d::Int)
#         check_q(q)
#         if isodd(q)
#             idx = full_index_from_q(q)
#             @inbounds return g″[idx, a, b, c, d]
#         else
#             gpp_half === nothing && _missing_half_grid_error("g″")
#             idx = half_index_from_q(q)
#             @inbounds return gpp_half[idx, a, b, c, d]
#         end
#     end

#     # Pure Hamiltonian phase cache over doubled relative q-grid.
#     phase_cache = Array{ComplexF64}(undef, const_Q_MAX, n_sys, n_sys)
#     @inbounds for qΔ in 1:const_Q_MAX, a in 1:n_sys, b in 1:n_sys
#         phase_cache[qΔ, a, b] = exp(-1.0im * (ϵ[a] - ϵ[b]) * (0.5 * (Float64(qΔ) - 1.0) * Δt) / hbar_f)
#     end

#     @inline function n_outer_nodes(curr_itr::Int, q_t::Int)
#         q_now = q_full(curr_itr)
#         return q_t > q_now ? curr_itr + 1 : curr_itr
#     end

#     @inline function outer_node_coord(node_idx::Int, curr_itr::Int, q_t::Int)
#         return node_idx <= curr_itr ? q_full(node_idx) : q_t
#     end

#     @inline function ∫weight_to_t(node_idx::Int, curr_itr::Int, q_t::Int)
#         n_nodes = n_outer_nodes(curr_itr, q_t)
#         n_nodes <= 1 && return 0.0

#         q = outer_node_coord(node_idx, curr_itr, q_t)
#         if node_idx == 1
#             q_next = outer_node_coord(node_idx + 1, curr_itr, q_t)
#             return 0.25 * Δt * Float64(q_next - q)
#         elseif node_idx == n_nodes
#             q_prev = outer_node_coord(node_idx - 1, curr_itr, q_t)
#             return 0.25 * Δt * Float64(q - q_prev)
#         else
#             q_prev = outer_node_coord(node_idx - 1, curr_itr, q_t)
#             q_next = outer_node_coord(node_idx + 1, curr_itr, q_t)
#             return 0.25 * Δt * Float64(q_next - q_prev)
#         end
#     end

#     @inline function n_inner_nodes(q_s::Int, q_t::Int)
#         q_t <= q_s && return 1
#         q_floor_full = isodd(q_t) ? q_t : q_t - 1
#         n_full = ((q_floor_full - q_s) >>> 1) + 1
#         return n_full + (iseven(q_t) ? 1 : 0)
#     end

#     @inline function inner_node_coord(node_idx::Int, q_s::Int, q_t::Int)
#         q_floor_full = isodd(q_t) ? q_t : q_t - 1
#         n_full = ((q_floor_full - q_s) >>> 1) + 1
#         return node_idx <= n_full ? q_s + 2 * (node_idx - 1) : q_t
#     end

#     @inline function ∫weight_between_coord(node_idx::Int, q_s::Int, q_t::Int)
#         q_t <= q_s && return 0.0
#         n_nodes = n_inner_nodes(q_s, q_t)
#         n_nodes <= 1 && return 0.0

#         q = inner_node_coord(node_idx, q_s, q_t)
#         if node_idx == 1
#             q_next = inner_node_coord(node_idx + 1, q_s, q_t)
#             return 0.25 * Δt * Float64(q_next - q)
#         elseif node_idx == n_nodes
#             q_prev = inner_node_coord(node_idx - 1, q_s, q_t)
#             return 0.25 * Δt * Float64(q - q_prev)
#         else
#             q_prev = inner_node_coord(node_idx - 1, q_s, q_t)
#             q_next = inner_node_coord(node_idx + 1, q_s, q_t)
#             return 0.25 * Δt * Float64(q_next - q_prev)
#         end
#     end

#     @inline function Δtime(qΔ::Int)
#         return 0.5 * (Float64(qΔ) - 1.0) * Δt
#     end

#     @inline function c_local(c_t::AbstractMatrix, a::Int, b::Int)
#         return (use_population_closure && a != b) ? const_ZERO_C : c_t[a, b]
#     end

#     @inline function c_mem(
#         c_t::AbstractMatrix,
#         q_s::Int,
#         q_t::Int,
#         a::Int,
#         b::Int,
#     )
#         if use_population_closure && a != b
#             return const_ZERO_C
#         end
#         if q_s == q_t
#             return c_t[a, b]
#         end
#         isodd(q_s) || error("non-endpoint memory q index must be a stored full-grid point, got q=$(q_s)")
#         s_idx = full_index_from_q(q_s)
#         @inbounds return c[a, b, s_idx]
#     end

#     # Backward-compatible names for the already-generated kernel functions.
#     # In the transported PT implementation these functions read the internal
#     # coordinate c, not the physical readout σ.
#     @inline σ_local(c_t::AbstractMatrix, a::Int, b::Int) = c_local(c_t, a, b)
#     @inline σ_mem(c_t::AbstractMatrix, q_s::Int, q_t::Int, a::Int, b::Int) = c_mem(c_t, q_s, q_t, a, b)

#     @inline function phase_exp(from_a::Int, from_b::Int, qΔ::Int)
#         check_q(qΔ)
#         @inbounds return phase_cache[qΔ, from_a, from_b]
#     end

#     # Markovian asymptote used only by the optional inner-G1 version of
#     # P L0 Q G1 Q L1 P.  Because Λ stores i*g′(∞), the replacement is
#     # g′(∞) = -im*Λ and g″(∞) = 0.
#     @inline Gp_markovian_G1(Δ_coord::Int, a::Int, b::Int, c::Int, d::Int) =
#         -1.0im * Λ[a, b, c, d]

#     @inline Gpp_markovian_G1(Δ_coord::Int, a::Int, b::Int, c::Int, d::Int) =
#         const_ZERO_C


#     @inline function is_secular_pair(
#         out_a::Int,
#         out_b::Int,
#         in_a::Int,
#         in_b::Int,
#     )
#         if !use_secular
#             return true
#         end
#         return abs(Ω(out_a, out_b) - Ω(in_a, in_b)) <= secular_tol
#     end

#     @inline function D_L0P(
#         itr::Int,
#         α::Int,
#         β::Int,
#     )
#         return (
#             -1.0im * ω(α, β) / hbar_f
#             +(
#                 -Gp(itr, α, α, α, α)
#                 +conj(Gp(itr, α, α, β, β))
#                 +Gp(itr, β, β, α, α)
#                 -conj(Gp(itr, β, β, β, β))
#             ) / hbar2
#         )
#     end


#     # ---------------------------------------------------------------------
#     # Generic transported-dual c-update branch helper functions.
#     #
#     # These implement the explicit contour g-closure for all four
#     # P_T L1 Q_T G0 Q_T L1 P_T branches: LL, LR, RL, RR.
#     #
#     # Endpoint codes used below:
#     #   0 -> 0,  1 -> s,  2 -> t.
#     # Thus t-s is represented by endpoint difference 2-1.
#     # Negative-time g identities are resolved as
#     #   g_ab,cd(-τ)=conj(g_dc,ba(τ)),
#     #   g'_ab,cd(-τ)=-conj(g'_dc,ba(τ)),
#     #   g''_ab,cd(-τ)=conj(g''_dc,ba(τ)).
#     # ---------------------------------------------------------------------

#     @inline G0(a::Int, b::Int, c::Int, d::Int) = G(1, a, b, c, d)
#     @inline Gp0(a::Int, b::Int, c::Int, d::Int) = Gp(1, a, b, c, d)
#     @inline Gpp0(a::Int, b::Int, c::Int, d::Int) = Gpp(1, a, b, c, d)

#     @inline Gneg(q::Int, a::Int, b::Int, c::Int, d::Int) = conj(G(q, d, c, b, a))
#     @inline Gpneg(q::Int, a::Int, b::Int, c::Int, d::Int) = -conj(Gp(q, d, c, b, a))
#     @inline Gppneg(q::Int, a::Int, b::Int, c::Int, d::Int) = conj(Gpp(q, d, c, b, a))

#     @inline function logD_PT(qT::Int, a::Int, b::Int)
#         return (
#             -G(qT, a, a, a, a)
#             -conj(G(qT, b, b, b, b))
#             +G(qT, b, b, a, a)
#             +conj(G(qT, a, a, b, b))
#         ) / hbar2
#     end

#     @inline function Gdiff_ep(s_itr::Int, Δ_itr::Int, t_itr::Int,
#                               e1::Int, e0::Int,
#                               a::Int, b::Int, c::Int, d::Int)
#         if e1 == e0
#             return G0(a, b, c, d)
#         elseif e1 == 1 && e0 == 0
#             return G(s_itr, a, b, c, d)
#         elseif e1 == 2 && e0 == 0
#             return G(t_itr, a, b, c, d)
#         elseif e1 == 2 && e0 == 1
#             return G(Δ_itr, a, b, c, d)
#         elseif e1 == 0 && e0 == 1
#             return Gneg(s_itr, a, b, c, d)
#         elseif e1 == 0 && e0 == 2
#             return Gneg(t_itr, a, b, c, d)
#         elseif e1 == 1 && e0 == 2
#             return Gneg(Δ_itr, a, b, c, d)
#         else
#             error("Unsupported endpoint difference in Gdiff_ep: $e1 - $e0")
#         end
#     end

#     @inline function Gpdiff_ep(s_itr::Int, Δ_itr::Int, t_itr::Int,
#                                e1::Int, e0::Int,
#                                a::Int, b::Int, c::Int, d::Int)
#         if e1 == e0
#             return Gp0(a, b, c, d)
#         elseif e1 == 1 && e0 == 0
#             return Gp(s_itr, a, b, c, d)
#         elseif e1 == 2 && e0 == 0
#             return Gp(t_itr, a, b, c, d)
#         elseif e1 == 2 && e0 == 1
#             return Gp(Δ_itr, a, b, c, d)
#         elseif e1 == 0 && e0 == 1
#             return Gpneg(s_itr, a, b, c, d)
#         elseif e1 == 0 && e0 == 2
#             return Gpneg(t_itr, a, b, c, d)
#         elseif e1 == 1 && e0 == 2
#             return Gpneg(Δ_itr, a, b, c, d)
#         else
#             error("Unsupported endpoint difference in Gpdiff_ep: $e1 - $e0")
#         end
#     end

#     @inline function Gppdiff_ep(s_itr::Int, Δ_itr::Int, t_itr::Int,
#                                 e1::Int, e0::Int,
#                                 a::Int, b::Int, c::Int, d::Int)
#         if e1 == e0
#             return Gpp0(a, b, c, d)
#         elseif e1 == 1 && e0 == 0
#             return Gpp(s_itr, a, b, c, d)
#         elseif e1 == 2 && e0 == 0
#             return Gpp(t_itr, a, b, c, d)
#         elseif e1 == 2 && e0 == 1
#             return Gpp(Δ_itr, a, b, c, d)
#         elseif e1 == 0 && e0 == 1
#             return Gppneg(s_itr, a, b, c, d)
#         elseif e1 == 0 && e0 == 2
#             return Gppneg(t_itr, a, b, c, d)
#         elseif e1 == 1 && e0 == 2
#             return Gppneg(Δ_itr, a, b, c, d)
#         else
#             error("Unsupported endpoint difference in Gppdiff_ep: $e1 - $e0")
#         end
#     end

#     @inline function seg_self_g(s_itr::Int, Δ_itr::Int, t_itr::Int, seg)
#         η = seg[1]
#         r = seg[2]
#         a = seg[3]
#         b = seg[4]
#         # Forward segment: g(b-a).  Backward segment: g(a-b).
#         if η == 1
#             return Gdiff_ep(s_itr, Δ_itr, t_itr, b, a, r, r, r, r)
#         else
#             return Gdiff_ep(s_itr, Δ_itr, t_itr, a, b, r, r, r, r)
#         end
#     end

#     @inline function Jseg(s_itr::Int, Δ_itr::Int, t_itr::Int, seg1, seg2)
#         r1 = seg1[2]
#         a1 = seg1[3]
#         b1 = seg1[4]
#         r2 = seg2[2]
#         a2 = seg2[3]
#         b2 = seg2[4]
#         return (
#             Gdiff_ep(s_itr, Δ_itr, t_itr, b1, a2, r1, r1, r2, r2)
#             -Gdiff_ep(s_itr, Δ_itr, t_itr, a1, a2, r1, r1, r2, r2)
#             -Gdiff_ep(s_itr, Δ_itr, t_itr, b1, b2, r1, r1, r2, r2)
#             +Gdiff_ep(s_itr, Δ_itr, t_itr, a1, b2, r1, r1, r2, r2)
#         )
#     end

#     @inline function theta_path(s_itr::Int, Δ_itr::Int, t_itr::Int, segs)
#         θ = 0.0 + 0.0im
#         N = length(segs)
#         for i in 1:N
#             θ -= seg_self_g(s_itr, Δ_itr, t_itr, segs[i])
#         end
#         for i in 2:N
#             ηi = segs[i][1]
#             for j in 1:(i - 1)
#                 ηj = segs[j][1]
#                 θ -= (ηi * ηj) * Jseg(s_itr, Δ_itr, t_itr, segs[i], segs[j])
#             end
#         end
#         return θ / hbar2
#     end

#     @inline function A_path(s_itr::Int, Δ_itr::Int, t_itr::Int,
#                             segs, pos::Int, τep::Int,
#                             iidx::Int, jidx::Int)
#         # Insertion V_{iidx,jidx} is placed between segs[pos] and segs[pos+1].
#         # Segments 1:pos are to the right of the insertion in the operator word.
#         # Segments pos+1:end are to the left of the insertion.
#         A = 0.0 + 0.0im
#         N = length(segs)

#         for k in 1:pos
#             seg = segs[k]
#             η = seg[1]
#             r = seg[2]
#             a = seg[3]
#             b = seg[4]
#             Iright = (
#                 Gpdiff_ep(s_itr, Δ_itr, t_itr, τep, a, iidx, jidx, r, r)
#                 -Gpdiff_ep(s_itr, Δ_itr, t_itr, τep, b, iidx, jidx, r, r)
#             )
#             A -= η * Iright
#         end

#         for k in (pos + 1):N
#             seg = segs[k]
#             η = seg[1]
#             r = seg[2]
#             a = seg[3]
#             b = seg[4]
#             Ileft = (
#                 Gpdiff_ep(s_itr, Δ_itr, t_itr, b, τep, r, r, iidx, jidx)
#                 -Gpdiff_ep(s_itr, Δ_itr, t_itr, a, τep, r, r, iidx, jidx)
#             )
#             A -= η * Ileft
#         end
#         return A
#     end

#     @inline function C_connected(s_itr::Int, Δ_itr::Int, t_itr::Int,
#                                  pos1::Int, τ1::Int, i1::Int, j1::Int,
#                                  pos2::Int, τ2::Int, i2::Int, j2::Int)
#         # The leftmost insertion in the operator word has the larger pos.
#         if pos1 > pos2
#             return Gppdiff_ep(s_itr, Δ_itr, t_itr, τ1, τ2, i1, j1, i2, j2)
#         else
#             return Gppdiff_ep(s_itr, Δ_itr, t_itr, τ2, τ1, i2, j2, i1, j1)
#         end
#     end

#     @inline function kernel_branch_path_closed(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         c_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         phase_left::Int,
#         phase_right::Int,
#         c_left::Int,
#         c_right::Int,
#         input_left::Int,
#         input_right::Int,
#         direct_segs,
#         inner_sub_segs,
#         outer_sub_segs,
#         pos_outer::Int,
#         τ_outer::Int,
#         i_outer::Int,
#         j_outer::Int,
#         pos_inner::Int,
#         τ_inner::Int,
#         i_inner::Int,
#         j_inner::Int,
#         sign_branch::Int,
#     )
#         BD = (
#             logD_PT(t_itr, α, β)
#             -logD_PT(s_itr, input_left, input_right)
#             +theta_path(s_itr, Δ_itr, t_itr, direct_segs)
#         )
#         BS = (
#             logD_PT(t_itr, α, β)
#             -logD_PT(s_itr, input_left, input_right)
#             +theta_path(s_itr, Δ_itr, t_itr, inner_sub_segs)
#             +theta_path(s_itr, Δ_itr, t_itr, outer_sub_segs)
#         )

#         AoutD = A_path(s_itr, Δ_itr, t_itr, direct_segs, pos_outer, τ_outer, i_outer, j_outer)
#         AinD  = A_path(s_itr, Δ_itr, t_itr, direct_segs, pos_inner, τ_inner, i_inner, j_inner)
#         Cdir  = C_connected(s_itr, Δ_itr, t_itr, pos_outer, τ_outer, i_outer, j_outer, pos_inner, τ_inner, i_inner, j_inner)

#         AinnerS = A_path(s_itr, Δ_itr, t_itr, inner_sub_segs, 1, τ_inner, i_inner, j_inner)
#         AouterS = A_path(s_itr, Δ_itr, t_itr, outer_sub_segs, 1, τ_outer, i_outer, j_outer)

#         direct = sign_branch * exp(BD) * ((hbar2 * Cdir - AoutD * AinD) / hbar4)
#         subtr  = sign_branch * exp(BS) * ((AinnerS * AouterS) / hbar4)

#         return c_mem(c_t, s_itr, t_itr, c_left, c_right) * phase_exp(phase_left, phase_right, Δ_itr) * (direct + subtr)
#     end

#     @inline function kernel_LL_path_closed(s_itr::Int, Δ_itr::Int, t_itr::Int,
#                                            c_t::AbstractMatrix, α::Int, β::Int, χ::Int, μ::Int)
#         direct_segs = ((1, μ, 0, 1), (1, χ, 1, 2), (-1, α, 0, 2))
#         inner_sub_segs = ((1, μ, 0, 1), (-1, χ, 0, 1))
#         outer_sub_segs = ((1, χ, 0, 2), (-1, α, 0, 2))
#         return kernel_branch_path_closed(
#             s_itr, Δ_itr, t_itr, c_t, α, β,
#             χ, β, μ, β, μ, β,
#             direct_segs, inner_sub_segs, outer_sub_segs,
#             2, 2, α, χ,
#             1, 1, χ, μ,
#             -1,
#         )
#     end

#     @inline function kernel_LR_path_closed(s_itr::Int, Δ_itr::Int, t_itr::Int,
#                                            c_t::AbstractMatrix, α::Int, β::Int, χ::Int, ν::Int)
#         # Word after expanding τ^T_{χν}(s):
#         #   χ[0,t] -> V_{αχ}(t) -> α[0,t]^* -> β[0,s] -> V_{νβ}(s) -> ν[0,s]^*.
#         direct_segs = ((1, χ, 0, 2), (-1, α, 0, 2), (1, β, 0, 1), (-1, ν, 0, 1))
#         inner_sub_segs = ((1, β, 0, 1), (-1, ν, 0, 1))
#         outer_sub_segs = ((1, χ, 0, 2), (-1, α, 0, 2))
#         return kernel_branch_path_closed(
#             s_itr, Δ_itr, t_itr, c_t, α, β,
#             χ, β, χ, ν, χ, ν,
#             direct_segs, inner_sub_segs, outer_sub_segs,
#             1, 2, α, χ,
#             3, 1, ν, β,
#             1,
#         )
#     end

#     @inline function kernel_RL_path_closed(s_itr::Int, Δ_itr::Int, t_itr::Int,
#                                            c_t::AbstractMatrix, α::Int, β::Int, χ::Int, μ::Int)
#         # Word after expanding τ^T_{μχ}(s):
#         #   μ[0,s] -> V_{αμ}(s) -> α[0,s]^* -> β[0,t] -> V_{χβ}(t) -> χ[0,t]^*.
#         direct_segs = ((1, μ, 0, 1), (-1, α, 0, 1), (1, β, 0, 2), (-1, χ, 0, 2))
#         inner_sub_segs = ((1, μ, 0, 1), (-1, α, 0, 1))
#         outer_sub_segs = ((1, β, 0, 2), (-1, χ, 0, 2))
#         return kernel_branch_path_closed(
#             s_itr, Δ_itr, t_itr, c_t, α, β,
#             α, χ, μ, χ, μ, χ,
#             direct_segs, inner_sub_segs, outer_sub_segs,
#             3, 2, χ, β,
#             1, 1, α, μ,
#             1,
#         )
#     end

#     @inline function kernel_RR_path_closed(s_itr::Int, Δ_itr::Int, t_itr::Int,
#                                            c_t::AbstractMatrix, α::Int, β::Int, χ::Int, ν::Int)
#         # Word after expanding τ^T_{αν}(s):
#         #   β[0,t] -> V_{χβ}(t) -> χ[s,t]^* -> V_{νχ}(s) -> ν[0,s]^*.
#         direct_segs = ((1, β, 0, 2), (-1, χ, 1, 2), (-1, ν, 0, 1))
#         inner_sub_segs = ((1, χ, 0, 1), (-1, ν, 0, 1))
#         outer_sub_segs = ((1, β, 0, 2), (-1, χ, 0, 2))
#         return kernel_branch_path_closed(
#             s_itr, Δ_itr, t_itr, c_t, α, β,
#             α, χ, α, ν, α, ν,
#             direct_segs, inner_sub_segs, outer_sub_segs,
#             1, 2, χ, β,
#             2, 1, ν, χ,
#             -1,
#         )
#     end


#     # ---------------------------------------------------------------------
#     # Physical σ-readout from transported internal coordinate c.
#     #
#     # σ_{αβ}(t) = c_{αβ}(t) + Tr_B{Q_T(t)ρ(t)}.
#     # At G0 level the Q_T readout has only one-L1 left/right source branches.
#     # The two branch kernels below were generated from the notebook's
#     # preprocessed g-closed readout expressions.
#     # ---------------------------------------------------------------------

#     @inline function readout_Q_L(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         c_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         μ::Int,
#     )
#         return c_mem(c_t, s_itr, t_itr, μ, β) * (-1 * (1 / hbar2) * ((((-1 * Gp(Δ_itr, β, β, α, μ)) + (-1 * conj(Gp(s_itr, μ, α, β, β))) + Gp(s_itr, α, μ, μ, μ) + Gp(Δ_itr, α, α, α, μ)) * exp(((1 / hbar2) * ((-1 * G(s_itr, μ, μ, μ, μ)) + (-1 * G(t_itr, α, α, μ, μ)) + (-1 * G(Δ_itr, α, α, α, α)) + (-1 * G(Δ_itr, β, β, μ, μ)) + (-1 * conj(G(t_itr, μ, μ, β, β))) + (-1 * conj(G(s_itr, α, α, β, β))) + G(t_itr, μ, μ, μ, μ) + G(s_itr, α, α, μ, μ) + G(Δ_itr, α, α, μ, μ) + G(Δ_itr, β, β, α, α) + conj(G(s_itr, μ, μ, β, β)) + conj(G(t_itr, α, α, β, β)))))) + (((-1 * Gp(s_itr, α, μ, μ, μ)) + conj(Gp(s_itr, μ, α, α, α))) * exp(((1 / hbar2) * ((-1 * G(s_itr, α, α, α, α)) + (-1 * conj(G(s_itr, β, β, β, β))) + G(s_itr, β, β, α, α) + conj(G(s_itr, α, α, β, β))))) * exp(((1 / hbar2) * ((-1 * G(s_itr, μ, μ, μ, μ)) + (-1 * G(t_itr, β, β, μ, μ)) + (-1 * conj(G(t_itr, μ, μ, β, β))) + (-1 * conj(G(s_itr, α, α, α, α))) + G(t_itr, μ, μ, μ, μ) + G(s_itr, α, α, μ, μ) + conj(G(s_itr, μ, μ, α, α)) + conj(G(t_itr, β, β, β, β))))) * exp(((1 / hbar2) * ((-1 * G(t_itr, α, α, α, α)) + (-1 * G(s_itr, β, β, α, α)) + (-1 * conj(G(s_itr, α, α, β, β))) + (-1 * conj(G(t_itr, β, β, β, β))) + G(s_itr, α, α, α, α) + G(t_itr, β, β, α, α) + conj(G(t_itr, α, α, β, β)) + conj(G(s_itr, β, β, β, β))))))) * exp(((1 / hbar_f) * ((1.0im * ϵ[β] * Δtime(Δ_itr)) + (-1 * 1.0im * ϵ[α] * Δtime(Δ_itr))))) * exp(((1 / hbar2) * ((-1 * G(t_itr, μ, μ, μ, μ)) + (-1 * G(s_itr, β, β, μ, μ)) + (-1 * conj(G(s_itr, μ, μ, β, β))) + (-1 * conj(G(t_itr, β, β, β, β))) + G(s_itr, μ, μ, μ, μ) + G(t_itr, β, β, μ, μ) + conj(G(t_itr, μ, μ, β, β)) + conj(G(s_itr, β, β, β, β))))))
#     end


#     @inline function readout_Q_R(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         c_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         ν::Int,
#     )
#         return c_mem(c_t, s_itr, t_itr, α, ν) * (-1 * (1 / hbar2) * ((((-1 * Gp(s_itr, ν, β, α, α)) + (-1 * conj(Gp(Δ_itr, α, α, β, ν))) + conj(Gp(s_itr, β, ν, ν, ν)) + conj(Gp(Δ_itr, β, β, β, ν))) * exp(((1 / hbar2) * ((-1 * G(t_itr, ν, ν, α, α)) + (-1 * G(s_itr, β, β, α, α)) + (-1 * conj(G(s_itr, ν, ν, ν, ν))) + (-1 * conj(G(Δ_itr, α, α, ν, ν))) + (-1 * conj(G(t_itr, β, β, ν, ν))) + (-1 * conj(G(Δ_itr, β, β, β, β))) + G(s_itr, ν, ν, α, α) + G(t_itr, β, β, α, α) + conj(G(t_itr, ν, ν, ν, ν)) + conj(G(Δ_itr, α, α, β, β)) + conj(G(s_itr, β, β, ν, ν)) + conj(G(Δ_itr, β, β, ν, ν)))))) + (-1 * ((-1 * Gp(s_itr, ν, β, β, β)) + conj(Gp(s_itr, β, ν, ν, ν))) * exp(((1 / hbar2) * ((-1 * G(s_itr, α, α, α, α)) + (-1 * conj(G(s_itr, β, β, β, β))) + G(s_itr, β, β, α, α) + conj(G(s_itr, α, α, β, β))))) * exp(((1 / hbar2) * ((-1 * G(t_itr, ν, ν, α, α)) + (-1 * G(s_itr, β, β, β, β)) + (-1 * conj(G(s_itr, ν, ν, ν, ν))) + (-1 * conj(G(t_itr, α, α, ν, ν))) + G(s_itr, ν, ν, β, β) + G(t_itr, α, α, α, α) + conj(G(t_itr, ν, ν, ν, ν)) + conj(G(s_itr, β, β, ν, ν))))) * exp(((1 / hbar2) * ((-1 * G(t_itr, α, α, α, α)) + (-1 * G(s_itr, β, β, α, α)) + (-1 * conj(G(s_itr, α, α, β, β))) + (-1 * conj(G(t_itr, β, β, β, β))) + G(s_itr, α, α, α, α) + G(t_itr, β, β, α, α) + conj(G(t_itr, α, α, β, β)) + conj(G(s_itr, β, β, β, β))))))) * exp(((1 / hbar_f) * ((1.0im * ϵ[β] * Δtime(Δ_itr)) + (-1 * 1.0im * ϵ[α] * Δtime(Δ_itr))))) * exp(((1 / hbar2) * ((-1 * G(s_itr, ν, ν, α, α)) + (-1 * G(t_itr, α, α, α, α)) + (-1 * conj(G(t_itr, ν, ν, ν, ν))) + (-1 * conj(G(s_itr, α, α, ν, ν))) + G(t_itr, ν, ν, α, α) + G(s_itr, α, α, α, α) + conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, α, α, ν, ν))))))
#     end


#     # ---------------------------------------------------------------------
#     # PT c-update kernels generated from Jupyter g-closed c_update_branches.
#     # This block contains ONLY local + P_T L1 Q_T G0 Q_T L1 P_T terms.
#     # No physical sigma readout and no PL0QG0/PL0QG1 memory is included here.
#     # ---------------------------------------------------------------------
#     @inline function local_PT_diag(t_eval_coord::Int, α::Int, β::Int)
#         # Correct normalized transported local L0 generator for the internal
#         # coordinate c_{αβ}.
#         #
#         # c_{αβ}(t) = D_{αβ}(t) Tr_B[U_α†(t) ρ_{αβ}(t) U_β(t)], so
#         # d/dt c contains both ell_T(t)[L0 ρ] and dot(ell_T)(t)[ρ].
#         # The dot(ell_T) contribution cancels the explicit transported bath
#         # motion and leaves the scalar line-shape generator d log D_{αβ}/dt
#         # plus the electronic phase.
#         #
#         # The earlier generated expression used only ell_T L0 R_T and missed
#         # dot(ell_T), producing a numerically noisy extra local term.
#         return D_L0P(t_eval_coord, α, β)
#     end

#     @inline function local_PT_L(t_eval_coord::Int, α::Int, β::Int, μ::Int)
#         return ((1 / hbar2) * ((-1 * Gp(t_eval_coord, α, μ, μ, μ)) + conj(Gp(t_eval_coord, μ, α, α, α))) * exp(((1 / hbar2) * ((-1 * G(t_eval_coord, α, α, α, α)) + (-1 * conj(G(t_eval_coord, β, β, β, β))) + G(t_eval_coord, β, β, α, α) + conj(G(t_eval_coord, α, α, β, β))))) * exp(((1 / hbar2) * ((-1 * G(t_eval_coord, β, β, μ, μ)) + (-1 * conj(G(t_eval_coord, μ, μ, β, β))) + (-1 * conj(G(t_eval_coord, α, α, α, α))) + G(t_eval_coord, α, α, μ, μ) + conj(G(t_eval_coord, μ, μ, α, α)) + conj(G(t_eval_coord, β, β, β, β))))))
#     end

#     @inline function local_PT_R(t_eval_coord::Int, α::Int, β::Int, ν::Int)
#         return (-1 * (1 / hbar2) * ((-1 * Gp(t_eval_coord, ν, β, β, β)) + conj(Gp(t_eval_coord, β, ν, ν, ν))) * exp(((1 / hbar2) * ((-1 * G(t_eval_coord, α, α, α, α)) + (-1 * conj(G(t_eval_coord, β, β, β, β))) + G(t_eval_coord, β, β, α, α) + conj(G(t_eval_coord, α, α, β, β))))) * exp(((1 / hbar2) * ((-1 * G(t_eval_coord, ν, ν, α, α)) + (-1 * G(t_eval_coord, β, β, β, β)) + (-1 * conj(G(t_eval_coord, α, α, ν, ν))) + G(t_eval_coord, ν, ν, β, β) + G(t_eval_coord, α, α, α, α) + conj(G(t_eval_coord, β, β, ν, ν))))))
#     end

#     @inline function kernel_LL(s_itr::Int, Δ_itr::Int, t_itr::Int, c_t::AbstractMatrix, α::Int, β::Int, χ::Int, μ::Int)
#         return kernel_LL_path_closed(s_itr, Δ_itr, t_itr, c_t, α, β, χ, μ)
#     end

#     @inline function kernel_LR(s_itr::Int, Δ_itr::Int, t_itr::Int, c_t::AbstractMatrix, α::Int, β::Int, χ::Int, ν::Int)
#         return kernel_LR_path_closed(s_itr, Δ_itr, t_itr, c_t, α, β, χ, ν)
#     end

#     @inline function kernel_RL(s_itr::Int, Δ_itr::Int, t_itr::Int, c_t::AbstractMatrix, α::Int, β::Int, χ::Int, μ::Int)
#         return kernel_RL_path_closed(s_itr, Δ_itr, t_itr, c_t, α, β, χ, μ)
#     end

#     @inline function kernel_RR(s_itr::Int, Δ_itr::Int, t_itr::Int, c_t::AbstractMatrix, α::Int, β::Int, χ::Int, ν::Int)
#         return kernel_RR_path_closed(s_itr, Δ_itr, t_itr, c_t, α, β, χ, ν)
#     end


#     # ---------------------------------------------------------------------
#     # Explicit P L0 Q G0 Q L1 P, g-closed branch prefactors and coefficients.
#     # Branch L : input (μ, β) -> output (α, β)
#     # Branch R : input (α, ν) -> output (α, β)
#     # ---------------------------------------------------------------------

#     @inline function exponent_L0_L(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         μ::Int,
#     )
#         # exp[-(i hbar ω_{αβ} Δ + G_L) / hbar^2]
#         # = phase_exp(α, β, Δ) * exp[-G_L / hbar^2]
#         return (
#             -G(Δ_itr, α, α, α, α)
#             +G(s_itr, α, α, μ, μ)
#             -G(t_itr, α, α, μ, μ)
#             +G(Δ_itr, α, α, μ, μ)
#             +G(Δ_itr, β, β, α, α)
#             -G(s_itr, β, β, μ, μ)
#             +G(t_itr, β, β, μ, μ)
#             -G(Δ_itr, β, β, μ, μ)
#             -conj(G(s_itr, α, α, β, β))
#             +conj(G(t_itr, α, α, β, β))
#             +conj(G(s_itr, β, β, β, β))
#             -conj(G(t_itr, β, β, β, β))
#         )
#     end

#     @inline function coef_L0_L(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         μ::Int,
#     )
#         left_bracket = (
#             Gp(Δ_itr, α, α, α, μ)
#             +Gp(s_itr, α, μ, μ, μ)
#             -Gp(Δ_itr, β, β, α, μ)
#             -conj(Gp(s_itr, μ, α, β, β))
#         )

#         right_bracket = (
#             Gp(t_itr, α, α, α, α)
#             -Gp(Δ_itr, α, α, α, α)
#             -Gp(t_itr, α, α, μ, μ)
#             +Gp(Δ_itr, α, α, μ, μ)
#             -Gp(t_itr, β, β, α, α)
#             +Gp(Δ_itr, β, β, α, α)
#             +Gp(t_itr, β, β, μ, μ)
#             -Gp(Δ_itr, β, β, μ, μ)
#         )

#         return (
#             hbar2 * Gpp(Δ_itr, α, α, α, μ)
#             -hbar2 * Gpp(Δ_itr, β, β, α, μ)
#             +left_bracket * right_bracket
#         )
#     end

#     @inline function kernel_L0_L(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         μ::Int,
#     )
#         return -(
#             σ_mem(σ_t, s_itr, t_itr, μ, β)
#             * phase_exp(α, β, Δ_itr)
#             * exp(exponent_L0_L(s_itr, Δ_itr, t_itr, α, β, μ) / hbar2)
#             * coef_L0_L(s_itr, Δ_itr, t_itr, α, β, μ) / hbar4
#         )
#     end

#     @inline function exponent_L0_R(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         ν::Int,
#     )
#         # exp[-(i hbar ω_{αβ} Δ + G_R) / hbar^2]
#         # = phase_exp(α, β, Δ) * exp[-G_R / hbar^2]
#         return (
#             +G(s_itr, α, α, α, α)
#             -G(t_itr, α, α, α, α)
#             -G(s_itr, β, β, α, α)
#             +G(t_itr, β, β, α, α)
#             +conj(G(Δ_itr, α, α, β, β))
#             -conj(G(s_itr, α, α, ν, ν))
#             +conj(G(t_itr, α, α, ν, ν))
#             -conj(G(Δ_itr, α, α, ν, ν))
#             -conj(G(Δ_itr, β, β, β, β))
#             +conj(G(s_itr, β, β, ν, ν))
#             -conj(G(t_itr, β, β, ν, ν))
#             +conj(G(Δ_itr, β, β, ν, ν))
#         )
#     end

#     @inline function coef_L0_R(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         α::Int,
#         β::Int,
#         ν::Int,
#     )
#         left_bracket = (
#             Gp(s_itr, ν, β, α, α)
#             +conj(Gp(Δ_itr, α, α, β, ν))
#             -conj(Gp(Δ_itr, β, β, β, ν))
#             -conj(Gp(s_itr, β, ν, ν, ν))
#         )

#         right_bracket = (
#             conj(Gp(t_itr, α, α, β, β))
#             -conj(Gp(Δ_itr, α, α, β, β))
#             -conj(Gp(t_itr, α, α, ν, ν))
#             +conj(Gp(Δ_itr, α, α, ν, ν))
#             -conj(Gp(t_itr, β, β, β, β))
#             +conj(Gp(Δ_itr, β, β, β, β))
#             +conj(Gp(t_itr, β, β, ν, ν))
#             -conj(Gp(Δ_itr, β, β, ν, ν))
#         )

#         return (
#             -hbar2 * conj(Gpp(Δ_itr, α, α, β, ν))
#             +hbar2 * conj(Gpp(Δ_itr, β, β, β, ν))
#             +left_bracket * right_bracket
#         )
#     end

#     @inline function kernel_L0_R(
#         s_itr::Int,
#         Δ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         ν::Int,
#     )
#         return -(
#             σ_mem(σ_t, s_itr, t_itr, α, ν)
#             * phase_exp(α, β, Δ_itr)
#             * exp(exponent_L0_R(s_itr, Δ_itr, t_itr, α, β, ν) / hbar2)
#             * coef_L0_R(s_itr, Δ_itr, t_itr, α, β, ν) / hbar4
#         )
#     end



#     # ---------------------------------------------------------------------
#     # Explicit P L0 Q G1 Q L1 P, g-closed 2-time branch kernels.
#     # The extra variable τ satisfies s <= τ <= t.  Each branch is the
#     # direct g-closed expression from I^{*,L0G1L1,g}_{αβ}(t,τ,s).
#     # ---------------------------------------------------------------------

#     Base.@noinline function kernel_L0G1_LL(
#         s_itr::Int,
#         τ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((hbar2 * Gpp(Δtτ_itr, α, α, α, χ) - hbar2 * Gpp(Δtτ_itr, β, β, α, χ) + (- Gp(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β))) * (- Gp(t_itr, α, α, α, α) + Gp(Δtτ_itr, α, α, α, α) + Gp(t_itr, α, α, χ, χ) - Gp(Δtτ_itr, α, α, χ, χ) + Gp(t_itr, β, β, α, α) - Gp(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp(Δtτ_itr, β, β, χ, χ))) * (Gp(Δτs_itr, β, β, χ, μ) - Gp(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) + G(t_itr, β, β, χ, χ) - G(τ_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(Δτs_itr, β, β, χ, χ) - G(t_itr, β, β, μ, μ) + G(τ_itr, β, β, μ, μ) - G(Δτs_itr, β, β, μ, μ) - G(Δτs_itr, χ, χ, χ, χ) + G(s_itr, χ, χ, μ, μ) - G(τ_itr, χ, χ, μ, μ) + G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(s_itr, χ, χ, β, β)) + conj(G(τ_itr, χ, χ, β, β)) + conj(G(s_itr, μ, μ, β, β)) - conj(G(t_itr, μ, μ, β, β))) / (hbar2))) + (hbar2 * (- Gpp(Δtτ_itr, α, α, α, χ) + Gpp(Δtτ_itr, β, β, α, χ)) * (- Gp(Δts_itr, α, α, χ, μ) + Gp(Δτs_itr, α, α, χ, μ) + Gp(Δts_itr, β, β, χ, μ) - Gp(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β))) + hbar2 * (- Gp(Δtτ_itr, β, β, α, α) - Gp(Δts_itr, β, β, χ, χ) + Gp(Δtτ_itr, β, β, χ, χ) - Gp(t_itr, β, β, μ, μ) + Gp(Δts_itr, β, β, μ, μ) + conj(Gp(t_itr, β, β, β, β))) * Gpp(Δτs_itr, α, χ, χ, μ) + (hbar2 * Gpp(Δτs_itr, α, χ, χ, μ) + (Gp(Δtτ_itr, α, α, α, χ) + Gp(Δτs_itr, α, χ, χ, χ) + Gp(τ_itr, α, χ, μ, μ) - Gp(Δτs_itr, α, χ, μ, μ) - Gp(Δtτ_itr, β, β, α, χ) - conj(Gp(τ_itr, χ, α, β, β))) * (- Gp(Δts_itr, α, α, χ, μ) + Gp(Δτs_itr, α, α, χ, μ) + Gp(Δts_itr, β, β, χ, μ) - Gp(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β)))) * (- Gp(t_itr, α, α, α, α) + Gp(Δtτ_itr, α, α, α, α) + Gp(Δts_itr, α, α, χ, χ) - Gp(Δtτ_itr, α, α, χ, χ) + Gp(t_itr, α, α, μ, μ) - Gp(Δts_itr, α, α, μ, μ) + Gp(t_itr, β, β, α, α) - conj(Gp(t_itr, β, β, β, β))) + (- hbar2 * Gpp(Δts_itr, α, α, χ, μ) + hbar2 * Gpp(Δts_itr, β, β, χ, μ) + (Gp(Δts_itr, α, α, χ, μ) - Gp(Δτs_itr, α, α, χ, μ) - Gp(Δts_itr, β, β, χ, μ) + Gp(Δτs_itr, χ, χ, χ, μ) + Gp(s_itr, χ, μ, μ, μ) - conj(Gp(s_itr, μ, χ, β, β))) * (- Gp(Δtτ_itr, β, β, α, α) - Gp(Δts_itr, β, β, χ, χ) + Gp(Δtτ_itr, β, β, χ, χ) - Gp(t_itr, β, β, μ, μ) + Gp(Δts_itr, β, β, μ, μ) + conj(Gp(t_itr, β, β, β, β)))) * (- Gp(Δtτ_itr, α, α, α, χ) - Gp(Δτs_itr, α, χ, χ, χ) - Gp(τ_itr, α, χ, μ, μ) + Gp(Δτs_itr, α, χ, μ, μ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(Δts_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δτs_itr, α, α, χ, χ) - G(t_itr, α, α, μ, μ) + G(τ_itr, α, α, μ, μ) + G(Δts_itr, α, α, μ, μ) - G(Δτs_itr, α, α, μ, μ) + G(Δtτ_itr, β, β, α, α) + G(Δts_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) - G(Δts_itr, β, β, μ, μ) - G(Δτs_itr, χ, χ, χ, χ) + G(s_itr, χ, χ, μ, μ) - G(τ_itr, χ, χ, μ, μ) + G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(s_itr, χ, χ, β, β)) + conj(G(τ_itr, χ, χ, β, β)) + conj(G(s_itr, μ, μ, β, β)) - conj(G(t_itr, μ, μ, β, β))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, μ, β) * exp(((- G(s_itr, β, β, μ, μ) + G(t_itr, β, β, μ, μ) + G(s_itr, μ, μ, μ, μ) - G(t_itr, μ, μ, μ, μ) + conj(G(s_itr, β, β, β, β)) - conj(G(t_itr, β, β, β, β)) - conj(G(s_itr, μ, μ, β, β)) + conj(G(t_itr, μ, μ, β, β))) / (hbar2)))
#         )
#     end

#     Base.@noinline function kernel_L0G1_LR(
#         s_itr::Int,
#         τ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((- hbar2 * Gpp(Δtτ_itr, α, α, α, χ) + hbar2 * Gpp(Δtτ_itr, β, β, α, χ) + (- Gp(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β))) * (Gp(t_itr, α, α, α, α) - Gp(Δtτ_itr, α, α, α, α) - Gp(t_itr, α, α, χ, χ) + Gp(Δtτ_itr, α, α, χ, χ) - Gp(t_itr, β, β, α, α) + Gp(Δtτ_itr, β, β, α, α) + Gp(t_itr, β, β, χ, χ) - Gp(Δtτ_itr, β, β, χ, χ))) * (- Gp(s_itr, ν, β, χ, χ) + conj(Gp(Δτs_itr, β, β, β, ν)) + conj(Gp(s_itr, β, ν, ν, ν)) - conj(Gp(Δτs_itr, χ, χ, β, ν))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) - G(s_itr, β, β, χ, χ) + G(t_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(t_itr, χ, χ, χ, χ) - G(τ_itr, χ, χ, χ, χ) + G(s_itr, ν, ν, χ, χ) - G(t_itr, ν, ν, χ, χ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(t_itr, β, β, β, β)) + conj(G(τ_itr, β, β, β, β)) - conj(G(Δτs_itr, β, β, β, β)) + conj(G(s_itr, β, β, ν, ν)) - conj(G(τ_itr, β, β, ν, ν)) + conj(G(Δτs_itr, β, β, ν, ν)) + conj(G(Δτs_itr, χ, χ, β, β)) - conj(G(t_itr, χ, χ, ν, ν)) + conj(G(τ_itr, χ, χ, ν, ν)) - conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2))) + (hbar2 * (conj(Gpp(Δts_itr, α, α, β, ν)) - conj(Gpp(Δts_itr, β, β, β, ν))) * (- Gp(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp(Δτs_itr, χ, α, ν, ν))) + hbar2 * (- Gp(Δtτ_itr, α, α, α, α) - Gp(t_itr, α, α, χ, χ) + Gp(Δtτ_itr, α, α, χ, χ) + conj(Gp(Δts_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, ν, ν)) - conj(Gp(Δts_itr, α, α, ν, ν))) * conj(Gpp(Δτs_itr, χ, α, β, ν)) - hbar2 * (- Gp(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp(Δtτ_itr, β, β, χ, χ) + conj(Gp(Δts_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp(Δts_itr, β, β, ν, ν))) * conj(Gpp(Δτs_itr, χ, α, β, ν)) + (hbar2 * conj(Gpp(Δτs_itr, χ, α, β, ν)) + (- Gp(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp(Δτs_itr, χ, α, ν, ν))) * (Gp(s_itr, ν, β, χ, χ) + conj(Gp(Δts_itr, α, α, β, ν)) - conj(Gp(Δτs_itr, α, α, β, ν)) - conj(Gp(Δts_itr, β, β, β, ν)) - conj(Gp(s_itr, β, ν, ν, ν)) + conj(Gp(Δτs_itr, χ, χ, β, ν)))) * (Gp(t_itr, α, α, α, α) - Gp(t_itr, β, β, α, α) - conj(Gp(t_itr, α, α, β, β)) + conj(Gp(t_itr, β, β, β, β))) + (hbar2 * Gpp(Δtτ_itr, α, α, α, χ) - hbar2 * Gpp(Δtτ_itr, β, β, α, χ) + (- Gp(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp(Δtτ_itr, β, β, α, χ) + conj(Gp(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp(Δτs_itr, χ, α, ν, ν))) * (Gp(Δtτ_itr, α, α, α, α) + Gp(t_itr, α, α, χ, χ) - Gp(Δtτ_itr, α, α, χ, χ) - Gp(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp(Δtτ_itr, β, β, χ, χ) - conj(Gp(Δts_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, ν, ν)) + conj(Gp(Δts_itr, α, α, ν, ν)) + conj(Gp(Δts_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp(Δts_itr, β, β, ν, ν)))) * (- Gp(s_itr, ν, β, χ, χ) - conj(Gp(Δts_itr, α, α, β, ν)) + conj(Gp(Δτs_itr, α, α, β, ν)) + conj(Gp(Δts_itr, β, β, β, ν)) + conj(Gp(s_itr, β, ν, ν, ν)) - conj(Gp(Δτs_itr, χ, χ, β, ν)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) - G(s_itr, β, β, χ, χ) + G(t_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(t_itr, χ, χ, χ, χ) - G(τ_itr, χ, χ, χ, χ) + G(s_itr, ν, ν, χ, χ) - G(t_itr, ν, ν, χ, χ) + conj(G(Δts_itr, α, α, β, β)) - conj(G(Δτs_itr, α, α, β, β)) + conj(G(t_itr, α, α, ν, ν)) - conj(G(τ_itr, α, α, ν, ν)) - conj(G(Δts_itr, α, α, ν, ν)) + conj(G(Δτs_itr, α, α, ν, ν)) - conj(G(Δts_itr, β, β, β, β)) + conj(G(s_itr, β, β, ν, ν)) - conj(G(t_itr, β, β, ν, ν)) + conj(G(Δts_itr, β, β, ν, ν)) + conj(G(Δτs_itr, χ, χ, β, β)) - conj(G(t_itr, χ, χ, ν, ν)) + conj(G(τ_itr, χ, χ, ν, ν)) - conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, χ, ν) * exp(((G(s_itr, χ, χ, χ, χ) - G(t_itr, χ, χ, χ, χ) - G(s_itr, ν, ν, χ, χ) + G(t_itr, ν, ν, χ, χ) - conj(G(s_itr, χ, χ, ν, ν)) + conj(G(t_itr, χ, χ, ν, ν)) + conj(G(s_itr, ν, ν, ν, ν)) - conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))
#         )
#     end

#     Base.@noinline function kernel_L0G1_RL(
#         s_itr::Int,
#         τ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((- hbar2 * conj(Gpp(Δtτ_itr, α, α, β, χ)) + hbar2 * conj(Gpp(Δtτ_itr, β, β, β, χ)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp(Δtτ_itr, α, α, β, χ)) + conj(Gp(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ))) * (- conj(Gp(t_itr, α, α, β, β)) + conj(Gp(Δtτ_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, χ, χ)) - conj(Gp(Δtτ_itr, α, α, χ, χ)) + conj(Gp(t_itr, β, β, β, β)) - conj(Gp(Δtτ_itr, β, β, β, β)) - conj(Gp(t_itr, β, β, χ, χ)) + conj(Gp(Δtτ_itr, β, β, χ, χ)))) * (- Gp(Δτs_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(t_itr, α, α, α, α) + G(τ_itr, α, α, α, α) - G(Δτs_itr, α, α, α, α) + G(s_itr, α, α, μ, μ) - G(τ_itr, α, α, μ, μ) + G(Δτs_itr, α, α, μ, μ) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) + G(Δτs_itr, χ, χ, α, α) - G(t_itr, χ, χ, μ, μ) + G(τ_itr, χ, χ, μ, μ) - G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(Δtτ_itr, α, α, β, β)) - conj(G(s_itr, α, α, χ, χ)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(t_itr, χ, χ, χ, χ)) - conj(G(τ_itr, χ, χ, χ, χ)) + conj(G(s_itr, μ, μ, χ, χ)) - conj(G(t_itr, μ, μ, χ, χ))) / (hbar2))) + (hbar2 * (conj(Gpp(Δtτ_itr, α, α, β, χ)) - conj(Gpp(Δtτ_itr, β, β, β, χ))) * (- Gp(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp(Δts_itr, β, β, α, μ) - Gp(Δτs_itr, β, β, α, μ) + Gp(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) + hbar2 * (- Gp(Δts_itr, α, α, α, α) - Gp(t_itr, α, α, μ, μ) + Gp(Δts_itr, α, α, μ, μ) + conj(Gp(Δtτ_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, χ, χ)) - conj(Gp(Δtτ_itr, α, α, χ, χ))) * Gpp(Δτs_itr, χ, β, α, μ) + (hbar2 * Gpp(Δτs_itr, χ, β, α, μ) + (- Gp(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp(Δts_itr, β, β, α, μ) - Gp(Δτs_itr, β, β, α, μ) + Gp(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) * (Gp(Δτs_itr, χ, β, α, α) + Gp(τ_itr, χ, β, μ, μ) - Gp(Δτs_itr, χ, β, μ, μ) + conj(Gp(Δtτ_itr, α, α, β, χ)) - conj(Gp(Δtτ_itr, β, β, β, χ)) - conj(Gp(τ_itr, β, χ, χ, χ)))) * (Gp(t_itr, α, α, α, α) - Gp(t_itr, β, β, α, α) + Gp(Δts_itr, β, β, α, α) + Gp(t_itr, β, β, μ, μ) - Gp(Δts_itr, β, β, μ, μ) - conj(Gp(t_itr, α, α, β, β)) + conj(Gp(t_itr, β, β, β, β)) - conj(Gp(Δtτ_itr, β, β, β, β)) - conj(Gp(t_itr, β, β, χ, χ)) + conj(Gp(Δtτ_itr, β, β, χ, χ))) + (hbar2 * Gpp(Δts_itr, α, α, α, μ) - hbar2 * Gpp(Δts_itr, β, β, α, μ) + (Gp(Δts_itr, α, α, α, α) + Gp(t_itr, α, α, μ, μ) - Gp(Δts_itr, α, α, μ, μ) - conj(Gp(Δtτ_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, χ, χ)) + conj(Gp(Δtτ_itr, α, α, χ, χ))) * (- Gp(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp(Δts_itr, β, β, α, μ) - Gp(Δτs_itr, β, β, α, μ) + Gp(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ)))) * (- Gp(Δτs_itr, χ, β, α, α) - Gp(τ_itr, χ, β, μ, μ) + Gp(Δτs_itr, χ, β, μ, μ) - conj(Gp(Δtτ_itr, α, α, β, χ)) + conj(Gp(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δts_itr, α, α, α, α) + G(s_itr, α, α, μ, μ) - G(t_itr, α, α, μ, μ) + G(Δts_itr, α, α, μ, μ) + G(Δts_itr, β, β, α, α) - G(Δτs_itr, β, β, α, α) + G(t_itr, β, β, μ, μ) - G(τ_itr, β, β, μ, μ) - G(Δts_itr, β, β, μ, μ) + G(Δτs_itr, β, β, μ, μ) + G(Δτs_itr, χ, χ, α, α) - G(t_itr, χ, χ, μ, μ) + G(τ_itr, χ, χ, μ, μ) - G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(Δtτ_itr, α, α, β, β)) - conj(G(s_itr, α, α, χ, χ)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(t_itr, χ, χ, χ, χ)) - conj(G(τ_itr, χ, χ, χ, χ)) + conj(G(s_itr, μ, μ, χ, χ)) - conj(G(t_itr, μ, μ, χ, χ))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, μ, χ) * exp(((- G(s_itr, χ, χ, μ, μ) + G(t_itr, χ, χ, μ, μ) + G(s_itr, μ, μ, μ, μ) - G(t_itr, μ, μ, μ, μ) + conj(G(s_itr, χ, χ, χ, χ)) - conj(G(t_itr, χ, χ, χ, χ)) - conj(G(s_itr, μ, μ, χ, χ)) + conj(G(t_itr, μ, μ, χ, χ))) / (hbar2)))
#         )
#     end

#     Base.@noinline function kernel_L0G1_RR(
#         s_itr::Int,
#         τ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((hbar2 * conj(Gpp(Δtτ_itr, α, α, β, χ)) - hbar2 * conj(Gpp(Δtτ_itr, β, β, β, χ)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp(Δtτ_itr, α, α, β, χ)) + conj(Gp(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ))) * (conj(Gp(t_itr, α, α, β, β)) - conj(Gp(Δtτ_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, χ, χ)) + conj(Gp(Δtτ_itr, α, α, χ, χ)) - conj(Gp(t_itr, β, β, β, β)) + conj(Gp(Δtτ_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, χ, χ)) - conj(Gp(Δtτ_itr, β, β, χ, χ)))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp(Δτs_itr, α, α, χ, ν)) + conj(Gp(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) - G(s_itr, χ, χ, α, α) + G(τ_itr, χ, χ, α, α) + G(s_itr, ν, ν, α, α) - G(t_itr, ν, ν, α, α) + conj(G(Δtτ_itr, α, α, β, β)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(τ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) + conj(G(Δτs_itr, α, α, χ, χ)) - conj(G(t_itr, α, α, ν, ν)) + conj(G(τ_itr, α, α, ν, ν)) - conj(G(Δτs_itr, α, α, ν, ν)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) - conj(G(Δτs_itr, χ, χ, χ, χ)) + conj(G(s_itr, χ, χ, ν, ν)) - conj(G(τ_itr, χ, χ, ν, ν)) + conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2))) + (hbar2 * (- conj(Gpp(Δtτ_itr, α, α, β, χ)) + conj(Gpp(Δtτ_itr, β, β, β, χ))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp(Δts_itr, α, α, χ, ν)) + conj(Gp(Δts_itr, β, β, χ, ν)) - conj(Gp(Δτs_itr, β, β, χ, ν)) + conj(Gp(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν))) + hbar2 * (- Gp(t_itr, β, β, α, α) + conj(Gp(Δtτ_itr, β, β, β, β)) + conj(Gp(Δts_itr, β, β, χ, χ)) - conj(Gp(Δtτ_itr, β, β, χ, χ)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp(Δts_itr, β, β, ν, ν))) * conj(Gpp(Δτs_itr, β, χ, χ, ν)) + (hbar2 * conj(Gpp(Δτs_itr, β, χ, χ, ν)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp(Δtτ_itr, α, α, β, χ)) + conj(Gp(Δtτ_itr, β, β, β, χ)) + conj(Gp(Δτs_itr, β, χ, χ, χ)) + conj(Gp(τ_itr, β, χ, ν, ν)) - conj(Gp(Δτs_itr, β, χ, ν, ν))) * (Gp(s_itr, ν, χ, α, α) + conj(Gp(Δts_itr, α, α, χ, ν)) - conj(Gp(Δts_itr, β, β, χ, ν)) + conj(Gp(Δτs_itr, β, β, χ, ν)) - conj(Gp(Δτs_itr, χ, χ, χ, ν)) - conj(Gp(s_itr, χ, ν, ν, ν)))) * (Gp(t_itr, β, β, α, α) + conj(Gp(t_itr, α, α, β, β)) - conj(Gp(Δtτ_itr, α, α, β, β)) - conj(Gp(Δts_itr, α, α, χ, χ)) + conj(Gp(Δtτ_itr, α, α, χ, χ)) - conj(Gp(t_itr, α, α, ν, ν)) + conj(Gp(Δts_itr, α, α, ν, ν)) - conj(Gp(t_itr, β, β, β, β))) + (- hbar2 * conj(Gpp(Δts_itr, α, α, χ, ν)) + hbar2 * conj(Gpp(Δts_itr, β, β, χ, ν)) + (Gp(t_itr, β, β, α, α) - conj(Gp(Δtτ_itr, β, β, β, β)) - conj(Gp(Δts_itr, β, β, χ, χ)) + conj(Gp(Δtτ_itr, β, β, χ, χ)) - conj(Gp(t_itr, β, β, ν, ν)) + conj(Gp(Δts_itr, β, β, ν, ν))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp(Δts_itr, α, α, χ, ν)) + conj(Gp(Δts_itr, β, β, χ, ν)) - conj(Gp(Δτs_itr, β, β, χ, ν)) + conj(Gp(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν)))) * (- Gp(τ_itr, χ, β, α, α) - conj(Gp(Δtτ_itr, α, α, β, χ)) + conj(Gp(Δtτ_itr, β, β, β, χ)) + conj(Gp(Δτs_itr, β, χ, χ, χ)) + conj(Gp(τ_itr, β, χ, ν, ν)) - conj(Gp(Δτs_itr, β, χ, ν, ν)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) - G(s_itr, χ, χ, α, α) + G(τ_itr, χ, χ, α, α) + G(s_itr, ν, ν, α, α) - G(t_itr, ν, ν, α, α) + conj(G(Δtτ_itr, α, α, β, β)) + conj(G(Δts_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δts_itr, α, α, ν, ν)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(Δts_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(Δτs_itr, β, β, χ, χ)) - conj(G(t_itr, β, β, ν, ν)) + conj(G(τ_itr, β, β, ν, ν)) + conj(G(Δts_itr, β, β, ν, ν)) - conj(G(Δτs_itr, β, β, ν, ν)) - conj(G(Δτs_itr, χ, χ, χ, χ)) + conj(G(s_itr, χ, χ, ν, ν)) - conj(G(τ_itr, χ, χ, ν, ν)) + conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, α, ν) * exp(((G(s_itr, α, α, α, α) - G(t_itr, α, α, α, α) - G(s_itr, ν, ν, α, α) + G(t_itr, ν, ν, α, α) - conj(G(s_itr, α, α, ν, ν)) + conj(G(t_itr, α, α, ν, ν)) + conj(G(s_itr, ν, ν, ν, ν)) - conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))
#         )
#     end


#     # ---------------------------------------------------------------------
#     # Markovianized inner-G1 variants of P L0 Q G1 Q L1 P.
#     #
#     # These are mechanically identical to the exact 2-time kernels above,
#     # except that relative-time derivative factors with arguments
#     # Δts_itr, Δtτ_itr, and Δτs_itr use
#     #
#     #     Gp  -> Gp_markovian_G1  = -im * Λ
#     #     Gpp -> Gpp_markovian_G1 = 0
#     #
#     # Absolute-time g/g′ factors and all g-exponential dressing factors are
#     # intentionally left unchanged.  Therefore this switch only Markovianizes
#     # the internal G1 derivative/correlation content, not the full outer
#     # P L0 Q ... Q L1 P memory kernel.
#     # ---------------------------------------------------------------------

#     Base.@noinline function kernel_L0G1_LL_markovian_G1(
#         s_itr::Int,
#         τ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((hbar2 * Gpp_markovian_G1(Δtτ_itr, α, α, α, χ) - hbar2 * Gpp_markovian_G1(Δtτ_itr, β, β, α, χ) + (- Gp_markovian_G1(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β))) * (- Gp(t_itr, α, α, α, α) + Gp_markovian_G1(Δtτ_itr, α, α, α, α) + Gp(t_itr, α, α, χ, χ) - Gp_markovian_G1(Δtτ_itr, α, α, χ, χ) + Gp(t_itr, β, β, α, α) - Gp_markovian_G1(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, χ, χ))) * (Gp_markovian_G1(Δτs_itr, β, β, χ, μ) - Gp_markovian_G1(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) + G(t_itr, β, β, χ, χ) - G(τ_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(Δτs_itr, β, β, χ, χ) - G(t_itr, β, β, μ, μ) + G(τ_itr, β, β, μ, μ) - G(Δτs_itr, β, β, μ, μ) - G(Δτs_itr, χ, χ, χ, χ) + G(s_itr, χ, χ, μ, μ) - G(τ_itr, χ, χ, μ, μ) + G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(s_itr, χ, χ, β, β)) + conj(G(τ_itr, χ, χ, β, β)) + conj(G(s_itr, μ, μ, β, β)) - conj(G(t_itr, μ, μ, β, β))) / (hbar2))) + (hbar2 * (- Gpp_markovian_G1(Δtτ_itr, α, α, α, χ) + Gpp_markovian_G1(Δtτ_itr, β, β, α, χ)) * (- Gp_markovian_G1(Δts_itr, α, α, χ, μ) + Gp_markovian_G1(Δτs_itr, α, α, χ, μ) + Gp_markovian_G1(Δts_itr, β, β, χ, μ) - Gp_markovian_G1(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β))) + hbar2 * (- Gp_markovian_G1(Δtτ_itr, β, β, α, α) - Gp_markovian_G1(Δts_itr, β, β, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, χ, χ) - Gp(t_itr, β, β, μ, μ) + Gp_markovian_G1(Δts_itr, β, β, μ, μ) + conj(Gp(t_itr, β, β, β, β))) * Gpp_markovian_G1(Δτs_itr, α, χ, χ, μ) + (hbar2 * Gpp_markovian_G1(Δτs_itr, α, χ, χ, μ) + (Gp_markovian_G1(Δtτ_itr, α, α, α, χ) + Gp_markovian_G1(Δτs_itr, α, χ, χ, χ) + Gp(τ_itr, α, χ, μ, μ) - Gp_markovian_G1(Δτs_itr, α, χ, μ, μ) - Gp_markovian_G1(Δtτ_itr, β, β, α, χ) - conj(Gp(τ_itr, χ, α, β, β))) * (- Gp_markovian_G1(Δts_itr, α, α, χ, μ) + Gp_markovian_G1(Δτs_itr, α, α, χ, μ) + Gp_markovian_G1(Δts_itr, β, β, χ, μ) - Gp_markovian_G1(Δτs_itr, χ, χ, χ, μ) - Gp(s_itr, χ, μ, μ, μ) + conj(Gp(s_itr, μ, χ, β, β)))) * (- Gp(t_itr, α, α, α, α) + Gp_markovian_G1(Δtτ_itr, α, α, α, α) + Gp_markovian_G1(Δts_itr, α, α, χ, χ) - Gp_markovian_G1(Δtτ_itr, α, α, χ, χ) + Gp(t_itr, α, α, μ, μ) - Gp_markovian_G1(Δts_itr, α, α, μ, μ) + Gp(t_itr, β, β, α, α) - conj(Gp(t_itr, β, β, β, β))) + (- hbar2 * Gpp_markovian_G1(Δts_itr, α, α, χ, μ) + hbar2 * Gpp_markovian_G1(Δts_itr, β, β, χ, μ) + (Gp_markovian_G1(Δts_itr, α, α, χ, μ) - Gp_markovian_G1(Δτs_itr, α, α, χ, μ) - Gp_markovian_G1(Δts_itr, β, β, χ, μ) + Gp_markovian_G1(Δτs_itr, χ, χ, χ, μ) + Gp(s_itr, χ, μ, μ, μ) - conj(Gp(s_itr, μ, χ, β, β))) * (- Gp_markovian_G1(Δtτ_itr, β, β, α, α) - Gp_markovian_G1(Δts_itr, β, β, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, χ, χ) - Gp(t_itr, β, β, μ, μ) + Gp_markovian_G1(Δts_itr, β, β, μ, μ) + conj(Gp(t_itr, β, β, β, β)))) * (- Gp_markovian_G1(Δtτ_itr, α, α, α, χ) - Gp_markovian_G1(Δτs_itr, α, χ, χ, χ) - Gp(τ_itr, α, χ, μ, μ) + Gp_markovian_G1(Δτs_itr, α, χ, μ, μ) + Gp_markovian_G1(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(Δts_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δτs_itr, α, α, χ, χ) - G(t_itr, α, α, μ, μ) + G(τ_itr, α, α, μ, μ) + G(Δts_itr, α, α, μ, μ) - G(Δτs_itr, α, α, μ, μ) + G(Δtτ_itr, β, β, α, α) + G(Δts_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) - G(Δts_itr, β, β, μ, μ) - G(Δτs_itr, χ, χ, χ, χ) + G(s_itr, χ, χ, μ, μ) - G(τ_itr, χ, χ, μ, μ) + G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(s_itr, χ, χ, β, β)) + conj(G(τ_itr, χ, χ, β, β)) + conj(G(s_itr, μ, μ, β, β)) - conj(G(t_itr, μ, μ, β, β))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, μ, β) * exp(((- G(s_itr, β, β, μ, μ) + G(t_itr, β, β, μ, μ) + G(s_itr, μ, μ, μ, μ) - G(t_itr, μ, μ, μ, μ) + conj(G(s_itr, β, β, β, β)) - conj(G(t_itr, β, β, β, β)) - conj(G(s_itr, μ, μ, β, β)) + conj(G(t_itr, μ, μ, β, β))) / (hbar2)))
#         )
#     end


#     Base.@noinline function kernel_L0G1_LR_markovian_G1(
#         s_itr::Int,
#         τ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((- hbar2 * Gpp_markovian_G1(Δtτ_itr, α, α, α, χ) + hbar2 * Gpp_markovian_G1(Δtτ_itr, β, β, α, χ) + (- Gp_markovian_G1(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, α, χ) + conj(Gp(τ_itr, χ, α, β, β))) * (Gp(t_itr, α, α, α, α) - Gp_markovian_G1(Δtτ_itr, α, α, α, α) - Gp(t_itr, α, α, χ, χ) + Gp_markovian_G1(Δtτ_itr, α, α, χ, χ) - Gp(t_itr, β, β, α, α) + Gp_markovian_G1(Δtτ_itr, β, β, α, α) + Gp(t_itr, β, β, χ, χ) - Gp_markovian_G1(Δtτ_itr, β, β, χ, χ))) * (- Gp(s_itr, ν, β, χ, χ) + conj(Gp_markovian_G1(Δτs_itr, β, β, β, ν)) + conj(Gp(s_itr, β, ν, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, χ, χ, β, ν))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) - G(s_itr, β, β, χ, χ) + G(t_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(t_itr, χ, χ, χ, χ) - G(τ_itr, χ, χ, χ, χ) + G(s_itr, ν, ν, χ, χ) - G(t_itr, ν, ν, χ, χ) + conj(G(t_itr, α, α, β, β)) - conj(G(τ_itr, α, α, β, β)) - conj(G(t_itr, β, β, β, β)) + conj(G(τ_itr, β, β, β, β)) - conj(G(Δτs_itr, β, β, β, β)) + conj(G(s_itr, β, β, ν, ν)) - conj(G(τ_itr, β, β, ν, ν)) + conj(G(Δτs_itr, β, β, ν, ν)) + conj(G(Δτs_itr, χ, χ, β, β)) - conj(G(t_itr, χ, χ, ν, ν)) + conj(G(τ_itr, χ, χ, ν, ν)) - conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2))) + (hbar2 * (conj(Gpp_markovian_G1(Δts_itr, α, α, β, ν)) - conj(Gpp_markovian_G1(Δts_itr, β, β, β, ν))) * (- Gp_markovian_G1(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, α, χ) + conj(Gp_markovian_G1(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, χ, α, ν, ν))) + hbar2 * (- Gp_markovian_G1(Δtτ_itr, α, α, α, α) - Gp(t_itr, α, α, χ, χ) + Gp_markovian_G1(Δtτ_itr, α, α, χ, χ) + conj(Gp_markovian_G1(Δts_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, ν, ν)) - conj(Gp_markovian_G1(Δts_itr, α, α, ν, ν))) * conj(Gpp_markovian_G1(Δτs_itr, χ, α, β, ν)) - hbar2 * (- Gp_markovian_G1(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, χ, χ) + conj(Gp_markovian_G1(Δts_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp_markovian_G1(Δts_itr, β, β, ν, ν))) * conj(Gpp_markovian_G1(Δτs_itr, χ, α, β, ν)) + (hbar2 * conj(Gpp_markovian_G1(Δτs_itr, χ, α, β, ν)) + (- Gp_markovian_G1(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, α, χ) + conj(Gp_markovian_G1(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, χ, α, ν, ν))) * (Gp(s_itr, ν, β, χ, χ) + conj(Gp_markovian_G1(Δts_itr, α, α, β, ν)) - conj(Gp_markovian_G1(Δτs_itr, α, α, β, ν)) - conj(Gp_markovian_G1(Δts_itr, β, β, β, ν)) - conj(Gp(s_itr, β, ν, ν, ν)) + conj(Gp_markovian_G1(Δτs_itr, χ, χ, β, ν)))) * (Gp(t_itr, α, α, α, α) - Gp(t_itr, β, β, α, α) - conj(Gp(t_itr, α, α, β, β)) + conj(Gp(t_itr, β, β, β, β))) + (hbar2 * Gpp_markovian_G1(Δtτ_itr, α, α, α, χ) - hbar2 * Gpp_markovian_G1(Δtτ_itr, β, β, α, χ) + (- Gp_markovian_G1(Δtτ_itr, α, α, α, χ) - Gp(τ_itr, α, χ, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, α, χ) + conj(Gp_markovian_G1(Δτs_itr, χ, α, β, β)) + conj(Gp(τ_itr, χ, α, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, χ, α, ν, ν))) * (Gp_markovian_G1(Δtτ_itr, α, α, α, α) + Gp(t_itr, α, α, χ, χ) - Gp_markovian_G1(Δtτ_itr, α, α, χ, χ) - Gp_markovian_G1(Δtτ_itr, β, β, α, α) - Gp(t_itr, β, β, χ, χ) + Gp_markovian_G1(Δtτ_itr, β, β, χ, χ) - conj(Gp_markovian_G1(Δts_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, ν, ν)) + conj(Gp_markovian_G1(Δts_itr, α, α, ν, ν)) + conj(Gp_markovian_G1(Δts_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp_markovian_G1(Δts_itr, β, β, ν, ν)))) * (- Gp(s_itr, ν, β, χ, χ) - conj(Gp_markovian_G1(Δts_itr, α, α, β, ν)) + conj(Gp_markovian_G1(Δτs_itr, α, α, β, ν)) + conj(Gp_markovian_G1(Δts_itr, β, β, β, ν)) + conj(Gp(s_itr, β, ν, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, χ, χ, β, ν)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δτs_itr) - 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δtτ_itr, α, α, α, α) - G(t_itr, α, α, χ, χ) + G(τ_itr, α, α, χ, χ) + G(Δtτ_itr, α, α, χ, χ) + G(Δtτ_itr, β, β, α, α) - G(s_itr, β, β, χ, χ) + G(t_itr, β, β, χ, χ) - G(Δtτ_itr, β, β, χ, χ) + G(t_itr, χ, χ, χ, χ) - G(τ_itr, χ, χ, χ, χ) + G(s_itr, ν, ν, χ, χ) - G(t_itr, ν, ν, χ, χ) + conj(G(Δts_itr, α, α, β, β)) - conj(G(Δτs_itr, α, α, β, β)) + conj(G(t_itr, α, α, ν, ν)) - conj(G(τ_itr, α, α, ν, ν)) - conj(G(Δts_itr, α, α, ν, ν)) + conj(G(Δτs_itr, α, α, ν, ν)) - conj(G(Δts_itr, β, β, β, β)) + conj(G(s_itr, β, β, ν, ν)) - conj(G(t_itr, β, β, ν, ν)) + conj(G(Δts_itr, β, β, ν, ν)) + conj(G(Δτs_itr, χ, χ, β, β)) - conj(G(t_itr, χ, χ, ν, ν)) + conj(G(τ_itr, χ, χ, ν, ν)) - conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, χ, ν) * exp(((G(s_itr, χ, χ, χ, χ) - G(t_itr, χ, χ, χ, χ) - G(s_itr, ν, ν, χ, χ) + G(t_itr, ν, ν, χ, χ) - conj(G(s_itr, χ, χ, ν, ν)) + conj(G(t_itr, χ, χ, ν, ν)) + conj(G(s_itr, ν, ν, ν, ν)) - conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))
#         )
#     end


#     Base.@noinline function kernel_L0G1_RL_markovian_G1(
#         s_itr::Int,
#         τ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         μ::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((- hbar2 * conj(Gpp_markovian_G1(Δtτ_itr, α, α, β, χ)) + hbar2 * conj(Gpp_markovian_G1(Δtτ_itr, β, β, β, χ)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ))) * (- conj(Gp(t_itr, α, α, β, β)) + conj(Gp_markovian_G1(Δtτ_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, χ, χ)) - conj(Gp_markovian_G1(Δtτ_itr, α, α, χ, χ)) + conj(Gp(t_itr, β, β, β, β)) - conj(Gp_markovian_G1(Δtτ_itr, β, β, β, β)) - conj(Gp(t_itr, β, β, χ, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, χ, χ)))) * (- Gp_markovian_G1(Δτs_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp_markovian_G1(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(t_itr, α, α, α, α) + G(τ_itr, α, α, α, α) - G(Δτs_itr, α, α, α, α) + G(s_itr, α, α, μ, μ) - G(τ_itr, α, α, μ, μ) + G(Δτs_itr, α, α, μ, μ) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) + G(Δτs_itr, χ, χ, α, α) - G(t_itr, χ, χ, μ, μ) + G(τ_itr, χ, χ, μ, μ) - G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(Δtτ_itr, α, α, β, β)) - conj(G(s_itr, α, α, χ, χ)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(t_itr, χ, χ, χ, χ)) - conj(G(τ_itr, χ, χ, χ, χ)) + conj(G(s_itr, μ, μ, χ, χ)) - conj(G(t_itr, μ, μ, χ, χ))) / (hbar2))) + (hbar2 * (conj(Gpp_markovian_G1(Δtτ_itr, α, α, β, χ)) - conj(Gpp_markovian_G1(Δtτ_itr, β, β, β, χ))) * (- Gp_markovian_G1(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp_markovian_G1(Δts_itr, β, β, α, μ) - Gp_markovian_G1(Δτs_itr, β, β, α, μ) + Gp_markovian_G1(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) + hbar2 * (- Gp_markovian_G1(Δts_itr, α, α, α, α) - Gp(t_itr, α, α, μ, μ) + Gp_markovian_G1(Δts_itr, α, α, μ, μ) + conj(Gp_markovian_G1(Δtτ_itr, α, α, β, β)) + conj(Gp(t_itr, α, α, χ, χ)) - conj(Gp_markovian_G1(Δtτ_itr, α, α, χ, χ))) * Gpp_markovian_G1(Δτs_itr, χ, β, α, μ) + (hbar2 * Gpp_markovian_G1(Δτs_itr, χ, β, α, μ) + (- Gp_markovian_G1(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp_markovian_G1(Δts_itr, β, β, α, μ) - Gp_markovian_G1(Δτs_itr, β, β, α, μ) + Gp_markovian_G1(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ))) * (Gp_markovian_G1(Δτs_itr, χ, β, α, α) + Gp(τ_itr, χ, β, μ, μ) - Gp_markovian_G1(Δτs_itr, χ, β, μ, μ) + conj(Gp_markovian_G1(Δtτ_itr, α, α, β, χ)) - conj(Gp_markovian_G1(Δtτ_itr, β, β, β, χ)) - conj(Gp(τ_itr, β, χ, χ, χ)))) * (Gp(t_itr, α, α, α, α) - Gp(t_itr, β, β, α, α) + Gp_markovian_G1(Δts_itr, β, β, α, α) + Gp(t_itr, β, β, μ, μ) - Gp_markovian_G1(Δts_itr, β, β, μ, μ) - conj(Gp(t_itr, α, α, β, β)) + conj(Gp(t_itr, β, β, β, β)) - conj(Gp_markovian_G1(Δtτ_itr, β, β, β, β)) - conj(Gp(t_itr, β, β, χ, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, χ, χ))) + (hbar2 * Gpp_markovian_G1(Δts_itr, α, α, α, μ) - hbar2 * Gpp_markovian_G1(Δts_itr, β, β, α, μ) + (Gp_markovian_G1(Δts_itr, α, α, α, α) + Gp(t_itr, α, α, μ, μ) - Gp_markovian_G1(Δts_itr, α, α, μ, μ) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, χ, χ)) + conj(Gp_markovian_G1(Δtτ_itr, α, α, χ, χ))) * (- Gp_markovian_G1(Δts_itr, α, α, α, μ) - Gp(s_itr, α, μ, μ, μ) + Gp_markovian_G1(Δts_itr, β, β, α, μ) - Gp_markovian_G1(Δτs_itr, β, β, α, μ) + Gp_markovian_G1(Δτs_itr, χ, χ, α, μ) + conj(Gp(s_itr, μ, α, χ, χ)))) * (- Gp_markovian_G1(Δτs_itr, χ, β, α, α) - Gp(τ_itr, χ, β, μ, μ) + Gp_markovian_G1(Δτs_itr, χ, β, μ, μ) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) - G(Δts_itr, α, α, α, α) + G(s_itr, α, α, μ, μ) - G(t_itr, α, α, μ, μ) + G(Δts_itr, α, α, μ, μ) + G(Δts_itr, β, β, α, α) - G(Δτs_itr, β, β, α, α) + G(t_itr, β, β, μ, μ) - G(τ_itr, β, β, μ, μ) - G(Δts_itr, β, β, μ, μ) + G(Δτs_itr, β, β, μ, μ) + G(Δτs_itr, χ, χ, α, α) - G(t_itr, χ, χ, μ, μ) + G(τ_itr, χ, χ, μ, μ) - G(Δτs_itr, χ, χ, μ, μ) - G(s_itr, μ, μ, μ, μ) + G(t_itr, μ, μ, μ, μ) + conj(G(Δtτ_itr, α, α, β, β)) - conj(G(s_itr, α, α, χ, χ)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(t_itr, χ, χ, χ, χ)) - conj(G(τ_itr, χ, χ, χ, χ)) + conj(G(s_itr, μ, μ, χ, χ)) - conj(G(t_itr, μ, μ, χ, χ))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, μ, χ) * exp(((- G(s_itr, χ, χ, μ, μ) + G(t_itr, χ, χ, μ, μ) + G(s_itr, μ, μ, μ, μ) - G(t_itr, μ, μ, μ, μ) + conj(G(s_itr, χ, χ, χ, χ)) - conj(G(t_itr, χ, χ, χ, χ)) - conj(G(s_itr, μ, μ, χ, χ)) + conj(G(t_itr, μ, μ, χ, χ))) / (hbar2)))
#         )
#     end


#     Base.@noinline function kernel_L0G1_RR_markovian_G1(
#         s_itr::Int,
#         τ_itr::Int,
#         t_itr::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#         χ::Int,
#         ν::Int,
#     )
#         Δts_itr = t_itr - s_itr + 1
#         Δtτ_itr = t_itr - τ_itr + 1
#         Δτs_itr = τ_itr - s_itr + 1

#         return (
#             1 * ((1) / (hbar6)) * ((hbar2 * conj(Gpp_markovian_G1(Δtτ_itr, α, α, β, χ)) - hbar2 * conj(Gpp_markovian_G1(Δtτ_itr, β, β, β, χ)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, χ)) + conj(Gp(τ_itr, β, χ, χ, χ))) * (conj(Gp(t_itr, α, α, β, β)) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, β)) - conj(Gp(t_itr, α, α, χ, χ)) + conj(Gp_markovian_G1(Δtτ_itr, α, α, χ, χ)) - conj(Gp(t_itr, β, β, β, β)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, β)) + conj(Gp(t_itr, β, β, χ, χ)) - conj(Gp_markovian_G1(Δtτ_itr, β, β, χ, χ)))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp_markovian_G1(Δτs_itr, α, α, χ, ν)) + conj(Gp_markovian_G1(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) - G(s_itr, χ, χ, α, α) + G(τ_itr, χ, χ, α, α) + G(s_itr, ν, ν, α, α) - G(t_itr, ν, ν, α, α) + conj(G(Δtτ_itr, α, α, β, β)) + conj(G(t_itr, α, α, χ, χ)) - conj(G(τ_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) + conj(G(Δτs_itr, α, α, χ, χ)) - conj(G(t_itr, α, α, ν, ν)) + conj(G(τ_itr, α, α, ν, ν)) - conj(G(Δτs_itr, α, α, ν, ν)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(t_itr, β, β, χ, χ)) + conj(G(τ_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) - conj(G(Δτs_itr, χ, χ, χ, χ)) + conj(G(s_itr, χ, χ, ν, ν)) - conj(G(τ_itr, χ, χ, ν, ν)) + conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2))) + (hbar2 * (- conj(Gpp_markovian_G1(Δtτ_itr, α, α, β, χ)) + conj(Gpp_markovian_G1(Δtτ_itr, β, β, β, χ))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp_markovian_G1(Δts_itr, α, α, χ, ν)) + conj(Gp_markovian_G1(Δts_itr, β, β, χ, ν)) - conj(Gp_markovian_G1(Δτs_itr, β, β, χ, ν)) + conj(Gp_markovian_G1(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν))) + hbar2 * (- Gp(t_itr, β, β, α, α) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, β)) + conj(Gp_markovian_G1(Δts_itr, β, β, χ, χ)) - conj(Gp_markovian_G1(Δtτ_itr, β, β, χ, χ)) + conj(Gp(t_itr, β, β, ν, ν)) - conj(Gp_markovian_G1(Δts_itr, β, β, ν, ν))) * conj(Gpp_markovian_G1(Δτs_itr, β, χ, χ, ν)) + (hbar2 * conj(Gpp_markovian_G1(Δτs_itr, β, χ, χ, ν)) + (- Gp(τ_itr, χ, β, α, α) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, χ)) + conj(Gp_markovian_G1(Δτs_itr, β, χ, χ, χ)) + conj(Gp(τ_itr, β, χ, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, β, χ, ν, ν))) * (Gp(s_itr, ν, χ, α, α) + conj(Gp_markovian_G1(Δts_itr, α, α, χ, ν)) - conj(Gp_markovian_G1(Δts_itr, β, β, χ, ν)) + conj(Gp_markovian_G1(Δτs_itr, β, β, χ, ν)) - conj(Gp_markovian_G1(Δτs_itr, χ, χ, χ, ν)) - conj(Gp(s_itr, χ, ν, ν, ν)))) * (Gp(t_itr, β, β, α, α) + conj(Gp(t_itr, α, α, β, β)) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, β)) - conj(Gp_markovian_G1(Δts_itr, α, α, χ, χ)) + conj(Gp_markovian_G1(Δtτ_itr, α, α, χ, χ)) - conj(Gp(t_itr, α, α, ν, ν)) + conj(Gp_markovian_G1(Δts_itr, α, α, ν, ν)) - conj(Gp(t_itr, β, β, β, β))) + (- hbar2 * conj(Gpp_markovian_G1(Δts_itr, α, α, χ, ν)) + hbar2 * conj(Gpp_markovian_G1(Δts_itr, β, β, χ, ν)) + (Gp(t_itr, β, β, α, α) - conj(Gp_markovian_G1(Δtτ_itr, β, β, β, β)) - conj(Gp_markovian_G1(Δts_itr, β, β, χ, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, χ, χ)) - conj(Gp(t_itr, β, β, ν, ν)) + conj(Gp_markovian_G1(Δts_itr, β, β, ν, ν))) * (- Gp(s_itr, ν, χ, α, α) - conj(Gp_markovian_G1(Δts_itr, α, α, χ, ν)) + conj(Gp_markovian_G1(Δts_itr, β, β, χ, ν)) - conj(Gp_markovian_G1(Δτs_itr, β, β, χ, ν)) + conj(Gp_markovian_G1(Δτs_itr, χ, χ, χ, ν)) + conj(Gp(s_itr, χ, ν, ν, ν)))) * (- Gp(τ_itr, χ, β, α, α) - conj(Gp_markovian_G1(Δtτ_itr, α, α, β, χ)) + conj(Gp_markovian_G1(Δtτ_itr, β, β, β, χ)) + conj(Gp_markovian_G1(Δτs_itr, β, χ, χ, χ)) + conj(Gp(τ_itr, β, χ, ν, ν)) - conj(Gp_markovian_G1(Δτs_itr, β, χ, ν, ν)))) * exp(((- 1.0im * hbar_f * ϵ[α] * Δtime(Δtτ_itr) - 1.0im * hbar_f * ϵ[α] * Δtime(Δτs_itr) + 1.0im * hbar_f * ϵ[β] * Δtime(Δtτ_itr) + 1.0im * hbar_f * ϵ[χ] * Δtime(Δτs_itr) + G(t_itr, β, β, α, α) - G(τ_itr, β, β, α, α) - G(s_itr, χ, χ, α, α) + G(τ_itr, χ, χ, α, α) + G(s_itr, ν, ν, α, α) - G(t_itr, ν, ν, α, α) + conj(G(Δtτ_itr, α, α, β, β)) + conj(G(Δts_itr, α, α, χ, χ)) - conj(G(Δtτ_itr, α, α, χ, χ)) - conj(G(Δts_itr, α, α, ν, ν)) - conj(G(Δtτ_itr, β, β, β, β)) - conj(G(Δts_itr, β, β, χ, χ)) + conj(G(Δtτ_itr, β, β, χ, χ)) + conj(G(Δτs_itr, β, β, χ, χ)) - conj(G(t_itr, β, β, ν, ν)) + conj(G(τ_itr, β, β, ν, ν)) + conj(G(Δts_itr, β, β, ν, ν)) - conj(G(Δτs_itr, β, β, ν, ν)) - conj(G(Δτs_itr, χ, χ, χ, χ)) + conj(G(s_itr, χ, χ, ν, ν)) - conj(G(τ_itr, χ, χ, ν, ν)) + conj(G(Δτs_itr, χ, χ, ν, ν)) - conj(G(s_itr, ν, ν, ν, ν)) + conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))) * σ_mem(σ_t, s_itr, t_itr, α, ν) * exp(((G(s_itr, α, α, α, α) - G(t_itr, α, α, α, α) - G(s_itr, ν, ν, α, α) + G(t_itr, ν, ν, α, α) - conj(G(s_itr, α, α, ν, ν)) + conj(G(t_itr, α, α, ν, ν)) + conj(G(s_itr, ν, ν, ν, ν)) - conj(G(t_itr, ν, ν, ν, ν))) / (hbar2)))
#         )
#     end


#     # Evaluate the four PL0QG1QL1P branch kernels at a single tau point.
#     # This helper is used by both the original nested quadrature and the
#     # Option-B collapsed-tau approximation below.
#     Base.@noinline function eval_L0G1_tau_kernel(
#         s_coord::Int,
#         τ_coord::Int,
#         t_eval_coord::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#     )
#         τ_kernel = 0.0 + 0.0im

#         # P L0 Q G1 Q L1 P, LL: input (μ, β), intermediate χ.
#         for χ in 1:n_sys
#             χ == α && continue
#             for μ in 1:n_sys
#                 μ == χ && continue
#                 if is_secular_pair(α, β, μ, β)
#                     τ_kernel += (
#                         effective_markovianize_PL0QG1QL1P_G1 ?
#                         kernel_L0G1_LL_markovian_G1(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, μ) :
#                         kernel_L0G1_LL(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, μ)
#                     )
#                 end
#             end
#         end

#         # P L0 Q G1 Q L1 P, RR: input (α, ν), intermediate χ.
#         for χ in 1:n_sys
#             χ == β && continue
#             for ν in 1:n_sys
#                 ν == χ && continue
#                 if is_secular_pair(α, β, α, ν)
#                     τ_kernel += (
#                         effective_markovianize_PL0QG1QL1P_G1 ?
#                         kernel_L0G1_RR_markovian_G1(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, ν) :
#                         kernel_L0G1_RR(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, ν)
#                     )
#                 end
#             end
#         end

#         # P L0 Q G1 Q L1 P, RL: input (μ, χ), left input μ, right intermediate χ.
#         for μ in 1:n_sys
#             μ == α && continue
#             for χ in 1:n_sys
#                 χ == β && continue
#                 if is_secular_pair(α, β, μ, χ)
#                     τ_kernel += (
#                         effective_markovianize_PL0QG1QL1P_G1 ?
#                         kernel_L0G1_RL_markovian_G1(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, μ) :
#                         kernel_L0G1_RL(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, μ)
#                     )
#                 end
#             end
#         end

#         # P L0 Q G1 Q L1 P, LR: input (χ, ν), left intermediate χ, right input ν.
#         for χ in 1:n_sys
#             χ == α && continue
#             for ν in 1:n_sys
#                 ν == β && continue
#                 if is_secular_pair(α, β, χ, ν)
#                     τ_kernel += (
#                         effective_markovianize_PL0QG1QL1P_G1 ?
#                         kernel_L0G1_LR_markovian_G1(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, ν) :
#                         kernel_L0G1_LR(s_coord, τ_coord, t_eval_coord, σ_t, α, β, χ, ν)
#                     )
#                 end
#             end
#         end

#         return τ_kernel
#     end


#     Base.@noinline function calc__rhs_element!(
#         rhs_mat::AbstractMatrix,
#         curr_itr::Int,
#         t_eval_coord::Int,
#         σ_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#     )
#         if use_population_closure && α != β
#             rhs_mat[α, β] = 0.0 + 0.0im
#             return nothing
#         end

#         # -------------------------------------------------------------
#         # Internal transported coordinate c-update.
#         # The RHS is local normalized-P_T dynamics plus
#         # P_T L1 Q_T G0 Q_T L1 P_T.  The physical σ readout is
#         # never fed back into this RHS.
#         # -------------------------------------------------------------
#         rhs = 0.0 + 0.0im

#         # Normalized tau^T leaves a scalar local L0/line-shape block in the
#         # internal c-equation.
#         if include_PL0P_local
#             rhs += local_PT_diag(t_eval_coord, α, β) * σ_local(σ_t, α, β)
#         end

#         if include_PL1P_local
#             for μ in 1:n_sys
#                 μ == α && continue
#                 if is_secular_pair(α, β, μ, β)
#                     rhs += local_PT_L(t_eval_coord, α, β, μ) * σ_local(σ_t, μ, β)
#                 end
#             end

#             for ν in 1:n_sys
#                 ν == β && continue
#                 if is_secular_pair(α, β, α, ν)
#                     rhs += local_PT_R(t_eval_coord, α, β, ν) * σ_local(σ_t, α, ν)
#                 end
#             end
#         end

#         if t_eval_coord > 1
#             integral = 0.0 + 0.0im
#             n_s_nodes = n_outer_nodes(curr_itr, t_eval_coord)

#             for s_node in 1:n_s_nodes
#                 s_coord = outer_node_coord(s_node, curr_itr, t_eval_coord)
#                 Δ_coord = t_eval_coord - s_coord + 1
#                 w_int = ∫weight_to_t(s_node, curr_itr, t_eval_coord)
#                 w_int == 0.0 && continue

#                 kernel = 0.0 + 0.0im

#                 if include_P_L0plusL1_QG0QL1P
#                     # P_T L1 Q_T G0 Q_T L1 P_T memory for internal c.
#                     # Despite the historical flag name, this block now contains
#                     # only the PL1QG0QL1P contribution generated from the
#                     # Jupyter PT c-update formulas.  Both L1 insertions keep
#                     # their off-diagonal constraints.

#                     # LL-like branch:
#                     #   first insertion α -> χ, second insertion χ -> μ,
#                     #   input (μ, β) -> output (α, β).
#                     for χ in 1:n_sys
#                         χ == α && continue
#                         for μ in 1:n_sys
#                             μ == χ && continue
#                             if is_secular_pair(α, β, μ, β)
#                                 kernel += kernel_LL(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, χ, μ)
#                             end
#                         end
#                     end

#                     # LR-like branch:
#                     #   first insertion α -> χ, second insertion ν -> β,
#                     #   input (χ, ν) -> output (α, β).
#                     for χ in 1:n_sys
#                         χ == α && continue
#                         for ν in 1:n_sys
#                             ν == β && continue
#                             if is_secular_pair(α, β, χ, ν)
#                                 kernel += kernel_LR(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, χ, ν)
#                             end
#                         end
#                     end

#                     # RL-like branch:
#                     #   first insertion χ -> β, second insertion α -> μ,
#                     #   input (μ, χ) -> output (α, β).
#                     for χ in 1:n_sys
#                         χ == β && continue
#                         for μ in 1:n_sys
#                             μ == α && continue
#                             if is_secular_pair(α, β, μ, χ)
#                                 kernel += kernel_RL(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, χ, μ)
#                             end
#                         end
#                     end

#                     # RR-like branch:
#                     #   first insertion χ -> β, second insertion ν -> χ,
#                     #   input (α, ν) -> output (α, β).
#                     for χ in 1:n_sys
#                         χ == β && continue
#                         for ν in 1:n_sys
#                             ν == χ && continue
#                             if is_secular_pair(α, β, α, ν)
#                                 kernel += kernel_RR(s_coord, Δ_coord, t_eval_coord, σ_t, α, β, χ, ν)
#                             end
#                         end
#                     end
#                 end

#                 if include_PL0QG1QL1P
#                     inner_kernel = 0.0 + 0.0im

#                     if collapse_tau_PL0QG1QL1P_G1
#                         # Option B: collapse the inner tau integral while keeping
#                         # the outer s-memory integral.  In grid coordinates,
#                         # physical interval length is 0.5*(q_t - q_s)*Δt in doubled q-grid.
#                         # This is a midpoint effective-kernel approximation:
#                         #   ∫_s^t dτ K(t,τ,s) ≈ (t-s) K(t,nearest_q((t+s)/2),s).
#                         τ_interval = 0.5 * Float64(t_eval_coord - s_coord) * Δt
#                         if τ_interval > 0.0
#                             τ_mid = min(max(div(t_eval_coord + s_coord + 1, 2), 1), const_Q_MAX)
#                             inner_kernel += τ_interval * eval_L0G1_tau_kernel(
#                                 s_coord,
#                                 τ_mid,
#                                 t_eval_coord,
#                                 σ_t,
#                                 α,
#                                 β,
#                             )
#                         end
#                     else
#                         n_τ_nodes = n_inner_nodes(s_coord, t_eval_coord)
#                         for τ_node in 1:n_τ_nodes
#                             τ_coord = inner_node_coord(τ_node, s_coord, t_eval_coord)
#                             w_τ = ∫weight_between_coord(τ_node, s_coord, t_eval_coord)
#                             w_τ == 0.0 && continue

#                             inner_kernel += w_τ * eval_L0G1_tau_kernel(
#                                 s_coord,
#                                 τ_coord,
#                                 t_eval_coord,
#                                 σ_t,
#                                 α,
#                                 β,
#                             )
#                         end
#                     end

#                     kernel += inner_kernel
#                 end

#                 integral += w_int * kernel
#             end

#             rhs += integral
#         end

#         rhs_mat[α, β] = rhs
#         return nothing
#     end

#     @inline function calc__rhs!(
#         rhs_mat::AbstractMatrix,
#         curr_itr::Int,
#         t_eval_coord::Int,
#         σ_t::AbstractMatrix,
#     )
#         fill!(rhs_mat, 0.0 + 0.0im)

#         n_components = n_sys * n_sys
#         heavy_memory = include_P_L0plusL1_QG0QL1P || include_PL0QG1QL1P
#         min_thread_components = max(16, 4 * Threads.nthreads())
#         if use_threads && Threads.nthreads() > 1 && (heavy_memory || n_components >= min_thread_components)
#             Threads.@threads for linear_idx in 1:n_components
#                 @inbounds begin
#                     α = ((linear_idx - 1) % n_sys) + 1
#                     β = ((linear_idx - 1) ÷ n_sys) + 1
#                     calc__rhs_element!(rhs_mat, curr_itr, t_eval_coord, σ_t, α, β)
#                 end
#             end
#         else
#             @inbounds for β in 1:n_sys, α in 1:n_sys
#                 calc__rhs_element!(rhs_mat, curr_itr, t_eval_coord, σ_t, α, β)
#             end
#         end

#         return rhs_mat
#     end

#     Base.@noinline function readout__σ_element!(
#         σ_out::AbstractMatrix,
#         curr_itr::Int,
#         t_eval_coord::Int,
#         c_t::AbstractMatrix,
#         α::Int,
#         β::Int,
#     )
#         if use_population_closure && α != β
#             σ_out[α, β] = 0.0 + 0.0im
#             return nothing
#         end

#         value = c_local(c_t, α, β)

#         if include_QT_G0_readout && t_eval_coord > 1
#             integral = 0.0 + 0.0im
#             n_s_nodes = n_outer_nodes(curr_itr, t_eval_coord)

#             for s_node in 1:n_s_nodes
#                 s_coord = outer_node_coord(s_node, curr_itr, t_eval_coord)
#                 Δ_coord = t_eval_coord - s_coord + 1
#                 w_int = ∫weight_to_t(s_node, curr_itr, t_eval_coord)
#                 w_int == 0.0 && continue

#                 kernel = 0.0 + 0.0im

#                 # Left source branch: c_{μβ}(s) -> σ_{αβ}(t).
#                 for μ in 1:n_sys
#                     μ == α && continue
#                     if is_secular_pair(α, β, μ, β)
#                         kernel += readout_Q_L(s_coord, Δ_coord, t_eval_coord, c_t, α, β, μ)
#                     end
#                 end

#                 # Right source branch: c_{αν}(s) -> σ_{αβ}(t).
#                 for ν in 1:n_sys
#                     ν == β && continue
#                     if is_secular_pair(α, β, α, ν)
#                         kernel += readout_Q_R(s_coord, Δ_coord, t_eval_coord, c_t, α, β, ν)
#                     end
#                 end

#                 integral += w_int * kernel
#             end

#             value += integral
#         end

#         σ_out[α, β] = value
#         return nothing
#     end

#     @inline function readout__σ!(
#         σ_out::AbstractMatrix,
#         curr_itr::Int,
#         t_eval_coord::Int,
#         c_t::AbstractMatrix,
#     )
#         fill!(σ_out, 0.0 + 0.0im)

#         n_components = n_sys * n_sys
#         heavy_readout = include_QT_G0_readout && (include_P_L0plusL1_QG0QL1P || include_PL1P_local || include_PL0P_local)
#         min_thread_components = max(16, 4 * Threads.nthreads())
#         if use_threads && Threads.nthreads() > 1 && (heavy_readout || n_components >= min_thread_components)
#             Threads.@threads for linear_idx in 1:n_components
#                 @inbounds begin
#                     α = ((linear_idx - 1) % n_sys) + 1
#                     β = ((linear_idx - 1) ÷ n_sys) + 1
#                     readout__σ_element!(σ_out, curr_itr, t_eval_coord, c_t, α, β)
#                 end
#             end
#         else
#             @inbounds for β in 1:n_sys, α in 1:n_sys
#                 readout__σ_element!(σ_out, curr_itr, t_eval_coord, c_t, α, β)
#             end
#         end

#         return σ_out
#     end

#     @inline function enforce_hermiticity!(σ_next::AbstractMatrix)
#         for i in 1:n_sys
#             σ_next[i, i] = real(σ_next[i, i]) + 0.0im
#         end

#         for i in 1:(n_sys - 1)
#             for j in (i + 1):n_sys
#                 # Do not name this local scalar `c`: the outer PT internal
#                 # coordinate array is also named `c`, and nested Julia
#                 # closures can capture/rebind it.  If that happens, c_mem
#                 # later sees `c::ComplexF64` instead of the 3D c-history array.
#                 c_sym = 0.5 * (σ_next[i, j] + conj(σ_next[j, i]))
#                 σ_next[i, j] = c_sym
#                 σ_next[j, i] = conj(c_sym)
#             end
#         end

#         return σ_next
#     end

#     @inline function enforce_population_closure!(σ_next::AbstractMatrix)
#         for j in 1:n_sys, i in 1:n_sys
#             if i != j
#                 σ_next[i, j] = 0.0 + 0.0im
#             else
#                 σ_next[i, i] = real(σ_next[i, i]) + 0.0im
#             end
#         end
#         return σ_next
#     end

#     # include_PL0QG1QL1P is rejected above in this G0-only PT c-update core.

#     c_stage = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k2      = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k3      = Matrix{ComplexF64}(undef, n_sys, n_sys)
#     k4      = Matrix{ComplexF64}(undef, n_sys, n_sys)

#     # Ensure the physical σ buffer is consistent with the current internal c
#     # state before advancing.  This matters when start_itr > 1 or when the
#     # caller inspects context.σ immediately after initialization.
#     σ_start = @view σ[:, :, start_itr]
#     c_start = @view c[:, :, start_itr]
#     readout__σ!(σ_start, start_itr, q_full(start_itr), c_start)

#     @inbounds for curr_itr in start_itr:(n_itr - 1)
#         if verbose && (curr_itr == start_itr || curr_itr == n_itr - 1 || ((curr_itr - start_itr) % verbose_every_f == 0))
#             @printf(
#                 stderr,
#                 "Current iteration: %6d / %6d  method=%s  threads=%d  use_threads=%s  secular=%s  QT_G0_readout=%s  PL0QG1_markovian_G1=%s  PL0QG1_collapse_tau=%s\n",
#                 curr_itr,
#                 n_itr,
#                 String(method_sym),
#                 Threads.nthreads(),
#                 string(use_threads),
#                 string(use_secular),
#                 string(include_QT_G0_readout),
#                 string(effective_markovianize_PL0QG1QL1P_G1),
#                 string(collapse_tau_PL0QG1QL1P_G1),
#             )
#         end

#         c_t    = @view c[:, :, curr_itr]
#         c_next = @view c[:, :, curr_itr + 1]
#         σ_next = @view σ[:, :, curr_itr + 1]
#         k1     = @view c′[:, :, curr_itr]

#         if method_sym == :euler
#             calc__rhs!(k1, curr_itr, q_full(curr_itr), c_t)
#             @. c_next = c_t + Δt * k1

#         elseif method_sym == :rk2
#             calc__rhs!(k1, curr_itr, q_full(curr_itr), c_t)
#             @. c_stage = c_t + 0.5 * Δt * k1
#             use_population_closure && enforce_population_closure!(c_stage)
#             calc__rhs!(k2, curr_itr, q_half_after(curr_itr), c_stage)
#             @. c_next = c_t + Δt * k2

#         elseif method_sym == :rk4
#             # print("시작0\n")
#             calc__rhs!(k1, curr_itr, q_full(curr_itr), c_t)

#             # print("시작1\n")
#             @. c_stage = c_t + 0.5 * Δt * k1
#             use_population_closure && enforce_population_closure!(c_stage)
#             calc__rhs!(k2, curr_itr, q_half_after(curr_itr), c_stage)

#             # print("시작2\n")
#             @. c_stage = c_t + 0.5 * Δt * k2
#             use_population_closure && enforce_population_closure!(c_stage)
#             calc__rhs!(k3, curr_itr, q_half_after(curr_itr), c_stage)

#             # print("시작3\n")
#             @. c_stage = c_t + Δt * k3
#             use_population_closure && enforce_population_closure!(c_stage)
#             calc__rhs!(k4, curr_itr, q_full(curr_itr + 1), c_stage)

#             @. c_next = c_t + (Δt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
#             # print("끝\n")
#         end

#         # Enforce the internal transported coordinate c first, then read out
#         # the physical σ from c and store it in context.σ.
#         if use_population_closure
#             enforce_population_closure!(c_next)
#         elseif enforce_hermitian
#             enforce_hermiticity!(c_next)
#         end

#         # Physical storage: σ = c + Tr_B{Q_T rho} at G0 readout level.
#         # If include_QT_G0_readout=false, readout__σ! reduces to projected-only σ≈c.
#         readout__σ!(σ_next, curr_itr + 1, q_full(curr_itr + 1), c_next)

#         if use_population_closure
#             enforce_population_closure!(σ_next)
#         elseif enforce_hermitian
#             enforce_hermiticity!(σ_next)
#         end

#         context.curr_itr = UInt64(curr_itr + 1)
#     end

#     return @view σ[:, :, Int(context.curr_itr)]
# end










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

