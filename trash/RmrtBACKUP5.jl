
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

function calc__σ_σ′_secular_core!(
    context::RmrtContext;
    use_population_closure::Bool = false,
    use_local_population_to_coherence::Bool = true,
    use_local_coherence_to_population::Bool = true,
    use_population_memory_population_input::Bool = true,
    use_population_memory_coherence_input::Bool = true,
    use_L0Q_memory_return::Bool = true,

    # ---------------------------------------------------------------------
    # Optional HEOM-reference injection mode.
    # If use_heom_input=true, the function reads heom_file and evaluates the
    # RMRT RHS/diagnostics along the HEOM reduced trajectory instead of the
    # self-propagated trajectory.  In this mode σ[:, :, itr] is overwritten by
    # the HEOM density at each grid point, while σ′[:, :, itr] stores the RMRT
    # RHS evaluated on that HEOM input.
    #
    # Expected HEOM text columns:
    #   time, p1, p2, ..., c12_re, c12_im, ...
    # Optional derivative columns:
    #   dp1, dp2, ..., cp12_re, cp12_im, ...
    # ---------------------------------------------------------------------
    heom_file::Union{Nothing,String} = nothing,
    use_heom_input::Bool = false,
    heom_time_tol::Float64 = 1.0e-8,

    # ---------------------------------------------------------------------
    # Teacher-forcing cutoff mode.
    # If heom_teacher_forcing_cutoff is not nothing, HEOM history is used up
    # to t_cut, and after t_cut the dynamics is self-propagated.
    #
    # Difference from use_heom_input=true:
    #   use_heom_input=true                 : force HEOM for all times.
    #   heom_teacher_forcing_cutoff = t_cut : force only t <= t_cut.
    #
    # With 1-based grid indexing t=(itr-1)*Δt, the cutoff index is the largest
    # index satisfying t <= t_cut.  The first self-propagated step starts from
    # that HEOM cutoff state.
    # ---------------------------------------------------------------------
    heom_teacher_forcing_cutoff::Union{Nothing,Real} = nothing,

    verify_L0Q_terms::Bool = false,
    verify_L0Q_every::Int = 1,
    verify_L0Q_pair::Union{Nothing,Tuple{Int,Int}} = nothing,
    verify_L0Q_io::IO = stderr,
    verify_L0Q_header::Bool = true,

    # ---------------------------------------------------------------------
    # Phase / instantaneous-frequency diagnostics for a coherence component.
    # This traces RHS contributions R_k through
    #
    #     omega_k(t) = Im(conj(c_ab) * R_k) / |c_ab|^2
    #
    # which is the RHS-based instantaneous phase velocity contribution.
    # Use trace_phase_pair=(1,2) to diagnose c12, or nothing for all coherences.
    # ---------------------------------------------------------------------
    trace_phase_terms::Bool = false,
    trace_phase_every::Int = 1,
    trace_phase_pair::Union{Nothing,Tuple{Int,Int}} = (1,2),
    trace_phase_eps::Float64 = 1.0e-8,
    trace_phase_io::IO = stderr,
    trace_phase_header::Bool = true,

    # ---------------------------------------------------------------------
    # Population-RHS diagnostics.
    # This prints R_aa^RMRT(t) = [dσ_aa/dt]_RMRT evaluated on the current
    # input history.  If use_heom_input=true and heom_file contains derivative
    # columns dp1, dp2, ..., the same line also prints the HEOM finite-difference
    # dp_a(t) and the RMRT-minus-HEOM derivative error.
    #
    # Use trace_population_indices=nothing for all populations, or e.g.
    # trace_population_indices=(1, 2) / [1, 2] for selected diagonal entries.
    # ---------------------------------------------------------------------
    trace_population_rhs::Bool = false,
    trace_population_every::Int = 1,
    trace_population_indices = nothing,
    trace_population_io::IO = stderr,
    trace_population_header::Bool = true,

    # ---------------------------------------------------------------------
    # Population-output RHS decomposition diagnostics.
    # This is intended for HEOM-forced RHS checks.  For each selected
    # population output (a,a), it prints the applied local P L1 P piece,
    # the applied P L1 Q G0 Q L1 P memory branches split by population vs
    # coherence input, and the residual against HEOM dp_a when available.
    #
    # The diagnostic is independent of trace_population_rhs; use a separate
    # IO stream if you want a clean table.
    # ---------------------------------------------------------------------
    trace_population_decomp::Bool = false,
    trace_population_decomp_every::Int = 1,
    trace_population_decomp_indices = nothing,
    trace_population_decomp_io::IO = stderr,
    trace_population_decomp_header::Bool = true,

    verify_L0P_transport::Bool = false,
    verify_L0P_every::Int = 100,
    verify_L0P_pair::Union{Nothing,Tuple{Int,Int}} = (1,2),
    verify_L0P_s_offset::Int = 100,
    verify_L0P_io::IO = stderr,
    verify_L0P_header::Bool = true,

    use_secular::Bool = false,
    secular_tol::Float64 = 1.0e-10,
    method::Union{Symbol,String} = :rk4,
    use_threads::Bool = true,

    # Half-shifted g-grid is allocated/calculated lazily.  This flag only
    # prepares the cache for stage-time RK2/RK4 support; the RHS refactor should
    # use the prepared cache explicitly rather than allocating it in inner loops.
    auto_prepare_half_shifted_grid::Bool = true,
    recompute_half_shifted_grid::Bool = false,

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

    function load__heom_reference_density_and_derivative(
        path::String,
        n_sys::Int,
        n_itr::Int,
        Δt::Float64;
        time_tol::Float64 = 1.0e-8,
    )
        isfile(path) || error("HEOM reference file not found: $(path)")

        lines = readlines(path)
        header = String[]
        rows = Vector{Vector{Float64}}()
        header_found = false

        for raw_line in lines
            line = strip(raw_line)
            isempty(line) && continue
            startswith(line, "#") && continue
            startswith(line, "----") && continue

            parts = split(line)
            isempty(parts) && continue

            if !header_found
                if any(x -> x == "time", parts) && any(x -> x == "p1", parts)
                    header = String.(parts)
                    header_found = true
                end
                continue
            end

            length(parts) < length(header) && continue

            vals = Vector{Float64}(undef, length(header))
            ok = true
            for j in eachindex(header)
                v = tryparse(Float64, parts[j])
                if v === nothing
                    ok = false
                    break
                end
                vals[j] = v
            end
            ok && push!(rows, vals)
        end

        header_found || error("Could not find HEOM header line with columns `time` and `p1` in $(path)")
        isempty(rows) && error("No numeric HEOM data rows found in $(path)")

        col = Dict{String,Int}()
        for (j, name) in pairs(header)
            col[name] = j
        end

        haskey(col, "time") || error("HEOM file is missing `time` column")
        times = [row[col["time"]] for row in rows]

        # Require monotonic time data.
        for j in 2:length(times)
            times[j] >= times[j - 1] || error("HEOM time column must be nondecreasing")
        end

        function interp_value(name::String, t::Float64)
            haskey(col, name) || error("HEOM file is missing required column `$(name)`")

            if t <= times[1] + time_tol
                return rows[1][col[name]]
            elseif t >= times[end] - time_tol
                return rows[end][col[name]]
            end

            j = searchsortedlast(times, t)
            j = clamp(j, 1, length(times) - 1)
            t0 = times[j]
            t1 = times[j + 1]

            if abs(t1 - t0) <= eps(Float64)
                return rows[j][col[name]]
            end

            λ = (t - t0) / (t1 - t0)
            return (1.0 - λ) * rows[j][col[name]] + λ * rows[j + 1][col[name]]
        end

        function interp_optional_value(name::String, t::Float64)
            haskey(col, name) || return NaN

            if t <= times[1] + time_tol
                return rows[1][col[name]]
            elseif t >= times[end] - time_tol
                return rows[end][col[name]]
            end

            j = searchsortedlast(times, t)
            j = clamp(j, 1, length(times) - 1)
            t0 = times[j]
            t1 = times[j + 1]

            if abs(t1 - t0) <= eps(Float64)
                return rows[j][col[name]]
            end

            λ = (t - t0) / (t1 - t0)
            return (1.0 - λ) * rows[j][col[name]] + λ * rows[j + 1][col[name]]
        end

        heom_sigma = Array{ComplexF64}(undef, n_sys, n_sys, n_itr)
        heom_sigmap = fill(ComplexF64(NaN, NaN), n_sys, n_sys, n_itr)

        for itr in 1:n_itr
            t = (itr - 1) * Δt

            for a in 1:n_sys
                pa = interp_value("p$(a)", t)
                heom_sigma[a, a, itr] = pa + 0.0im

                dpa = interp_optional_value("dp$(a)", t)
                if isfinite(dpa)
                    heom_sigmap[a, a, itr] = dpa + 0.0im
                end
            end

            for a in 1:(n_sys - 1)
                for b in (a + 1):n_sys
                    re = interp_value("c$(a)$(b)_re", t)
                    imv = interp_value("c$(a)$(b)_im", t)
                    cab = re + 1.0im * imv
                    heom_sigma[a, b, itr] = cab
                    heom_sigma[b, a, itr] = conj(cab)

                    dre = interp_optional_value("cp$(a)$(b)_re", t)
                    dimv = interp_optional_value("cp$(a)$(b)_im", t)
                    if isfinite(dre) && isfinite(dimv)
                        dcab = dre + 1.0im * dimv
                        heom_sigmap[a, b, itr] = dcab
                        heom_sigmap[b, a, itr] = conj(dcab)
                    end
                end
            end
        end

        return heom_sigma, heom_sigmap
    end

    heom_sigma_ref = nothing
    heom_sigmap_ref = nothing

    teacher_forcing_enabled = heom_teacher_forcing_cutoff !== nothing
    teacher_cutoff_itr = 0

    if teacher_forcing_enabled
        t_cut = Float64(heom_teacher_forcing_cutoff)
        t_cut >= 0.0 || error("heom_teacher_forcing_cutoff must be >= 0")

        # 1-based index with t = (itr - 1) * Δt.
        # Choose the largest index with t <= t_cut, up to tiny roundoff.
        teacher_cutoff_itr = clamp(
            floor(Int, t_cut / Δt + 1.0e-9) + 1,
            1,
            n_itr,
        )
    end

    if use_heom_input || teacher_forcing_enabled
        heom_file === nothing && error(
            "HEOM forcing requires heom_file=\"path/to/heom.txt\""
        )

        heom_sigma_ref, heom_sigmap_ref = load__heom_reference_density_and_derivative(
            String(heom_file),
            n_sys,
            n_itr,
            Δt;
            time_tol = heom_time_tol,
        )
    end

    @inline function use_heom_for_storage_itr(itr::Int)
        # Store exact HEOM state up to and including the cutoff point.
        return use_heom_input ||
               (teacher_forcing_enabled && itr <= teacher_cutoff_itr)
    end

    @inline function use_heom_for_rhs_current_itr(itr::Int)
        # Before cutoff, RHS diagnostics are evaluated on HEOM current.
        # At the cutoff point, RK stages use σ_t so that the first self step
        # starts from the HEOM cutoff state.
        return use_heom_input ||
               (teacher_forcing_enabled && itr < teacher_cutoff_itr)
    end

    @inline function use_heom_for_history_itr(itr::Int)
        # Memory history strictly before cutoff is HEOM.  The cutoff point is
        # already stored in σ as HEOM and then treated as the self initial state.
        return use_heom_input ||
               (teacher_forcing_enabled && itr < teacher_cutoff_itr)
    end

    method_sym = Symbol(lowercase(String(method)))
    method_sym in (:euler, :rk2, :rk4) || error(
        "Unsupported integration method: $(method). " *
        "Use :euler, :rk2, or :rk4."
    )
    needs_half_shifted_grid = method_sym in (:rk2, :rk4)

    if needs_half_shifted_grid
        if auto_prepare_half_shifted_grid
            ensure__half_shifted_grid!(
                context;
                recompute = recompute_half_shifted_grid,
                use_threads = use_threads,
                verbose = verbose,
            )
        elseif !context.using_half_shifted_grid || !has__half_shifted_grid(context)
            error(
                "method=$(method_sym) requires half-shifted g-series cache. " *
                "Call ensure__half_shifted_grid!(context) or " *
                "calc__g_g′_g″_half_shifted!(context) before propagation, " *
                "or set auto_prepare_half_shifted_grid=true."
            )
        end
    end

    verify_L0Q_every >= 1 || error("verify_L0Q_every must be >= 1")
    trace_phase_every >= 1 || error("trace_phase_every must be >= 1")
    trace_phase_eps > 0.0 || error("trace_phase_eps must be > 0")
    trace_population_every >= 1 || error("trace_population_every must be >= 1")
    trace_population_decomp_every >= 1 || error("trace_population_decomp_every must be >= 1")
    verify_L0P_every >= 1 || error("verify_L0P_every must be >= 1")
    verify_L0P_s_offset >= 1 || error("verify_L0P_s_offset must be >= 1")

    start_itr < n_itr || return @view σ[:, :, n_itr]

    @inline ω(a::Int, b::Int) = ϵ[a] - ϵ[b]

    # -------------------------------------------------------------------------
    # Verification for QL0P - dot(P)P = 0 convention
    #
    # We check consistency between:
    #
    #   D_code[a,b](t)
    #     = -iω_ab - g′_aaaa + conj(g′_aabb) + g′_bbaa - conj(g′_bbbb)
    #
    # and
    #
    #   D_N[a,b](t)
    #     = -iω_ab + d/dt log N_ab(t)
    #
    # where
    #
    #   log N_ab(t)
    #     = -g_aaaa(t) - conj(g_bbbb(t))
    #       + g_bbaa(t) + conj(g_aabb(t)).
    #
    # If D_code - D_N is not small, the L0-transport convention,
    # N_ab convention, or g′ sign/conjugation convention is inconsistent.
    # -------------------------------------------------------------------------

    verify_L0P_header_printed = Ref(false)
    verify_L0P_lock = ReentrantLock()

    @inline function logN_bath(
        itr::Int,
        a::Int,
        b::Int,
    )
        return (
            -g[itr, a, a, a, a]
            -conj(g[itr, b, b, b, b])
            +g[itr, b, b, a, a]
            +conj(g[itr, a, a, b, b])
        )
    end

    @inline function D_L0P_code(
        itr::Int,
        a::Int,
        b::Int,
    )
        return (
            -1.0im * ω(a, b)
            -g′[itr, a, a, a, a]
            +conj(g′[itr, a, a, b, b])
            +g′[itr, b, b, a, a]
            -conj(g′[itr, b, b, b, b])
        )
    end

    @inline function dlogN_bath_gprime(
        itr::Int,
        a::Int,
        b::Int,
    )
        return (
            -g′[itr, a, a, a, a]
            -conj(g′[itr, b, b, b, b])
            +g′[itr, b, b, a, a]
            +conj(g′[itr, a, a, b, b])
        )
    end

    @inline function D_L0P_from_N_gprime(
        itr::Int,
        a::Int,
        b::Int,
    )
        return -1.0im * ω(a, b) + dlogN_bath_gprime(itr, a, b)
    end

    @inline function dlogN_bath_finite_difference(
        itr::Int,
        a::Int,
        b::Int,
    )
        if itr <= 1
            return (logN_bath(2, a, b) - logN_bath(1, a, b)) / Δt
        elseif itr >= n_itr
            return (logN_bath(n_itr, a, b) - logN_bath(n_itr - 1, a, b)) / Δt
        else
            return (logN_bath(itr + 1, a, b) - logN_bath(itr - 1, a, b)) / (2.0 * Δt)
        end
    end

    @inline function D_L0P_from_N_fd(
        itr::Int,
        a::Int,
        b::Int,
    )
        return -1.0im * ω(a, b) + dlogN_bath_finite_difference(itr, a, b)
    end

    @inline function should_verify_L0P(
        curr_itr::Int,
        a::Int,
        b::Int,
    )
        if !verify_L0P_transport || a == b
            return false
        end

        ((curr_itr - start_itr) % verify_L0P_every == 0) || return false

        verify_L0P_pair === nothing && return true

        return verify_L0P_pair == (a, b)
    end

    function print_verify_L0P_header_if_needed!()
        if verify_L0P_header && !verify_L0P_header_printed[]
            lock(verify_L0P_lock)
            try
                if !verify_L0P_header_printed[]
                    println(
                        verify_L0P_io,
                        "# L0P_VERIFY columns: itr t a b s_itr Δ " *
                        "D_code D_N_gprime D_N_fd " *
                        "D_code_minus_D_N_gprime D_code_minus_D_N_fd " *
                        "log_transport_exact log_transport_quad transport_error"
                    )
                    verify_L0P_header_printed[] = true
                end
            finally
                unlock(verify_L0P_lock)
            end
        end

        return nothing
    end

    @inline function fmt__L0P_real(x)
        return Printf.@sprintf("%+.12e", Float64(real(x)))
    end

    @inline function fmt__L0P_time(x)
        return Printf.@sprintf("%.10e", Float64(x))
    end

    @inline function fmt__L0P_complex(z)
        return "(" * fmt__L0P_real(real(z)) * "," * fmt__L0P_real(imag(z)) * ")"
    end

    @inline function log_transport_exact(
        s_itr::Int,
        t_itr::Int,
        a::Int,
        b::Int,
    )
        Δ = (t_itr - s_itr) * Δt

        return (
            -1.0im * ω(a, b) * Δ
            + logN_bath(t_itr, a, b)
            - logN_bath(s_itr, a, b)
        )
    end

    function log_transport_quad(
        s_itr::Int,
        t_itr::Int,
        a::Int,
        b::Int,
    )
        if s_itr == t_itr
            return 0.0 + 0.0im
        end

        acc = 0.0 + 0.0im

        @inbounds for itr in s_itr:t_itr
            w = (itr == s_itr || itr == t_itr) ? 0.5 * Δt : Δt
            acc += w * D_L0P_code(itr, a, b)
        end

        return acc
    end

    function print__L0P_verify_line!(
        io::IO,
        curr_itr::Int,
        time::Float64,
        a::Int,
        b::Int,
        s_itr::Int,
        Δ::Float64,
        D_code,
        D_N_gprime,
        D_N_fd,
        D_diff_gprime,
        D_diff_fd,
        log_exact,
        log_quad,
        log_err,
    )
        println(
            io,
            "L0P_VERIFY",
            " itr=", curr_itr,
            " t=", fmt__L0P_time(time),
            " a=", a,
            " b=", b,
            " s_itr=", s_itr,
            " Delta=", fmt__L0P_time(Δ),
            " D_code=", fmt__L0P_complex(D_code),
            " D_N_gprime=", fmt__L0P_complex(D_N_gprime),
            " D_N_fd=", fmt__L0P_complex(D_N_fd),
            " D_code_minus_D_N_gprime=", fmt__L0P_complex(D_diff_gprime),
            " D_code_minus_D_N_fd=", fmt__L0P_complex(D_diff_fd),
            " log_transport_exact=", fmt__L0P_complex(log_exact),
            " log_transport_quad=", fmt__L0P_complex(log_quad),
            " transport_error=", fmt__L0P_complex(log_err),
        )

        return nothing
    end

    function verify__L0P_transport_at!(
        curr_itr::Int,
    )
        verify_L0P_transport || return nothing

        time = (curr_itr - 1) * Δt
        s_itr = max(1, curr_itr - verify_L0P_s_offset)
        Δ = (curr_itr - s_itr) * Δt

        print_verify_L0P_header_if_needed!()

        @inbounds for b in 1:n_sys
            for a in 1:n_sys
                should_verify_L0P(curr_itr, a, b) || continue

                D_code = D_L0P_code(curr_itr, a, b)
                D_N_gprime = D_L0P_from_N_gprime(curr_itr, a, b)
                D_N_fd = D_L0P_from_N_fd(curr_itr, a, b)

                D_diff_gprime = D_code - D_N_gprime
                D_diff_fd = D_code - D_N_fd

                log_exact = log_transport_exact(s_itr, curr_itr, a, b)
                log_quad = log_transport_quad(s_itr, curr_itr, a, b)
                log_err = log_exact - log_quad

                lock(verify_L0P_lock)
                try
                    print__L0P_verify_line!(
                        verify_L0P_io,
                        curr_itr,
                        time,
                        a,
                        b,
                        s_itr,
                        Δ,
                        D_code,
                        D_N_gprime,
                        D_N_fd,
                        D_diff_gprime,
                        D_diff_fd,
                        log_exact,
                        log_quad,
                        log_err,
                    )
                finally
                    unlock(verify_L0P_lock)
                end
            end
        end

        return nothing
    end

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

    @inline function is_local_coherence_to_population(
        out_a::Int,
        out_b::Int,
        in_a::Int,
        in_b::Int,
    )
        return out_a == out_b && in_a != in_b
    end

    @inline function is_population_input(in_a::Int, in_b::Int)
        return in_a == in_b
    end

    @inline function is_coherence_input(in_a::Int, in_b::Int)
        return in_a != in_b
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

    @inline function ∫weight(s_itr::Int, curr_itr::Int)
        return (s_itr == 1 || s_itr == curr_itr) ? 0.5 * Δt : Δt
    end

    @inline function phase(a::Int, b::Int, Δ_itr::Int)
        Δ = (Δ_itr - 1) * Δt
        return exp(-1.0im * ω(a, b) * Δ)
    end

    # -------------------------------------------------------------------------
    # Non-time-localized memory density access
    # -------------------------------------------------------------------------
    # Memory integrals must use the density at the integration time s.
    # For the endpoint s = t during RK stages, use the current trial matrix σ_t;
    # for earlier times, use the already stored trajectory σ[:, :, s_itr].
    # This removes the previous time-localization σ(s) -> σ(t) in the memory
    # integrals while preserving RK stage consistency at the moving endpoint.
    # -------------------------------------------------------------------------
    @inline function σ_mem(
        σ_t::AbstractMatrix,
        s_itr::Int,
        curr_itr::Int,
        a::Int,
        b::Int,
    )
        if use_heom_for_history_itr(s_itr)
            return heom_sigma_ref[a, b, s_itr]
        end

        return s_itr == curr_itr ? σ_t[a, b] : σ[a, b, s_itr]
    end

    @inline function σ_now(
        σ_t::AbstractMatrix,
        curr_itr::Int,
        a::Int,
        b::Int,
    )
        if use_heom_for_rhs_current_itr(curr_itr)
            return heom_sigma_ref[a, b, curr_itr]
        end

        return σ_t[a, b]
    end

    @inline function σprime_heom_now(
        curr_itr::Int,
        a::Int,
        b::Int,
    )
        if use_heom_input || teacher_forcing_enabled
            return heom_sigmap_ref[a, b, curr_itr]
        end

        return ComplexF64(NaN, NaN)
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
        return fmt__real(real(z)) * " " * fmt__real(imag(z))
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

    # -------------------------------------------------------------------------
    # Phase / instantaneous-frequency diagnostics
    # -------------------------------------------------------------------------

    trace_phase_header_printed = Ref(false)
    trace_phase_lock = ReentrantLock()

    @inline function should_trace_phase(curr_itr::Int, α::Int, β::Int)
        if !trace_phase_terms || α == β
            return false
        end

        ((curr_itr - start_itr) % trace_phase_every == 0) || return false

        trace_phase_pair === nothing && return true

        return trace_phase_pair == (α, β)
    end

    @inline function omega_from_rhs_term(term, c)
        denom = abs2(c)
        if denom <= trace_phase_eps^2
            return NaN
        end
        return imag(conj(c) * term) / denom
    end

    @inline function growth_from_rhs_term(term, c)
        denom = abs2(c)
        if denom <= trace_phase_eps^2
            return NaN
        end
        return real(conj(c) * term) / denom
    end

    function print_trace_phase_header_if_needed!()
        if trace_phase_header && !trace_phase_header_printed[]
            lock(trace_phase_lock)
            try
                if !trace_phase_header_printed[]
                    println(
                        trace_phase_io,
                        "PHASE_TRACE itr t a b abs_c " *
                        "REAL_c IMAG_c REAL_rhs_diag IMAG_rhs_diag REAL_rhs_local_raw IMAG_rhs_local_raw REAL_rhs_local_app IMAG_rhs_local_app " *
                        "REAL_rhs_mem_core_app IMAG_rhs_mem_core_app REAL_rhs_L0Q_raw IMAG_rhs_L0Q_raw REAL_rhs_L0Q_app IMAG_rhs_L0Q_app REAL_rhs_total IMAG_rhs_total REAL_rhs_heom_fd IMAG_rhs_heom_fd " *
                        "omega_total omega_heom_fd omega_diag omega_local_raw omega_local_app " *
                        "omega_mem_core_app omega_L0Q_raw omega_L0Q_app " *
                        "omega_L0Q_A_raw omega_L0Q_B_raw " *
                        "growth_total growth_heom_fd growth_L0Q_raw growth_L0Q_app"
                    )
                    trace_phase_header_printed[] = true
                end
            finally
                unlock(trace_phase_lock)
            end
        end
        return nothing
    end

    function print__phase_trace_line!(
        io::IO,
        curr_itr::Int,
        t::Float64,
        α::Int,
        β::Int,
        c,
        rhs_diag,
        rhs_local_raw,
        rhs_local_app,
        rhs_mem_core_app,
        rhs_L0Q_raw,
        rhs_L0Q_app,
        rhs_L0Q_A_raw,
        rhs_L0Q_B_raw,
        rhs_total,
        rhs_heom_fd,
    )
        omega_total        = omega_from_rhs_term(rhs_total, c)
        omega_heom_fd      = omega_from_rhs_term(rhs_heom_fd, c)
        omega_diag         = omega_from_rhs_term(rhs_diag, c)
        omega_local_raw    = omega_from_rhs_term(rhs_local_raw, c)
        omega_local_app    = omega_from_rhs_term(rhs_local_app, c)
        omega_mem_core_app = omega_from_rhs_term(rhs_mem_core_app, c)
        omega_L0Q_raw      = omega_from_rhs_term(rhs_L0Q_raw, c)
        omega_L0Q_app      = omega_from_rhs_term(rhs_L0Q_app, c)
        omega_L0Q_A_raw    = omega_from_rhs_term(rhs_L0Q_A_raw, c)
        omega_L0Q_B_raw    = omega_from_rhs_term(rhs_L0Q_B_raw, c)

        growth_total       = growth_from_rhs_term(rhs_total, c)
        growth_heom_fd     = growth_from_rhs_term(rhs_heom_fd, c)
        growth_L0Q_raw     = growth_from_rhs_term(rhs_L0Q_raw, c)
        growth_L0Q_app     = growth_from_rhs_term(rhs_L0Q_app, c)

        println(
            io,
            "PHASE_TRACE",                          " ",
            # " itr=",                curr_itr,
            # " t=",                  fmt__time(t),
            # " a=",                  α,
            # " b=",                  β,
            # " abs_c=",              fmt__real(abs(c)),
            # " c=",                  fmt__complex(c),
            # " rhs_diag=",           fmt__complex(rhs_diag),
            # " rhs_local_raw=",      fmt__complex(rhs_local_raw),
            # " rhs_local_app=",      fmt__complex(rhs_local_app),
            # " rhs_mem_core_app=",   fmt__complex(rhs_mem_core_app),
            # " rhs_L0Q_raw=",        fmt__complex(rhs_L0Q_raw),
            # " rhs_L0Q_app=",        fmt__complex(rhs_L0Q_app),
            # " rhs_total=",          fmt__complex(rhs_total),
            # " omega_total=",        fmt__real(omega_total),
            # " omega_diag=",         fmt__real(omega_diag),
            # " omega_local_raw=",    fmt__real(omega_local_raw),
            # " omega_local_app=",    fmt__real(omega_local_app),
            # " omega_mem_core_app=", fmt__real(omega_mem_core_app),
            # " omega_L0Q_raw=",      fmt__real(omega_L0Q_raw),
            # " omega_L0Q_app=",      fmt__real(omega_L0Q_app),
            # " omega_L0Q_A_raw=",    fmt__real(omega_L0Q_A_raw),
            # " omega_L0Q_B_raw=",    fmt__real(omega_L0Q_B_raw),
            # " growth_total=",       fmt__real(growth_total),
            # " growth_L0Q_raw=",     fmt__real(growth_L0Q_raw),
            # " growth_L0Q_app=",     fmt__real(growth_L0Q_app),
            curr_itr,                           " ",
            fmt__time(t),                       " ",
            α,                                  " ",
            β,                                  " ",
            fmt__real(abs(c)),                  " ",                
            fmt__complex(c),                    " ",              
            fmt__complex(rhs_diag),             " ",                     
            fmt__complex(rhs_local_raw),        " ",         
            fmt__complex(rhs_local_app),        " ",         
            fmt__complex(rhs_mem_core_app),     " ",            
            fmt__complex(rhs_L0Q_raw),          " ",       
            fmt__complex(rhs_L0Q_app),          " ",       
            fmt__complex(rhs_total),            " ",     
            fmt__complex(rhs_heom_fd),          " ",
            fmt__real(omega_total),             " ",    
            fmt__real(omega_heom_fd),           " ",
            fmt__real(omega_diag),              " ",   
            fmt__real(omega_local_raw),         " ",        
            fmt__real(omega_local_app),         " ",        
            fmt__real(omega_mem_core_app),      " ",           
            fmt__real(omega_L0Q_raw),           " ",      
            fmt__real(omega_L0Q_app),           " ",      
            fmt__real(omega_L0Q_A_raw),         " ",        
            fmt__real(omega_L0Q_B_raw),         " ",        
            fmt__real(growth_total),            " ",     
            fmt__real(growth_heom_fd),          " ",
            fmt__real(growth_L0Q_raw),          " ",       
            fmt__real(growth_L0Q_app),          " ",       
        )
        return nothing
    end


    # -------------------------------------------------------------------------
    # Population RHS diagnostics
    # -------------------------------------------------------------------------

    trace_population_header_printed = Ref(false)
    trace_population_lock = ReentrantLock()

    @inline function should_trace_population(curr_itr::Int, a::Int)
        trace_population_rhs || return false
        ((curr_itr - start_itr) % trace_population_every == 0) || return false
        trace_population_indices === nothing && return true
        return a in trace_population_indices
    end

    function print_trace_population_header_if_needed!()
        if trace_population_header && !trace_population_header_printed[]
            lock(trace_population_lock)
            try
                if !trace_population_header_printed[]
                    println(
                        trace_population_io,
                        "POP_RHS_TRACE itr t a p_a " *
                        "REAL_Raa_RMRT IMAG_Raa_RMRT " *
                        "REAL_dp_a_HEOM IMAG_dp_a_HEOM " *
                        "REAL_R_minus_dp IMAG_R_minus_dp"
                    )
                    trace_population_header_printed[] = true
                end
            finally
                unlock(trace_population_lock)
            end
        end
        return nothing
    end

    function print__population_rhs_trace_line!(
        io::IO,
        curr_itr::Int,
        t::Float64,
        a::Int,
        p_a,
        rhs_aa,
        heom_dp_aa,
    )
        diff = rhs_aa - heom_dp_aa

        println(
            io,
            "POP_RHS_TRACE", " ",
            curr_itr, " ",
            fmt__time(t), " ",
            a, " ",
            fmt__real(real(p_a)), " ",
            fmt__complex(rhs_aa), " ",
            fmt__complex(heom_dp_aa), " ",
            fmt__complex(diff),
        )

        return nothing
    end

    @inline function trace__population_rhs_if_needed!(
        curr_itr::Int,
        σ_t::AbstractMatrix,
        a::Int,
        rhs_aa,
    )
        should_trace_population(curr_itr, a) || return nothing

        time = (curr_itr - 1) * Δt
        p_a = real(σ_now(σ_t, curr_itr, a, a))
        heom_dp_aa = σprime_heom_now(curr_itr, a, a)

        print_trace_population_header_if_needed!()
        lock(trace_population_lock)
        try
            print__population_rhs_trace_line!(
                trace_population_io,
                curr_itr,
                time,
                a,
                p_a,
                rhs_aa,
                heom_dp_aa,
            )
        finally
            unlock(trace_population_lock)
        end

        return nothing
    end


    # -------------------------------------------------------------------------
    # Population-output RHS decomposition diagnostics
    # -------------------------------------------------------------------------

    trace_population_decomp_header_printed = Ref(false)
    trace_population_decomp_lock = ReentrantLock()

    @inline function should_trace_population_decomp(curr_itr::Int, a::Int)
        trace_population_decomp || return false
        ((curr_itr - start_itr) % trace_population_decomp_every == 0) || return false
        trace_population_decomp_indices === nothing && return true
        return a in trace_population_decomp_indices
    end

    function print_trace_population_decomp_header_if_needed!()
        if trace_population_decomp_header && !trace_population_decomp_header_printed[]
            lock(trace_population_decomp_lock)
            try
                if !trace_population_decomp_header_printed[]
                    println(
                        trace_population_decomp_io,
                        "POP_DECOMP_TRACE itr t a p_a " *
                        "REAL_rhs_diag IMAG_rhs_diag " *
                        "REAL_local_A_raw IMAG_local_A_raw REAL_local_B_raw IMAG_local_B_raw REAL_local_raw IMAG_local_raw " *
                        "REAL_local_A_app IMAG_local_A_app REAL_local_B_app IMAG_local_B_app REAL_local_app IMAG_local_app " *
                        "REAL_mem_b1_pop_app IMAG_mem_b1_pop_app REAL_mem_b1_coh_app IMAG_mem_b1_coh_app " *
                        "REAL_mem_b2_pop_app IMAG_mem_b2_pop_app REAL_mem_b2_coh_app IMAG_mem_b2_coh_app " *
                        "REAL_mem_b3_pop_app IMAG_mem_b3_pop_app REAL_mem_b3_coh_app IMAG_mem_b3_coh_app " *
                        "REAL_mem_b4_pop_app IMAG_mem_b4_pop_app REAL_mem_b4_coh_app IMAG_mem_b4_coh_app " *
                        "REAL_mem_pop_app IMAG_mem_pop_app REAL_mem_coh_app IMAG_mem_coh_app REAL_mem_core_app IMAG_mem_core_app " *
                        "REAL_rhs_total IMAG_rhs_total REAL_dp_HEOM IMAG_dp_HEOM REAL_diff IMAG_diff " *
                        "REAL_diff_minus_local_app IMAG_diff_minus_local_app " *
                        "REAL_diff_minus_mem_coh_app IMAG_diff_minus_mem_coh_app " *
                        "REAL_diff_minus_local_and_mem_coh_app IMAG_diff_minus_local_and_mem_coh_app"
                    )
                    trace_population_decomp_header_printed[] = true
                end
            finally
                unlock(trace_population_decomp_lock)
            end
        end
        return nothing
    end

    function print__population_decomp_line!(
        io::IO,
        curr_itr::Int,
        t::Float64,
        a::Int,
        p_a,
        rhs_diag,
        local_A_raw,
        local_B_raw,
        local_A_app,
        local_B_app,
        mem_b1_pop_app,
        mem_b1_coh_app,
        mem_b2_pop_app,
        mem_b2_coh_app,
        mem_b3_pop_app,
        mem_b3_coh_app,
        mem_b4_pop_app,
        mem_b4_coh_app,
        rhs_total,
        heom_dp_aa,
    )
        local_raw = local_A_raw + local_B_raw
        local_app = local_A_app + local_B_app

        mem_pop_app = mem_b1_pop_app + mem_b2_pop_app + mem_b3_pop_app + mem_b4_pop_app
        mem_coh_app = mem_b1_coh_app + mem_b2_coh_app + mem_b3_coh_app + mem_b4_coh_app
        mem_core_app = mem_pop_app + mem_coh_app

        diff = rhs_total - heom_dp_aa
        diff_minus_local_app = diff - local_app
        diff_minus_mem_coh_app = diff - mem_coh_app
        diff_minus_local_and_mem_coh_app = diff - local_app - mem_coh_app

        println(
            io,
            "POP_DECOMP_TRACE", " ",
            curr_itr, " ",
            fmt__time(t), " ",
            a, " ",
            fmt__real(real(p_a)), " ",
            fmt__complex(rhs_diag), " ",
            fmt__complex(local_A_raw), " ",
            fmt__complex(local_B_raw), " ",
            fmt__complex(local_raw), " ",
            fmt__complex(local_A_app), " ",
            fmt__complex(local_B_app), " ",
            fmt__complex(local_app), " ",
            fmt__complex(mem_b1_pop_app), " ",
            fmt__complex(mem_b1_coh_app), " ",
            fmt__complex(mem_b2_pop_app), " ",
            fmt__complex(mem_b2_coh_app), " ",
            fmt__complex(mem_b3_pop_app), " ",
            fmt__complex(mem_b3_coh_app), " ",
            fmt__complex(mem_b4_pop_app), " ",
            fmt__complex(mem_b4_coh_app), " ",
            fmt__complex(mem_pop_app), " ",
            fmt__complex(mem_coh_app), " ",
            fmt__complex(mem_core_app), " ",
            fmt__complex(rhs_total), " ",
            fmt__complex(heom_dp_aa), " ",
            fmt__complex(diff), " ",
            fmt__complex(diff_minus_local_app), " ",
            fmt__complex(diff_minus_mem_coh_app), " ",
            fmt__complex(diff_minus_local_and_mem_coh_app),
        )

        return nothing
    end

    @inline function trace__population_decomp_if_needed!(
        curr_itr::Int,
        σ_t::AbstractMatrix,
        a::Int,
        rhs_diag,
        local_A_raw,
        local_B_raw,
        local_A_app,
        local_B_app,
        mem_b1_pop_app,
        mem_b1_coh_app,
        mem_b2_pop_app,
        mem_b2_coh_app,
        mem_b3_pop_app,
        mem_b3_coh_app,
        mem_b4_pop_app,
        mem_b4_coh_app,
        rhs_total,
    )
        should_trace_population_decomp(curr_itr, a) || return nothing

        time = (curr_itr - 1) * Δt
        p_a = real(σ_now(σ_t, curr_itr, a, a))
        heom_dp_aa = σprime_heom_now(curr_itr, a, a)

        print_trace_population_decomp_header_if_needed!()
        lock(trace_population_decomp_lock)
        try
            print__population_decomp_line!(
                trace_population_decomp_io,
                curr_itr,
                time,
                a,
                p_a,
                rhs_diag,
                local_A_raw,
                local_B_raw,
                local_A_app,
                local_B_app,
                mem_b1_pop_app,
                mem_b1_coh_app,
                mem_b2_pop_app,
                mem_b2_coh_app,
                mem_b3_pop_app,
                mem_b3_coh_app,
                mem_b4_pop_app,
                mem_b4_coh_app,
                rhs_total,
                heom_dp_aa,
            )
        finally
            unlock(trace_population_decomp_lock)
        end

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


    # -------------------------------------------------------------------------
    # Non-time-localized exponent helpers
    # -------------------------------------------------------------------------
    # The original exponent_type_1/type_2 functions are strict transported
    # time-localized prefactors.  They include the input-channel backward
    # transport factor
    #
    #     T_in(s,t) = exp(+i omega_in Δ) * N_in^bath(s)/N_in^bath(t).
    #
    # For a genuinely non-time-localized kernel multiplying σ(input, s), the
    # scalar phase and the bath exponent must both be divided by this input
    # transport factor.  The phase division is handled explicitly in each branch
    # by changing phase(...).  The bath part is removed here by subtracting
    # logN_bath(s,input)-logN_bath(t,input).

    @inline function logT_input_bath(
        s_itr::Int,
        t_itr::Int,
        in_a::Int,
        in_b::Int,
    )
        return logN_bath(s_itr, in_a, in_b) - logN_bath(t_itr, in_a, in_b)
    end

    @inline function gen__exponent_type_1_nonlocal(
        s_itr::Int,
        Δ_itr::Int,
        t_itr::Int,
        α⁻::Int,
        α⁼::Int,
        β::Int,
    )
        # Strict-TL type-1 input channel is (α⁼, β).
        return gen__exponent_type_1(
            s_itr,
            Δ_itr,
            t_itr,
            α⁻,
            α⁼,
            β,
        ) - logT_input_bath(s_itr, t_itr, α⁼, β)
    end

    @inline function gen__exponent_type_2_nonlocal(
        s_itr::Int,
        Δ_itr::Int,
        t_itr::Int,
        a::Int,
        β⁼::Int,
        β⁻::Int,
    )
        # Strict-TL type-2 input channel is (a, β⁼).
        return gen__exponent_type_2(
            s_itr,
            Δ_itr,
            t_itr,
            a,
            β⁼,
            β⁻,
        ) - logT_input_bath(s_itr, t_itr, a, β⁼)
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
    # Non-time-localized, g-closed raw-trace form.
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
        # Non-time-localized Branch A raw trace prefactor.
        #
        # This corresponds to the direct raw block
        #
        #   exp(-iω[α,β]Δ)
        #   TrB{ exp(-iH_α Δ) B_{αf} τ_{fβ}(s) exp(+iH_β Δ) }
        #
        # plus the Q(s)- and Q(t)-subtracted covariance structure.  No
        # transported σ[f,β](s) -> σ[f,β](t) replacement is used here.
        # Therefore this is the old E_A(t,s,Δ), not Λ_A[f,α|β](t,s).
        return (
            -conj(g[t_itr, β, β, β, β] - g[s_itr, β, β, β, β])
            +conj(g[t_itr, α, α, β, β] - g[s_itr, α, α, β, β])

            +(g[t_itr, β, β, f, f] - g[s_itr, β, β, f, f] - g[Δ_itr, β, β, f, f])
            -(g[t_itr, α, α, f, f] - g[s_itr, α, α, f, f] - g[Δ_itr, α, α, f, f])

            -g[Δ_itr, α, α, α, α]
            +g[Δ_itr, β, β, α, α]
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
        # Non-time-localized Branch B raw trace prefactor.
        #
        # This corresponds to the direct raw block
        #
        #   exp(-iω[α,β]Δ)
        #   TrB{ exp(-iH_α Δ) τ_{αf}(s) B_{fβ} exp(+iH_β Δ) }
        #
        # plus the Q(s)- and Q(t)-subtracted covariance structure.  No
        # transported σ[α,f](s) -> σ[α,f](t) replacement is used here.
        # Therefore this is the old E_B(t,s,Δ), not Λ_B[α|f,β](t,s).
        return (
            -(g[t_itr, α, α, α, α] - g[s_itr, α, α, α, α])
            +(g[t_itr, β, β, α, α] - g[s_itr, β, β, α, α])

            +conj(g[t_itr, α, α, f, f] - g[s_itr, α, α, f, f] - g[Δ_itr, α, α, f, f])
            -conj(g[t_itr, β, β, f, f] - g[s_itr, β, β, f, f] - g[Δ_itr, β, β, f, f])

            -conj(g[Δ_itr, β, β, β, β])
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

        # if g″[Δ_itr, β, f, α, α] == g″[Δ_itr, α, α, β, f]
        #     @printf("meaningless1! ")
        # end
        # if g″[Δ_itr, β, f, β, β] == g″[Δ_itr, β, β, β, f]
        #     @printf("meaningless2! \n")
        # end

        return (
            conj(g″[Δ_itr, β, f, α, α])
            -conj(g″[Δ_itr, β, f, β, β])
            # conj(g″[Δ_itr, α, α, β, f])
            # -conj(g″[Δ_itr, β, β, β, f])
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

        for s_itr in 1:curr_itr
            Δ_itr = curr_itr - s_itr + 1
            w_int = ∫weight(s_itr, curr_itr)
            pα = real(σ_mem(σ_t, s_itr, curr_itr, α, α))

            for f in 1:n_sys
                f == α && continue

                pf = real(σ_mem(σ_t, s_itr, curr_itr, f, f))

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
    # The memory integrals use σ(s_itr) for s < t and σ_t only at the moving
    # endpoint s = t.  This is the non-time-localized memory form.
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
            trace__population_rhs_if_needed!(curr_itr, σ_t, α, rhs_pop)
            trace__population_decomp_if_needed!(
                curr_itr,
                σ_t,
                α,
                0.0 + 0.0im,
                0.0 + 0.0im,
                0.0 + 0.0im,
                0.0 + 0.0im,
                0.0 + 0.0im,
                rhs_pop,
                0.0 + 0.0im,
                0.0 + 0.0im,
                0.0 + 0.0im,
                0.0 + 0.0im,
                0.0 + 0.0im,
                0.0 + 0.0im,
                0.0 + 0.0im,
                rhs_pop,
            )
            return nothing
        end

        # -----------------------------------------------------------------
        # t-local diagonal / coherence phase block.
        # This is self-coupling, so it is always secular.
        # For α == β this block algebraically vanishes.
        # -----------------------------------------------------------------
        rhs_diag = (
            -1.0im * ω(α, β)
            -g′[curr_itr, α, α, α, α]
            +conj(g′[curr_itr, α, α, β, β])
            +g′[curr_itr, β, β, α, α]
            -conj(g′[curr_itr, β, β, β, β])
        ) * σ_now(σ_t, curr_itr, α, β)

        rhs = rhs_diag

        verify_this = should_verify_L0Q(curr_itr, α, β)
        trace_phase_this = should_trace_phase(curr_itr, α, β)
        trace_pop_decomp_this = (α == β) && should_trace_population_decomp(curr_itr, α)

        phase_local_A_raw = 0.0 + 0.0im
        phase_local_B_raw = 0.0 + 0.0im
        phase_local_A_app = 0.0 + 0.0im
        phase_local_B_app = 0.0 + 0.0im
        phase_mem_core_app = 0.0 + 0.0im
        phase_L0Q_A_raw = 0.0 + 0.0im
        phase_L0Q_B_raw = 0.0 + 0.0im
        phase_L0Q_A_app = 0.0 + 0.0im
        phase_L0Q_B_app = 0.0 + 0.0im
        verify_local_A_pop_to_coh = 0.0 + 0.0im
        verify_local_B_pop_to_coh = 0.0 + 0.0im
        verify_local_pop_to_coh   = 0.0 + 0.0im
        verify_L0Q_A_pop          = 0.0 + 0.0im
        verify_L0Q_B_pop          = 0.0 + 0.0im
        verify_L0Q_A_all          = 0.0 + 0.0im
        verify_L0Q_B_all          = 0.0 + 0.0im

        pop_local_A_raw = 0.0 + 0.0im
        pop_local_B_raw = 0.0 + 0.0im
        pop_local_A_app = 0.0 + 0.0im
        pop_local_B_app = 0.0 + 0.0im

        pop_mem_b1_pop_app = 0.0 + 0.0im
        pop_mem_b1_coh_app = 0.0 + 0.0im
        pop_mem_b2_pop_app = 0.0 + 0.0im
        pop_mem_b2_coh_app = 0.0 + 0.0im
        pop_mem_b3_pop_app = 0.0 + 0.0im
        pop_mem_b3_coh_app = 0.0 + 0.0im
        pop_mem_b4_pop_app = 0.0 + 0.0im
        pop_mem_b4_coh_app = 0.0 + 0.0im

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
                local_left_term = σ_now(σ_t, curr_itr, α⁻, β) * (
                    g′[curr_itr, α, α⁻, α⁻, α⁻]
                    - conj(g′[curr_itr, α⁻, α, β, β])
                )

                local_left_contrib = -local_left_term
                phase_local_A_raw += local_left_contrib

                if trace_pop_decomp_this
                    pop_local_A_raw += local_left_contrib
                end

                if verify_this && is_local_population_to_coherence(α, β, α⁻, β)
                    verify_local_A_pop_to_coh += local_left_contrib
                    verify_local_pop_to_coh   += local_left_contrib
                end

                if !use_local_population_to_coherence &&
                   is_local_population_to_coherence(α, β, α⁻, β)
                    continue
                end

                if !use_local_coherence_to_population &&
                   is_local_coherence_to_population(α, β, α⁻, β)
                    continue
                end

                phase_local_A_app += local_left_contrib
                if trace_pop_decomp_this
                    pop_local_A_app += local_left_contrib
                end
                rhs += local_left_contrib
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
                local_right_term = σ_now(σ_t, curr_itr, α, β⁻) * (
                    g′[curr_itr, β⁻, β, α, α]
                    - conj(g′[curr_itr, β, β⁻, β⁻, β⁻])
                )

                local_right_contrib = local_right_term
                phase_local_B_raw += local_right_contrib

                if trace_pop_decomp_this
                    pop_local_B_raw += local_right_contrib
                end

                if verify_this && is_local_population_to_coherence(α, β, α, β⁻)
                    verify_local_B_pop_to_coh += local_right_contrib
                    verify_local_pop_to_coh   += local_right_contrib
                end

                if !use_local_population_to_coherence &&
                   is_local_population_to_coherence(α, β, α, β⁻)
                    continue
                end

                if !use_local_coherence_to_population &&
                   is_local_coherence_to_population(α, β, α, β⁻)
                    continue
                end

                phase_local_B_app += local_right_contrib
                if trace_pop_decomp_this
                    pop_local_B_app += local_right_contrib
                end
                rhs += local_right_contrib
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
                            branch1_contrib = -(
                                σ_mem(σ_t, s_itr, curr_itr, α⁼, β)
                                * phase(α⁻, β, Δ_itr)
                                * exp(gen__exponent_type_1_nonlocal(
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
                            if should_skip_population_memory_input(α, β, α⁼, β)
                                continue
                            end

                            kernel += branch1_contrib
                            phase_mem_core_app += w_int * branch1_contrib

                            if trace_pop_decomp_this
                                if is_population_input(α⁼, β)
                                    pop_mem_b1_pop_app += w_int * branch1_contrib
                                else
                                    pop_mem_b1_coh_app += w_int * branch1_contrib
                                end
                            end
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
                            branch2_contrib = -(
                                σ_mem(σ_t, s_itr, curr_itr, α, β⁼)
                                * phase(α, β⁻, Δ_itr)
                                * exp(gen__exponent_type_2_nonlocal(
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
                            if should_skip_population_memory_input(α, β, α, β⁼)
                                continue
                            end

                            kernel += branch2_contrib
                            phase_mem_core_app += w_int * branch2_contrib

                            if trace_pop_decomp_this
                                if is_population_input(α, β⁼)
                                    pop_mem_b2_pop_app += w_int * branch2_contrib
                                else
                                    pop_mem_b2_coh_app += w_int * branch2_contrib
                                end
                            end
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
                            branch3_contrib = (
                                σ_mem(σ_t, s_itr, curr_itr, α⁻, β⁻)
                                * phase(α, β⁻, Δ_itr)
                                * exp(gen__exponent_type_1_nonlocal(
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
                            if should_skip_population_memory_input(α, β, α⁻, β⁻)
                                continue
                            end

                            kernel += branch3_contrib
                            phase_mem_core_app += w_int * branch3_contrib

                            if trace_pop_decomp_this
                                if is_population_input(α⁻, β⁻)
                                    pop_mem_b3_pop_app += w_int * branch3_contrib
                                else
                                    pop_mem_b3_coh_app += w_int * branch3_contrib
                                end
                            end
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
                            branch4_contrib = (
                                σ_mem(σ_t, s_itr, curr_itr, α⁻, β⁻)
                                * phase(α⁻, β, Δ_itr)
                                * exp(gen__exponent_type_2_nonlocal(
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
                            if should_skip_population_memory_input(α, β, α⁻, β⁻)
                                continue
                            end

                            kernel += branch4_contrib
                            phase_mem_core_app += w_int * branch4_contrib

                            if trace_pop_decomp_this
                                if is_population_input(α⁻, β⁻)
                                    pop_mem_b4_pop_app += w_int * branch4_contrib
                                else
                                    pop_mem_b4_coh_app += w_int * branch4_contrib
                                end
                            end
                        end
                    end
                end

                # ---------------------------------------------------------
                # Missing first-order memory-return term
                #   P L0 Q exp(Δ L0) Q L1 P
                # This term vanishes for population output (α == β), but is
                # generally needed for coherence output.
                #
                # Non-time-localized form is used here:
                #   Branch A uses σ[f,β](s), not σ[f,β](t).
                #   Branch B uses σ[α,f](s), not σ[α,f](t).
                # Therefore the raw output-block propagation phase remains
                # phase(α,β,Δ), and the raw E_A/E_B prefactors are used rather
                # than the transported Λ_A/Λ_B prefactors.
                # ---------------------------------------------------------
                if (use_L0Q_memory_return || verify_this || trace_phase_this) && α != β
                    # Branch A: input (f, β) -> output (α, β), sign = -
                    for f in 1:n_sys
                        f == α && continue

                        if is_secular_pair(α, β, f, β)
                            L0Q_A_term = (
                                σ_mem(σ_t, s_itr, curr_itr, f, β)
                                * phase(α, β, Δ_itr)
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

                            phase_L0Q_A_raw += w_int * L0Q_A_contrib

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
                                phase_L0Q_A_app += w_int * L0Q_A_contrib
                            end
                        end
                    end

                    # Branch B: input (α, f) -> output (α, β), sign = +
                    for f in 1:n_sys
                        f == β && continue

                        if is_secular_pair(α, β, α, f)
                            L0Q_B_contrib = (
                                σ_mem(σ_t, s_itr, curr_itr, α, f)
                                * phase(α, β, Δ_itr)
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

                            phase_L0Q_B_raw += w_int * L0Q_B_contrib

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
                                phase_L0Q_B_app += w_int * L0Q_B_contrib
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

        if trace_phase_this
            time = (curr_itr - 1) * Δt
            c_ab = σ_now(σ_t, curr_itr, α, β)
            rhs_local_raw = phase_local_A_raw + phase_local_B_raw
            rhs_local_app = phase_local_A_app + phase_local_B_app
            rhs_L0Q_raw = phase_L0Q_A_raw + phase_L0Q_B_raw
            rhs_L0Q_app = phase_L0Q_A_app + phase_L0Q_B_app

            print_trace_phase_header_if_needed!()
            lock(trace_phase_lock)
            try
                print__phase_trace_line!(
                    trace_phase_io,
                    curr_itr,
                    time,
                    α,
                    β,
                    c_ab,
                    rhs_diag,
                    rhs_local_raw,
                    rhs_local_app,
                    phase_mem_core_app,
                    rhs_L0Q_raw,
                    rhs_L0Q_app,
                    phase_L0Q_A_raw,
                    phase_L0Q_B_raw,
                    rhs,
                    σprime_heom_now(curr_itr, α, β),
                )
            finally
                unlock(trace_phase_lock)
            end
        end

        if trace_pop_decomp_this
            trace__population_decomp_if_needed!(
                curr_itr,
                σ_t,
                α,
                rhs_diag,
                pop_local_A_raw,
                pop_local_B_raw,
                pop_local_A_app,
                pop_local_B_app,
                pop_mem_b1_pop_app,
                pop_mem_b1_coh_app,
                pop_mem_b2_pop_app,
                pop_mem_b2_coh_app,
                pop_mem_b3_pop_app,
                pop_mem_b3_coh_app,
                pop_mem_b4_pop_app,
                pop_mem_b4_coh_app,
                rhs,
            )
        end

        rhs_mat[α, β] = rhs

        if α == β
            trace__population_rhs_if_needed!(curr_itr, σ_t, α, rhs)
        end

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
                "Current iteration: %6d / %6d  method=%s  threads=%d  use_threads=%s  local_pop_to_coh=%s  local_coh_to_pop=%s  mem_pop_input=%s  mem_coh_input=%s  L0Q_return=%s  verify_L0Q=%s  heom_input=%s  teacher_cutoff=%s
",
                curr_itr,
                n_itr,
                String(method_sym),
                Threads.nthreads(),
                string(use_threads),
                string(use_local_population_to_coherence),
                string(use_local_coherence_to_population),
                string(use_population_memory_population_input),
                string(use_population_memory_coherence_input),
                string(use_L0Q_memory_return),
                string(verify_L0Q_terms),
                string(use_heom_input),
                teacher_forcing_enabled ? string(Float64(heom_teacher_forcing_cutoff)) : "nothing",
            )
        end

        verify__L0P_transport_at!(curr_itr)

        if use_heom_for_storage_itr(curr_itr)
            @views σ[:, :, curr_itr] .= heom_sigma_ref[:, :, curr_itr]
        end

        σ_t    = @view σ[:, :, curr_itr]
        σ_next = @view σ[:, :, curr_itr + 1]
        k1     = @view σ′[:, :, curr_itr]

        if use_heom_input ||
           (teacher_forcing_enabled && curr_itr < teacher_cutoff_itr)

            # Before the teacher-forcing cutoff, evaluate diagnostics/RHS on
            # HEOM and overwrite the next stored state by HEOM.
            # At curr_itr == teacher_cutoff_itr, this branch is skipped and the
            # first self-propagated step begins from the HEOM cutoff state.
            calc__rhs!(k1, curr_itr, σ_t)
            @views σ_next .= heom_sigma_ref[:, :, curr_itr + 1]

        elseif method_sym == :euler
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

