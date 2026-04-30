
module CoherenceMrt

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

@show @isdefined(Patternized_g)
@show @isdefined(Patternized_g′)
@show @isdefined(Patternized_g″)
@show @isdefined(Patternized_Λ)

@isdefined(Patternized_g)  && error("Patternized_g already defined before this file finished loading")
@isdefined(Patternized_g′) && error("Patternized_g′ already defined before this file finished loading")
@isdefined(Patternized_g″) && error("Patternized_g″ already defined before this file finished loading")
@isdefined(Patternized_Λ)  && error("Patternized_Λ already defined before this file finished loading")

@patternized Patternized_Λ (n_sys::Int) (a::Int, b::Int, c::Int, d::Int) begin
    rule(ααββ, Matrix{T}, zeros(T, n_sys, n_sys), (a, a), a == b && b == c && c == d)  # aaaa bbbb
    rule(ααββ, Matrix{T}, zeros(T, n_sys, n_sys), (a, c), a == b && c == d)            # aabb
    rule(αβαα, Matrix{T}, zeros(T, n_sys, n_sys), (a, b), a == c && c == d)            # abaa
    rule(αβββ, Matrix{T}, zeros(T, n_sys, n_sys), (a, b), b == c && c == d)            # abbb
end

@patternized Patternized_g (n_sys::Int, n_itr::Int) (t::Int, a::Int, b::Int, c::Int, d::Int) begin
    rule(ααββ, Array{T,3}, zeros(T, n_itr, n_sys, n_sys), (t, a, c), a == b && c == d)
    rule(αβββ, Array{T,3}, zeros(T, n_itr, n_sys, n_sys), (t, a, b), b == c && c == d)
end

@patternized Patternized_g′ (n_sys::Int, n_itr::Int) (t::Int, a::Int, b::Int, c::Int, d::Int) begin
    rule(ααββ, Array{T,3}, zeros(T, n_itr, n_sys, n_sys), (t, a, c), a == b && c == d)
    rule(αβββ, Array{T,3}, zeros(T, n_itr, n_sys, n_sys), (t, a, b), b == c && c == d)
    rule(αβαα, Array{T,3}, zeros(T, n_itr, n_sys, n_sys), (t, a, b), a == c && c == d)
end

@patternized Patternized_g″ (n_sys::Int, n_itr::Int) (t::Int, a::Int, b::Int, c::Int, d::Int) begin
    rule(αβαβ, Array{T,3}, zeros(T, n_itr, n_sys, n_sys), (t, a, b), a == c && b == d)
    rule(αββα, Array{T,3}, zeros(T, n_itr, n_sys, n_sys), (t, a, b), a == d && b == c)
end

# 매크로가 잘 선언되었나 확인.
@show @isdefined(Patternized_g)
@show @isdefined(Patternized_g′)
@show @isdefined(Patternized_g″)
@show @isdefined(Patternized_Λ)
println(filter(x -> occursin("Patternized", String(x)), string.(names(Main, all=true))))

mutable struct MrtContext
    # input
    system              ::System
    environment         ::Environment
    simulation_details  ::SimulationDetails

    # computing context
    γ_exci              ::Array{ComplexF64, 3}    # 원래는 oscillator 안에 들어가는 coupling strenght 의 exciton verison인데, 음.
    ϵ_exci              ::Vector{Float64}               # energy in exciton basis
    ϵ_exci_0            ::Vector{Float64}               # energy - reorganization energy
    U_sys               ::Matrix{ComplexF64}            # eigenvector matrix
    g                   ::Patternized_g{ComplexF64}
    g′                  ::Patternized_g′{ComplexF64}
    g″                  ::Patternized_g″{ComplexF64}
    Λ                   ::Patternized_Λ{Float64}
    Γ                   ::Array{ComplexF64, 2}

    # output
    transition_rate     ::Array{Float64, 2}
    dissipation         ::Array{Dissipation, 1}


    function MrtContext(system::System, environment::Environment, simulation_details::SimulationDetails)
        
        n_itr = simulation_details.num_of_iteration
        n_sys = system.n_sys
        n_osc = environment.num_of_effective_oscillators

        # H_sys0  = diag(system.H_sys) # 대각성분만 추출.
        γ_exci  = zeros(ComplexF64, (n_osc, n_sys, n_sys))
        ϵ_exci  = zeros(Float64, n_sys)
        ϵ_exci_0= zeros(Float64, n_sys)
        U_sys   = zeros(ComplexF64, (n_sys, n_sys))
        g       = Patternized_g{ComplexF64}(n_sys, n_itr)
        g′      = Patternized_g′{ComplexF64}(n_sys, n_itr)
        g″      = Patternized_g″{ComplexF64}(n_sys, n_itr)
        Λ       = Patternized_Λ{Float64}(n_sys)
        Γ       = zeros(ComplexF64, (n_sys, n_sys))

        transition_rate     = zeros(Float64, (n_sys, n_sys))
        dissipation         = [Dissipation(i, n_sys) for i in 1:n_osc]

        new(system, environment, simulation_details, γ_exci, ϵ_exci, ϵ_exci_0, U_sys, g, g′, g″, Λ, Γ, transition_rate, dissipation)
    end
end

function create__mrt_context(
    system      ::System,
    environment ::Environment,
    simulation_details  ::SimulationDetails
)
    return MrtContext(system, environment, simulation_details)
end

# Equation 65
function calc__Λ!(context::MrtContext)

    n_sys       = context.system.n_sys
    n_osc       = context.environment.num_of_effective_oscillators
    oscs        = context.environment.effective_oscillators

    H_sys       = context.system.H_sys
    U_sys       = context.U_sys
    ϵ_exci      = context.ϵ_exci
    ϵ_exci_0    = context.ϵ_exci_0
    γ_exci      = context.γ_exci    

    Λ           = context.Λ

    eigen_result = eigen!(Hermitian(H_sys))
    ϵ_exci      .= eigen_result.values      # copy
    U_sys       .= eigen_result.vectors     # copy


    # Algorithm start
    fill!(γ_exci, 0)
    # fill!(Λ, 0)

    @inbounds for osc_idx = 1:n_osc
        ω       = oscs[osc_idx].freq
        γ       = oscs[osc_idx].site_bath_coupling_strength
       
        # U_sys is unitary
        γ_exci[osc_idx,:,:] .= U_sys' * γ * U_sys
        
        # 나중에 바꾸고 싶으면 반쪽해서 복사 하든 / 지금은 그냥...
        for β in 1:n_sys, α in 1:n_sys
            if α == β
                Λ[α,α,α,α] += ω * γ_exci[osc_idx,α,α] * γ_exci[osc_idx,α,α]
                continue
            else
                Λ[α,α,β,β] += ω * γ_exci[osc_idx,α,α] * γ_exci[osc_idx,β,β]
                Λ[α,β,α,α] += ω * γ_exci[osc_idx,α,β] * γ_exci[osc_idx,α,α]
                Λ[α,β,β,β] += ω * γ_exci[osc_idx,α,β] * γ_exci[osc_idx,β,β]
            end
        end
    end

    # ϵ_exci_0 생성
    for α in 1:n_sys
        ϵ_exci_0[α] = ϵ_exci[α] - Λ[α,α,α,α]
    end
end

# Equation 65
function calc__Γ!(context::MrtContext)
    
    n_sys       = context.system.n_sys
    n_osc       = context.environment.num_of_effective_oscillators
    γ_exci      = context.γ_exci    
    Γ           = context.Γ

    for β in 1:n_sys, α in 1:n_sys

        Γ_αβ = 0.0im
        for osc_idx in 1:n_osc
            Γ_αβ += 0
            # Γ_αβ += γ_exci[osc_idx,α,β]
        end

        Γ[α,β] = Γ_αβ
    end
end

function calc__g_g′_and_g″!(context::MrtContext)
    n_sys   = context.system.n_sys
    n_osc   = context.environment.num_of_effective_oscillators
    oscs    = context.environment.effective_oscillators

    n_itr   = context.simulation_details.num_of_iteration
    Δt      = context.simulation_details.Δt

    g       = context.g
    g′      = context.g′
    g″      = context.g″

    γ_exci  = context.γ_exci

    # fill!(g , 0)
    # fill!(g′, 0)
    # fill!(g″, 0)

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

                # 같을 때만 따로 처리... 
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

function calc__g_g′_and_g″_with_threads!(context::MrtContext)
    n_sys   = context.system.n_sys
    n_osc   = context.environment.num_of_effective_oscillators
    oscs    = context.environment.effective_oscillators

    n_itr   = context.simulation_details.num_of_iteration
    Δt      = context.simulation_details.Δt

    g       = context.g
    g′      = context.g′
    g″      = context.g″

    γ_exci  = context.γ_exci

    # fill!(g , 0)
    # fill!(g′, 0)
    # fill!(g″, 0)

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

function calc__rates!(context::MrtContext)

    n_sys       = context.system.n_sys

    n_itr       = context.simulation_details.num_of_iteration
    Δt          = context.simulation_details.Δt

    g           = context.g
    g′          = context.g′
    g″          = context.g″
    Λ           = context.Λ

    ϵ_exci_0    = context.ϵ_exci_0

    rate        = context.transition_rate
    
    @inbounds for β in 1:n_sys, α in 1:n_sys
        if α == β; continue; end

        ϵ_α0 = ϵ_exci_0[α]
        ϵ_β0 = ϵ_exci_0[β]

        Λ_αααα,  Λ_ααββ,  Λ_ββββ  = Λ[α,α,α,α], Λ[α,α,β,β], Λ[β,β,β,β]
        Λ_αβαα,  Λ_βααα           = Λ[α,β,α,α], Λ[β,α,α,α]

        integral = 0.0
        @inbounds for time_idx in 1:n_itr
            t = (time_idx - 1) * Δt

            g_αααα,  g_ααββ,  g_ββββ  = g[time_idx, α,α,α,α], g[time_idx, α,α,β,β], g[time_idx, β,β,β,β]
            g′_αβββ, g′_αβαα          = g′[time_idx, α,β,β,β], g′[time_idx, α,β,α,α]
            g′_βαββ, g′_βααα          = g′[time_idx, β,α,β,β], g′[time_idx, β,α,α,α]
            g″_αββα                   = g″[time_idx, α,β,β,α]

            # Equation 72
            exponent = -1.0im*t*(ϵ_β0 - ϵ_α0) - (g_αααα - 2.0*g_ααββ + g_ββββ) - 1.0im*t*(Λ_αααα - 2.0*Λ_ααββ + Λ_ββββ)
            integrand = ( -(g′_αβββ - g′_αβαα - 2.0im*Λ_αβαα) * (g′_βαββ - g′_βααα - 2.0im*Λ_βααα) + g″_αββα ) * exp(exponent)

            # trapezoidal method 적분 시작
            trapezoidal_weight = (time_idx == 1 || time_idx == n_itr) ? 0.5 : 1.0

            integral += trapezoidal_weight * real(integrand)
        end

        prefactor = 2.0 * Δt
        rate[α,β] = prefactor * integral
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

function calc__dissipations!(context::MrtContext)

    n_sys       = context.system.n_sys
    n_osc       = context.environment.num_of_effective_oscillators
    oscs        = context.environment.effective_oscillators

    dissipation = context.dissipation

    n_itr       = context.simulation_details.num_of_iteration
    Δt          = context.simulation_details.Δt

    γ_exci     = context.γ_exci
    ϵ_exci      = context.ϵ_exci
    ϵ_exci_0    = context.ϵ_exci_0
    Λ           = context.Λ
    Γ           = context.Γ
    g           = context.g
    g′          = context.g′
    g″          = context.g″

    @printf(stderr, "---- Calculating MRT dissipation parameters ----\n")

    @inbounds for β in 1:n_sys, α in 1:n_sys

        if β == α continue end

        ϵ_α, ϵ_β = ϵ_exci[α], ϵ_exci[β]
        ϵ_α0, ϵ_β0 = ϵ_exci_0[α], ϵ_exci_0[β]

        Λ_αααα,  Λ_ααββ, Λ_ββββ     = Λ[α,α,α,α], Λ[α,α,β,β], Λ[β,β,β,β]
        Λ_αβαα,  Λ_βααα             = Λ[α,β,α,α], Λ[β,α,α,α]
    
        # 이것도 마찬가지 에너지 shift 합이지. exciton coupling gamma의 합이 아님
        Γ_αα, Γ_αβ, Γ_βα, Γ_ββ      = Γ[α,α], Γ[α,β], Γ[β,α], Γ[β,β]
        Γ_αα, Γ_αβ, Γ_βα, Γ_ββ      = 0.0, 0.0, 0.0, 0.0

        @inbounds for osc_idx in 1:n_osc

            ω        = oscs[osc_idx].freq
            coth     = oscs[osc_idx].coth
            spreadʲ  = oscs[osc_idx].spread
            γʲ_exci  = @views γ_exci[osc_idx,:,:]

            # λʲ 까지 객체로 저장하면, 메모리 낭비 심하니.. iteration에서 바로 계산.
            γʲ_αα, γʲ_αβ, γʲ_βα, γʲ_ββ  = γʲ_exci[α,α], γʲ_exci[α,β], γʲ_exci[β,α], γʲ_exci[β,β]

            λʲ_αααα, λʲ_ββββ            = ω * γʲ_αα * γʲ_αα, ω * γʲ_ββ * γʲ_ββ
            λʲ_ααββ, λʲ_αββα            = ω * γʲ_αα * γʲ_ββ, ω * γʲ_αβ * γʲ_βα
            λʲ_βαββ, λʲ_βααα            = ω * γʲ_βα * γʲ_ββ, ω * γʲ_βα * γʲ_αα
            λʲ_αβββ, λʲ_αβαα            = ω * γʲ_αβ * γʲ_ββ, ω * γʲ_αβ * γʲ_αα

            # 착각했어 γʲ_αα, γʲ_αβ, γʲ_βα, γʲ_ββ 는 에너지 shift이지,. exciton coupling gamma가 아님!!!
            # 아마 ω dʲ_αα/ √2 였어야 하는건데.
            γʲ_αα, γʲ_αβ, γʲ_βα, γʲ_ββ  = 0.0, 0.0, 0.0, 0.0

            integral = 0.0
            @inbounds for time_idx in 1:n_itr

                g_αααα, g_ααββ, g_ββββ    = g[time_idx, α,α,α,α], g[time_idx, α,α,β,β], g[time_idx, β,β,β,β]

                g′_αααα, g′_ααββ, g′_ββββ = g′[time_idx, α,α,α,α], g′[time_idx, α,α,β,β], g′[time_idx, β,β,β,β]
                g′_αβββ, g′_αβαα          = g′[time_idx, α,β,β,β], g′[time_idx, α,β,α,α]
                g′_βαββ, g′_βααα          = g′[time_idx, β,α,β,β], g′[time_idx, β,α,α,α]
                g″_αββα                   = g″[time_idx, α,β,β,α]

                t   = Δt * (time_idx - 1)
                ωt  = ω*t

                sin_ωt, cos_ωt = sincos(ωt)

                # Equation 62, f
                f′ = (coth * sin_ωt)            + 1.0im*(cos_ωt - 1)
                f″ = (coth * cos_ωt * ω)        + 1.0im*(-sin_ωt * ω)
                f‴ = (coth * (-sin_ωt) * ω^2)   + 1.0im*(-cos_ωt * ω^2)

                # Equation 60, 61
                Gʲ_βα   = (λʲ_αααα - 2.0*λʲ_ααββ + λʲ_ββββ)
                Δʲ_βα   = (λʲ_αααα - λʲ_ββββ) - (γʲ_αα - γʲ_ββ)
                # Equation 64
                W_βα    = -1.0im*(g′_αβββ - g′_αβαα) - 2.0*Λ_αβαα + Γ_αβ                
                X_βα    = -1.0im*(g′_βαββ - g′_βααα) - 2.0*Λ_βααα + Γ_βα    
                Π_βα    = exp(-1.0im*t*(2.0*Λ_αααα - 2.0*Λ_ααββ - Γ_αα + Γ_ββ) - (g_αααα -2.0*g_ααββ + g_ββββ))
                # Equation 69 ??? 25 ???
                # 𝒩_βα
                # Equation 69는 Γ를 0으로 둔 표준 MRT 기준으로 전개했음...
                S_βα    = ( -(g′_αβββ - g′_αβαα - 2.0im*Λ_αβαα) * (g′_βαββ - g′_βααα - 2.0im*Λ_βααα) + g″_αββα ) * exp(-2.0im*t*(Λ_αααα-Λ_ααββ) - (g_αααα -2.0*g_ααββ + g_ββββ))
                # println(S_βα, " =?= ", (W_βα * X_βα + g″_αββα) * Π_βα)
            

                # Equation 76
                Jʲ_βα   = 
                    (Δʲ_βα + Gʲ_βα - 1.0im * Gʲ_βα * f′) * S_βα + 
                    (
                        (λʲ_βαββ - λʲ_βααα) * f″ * W_βα + 
                        (λʲ_αβββ - λʲ_αβαα) * f″ * X_βα +
                        1.0im * λʲ_αββα * f‴ 
                    ) * Π_βα

                trapezoidal_weight  = (time_idx == 1 || time_idx == n_itr) ? 0.5 : 1.0
                exponent            = -1.0im * t * (ϵ_β - ϵ_α)
                integrand           = Jʲ_βα * exp(exponent)
                # 𝒥ʲ_βα =
                
                # Equation 36
                integral += trapezoidal_weight * real(integrand)
                # @printf(stderr, "%15.6f \n", integral)
                # @printf(stderr, "%15.6f \n", real(integrand))
            end

            dissipation[osc_idx].k_dissipation[α,β]   = 2.0 * Δt * integral
            # dissipation.i_dissipation   = dissipation.k_dissipation / γ_exci
            dissipation[osc_idx].j_dissipation[α,β]   = dissipation[osc_idx].k_dissipation[α,β] / spreadʲ

            # 𝒦ʲ_

            if osc_idx % 100 == 0
                @printf(stderr, "%3d -> %3d    OSC %6d / %6d\n", α, β, osc_idx, n_osc)
            end
        end
    end
end


function calc__dissipations_with_threads!(context::MrtContext)

    n_sys       = context.system.n_sys
    n_osc       = context.environment.num_of_effective_oscillators
    oscs        = context.environment.effective_oscillators

    dissipation = context.dissipation

    n_itr       = context.simulation_details.num_of_iteration
    Δt          = context.simulation_details.Δt

    γ_exci     = context.γ_exci
    ϵ_exci      = context.ϵ_exci
    ϵ_exci_0    = context.ϵ_exci_0
    Λ           = context.Λ
    Γ           = context.Γ
    g           = context.g
    g′          = context.g′
    g″          = context.g″

    @printf(stderr, "---- Calculating MRT dissipation parameters ----\n")

    @inbounds for β in 1:n_sys, α in 1:n_sys

        if β == α continue end

        ϵ_α, ϵ_β = ϵ_exci[α], ϵ_exci[β]

        Λ_αααα,  Λ_ααββ, Λ_ββββ     = Λ[α,α,α,α], Λ[α,α,β,β], Λ[β,β,β,β]
        Λ_αβαα,  Λ_βααα             = Λ[α,β,α,α], Λ[β,α,α,α]
    
        # 이것도 마찬가지 에너지 shift 합이지. exciton coupling gamma의 합이 아님
        # 근데 지금 이거 재할당의해서 캡처시 Boxed 되어버리니까 일단 주석처리.
        # Γ_αα, Γ_αβ, Γ_βα, Γ_ββ      = Γ[α,α], Γ[α,β], Γ[β,α], Γ[β,β]
        Γ_αα, Γ_αβ, Γ_βα, Γ_ββ      = 0.0, 0.0, 0.0, 0.0

        # Oscillator 별 독립적인 계산이라 local 변수도 필요 없고
        # 멀티쓰레드 시, 작업 크기도 거의 비슷하니... static scheduling ( :static )
        @inbounds @threads :static for osc_idx in 1:n_osc

            ω        = oscs[osc_idx].freq
            coth     = oscs[osc_idx].coth
            spreadʲ  = oscs[osc_idx].spread
            γʲ_exci  = @views γ_exci[osc_idx,:,:]

            # λʲ 까지 객체로 저장하면, 메모리 낭비 심하니.. iteration에서 바로 계산.
            γʲ_αα, γʲ_αβ, γʲ_βα, γʲ_ββ  = γʲ_exci[α,α], γʲ_exci[α,β], γʲ_exci[β,α], γʲ_exci[β,β]

            λʲ_αααα, λʲ_ββββ            = ω * γʲ_αα * γʲ_αα, ω * γʲ_ββ * γʲ_ββ
            λʲ_ααββ, λʲ_αββα            = ω * γʲ_αα * γʲ_ββ, ω * γʲ_αβ * γʲ_βα
            λʲ_βαββ, λʲ_βααα            = ω * γʲ_βα * γʲ_ββ, ω * γʲ_βα * γʲ_αα
            λʲ_αβββ, λʲ_αβαα            = ω * γʲ_αβ * γʲ_ββ, ω * γʲ_αβ * γʲ_αα

            # 착각했어 γʲ_αα, γʲ_αβ, γʲ_βα, γʲ_ββ 는 에너지 shift이지,. exciton coupling gamma가 아님!!!
            # 아마 ω dʲ_αα/ √2 였어야 하는건데.
            γʲ_αα, γʲ_αβ, γʲ_βα, γʲ_ββ  = 0.0, 0.0, 0.0, 0.0

            ####################################################

            # Equation 60, 61
            Gʲ_βα   = (λʲ_αααα - 2.0*λʲ_ααββ + λʲ_ββββ)
            Δʲ_βα   = (λʲ_αααα - λʲ_ββββ) - (γʲ_αα - γʲ_ββ)

            integral = 0.0
            @inbounds for time_idx in 1:n_itr

                g_αααα, g_ααββ, g_ββββ    = g[time_idx, α,α,α,α], g[time_idx, α,α,β,β], g[time_idx, β,β,β,β]

                g′_αβββ, g′_αβαα          = g′[time_idx, α,β,β,β], g′[time_idx, α,β,α,α]
                g′_βαββ, g′_βααα          = g′[time_idx, β,α,β,β], g′[time_idx, β,α,α,α]
                g″_αββα                   = g″[time_idx, α,β,β,α]

                t   = Δt * (time_idx - 1)
                ωt  = ω*t

                sin_ωt, cos_ωt = sincos(ωt)

                # Equation 62, f
                f′ = (coth * sin_ωt)            + 1.0im*(cos_ωt - 1)
                f″ = (coth * cos_ωt * ω)        + 1.0im*(-sin_ωt * ω)
                f‴ = (coth * (-sin_ωt) * ω^2)   + 1.0im*(-cos_ωt * ω^2)

                # Equation 64
                W_βα    = -1.0im*(g′_αβββ - g′_αβαα) - 2.0*Λ_αβαα + Γ_αβ                
                X_βα    = -1.0im*(g′_βαββ - g′_βααα) - 2.0*Λ_βααα + Γ_βα    
                Π_βα    = exp(-1.0im*t*(2.0*Λ_αααα - 2.0*Λ_ααββ - Γ_αα + Γ_ββ) - (g_αααα -2.0*g_ααββ + g_ββββ))
                # Equation 25
                S_βα    = (W_βα * X_βα + g″_αββα) * Π_βα
            
                # Equation 76
                𝒥ʲ_βα   = 
                    (Δʲ_βα + Gʲ_βα - 1.0im * Gʲ_βα * f′) * S_βα + 
                    (
                        (λʲ_βαββ - λʲ_βααα) * f″ * W_βα + 
                        (λʲ_αβββ - λʲ_αβαα) * f″ * X_βα +
                        1.0im * λʲ_αββα * f‴ 
                    ) * Π_βα

                trapezoidal_weight  = (time_idx == 1 || time_idx == n_itr) ? 0.5 : 1.0
                exponent            = -1.0im * t * (ϵ_β - ϵ_α)
                integrand           = exp(exponent) * 𝒥ʲ_βα

                # Equation 36
                integral += trapezoidal_weight * real(integrand)
            end

            dissipation[osc_idx].k_dissipation[α,β]   = 2.0 * Δt * integral
            # dissipation.i_dissipation   = dissipation.k_dissipation / γ_exci
            dissipation[osc_idx].j_dissipation[α,β]   = dissipation[osc_idx].k_dissipation[α,β] / spreadʲ

            if osc_idx % 100 == 0
                @printf(stderr, "%3d -> %3d    OSC %6d / %6d\n", α, β, osc_idx, n_osc)
            end
        end
    end
end

function check__physics(context::MrtContext)

    n_sys       = context.system.n_sys
    n_osc       = context.environment.num_of_effective_oscillators
    oscs        = context.environment.effective_oscillators

    transition_rate = context.transition_rate
    dissipation     = context.dissipation
    ϵ_exci_0        = context.ϵ_exci_0
    ϵ_exci          = context.ϵ_exci

    # temporary T for termperature of first oscillator
    T           = oscs[1].temperature


    @printf(stderr, "Detailed Balance for Dissipation\n")

    # for β in 1:n_sys, α in 1:(β-1)

    #     for osc_idx in 1:n_osc

    #         ω               = oscs[osc_idx].freq
    #         j_dissipation   = dissipation[osc_idx].j_dissipation

    #         # Equation 46
    #         diss_ratio      = -j_dissipation[α,β] / j_dissipation[β,α]
    #         boltz_ratio     = exp(-(ϵ_exci_0[α] - ϵ_exci_0[β]) / T)

    #         @printf(stderr,
    #             "%3d -> %3d, OSC %5d   freq %13.6le   diss_ratio %13.6le   Boltzmann_ratio %13.6le\n",
    #             α, β, osc_idx, ω, diss_ratio, boltz_ratio
    #         )
    #     end
    # end

    # for β in 1:n_sys, α in 1:(β-1)
    #     for osc_idx in 1:n_osc

    #         ω               = oscs[osc_idx].freq
    #         k_dissipation   = dissipation[osc_idx].k_dissipation

    #         # Equation 46, 52
    #         diss_ratio      = -k_dissipation[α,β] / k_dissipation[β,α]
    #         boltz_ratio     = exp(-(ϵ_exci[α] - ϵ_exci[β]) / T)

    #         @printf(stderr,
    #             "%3d -> %3d, OSC %5d   freq %13.6le   diss_ratio %13.6le   Boltzmann_ratio %13.6le\n",
    #             α, β, osc_idx, ω, diss_ratio, boltz_ratio
    #         )
    #     end
    # end

    for β in 1:n_sys, α in 1:(β-1)
        for osc_idx in 1:n_osc
            ω               = oscs[osc_idx].freq

            𝒦ʲ      = dissipation[osc_idx].k_dissipation
            𝒦ʲ_αβ   = 𝒦ʲ[α,β]
            𝒦ʲ_βα   = 𝒦ʲ[β,α]

            K_αβ = transition_rate[α,β]
            K_βα = transition_rate[β,α]

            # Equation 46, 52
            diss_ratio      = -𝒦ʲ_αβ / 𝒦ʲ_βα
            rate_ratio      = K_αβ / K_βα
            boltz_ratio     = exp(-(ϵ_exci[β] - ϵ_exci[α]) / T)
            # 실제로는 exp E_alpha + V_alphaalpha 일텐데, boltz ratio는 그냥 

            if (osc_idx-1) % 50 == 0
                @printf(stderr,
                    "%3d -> %3d, OSC %5d | freq %13.6le | diss_ratio %13.6le | rate_ratio %13.6le |  boltz_ratio %13.6le\n",
                    α, β, osc_idx, ω, diss_ratio, rate_ratio, boltz_ratio
                )
            end
        end
    end
    
    

    @printf(stderr, "Energy Conservation\n")

    # for β in 1:n_sys, α in 1:(β-1)

    #     Edec_elec_αβ = -(ϵ_exci_0[α] - ϵ_exci_0[β]) * transition_rate[α,β]  # C: rate[i + nsys*j]
    #     Edec_elec_βα = -(ϵ_exci_0[β] - ϵ_exci_0[α]) * transition_rate[β,α]  # C: rate[j + nsys*i]

    #     Einc_diss_αβ = 0.0
    #     Einc_diss_βα = 0.0

    #     for osc_idx in 1:n_osc

    #         k_dissipation   = dissipation[osc_idx].k_dissipation

    #         Einc_diss_αβ += k_dissipation[α,β]
    #         Einc_diss_βα += k_dissipation[β,α]
    #     end

    #     @printf(stderr, "%2d -> %2d  Elec %13.6le  Diss %13.6le\n", α, β, Edec_elec_βα, Einc_diss_βα)
    #     @printf(stderr, "%2d -> %2d  Elec %13.6le  Diss %13.6le\n", β, α, Edec_elec_αβ, Einc_diss_αβ)
    # end

    for β in 1:n_sys, α in 1:(β-1)

        Edec_elec_αβ = -(ϵ_exci[α] - ϵ_exci[β]) * transition_rate[α,β]  # C: rate[i + nsys*j]
        Edec_elec_βα = -(ϵ_exci[β] - ϵ_exci[α]) * transition_rate[β,α]  # C: rate[j + nsys*i]

        Einc_diss_αβ = 0.0
        Einc_diss_βα = 0.0

        for osc_idx in 1:n_osc
            j      = osc_idx
            𝒦ʲ      = dissipation[j].k_dissipation
            𝒦ʲ_αβ   = 𝒦ʲ[α,β]
            𝒦ʲ_βα   = 𝒦ʲ[β,α]

            Einc_diss_αβ += 𝒦ʲ_αβ
            Einc_diss_βα += 𝒦ʲ_βα
        end

        @printf(stderr, "%2d -> %2d  Elec %13.6le  Diss %13.6le\n", α, β, Edec_elec_βα, Einc_diss_βα)
        @printf(stderr, "%2d -> %2d  Elec %13.6le  Diss %13.6le\n", β, α, Edec_elec_αβ, Einc_diss_αβ)
    end

    return nothing
end



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
    context             ::MrtContext,
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
