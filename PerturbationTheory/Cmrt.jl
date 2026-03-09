
include("../Physics/Physics.jl")
using Base.Threads
using LinearAlgebra
import Base: getindex

mutable struct Patternized_Λ{T}
    ααββ::Matrix{T}   # (α, β)  — includes αααα, ααββ, ββββ
    αβαα::Matrix{T}   # (α, β)  — only for α ≠ β
    αβββ::Matrix{T}

    function Patternized_Λ{T}(n_sys::Int) where {T}
        new(zeros(T, n_sys, n_sys), zeros(T, n_sys, n_sys), zeros(T, n_sys, n_sys))
    end
end

mutable struct Patternized_g{T}
    ααββ::Array{T,3}   # (t, α, β)
    αβββ::Array{T,3}   # (t, α, β)
    function Patternized_g{T}(n_sys::Int, n_itr::Int) where {T}
        new(zeros(T, n_itr, n_sys, n_sys), zeros(T, n_itr, n_sys, n_sys))
    end
end

mutable struct Patternized_g′{T}
    ααββ::Array{T,3}   # (t, α, β)
    αβββ::Array{T,3}   # (t, α, β)
    αβαα::Array{T,3}   # (t, α, β)
    function Patternized_g′{T}(n_sys::Int, n_itr::Int) where {T}
        new(zeros(T, n_itr, n_sys, n_sys), zeros(T, n_itr, n_sys, n_sys), zeros(T, n_itr, n_sys, n_sys))
    end
end

mutable struct Patternized_g″{T}
    αβαβ::Array{T,3}   # (t, α, β)
    αββα::Array{T,3}   # (t, α, β)
    function Patternized_g″{T}(n_sys::Int, n_itr::Int) where {T}
        new(zeros(T, n_itr, n_sys, n_sys), zeros(T, n_itr, n_sys, n_sys))
    end
end

@inline Base.getindex(Λ::Patternized_Λ{T}, a::Int,b::Int,c::Int,d::Int) where {T} =
    a==b && b==c && c==d ? Λ.ααββ[a,a] : # aaaa bbbb
    a==b && c==d         ? Λ.ααββ[a,c] : # aabb
    a==c && c==d         ? Λ.αβαα[a,b] : # abaa
    b==c && c==d         ? Λ.αβββ[a,b] : # abbb
    error("unsupported Λ pattern")

@inline Base.setindex!(Λ::Patternized_Λ{T}, v, a::Int,b::Int,c::Int,d::Int) where {T} =
    a==b && b==c && c==d ? (Λ.ααββ[a,a]=v) :
    a==b && c==d         ? (Λ.ααββ[a,c]=v) :
    a==c && c==d         ? (Λ.αβαα[a,b]=v) :
    b==c && c==d         ? (Λ.αβββ[a,b]=v) :
    error("unsupported Λ pattern")

@inline Base.getindex(g::Patternized_g{T}, t::Int,a::Int,b::Int,c::Int,d::Int) where {T} =
    a==b && c==d ? g.ααββ[t,a,c] :
    b==c && c==d ? g.αβββ[t,a,b] :
    error("unsupported g pattern")

@inline Base.setindex!(g::Patternized_g{T}, v, t::Int,a::Int,b::Int,c::Int,d::Int) where {T} =
    a==b && c==d ? (g.ααββ[t,a,c]=v) :
    b==c && c==d ? (g.αβββ[t,a,b]=v) :
    error("unsupported g pattern")

@inline Base.getindex(g′::Patternized_g′{T}, t::Int,a::Int,b::Int,c::Int,d::Int) where {T} =
    a==b && c==d            ? g′.ααββ[t,a,c] :
    b==c && c==d            ? g′.αβββ[t,a,b] :
    a==c && c==d            ? g′.αβαα[t,a,b] :
    error("unsupported g′ pattern")

@inline Base.setindex!(g′::Patternized_g′{T}, v, t::Int,a::Int,b::Int,c::Int,d::Int) where {T} =
    a==b && c==d            ? (g′.ααββ[t,a,c]=v) :
    b==c && c==d            ? (g′.αβββ[t,a,b]=v) :
    a==c && c==d            ? (g′.αβαα[t,a,b]=v) :
    error("unsupported g′ pattern")

@inline Base.getindex(g″::Patternized_g″{T}, t::Int,a::Int,b::Int,c::Int,d::Int) where {T} =
    a==c && b==d ? g″.αβαβ[t,a,b] :
    a==d && b==c ? g″.αββα[t,a,b] :
    error("unsupported g″ pattern")

@inline Base.setindex!(g″::Patternized_g″{T}, v, t::Int,a::Int,b::Int,c::Int,d::Int) where {T} =
    a==c && b==d ? (g″.αβαβ[t,a,b]=v) :
    a==d && b==c ? (g″.αββα[t,a,b]=v) :
    error("unsupported g″ pattern")

# for threads
@inline function inplace_add!(dest::Patternized_g{T}, src::Patternized_g{T}) where {T}
    dest.ααββ .+= src.ααββ
    return dest
end
@inline function inplace_add!(dest::Patternized_g′{T}, src::Patternized_g′{T}) where {T}
    dest.ααββ .+= src.ααββ 
    dest.αβββ .+= src.αβββ
    dest.αβαα .+= src.αβαα
    return dest
end
@inline function inplace_add!(dest::Patternized_g″{T}, src::Patternized_g″{T}) where {T}
    dest.αββα .+= src.αββα
    return dest
end

mutable struct CmrtContext
    # input
    system              ::System
    environment         ::Environment
    simulation_details  ::SimulationDetails

    # computing context
    γ_exci              ::Array{ComplexF64, 3}    # 원래는 oscillator 안에 들어가는 coupling strenght 의 exciton verison인데, 음.
    ϵ_exci              ::Vector{Float64}               # energy in exciton basis
    ϵ_exci_0            ::Vector{Float64}               # energy - reorganization energy
    U_sys               ::Matrix{ComplexF64}            # engenvector matrix
    g                   ::Patternized_g{ComplexF64}
    g′                  ::Patternized_g′{ComplexF64}
    g″                  ::Patternized_g″{ComplexF64}
    Λ                   ::Patternized_Λ{Float64}
    Γ                   ::Array{ComplexF64, 2}

    # output
    transition_rate     ::Array{Float64, 2}
    dissipation         ::Array{Dissipation, 1}


    function CmrtContext(system::System, environment::Environment, simulation_details::SimulationDetails)
        
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

function create__cmrt_context(
    system      ::System,
    environment ::Environment,
    simulation_details  ::SimulationDetails
)
    return CmrtContext(system, environment, simulation_details)
end

# Equation 65
function calc__Λ!(context::CmrtContext)

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
function calc__Γ!(context::CmrtContext)
    
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

function calc__g_g′_and_g″!(context::CmrtContext)
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

function calc__g_g′_and_g″_with_threads!(context::CmrtContext)
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

function calc__rates!(context::CmrtContext)

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

function calc__dissipations!(context::CmrtContext)

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


function calc__dissipations_with_threads!(context::CmrtContext)

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

function check__physics(context::CmrtContext)

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

# CMRT 논문에서... 실제 rate 를 '진화' 시키는코드, 처음으로 초기값이 들어감!
function calc__σ(context::CmrtContext)

    n_sys       = context.system.n_sys
    n_itr       = context.simulation_details.num_of_iteration
    Δt          = context.simulation_details.Δt

    g           = context.g
    g′          = context.g′
    g″          = context.g″

    σ           = zeros(ComplexF64, (n_sys, n_sys, n_itr))
    current_σ′  = zeros(ComplexF64, (n_sys, n_sys))

    @inbounds for time_idx in 1:n_itr
        t = (time_idx - 1) * Δt

        for β in 1:n_sys, α in 1:n_sys

            # σ[α,β]
        end
    end
end

# pure dephasing 항을 구해요.
function calc__R_pd!(ctx::CmrtContext)
    n_sys   = ctx.system.n_sys
    n_itr   = ctx.simulation_details.num_of_iteration

    g′      = ctx.g′

    R_pd = zeros(ComplexF64, n_itr, n_sys, n_sys)

    @inbounds for time_idx in 1:n_itr
        for β in 1:n_sys, α in 1:n_sys
            if α == β
                R_pd[time_idx,α,β] = 0.0 + 0.0im
                continue
            end
            g′_αααα = g′[time_idx, α,α,α,α]
            g′_ββββ = g′[time_idx, β,β,β,β]
            g′_ααββ = g′[time_idx, α,α,β,β]

            real_part = real(g′_αααα + g′_ββββ - 2.0*g′_ααββ)
            imag_part = imag(g′_αααα - g′_ββββ)   # this goes into i*Im[...]

            R_pd[time_idx,α,β] = real_part + 1.0im*imag_part
        end
    end

    return R_pd
end

#dissipation rate (population transfer rate) 를 구해요.

"""
Compute time-dependent population transfer rate R_hist[t,a,b] defined by:

R_ab(t) = 2 Re ∫_0^t dτ  F_b^*(τ) A_a(τ) X_ab(τ)

We discretize τ on the same grid as (g,g′,g″) and build cumulative integrals
with the trapezoidal rule, so R_hist[ti,a,b] corresponds to time t=(ti-1)*dt.

Returns:
- R_hist :: Array{Float64,3}  size (n_itr,n_sys,n_sys), with R[:,a,a]=0
"""
function calc__R_hist!(ctx::CmrtContext)
    n_sys = ctx.system.n_sys
    n_itr = ctx.simulation_details.num_of_iteration
    dt    = ctx.simulation_details.Δt

    ϵ = ctx.ϵ_exci
    Λ = ctx.Λ
    g = ctx.g
    g′ = ctx.g′
    g″ = ctx.g″

    R_hist = zeros(Float64, n_itr, n_sys, n_sys)

    # loop over (a,b), a!=b
    @inbounds for a in 1:n_sys, b in 1:n_sys
        if a == b
            continue
        end

        λ_bbbb = Λ[b,b,b,b]
        λ_aabb = Λ[a,a,b,b]
        λ_babb = Λ[b,a,b,b]   # (b,a,b,b)
        λ_abbb = Λ[a,b,b,b]   # (a,b,b,b)

        # cummulator
        I = 0.0 + 0.0im

        # ti=1 corresponds to t=0 -> integral 0
        R_hist[1,a,b] = 0.0

        for ti in 2:n_itr
            # trapezoid on [t_{ti-1}, t_{ti}]
            t1 = (ti-2)*dt
            t2 = (ti-1)*dt

            # integrand at ti-1 and ti
            f1 = _cmrt_integrand(ctx, a, b, ti-1, t1, ϵ, Λ, g, g′, g″, λ_bbbb, λ_aabb, λ_babb, λ_abbb)
            f2 = _cmrt_integrand(ctx, a, b, ti,   t2, ϵ, Λ, g, g′, g″, λ_bbbb, λ_aabb, λ_babb, λ_abbb)

            I += 0.5*dt*(f1 + f2)
            R_hist[ti,a,b] = 2.0*real(I)
        end
    end

    return R_hist
end


# --- internal helper: integrand = F_b^* A_a X_ab ---
@inline function _cmrt_integrand(
    ctx::CmrtContext,
    a::Int, b::Int, ti::Int, t::Float64,
    ϵ, Λ, g, g′, g″,
    λ_bbbb, λ_aabb, λ_babb, λ_abbb
)
    # A_a(t) = exp(-i ε_a t - g_aaaa(t))
    g_aaaa = g[ti, a,a,a,a]
    A_a = exp(-1.0im*ϵ[a]*t - g_aaaa)

    # F_b(t) = exp(-i(ε_b - 2λ_bbbb) t - g_bbbb^*(t))
    # so F_b^*(t) = exp(+i(ε_b - 2λ_bbbb) t - g_bbbb(t))
    g_bbbb = g[ti, b,b,b,b]
    F_b_conj = exp(+1.0im*(ϵ[b] - 2.0*λ_bbbb)*t - g_bbbb)

    # X_ab(t):
    # pref = exp( 2*( g_aabb(t) + i λ_aabb t ) )
    g_aabb = g[ti, a,a,b,b]
    pref = exp( 2.0*(g_aabb + 1.0im*λ_aabb*t) )

    # ddot g_{b a a b}(t) = g″[ti, b,a,a,b]  -> pattern a==d && b==c
    ddg_baab = g″[ti, b,a,a,b]

    # dot g terms:
    # dot g_{b a a a} = g′[ti, b,a,a,a]  -> αβββ with (b,a)
    dg_baaa = g′[ti, b,a,a,a]
    # dot g_{b a b b} = g′[ti, b,a,b,b]  -> αβαα with (b,a)
    dg_babb = g′[ti, b,a,b,b]

    # dot g_{a b a a} = g′[ti, a,b,a,a]  -> αβαα with (a,b)
    dg_abaa = g′[ti, a,b,a,a]
    # dot g_{a b b b} = g′[ti, a,b,b,b]  -> αβββ with (a,b)
    dg_abbb = g′[ti, a,b,b,b]

    term1 = dg_baaa - dg_babb - 2.0im*λ_babb
    term2 = dg_abaa - dg_abbb - 2.0im*λ_abbb

    X_ab = pref * (ddg_baab - term1*term2)

    return F_b_conj * A_a * X_ab
end

# ------------------------------------------------------------
# Reduced dynamics with time-dependent R(t), Rpd(t)
# ------------------------------------------------------------

@inline function _interp_val(v1, v2, θ)
    return (1.0-θ)*v1 + θ*v2
end

"""
Simulate σ(t) using time-dependent R_hist and Rpd_hist.

- R_hist[ti,a,b] = R_ab(t_i) with t_i = (ti-1)dt, and R_aa=0
- Rpd_hist[ti,a,b] = Rpd_ab(t_i), complex allowed (as in your formula)

Returns:
- σ_hist :: Array{ComplexF64,3} (n_sys,n_sys,n_itr)
"""
function simulate__sigma_cmrt_time_dependent(
    ctx::CmrtContext;
    σ0::AbstractMatrix{<:Complex},
    R_hist::Array{Float64,3},
    Rpd_hist::Array{ComplexF64,3},
    method::Symbol = :rk4,
)
    n_sys = ctx.system.n_sys
    n_itr = ctx.simulation_details.num_of_iteration
    dt    = ctx.simulation_details.Δt
    ϵ     = ctx.ϵ_exci

    @assert size(σ0,1)==n_sys && size(σ0,2)==n_sys
    @assert size(R_hist,1)==n_itr && size(R_hist,2)==n_sys && size(R_hist,3)==n_sys
    @assert size(Rpd_hist,1)==n_itr && size(Rpd_hist,2)==n_sys && size(Rpd_hist,3)==n_sys

    σ = Matrix{ComplexF64}(σ0)
    σ_hist = zeros(ComplexF64, n_sys, n_sys, n_itr)
    σ_hist[:,:,1] .= σ

    # RHS with linear interpolation between ti and ti+1:
    # at time index ti with stage θ in {0,0.5,1} uses (ti, ti+1)
    function rhs!(dσ, σ, ti::Int, θ::Float64)
        fill!(dσ, 0.0 + 0.0im)

        # interpolate R and Rpd at (ti,ti+1)
        # also need out[a] = sum_f R[f,a]
        out = zeros(Float64, n_sys)
        @inbounds for a in 1:n_sys
            s = 0.0
            for f in 1:n_sys
                if f == a; continue; end
                Rfa = _interp_val(R_hist[ti,f,a], R_hist[ti+1,f,a], θ)
                s += Rfa
            end
            out[a] = s
        end

        @inbounds for a in 1:n_sys, b in 1:n_sys
            if a == b
                acc = 0.0 + 0.0im
                σaa = σ[a,a]
                for f in 1:n_sys
                    if f == a; continue; end
                    Raf = _interp_val(R_hist[ti,a,f], R_hist[ti+1,a,f], θ)
                    Rfa = _interp_val(R_hist[ti,f,a], R_hist[ti+1,f,a], θ)
                    acc += Raf*σ[f,f] - Rfa*σaa
                end
                dσ[a,a] = acc
            else
                Rpd_ab = _interp_val(Rpd_hist[ti,a,b], Rpd_hist[ti+1,a,b], θ)
                damping = Rpd_ab + 0.5*(out[a] + out[b])
                ωab = ϵ[a] - ϵ[b]
                dσ[a,b] = (-1.0im*ωab - damping) * σ[a,b]
            end
        end

        return dσ
    end

    if method == :euler
        dσ = zeros(ComplexF64, n_sys, n_sys)
        @inbounds for ti in 1:(n_itr-1)
            rhs!(dσ, σ, ti, 0.0)
            @. σ = σ + dt*dσ
            σ_hist[:,:,ti+1] .= σ
        end

    elseif method == :rk4
        k1 = zeros(ComplexF64, n_sys, n_sys)
        k2 = zeros(ComplexF64, n_sys, n_sys)
        k3 = zeros(ComplexF64, n_sys, n_sys)
        k4 = zeros(ComplexF64, n_sys, n_sys)
        tmp = zeros(ComplexF64, n_sys, n_sys)

        @inbounds for ti in 1:(n_itr-1)
            rhs!(k1, σ, ti, 0.0)

            @. tmp = σ + (0.5*dt)*k1
            rhs!(k2, tmp, ti, 0.5)

            @. tmp = σ + (0.5*dt)*k2
            rhs!(k3, tmp, ti, 0.5)

            @. tmp = σ + dt*k3
            rhs!(k4, tmp, ti, 1.0)

            @. σ = σ + (dt/6.0)*(k1 + 2k2 + 2k3 + k4)
            σ_hist[:,:,ti+1] .= σ
        end
    else
        error("Unknown method=$method. Use :euler or :rk4")
    end

    return σ_hist
end