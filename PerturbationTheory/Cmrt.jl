

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
    function Patternized_g{T}(n_sys::Int, n_itr::Int) where {T}
        new(zeros(T, n_itr, n_sys, n_sys))
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
    a==b && c==d            ? g′.ααββ[t,a,b] :
    b==c && c==d            ? g′.αβββ[t,a,b] :
    a==c && c==d            ? g′.αβαα[t,a,b] :
    error("unsupported g′ pattern")

@inline Base.setindex!(g′::Patternized_g′{T}, v, t::Int,a::Int,b::Int,c::Int,d::Int) where {T} =
    a==b && c==d            ? (g′.ααββ[t,a,b]=v) :
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

function create__mrt_context(
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
            Γ_αβ += γ_exci[osc_idx,α,β]
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

            @inbounds for time_idx = 1:n_itr
                t   = (time_idx - 1) * Δt
                ωt  = ω * t
                sin_ωt, cos_ωt = sincos(ωt)
                
                g[time_idx,α,α,β,β]    += (γʲ_ααββ      * ((coth * (1.0 - cos_ωt)) + 1.0im*(sin_ωt - ωt)))

                # 같을 때만 따로 처리... 
                if α == β
                    g′[time_idx,α,α,α,α]    += ω_γʲ_αβββ    * ((coth * sin_ωt) + 1.0im*(cos_ωt - 1.0))
                else
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

            @inbounds for time_idx = 1:n_itr
                t   = (time_idx - 1) * Δt
                ωt  = ω * t

                sin_ωt, cos_ωt = sin(ωt), cos(ωt)
                
                g_local[time_idx,α,α,β,β]    += (hr_ααββ      * ((coth * (1.0 - cos_ωt)) + 1.0im*(sin_ωt - ωt)))

                if α == β
                    g′_local[time_idx,α,α,α,α]    += hr_αβββ_ω    * ((coth * sin_ωt) + 1.0im*(cos_ωt - 1.0))
                else
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