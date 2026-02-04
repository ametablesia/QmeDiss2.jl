
# include("../Physics/Physics.jl")
# using Base.Threads
# using LinearAlgebra
# import Base: getindex


# mutable struct CmrtContext
#     # input
#     system              ::System
#     environment         ::Environment
#     simulation_details  ::SimulationDetails

#     # computing context
#     γ_exci              ::Array{ComplexF64, 3}    # 원래는 oscillator 안에 들어가는 coupling strenght 의 exciton verison인데, 음.
#     ϵ_exci              ::Vector{Float64}               # energy in exciton basis
#     ϵ_exci_0            ::Vector{Float64}               # energy - reorganization energy
#     U_sys               ::Matrix{ComplexF64}            # engenvector matrix
#     g                   ::Patternized_g{ComplexF64}
#     g′                  ::Patternized_g′{ComplexF64}
#     g″                  ::Patternized_g″{ComplexF64}
#     Λ                   ::Patternized_Λ{Float64}

#     # output
#     transition_rate     ::Array{Float64, 2}


#     function CmrtContext(system::System, environment::Environment, simulation_details::SimulationDetails)
        
#         n_itr = simulation_details.num_of_iteration
#         n_sys = system.n_sys
#         n_osc = environment.num_of_effective_oscillators

#         # H_sys0  = diag(system.H_sys) # 대각성분만 추출.
#         γ_exci  = zeros(ComplexF64, (n_osc, n_sys, n_sys))
#         ϵ_exci  = zeros(Float64, n_sys)
#         ϵ_exci_0= zeros(Float64, n_sys)
#         U_sys   = zeros(ComplexF64, (n_sys, n_sys))
#         g       = Patternized_g{ComplexF64}(n_sys, n_itr)
#         g′      = Patternized_g′{ComplexF64}(n_sys, n_itr)
#         g″      = Patternized_g″{ComplexF64}(n_sys, n_itr)
#         Λ       = Patternized_Λ{Float64}(n_sys)

#         transition_rate     = zeros(Float64, (n_sys, n_sys))

#         new(system, environment, simulation_details, γ_exci, ϵ_exci, ϵ_exci_0, U_sys, g, g′, g″, Λ, transition_rate)
#     end
# end

# function create__cmrt_context(
#     system              ::System,
#     environment         ::Environment,
#     simulation_details  ::SimulationDetails
# )
#     return CmrtContext(system, environment, simulation_details)
# end

# function calc__Λ!(context::MrtContext)

#     n_sys       = context.system.n_sys
#     n_osc       = context.environment.num_of_effective_oscillators
#     oscs        = context.environment.effective_oscillators

#     H_sys       = context.system.H_sys
#     U_sys       = context.U_sys
#     ϵ_exci      = context.ϵ_exci
#     ϵ_exci_0    = context.ϵ_exci_0
#     γ_exci      = context.γ_exci    

#     Λ           = context.Λ

#     eigen_result = eigen!(Hermitian(H_sys))
#     ϵ_exci      .= eigen_result.values      # copy
#     U_sys       .= eigen_result.vectors     # copy


#     # Algorithm start
#     fill!(γ_exci, 0)
#     # fill!(Λ, 0)

#     @inbounds for osc_idx = 1:n_osc
#         ω       = oscs[osc_idx].freq
#         γ       = oscs[osc_idx].site_bath_coupling_strength
       
#         # U_sys is unitary
#         γ_exci[osc_idx,:,:] .= U_sys' * γ * U_sys
        
#         # 나중에 바꾸고 싶으면 반쪽해서 복사 하든 / 지금은 그냥...
#         for β in 1:n_sys, α in 1:n_sys
#             if α == β
#                 Λ[α,α,α,α] += ω * γ_exci[osc_idx,α,α] * γ_exci[osc_idx,α,α]
#                 continue
#             else
#                 Λ[α,α,β,β] += ω * γ_exci[osc_idx,α,α] * γ_exci[osc_idx,β,β]
#                 Λ[α,β,α,α] += ω * γ_exci[osc_idx,α,β] * γ_exci[osc_idx,α,α]
#                 Λ[α,β,β,β] += ω * γ_exci[osc_idx,α,β] * γ_exci[osc_idx,β,β]
#             end
#         end
#     end

#     # ϵ_exci_0 생성
#     for α in 1:n_sys
#         ϵ_exci_0[α] = ϵ_exci[α] - Λ[α,α,α,α]
#     end
# end

# function calc__g!(context::CmrtContext)


# end