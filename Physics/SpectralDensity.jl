
using Printf

abstract type SpectralDensity end

mutable struct DrudeLorentzSpectralDensity <: SpectralDensity
    λ   ::Float64   # reorganization energy
    γ   ::Float64   # bath correation decay rate (γ = 1/τ_c)
end

mutable struct BrownianSpectralDensity <: SpectralDensity
    λ   ::Float64   # reorganization energy
    γ   ::Float64   # damping constant
    ω_0 ::Float64   # peak freq
end

mutable struct LogNormalSpectralDensity <: SpectralDensity
    η   ::Float64   # coupling strength scale
    μ   ::Float64   # peak freq in log scale, ln(ω_0)
    σ   ::Float16   # variance
end

mutable struct OhmicTypeSpectralDensity <: SpectralDensity
    η   ::Float64   # coupling strength scale
    s   ::Float64   # power
    ω_c ::Float64   # cutoff freq
end

## WRAN pi가 나눠져 있어요.
@inline function J(info::DrudeLorentzSpectralDensity, ω)
    λ = info.λ
    γ = info.γ
    
    return (2.0*λ / π) * (ω*γ) / (ω^2 + γ^2)
end

@inline function J(info::BrownianSpectralDensity, ω)
    λ   = info.λ  
    γ   = info.γ        
    ω_0 = info.ω_0

    return (2*λ*γ*ω_0^2) / ((ω_0^2 - ω^2)^2 + γ^2*ω^2)
end

@inline function J(info::LogNormalSpectralDensity, ω)
    η   = info.η
    μ   = info.μ
    σ   = info.σ

    return η / (ω*σ*sqrt(2*π)) * exp(- (ln(ω) - μ)^2 / 2*σ^2)
end

@inline function J(info::OhmicTypeSpectralDensity, ω)
    η   = info.η
    s   = info.s
    ω_c = info.ω_c
    
    return η * ω^s * exp(-ω/ω_c)
end

@inline function calc__spectral_density(info::SpectralDensity, freq)
    return J(info, freq)
end

@inline function C(info::SpectralDensity, time)
    # Need to implement Matsubara expansion.
end

@inline function calc__correlation_function(info::SpectralDensity, time)
    return C(info, time)
end

mutable struct EffectiveOscillator
    spectral_density_id             ::Int64
    freq                            ::Float64
    coth                            ::Float64
    spread                          ::Float64
    temperature                     ::Float64
    site_bath_coupling_strength     ::Matrix{ComplexF64}

    function EffectiveOscillator(;
        spectral_density_id,    
        freq,                           
        coth,                           
        spread,                         
        temperature,                    
        site_bath_coupling_strength    
    )
        new(spectral_density_id, freq, coth, spread, temperature, site_bath_coupling_strength)
    end
end

mutable struct SpectralDensityDecomposeInfo
    num_of_sampled_freq             ::Int64     # is same with num of effective oscillator
    freq_max                        ::Float64
    temperature                     ::Float64
    site_bath_coupling_strength     ::Array{Float64, 2}
end

# function decompose__spectral_density_into_effective_oscillators!(
#     oscillators         ::Vector{EffectiveOscillator},
#     spectral_density    ::SpectralDensity,
#     decompose_info      ::SpectralDensityDecomposeInfo
# )

#     n_sampled   ::Int64     = decompose_info.num_of_sampled_freq
#     ω_max       ::Float64   = decompose_info.freq_max
#     temp        ::Float64   = decompose_info.temperature
#     β_th        ::Float64   = 1.0 / temp
#     coup        ::Float64   = decompose_info.site_bath_coupling_strength

#     resize!(oscillators, n_sampled)

#     Δx  ::Float = 1.0 / n_sampled
#     @inbounds for j in 1:n_sampled

#         ### Quadratic mapping (low ω -> dense sampling)
#         x   = j * Δx
#         ω_j = ω_max * x^2

#         ### dω(x) = ω_max x^2
#         Δω = 2.0 * ω_max * x * Δx

#         ### J(ω_j)
#         J_ω_j = J(spectral_density, ω_j)

#         ### effective coupling
#         g_j = sqrt(J_ω_j * Δω)

#         ### coth(1/2 * β \omega)
#         coth_j = coth(0.5 * β_th * ω_j)

#         ### Effective oscillator
#         oscillators[j] = Oscillators(
#             freq   = ω_j,
#             g      = g_j,
#             coth   = coth_j
#         )
#     end

#     return nothing
# end

const HBAR_MKS = 1.0545718e-34
const kb_MKS = 1.38064852e-23
const C_CGS = 2.997992458e+10
const TUNIT_TO_S = HBAR_MKS / kb_MKS
const WAVENUMBER_TO_ENERGY =  (C_CGS * 2*π * TUNIT_TO_S)

function decompose__spectral_density_into_effective_oscillators!(
    oscillators         ::Vector{EffectiveOscillator},
    spectral_density    ::DrudeLorentzSpectralDensity,
    decompose_info      ::SpectralDensityDecomposeInfo
)
    n_sampled   ::Int64             = decompose_info.num_of_sampled_freq
    ω_max       ::Float64           = decompose_info.freq_max
    temp        ::Float64           = decompose_info.temperature
    coup        ::Matrix{Float64}   = decompose_info.site_bath_coupling_strength
    λ_origin    ::Float64           = spectral_density.λ
    γ           ::Float64           = spectral_density.γ

    β_th        ::Float64           = 1.0 / temp

    # check for reorganization energy summation
    λ_sum       ::Float64           = 0.0

    resize!(oscillators, n_sampled)
    for j in 1:n_sampled
        oscillators[j] = EffectiveOscillator(
            spectral_density_id             = 0,
            freq                            = 0.0,
            coth                            = 0.0,
            spread                          = 0.0,
            temperature                     = temp,
            site_bath_coupling_strength     = zeros(size(coup))
        )
    end

    Δx  ::Float64 = 1.0 / n_sampled
    @inbounds for j in 1:n_sampled

        ### Quadratic mapping (low ω -> dense sampling)
        x_j         = j * Δx
        ω_j         = ω_max * x_j^2
        Δω          = 2.0 * ω_max * x_j * Δx       # dω(x) = ω_max x^2

        J_ω_j       = J(spectral_density, ω_j)   # J(ω_j)
        weight      = sqrt(J_ω_j * Δω) / ω_j     # √ gaussian quadrature weight * J(ω) Δω / ω^2 

        coth_j      = coth(0.5 * β_th * ω_j)     # coth(1/2 * β ω)
        spread_j    = Δω
        #spread_j    = 2.0 * ω_j * (x / Δx)       # spread
        # coup_j      = weight * coup           # 행렬복사 방지 위해서...
        
        ### Effective oscillator
        # oscillators[j] = EffectiveOscillator(
        #     spectral_density_id             = 0,
        #     freq                            = ω_j,
        #     coth                            = coth_j,
        #     spread                          = spread_j,
        #     temperature                     = temp,
        #     site_bath_coupling_strength     = coup_j
        # )
        oscillators[j].spectral_density_id             = 0
        oscillators[j].freq                            = ω_j
        oscillators[j].coth                            = coth_j
        oscillators[j].spread                          = spread_j
        oscillators[j].temperature                     = temp
        @. oscillators[j].site_bath_coupling_strength     = weight * coup

        λ_sum += ω_j * weight * weight

        # @printf(stderr, "J_ω_j          %15.6e a.u.\n", J_ω_j)
        # @printf(stderr, "ωj          %15.6e a.u.\n", ω_j)
        # @printf(stderr, "gamma_raw   %15.6e a.u.\n", weight)
    end

    @printf(stderr, "--------------------------------------------------\n")
    @printf(stderr, "Spectral density file - Drude-Lorentz\n")
    @printf(stderr, "Reorg E pristine  %15.6e a.u.   %15.6e waveno.\n",             λ_origin, λ_origin / WAVENUMBER_TO_ENERGY)
    @printf(stderr, "Reorg E osc sum   %15.6e a.u.   %15.6e waveno.  (%6.2f %%)\n", λ_sum, λ_sum / WAVENUMBER_TO_ENERGY, λ_sum / λ_origin * 100.0)
    @printf(stderr, "w_cutoff          %15.6e a.u.   %15.6e waveno.\n",γ, γ / WAVENUMBER_TO_ENERGY)
    @printf(stderr, "ω_max             %15.6e a.u.   %15.6e waveno.\n", ω_max, ω_max / WAVENUMBER_TO_ENERGY)
    
    @printf(stderr, "Coupling\n")
    # n_row, n_col = size(coup)
    @inbounds for i in axes(coup, 1)
        for j in axes(coup, 2)
            @printf(stderr, "%8.3f", coup[i, j])
        end
        @printf(stderr, "\n")
    end

    return nothing
end

# function alias for short-name lovers.
const decompose__spd_into_osc! = decompose__spectral_density_into_effective_oscillators!



## TESTING CODE
# oscillators = Vector{EffectiveOscillator}()
# spectral_density = DrudeLorentzSpectralDensity(0.05, 1.0)
# decompose_info = SpectralDensityDecomposeInfo(2000, 30.0, 1.0, [1.0 0.0; 0.0 -1.0])

# decompose__spd_into_osc!(oscillators, spectral_density, decompose_info)

# using BenchmarkTools
# @btime decompose__spd_into_osc!(oscillators, spectral_density, decompose_info)
