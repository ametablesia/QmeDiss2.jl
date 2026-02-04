
include("SpectralDensity.jl")

mutable struct Environment
    # internal
    num_of_effective_oscillators    ::Int64
    spectral_densities              ::Vector{SpectralDensity}
    effective_oscillators           ::Vector{EffectiveOscillator}

    function Environment()
        new(0, Vector{SpectralDensity}(), Vector{EffectiveOscillator}())
    end
end

function add__spectral_density!(
    environment         ::Environment,
    spectral_density    ::DrudeLorentzSpectralDensity,
    decompose_info      ::SpectralDensityDecomposeInfo
)
    # decompose to oscs
    new_oscs = Vector{EffectiveOscillator}()
    decompose__spd_into_osc!(new_oscs, spectral_density, decompose_info)
    
    # assign spectral density id
    spectral_density_id = length(environment.spectral_densities)
    @inbounds for i in eachindex(new_oscs)
        new_oscs[i].spectral_density_id = spectral_density_id
    end

    # update environment member vars
    push!(environment.spectral_densities, spectral_density)
    append!(environment.effective_oscillators, new_oscs)
    environment.num_of_effective_oscillators = length(environment.effective_oscillators)
end

