
mutable struct Dissipation
    effective_oscillator_id ::Int64
    
    i_dissipation   ::Array{ComplexF64, 2}
    j_dissipation   ::Array{ComplexF64, 2}
    k_dissipation   ::Array{ComplexF64, 2}

    function Dissipation(osc_id::Int64, n_sys::Int64)
        return new(osc_id, zeros(ComplexF64, (n_sys, n_sys)), zeros(ComplexF64, (n_sys, n_sys)), zeros(ComplexF64, (n_sys, n_sys)))
    end
end
