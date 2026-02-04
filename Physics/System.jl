

mutable struct System
    n_sys   ::Int64
    H_sys   ::Matrix{ComplexF64}
    #
    # system_hamiltonian_in_
    # system_hamiltonian_in_exciton_basis

    # function System(; size_of_system::Int64, system_hamiltonian::Matrix{ComplexF64})
    #     new(size_of_system, system_hamiltonian)
    # end 
end

# constructors
System(size_of_system::Int64) = System(size_of_system, Matrix{Complex(undef, (size_of_system, size_of_system))})
System(; size_of_system::Int64) = System(size_of_system, Matrix{Complex(undef, (size_of_system, size_of_system))})
System(; size_of_system::Int64, system_hamiltonian::Matrix{ComplexF64}) = System(size_of_system, system_hamiltonian)

function set__system_hamiltonian!(
    system  ::System,
    H_sys   ::Matrix{ComplexF64}
)
    if ishermitian(H_sys) == false
        println("Hsys is not Hermitian!")
    end

    system.H_sys = H_sys
end