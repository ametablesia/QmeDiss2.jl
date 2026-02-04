
mutable struct Patternized_Λ{T}
    ααββ::Matrix{T}   # (α, β)  — includes αααα, ααββ, ββββ
    αβαα::Matrix{T}   # (α, β)  — only for α ≠ β
    
    function Patternized_Λ{T}(n_sys::Int) where {T}
        new(zeros(T, n_sys, n_sys), zeros(T, n_sys, n_sys))
    end
end

mutable struct Patternized_g{T}
    ααββ::Array{T,3}   # (t, α, β)
    function Patternized_g{T}(n_sys::Int, n_itr::Int) where {T}
        new(zeros(T, n_itr, n_sys, n_sys))
    end
end

mutable struct Patternized_g′{T}
    αβββ::Array{T,3}   # (t, α, β)
    αβαα::Array{T,3}   # (t, α, β)
    function Patternized_g′{T}(n_sys::Int, n_itr::Int) where {T}
        new(zeros(T, n_itr, n_sys, n_sys), zeros(T, n_itr, n_sys, n_sys))
    end
end

mutable struct Patternized_g″{T}
    αβαβ::Array{T,3}   # (t, α, β)
    αββα::Array{T,3}   # (t, α, β)
    function Patternized_g″{T}(n_sys::Int, n_itr::Int) where {T}
        new(zeros(T, n_itr, n_sys, n_sys), zeros(T, n_itr, n_sys, n_sys))
    end
end

# Λ_αααα, Λ_ααββ, Λ_ββββ, Λ_αβαα, Λ_βααα         
@inline Base.getindex(Λ::Patternized_Λ{T}, ::Val{:αααα}, α::Int, α2::Int, α3::Int, α4::Int) where {T}               = Λ.ααββ[α, α]
@inline Base.getindex(Λ::Patternized_Λ{T}, ::Val{:ααββ}, α::Int, α2::Int, β::Int, β2::Int) where {T}                = Λ.ααββ[α, β]
@inline Base.getindex(Λ::Patternized_Λ{T}, ::Val{:ββββ}, β::Int, β2::Int, β3::Int, β4::Int) where {T}               = Λ.ααββ[β, β]
@inline Base.getindex(Λ::Patternized_Λ{T}, ::Val{:αβαα}, α::Int, β::Int, α2::Int, α3::Int) where {T}                = (α == β ? Λ.ααββ[α, α] : Λ.αβαα[α, β])
@inline Base.getindex(Λ::Patternized_Λ{T}, ::Val{:βααα}, β::Int, α::Int, α2::Int, α3::Int) where {T}                = (α == β ? Λ.ααββ[α, α] : Λ.αβαα[β, α])

@inline Base.setindex!(Λ::Patternized_Λ{T}, v, ::Val{:αααα}, α::Int, α2::Int, α3::Int, α4::Int) where {T}           = (Λ.ααββ[α, α] = v)
@inline Base.setindex!(Λ::Patternized_Λ{T}, v, ::Val{:ααββ}, α::Int, α2::Int, β::Int, β2::Int) where {T}            = (Λ.ααββ[α, β] = v)
@inline Base.setindex!(Λ::Patternized_Λ{T}, v, ::Val{:ββββ}, β::Int, β2::Int, β3::Int, β4::Int) where {T}           = (Λ.ααββ[β, β] = v)
@inline Base.setindex!(Λ::Patternized_Λ{T}, v, ::Val{:αβαα}, α::Int, β::Int, α2::Int, α3::Int) where {T}            = (α == β ? (Λ.ααββ[α, α] = v) : (Λ.αβαα[α, β] = v))
@inline Base.setindex!(Λ::Patternized_Λ{T}, v, ::Val{:βααα}, β::Int, α::Int, α2::Int, α3::Int) where {T}            = (α == β ? (Λ.ααββ[α, α] = v) : (Λ.αβαα[β, α] = v))

# g_αααα, g_ααββ, g_ββββ
@inline Base.getindex(g::Patternized_g{T}, ::Val{:αααα}, t::Int, α::Int, α2::Int, α3::Int, α4::Int) where {T}       = g.ααββ[t, α, α]
@inline Base.getindex(g::Patternized_g{T}, ::Val{:ααββ}, t::Int, α::Int, α2::Int, β::Int, β2::Int) where {T}        = g.ααββ[t, α, β]
@inline Base.getindex(g::Patternized_g{T}, ::Val{:ββββ}, t::Int, β::Int, β2::Int, β3::Int, β4::Int) where {T}       = g.ααββ[t, β, β]

@inline Base.setindex!(g::Patternized_g{T}, v, ::Val{:αααα}, t::Int, α::Int, α2::Int, α3::Int, α4::Int) where {T}   = (g.ααββ[t, α, α] = v)
@inline Base.setindex!(g::Patternized_g{T}, v, ::Val{:ααββ}, t::Int, α::Int, α2::Int, β::Int, β2::Int) where {T}    = (g.ααββ[t, α, β] = v)
@inline Base.setindex!(g::Patternized_g{T}, v, ::Val{:ββββ}, t::Int, β::Int, β2::Int, β3::Int, β4::Int) where {T}   = (g.ααββ[t, β, β] = v)

# g′_αβββ, g′_αβαα, g′_βαββ, g′_βααα   
@inline Base.getindex(g′::Patternized_g′{T}, ::Val{:αβββ}, t::Int, α::Int, β::Int, β2::Int) where {T}               = (α == β ? g′.αβαα[t, α, α] : g′.αβββ[t, α, β])
@inline Base.getindex(g′::Patternized_g′{T}, ::Val{:αβαα}, t::Int, α::Int, β::Int, α2::Int) where {T}               = g′.αβαα[t, α, β]
@inline Base.getindex(g′::Patternized_g′{T}, ::Val{:βαββ}, t::Int, β::Int, α::Int, β2::Int) where {T}               = (α == β ? g′.αβαα[t, α, α] : g′.αβββ[t, β, α])
@inline Base.getindex(g′::Patternized_g′{T}, ::Val{:βααα}, t::Int, β::Int, α::Int, α2::Int) where {T}               = g′.αβαα[t, α, β]

@inline Base.setindex!(g′::Patternized_g′{T}, v, ::Val{:αβββ}, t::Int, α::Int, β::Int, β2::Int) where {T}           = (α == β ? (g′.αβαα[t, α, α] = v) : (g′.αβββ[t, α, β] = v))
@inline Base.setindex!(g′::Patternized_g′{T}, v, ::Val{:αβαα}, t::Int, α::Int, β::Int, α2::Int) where {T}           = (g′.αβαα[t, α, β] = v)
@inline Base.setindex!(g′::Patternized_g′{T}, v, ::Val{:βαββ}, t::Int, β::Int, α::Int, β2::Int) where {T}           = (α == β ? (g′.αβαα[t, α, α] = v) : (g′.αβββ[t, β, α] = v))
@inline Base.setindex!(g′::Patternized_g′{T}, v, ::Val{:βααα}, t::Int, β::Int, α::Int, α2::Int) where {T}           = (g′.αβαα[t, α, β] = v)

# g″_αββα 
@inline Base.getindex(g″::Patternized_g″{T}, ::Val{:αββα}, t::Int, α::Int, β::Int, β2::Int, α2::Int) where {T}      = g″.αββα[t, α, β]

function _infer_pattern(vars::Vector{Symbol})
    dict = Dict{Symbol,Symbol}()
    next = 'α'
    pat = Symbol[]
    for v in vars
        if !haskey(dict, v)
            dict[v] = Symbol(Char(next))
            next += 1
        end
        push!(pat, dict[v])
    end
    return Symbol(join(pat))
end

macro pattern(expr)
    expr.head == :ref || error("usage: @pattern obj[i,j,k,l] or @pattern obj[t,i,j,k,l]")
    obj = expr.args[1]
    inds = expr.args[2:end]
    n = length(inds)

    (n == 4 || n == 5) || error("pattern requires exactly 4 indices (or 5 with time index)")

    pat_inds_any = n == 4 ? inds : inds[2:end]

    all(v -> v isa Symbol, pat_inds_any) || error("pattern indices must be variables (numeric literals or expressions are not allowed)")

    pat_inds = Symbol[v for v in pat_inds_any]
    pat = _infer_pattern(pat_inds)

    esc(Expr(:ref, obj, :(Val($(QuoteNode(pat)))), inds...))
end


α = 5
β = 3
g = Patternized_g{ComplexF64}(10, 10)
# println(@macroexpand @pattern Λ[α, α, β, β])
println(@pattern g[10, α, α, α, α])