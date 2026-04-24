
using ..Physics
using Base.Threads
using LinearAlgebra
import Base: getindex

abstract type Patternized{T} end

"""
    @patternized TypeName (ctor_args...) (access_args...) begin
        rule(field, fieldtype, init_expr, stored_indices_tuple, condition_expr)
        ...
    end

예시:
@patternized Patternized_Λ (n_sys::Int) (α, β, γ, δ) begin
    rule(ααββ, Matrix{T}, zeros(T, n_sys, n_sys), (α, α), α == β && β == γ && γ == δ)
    ...
end
"""

macro patternized(type_name, constructor_args_expr, access_args_expr, body)

    #
    # < 매크로 입력 전처리 >
    #
    # 생성자 인자가 tuple 형식이면, args를 꺼내서 list 형식으로 만듦.
    constructor_args = constructor_args_expr isa Expr && constructor_args_expr.head == :tuple ? constructor_args_expr.args : [constructor_args_expr]
    # 접근자 인자가 tuple 형식이면, args를 꺼내서 list 형식으로 만듦.
    access_args = access_args_expr isa Expr && access_args_expr.head == :tuple ? access_args_expr.args : [access_args_expr]
    # body에는 rule 들'만' 선언되고, 각 rule을 뽑아 낼 수 있도록 만듦.
    rules_exprs = body isa Expr && body.head == :block ? [ex for ex in body.args if !(ex isa LineNumberNode)] : [body]

    # 
    # < 필드, 멤버 변수 생성>
    # 
    # 필드 변수명, 타입, 초기값 설정.
    field_symbols   = Symbol[]
    field_types     = Any[]
    field_inits     = Any[] 

    # 분기 관련 코드에서, 어느 필드로 보낼지, 어떤 인덱스로 보낼지, 언제 condition을 적용할지.
    rule_fields         = Symbol[]
    rule_index_tuples   = Any[]
    rule_conditions     = Any[]

    for ex in rules_exprs
        if !(ex isa Expr && ex.head == :call && ex.args[1] == :rule && length(ex.args) == 6)
            error("Each line inside @patternized must be: rule(field, fieldtype, init, indices, condition)")
        end

        # 일단 잠깐 가져와서,
        field_symbol, field_type, field_init, index_tuple, condition = ex.args[2], ex.args[3], ex.args[4], ex.args[5], ex.args[6]

        field_symbol isa Symbol || error("field name must be a Symbol")

        # 추가하려는 필드가 중복이면 안되므로, 검사를 수행
        finding_position = findfirst(==(field_symbol), field_symbols)
        if finding_position === nothing
            # 중복되지 않으면, 추가해도 됨.
            push!(field_symbols, field_symbol)
            push!(field_types, field_type)
            push!(field_inits, field_init)
        else
            # 중복되면, 에러.
            if field_types[finding_position] != field_type
                error("field $field_symbol is already added to patternized container with different field type.")
            end
            if field_inits[finding_position] != field_init
                error("field $field_symbol is already added to patternized container with different initializer.")
            end
        end

        # 추가하려는 분기는 따로 검사 안할래 (귀찮음.)
        push!(rule_fields, field_symbol)
        push!(rule_index_tuples, index_tuple)
        push!(rule_conditions, condition)
    end

    # gensym 해서, 내부용 함수 인자 이름을 처리할거임.
    x = gensym(:x)          # getindex, setindex! 의 객체 본체
    dst = gensym(:dst)      # inplace_add! 인자
    src = gensym(:src)      # inplace_add! 인자
    val = gensym(:val)      # setindex!에서 대입할 값.

    # 필드 자동 생성
    struct_fields = [
        :($(field_symbols[i])::$(field_types[i])) for i in eachindex(field_symbols)
    ]

    # 생성자 자동 생성
    constructor_body = Expr(:call, Expr(:curly, type_name, :T), field_inits...)
    constructor_def = quote
        function $(Expr(:curly, type_name, :T))($(constructor_args...)) where {T}
            return $constructor_body
        end
    end

     # getindex branches
    get_branches = Any[]
    for i in eachindex(rule_fields)
        idx_tuple = rule_index_tuples[i]
        idx_args = idx_tuple isa Expr && idx_tuple.head == :tuple ? idx_tuple.args : [idx_tuple]
        push!(get_branches, :( $(rule_conditions[i]) => @inbounds($(x).$(rule_fields[i])[$(idx_args...)] ) ))
    end

    getindex_body = nothing
    for i in reverse(eachindex(get_branches))
        cond = get_branches[i].args[2]
        rhs  = get_branches[i].args[3]
        if getindex_body === nothing
            getindex_body = quote
                if $cond
                    return $rhs
                else
                    throw(ArgumentError("unsupported pattern for $(string($(QuoteNode(type_name))))"))
                end
            end
        else
            getindex_body = quote
                if $cond
                    return $rhs
                else
                    $getindex_body
                end
            end
        end
    end

    getindex_def = quote
        @inline function Base.getindex($(x)::$(Expr(:curly, type_name, :T)),
                                       $(access_args...)) where {T}
            $getindex_body
        end
    end

    # setindex! branches
    setindex_body = nothing
    for i in reverse(eachindex(rule_fields))
        idx_tuple = rule_index_tuples[i]
        idx_args = idx_tuple isa Expr && idx_tuple.head == :tuple ? idx_tuple.args : [idx_tuple]
        cond = rule_conditions[i]
        rhs = :(@inbounds($(x).$(rule_fields[i])[$(idx_args...)] = $(val)))
        if setindex_body === nothing
            setindex_body = quote
                if $cond
                    $rhs
                    return $(val)
                else
                    throw(ArgumentError("unsupported pattern for $(string($(QuoteNode(type_name))))"))
                end
            end
        else
            setindex_body = quote
                if $cond
                    $rhs
                    return $(val)
                else
                    $setindex_body
                end
            end
        end
    end

    setindex_def = quote
        @inline function Base.setindex!($(x)::$(Expr(:curly, type_name, :T)),
                                        $(val),
                                        $(access_args...)) where {T}
            $setindex_body
        end
    end

    # inplace_add!
    add_lines = [:( $(dst).$(fs) .+= $(src).$(fs) ) for fs in field_symbols]
    add_def = quote
        @inline function inplace_add!($(dst)::$(Expr(:curly, type_name, :T)),
                                      $(src)::$(Expr(:curly, type_name, :T))) where {T}
            $(add_lines...)
            return $(dst)
        end
    end

    q = quote
        mutable struct $(Expr(:curly, type_name, :T)) <: Patternized{T}
            $(struct_fields...)
        end

        $constructor_def
        $getindex_def
        $setindex_def
        $add_def
    end

    return esc(q)
end


# mutable struct Patternized_Λ{T}
#     ααββ::Matrix{T}   # (α, β)  — includes αααα, ααββ, ββββ
#     αβαα::Matrix{T}   # (α, β)  — only for α ≠ β
#     αβββ::Matrix{T}

#     function Patternized_Λ{T}(n_sys::Int) where {T}
#         new(zeros(T, n_sys, n_sys), zeros(T, n_sys, n_sys), zeros(T, n_sys, n_sys))
#     end
# end

# mutable struct Patternized_g{T}
#     ααββ::Array{T,3}   # (t, α, β)
#     αβββ::Array{T,3}   # (t, α, β)
#     function Patternized_g{T}(n_sys::Int, n_itr::Int) where {T}
#         new(zeros(T, n_itr, n_sys, n_sys), zeros(T, n_itr, n_sys, n_sys))
#     end
# end

# mutable struct Patternized_g′{T}
#     ααββ::Array{T,3}   # (t, α, β)
#     αβββ::Array{T,3}   # (t, α, β)
#     αβαα::Array{T,3}   # (t, α, β)
#     function Patternized_g′{T}(n_sys::Int, n_itr::Int) where {T}
#         new(zeros(T, n_itr, n_sys, n_sys), zeros(T, n_itr, n_sys, n_sys), zeros(T, n_itr, n_sys, n_sys))
#     end
# end

# mutable struct Patternized_g″{T}
#     αβαβ::Array{T,3}   # (t, α, β)
#     αββα::Array{T,3}   # (t, α, β)
#     function Patternized_g″{T}(n_sys::Int, n_itr::Int) where {T}
#         new(zeros(T, n_itr, n_sys, n_sys), zeros(T, n_itr, n_sys, n_sys))
#     end
# end

# @inline Base.getindex(Λ::Patternized_Λ{T}, a::Int,b::Int,c::Int,d::Int) where {T} =
#     a==b && b==c && c==d ? Λ.ααββ[a,a] : # aaaa bbbb
#     a==b && c==d         ? Λ.ααββ[a,c] : # aabb
#     a==c && c==d         ? Λ.αβαα[a,b] : # abaa
#     b==c && c==d         ? Λ.αβββ[a,b] : # abbb
#     error("unsupported Λ pattern")

# @inline Base.setindex!(Λ::Patternized_Λ{T}, v, a::Int,b::Int,c::Int,d::Int) where {T} =
#     a==b && b==c && c==d ? (Λ.ααββ[a,a]=v) :
#     a==b && c==d         ? (Λ.ααββ[a,c]=v) :
#     a==c && c==d         ? (Λ.αβαα[a,b]=v) :
#     b==c && c==d         ? (Λ.αβββ[a,b]=v) :
#     error("unsupported Λ pattern")

# @inline Base.getindex(g::Patternized_g{T}, t::Int,a::Int,b::Int,c::Int,d::Int) where {T} =
#     a==b && c==d ? g.ααββ[t,a,c] :
#     b==c && c==d ? g.αβββ[t,a,b] :
#     error("unsupported g pattern")

# @inline Base.setindex!(g::Patternized_g{T}, v, t::Int,a::Int,b::Int,c::Int,d::Int) where {T} =
#     a==b && c==d ? (g.ααββ[t,a,c]=v) :
#     b==c && c==d ? (g.αβββ[t,a,b]=v) :
#     error("unsupported g pattern")

# @inline Base.getindex(g′::Patternized_g′{T}, t::Int,a::Int,b::Int,c::Int,d::Int) where {T} =
#     a==b && c==d            ? g′.ααββ[t,a,c] :
#     b==c && c==d            ? g′.αβββ[t,a,b] :
#     a==c && c==d            ? g′.αβαα[t,a,b] :
#     error("unsupported g′ pattern")

# @inline Base.setindex!(g′::Patternized_g′{T}, v, t::Int,a::Int,b::Int,c::Int,d::Int) where {T} =
#     a==b && c==d            ? (g′.ααββ[t,a,c]=v) :
#     b==c && c==d            ? (g′.αβββ[t,a,b]=v) :
#     a==c && c==d            ? (g′.αβαα[t,a,b]=v) :
#     error("unsupported g′ pattern")

# @inline Base.getindex(g″::Patternized_g″{T}, t::Int,a::Int,b::Int,c::Int,d::Int) where {T} =
#     a==c && b==d ? g″.αβαβ[t,a,b] :
#     a==d && b==c ? g″.αββα[t,a,b] :
#     error("unsupported g″ pattern")

# @inline Base.setindex!(g″::Patternized_g″{T}, v, t::Int,a::Int,b::Int,c::Int,d::Int) where {T} =
#     a==c && b==d ? (g″.αβαβ[t,a,b]=v) :
#     a==d && b==c ? (g″.αββα[t,a,b]=v) :
#     error("unsupported g″ pattern")

# # for threads
# @inline function inplace_add!(dest::Patternized_g{T}, src::Patternized_g{T}) where {T}
#     dest.ααββ .+= src.ααββ
#     return dest
# end
# @inline function inplace_add!(dest::Patternized_g′{T}, src::Patternized_g′{T}) where {T}
#     dest.ααββ .+= src.ααββ 
#     dest.αβββ .+= src.αβββ
#     dest.αβαα .+= src.αβαα
#     return dest
# end
# @inline function inplace_add!(dest::Patternized_g″{T}, src::Patternized_g″{T}) where {T}
#     dest.αββα .+= src.αββα
#     return dest
# end

