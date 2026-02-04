
struct Config
    n::Int
    dt::Float64
    temperature::Float64
    beta::Float64

    function Config(; n, dt, temperature, beta) # ; 뒤에는 이름으로 매칭 not 위치매칭
        new(n, dt, temperature, beta)   # new in Julia 는 완전 다른 뜻을 가짐... (필드에 값을 채워라...)
    end
end

using Base

macro fatal(msg)
    quote
        printstyled("ERROR: "; color=:red, bold=true)
        println($msg)
        exit(1)
    end
end

function log_error(msg)
    if isa(stdout, Base.TTY)
        printstyled(stderr, "ERROR: $msg\n"; color=:red, bold=true)
    else
        println(stderr, "ERROR: $msg")
    end
    exit(1)
end

# dos rebel
println()
println("    ██████    ██████████   ███████████     |   ")
println("  ███▒▒▒▒███ ▒▒███▒▒▒▒███ ▒█▒▒▒███▒▒▒█     |   ")
println(" ███    ▒▒███ ▒███   ▒▒███▒   ▒███  ▒      |   ")
println("▒███   ██▒███ ▒███    ▒███    ▒███         |   ")
println("▒▒███ ▒▒████  ▒███    ███     ▒███         |   ")
println(" ▒▒▒██████▒██ ██████████      █████        |   ")
println("   ▒▒▒▒▒▒ ▒▒ ▒▒▒▒▒▒▒▒▒▒      ▒▒▒▒▒         |   ")
println()                                    
                                      
println("v0.0.1")                                      

if length(ARGS) < 1
    i = 5
    log_error("Usage: julia Launcher.jl $i <params_path>")    
end

params = include("config.jl")
print(params)