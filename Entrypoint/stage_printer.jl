# stage_printer.jl
#
# A small hierarchical terminal printer for Julia.
#
# Features:
# - Colored output via printstyled
# - Indentation by nested stages
# - @stage macro for structured task blocks
# - @emit macro for indented messages
# - Live single-line update with emit_live
# - Safe indentation cleanup using try/finally
#
# ------------------------------------------------------------
# Example usage:
#
# include("stage_printer.jl")
# using .StagePrinter
#
# @stage "Build agent" begin
#     @emit "creating task graph" color=:cyan
#
#     @stage "Analyze task" begin
#         @emit "checking dependencies"
#         @emit "some dependency is optional" color=:yellow
#     end
#
#     @stage "Execute task" begin
#         for i in 1:10
#             emit_live("progress: $i / 10"; color=:yellow)
#             sleep(0.05)
#         end
#         finish_live()
#         @emit "execution completed" color=:green
#     end
# end
#
# ------------------------------------------------------------
# Expected output:
#
# ▶ Build agent
#   creating task graph
#   ▶ Analyze task
#     checking dependencies
#     some dependency is optional
#   ✓ Analyze task
#   ▶ Execute task
#     progress: 10 / 10
#     execution completed
#   ✓ Execute task
# ✓ Build agent
#
# If an error happens inside a stage:
#
# @stage "Simulation" begin
#     @emit "initializing"
#
#     @stage "GPU kernel" begin
#         @emit "launching kernel" color=:yellow
#         error("kernel failed")
#     end
# end
#
# Expected output:
#
# ▶ Simulation
#   initializing
#   ▶ GPU kernel
#     launching kernel
#   ✗ GPU kernel
# ✗ Simulation
# ERROR: kernel failed
#
# ------------------------------------------------------------

module StagePrinter

export Printer,
       DEFAULT_PRINTER,
       emit,
       emit_live,
       finish_live,
       info,
       warn,
       success,
       fail,
       with_stage,
       @stage,
       @emit,
       @info,
       @warn,
       @success,
       @fail,
       enable_color!,
       disable_color!,
       set_indent_unit!

"""
    Printer

State object for hierarchical terminal output.

Fields:
- `indent`: current indentation level
- `indent_unit`: text repeated for one indentation level
- `use_color`: whether to use Julia's `printstyled`
- `live_active`: whether the last output was a live line
"""
mutable struct Printer
    indent::Int
    indent_unit::String
    use_color::Bool
    live_active::Bool
end

"""
    DEFAULT_PRINTER

Global default printer used by `@stage`, `@emit`, `emit`, etc.
"""
const DEFAULT_PRINTER = Ref(Printer(0, "  ", true, false))

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

function enable_color!(printer::Printer=DEFAULT_PRINTER[])
    printer.use_color = true
    return printer
end

function disable_color!(printer::Printer=DEFAULT_PRINTER[])
    printer.use_color = false
    return printer
end

function set_indent_unit!(unit::AbstractString, printer::Printer=DEFAULT_PRINTER[])
    printer.indent_unit = String(unit)
    return printer
end

# ------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------

_prefix(printer::Printer) = repeat(printer.indent_unit, printer.indent)

function _styled_println(text::AbstractString; color=:normal, bold=false, printer::Printer=DEFAULT_PRINTER[])
    if printer.use_color && (color != :normal || bold)
        if color == :normal
            printstyled(text; bold=bold)
        else
            printstyled(text; color=color, bold=bold)
        end
        print('\n')
    else
        println(text)
    end

    flush(stdout)
    return nothing
end

function _styled_print(text::AbstractString; color=:normal, bold=false, printer::Printer=DEFAULT_PRINTER[])
    if printer.use_color && (color != :normal || bold)
        if color == :normal
            printstyled(text; bold=bold)
        else
            printstyled(text; color=color, bold=bold)
        end
    else
        print(text)
    end

    flush(stdout)
    return nothing
end

"""
    finish_live(; printer=DEFAULT_PRINTER[])

Finish a live-updated line by printing a newline if needed.
Use this after a sequence of `emit_live(...)` calls.
"""
function finish_live(; printer::Printer=DEFAULT_PRINTER[])
    if printer.live_active
        print('\n')
        flush(stdout)
        printer.live_active = false
    end

    return nothing
end

# ------------------------------------------------------------
# Public printing API
# ------------------------------------------------------------

"""
    emit(msg; color=:normal, bold=false, printer=DEFAULT_PRINTER[])

Print a normal indented line.
"""
function emit(msg; color=:normal, bold=false, printer::Printer=DEFAULT_PRINTER[])
    finish_live(; printer=printer)

    text = _prefix(printer) * string(msg)
    _styled_println(text; color=color, bold=bold, printer=printer)

    return nothing
end

"""
    emit_live(msg; color=:normal, bold=false, printer=DEFAULT_PRINTER[])

Update the current terminal line in-place.

Useful for progress-like output.
Call `finish_live()` before normal multi-line output if needed.
"""
function emit_live(msg; color=:normal, bold=false, printer::Printer=DEFAULT_PRINTER[])
    # '\r' moves to beginning of current line.
    # '\e[2K' clears the entire current line.
    print("\r\e[2K")

    text = _prefix(printer) * string(msg)
    _styled_print(text; color=color, bold=bold, printer=printer)

    printer.live_active = true
    return nothing
end

function info(msg; printer::Printer=DEFAULT_PRINTER[])
    emit("ℹ " * string(msg); color=:cyan, printer=printer)
end

function warn(msg; printer::Printer=DEFAULT_PRINTER[])
    emit("⚠ " * string(msg); color=:yellow, bold=true, printer=printer)
end

function success(msg; printer::Printer=DEFAULT_PRINTER[])
    emit("✓ " * string(msg); color=:green, printer=printer)
end

function fail(msg; printer::Printer=DEFAULT_PRINTER[])
    emit("✗ " * string(msg); color=:red, bold=true, printer=printer)
end

"""
    with_stage(f, title; printer=DEFAULT_PRINTER[], color=:cyan, done_color=:green)

Run function `f` inside an indented stage.

The indentation level is restored even when an exception occurs.
"""
function with_stage(
    f,
    title;
    printer::Printer=DEFAULT_PRINTER[],
    color=:cyan,
    done_color=:green,
)
    finish_live(; printer=printer)

    title_string = string(title)
    emit("▶ " * title_string; color=color, bold=true, printer=printer)

    printer.indent += 1
    success_flag = false

    try
        result = f()
        success_flag = true
        return result
    finally
        finish_live(; printer=printer)

        # Make sure indentation is restored even after an error.
        printer.indent = max(printer.indent - 1, 0)

        if success_flag
            emit("✓ " * title_string; color=done_color, printer=printer)
        else
            emit("✗ " * title_string; color=:red, bold=true, printer=printer)
        end
    end
end

# ------------------------------------------------------------
# Macros
# ------------------------------------------------------------

"""
    @stage "Title" begin
        ...
    end

Run a block inside an indented stage.

Example:

    @stage "Load data" begin
        @emit "opening file"
        @stage "Parse" begin
            @emit "building AST"
        end
    end
"""
macro stage(title, block)
    return quote
        StagePrinter.with_stage(() -> begin
            $(esc(block))
        end, $(esc(title)))
    end
end

"""
    @emit "message" color=:yellow bold=true

Macro form of `emit(...)`.
"""
macro emit(args...)
    return Expr(:call, :(StagePrinter.emit), map(esc, args)...)
end

macro info(args...)
    return Expr(:call, :(StagePrinter.info), map(esc, args)...)
end

macro warn(args...)
    return Expr(:call, :(StagePrinter.warn), map(esc, args)...)
end

macro success(args...)
    return Expr(:call, :(StagePrinter.success), map(esc, args)...)
end

macro fail(args...)
    return Expr(:call, :(StagePrinter.fail), map(esc, args)...)
end

# ------------------------------------------------------------
# Demo
# ------------------------------------------------------------

"""
    demo_success()

Run a successful nested-stage demo.
"""
function demo_success()
    @stage "Build agent" begin
        @emit "creating task graph" color=:cyan

        @stage "Analyze task" begin
            @emit "checking dependencies"
            @emit "some dependency is optional" color=:yellow
        end

        @stage "Execute task" begin
            for i in 1:10
                emit_live("progress: $i / 10"; color=:yellow)
                sleep(0.05)
            end

            finish_live()
            @emit "execution completed" color=:green
        end
    end

    return nothing
end

"""
    demo_error()

Run a nested-stage demo that intentionally throws an error.
"""
function demo_error()
    @stage "Simulation" begin
        @emit "initializing"

        @stage "GPU kernel" begin
            @emit "launching kernel" color=:yellow
            error("kernel failed")
        end
    end

    return nothing
end

end # module StagePrinter

# ------------------------------------------------------------
# Run this file directly:
#
#     julia stage_printer.jl
#
# or include it from another file:
#
#     include("stage_printer.jl")
#     using .StagePrinter
#     StagePrinter.demo_success()
# ------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    using .StagePrinter

    StagePrinter.demo_success()

    println()
    println("Now running error demo. The error is expected.")
    println()

    try
        StagePrinter.demo_error()
    catch err
        println("Caught error: ", err)
    end
end
