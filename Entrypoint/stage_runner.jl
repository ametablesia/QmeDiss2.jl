# stage_runner.jl
#
# Hierarchical + parallel-aware stage logger for Julia.
#
# This file contains two related layers:
#
# 1. Terminal output layer
#    - colored output
#    - indented stage output
#    - live single-line progress output
#
# 2. Run-state data layer
#    - stores every stage as a StepNode
#    - stores status: PENDING / RUNNING / SUCCESS / FAILED / CANCELLED / SKIPPED
#    - stores parent-child relations
#    - stores events emitted inside stages
#    - supports parallel branches via @async_stage
#
# ------------------------------------------------------------
# Quick usage:
#
# include("stage_runner.jl")
# using .StageRunner
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
# print_tree()
#
# ------------------------------------------------------------
# Expected terminal output:
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
# Run tree:
# ✓ [1] Build agent
#   ✓ [2] Analyze task
#   ✓ [3] Execute task
#
# ------------------------------------------------------------
# Parallel usage:
#
# @stage "Parallel parent" begin
#     t1 = @async_stage "Worker A" begin
#         @emit "A started"
#         sleep(0.2)
#         @emit "A done"
#     end
#
#     t2 = @async_stage "Worker B" begin
#         @emit "B started"
#         sleep(0.1)
#         @emit "B done"
#     end
#
#     wait(t1)
#     wait(t2)
# end
#
# print_tree()
#
# Expected tree shape:
#
# ✓ [1] Parallel parent
#   ✓ [2] Worker A
#   ✓ [3] Worker B
#
# The exact terminal line order may vary because Worker A and Worker B run concurrently.
#
# ------------------------------------------------------------
# Error usage:
#
# try
#     @stage "Simulation" begin
#         @emit "initializing"
#
#         @stage "GPU kernel" begin
#             @emit "launching kernel" color=:yellow
#             error("kernel failed")
#         end
#     end
# catch err
#     println("Caught error: ", err)
# end
#
# Expected terminal output:
#
# ▶ Simulation
#   initializing
#   ▶ GPU kernel
#     launching kernel
#   ✗ GPU kernel
# ✗ Simulation
# Caught error: ErrorException("kernel failed")
#
# Expected tree:
#
# ✗ [1] Simulation
#   ✗ [2] GPU kernel
#
# ------------------------------------------------------------

module StageRunner

using Dates

export StepStatus,
       PENDING,
       RUNNING,
       SUCCESS,
       FAILED,
       CANCELLED,
       SKIPPED,
       StepNode,
       StepEvent,
       RunState,
       Printer,
       DEFAULT_RUN,
       DEFAULT_PRINTER,
       reset_run!,
       current_step_id,
       running_steps,
       failed_steps,
       all_steps,
       tree_lines,
       print_tree,
       emit,
       emit_live,
       finish_live,
       info,
       warn,
       success,
       fail,
       with_stage,
       async_stage,
       @stage,
       @async_stage,
       @emit,
       @info,
       @warn,
       @success,
       @fail,
       enable_color!,
       disable_color!,
       set_indent_unit!,
       demo_success,
       demo_parallel,
       demo_error

# ------------------------------------------------------------
# Status
# ------------------------------------------------------------

@enum StepStatus begin
    PENDING
    RUNNING
    SUCCESS
    FAILED
    CANCELLED
    SKIPPED
end

function status_icon(status::StepStatus)
    if status == PENDING
        return "○"
    elseif status == RUNNING
        return "…"
    elseif status == SUCCESS
        return "✓"
    elseif status == FAILED
        return "✗"
    elseif status == CANCELLED
        return "!"
    elseif status == SKIPPED
        return "-"
    else
        return "?"
    end
end

# ------------------------------------------------------------
# Data layer
# ------------------------------------------------------------

"""
    StepNode

A node in the execution tree.

Fields:
- `id`: integer id
- `title`: stage title
- `status`: PENDING, RUNNING, SUCCESS, FAILED, CANCELLED, or SKIPPED
- `parent`: parent node id, or `nothing`
- `children`: child node ids
- `depth`: tree depth used for display
- `created_at`, `started_at`, `ended_at`: timestamps
- `error`: error string if the stage failed
- `result_summary`: optional result summary
- `metadata`: user-defined metadata
- `task`: Julia Task that created the node
"""
mutable struct StepNode
    id::Int
    title::String
    status::StepStatus
    parent::Union{Int, Nothing}
    children::Vector{Int}
    depth::Int
    created_at::DateTime
    started_at::Union{DateTime, Nothing}
    ended_at::Union{DateTime, Nothing}
    error::Union{String, Nothing}
    result_summary::Union{String, Nothing}
    metadata::Dict{Symbol, Any}
    task::Task
end

"""
    StepEvent

An event emitted during a run.

`step_id` is `nothing` when the event happened outside any stage.
"""
struct StepEvent
    at::DateTime
    step_id::Union{Int, Nothing}
    kind::Symbol
    message::String
    status::Union{StepStatus, Nothing}
    metadata::Dict{Symbol, Any}
end

"""
    RunState

Global execution state.

Important fields:
- `nodes`: id => StepNode
- `roots`: root stage ids
- `events`: emitted events
- `stacks`: task-local stage stacks. This is what makes nested stages work with parallel tasks.
- `lock`: protects all state mutations
"""
mutable struct RunState
    nodes::Dict{Int, StepNode}
    roots::Vector{Int}
    events::Vector{StepEvent}
    next_id::Int
    stacks::IdDict{Task, Vector{Int}}
    lock::ReentrantLock
end

function RunState()
    return RunState(
        Dict{Int, StepNode}(),
        Int[],
        StepEvent[],
        1,
        IdDict{Task, Vector{Int}}(),
        ReentrantLock(),
    )
end

const DEFAULT_RUN = Ref(RunState())

function reset_run!(run::RunState=DEFAULT_RUN[])
    lock(run.lock)
    try
        empty!(run.nodes)
        empty!(run.roots)
        empty!(run.events)
        empty!(run.stacks)
        run.next_id = 1
    finally
        unlock(run.lock)
    end

    return run
end

function _get_stack_unlocked(run::RunState, task::Task=current_task())
    return get!(run.stacks, task, Int[])
end

function _copy_current_stack(run::RunState)
    lock(run.lock)
    try
        return copy(_get_stack_unlocked(run))
    finally
        unlock(run.lock)
    end
end

function _set_current_stack!(run::RunState, stack::Vector{Int})
    lock(run.lock)
    try
        run.stacks[current_task()] = copy(stack)
    finally
        unlock(run.lock)
    end

    return nothing
end

function current_step_id(run::RunState=DEFAULT_RUN[])
    lock(run.lock)
    try
        stack = _get_stack_unlocked(run)
        return isempty(stack) ? nothing : stack[end]
    finally
        unlock(run.lock)
    end
end

function _current_depth(run::RunState)
    lock(run.lock)
    try
        return length(_get_stack_unlocked(run))
    finally
        unlock(run.lock)
    end
end

function record_event!(
    run::RunState,
    step_id::Union{Int, Nothing},
    kind::Symbol,
    message::AbstractString;
    status::Union{StepStatus, Nothing}=nothing,
    metadata::Dict{Symbol, Any}=Dict{Symbol, Any}(),
)
    lock(run.lock)
    try
        push!(
            run.events,
            StepEvent(
                now(),
                step_id,
                kind,
                String(message),
                status,
                metadata,
            ),
        )
    finally
        unlock(run.lock)
    end

    return nothing
end

function _start_step!(
    run::RunState,
    title;
    metadata::Dict{Symbol, Any}=Dict{Symbol, Any}(),
)
    lock(run.lock)
    try
        stack = _get_stack_unlocked(run)
        parent_id = isempty(stack) ? nothing : stack[end]

        depth =
            parent_id === nothing ?
            0 :
            run.nodes[parent_id].depth + 1

        id = run.next_id
        run.next_id += 1

        node = StepNode(
            id,
            string(title),
            RUNNING,
            parent_id,
            Int[],
            depth,
            now(),
            now(),
            nothing,
            nothing,
            nothing,
            copy(metadata),
            current_task(),
        )

        run.nodes[id] = node

        if parent_id === nothing
            push!(run.roots, id)
        else
            push!(run.nodes[parent_id].children, id)
        end

        push!(stack, id)

        push!(
            run.events,
            StepEvent(now(), id, :stage_started, node.title, RUNNING, Dict{Symbol, Any}()),
        )

        return node
    finally
        unlock(run.lock)
    end
end

function _finish_step!(
    run::RunState,
    id::Int,
    status::StepStatus;
    error::Union{String, Nothing}=nothing,
    result_summary::Union{String, Nothing}=nothing,
)
    lock(run.lock)
    try
        node = run.nodes[id]
        node.status = status
        node.ended_at = now()
        node.error = error
        node.result_summary = result_summary

        stack = _get_stack_unlocked(run)
        if !isempty(stack) && stack[end] == id
            pop!(stack)
        else
            # This should not happen in normal usage. Keep the state recoverable.
            idx = findlast(==(id), stack)
            if idx !== nothing
                deleteat!(stack, idx)
            end
        end

        push!(
            run.events,
            StepEvent(now(), id, :stage_finished, node.title, status, Dict{Symbol, Any}()),
        )

        return node
    finally
        unlock(run.lock)
    end
end

function all_steps(run::RunState=DEFAULT_RUN[])
    lock(run.lock)
    try
        return [run.nodes[id] for id in sort(collect(keys(run.nodes)))]
    finally
        unlock(run.lock)
    end
end

function running_steps(run::RunState=DEFAULT_RUN[])
    lock(run.lock)
    try
        return [node for node in values(run.nodes) if node.status == RUNNING]
    finally
        unlock(run.lock)
    end
end

function failed_steps(run::RunState=DEFAULT_RUN[])
    lock(run.lock)
    try
        return [node for node in values(run.nodes) if node.status == FAILED]
    finally
        unlock(run.lock)
    end
end

# ------------------------------------------------------------
# Tree formatting
# ------------------------------------------------------------

function _duration_string(node::StepNode)
    if node.started_at === nothing
        return ""
    end

    end_time = node.ended_at === nothing ? now() : node.ended_at
    ms = Dates.value(end_time - node.started_at)

    if ms < 1000
        return " ($(ms) ms)"
    else
        return " ($(round(ms / 1000; digits=3)) s)"
    end
end

function _tree_lines_unlocked(run::RunState, id::Int, lines::Vector{String})
    node = run.nodes[id]
    prefix = repeat("  ", node.depth)
    line = string(
        prefix,
        status_icon(node.status),
        " [",
        node.id,
        "] ",
        node.title,
        _duration_string(node),
    )

    if node.error !== nothing
        line *= " -- " * node.error
    end

    push!(lines, line)

    for child_id in node.children
        _tree_lines_unlocked(run, child_id, lines)
    end

    return lines
end

"""
    tree_lines(; run=DEFAULT_RUN[])

Return a text representation of the current run tree.
"""
function tree_lines(run::RunState=DEFAULT_RUN[])
    lock(run.lock)
    try
        lines = String[]
        for root_id in run.roots
            _tree_lines_unlocked(run, root_id, lines)
        end
        return lines
    finally
        unlock(run.lock)
    end
end

"""
    print_tree(; run=DEFAULT_RUN[])

Print the current run tree.
"""
function print_tree(run::RunState=DEFAULT_RUN[])
    println("Run tree:")
    for line in tree_lines(run)
        println(line)
    end
    return nothing
end

# ------------------------------------------------------------
# Terminal printer layer
# ------------------------------------------------------------

mutable struct Printer
    indent_unit::String
    use_color::Bool
    live_active::Bool
    lock::ReentrantLock
end

function Printer(; indent_unit="  ", use_color=true)
    return Printer(String(indent_unit), Bool(use_color), false, ReentrantLock())
end

const DEFAULT_PRINTER = Ref(Printer())

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

function _styled_println_locked(
    text::AbstractString;
    color=:normal,
    bold=false,
    printer::Printer=DEFAULT_PRINTER[],
)
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

function _styled_print_locked(
    text::AbstractString;
    color=:normal,
    bold=false,
    printer::Printer=DEFAULT_PRINTER[],
)
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

function _print_line(
    msg;
    depth::Int,
    color=:normal,
    bold=false,
    printer::Printer=DEFAULT_PRINTER[],
)
    lock(printer.lock)
    try
        if printer.live_active
            print('\n')
            printer.live_active = false
        end

        text = repeat(printer.indent_unit, depth) * string(msg)
        _styled_println_locked(text; color=color, bold=bold, printer=printer)
    finally
        unlock(printer.lock)
    end

    return nothing
end

"""
    finish_live(; printer=DEFAULT_PRINTER[])

Finish a live-updated line by printing a newline if needed.
"""
function finish_live(; printer::Printer=DEFAULT_PRINTER[])
    lock(printer.lock)
    try
        if printer.live_active
            print('\n')
            flush(stdout)
            printer.live_active = false
        end
    finally
        unlock(printer.lock)
    end

    return nothing
end

"""
    emit(msg; color=:normal, bold=false, printer=DEFAULT_PRINTER[], run=DEFAULT_RUN[], record=true)

Print a normal indented line and store it as an event.
"""
function emit(
    msg;
    color=:normal,
    bold=false,
    printer::Printer=DEFAULT_PRINTER[],
    run::RunState=DEFAULT_RUN[],
    record::Bool=true,
)
    depth = _current_depth(run)
    _print_line(msg; depth=depth, color=color, bold=bold, printer=printer)

    if record
        step_id = current_step_id(run)
        record_event!(
            run,
            step_id,
            :message,
            string(msg);
            metadata=Dict{Symbol, Any}(:color => color, :bold => bold),
        )
    end

    return nothing
end

"""
    emit_live(msg; color=:normal, bold=false, printer=DEFAULT_PRINTER[], run=DEFAULT_RUN[], record=false)

Update the current terminal line in-place.

By default, live updates are not stored as events because progress updates can be very frequent.
"""
function emit_live(
    msg;
    color=:normal,
    bold=false,
    printer::Printer=DEFAULT_PRINTER[],
    run::RunState=DEFAULT_RUN[],
    record::Bool=false,
)
    depth = _current_depth(run)

    lock(printer.lock)
    try
        print("\r\e[2K")
        text = repeat(printer.indent_unit, depth) * string(msg)
        _styled_print_locked(text; color=color, bold=bold, printer=printer)
        printer.live_active = true
    finally
        unlock(printer.lock)
    end

    if record
        step_id = current_step_id(run)
        record_event!(
            run,
            step_id,
            :live,
            string(msg);
            metadata=Dict{Symbol, Any}(:color => color, :bold => bold),
        )
    end

    return nothing
end

function info(msg; printer::Printer=DEFAULT_PRINTER[], run::RunState=DEFAULT_RUN[])
    emit("ℹ " * string(msg); color=:cyan, printer=printer, run=run)
end

function warn(msg; printer::Printer=DEFAULT_PRINTER[], run::RunState=DEFAULT_RUN[])
    emit("⚠ " * string(msg); color=:yellow, bold=true, printer=printer, run=run)
end

function success(msg; printer::Printer=DEFAULT_PRINTER[], run::RunState=DEFAULT_RUN[])
    emit("✓ " * string(msg); color=:green, printer=printer, run=run)
end

function fail(msg; printer::Printer=DEFAULT_PRINTER[], run::RunState=DEFAULT_RUN[])
    emit("✗ " * string(msg); color=:red, bold=true, printer=printer, run=run)
end

# ------------------------------------------------------------
# Stage execution
# ------------------------------------------------------------

function _error_string(err, bt)
    return sprint(showerror, err, bt)
end

"""
    with_stage(f, title; run=DEFAULT_RUN[], printer=DEFAULT_PRINTER[], color=:cyan, done_color=:green, metadata=Dict())

Run function `f` inside a stage.

The stage is stored in `RunState`.
The indentation stack is restored even if an exception occurs.
"""
function with_stage(
    f,
    title;
    run::RunState=DEFAULT_RUN[],
    printer::Printer=DEFAULT_PRINTER[],
    color=:cyan,
    done_color=:green,
    metadata::Dict{Symbol, Any}=Dict{Symbol, Any}(),
)
    node = _start_step!(run, title; metadata=metadata)
    _print_line("▶ " * node.title; depth=node.depth, color=color, bold=true, printer=printer)

    success_flag = false
    err_string = nothing
    result_summary = nothing

    try
        result = f()
        success_flag = true

        # Keep this compact. Large results should not be stored here.
        if result !== nothing
            result_summary = repr(result)
        end

        return result
    catch err
        bt = catch_backtrace()
        err_string = _error_string(err, bt)
        rethrow()
    finally
        finish_live(; printer=printer)

        final_status = success_flag ? SUCCESS : FAILED
        finished_node = _finish_step!(
            run,
            node.id,
            final_status;
            error=err_string,
            result_summary=result_summary,
        )

        if success_flag
            _print_line("✓ " * finished_node.title; depth=finished_node.depth, color=done_color, printer=printer)
        else
            _print_line("✗ " * finished_node.title; depth=finished_node.depth, color=:red, bold=true, printer=printer)
        end
    end
end

"""
    async_stage(f, title; run=DEFAULT_RUN[], printer=DEFAULT_PRINTER[], ...)

Run a stage in a new Julia Task using `@async`.

The new task inherits the current stage stack, so the async stage becomes a child of
the stage in which it was created.
"""
function async_stage(
    f,
    title;
    run::RunState=DEFAULT_RUN[],
    printer::Printer=DEFAULT_PRINTER[],
    color=:cyan,
    done_color=:green,
    metadata::Dict{Symbol, Any}=Dict{Symbol, Any}(),
)
    parent_stack = _copy_current_stack(run)

    return @async begin
        _set_current_stack!(run, parent_stack)
        with_stage(
            f,
            title;
            run=run,
            printer=printer,
            color=color,
            done_color=done_color,
            metadata=metadata,
        )
    end
end

# ------------------------------------------------------------
# Macro helpers
# ------------------------------------------------------------

function _kw_expr(ex)
    # Macro keyword-like arguments are commonly parsed as Expr(:(=), name, value).
    # A normal function call with keywords needs Expr(:parameters, Expr(:kw, name, value), ...).
    if ex isa Expr && ex.head == :(=) && length(ex.args) == 2 && ex.args[1] isa Symbol
        return Expr(:kw, ex.args[1], esc(ex.args[2]))
    elseif ex isa Expr && ex.head == :kw
        return Expr(:kw, ex.args[1], esc(ex.args[2]))
    else
        return nothing
    end
end

function _call_with_kwargs(func, args)
    positional = Any[]
    keywords = Any[]

    for arg in args
        kw = _kw_expr(arg)
        if kw === nothing
            push!(positional, esc(arg))
        else
            push!(keywords, kw)
        end
    end

    if isempty(keywords)
        return Expr(:call, func, positional...)
    else
        return Expr(:call, func, Expr(:parameters, keywords...), positional...)
    end
end

# ------------------------------------------------------------
# Macros
# ------------------------------------------------------------

"""
    @stage "Title" begin
        ...
    end

Run a block inside an indented and state-tracked stage.
"""
macro stage(title, block)
    return quote
        StageRunner.with_stage(() -> begin
            $(esc(block))
        end, $(esc(title)))
    end
end

"""
    @async_stage "Title" begin
        ...
    end

Run a block as an async child stage.
Returns the created Task.

Use with `wait(task)` or `@sync`.
"""
macro async_stage(title, block)
    return quote
        StageRunner.async_stage(() -> begin
            $(esc(block))
        end, $(esc(title)))
    end
end

"""
    @emit "message" color=:yellow bold=true

Macro form of `emit(...)`.

This macro correctly preserves keyword arguments.
"""
macro emit(args...)
    return _call_with_kwargs(:(StageRunner.emit), args)
end

macro info(args...)
    return _call_with_kwargs(:(StageRunner.info), args)
end

macro warn(args...)
    return _call_with_kwargs(:(StageRunner.warn), args)
end

macro success(args...)
    return _call_with_kwargs(:(StageRunner.success), args)
end

macro fail(args...)
    return _call_with_kwargs(:(StageRunner.fail), args)
end

# ------------------------------------------------------------
# Demos
# ------------------------------------------------------------

function demo_success()
    reset_run!()

    @stage "Build agent" begin
        @emit "creating task graph" color=:cyan

        @stage "Analyze task" begin
            @emit "checking dependencies"
            @emit "some dependency is optional" color=:yellow
        end

        @stage "Execute task" begin
            for i in 1:10
                emit_live("progress: $i / 10"; color=:yellow)
                sleep(0.5)
            end

            finish_live()
            @emit "execution completed" color=:green
        end
    end

    println()
    print_tree()

    return nothing
end

function demo_parallel()
    reset_run!()

    @stage "Parallel parent" begin
        t1 = @async_stage "Worker A" begin
            @emit "A started"
            sleep(0.2)
            @emit "A done"
        end

        t2 = @async_stage "Worker B" begin
            @emit "B started"
            sleep(0.1)
            @emit "B done"
        end

        wait(t1)
        wait(t2)

        @emit "all workers joined" color=:green
    end

    println()
    print_tree()

    return nothing
end

function demo_error()
    reset_run!()

    try
        @stage "Simulation" begin
            @emit "initializing"

            @stage "GPU kernel" begin
                @emit "launching kernel" color=:yellow
                error("kernel failed")
            end
        end
    catch err
        println("Caught error: ", err)
    end

    println()
    print_tree()

    return nothing
end

end # module StageRunner

# ------------------------------------------------------------
# Run this file directly:
#
#     julia stage_runner.jl
#
# or include it:
#
#     include("stage_runner.jl")
#     using .StageRunner
#     StageRunner.demo_success()
#     StageRunner.demo_parallel()
#     StageRunner.demo_error()
# ------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    using .StageRunner

    println("=== success demo ===")
    StageRunner.demo_success()

    println()
    println("=== parallel demo ===")
    StageRunner.demo_parallel()

    println()
    println("=== error demo ===")
    StageRunner.demo_error()
end
