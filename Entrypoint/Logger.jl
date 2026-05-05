# corner_live_tree_printer.jl
#
# Live tree-style terminal printer for Julia.
#
# Goal:
# - Print stages like a real tree:
#
#   … Main task
#   ├── ✓ Understand request
#   ├── ✓ Load context
#   ├── … Execute plan
#   │   ├── ✓ Generate code
#   │   ├── ✗ Run test
#   │   │   ╰── ! MethodError: emit(::String, ::Symbol)
#   │   ╰── … Patch code
#   ╰── ○ Final response
#
# - No duplicated "stage start" / "stage finish" lines.
#   A stage appears once, and its status icon changes from … to ✓ or ✗.
#
# - Tree connector symbols are NOT colored.
#   Only status icons and labels can be colored.
#
# - blank() / @blank can insert true blank lines into the rendered tree.
#
# - This is a terminal renderer, not a full execution-state database.
#   It keeps a small in-memory tree only so it can redraw the terminal.
#
# ------------------------------------------------------------
# Basic usage:
#
# include("corner_live_tree_printer.jl")
# using .CornerLiveTreePrinter
#
# @stage "Simulation" begin
#     @info "initializing context"
#
#     @stage "Time evolution" begin
#         for i in 1:10
#             loss = exp(-i / 3)
#             @emit_livef "iteration = %03d / %03d, loss = %.6e" i 10 loss color=:yellow
#             sleep(0.05)
#         end
#         finish_live()
#     end
#
#     @success "done"
# end
#
# ------------------------------------------------------------
# Direct status demo:
#
# CornerLiveTreePrinter.demo_status()
#
# Expected final output:
#
# … Main task
# ├── ✓ Understand request
# ├── ✓ Load context
# ├── … Execute plan
# │   ├── ✓ Generate code
# │   ├── ✗ Run test
# │   │   ╰── ! MethodError: emit(::String, ::Symbol)
# │   ╰── … Patch code
# ╰── ○ Final response
#
# ------------------------------------------------------------

module CornerLiveTreePrinter

using Printf

export Printer,
       TreeNode,
       DEFAULT_PRINTER,
       reset_printer!,
       clear_render!,
       render!,
       emit,
       emit_live,
       emitf,
       emit_livef,
       finish_live,
       info,
       warn,
       success,
       fail,
       pending,
       blank,
       with_stage,
       @stage,
       @emit,
       @emit_live,
       @emitf,
       @emit_livef,
       @info,
       @warn,
       @success,
       @fail,
       @pending,
       @blank,
       enable_color!,
       disable_color!,
       demo_success,
       demo_status,
       demo_error

# ------------------------------------------------------------
# Tree node
# ------------------------------------------------------------

mutable struct TreeNode
    label::String
    status::Symbol
    color::Symbol
    bold::Bool
    children::Vector{TreeNode}
    parent::Union{TreeNode, Nothing}
end

function TreeNode(
    label;
    status::Symbol=:message,
    color::Symbol=:normal,
    bold::Bool=false,
    parent::Union{TreeNode, Nothing}=nothing,
)
    return TreeNode(string(label), status, color, bold, TreeNode[], parent)
end

# ------------------------------------------------------------
# Printer
# ------------------------------------------------------------

mutable struct Printer
    roots::Vector{TreeNode}
    stack::Vector{TreeNode}
    rendered_lines::Int
    use_color::Bool
    live_node::Union{TreeNode, Nothing}
    lock::ReentrantLock
end

function Printer(; use_color::Bool=true)
    return Printer(TreeNode[], TreeNode[], 0, use_color, nothing, ReentrantLock())
end

const DEFAULT_PRINTER = Ref(Printer())

function reset_printer!(printer::Printer=DEFAULT_PRINTER[])
    lock(printer.lock)
    try
        printer.roots = TreeNode[]
        printer.stack = TreeNode[]
        printer.rendered_lines = 0
        printer.live_node = nothing
    finally
        unlock(printer.lock)
    end

    return printer
end

function enable_color!(printer::Printer=DEFAULT_PRINTER[])
    printer.use_color = true
    return printer
end

function disable_color!(printer::Printer=DEFAULT_PRINTER[])
    printer.use_color = false
    return printer
end

# ------------------------------------------------------------
# Status icons/colors
# ------------------------------------------------------------

function _status_icon(status::Symbol)
    if status == :blank
        return ""
    elseif status == :pending
        return "○"
    elseif status == :running
        return "…"
    elseif status == :success
        return "✓"
    elseif status == :failed
        return "✗"
    elseif status == :warning
        return "⚠"
    elseif status == :info
        return "ℹ"
    elseif status == :cancelled
        return "!"
    elseif status == :message
        return ""
    else
        return string(status)
    end
end

function _status_color(status::Symbol)
    if status == :blank
        return :normal
    elseif status == :pending
        return :normal
    elseif status == :running
        return :yellow
    elseif status == :success
        return :green
    elseif status == :failed
        return :red
    elseif status == :warning
        return :yellow
    elseif status == :info
        return :cyan
    elseif status == :cancelled
        return :red
    elseif status == :message
        return :normal
    else
        return :normal
    end
end

function _effective_color(node::TreeNode)
    return node.color == :normal ? _status_color(node.status) : node.color
end

# ------------------------------------------------------------
# Low-level styled print
# ------------------------------------------------------------

function _styled_print(
    printer::Printer,
    text::AbstractString;
    color::Symbol=:normal,
    bold::Bool=false,
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

    return nothing
end

function _print_node_text(printer::Printer, node::TreeNode)
    icon = _status_icon(node.status)
    color = _effective_color(node)

    if isempty(icon)
        _styled_print(printer, node.label; color=color, bold=node.bold)
    else
        _styled_print(printer, icon; color=color, bold=true)
        print(" ")
        _styled_print(printer, node.label; color=color, bold=node.bold)
    end

    return nothing
end

# ------------------------------------------------------------
# Tree rendering
# ------------------------------------------------------------

# const BRANCH_MIDDLE = "├─ "
# const BRANCH_LAST   = "╰─ "
# const PIPE          = "│  "
# const SPACE         = "   "

const BRANCH_MIDDLE = "│  "
const BRANCH_LAST   = "╰─ "
const PIPE          = "│  "
const SPACE         = "   "

function _count_lines(nodes::Vector{TreeNode})
    total = 0

    for node in nodes
        total += 1
        total += _count_lines(node.children)
    end

    return total
end

function _clear_previous_render_unlocked(printer::Printer)
    n = printer.rendered_lines

    if n <= 0
        return nothing
    end

    for _ in 1:n
        print("\e[1A")
        print("\e[2K")
    end

    printer.rendered_lines = 0
    return nothing
end

function clear_render!(printer::Printer=DEFAULT_PRINTER[])
    lock(printer.lock)
    try
        _clear_previous_render_unlocked(printer)
        flush(stdout)
    finally
        unlock(printer.lock)
    end

    return nothing
end

function _render_node_unlocked(
    printer::Printer,
    node::TreeNode,
    prefix::String,
    connector::String,
)
    if node.status == :blank
        # A blank node intentionally produces a true empty line.
        # It does not print tree connector symbols, status icons, or colored text.
        print('\n')
        return nothing
    end

    # Connector symbols are intentionally printed without color.
    print(prefix)
    print(connector)
    _print_node_text(printer, node)
    print('\n')

    n = length(node.children)

    for (i, child) in enumerate(node.children)
        is_last = i == n
        child_connector = is_last ? BRANCH_LAST : BRANCH_MIDDLE
        child_prefix = prefix * (isempty(connector) ? "" : (connector == BRANCH_LAST ? SPACE : PIPE))

        _render_node_unlocked(printer, child, child_prefix, child_connector)
    end

    return nothing
end

function _render_unlocked(printer::Printer)
    _clear_previous_render_unlocked(printer)

    for root in printer.roots
        _render_node_unlocked(printer, root, "", "")
    end

    printer.rendered_lines = _count_lines(printer.roots)
    flush(stdout)

    return nothing
end

"""
    render!(; printer=DEFAULT_PRINTER[])

Redraw the current tree.
"""
function render!(; printer::Printer=DEFAULT_PRINTER[])
    lock(printer.lock)
    try
        _render_unlocked(printer)
    finally
        unlock(printer.lock)
    end

    return nothing
end

# ------------------------------------------------------------
# Node operations
# ------------------------------------------------------------

function _current_parent(printer::Printer)
    return isempty(printer.stack) ? nothing : printer.stack[end]
end

function _append_node_unlocked(
    printer::Printer,
    label;
    status::Symbol=:message,
    color::Symbol=:normal,
    bold::Bool=false,
)
    parent = _current_parent(printer)
    node = TreeNode(label; status=status, color=color, bold=bold, parent=parent)

    if parent === nothing
        push!(printer.roots, node)
    else
        push!(parent.children, node)
    end

    return node
end

function _update_node_unlocked!(
    node::TreeNode,
    label;
    status::Symbol=node.status,
    color::Symbol=node.color,
    bold::Bool=node.bold,
)
    node.label = string(label)
    node.status = status
    node.color = color
    node.bold = bold
    return node
end

# ------------------------------------------------------------
# Public emit functions
# ------------------------------------------------------------

"""
    emit(msg; status=:message, color=:normal, bold=false, printer=DEFAULT_PRINTER[])

Append a message node and redraw the tree.
"""
function emit(
    msg;
    status::Symbol=:message,
    color::Symbol=:normal,
    bold::Bool=false,
    printer::Printer=DEFAULT_PRINTER[],
)
    lock(printer.lock)
    try
        printer.live_node = nothing
        _append_node_unlocked(printer, msg; status=status, color=color, bold=bold)
        _render_unlocked(printer)
    finally
        unlock(printer.lock)
    end

    return nothing
end

"""
    emit_live(msg; status=:message, color=:normal, bold=false, printer=DEFAULT_PRINTER[])

Update one live node under the current stage.

Repeated calls update the same line until `finish_live()` is called.
"""
function emit_live(
    msg;
    status::Symbol=:message,
    color::Symbol=:normal,
    bold::Bool=false,
    printer::Printer=DEFAULT_PRINTER[],
)
    lock(printer.lock)
    try
        if printer.live_node === nothing
            printer.live_node = _append_node_unlocked(printer, msg; status=status, color=color, bold=bold)
        else
            _update_node_unlocked!(printer.live_node, msg; status=status, color=color, bold=bold)
        end

        _render_unlocked(printer)
    finally
        unlock(printer.lock)
    end

    return nothing
end

"""
    finish_live(; printer=DEFAULT_PRINTER[])

End the current live node. The line remains in the tree with its latest text.
"""
function finish_live(; printer::Printer=DEFAULT_PRINTER[])
    lock(printer.lock)
    try
        printer.live_node = nothing
        _render_unlocked(printer)
    finally
        unlock(printer.lock)
    end

    return nothing
end

function emitf(
    fmt::AbstractString,
    args...;
    status::Symbol=:message,
    color::Symbol=:normal,
    bold::Bool=false,
    printer::Printer=DEFAULT_PRINTER[],
)
    msg = Printf.format(Printf.Format(fmt), args...)
    emit(msg; status=status, color=color, bold=bold, printer=printer)
    return nothing
end

function emit_livef(
    fmt::AbstractString,
    args...;
    status::Symbol=:message,
    color::Symbol=:normal,
    bold::Bool=false,
    printer::Printer=DEFAULT_PRINTER[],
)
    msg = Printf.format(Printf.Format(fmt), args...)
    emit_live(msg; status=status, color=color, bold=bold, printer=printer)
    return nothing
end

function info(msg; printer::Printer=DEFAULT_PRINTER[])
    emit(msg; status=:info, printer=printer)
end

function warn(msg; printer::Printer=DEFAULT_PRINTER[])
    emit(msg; status=:warning, bold=true, printer=printer)
end

function success(msg; printer::Printer=DEFAULT_PRINTER[])
    emit(msg; status=:success, printer=printer)
end

function fail(msg; printer::Printer=DEFAULT_PRINTER[])
    emit(msg; status=:failed, bold=true, printer=printer)
end

function pending(msg; printer::Printer=DEFAULT_PRINTER[])
    emit(msg; status=:pending, printer=printer)
end

"""
    blank(n::Integer=1; printer=DEFAULT_PRINTER[])

Insert true blank line nodes into the rendered tree.

Unlike `emit("")`, this does not print tree connector symbols, status icons,
or colored text. The blank line is still counted in `rendered_lines` so live
redraw remains stable.
"""
function blank(n::Integer=1; printer::Printer=DEFAULT_PRINTER[])
    if n < 0
        throw(ArgumentError("blank line count must be non-negative"))
    end

    lock(printer.lock)
    try
        printer.live_node = nothing

        for _ in 1:n
            _append_node_unlocked(printer, ""; status=:blank)
        end

        _render_unlocked(printer)
    finally
        unlock(printer.lock)
    end

    return nothing
end

# ------------------------------------------------------------
# Stage
# ------------------------------------------------------------

"""
    with_stage(f, title; printer=DEFAULT_PRINTER[], color=:normal, bold=false)

Create a stage node, mark it running, run `f`, then mark it success or failed.

The stage appears as one line. It is not printed once for start and again for finish.
"""
function with_stage(
    f,
    title;
    printer::Printer=DEFAULT_PRINTER[],
    color::Symbol=:normal,
    bold::Bool=true,
)
    local node

    lock(printer.lock)
    try
        printer.live_node = nothing
        node = _append_node_unlocked(
            printer,
            title;
            status=:running,
            color=color,
            bold=bold,
        )
        push!(printer.stack, node)
        _render_unlocked(printer)
    finally
        unlock(printer.lock)
    end

    success_flag = false

    try
        result = f()
        success_flag = true
        return result
    finally
        lock(printer.lock)
        try
            printer.live_node = nothing

            if !isempty(printer.stack) && printer.stack[end] === node
                pop!(printer.stack)
            else
                idx = findlast(x -> x === node, printer.stack)
                if idx !== nothing
                    deleteat!(printer.stack, idx)
                end
            end

            node.status = success_flag ? :success : :failed
            node.color = success_flag ? :normal : :red
            node.bold = !success_flag

            _render_unlocked(printer)
        finally
            unlock(printer.lock)
        end
    end
end

# ------------------------------------------------------------
# Macro keyword handling
# ------------------------------------------------------------

function _kw_expr(ex)
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

macro stage(title, block)
    return quote
        $(GlobalRef(CornerLiveTreePrinter, :with_stage))(() -> begin
            $(esc(block))
        end, $(esc(title)))
    end
end

macro emit(args...)
    return _call_with_kwargs(GlobalRef(CornerLiveTreePrinter, :emit), args)
end

macro emit_live(args...)
    return _call_with_kwargs(GlobalRef(CornerLiveTreePrinter, :emit_live), args)
end

macro emitf(args...)
    return _call_with_kwargs(GlobalRef(CornerLiveTreePrinter, :emitf), args)
end

macro emit_livef(args...)
    return _call_with_kwargs(GlobalRef(CornerLiveTreePrinter, :emit_livef), args)
end

macro info(args...)
    return _call_with_kwargs(GlobalRef(CornerLiveTreePrinter, :info), args)
end

macro warn(args...)
    return _call_with_kwargs(GlobalRef(CornerLiveTreePrinter, :warn), args)
end

macro success(args...)
    return _call_with_kwargs(GlobalRef(CornerLiveTreePrinter, :success), args)
end

macro fail(args...)
    return _call_with_kwargs(GlobalRef(CornerLiveTreePrinter, :fail), args)
end

macro pending(args...)
    return _call_with_kwargs(GlobalRef(CornerLiveTreePrinter, :pending), args)
end

macro blank(args...)
    return _call_with_kwargs(GlobalRef(CornerLiveTreePrinter, :blank), args)
end

# ------------------------------------------------------------
# Demos
# ------------------------------------------------------------

function demo_success()
    reset_printer!()

    @stage "Simulation" begin
        @info "initializing context"

        @stage "Time evolution" begin
            for i in 1:10
                loss = exp(-i / 3)
                @emit_livef "iteration = %03d / %03d, loss = %.6e" i 10 loss color=:yellow
                sleep(0.5)
            end

            finish_live()
        end

        @success "done"
    end

    return nothing
end

function demo_status()
    reset_printer!()

    lock(DEFAULT_PRINTER[].lock)
    try
        p = DEFAULT_PRINTER[]

        main = _append_node_unlocked(p, "Main task"; status=:running)

        push!(p.stack, main)
        _append_node_unlocked(p, "Understand request"; status=:success)
        _append_node_unlocked(p, "Load context"; status=:success)

        execute = _append_node_unlocked(p, "Execute plan"; status=:running)
        push!(p.stack, execute)

        _append_node_unlocked(p, "Generate code"; status=:success)

        run_test = _append_node_unlocked(p, "Run test"; status=:failed, bold=true)
        push!(p.stack, run_test)
        _append_node_unlocked(p, "MethodError: emit(::String, ::Symbol)"; status=:cancelled)
        pop!(p.stack)

        _append_node_unlocked(p, "Patch code"; status=:running)
        pop!(p.stack)

        _append_node_unlocked(p, "Final response"; status=:pending)
        pop!(p.stack)

        _render_unlocked(p)
    finally
        unlock(DEFAULT_PRINTER[].lock)
    end

    return nothing
end

function demo_error()
    reset_printer!()

    @stage "Simulation" begin
        @info "initializing"

        @stage "GPU kernel" begin
            @emit "launching kernel" color=:yellow
            error("kernel failed")
        end
    end

    return nothing
end

function sss()

    reset_printer!()

    @blank

    @stage "Effective Oscillator로 분해하기" begin
        @emit "initializing"
        @success "done!"
    end

    @blank

    @stage "g, g′, g″ 구하기" begin
        @emit "initializing"

        for i in 1:10
            @emit_livef "iteration = %03d / %03d" i 10
            sleep(0.5)
        end

        finish_live()

        @success "done!"
    end
end

end # module CornerLiveTreePrinter

# ------------------------------------------------------------
# Run this file directly:
#
#     julia corner_live_tree_printer.jl
#
# or include it:
#
#     include("corner_live_tree_printer.jl")
#     using .CornerLiveTreePrinter
#     CornerLiveTreePrinter.demo_success()
#     CornerLiveTreePrinter.demo_status()
# ------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    using .CornerLiveTreePrinter

    CornerLiveTreePrinter.sss()


    println("=== status demo ===")
    CornerLiveTreePrinter.demo_status()

    println()
    println("=== success demo ===")
    CornerLiveTreePrinter.demo_success()

    println()
    println("=== error demo ===")
    try
        CornerLiveTreePrinter.demo_error()
    catch err
        println("Caught error: ", err)
    end
end
