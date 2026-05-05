# corner_tree_printer.jl
#
# Minimal Unicode corner-style tree printer for Julia.
#
# This file intentionally focuses only on rendering a tree.
# It does not contain progress tracking, live terminal update, async state, or task logging.
#
# Style:
#
# Root
# ├── Child A
# │   ├── Child A1
# │   ╰── Child A2
# ╰── Child B
#     ╰── Child B1
#
# Main characters:
#
# ├──   middle child
# ╰──   last child, rounded corner style
# │     vertical continuation line
#
# ------------------------------------------------------------
# Quick usage:
#
# include("corner_tree_printer.jl")
# using .CornerTreePrinter
#
# root = node("Project",
#     node("Load data",
#         node("Read file"),
#         node("Parse rows"),
#         node("Validate schema"),
#     ),
#     node("Build model",
#         node("Initialize"),
#         node("Compile"),
#     ),
#     node("Run simulation",
#         node("Step 1"),
#         node("Step 2"),
#         node("Save result"),
#     ),
# )
#
# print_tree(root)
#
# ------------------------------------------------------------
# Expected output:
#
# Project
# ├── Load data
# │   ├── Read file
# │   ├── Parse rows
# │   ╰── Validate schema
# ├── Build model
# │   ├── Initialize
# │   ╰── Compile
# ╰── Run simulation
#     ├── Step 1
#     ├── Step 2
#     ╰── Save result
#
# ------------------------------------------------------------
# Status-like labels can be embedded directly in the node label:
#
# root = node("… Main task",
#     node("✓ Understand request"),
#     node("✓ Load context"),
#     node("… Execute plan",
#         node("✓ Generate code"),
#         node("✗ Run test",
#             node("! MethodError: emit(::String, ::Symbol)"),
#         ),
#         node("… Patch code"),
#     ),
#     node("○ Final response"),
# )
#
# print_tree(root)
#
# Expected output:
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

module CornerTreePrinter

export TreeNode,
       node,
       add_child!,
       tree_lines,
       print_tree,
       demo_basic,
       demo_status

"""
    TreeNode

A minimal tree node.

Fields:
- `label`: text displayed for the node
- `children`: child nodes
"""
mutable struct TreeNode
    label::String
    children::Vector{TreeNode}
end

"""
    TreeNode(label)

Create a tree node with no children.
"""
TreeNode(label) = TreeNode(string(label), TreeNode[])

"""
    node(label, children...)

Convenience constructor.

Example:

    root = node("Root",
        node("A"),
        node("B",
            node("B1"),
            node("B2"),
        ),
    )
"""
function node(label, children::TreeNode...)
    return TreeNode(string(label), collect(children))
end

"""
    add_child!(parent, child)

Append a child node to a parent node.
"""
function add_child!(parent::TreeNode, child::TreeNode)
    push!(parent.children, child)
    return parent
end

# ------------------------------------------------------------
# Rendering
# ------------------------------------------------------------

const BRANCH_MIDDLE = "├── "
const BRANCH_LAST   = "╰── "
const PIPE          = "│   "
const SPACE         = "    "

function _render_children!(
    lines::Vector{String},
    children::Vector{TreeNode},
    prefix::String,
)
    n = length(children)

    for (i, child) in enumerate(children)
        is_last = i == n
        connector = is_last ? BRANCH_LAST : BRANCH_MIDDLE

        push!(lines, prefix * connector * child.label)

        child_prefix = prefix * (is_last ? SPACE : PIPE)

        if !isempty(child.children)
            _render_children!(lines, child.children, child_prefix)
        end
    end

    return lines
end

"""
    tree_lines(root; show_root=true)

Return the rendered tree as a vector of strings.

If `show_root=false`, only the root's children are printed.
"""
function tree_lines(root::TreeNode; show_root::Bool=true)
    lines = String[]

    if show_root
        push!(lines, root.label)
        _render_children!(lines, root.children, "")
    else
        _render_children!(lines, root.children, "")
    end

    return lines
end

"""
    print_tree([io], root; show_root=true)

Print a tree to `io`.

Examples:

    print_tree(root)
    print_tree(stderr, root)
"""
function print_tree(io::IO, root::TreeNode; show_root::Bool=true)
    for line in tree_lines(root; show_root=show_root)
        println(io, line)
    end

    return nothing
end

function print_tree(root::TreeNode; show_root::Bool=true)
    return print_tree(stdout, root; show_root=show_root)
end

# ------------------------------------------------------------
# Demos
# ------------------------------------------------------------

function demo_basic()
    root = node("Project",
        node("Load data",
            node("Read file"),
            node("Parse rows"),
            node("Validate schema"),
        ),
        node("Build model",
            node("Initialize"),
            node("Compile"),
        ),
        node("Run simulation",
            node("Step 1"),
            node("Step 2"),
            node("Save result"),
        ),
    )

    print_tree(root)
    return root
end

function demo_status()
    root = node("… Main task",
        node("✓ Understand request"),
        node("✓ Load context"),
        node("… Execute plan",
            node("✓ Generate code"),
            node("✗ Run test",
                node("! MethodError: emit(::String, ::Symbol)"),
            ),
            node("… Patch code"),
        ),
        node("○ Final response"),
    )

    print_tree(root)
    return root
end

end # module CornerTreePrinter

# ------------------------------------------------------------
# Run this file directly:
#
#     julia corner_tree_printer.jl
#
# or include it:
#
#     include("corner_tree_printer.jl")
#     using .CornerTreePrinter
#     CornerTreePrinter.demo_basic()
#     CornerTreePrinter.demo_status()
# ------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    using .CornerTreePrinter

    println("=== basic demo ===")
    CornerTreePrinter.demo_basic()

    println()
    println("=== status demo ===")
    CornerTreePrinter.demo_status()
end
