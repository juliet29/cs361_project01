#=
        project1.jl -- This is where the magic happens!

    All of your code must either live in this file, or be `include`d here.
=#

#=
    If you want to use packages, please do so up here.
    Note that you may use any packages in the julia standard library
    (i.e. ones that ship with the julia language) as well as Statistics
    (since we use it in the backend already anyway)
=#

# Example:
using LinearAlgebra

#=
    If you're going to include files, please do so up here. Note that they
    must be saved in project1_jl and you must use the relative path
    (not the absolute path) of the file in the include statement.

    [Good]  include("somefile.jl")
    [Bad]   include("/pathto/project1_jl/somefile.jl")
=#

# Example
include("myhelpers.jl")
include("helpers.jl")

mutable struct BFGS
    Q #modifying matrix
end

function init!(M::BFGS, x)
    # initialize matrix Q with the identity matrix
    # will have dimensions of m * m, wher m is the size of input vector x
    m = length(x)
    M.Q = Matrix(1.0 * I, m, m)
    return M
end

function step!(M::BFGS, f, ∇f, x,  alpha_searches=5, step_lim=10, line_search_lim=10, avail_evals=10)
    num_evals = count(f,∇f)
    # print("\n step! START function calls $num_evals")
    
    Q, g = M.Q, ∇f(x)

    # x′ = line_search(f, x, -Q*g)
    # if iterations == 1
    #     x′, α = line_search_fix_alpha(f=f,x=x, d=-Q * g,g_for_count= g, n=alpha_searches, iterations=iterations, α)
    # else
    x′ = line_search_fix_alpha(f, x, -Q * g, ∇f, alpha_searches, line_search_lim, avail_evals)
    # end
    if x′ == false
        return false
    end

    num_evals = count(f,∇f)
    # print("\n step! function calls $num_evals")
    if num_evals > avail_evals - step_lim
        # print("\n overload in step! $avail_evals")
        return false
    end


    g′ = ∇f(x′)
    δ = x′ - x
    γ = g′ - g

    Q[:] = Q - (δ * γ' * Q + Q * γ * δ') / (δ' * γ) + (1+(γ'*Q*γ)/(δ'*γ))[1] * (δ * δ') / (δ' * γ)
    # print("\n x′",x′)

    return x′
end


"""
    optimize(f, g, x0, n, prob)

Arguments:
    - `f`: Function to be optimized
    - `g`: Gradient function for `f`
    - `x0`: (Vector) Initial position to start from
    - `n`: (Int) Number of evaluations allowed. Remember `g` costs twice of `f`
    - `prob`: (String) Name of the problem. So you can use a different strategy for each problem. E.g. "simple1", "secret2", etc.

Returns:
    - The location of the minimum
"""
function optimize(f, g, x0, avail_evals, probname)
    # TODO keep track of n 
    # using a problem agnostic BFGS method
    # starting matrix to be initialized 
    undef_m = BFGS(undef)
    init_m = init!(undef_m, x0)

    # problem specific
    # probname = PROBS[prob]
    println("probname $probname")
    if probname == "simple1"
        alpha_searches = 12
        step_lim = 2
        line_search_lim = 9
    elseif probname == "simple2"
        alpha_searches = 15
        step_lim = 5
        line_search_lim = 20

    elseif probname  == "simple3"
        alpha_searches = 10
        step_lim = 5
        line_search_lim = 20

    else
        alpha_searches = 5
        step_lim = 5
        line_search_lim = 5
    end


    

    # take steps w BFGS while we less than avail_evals steps taken 
    num_evals = count(f, g)
    # print("\n num_evals start $num_evals")
    iterations = 1
    x_ints = []

    while num_evals < avail_evals - 5
        x_int = step!(init_m, f, g, x0, alpha_searches, step_lim, line_search_lim, avail_evals)
        if x_int == false
            push!(x_ints, x0)
            break
        end
        push!(x_ints, x_int)
        num_evals = count(f, g)
        # print("\n num_evals $num_evals, n $avail_evals, iterations $iterations \n")
        iterations = iterations + 1
    end

    num_evals = count(f, g)
    # print("\n num_evals end $num_evals \n")
    
    x_best = x_ints[1] #x_int

    # println("$probname: alpha_searches $alpha_searches - step_lim $step_lim - line_search_lim $line_search_lim ")

    # plot_opt(x_ints, prob)

    return x_best
end

main("simple2", 10, optimize)

# mymain("simple3", 10, optimize)

