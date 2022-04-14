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
# using LinearAlgebra

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

function step!(M::BFGS, f, ∇f, x, alpha_searches=5, iterations=1, α=0.1, n=10)
    num_evals = count(f,∇f)
    print("\n step! START function calls $num_evals")
    
    Q, g = M.Q, ∇f(x)

    # x′ = line_search(f, x, -Q*g)
    # if iterations == 1
    #     x′, α = line_search_fix_alpha(f=f,x=x, d=-Q * g,g_for_count= g, n=alpha_searches, iterations=iterations, α)
    # else
    x′, α = line_search_fix_alpha(f, x, -Q * g, ∇f, alpha_searches, iterations, α)
    # end

    num_evals = count(f,∇f)
    print("\n step! function calls $num_evals")
    # if num_evals > n -1
    #     print("\n overload in step! $n")
    #     return false, α
    # end


    g′ = ∇f(x′)
    δ = x′ - x
    γ = g′ - g

    Q[:] = Q - (δ * γ' * Q + Q * γ * δ') / (δ' * γ) + (1+(γ'*Q*γ)/(δ'*γ))[1] * (δ * δ') / (δ' * γ)
    # print("\n x′",x′)

    return x′, α
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
function optimize(f, g, x0, n, prob)
    # TODO keep track of n 
    # using a problem agnostic BFGS method
    # starting matrix to be initialized 
    undef_m = BFGS(undef)
    init_m = init!(undef_m, x0)

    # take steps w BFGS while we are still improving + less than n steps taken 
    num_evals = count(f, g)
    print("\n num_evals start $num_evals")
    iterations = 1
    α = 0.1
    x_ints = []

    while num_evals < n - 5
        x_int, α = step!(init_m, f, g, x0, 3, iterations, α, n)
        if x_int == false
            push!(x_ints, x0)
            break
        end
        push!(x_ints, x_int)
        num_evals = count(f, g)
        print("\n num_evals $num_evals, n $n, iterations $iterations \n")
        iterations = iterations + 1
    end

    print("\n num_evals end $num_evals \n")
    x_best = last(x_ints) #x_int

    # plot_opt(x_ints, prob)

    return x_best, x_ints
end

main("simple2", 5, optimize)

