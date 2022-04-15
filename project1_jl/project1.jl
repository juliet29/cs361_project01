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
# using PlotlyJS
# using Kaleido
using Plots
# using PyPlot
# pyplot()
#  Kaleido

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

"cs361_week01/AA222Project1/project1_jl/project1.jl"

"""
BFGS works well for most problems 
"""

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
    
    Q, g = M.Q, ∇f(x)
    x′ = line_search_fix_alpha(f, x, -Q * g, ∇f, alpha_searches, line_search_lim, avail_evals)
    # returning from line search with too many evals 
    if x′ == false
        return false
    end
    # check for too many evals before proceeding 
    num_evals = count(f,∇f)
    if num_evals > avail_evals - step_lim
        return false
    end

    g′ = ∇f(x′)
    δ = x′ - x
    γ = g′ - g

    Q[:] = Q - (δ * γ' * Q + Q * γ * δ') / (δ' * γ) + (1+(γ'*Q*γ)/(δ'*γ))[1] * (δ * δ') / (δ' * γ)

    return x′
end


function BFGS_opt(f, g, x0, avail_evals, probname, alpha_searches, step_lim, line_search_lim)
    # using a problem agnostic BFGS method
    # starting matrix to be initialized 
    undef_m = BFGS(undef)
    init_m = init!(undef_m, x0)

    # take steps w BFGS while we less than avail_evals steps taken 
    num_evals = count(f, g)
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
        iterations = iterations + 1
    end

    x_best = first(x_ints) #x_int
    # println(x_best)

    # when call my main for plotting 
    # return x_best, x_ints
    return x_best
    
end

"""
Going to try grad descent for others 
"""
abstract type DescentMethod end
struct GradientDescent <: DescentMethod 
    α
end

function stepGrad!(M::GradientDescent, f, ∇f, x)
    
    α, g = M.α, ∇f(x)
    g_to_norm = g[1]^2 + g[2]^2
    g_norm = g/g_to_norm
    # println("g $g")
    # println("g_to_norm $g_to_norm")
    # println("g_norm $g_norm")
    return x - α*g_norm
end

mutable struct NoisyDescent <: DescentMethod
    submethod 
    σ
    k
end

function initNoisy!(M::NoisyDescent)
    M.k = 1
end

function stepNoisy!(M::NoisyDescent, G::GradientDescent, f,∇f, x)
    x = stepGrad!(G, f,∇f, x)
    # σ = 0
    σ = M.σ(M.k)
    x += σ.*randn(length(x))
    M.k +=1

    # println("k $(M.k) σ $σ")
    return x
end

function make_sigma(k)
    return 1/k
end


function noisy_opt(f,∇f, x, probname, avail_evals, conv_plot=undef, cont_plot=undef)
    grad_struct = GradientDescent(20)
    noisy_struct = NoisyDescent(GradientDescent, make_sigma, 10)
    # init_struct = initNoisy!(mut_struct)
    int_x = []
    int_f = Float64[] 
    while count(f, ∇f) < avail_evals - 5
        x = stepNoisy!(noisy_struct, grad_struct, f, ∇f, x)
        f_eval = f(x)
        if isnan(f_eval) == false && isinf(f_eval) == false
            push!(int_x, x)
            push!(int_f, f_eval)
        end
    end

    steps = [i for (i, v) in enumerate(int_x)]
    println("results $steps, $int_f")


    x_best = int_x[argmin(int_f)]
    f_best = minimum(int_f)
    println("best_f_index $(argmin(int_f)) -- best_x $x_best -- best $f_best")   

    plot_contour(int_x, x_best, probname, cont_plot)
    plot_convergence(int_x, int_f, probname, conv_plot)

    return x_best
end


function plot_convergence(int_x, int_f, probname, c_plot)
        x = [index for (index, value) in enumerate(int_x)]
        plot!(c_plot, x, int_f, markershape = :circle, linestyle = :solid,title=probname)

        fname = "figures_converg/$probname"
        savefig(c_plot, fname)
    
end

function plot_contour(int_x, x_best, probname, c_plot)
    prob = PROBS[probname]
    # make contour trace
    # contour values
    prob_range = range(-5.0, 5.0, 30)
    scale_factor = 1
    eval_m = zeros(Float32, length(prob_range), 2)
    eval_m[:,1] = scale_factor*prob_range
    eval_m[:,2] = scale_factor*prob_range
    x=eval_m[:,1]
    y =eval_m[:,2]
    z = [prob.f([x1, x2]) for x1=eval_m[:,1], x2=eval_m[:,2] ]

    contour!(c_plot, x, y, z, fill = false, title=probname, dpi=300, size=(700, 300), contourlabels=false, levels=30)

    # function performance 
    x1 = [row[1] for row in int_x]
    x2 = [row[2] for row in int_x]
    
    plot!(c_plot, x1, x2, markershape = :circle, linestyle = :solid, label="Intermediate")
    # indicate starting point
    plot!(c_plot, [x1[1]], [x2[1]], markershape = :circle, linestyle = :solid, markercolor = :blue, label="Start")
    # indicate end
    plot!(c_plot, [last(x1)], [last(x2)], markershape = :circle, linestyle = :solid, markercolor = :yellow, label="End")
    # indicate best
    plot!(c_plot, [x_best[1]], [x_best[2]], markershape = :circle, linestyle = :solid, markercolor = :red, label="Best")

    fname = "figures_contour/$probname"
    savefig(c_plot, fname)
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
function optimize(f, g, x0, avail_evals, probname, conv_plot=undef, cont_plot=undef)

    if probname == "simple1"
        alpha_searches = 12
        step_lim = 2
        line_search_lim = 9
        return BFGS_opt(f, g, x0, avail_evals, probname, alpha_searches, step_lim, line_search_lim)

    elseif probname == "simple2"
        alpha_searches = 10
        step_lim = 5
        line_search_lim = 20
        # return noisy_opt(f, g, x0, probname, avail_evals)
        return noisy_opt(f, g, x0, probname, avail_evals, conv_plot, cont_plot)

    elseif probname  == "simple3"
        alpha_searches = 10
        step_lim = 5
        line_search_lim = 20
        return BFGS_opt(f, g, x0, avail_evals, probname, alpha_searches, step_lim, line_search_lim)

    else
        alpha_searches = 10
        step_lim = 5
        line_search_lim = 20
        return BFGS_opt(f, g, x0, avail_evals, probname, alpha_searches, step_lim, line_search_lim)
    end

end

mymain("simple2", 2, optimize)

# mymain("simple3", 10, optimize)

