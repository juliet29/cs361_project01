# using Statistics
# using Plots
# using Symbolics
using LinearAlgebra
using Plots


function bracket_minimum(f,g_for_count, x=0, s=1e-2, k=2)
    """Used for univariate problems"""
    num_evals = count(f, g_for_count)
    print("\n bracket_min START function calls $num_evals")
    a, ya = x, f(x)
    b, yb = a .+ s, f(a .+ s)
    # print("\n yb ", yb, " ya ", ya)
    if yb > ya
        # switch b to be moving closer to the minimum
        a, b = b, a
        ya, yb = yb, ya
        # switch direction we are moving in 
        s = -s
    end
    while true
        # moving b along, taking steps past b, and calling it c

        c, yc = b + s, f(b + s)
        if yc > yb
            # yc should be moving into the valley, so if its > b, probably going up a hill.. switch a and c
            num_evals = count(f, g_for_count)
            print("\n bracket_min function calls $num_evals")
            return a < c ? (a, c) : (c, a)
        end
        # move all the vals over, a ->b, b -> c 
        a, ya, b, yb = b, yb, c, yc
        # double the step size
        s *= k
    end

end

function golden_section_search(f, a, b, n, g_for_count)
    "minimize univariate function, given an a bracket interval defined by a and b within n iterations"
    num_evals = count(f, g_for_count)
    print("\n gold_sect START function calls $num_evals")
    φ = MathConstants.golden
    ρ = φ - 1
    d = ρ * b + (1 - ρ) * a
    yd = f(d)
    for i = 1:n-1
        c = ρ * a + (1 - ρ) * b
        yc = f(c)
        # print("\n gold section, $i")
        if yc < yd
            b, d, yd = d, c, yc
        else
            a, b = b, c
        end
    end
    num_evals = count(f, g_for_count)
    print("\n gold_sect function calls $num_evals")
    return a < b ? (a, b) : (b, a)
end

function interval_middle(a, b)
    "get the value at the center of an interval"
    return (a + b) / 2
end

function minimize(f, a, b, n, g_for_count=0)
    "golden section search returning center of an interval"
    low, high = golden_section_search(f, a, b, n, g_for_count)
    return interval_middle(low, high)
end


function line_search_fix_alpha(f, x, d, g_for_count, n=5, iterations=1, α=0.1)
    num_evals = count(f, g_for_count)
    print("\n line_search START function calls $num_evals")
    objective = α -> f(x + α * d)
    # print(f(x), ' ', x, ' ', d, '\n')
    # ccall(:jl_exit, Cvoid, (Int32,), 86)
    a, b = bracket_minimum(objective, g_for_count)

    α = minimize(objective, a, b, n, g_for_count)


    num_evals = count(f, g_for_count)
    print("\n line_search function calls $num_evals")
    # print(x + α * d, ' ',  x, ' ', α, ' ', d)

    return x + α * d, α
end


function plot_opt(array_x, probname)
    prob = PROBS[probname]
    y = [prob.f(x) for x in array_x]
    x = [index for (index, value) in enumerate(array_x)]
    println(x, y)
    # plot(array_x, y)
    plot(x, y, title=probname)
    savefig("figures/$probname.png") 
end






# function line_search(f, x, d, n=12)
#     objective = α -> f(x + α * d)
#     # print(f(x), ' ', x, ' ', d, '\n')
#     # ccall(:jl_exit, Cvoid, (Int32,), 86)
#     a, b = bracket_minimum(objective)
#     # print('\n', a, ' ', b)
#     α = minimize(objective, a, b, n)
#     # print(x + α * d, ' ',  x, ' ', α, ' ', d)
#     # print("\n α, $α")
#     return x + α * d
# end