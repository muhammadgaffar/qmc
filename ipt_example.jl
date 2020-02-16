include("gf.jl")

using Plots

# iterated pertubation theory impurity solver
# easiest one
function solve_IPT(g,U,beta,iter=25)
    for i in 1:iter
        g0 = inv(iwn - 0.5^2 * g)
        g0t = invFourier(g0)
        sigma_t = U^2 * g0t * g0t * g0t
        sigma_iw = Fourier(sigma_t)
        g = inv(inv(g0) - sigma_iw)
    end
    return g
end

# example
n = 2^10; beta = 100
g0 = setGiwn(n,beta,name="GF")
setfromToy!(g0,:bethe)

gw  = setfromPade(g0, nw=501,wrange=(-3,3),npoints=200)
p = plot(gw.mesh,-imag(gw.data), framestyle=:box, title="dos ipt for various U", label="U = 0 eV")

#phase transition at U = 1.8,
for U in [1.0, 1.5, 1.8, 3.2]
    @info "For U = $U eV"
    if U == 1.8
        g_int = solve_IPT(g0, U,beta, 200)
    else
        g_int = solve_IPT(g0, U,beta)
    end
    gw  = setfromPade(g_int, nw=501,wrange=(-3,3),npoints=200)
    plot!(p, gw.mesh,-imag(gw.data), label="U = $U eV")
end

xlims!(p, (-3,3))
xlabel!(p, "\\omega")
ylabel!(p, "\\rho (\\omega)")
savefig(p, "ipt_example_result.pdf")
