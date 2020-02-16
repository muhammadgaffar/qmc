using Test
using BenchmarkTools
using Plots

include("gf.jl")
include("transform.jl")
include("util.jl")

# TEST GREEN FUNCTION TRANSFORMATION
G0w = setGw(nw=501,wrange=(-2,2),name="nonInteracting")
setfromToy!(G0w,:bethe)
pgw = plot(G0w.mesh, real(G0w.data),label="real")
plot!(pgw,G0w.mesh, imag(G0w.data), label="imag")

Giwn = setGiwn(256,20,name="nonInteracting")
setfromToy!(Giwn,:bethe)
pgiw = plot(Giwn.mesh,imag(Giwn.data))

Gtau = setGtau(256,20,name="nonInteracting")
setfromToy!(Gtau,:bethe)
pgtau = plot(Gtau.mesh,real(Gtau.data))

Gtau2 = invFourier(Giwn)
plot!(pgtau, Gtau2.mesh,real(Gtau2.data))

Giwn2 = Fourier(Gtau)
plot!(pgiw,Giwn.mesh,imag(Giwn.data))

G0w2 = setfromPade(Giwn, nw=501,wrange=(-2,2),npoints=100)
plot!(pgw,G0w2.mesh, imag(G0w2.data))
plot!(pgw,G0w2.mesh, real(G0w2.data))

# TEST GREEN FUNCTION OPERATOR
@testset "operator" begin
    G = G0w
    g = copy(G)
    @test g.data == G.data
    @test g.mesh == G.mesh

    g = (2G + G) / 5G
    @test g.data == (2*G.data + G.data) ./ (5*G.data)

    g = (iwn + Giwn) / (Giwn - iwn)
    @test g.data == (Giwn.mesh .+ Giwn.data) ./ (Giwn.data - Giwn.mesh)

    gg = inv(g)
    @test gg.data == inv.(g.data)
end;
