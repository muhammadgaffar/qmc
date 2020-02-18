using Test
using BenchmarkTools
using Plots

include("gf.jl")
include("transform.jl")
include("util.jl")

# TEST GREEN FUNCTION ALLOCATION
@testset "GF Allocation" begin
    gw = setGw((1,2,3), 501,(-2,2))
    @test size(gw.data) == (3,3,501)
    giw = setGiwn(("e2g","t2g"), 2^8,50)
    @test size(giw.data) == (2,2,2*2^8+1)
    gt = setGtau(("s", 2),2487,50)
    @test size(gt.data) == (2,2,2487)
end;

# TEST GREEN FUNCTION OPERATOR
@testset "Operator" begin
    gf = setGtau(("e2g","t2g"),2^8,50)
    gf.data = rand(ComplexF64,size(gf.data)...)
    gf2 = setGtau(("e2g","t2g"),2^8,50)
    gf2.data = rand(ComplexF64,size(gf.data)...)

    gf2 = copy(gf)
    @test size(gf2.data) == size(gf.data)
    @test size(gf2.mesh) == size(gf.mesh)

    g = gf + gf2
    @test g.data == gf.data .+ gf2.data

    g = -gf2
    @test g.data == -gf2.data

    g = gf - gf2
    @test g.data == gf.data .- gf2.data

    g = gf * gf2
    @test g.data[:,:,174] == gf.data[:,:,174] * gf2.data[:,:,174]

    g = (2.0 + 3im) * gf
    g2 = gf * (2.0 + 3im)
    @test g.data == (2.0 + 3im) .* gf.data
    @test g.data == g2.data

    v = rand(length(gf.mesh))
    g = v * gf
    g2 = gf * v
    @test g.data[:,:,54] == v[54] .* gf.data[:,:,54]
    @test g.data == g2.data

    g = gf / gf2
    @test g.data[:,:,174] ≈ gf.data[:,:,174] / gf2.data[:,:,174]

    g = 89/gf
    @test g.data[:,:,98] == 89 .* inv( gf.data[:,:,98] )

    g = gf2 / 24
    @test g.data[:,:,148] == gf.data[:,:,148] ./ 24

    gf = setGiwn(("s","p"), 128, 40)
    gf.data = rand(ComplexF64,size(gf.data)...)
    g = iwn + 5gf
    @test g.data[:,:,2] == 1im*gf.mesh[2]*Matrix(I,2,2) + 5 * gf.data[:,:,2]
    g =  8 / gf - iwn
    @test g.data[:,:,87] == 8 .* inv(gf.data[:,:,87]) - 1im*gf.mesh[87]*Matrix(I,2,2)

    g = inv(gf)
    @test g.data[:,:,111] == inv(gf.data[:,:,111])
end;


# TEST GREEN FUNCTION TRANSFORMATION
# CONFIRMATION WITH PLOT
G0w = setGw(("up","dw"),501,(-2,2))
setfromToy!(G0w,:bethe)
re_pgw = plot(G0w.mesh, real(G0w.data[1,1,:]),label="real_up")
plot!(re_pgw,G0w.mesh, real(G0w.data[2,2,:]),label="real_dw")
im_pgw = plot(G0w.mesh, imag(G0w.data[1,1,:]),label="imag_up")
plot!(im_pgw,G0w.mesh, imag(G0w.data[2,2,:]), label="imag_dw")

Giwn = setGiwn(("up","dw"),256,20.0)
setfromToy!(Giwn,:bethe)
pgiw = plot(Giwn.mesh,imag(Giwn.data[1,1,:]),label="imag_up", xlims=(-10,10))
plot!(pgiw,Giwn.mesh,imag(Giwn.data[2,2,:]),label="imag_dw", xlims=(-10,10))

Gtau = setGtau(("s","p"),256,20)
setfromToy!(Gtau,:bethe)
pgtau = plot(Gtau.mesh,real(Gtau.data[1,1,:]),label="up")
plot!(pgtau,Gtau.mesh,real(Gtau.data[2,2,:]),label="dw")

Gtau2 = invFourier(Giwn)
plot!(pgtau, Gtau2.mesh,real(Gtau2.data[1,1,:]),label="invFourier_up")

Giwn2 = Fourier(Gtau)
plot!(pgiw,Giwn2.mesh,imag(Giwn2.data[1,1,:]),label="Fourier_up")

G0w2 = setfromPade(Giwn, nw=501,wrange=(-2,2),npoints=100)
plot!(im_pgw,G0w2.mesh, imag(G0w2.data[1,1,:]))
plot!(re_pgw,G0w2.mesh, real(G0w2.data[2,2,:]))
