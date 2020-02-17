using Test
using BenchmarkTools
using Plots

include("gf.jl")
include("transform.jl")
include("util.jl")

# TEST GREEN FUNCTION ALLOCATION
@testset "GfreFreq Allocation" begin
    gw = setGw((1,2,3),501,(-2,2),name="nonInteracting")
    @test eltype(gw.mesh) == Float64
    @test size(gw.data) == (3,3,501)
    gw2 = setGw(orbs=(1,2,3),nw=501,wrange=(-2,2),name="nonInteracting")
    @test gw2.mesh == gw.mesh
    @test gw2.data == gw.data
    println("Checking memory and time allocation :")
    @time gw = setGw((1,2,3),501,(-2,2),name="nonInteracting")
    println("Checking show output : ")
    @show gw
end;

@testset "GfimFreq Allocation" begin
    gf = setGiwn(("e2g","t2g"),2^8,50,name="nonInteracting")
    @test eltype(gf.mesh) == Float64
    @test size(gf.data) == (2,2,2*2^8+1)
    gf2 = setGiwn(orbs=("e2g","t2g"),n=2^8,beta=50,name="nonInteracting")
    @test gf2.mesh == gf.mesh
    @test gf2.data == gf.data
    println("Checking memory and time allocation :")
    @time gf = setGiwn(("e2g","t2g"),2^8,50,name="nonInteracting")
    println("Checking show output : ")
    @show gf
end;

@testset "GfimTime Allocation" begin
    gf = setGtau(("e2g","t2g"),2^8,50,name="nonInteracting")
    @test eltype(gf.mesh) == Float64
    @test size(gf.data) == (2,2,2^8)
    gf2 = setGtau(orbs=("e2g","t2g"),nslices=2^8,beta=50,name="nonInteracting")
    @test gf2.mesh == gf.mesh
    @test gf2.data == gf.data
    println("Checking memory and time allocation :")
    @time gf = setGtau(("e2g","t2g"),2^8,50,name="nonInteracting")
    println("Checking show output : ")
    @show gf
end;

# TEST GREEN FUNCTION OPERATOR
@testset "Operator" begin
    gf = setGtau(("e2g","t2g"),2^8,50,name="gtest1")
    gf.data = rand(ComplexF64,size(gf.data)...)
    gf2 = setGtau(("e2g","t2g"),2^8,50,name="gtest2")
    gf2.data = rand(ComplexF64,size(gf.data)...)

    gf2 = copy(gf)
    @test size(gf2.data) == size(gf.data)
    @test size(gf2.mesh) == size(gf.mesh)
    println("Checking memory and time allocation : copy")
    @time gf2 = copy(gf)

    g = gf + gf2
    @test g.data == gf.data .+ gf2.data
    println("Checking memory and time allocation : G1+G2")
    @time gf + gf2

    g = -gf2
    @test g.data == -gf2.data
    println("Checking memory and time allocation : -G1")
    @time -gf2

    g = gf - gf2
    @test g.data == gf.data .- gf2.data
    println("Checking memory and time allocation : G1-G2")
    @time gf - gf2

    g = gf * gf2
    @test g.data[:,:,174] == gf.data[:,:,174] * gf2.data[:,:,174]
    println("Checking memory and time allocation : G1*G2")
    @time gf * gf2

    g = (2.0 + 3im) * gf
    g2 = gf * (2.0 + 3im)
    @test g.data == (2.0 + 3im) .* gf.data
    @test g.data == g2.data
    println("Checking memory and time allocation : aG")
    @time (2.0 + 3im) * gf

    v = rand(length(gf.mesh))
    g = v * gf
    g2 = gf * v
    @test g.data[:,:,54] == v[54] .* gf.data[:,:,54]
    @test g.data == g2.data
    println("Checking memory and time allocation : vG")
    @time v * gf

    g = gf / gf2
    @test g.data[:,:,174] ≈ gf.data[:,:,174] / gf2.data[:,:,174]
    println("Checking memory and time allocation : G1/G2")
    @time gf / gf2

    g = 89/gf
    @test g.data[:,:,98] == 89 .* inv( gf.data[:,:,98] )
    println("Checking memory and time allocation : a/G")
    @time 89 / gf

    g = gf2 / 24
    @test g.data[:,:,148] == gf.data[:,:,148] ./ 24
    println("Checking memory and time allocation : G1/a")
    @time gf2 / 24

    gf = setGiwn(("s","p"), 128, 40, name="GF")
    gf.data = rand(ComplexF64,size(gf.data)...)
    g = iwn + 5gf
    @test g.data[:,:,2] == 1im*gf.mesh[2]*Matrix(I,2,2) + 5 * gf.data[:,:,2]
    g =  8 / gf - iwn
    @test g.data[:,:,87] == 8 .* inv(gf.data[:,:,87]) - 1im*gf.mesh[87]*Matrix(I,2,2)
    println("Checking memory and time allocation : iwn +- G")
    @time iwn + gf

    g = inv(gf)
    @test g.data[:,:,111] == inv(gf.data[:,:,111])
    println("Checking memory and time allocation : G^{-1}")
    @time g = inv(gf)
end;


# TEST GREEN FUNCTION TRANSFORMATION
# CONFIRMATION WITH PLOT
G0w = setGw(("up","dw"),501,(-2,2),name="nonInteracting")
setfromToy!(G0w,:bethe)
re_pgw = plot(G0w.mesh, real(G0w.data[1,1,:]),label="real_up")
plot!(re_pgw,G0w.mesh, real(G0w.data[2,2,:]),label="real_dw")
im_pgw = plot(G0w.mesh, imag(G0w.data[1,1,:]),label="imag_up")
plot!(im_pgw,G0w.mesh, imag(G0w.data[2,2,:]), label="imag_dw")

Giwn = setGiwn(("up","dw"),256,20.0,name="nonInteracting")
setfromToy!(Giwn,:bethe)
pgiw = plot(Giwn.mesh,imag(Giwn.data[1,1,:]),label="imag_up", xlims=(-10,10))
plot!(pgiw,Giwn.mesh,imag(Giwn.data[2,2,:]),label="imag_dw", xlims=(-10,10))

Gtau = setGtau(("s","p"),256,20,name="nonInteracting")
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
