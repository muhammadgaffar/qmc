# GREEN FUNCTION MODULES
using LinearAlgebra
using BenchmarkTools
using Dierckx

import Base: +, -, *, /
import Base: copy, inv, convert

## CONSTRUCT BASIS OF GREEN FUNCTION ===============================||

abstract type GreenFunction end
abstract type Descriptor end

struct Omega <: Descriptor end
const omega = Omega()
const om    = Omega()
Base.show(io::IO,D::Omega) = print(io,"ω")

struct iOmega_n <: Descriptor end
const iomega_n = iOmega_n()
const iwn      = iOmega_n()
Base.show(io::IO,D::iOmega_n) = print(io,"iωₙ")

mutable struct GfrealFreq{T} <: GreenFunction
    name::String
    mesh::Vector{T}
    data::Vector{Complex{T}}
end

mutable struct GfimFreq{T} <: GreenFunction
    name::String
    mesh::Vector{T}
    data::Vector{Complex{T}}
    beta::T
end

mutable struct GfimTime{T} <: GreenFunction
    name::String
    mesh::Vector{T}
    data::Vector{Complex{T}}
    beta::T
end

## for Gw
function setGw(nw::Int,wrange::T;name::String,precision=Float64) where T <: Tuple{Number,Number}
    w = LinRange(wrange...,nw)
    initzeros = zeros(length(w))
    return GfrealFreq{precision}(name,w,initzeros)
end
setGw(;nw,wrange,name,precision=Float64) = setGw(nw,wrange,name=name,precision=precision)

Base.show(io::IO,gf::GfrealFreq{T}) where T = print(io,
"GfRealFreq{$T} : $(gf.name)
    mesh    : $(length(gf.mesh))
    range   : ($(gf.mesh[1]),$(gf.mesh[end]))")

## for Giwn
function setGiwn(n::Int,beta::T;name::String,precision=Float64) where T <: Number
    wn = (2*collect(-n:n) .+ 1) * π / beta
    initzeros = zeros(2n+1)
    return GfimFreq{precision}(name,wn,initzeros,beta)
end
setGiwn(;n,wrange,name,beta,precision=Float64) = setGiwn(n,beta,name=name,precision=precision)

Base.show(io::IO,gf::GfimFreq{T}) where T = print(io,
"GfimFreq{$T} : $(gf.name)
    nwn     : $(Int(0.5*(length(gf.mesh)-1)))
    beta    : $(gf.beta)")

## for Gtau
function setGtau(nslices::Int,beta::T;name::String,precision=Float64) where T <: Number
    tau = LinRange(0,beta,nslices)
    initzeros = zeros(nslices)
    return GfimTime{precision}(name,tau,initzeros,beta)
end
setGtau(;nslices,beta,name,precision=Float64) = setGtau(nslices,beta,name=name,precision=precision)

Base.show(io::IO,gf::GfimTime{T}) where T = print(io,
"GfimTime{$T} : $(gf.name)
    L slices: $(length(gf.mesh))
    beta    : $(gf.beta)")

## WHAT CAN WE DO WITH GREEN FUNCTION ===============================||

function copy(G::GreenFunction)
    g = (typeof(G) <: GfrealFreq ? typeof(G)("$(G.name)",G.mesh,G.data) : typeof(G)("$(G.name)",G.mesh,G.data,G.beta))
    return g
end

function +(G1::GreenFunction,G2::GreenFunction)
    if typeof(G1) != typeof(G2)
        throw("you cannot add different type green function")
    end
    g = copy(G1)
    g.data = G1.data .+ G2.data
    g.name = "GreenFunction"
    return g
end

function -(G::GreenFunction)
    g = copy(G)
    g.data = -G.data
    return g
end

function -(G1::GreenFunction,G2::GreenFunction)
    if typeof(G1) != typeof(G2)
        throw("you cannot substract different type green function")
    end
    g = copy(G1)
    g.data = G1.data .- G2.data
    g.name = "GreenFunction"
    return g
end

function *(G1::GreenFunction,G2::GreenFunction)
    if typeof(G1) != typeof(G2)
        throw("you cannot multiply different type green function")
    end
    g = copy(G1)
    g.data = G1.data .* G2.data
    g.name = "GreenFunction"
    return g
end

function *(a::Number,G::GreenFunction)
    g = copy(G)
    g.data = a .* G.data
    g.name = "GreenFunction"
    return g
end
*(G::GreenFunction,a::Number) = *(a::Number,G::GreenFunction)

function *(a::AbstractVector,G::GreenFunction)
    @assert length(a) == length(G.data) "DimensionMisMatch"
    g = copy(G)
    g.data = a .* G.data
    g.name = "GreenFunction"
    return g
end
*(G::GreenFunction,a::AbstractVector) = *(a::AbstractVector,G::GreenFunction)

function /(G1::GreenFunction,G2::GreenFunction)
    if typeof(G1) != typeof(G2)
        throw("you cannot multiply different type green function")
    end
    g = copy(G1)
    g.data = G1.data ./ G2.data
    g.name = "GreenFunction"
    return g
end

function /(a::Number,G::GreenFunction)
    g = copy(G)
    g.data = a ./ G.data
    g.name = "GreenFunction"
    return g
end
function /(G::GreenFunction,a::Number)
    g = copy(G)
    g.data = G.data ./ a
    g.name = "GreenFunction"
    return g
end

function +(iwn::iOmega_n,G::GfimFreq)
    g = copy(G)
    g.data = G.mesh + G.data
    return g
end
+(G::GfimFreq,iwn::iOmega_n) = +(iwn::iOmega_n,G::GfimFreq)

function -(iwn::iOmega_n,G::GfimFreq)
    g = copy(G)
    g.data = G.mesh .- G.data
    return g
end
-(G::GfimFreq,iwn::iOmega_n) = -(iwn - G)

function inv(G::GreenFunction)
    g = copy(G)
    g.data = inv.(G.data)
    g.name = "GreenFunction"
    return g
end

### for non-interacting cases
function setfromToy!(G0::GreenFunction,toy::Symbol)
    check = typeof(G0) # check green function type
    if (check <: GfrealFreq) freq = G0.mesh .+ 0.01im end
    if (check <: GfimFreq) freq = 1im*G0.mesh end

    # get A0w from toy model
    wmesh = 501 # i think this is enough for integration purpose, maybe.
    if toy == :bethe
        wmesh = LinRange(-1.01,1.01,wmesh) #for default t =0.5 the band has value in w ∈ [-1,1]
        A0w = A0bethe.(wmesh)
    else
        @error "Not yet implemented, try :bethe" # help me.
    end

    # now do hilbert transform.
    if check <: GfimFreq || check <: GfrealFreq
        for iw in 1:length(freq)
            div = (A0w ./ (freq[iw] .- wmesh))
            G0.data[iw] = integrate1d(wmesh,div)
        end
    elseif check <: GfimTime
        for t in 1:length(G0.mesh)
            fm = 1.0 ./ (exp.(G0.beta .* wmesh) .+ 1.0 )
            intg = A0w .* (fm .- 1.0) .* exp.(-wmesh .* G0.mesh[t])
            G0.data[t] = integrate1d(wmesh,intg)
        end
    end
    return G0
end

### list of available toy model
A0bethe(w; t=0.5) = (abs(w) < 2t ? sqrt(4t^2 - w^2) / (2π*t^2) : 0)
#A0cubic(w,t) >> help me

## GREEN FUNCTION TRANSFORMATION ===============================||

## G(τ) = invFourier(G(iωₙ))
function invFourier(Giwn::GfimFreq)
    # data and parameter
    nwn  = Int(0.5*(length(Giwn.mesh)-1))
    beta = Giwn.beta

    # tail coeff, ntail = 128 is enough, maybe.
    coeff = tail_coeff(Giwn.mesh[nwn+1:end],Giwn.data[nwn+1:end], 128)

    # inverse Fourier, produce 2n+1 data (becuase -n:n data)
    tau,gtau = fwn_to_ftau(Giwn.data,Giwn.mesh,beta, coeff)

    #spline for n data only..
    spl = Spline1D(tau,real(gtau))
    tau = LinRange(0,beta,nwn)
    gtau = spl.(tau)

    return GfimTime{eltype(Giwn.mesh)}(Giwn.name,tau,gtau,beta)
end

### G(iωₙ) = Fourier(G(τ))
function Fourier(Gtau::GfimTime)
    # parameter
    ntau = length(Gtau.mesh)

    # get G(iωₙ), not using fft.
    wn,gwn = ftau_to_fwn(Gtau.data,Gtau.mesh,Gtau.beta)
    return GfimFreq{eltype(Gtau.mesh)}(Gtau.name,wn,gwn,Gtau.beta)
end

## G(ω) = Analytical Continuation G(iωₙ), using Pade.
function setfromPade(Giwn::GfimFreq; nw::Int, wrange::T,
                npoints=nothing, broadening=0.01) where T <: Tuple{Number,Number}

    ## only use non-negative n for pade recursion
    nwn = Int((length(Giwn.mesh) - 1) / 2)
    wn  = 1im.*Giwn.mesh[length(Giwn.data)-nwn+1:end]
    gwn = Giwn.data[length(Giwn.data)-nwn+1:end]
    # initialization
    w = LinRange(wrange...,nw) .+ 1im.*broadening
    w = eltype(wn).(w)

    # matsubara points to sample for pade recursion,
    # if too large, or precision too low there are numerical error, NAN.
    # warn my precision
    if eltype(Giwn.mesh) == Float32
        @warn "Your precision is Float32, too low. use Float64, or BigFloat for accurate result."
    end
    npoints = (npoints == nothing ? length(wn) : npoints)

    # get pade coeffision
    coeff = pade_coeff(wn,gwn)

    # pade recursion approximation
    w, gw = pade_recursion(w, wn,gwn,npoints,coeff)

    return GfrealFreq{eltype(Giwn.mesh)}(Giwn.name,real(w),gw)
end
