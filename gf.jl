# GREEN FUNCTION MODULES
using LinearAlgebra
using BenchmarkTools
using Dierckx

import Base: +, -, *, /
import Base: copy, inv

## CONSTRUCT BASIS OF GREEN FUNCTION ===============================||

abstract type GreenFunction end
abstract type Descriptor end

struct Omega <: Descriptor end
omega = Omega()
om    = Omega()
Base.show(io::IO,D::Omega) = print(io,"ω")

struct iOmega_n <: Descriptor end
iomega_n = iOmega_n()
iwn      = iOmega_n()
Base.show(io::IO,D::iOmega_n) = print(io,"iωₙ")

mutable struct GfrealFreq{T} <: GreenFunction
    name::String
    w   ::Vector{T}
    data::Vector{Complex{T}}
end

mutable struct GfimFreq{T} <: GreenFunction
    name::String
    wn  ::Vector{T}
    data::Vector{Complex{T}}
    beta::T
end

mutable struct GfimTime{T} <: GreenFunction
    name::String
    tau ::Vector{T}
    data::Vector{Complex{T}}
    beta::T
end

## for Gw
function setGw(nw::Int,wrange::T;name::String,precision=Float32) where T <: Tuple{Number,Number}
    w = LinRange(wrange...,nw)
    initzeros = zeros(length(w))
    return GfrealFreq{precision}(name,w,initzeros)
end
setGw(;nw,wrange,name,precision=Float32) = setGw(nw,wrange,name=name,precision=precision)

Base.show(io::IO,gf::GfrealFreq{T}) where T = print(io,
"GfRealFreq{$T} : $(string(gf.name))
    mesh    : $(length(gf.w))
    range   : ($(gf.w[1]),$(gf.w[end]))")

## for Giwn
function setGiwn(n::Int,beta::T;name::String,precision=Float32) where T <: Number
    wn = (2*collect(-n:n) .+ 1) * π / beta
    initzeros = zeros(2n+1)
    return GfimFreq{precision}(name,wn,initzeros,beta)
end
setGiwn(;n,wrange,name,beta,precision=Float32) = setGiwn(n,beta,name=name,precision=precision)

Base.show(io::IO,gf::GfimFreq{T}) where T = print(io,
"GfimFreq{$T} : $(string(gf.name))
    nwn     : $(Int(0.5*(length(gf.wn)-1)))
    beta    : $(gf.beta)")

## for Gtau
function setGtau(nslices::Int,beta::T;name::String,precision=Float32) where T <: Number
    tau = LinRange(0,beta,nslices)
    initzeros = zeros(nslices)
    return GfimTime{precision}(name,tau,initzeros,beta)
end
setGtau(;nslices,beta,name,precision=Float32) = setGtau(nslices,beta,name=name,precision=precision)

Base.show(io::IO,gf::GfimTime{T}) where T = print(io,
"GfimTime{$T} : $(string(gf.name))
    L slices: $(length(gf.tau))
    beta    : $(gf.beta)")

## WHAT CAN WE DO WITH GREEN FUNCTION ===============================||

function copy(G::GreenFunction)
    data = copy(G.data)
    if typeof(G) <: GfimFreq mesh = copy(G.wn) end
    if typeof(G) <: GfrealFreq mesh = copy(G.w) end
    if typeof(G) <: GfimTime mesh = copy(G.tau) end

    if typeof(G) <: GfrealFreq g = typeof(G)("GF",mesh,data) end
    if typeof(G) <: GfimFreq || typeof(G) <: GfimTime
        beta = G.beta
        g = typeof(G)("GF",mesh,data,beta)
    end
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

function /(a::AbstractVector,G::GreenFunction)
    @assert length(a) == length(G.data) "DimensionMisMatch"
    g = copy(G)
    g.data = a ./ G.data
    g.name = "GreenFunction"
    return g
end
function /(G::GreenFunction,a::AbstractVector)
    @assert length(a) == length(G.data) "DimensionMisMatch"
    g = copy(G)
    g.data = G.data ./ a
    g.name = "GreenFunction"
    return g
end

function +(iwn::iOmega_n,G::GfimFreq)
    G.data = G.data .+ G.wn
    return G
end
+(G::GfimFreq,iwn::iOmega_n) = +(iwn::iOmega_n,G::GfimFreq)

function -(iwn::iOmega_n,G::GfimFreq)
    G.data = G.wn .- G.data
    return G
end
function -(G::GfimFreq,iwn::iOmega_n)
    G.data = G.data .- G.wn
    return G
end

function inv(G::GreenFunction)
    g = copy(G)
    g.data = inv.(G.data)
    g.name = "GreenFunction"
    return g
end

### for non-interacting cases
function setfromToy!(G0::GreenFunction,toy::Symbol)
    check = typeof(G0) # check green function type
    if (check <: GfrealFreq) freq = G0.w .+ 0.01im end
    if (check <: GfimFreq) freq = 1im*G0.wn end

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
        for t in 1:length(G0.tau)
            fm = 1.0 ./ (exp.(G0.beta .* wmesh) .+ 1.0 )
            intg = A0w .* (fm .- 1.0) .* exp.(-wmesh .* G0.tau[t])
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
    nwn  = Int(0.5*(length(Giwn.wn)-1))
    beta = Giwn.beta

    # tail coeff, ntail = 128 is enough, maybe.
    coeff = tail_coeff(Giwn.wn[nwn+1:end],Giwn.data[nwn+1:end], 128)

    # inverse Fourier, produce 2n+1 data (becuase -n:n data)
    tau,gtau = fwn_to_ftau(Giwn.data,Giwn.wn,beta, coeff)

    #spline for n data only..
    spl = Spline1D(tau,real(gtau))
    tau = LinRange(0,beta,nwn)
    gtau = spl.(tau)

    return GfimTime{eltype(Giwn.wn)}(Giwn.name,tau,gtau,beta)
end

### G(iωₙ) = Fourier(G(τ))
function Fourier(Gtau::GfimTime)
    # parameter
    ntau = length(Gtau.tau)

    # get G(iωₙ), not using fft.
    wn,gwn = ftau_to_fwn(Gtau.data,Gtau.tau,Gtau.beta)
    return GfimFreq{eltype(Gtau.tau)}(Gtau.name,wn,gwn,Gtau.beta)
end

## G(ω) = Analytical Continuation G(iωₙ), using Pade.
function setfromPade(Giwn::GfimFreq; nw::Int, wrange::T,
                npoints=nothing, broadening=0.01) where T <: Tuple{Number,Number}

    ## only use non-negative n for pade recursion
    nwn = Int((length(Giwn.wn) - 1) / 2)
    wn  = 1im.*Giwn.wn[length(Giwn.data)-nwn+1:end]
    gwn = Giwn.data[length(Giwn.data)-nwn+1:end]
    # initialization
    eta = broadening
    w = LinRange(wrange...,nw) .+ 1im.*eta
    w = eltype(wn).(w)

    # matsubara points to sample for pade recursion,
    # if too large, or precision too low there are numerical error, NAN.
    # warn my precision
    if eltype(Giwn.wn) == Float32
        @warn "Your precision is Float32, convert to Float64, or BigFloat"
    end
    npoints = (npoints == nothing ? length(wn) : npoints)

    # get pade coeffision
    coeff = pade_coeff(wn,gwn)

    # pade recursion approximation
    w, gw = pade_recursion(w, wn,gwn,npoints,coeff)

    return GfrealFreq{eltype(Giwn.wn)}(Giwn.name,real(w),gw)
end
