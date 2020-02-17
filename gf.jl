# GREEN FUNCTION MODULES
using LinearAlgebra
using BenchmarkTools
using Dierckx

import Base: +, -, *, /
import Base: copy, inv, convert

include("util.jl")
include("transform.jl")

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
    orbs::Tuple
    data::Array{Complex{T},3}
end

mutable struct GfimFreq{T} <: GreenFunction
    name::String
    mesh::Vector{T}
    orbs::Tuple
    data::Array{Complex{T},3}
    beta::T
end

mutable struct GfimTime{T} <: GreenFunction
    name::String
    mesh::Vector{T}
    orbs::Tuple
    data::Array{Complex{T},3}
    beta::T
end

## for Gw
function setGw(orbs::T1, nw::Int,wrange::T2;
            name::String,precision=Float64) where {T1<:Tuple,T2<:Tuple}
    w = LinRange(wrange...,nw)
    initzeros = zeros(length(orbs),length(orbs), nw)
    return GfrealFreq{precision}(name,w,orbs,initzeros)
end
setGw(;orbs,nw,wrange,name,precision=Float64) = setGw(orbs,nw,wrange,name=name,precision=precision)

Base.show(io::IO,gf::GfrealFreq{T}) where T = print(io,
"GfRealFreq{$T} : $(gf.name)
    orbs($(length(gf.orbs))) : $(gf.orbs)
    mesh    : $(length(gf.mesh))
    range   : ($(gf.mesh[1]),$(gf.mesh[end]))")

## for Giwn
function setGiwn(orbs::T,n::Int,beta::N;
                name::String,precision=Float64) where {N<:Number,T<:Tuple}
    wn = (2*collect(-n:n) .+ 1) * π / beta
    initzeros = zeros(length(orbs),length(orbs), 2n+1)
    return GfimFreq{precision}(name,wn,orbs,initzeros,beta)
end
setGiwn(;orbs,n,beta,name,precision=Float64) = setGiwn(orbs,n,beta,name=name,precision=precision)

Base.show(io::IO,gf::GfimFreq{T}) where T = print(io,
"GfimFreq{$T} : $(gf.name)
    orbs($(length(gf.orbs))) : $(gf.orbs)
    nwn     : $(Int(0.5*(length(gf.mesh)-1)))
    beta    : $(gf.beta)")

## for Gtau
function setGtau(orbs::T,nslices::Int,beta::N;
                name::String,precision=Float64) where {T<:Tuple,N<:Number}
    tau = LinRange(0,beta,nslices)
    initzeros = zeros(length(orbs),length(orbs), nslices)
    return GfimTime{precision}(name,tau,orbs,initzeros,beta)
end
setGtau(;orbs,nslices,beta,name,precision=Float64) = setGtau(orbs,nslices,beta,name=name,precision=precision)

Base.show(io::IO,gf::GfimTime{T}) where T = print(io,
"GfimTime{$T} : $(gf.name)
    orbs($(length(gf.orbs))) : $(gf.orbs)
    L slices: $(length(gf.mesh))
    beta    : $(gf.beta)")

## OPERATOR IN GREEN FUNCTION ===============================||

# copy is one of the most fundamental operator
function copy(G::GreenFunction)
    g = (typeof(G) <: GfrealFreq ? typeof(G)(G.name,G.mesh,G.orbs,G.data) :
        typeof(G)(G.name,G.mesh,G.orbs,G.data,G.beta))
    return g
end

# addition operator is simply element-wise operation
function +(G1::GreenFunction,G2::GreenFunction)
    if typeof(G1) != typeof(G2) throw("you cannot add different type green function") end
    g = copy(G1)
    g.data = G1.data .+ G2.data; g.name = "GF"
    return g
end

# negative of operator is simply element-wise operation
function -(G::GreenFunction)
    g = copy(G)
    g.data = -G.data
    return g
end

# substraction operator is simply element-wise operation
# why not G1-G2 =  G1+(-G2), it takes more allocation because we use -G operator first.
function -(G1::GreenFunction,G2::GreenFunction)
    if typeof(G1) != typeof(G2) throw("you cannot add different type green function") end
    g = copy(G1)
    g.data = G1.data .- G2.data; g.name = "GF"
    return g
end

# multiplication operator between GF is matrix multiplication
function *(G1::GreenFunction,G2::GreenFunction)
    if typeof(G1) != typeof(G2) throw("you cannot multiply different type green function") end
    g = copy(G1)
    g.data = zeros(size(g.data)...) # this is very weird, wth we need to init zeros first to make success.
    for i in 1:length(g.mesh) g.data[:,:,i] = G1.data[:,:,i] * G2.data[:,:,i] end
    g.name = "GF"
    return g
end

# multiplication operator between GF and a number is element-wise operation
function *(a::Number,G::GreenFunction)
    g = copy(G)
    g.data = a .* G.data; g.name = "GF"
    return g
end
*(G::GreenFunction,a::Number) = *(a::Number,G::GreenFunction)

# multiplication operator between GF and vector(must be mesh size) is element-wise operation
function *(v::AbstractVector,G::GreenFunction)
    if length(v) != length(G.mesh)
        throw("DimensionMisMatch: got a dimension with lengths $(length(v)) and $(length(G.mesh))")
    end
    g = copy(G)
    g.data = zeros(size(g.data)...) # this is very weird, wth we need to init zeros first to make success.
    for i in 1:length(g.mesh) g.data[:,:,i] = v[i] .* G.data[:,:,i] end
    g.name = "GF"
    return g
end
*(G::GreenFunction,v::AbstractVector) = *(v::AbstractVector,G::GreenFunction)

# divide operator between GF is matrix operation
# G1 * inv(G2) more faster this way
function /(G1::GreenFunction,G2::GreenFunction)
    if typeof(G1) != typeof(G2) throw("you cannot divide different type green function") end
    g = copy(G1)
    g.data = zeros(size(g.data)...) # this is very weird, wth we need to init zeros first to make success.
    for i in 1:length(g.mesh) g.data[:,:,i] = G1.data[:,:,i] * inv(G2.data[:,:,i]) end
    g.name = "GF"
    return g
end

# divide operator between GF and a number is a * inv(G)
# this type operation is faster than [I]a / G
function /(a::Number,G::GreenFunction)
    g = copy(G)
    g.data = zeros(size(g.data)...) # this is very weird, wth we need to init zeros first to make success.
    for i in 1:length(g.mesh) g.data[:,:,i] = a .* inv(G.data[:,:,i]) end
    g.name = "GF"
    return g
end
# this is simply element-wise operation
function /(G::GreenFunction,a::Number)
    g = copy(G)
    g.data = G.data ./ a; g.name = "GF"
    return g
end

# this is unusual operation
# but we need it because sometimes we found that GF has operation with its mesh
# instead doing G.mesh (+,-,*,/) G for user
# let's make more intutive, inspired from TRIQS

# addition operation is simply w[I] + [G]
function +(iwn::iOmega_n,G::GfimFreq)
    g = copy(G)
    ID = Matrix(I,size(G.data,1),size(G.data,2))
    for i in 1:length(g.mesh) g.data[:,:,i] = 1im*G.mesh[i]*ID + G.data[:,:,i] end
    g.name = "GF"
    return g
end
+(G::GfimFreq,iwn::iOmega_n) = +(iwn::iOmega_n,G::GfimFreq)
-(iwn::iOmega_n,G::GfimFreq) = +(iwn::iOmega_n,-G::GfimFreq)
-(G::GfimFreq,iwn::iOmega_n) = -(iwn - G)

# inverse of green function is matrix inversion
function inv(G::GreenFunction)
    g = copy(G)
    g.data = zeros(size(g.data)...) # this is very weird, wth we need to init zeros first to make success.
    for i in 1:length(g.mesh) g.data[:,:,i] = inv(G.data[:,:,i]) end
    g.name = "GF"
    return g
end

## ASSIGNMENT IN GREEN FUNCTION ===============================||

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
            G0.data[1,1,iw] = integrate1d(wmesh,div)
        end
        for iorb in 2:length(G0.orbs)
            G0.data[iorb,iorb,:] = G0.data[1,1,:]
        end
    elseif check <: GfimTime
        for t in 1:length(G0.mesh)
            fm = 1.0 ./ (exp.(G0.beta .* wmesh) .+ 1.0 )
            intg = A0w .* (fm .- 1.0) .* exp.(-wmesh .* G0.mesh[t])
            G0.data[1,1,t] = integrate1d(wmesh,intg)
        end
        for iorb in 2:length(G0.orbs)
            G0.data[iorb,iorb,:] = G0.data[1,1,:]
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

    tau = LinRange(0,beta,nwn)
    gtau = zeros(eltype(Giwn.data), length(Giwn.orbs),length(Giwn.orbs), length(tau))

    for iorb in 1:length(Giwn.orbs), jorb in 1:length(Giwn.orbs)
        # tail coeff, ntail = 128 is enough, maybe.
        coeff = tail_coeff(Giwn.mesh[nwn+1:end],Giwn.data[iorb,jorb,nwn+1:end], 128)

        # inverse Fourier, produce 2n+1 data (becuase -n:n data)
        tau_,ftau = fwn_to_ftau(Giwn.data[iorb,jorb,:],Giwn.mesh,beta, coeff)

        #spline for n data only..
        spl = Spline1D(tau_,real(ftau))
        gtau[iorb,jorb,:] = spl.(tau)
    end

    return GfimTime{eltype(Giwn.mesh)}(Giwn.name,tau,Giwn.orbs,gtau,beta)
end

### G(iωₙ) = Fourier(G(τ))
function Fourier(Gtau::GfimTime)
    # parameter
    n = length(Gtau.mesh)

    # allocate giwn
    giwn = zeros(eltype(Gtau.data), length(Gtau.orbs), length(Gtau.orbs), 2n+1)
    wn = (2*collect(-n:n) .+ 1) * π / Gtau.beta

    # get G(iωₙ), not using fft.
    for iorb in 1:length(Giwn.orbs), jorb in 1:length(Giwn.orbs)
        _,giwn[iorb,jorb,:] = ftau_to_fwn(Gtau.data[iorb,jorb,:],Gtau.mesh,Gtau.beta)
    end

    return GfimFreq{eltype(Gtau.mesh)}(Gtau.name,wn,Gtau.orbs,giwn,Gtau.beta)
end

## G(ω) = Analytical Continuation G(iωₙ), using Pade.
function setfromPade(Giwn::GfimFreq; nw::Int, wrange::T,
                npoints=nothing, broadening=0.01) where T <: Tuple{Number,Number}

    ## only use non-negative n for pade recursion
    nwn = Int((length(Giwn.mesh) - 1) / 2)
    wn  = 1im.*Giwn.mesh[length(Giwn.mesh)-nwn+1:end]
    # initialization
    w = LinRange(wrange...,nw) .+ 1im.*broadening
    w = eltype(wn).(w)

    # matsubara points to samplings pade recursion,
    # if too large, or precision too low there are numerical error, NAN.
    # warn my precision
    if eltype(Giwn.mesh) == Float32
        @warn "Your precision is Float32, too low. use Float64, for more accurate result."
    end
    npoints = (npoints == nothing ? length(wn) : npoints)

    gw = zeros(eltype(Giwn.data), length(Giwn.orbs),length(Giwn.orbs), length(w))
    for iorb in 1:length(Giwn.orbs), jorb in 1:length(Giwn.orbs)
        gwn = Giwn.data[iorb,jorb,length(Giwn.mesh)-nwn+1:end]

        # get pade coeffision
        coeff = pade_coeff(wn,gwn)

        # pade recursion approximation
        gw[iorb,jorb,:] = pade_recursion(w, wn,gwn,npoints,coeff)
    end

    return GfrealFreq{eltype(Giwn.mesh)}(Giwn.name,real(w),Giwn.orbs,gw)
end
