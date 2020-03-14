# GREEN FUNCTION MODULES
using LinearAlgebra
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


# a note, why the gf object is restricted to float64?
# I know for it's good if we have flexibel precision like i was intended to be
# but I began realized that this was a bad practice for general green function process
# causing too much degree of freedom is not good
# if we do really need to kills precision for better performance, use it in process that-
# -takes much cpu time, usually involve high number of iteration such as monte carlo

mutable struct GfrealFreq{N} <: GreenFunction
    orbs::Tuple
    mesh::Vector{Float64}
    data::Array{Complex{Float64},3}
end

mutable struct GfimFreq{N} <: GreenFunction
    orbs::Tuple
    mesh::Vector{Float64}
    data::Array{Complex{Float64},3}
end

mutable struct GfimTime{N} <: GreenFunction
    orbs::Tuple
    mesh::Vector{Float64}
    data::Array{Complex{Float64},3}
end

## for Gw
function setGw(orbs::T1, nw::Int,wrange::T2) where {T1<:Tuple,T2<:Tuple}
    w = LinRange(wrange...,nw)
    norb = length(orbs)
    initzeros = zeros(norb,norb, nw)
    return GfrealFreq{norb}(orbs,w,initzeros)
end
setGw(;orbs,nw,wrange) = setGw(orbs,nw,wrange)

Base.show(io::IO,gf::GfrealFreq{N}) where N = print(io,
"Green Function{Real Frequency} :  Matrix$((length(gf.orbs),length(gf.orbs)))
    orbs    : $(gf.orbs)
    mesh    : $(length(gf.mesh))
    range   : ($(gf.mesh[1]),$(gf.mesh[end]))")

## for Giwn
function setGiwn(orbs::T,n::Int,beta::R) where {T<:Tuple,R<:Real}
    wn = (2*collect(-n:n) .+ 1) * π / beta
    norb = length(orbs)
    initzeros = zeros(length(orbs),length(orbs), 2n+1)
    return GfimFreq{norb}(orbs,wn,initzeros)
end
setGiwn(;orbs,n,beta) = setGiwn(orbs,n,beta)

function Base.show(io::IO,gf::GfimFreq{N}) where N
    nwn = Int(0.5*(length(gf.mesh)-1))
    beta = π / gf.mesh[nwn+1]
    print(io,
"Green Function{Imaginary Frequency} : Matrix$((length(gf.orbs),length(gf.orbs)))
    orbs    : $(gf.orbs)
    Nwn     : $nwn
    beta    : $beta")
end

## for Gtau
function setGtau(orbs::T,nslices::Int,beta::R) where {T<:Tuple,R<:Real}
    tau = LinRange(0,beta,nslices)
    norb = length(orbs)
    initzeros = zeros(length(orbs),length(orbs), nslices)
    return GfimTime{norb}(orbs,tau,initzeros)
end
setGtau(;orbs,nslices,beta) = setGtau(orbs,nslices,beta)

function Base.show(io::IO,gf::GfimTime{N}) where N
    ntau = length(gf.mesh)
     print(io,
"GfimTime{Imaginary Time} : Matrix$((length(gf.orbs),length(gf.orbs)))
    orbs    : $(gf.orbs)
    Ntau    : $(length(gf.mesh))
    beta    : $(gf.mesh[end])")
end

## OPERATOR IN GREEN FUNCTION ===============================||

# copy is one of the most fundamental operator
copy(G::GreenFunction) = typeof(G)(G.orbs,G.mesh,G.data)

# addition operator is simply element-wise operation
function +(G1::GreenFunction,G2::GreenFunction)
    if typeof(G1) != typeof(G2) throw("GF type/size MisMatch") end
    g = copy(G1); g.data = G1.data .+ G2.data
    return g
end

# negative of operator is simply element-wise operation
function -(G::GreenFunction)
    g = copy(G); g.data = -G.data
    return g
end

# substraction operator is simply element-wise operation
# why not G1-G2 =  G1+(-G2), it takes more allocation because we use -G operator first.
function -(G1::GreenFunction,G2::GreenFunction)
    if typeof(G1) != typeof(G2) throw("GF type/size MisMatch") end
    g = copy(G1); g.data = G1.data .- G2.data
    return g
end

# multiplication operator between GF is matrix multiplication
function *(G1::GreenFunction,G2::GreenFunction)
    if typeof(G1) != typeof(G2) throw("GF type/size MisMatch") end
    g = copy(G1); g.data = zeros(size(g.data)...) # this is very weird, wth we need to init zeros first to make success.
    @inbounds for i in 1:length(g.mesh) g.data[:,:,i] = G1.data[:,:,i] * G2.data[:,:,i] end
    return g
end

# multiplication operator between GF and a number is element-wise operation
function *(a::Number,G::GreenFunction)
    g = copy(G); g.data = a .* G.data;
    return g
end
*(G::GreenFunction,a::Number) = *(a::Number,G::GreenFunction)

# multiplication operator between GF and vector(must be mesh size) is element-wise operation
function *(v::AbstractVector,G::GreenFunction)
    if length(v) != length(G.mesh)
        throw("DimensionMisMatch: got a dimension with lengths $(length(v)) and $(length(G.mesh))")
    end
    g = copy(G); g.data = zeros(size(g.data)...) # this is very weird, wth we need to init zeros first to make success.
    @inbounds for i in 1:length(g.mesh) g.data[:,:,i] = v[i] .* G.data[:,:,i] end
    return g
end
*(G::GreenFunction,v::AbstractVector) = *(v::AbstractVector,G::GreenFunction)

# divide operator between GF is matrix operation
# G1 * inv(G2) more faster this way
function /(G1::GreenFunction,G2::GreenFunction)
    if typeof(G1) != typeof(G2) throw("GF type/size MisMatch") end
    g = copy(G1); g.data = zeros(size(g.data)...) # this is very weird, wth we need to init zeros first to make success.
    @inbounds for i in 1:length(g.mesh) g.data[:,:,i] = G1.data[:,:,i] * inv(G2.data[:,:,i]) end
    return g
end

# divide operator between GF and a number is a * inv(G)
# this type operation is faster than [I]a / G
function /(a::Number,G::GreenFunction)
    g = copy(G); g.data = zeros(size(g.data)...) # this is very weird, wth we need to init zeros first to make success.
    @inbounds for i in 1:length(g.mesh) g.data[:,:,i] = a .* inv(G.data[:,:,i]) end
    return g
end
# this is simply element-wise operation
function /(G::GreenFunction,a::Number)
    g = copy(G); g.data = G.data ./ a
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
    @inbounds for i in 1:length(g.mesh) g.data[:,:,i] = 1im*G.mesh[i]*ID + G.data[:,:,i] end
    return g
end
+(G::GfimFreq,iwn::iOmega_n) = +(iwn::iOmega_n,G::GfimFreq)
-(iwn::iOmega_n,G::GfimFreq) = +(iwn::iOmega_n,-G::GfimFreq)
-(G::GfimFreq,iwn::iOmega_n) = -(iwn - G)

# inverse of green function is matrix inversion
function inv(G::GreenFunction)
    g = copy(G); g.data = zeros(size(g.data)...) # this is very weird, wth we need to init zeros first to make success.
    @inbounds for i in 1:length(g.mesh) g.data[:,:,i] = inv(G.data[:,:,i]) end
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
        @inbounds for iorb in 2:length(G0.orbs)
            G0.data[iorb,iorb,:] = G0.data[1,1,:]
        end
    elseif check <: GfimTime
        beta = G0.mesh[end]
        for t in 1:length(G0.mesh)
            fm = 1.0 ./ (exp.(beta .* wmesh) .+ 1.0 )
            intg = A0w .* (fm .- 1.0) .* exp.(-wmesh .* G0.mesh[t])
            G0.data[1,1,t] = integrate1d(wmesh,intg)
        end
        @inbounds for iorb in 2:length(G0.orbs)
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
    beta = π / Giwn.mesh[nwn+1]

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

    return GfimTime{length(Giwn.orbs)}(Giwn.orbs,tau,gtau)
end

# another way to do invfourier
function invFourier!(Giwn::GfimFreq, Gtau::GfimTime)
    # data and parameter
    nwn  = Int(0.5*(length(Giwn.mesh)-1))
    beta = π / Giwn.mesh[nwn+1]

    for iorb in 1:length(Giwn.orbs), jorb in 1:length(Giwn.orbs)
        # tail coeff, ntail = 128 is enough, maybe.
        coeff = tail_coeff(Giwn.mesh[nwn+1:end],Giwn.data[iorb,jorb,nwn+1:end], 128)

        # inverse Fourier, produce 2n+1 data (becuase -n:n data)
        tau_,ftau = fwn_to_ftau(Giwn.data[iorb,jorb,:],Giwn.mesh,beta, coeff)

        #spline for n data only..
        spl = Spline1D(tau_,real(ftau))
        Gtau.data[iorb,jorb,:] = spl.(Gtau.mesh)
    end

    return Gtau
end

### G(iωₙ) = Fourier(G(τ))
function Fourier(Gtau::GfimTime)
    # parameter
    n = length(Gtau.mesh)
    beta = Gtau.mesh[end]

    # allocate giwn
    giwn = zeros(eltype(Gtau.data), length(Gtau.orbs), length(Gtau.orbs), 2n+1)
    wn = (2*collect(-n:n) .+ 1) * π / beta

    # get G(iωₙ), not using fft.
    for iorb in 1:length(Gtau.orbs), jorb in 1:length(Gtau.orbs)
        _,giwn[iorb,jorb,:] = ftau_to_fwn(Gtau.data[iorb,jorb,:],Gtau.mesh, beta, n)
    end

    return GfimFreq{eltype(Gtau.mesh)}(Gtau.orbs,wn,giwn)
end

# another way to do fourier
function Fourier!(Gtau::GfimTime, Giwn::GfimFreq)
    # parameter
    n = length(Giwn.mesh)
    beta = Gtau.mesh[end]

    # allocate giwn
    giwn = zeros(eltype(Gtau.data), length(Gtau.orbs), length(Gtau.orbs), 2n+1)
    wn = (2*collect(-n:n) .+ 1) * π / beta

    # get G(iωₙ), not using fft.
    for iorb in 1:length(Gtau.orbs), jorb in 1:length(Gtau.orbs)
        Giwn.mesh,Giwn.data[iorb,jorb,:] = ftau_to_fwn(Gtau.data[iorb,jorb,:],Gtau.mesh, beta, Int(0.5*(n-1)))
    end

    Giwn.orbs = Gtau.orbs
    return Giwn
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
    npoints = (npoints == nothing ? length(wn) : npoints)

    gw = zeros(eltype(Giwn.data), length(Giwn.orbs),length(Giwn.orbs), length(w))
    for iorb in 1:length(Giwn.orbs), jorb in 1:length(Giwn.orbs)
        gwn = Giwn.data[iorb,jorb,length(Giwn.mesh)-nwn+1:end]

        # get pade coeffision
        coeff = pade_coeff(wn,gwn)

        # pade recursion approximation
        gw[iorb,jorb,:] = pade_recursion(w, wn,gwn,npoints,coeff)
    end

    return GfrealFreq{eltype(Giwn.mesh)}(Giwn.orbs,real(w),gw)
end;
