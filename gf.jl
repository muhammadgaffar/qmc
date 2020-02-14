# GREEN FUNCTION MODULES

## CONSTRUCT BASIS OF GREEN FUNCTION ===============================||

using BenchmarkTools
using Dierckx

abstract type GreenFunction end

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
    wn = (2*collect(1:n) .+ 1) * π / beta
    initzeros = zeros(n)
    return GfimFreq{precision}(name,wn,initzeros,beta)
end
setGiwn(;n,wrange,name,beta,precision=Float32) = setGiwn(n,beta,name=name,precision=precision)

Base.show(io::IO,gf::GfimFreq{T}) where T = print(io,
"GfimFreq{$T} : $(string(gf.name))
    nwn     : $(length(gf.wn))
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

### for non-interacting cases
function setfromToy!(G0::GreenFunction,toy::Symbol)
    check = typeof(G0) # check green function type
    if check <: GfimTime # gtau is not yet implemented for direct toy model, use fourier transformation instead
        throw("use Fourier(Giwn) to get Gtau instead")
    end
    freq = (check <: GfrealFreq ? G0.w .+ 0.01im : 1im*G0.wn) # Frequency type for hilbert transform

    # get A0w from toy model
    wmesh = 501 # i think this is enough for integration purpose, maybe.
    if toy == :bethe
        wmesh = LinRange(-1.01,1.01,wmesh) #for bethe, the A0w for default t =0.5
                                          #the band has value in w ∈ [-1,1]
        A0w = A0bethe.(wmesh)
    else
        @error "Not yet implemented, try :bethe" # help me.
    end

    # now do hilbert transform.
    for iw in 1:length(freq)
        div = (A0w ./ (freq[iw] .- wmesh))
        G0.data[iw] = integrate1d(wmesh,div)
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
    beta = Giwn.beta
    wn   = Giwn.wn
    nwn  = length(Giwn.wn)

    # tail coeff, N moments = 32 is good enough, maybe.
    coeff = tail_coeff(128,Giwn)
    # tail expansion, first order
    wn_tail = 1.0 ./ (1im.*wn)
    tau_tail = -0.5

    # i do not know why when we stretch n to 2n do the trick for fft purpose.
    # whatever... it works
    gwn = zeros(eltype(Giwn.data),2nwn)
    gwn[2:2:end] = Giwn.data .- coeff * wn_tail

    # fourier transform
    gtau = fft(gwn)
    # now take value for [0,β) only
    gtau = gtau[1:nwn] .* (2 ./ beta) .+ coeff * tau_tail
    # little correction, still do not know what is this
    a = real(gwn[end])*wn[end] / π
    gtau[1] += a
    gtau[end] -= a

    # get G(τ)
    tau = LinRange(0,beta,length(gtau))
    return GfimTime{eltype(Giwn.wn)}(Giwn.name,tau,real(gtau),beta)
end

### G(iωₙ) = Fourier(G(τ))
function Fourier(Gtau::GfimTime)
    # parameter
    beta = Gtau.beta
    tau  = Gtau.tau
    ntau = length(Gtau.tau)
    wn = (2*collect(1:ntau) .+ 1) * π / beta

    # get G(iωₙ)
    gwn = -ft_forward(length(wn),ntau,beta,Gtau.data,tau,wn)
    return GfimFreq{eltype(Gtau.tau)}(Gtau.name,wn,gwn,beta)
end

## G(ω) = Analytical Continuation G(iωₙ), using Pade.
function setfromPade(Giwn::GfimFreq; nw::Int, wrange::T,
                npoints=nothing, broadening=0.0001) where T <: Tuple{Number,Number}
    # initialization
    eta = broadening
    wn  = 1im.*Giwn.wn
    if npoints == nothing # npoints to sample, if too large, there are numerical error, NAN.
        nwn = length(wn)
    else
        nwn = npoints
    end
    r   = floor(Int,nwn/2)
    w   = LinRange(wrange...,nw)
    w   = collect(w) .+ 1im.*eta

    # get pade Coeff
    coeff = pade_coeff(Giwn)

    # start pade recursion
    an_prev = 0.0
    an      = coeff[1]
    bn_prev = 1.0
    bn      = 1.0
    for i in 2:r
        an_next = an .+ (w .- wn[i-1]) .* coeff[i] .* an_prev
        bn_next = bn .+ (w .- wn[i-1]) .* coeff[i] .* bn_prev
        an_prev, an = an, an_next
        bn_prev, bn = bn, bn_next
    end

    # get Gw from pade
    gw = an ./ bn
    return GfrealFreq{eltype(Giwn.wn)}(Giwn.name,real(w),gw)
end
