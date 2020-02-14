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


## for non-interacting cases
function setfromToy!(G0::GreenFunction,toy::Symbol)
    # check green function
    check = typeof(G0)
    ## for gtau, use invfourier of giwn
    if check <: GfimTime
        throw("use Fourier(Giwn) to get Gtau instead")
    end
    # make sure using correct type frequency
    freq = (check <: GfrealFreq ? G0.w .+ 0.01im : 1im*G0.wn)

    # get A0w from every chosen toy model
    wmesh = 501 # i think this is enough for integration purpose
    if toy == :bethe
        wmesh = LinRange(-1.01,1.01,wmesh) #for bethe, the A0w for default t =0.5
                                          #the band has value in w ∈ [-1,1]
        A0w = A0bethe.(wmesh)
    else
        @error "Not yet implemented, try :bethe"
    end

    for iw in 1:length(freq)
        div = (A0w ./ (freq[iw] .- wmesh))
        G0.data[iw] = integrate1d(wmesh,div)
    end
    return G0
end

A0bethe(w; t=0.5) = (abs(w) < 2t ? sqrt(4t^2 - w^2) / (2π*t^2) : 0)
#A0cubic(w,t)
#A0square(w,t)
#setfromHk()
#setfromAw(Aw,iw)

# transform green function to one another

## transform giwn to gtau
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

    # i do not know why when we stretch n to 2n do the trick
    # whatever... it works
    gwn = zeros(eltype(Giwn.data),2nwn)
    gwn[2:2:end] = Giwn.data .- coeff * wn_tail

    # fourier transform
    gtau = fft(gwn)
    # now take value for [0,β] only
    gtau = gtau[1:nwn] .* (2 ./ beta) .+ coeff * tau_tail
    # little correction, do not know what is this
    a = real(gwn[end])*wn[end] / π
    gtau[1] += a
    gtau[end] -= a

    # tau
    tau = LinRange(0,beta,length(gtau))
    return GfimTime{eltype(Giwn.wn)}(Giwn.name,tau,real(gtau),beta)
end

function Fourier(Gtau::GfimTime)
    beta = Gtau.beta
    tau  = Gtau.tau
    ntau = length(Gtau.tau)
    wn = (2*collect(1:ntau) .+ 1) * π / beta

    gwn = -ft_forward(length(wn),ntau,beta,Gtau.data,tau,wn)
    return GfimFreq{eltype(Gtau.tau)}(Gtau.name,wn,gwn,beta)
end

##setfromPade
