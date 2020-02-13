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
    mesh    : $(length(gf.w))")

## for Giwn
function setGiwn(n::Int,beta::T;name::String,precision=Float32) where T <: Number
    wn = (2*collect(1:n) .+ 1) * π / beta
    initzeros = zeros(n)
    return GfimFreq{precision}(name,wn,initzeros,beta)
end
setGiwn(;n,wrange,name,beta,precision=Float32) = setGiwn(n,beta,name=name,precision=precision)

Base.show(io::IO,gf::GfimFreq{T}) where T = print(io,
"GfimFreq{$T} : $(string(gf.name))
    n       : $(length(gf.wn))
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
    n       : $(length(gf.tau))
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
function invfourierGiwn(gtau::GfimTime,giwn::GfimFreq)
    # warning if have different beta, set gtau.beta = giwn.beta
    # and set reset tau
    if gtau.beta != giwn.beta
        @warn "you have different β! β is forced to be same as giwn system"
        gtau.beta = giwn.beta
        gtau.tau  = LinRange(0,gtau.beta,length(gtau.tau))
    end
    tail = matsubaraTail(4,giwn) # retrive tail of matsubara

    gdummy = zeros(eltype(giwn.data),2length(giwn.wn))
    for i in 1:length(giwn.wn)
        gdummy[2i] = giwn.data[i] - tail ./ (1im*giwn.wn[i])
    end

    # fourier transform !
    fft!(gdummy)
    gdummy = real(gdummy)*(2/giwn.beta) .- 0.5tail
    a = real(giwn.data[end])*giwn.wn[end] / π
    gdummy[1] += a
    gdummy[end] -= a

    #spline interpolation! for ntau
    ntaudummy = floor(Int,length(gdummy)/2)
    taudummy = LinRange(0,gtau.beta,ntaudummy)
    spl = Spline1D(taudummy,gdummy[1:ntaudummy])
    gtau.data = spl.(gtau.tau)
    return gtau
end

## transform gtau to giwn
function fourierGtau(giwn::GfimFreq,gtau::GfimTime)
    τ_tail = timeTail(gtau)
    freq_tail = matsubaraTail(64,giwn)

    gdummy = gtau.beta .* ifft((gtau.data-τ_tail) .* exp.(im*π .* gtau.tau ./ gtau.beta)) .+ freq_tail

    #spline interpolation! for nwn
    wn = (2*collect(1:length(gdummy)) .+ 1) ./ giwn.beta
    imspl = Spline1D(wn,imag(gdummy))
    respl = Spline1D(wn,imag(gdummy))
    giwn.data = respl.(giwn.wn) + 1im*imspl.(giwn.wn)
    return giwn
end

##setfromPade
x = setGw(nw=201,wrange=(-2,2),name="mada")
y = setGiwn(1024,16,name="imaginary")
z = setGtau(64,16;name="imtime")
setfromToy!(x,:bethe)
setfromToy!(y,:bethe)

z=invfourierGiwn(z,y);
y=fourierGtau(y,z)

plot(z.tau,real(z.data))
plot(y.wn,imag(y.data))

include("util.jl")
include("plots.jl")
