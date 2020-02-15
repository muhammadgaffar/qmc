using Test
include("gf.jl")
include("util.jl")
include("plots.jl")

G0w = setGw(nw=201,wrange=(-2,2),name="nonInteracting")
setfromToy!(G0w,:bethe)
plot(G0w)

Giwn = setGiwn(2^12,24,name="nonInteracting",precision=Float64)
setfromToy!(Giwn,:bethe)
plot(Giwn.wn,imag(Giwn.data))

Gtau = setGtau(256,24,name="nonInteracting")
setfromToy!(Gtau,:bethe)
plot(Gtau.tau,real(Gtau.data))

Gw = setfromPade(Giwn, nw=101,wrange=(-5,5),npoints=512)
plot(Gw)

Gtau = invFourier(Giwn)
Giwn = Fourier(Gtau)
plot!(Gtau.tau,real(Gtau.data))

Giwn = fourierGtau(Giwn,Gtau)
plot(Giwn.wn,imag(Giwn.data))

include("util.jl")
include("plots.jl")
