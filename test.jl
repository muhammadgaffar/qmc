using Test
include("gf.jl")
include("util.jl")
include("plots.jl")

G0w = setGw(nw=201,wrange=(-2,2),name="nonInteracting")
setfromToy!(G0w,:bethe)
plot(G0w)

Giwn = setGiwn(1024,24,name="nonInteracting",precision=Float64)
setfromToy!(Giwn,:bethe)
plot(Giwn.wn,imag(Giwn.data))

Gtau = invFourier(Giwn)
Giwn = Fourier(Gtau)
plot!(Gtau.tau,real(Gtau.data))

Giwn = fourierGtau(Giwn,Gtau)
plot(Giwn.wn,imag(Giwn.data))

include("util.jl")
include("plots.jl")
