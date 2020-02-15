using FFTW
using LinearAlgebra

function integrate1d(x::AbstractVector, y::AbstractVector,method=:simpsonEven)
    HALF = 0.5
    if method == :trapezoidal
        retval = 0
        @fastmath @simd for i in 1 : length(y)-1
            @inbounds retval += (x[i+1] - x[i]) * (y[i] + y[i+1])
        end
        return HALF * retval
    elseif method == :trapezoidalEven
        retval = 0
        N = length(y) - 1
        @fastmath @simd for i in 2 : N
            @inbounds retval += y[i]
        end
        return (x[2] - x[1]) * (retval + HALF*y[1] + HALF*y[end])
    elseif method == :simpsonEven
        retval = (17*y[1] + 59*y[2] + 43*y[3] + 49*y[4] + 49*y[end-3] + 43*y[end-2] + 59*y[end-1] + 17*y[end]) / 48
        @fastmath @simd for i in 5 : length(y) - 1
            @inbounds retval += y[i]
        end
        return (x[2] - x[1]) * retval
    end
end

function tail_coeff(n_moment::Int,Giwn::GfimFreq)
    wn = Giwn.wn
    gwn = Giwn.data
    nwn = length(wn)

    Sn, Sx, Sy, Sxx, Sxy = (0, 0, 0, 0, 0)
    for j in nwn-n_moment:nwn
        ωn = wn[j]
        Sn += 1
        Sx += 1/ωn^2
        Sy += imag(gwn[j])*ωn
        Sxx += 1/ωn^4
        Sxy += imag(gwn[j])/ωn
    end
    return (Sx*Sxy-Sxx*Sy)/(Sn*Sxx - Sx*Sx)
end

function ft_forward(mfreq,ntime,β,vτ,τmesh,ωmesh)
        vω = zeros(Complex{Float64},mfreq)
        for i in 1:mfreq
            ωn = ωmesh[i]
            for j in 1:ntime-1
                fa = vτ[j+1]
                fb = vτ[j]
                a = τmesh[j+1]
                b = τmesh[j]
                vω[i] += exp(im*a*ωn)*(-fa+fb+im*(a-b)*fa*ωn)/((a-b)*ωn^2)
                vω[i] += exp(im*b*ωn)*(fa-fb-im*(a-b)*fb*ωn)/((a-b)*ωn^2)
            end
        end
        return vω
end

#get pade coefficient
function pade_coeff(Giwn::GfimFreq)
    wn = 1im.*Giwn.wn
    nwn = length(Giwn.wn)
    coeff = zeros(eltype(Giwn.data),(nwn,nwn))
    coeff[1,:] = Giwn.data
    for i in 2:nwn
        coeff[i,:] = (coeff[i-1,i-1] ./ coeff[i-1,:] .- 1.0) ./ (wn[i-1].-wn)
    end
    return diag(coeff)
end
