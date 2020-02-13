using FFTW

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
        @inbounds return (x[2] - x[1]) * (retval + HALF*y[1] + HALF*y[end])
    elseif method == :simpsonEven
        @assert length(x) == length(y) "x and y vectors must be of the same length!"
        retval = (17*y[1] + 59*y[2] + 43*y[3] + 49*y[4] + 49*y[end-3] + 43*y[end-2] + 59*y[end-1] + 17*y[end]) / 48
        for i in 5 : length(y) - 1
            retval += y[i]
        end
        return (x[2] - x[1]) * retval
    end
end

function matsubaraTail(ntail::Int,Giwn::GfimFreq)
    if ntail >= length(Giwn.wn)
        throw("your matsubara frequency is fewer than 64, make it larger!")
    end
    Sn = 0
    Sx = 0
    Sy = 0
    Sxx = 0
    Sxy = 0

    for i in length(Giwn.wn)-ntail:length(Giwn.wn)
        wn = Giwn.wn[i]
        Sn += 1
        Sx += 1 / wn^2
        Sy += imag(Giwn.data[i])*wn
        Sxx += 1 / wn^4
        Sxy += imag(Giwn.data[i])/wn
    end

    return (Sx*Sxy-Sxx*Sy)/(Sn*Sxx-Sx^2)
end

function timeTail(Gtau::GfimTime)
    time_tail = -0.5 .+ 0.5 .* (Gtau.tau .- Gtau.beta ./ 2) .- 0.25 .* (Gtau.tau.^2 .- Gtau.beta .* Gtau.tau)
end
