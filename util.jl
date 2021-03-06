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
