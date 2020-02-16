### flexible functions for function transforms
### not necessarily for green function object
### much easier for expand and debugging

using FFTW

function tail_coeff(wn,fwn, ntail)
    Sn, Sx, Sy, Sxx, Sxy = (0, 0, 0, 0, 0)
    for j in length(wn)-ntail:length(wn)
        Sn += 1.0
        Sx += 1/wn[j]^2
        Sy += imag(fwn[j])*wn[j]
        Sxx += 1/wn[j]^4
        Sxy += imag(fwn[j])/wn[j]
    end
    c1 = (Sx*Sxy-Sxx*Sy)/(Sn*Sxx - Sx*Sx)

    # not yet implemented, help me
    c2 = 0.0 # most likely to be μ
    c3 = 0.0
    return c1,c2,c3
end

function highfreq_tail(wn,tau,beta; coeff=(1.0,0.0,0.0))
    c1,c2,c3 = coeff
    wn_tail = c1 ./ (1im .* wn) +
              c2 ./ (1im .* wn).^2 +
              c3 ./ (1im .* wn).^3

    tau_tail = - c1 ./ 2 .+
                 c2 ./ 2 .* (tau .- beta/2) +
               - c3 ./ 4 .* (tau.^2 - beta .* tau)

    return wn_tail, tau_tail
end

function fwn_to_ftau(fwn,wn,beta,tail_coeff)
    # initialize tau
    tau = LinRange(0,beta,length(wn))
    # get the tail
    fwn_tail, ftau_tail = highfreq_tail(wn,tau,beta, coeff=tail_coeff)
    # substract to its tail
    fwn  = fwn - fwn_tail
    # fft!
    ftau = fft(fwn)
    ftau .*= exp.(-1im .* π .* (1 + length(wn)) .* tau ./ beta) / beta
    # re add to its tail
    ftau .+= ftau_tail
    # little correction
    ftau[end] = -(ftau[1]+tail_coeff[1])
    return tau, ftau
end

function ftau_to_fwn(ftau,tau,beta)
    #initialize wn and fwn
    n = length(tau)
    wn = (2*collect(-n:n) .+ 1) * π / beta
    fwn = zeros(eltype(ftau),2n+1)

    #standard fourier transform.
    for i in 1:length(fwn)
        for t in 1:n-1
            ft2 = ftau[t+1]; ft1 = ftau[t]
            t2  = tau[t+1]; t1 = tau[t]
            fwn[i] -= exp(1im*t2*wn[i])*(-ft2+ft1+im*(t2-t1)*ft2*wn[i]) / ((t2-t1)*wn[i]^2)
            fwn[i] -= exp(1im*t1*wn[i])*( ft2-ft1-im*(t2-t1)*ft1*wn[i]) / ((t2-t1)*wn[i]^2)
        end
    end
    return wn,fwn
end

function pade_coeff(wn,fwn)
    # allocate array
    coeff = zeros(eltype(fwn),(length(wn),length(wn)))
    # start calculating coefficient for recursion
    coeff[1,:] = fwn
    one = eltype(wn)(1.0)
    for i in 2:length(wn)
        coeff[i,:] = (coeff[i-1,i-1] ./ coeff[i-1,:] .- one) ./ (wn .- wn[i-1])
    end
    return diag(coeff)
end

function pade_recursion(w, wn,gwn,npoints,coeff)
    #start recursion
    an_prev = eltype(wn)(0.0)
    an      = coeff[1]
    bn_prev = eltype(wn)(1.0)
    bn      = eltype(wn)(1.0)
    for i in 2:npoints
        an_next = an .+ (w .- wn[i-1]) .* coeff[i] .* an_prev
        bn_next = bn .+ (w .- wn[i-1]) .* coeff[i] .* bn_prev
        an_prev, an = an, an_next
        bn_prev, bn = bn, bn_next
    end
    return w, an ./ bn
end
