# important module
using Plots
using LinearAlgebra
using BenchmarkTools
using FFTW
using Test

## parameter
beta=16.0;
U=2;
Ed=1;
J=0;
Nd=2;
L=64;
ncor=3.0;
binsize=1000;
nbin=50;
ndirty=100;
nwarm0=100;
nwarm1=10;
random_site=true;
Metropolis=true;
Bethe=true;
ph_symmetry=true;
cure=true;
incf=-1;
niter=20;
mix=0.5;
m0=8;
real_w_cutof=5;
N_real_w=500;
Niwn=500;

## QMC
### Return Σ(w)
function HF_qmc(G0iwn,params)
    # get G0tau by fourier transform
    G0tau = invfourier_giwn(G0iwn)
    # Make matrix L * L of G0tau
    Glltau = get_glltau(G0tau)
    # get starting ising configuration
    S = startIsingConfig(params, from_file=false)

    ## Start QMC loop
    for iter in 1:niter
    end

    print_result()

    # get Observable
end

function readG0tau(G0tau,L)
    L = length(G0tau)
    Glltau = zeros(ComplexF64,L,L)
    for i in 0:L-1, j in 0:L-1
        ind = i-j
        if sign(ind) == -1
            Glltau[i+1,j+1] = -G0tau[ind+L]
        else
            Glltau[i+1,j+1] = -G0tau[ind+1]
        end
    end
    return Glltau
end

function startIsingConfig(nf,L)
    return rand([-1,1],nf*L)
end

function freq_tail(coef, β, τ, wn)
    coef = coef*1
    giwn_tail = coef[1] ./ (1im .* wn) .+
            coef[2] ./ (1im*wn).^2 .+
            coef[3] ./ (1im*wn).^3

    gtau_tail = coef[1] .* (-0.5) .+
            coef[2] .* (0.5 * (τ .- 0.5 .* β)) .+
            coef[3] .* (-0.25 * (τ.^2 .- β*τ))

    return giwn_tail, gtau_tail
end

function invfourier_gwn(giwn,wn,τ,tail_coef=[true,true,false])
    β = τ[1] + τ[end]
    giwn_tail, gtau_tail = freq_tail(tail_coef,β,τ,wn)

    giwn -= giwn_tail
    # set giwn size same as τ
    tmp_giwn = copy(giwn)
    giwn = zeros(ComplexF64,length(τ))
    for i in 1:length(τ)
        ind = floor(Int64,(i-1)*(length(wn)-1)/(length(τ)-1))
        giwn[i] = tmp_giwn[ind+1]
    end
    gtau = fft(giwn)

    return tmp_giwn,imag((gtau*2/β)) + gtau_tail
end
