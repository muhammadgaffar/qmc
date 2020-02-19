using LinearAlgebra
using Dierckx
const lapack = LinearAlgebra.LAPACK
const blas   = LinearAlgebra.BLAS

import Dates

include("gf.jl")

function HF_Solve(Giwn::GfimFreq,params::Dict)
    # check parameters, and initialization
    U,J,L,beta,file,niter,ntherm,nsweep,nmeas,ndirty = checkParams(params)
    if (beta == -1) beta = π / Giwn.mesh[1] end # if beta is not set in parameter
    Ui,pair,fs = getisingParams(U,J, size(Giwn.data,1))
    nIsing = length(Ui)
    nwarmup = ntherm*nIsing

    # PRINTOUT INTRODUCTION OF PROGRAMS
    printIntro(U,J,L,beta,niter,ntherm,nsweep,nmeas,ndirty,nIsing)

    ## get random ising quantities over L slices and N orbs
    ## This quantitiy is V for use in e^{-V}, this is result of hubbard-stranovich
    Vs = startIsing(Ui,beta,L)

    # monte carlo loop
    ## warming up iteration first, we do not collect measurement in warming up
    naccept = 0
    for iter in 1:niter
        ## Fourier transform Giwn
        Gtau = invFourier(Giwn)
        ## Make LxL matrix out of Gtau
        ## using spline interpolation for new tau
        tau, g0tau = get_G0tau(Gtau, L,beta)
        ## make clean update first, calculate g from A*g = g0
        g = cleanUpdate(Vs,fs,g0tau)

        #start sweep iteration
        for isweep in 1:(nsweep+ntherm)
            visited_avg = isweep / (L*nIsing)
            r_site = floor(Int, rand()*L*nIsing)
            l = floor(Int, (r_site % L)); if (l == 0) l = 1 end # which time slices
            z = floor(Int, (r_site / L) ) + 1 #which state

            P,a = detRatio(z,l,g,Vs,pair) # metropolis

            if (P > rand()) acceptMove!(g,Vs,a,fs,pair, naccept,ndirty, z, l) end
        end

    end

    # fitting

    # measurement

    # print result to HDF5

    # only return interacting giwn and sigma_iwn, rest in hdf5
    return
end

function checkParams(params::Dict)
    U = try float(params["U"]) catch; throw("Input U is missing") end
    if typeof(U) <: Real 1 else throw("Input U is not a real number") end

    J = try float(params["J"]) catch; throw("Input J is missing") end
    if typeof(J) <: Real 1 else throw("Input J is not a real number") end

    beta = try float(params["beta"])
        catch
             @warn("Input beta is missing, program will use beta from G0(iωn)")
             -1
        end
    if typeof(beta) <: Float64 1 else throw("Input beta is not a real number") end

    L = try params["L_slices"] catch; throw("Input L_slices is missing") end
    if typeof(L) <: Int 1 else throw("Input L_slices is not an integer") end

    file = try params["filename"] catch; throw("Input filename is missing") end
    if typeof(file) <: String 1 else throw("Input filename is not a string") end

    niter = try params["niter"] catch; throw("Input niter is missing") end
    if typeof(niter) <: Int 1 else throw("Input niter is not an integer") end

    ntherm = try params["ntherm"] catch; throw("Input ntherm is missing") end
    if typeof(ntherm) <: Int 1 else throw("Input ntherm is not an integer") end

    nsweep = try params["nsweep"] catch; throw("Input nsweep is missing") end
    if typeof(nsweep) <: Int 1 else throw("Input nsweep is not an integer") end

    nmeas = try params["measurement"] catch; throw("Input measurement is missing") end
    if typeof(nmeas) <: Int 1 else throw("Input measurement is not an integer") end

    ndirty = try params["ndirty"] catch;; throw("Input ndirty is missing") end
    if typeof(ndirty) <: Int 1 else throw("Input ndirty is not an integer") end

    return U,J,L,beta,file,niter,ntherm,nsweep,nmeas,ndirty
end

function printIntro(U,J,L,beta,niter,ntherm,nsweep,nmeas,ndirty,nIsing)
    println("
    ======================================
    HIRCSH-FYE QUANTUM MONTE CARLO SOLVER
    ======================================
    Copyright © Muhammad Gaffar, 19.02.2020

    Start Running at = $(Dates.now())

    Physical Parameters:
    β   (Inverse Temperature)   = $beta eV
    U   (Coulomb Repulsion  )   = $U eV
    J   (Hund's Effect      )   = $J eV

    Numerical Parameters:
    L       (Number of Time slices   ) = $L
    nIsing  (Number of Ising states  ) = $nIsing
    niter   (Number of MC iteration  ) = $niter
    ntherm  (Number of Thermalization) = $ntherm
    nsweep  (Number of Ising Sweep   ) = $nsweep
    nmeas   (Number of Measurement   ) = $nmeas
    ndirty  (Number of Dirity update ) = $ndirty
    ")
end

function get_G0tau(Gt::GfimTime, L,beta)
    # this function read input Gtau, and make matrix LxL out of it
    # g0tau = g_{i,j} = Gtau(τi - τj), and its antiperodicity
    # Gtau(τ + β) = -Gtau(τ)

    # spline for new L
    tau = LinRange(0,beta,L)
    g0tau = zeros(length(Gt.orbs),length(Gt.orbs), L,L)
    for iorb in 1:length(Gt.orbs), jorb in 1:length(Gt.orbs)
        spl = Spline1D(Gt.mesh,real(Gt.data[iorb,jorb,:]))
        g0  = spl.(tau)
        # get G[l1,l2]
        for i in 1:L, j in 1:L
            g0tau[iorb,jorb,i,j] = (i-j>=0 ? -g0[i-j+1] : g0[L+i-j])
        end
    end
    return tau,g0tau
end

function getisingParams(U,J,nb)
    # now we want to make parameters as result of trotter decomposition
    # the Hamiltonian first is written as
    # H = ∑_i U_i * [(ni↑ * ni↓) - 0.5 *(ni↑ + ni↓)]
    # with trotter decompisition, it can be written as
    # H = 0.5 ∑_i ∑_j λ_i * S_i * fs_ij
    #--------------------------------------------------------------------------------
    # Example 3band model:
    #  ij  |pair(0,1)|        pair state        |  Ui       |f_ji| 0  1  2  3  4  5
    #-------------------------------------------------------|-------------------------
    #   0  |  0,1    |up,down;   0   ;   0   >  |  U+J      | 0  | 1 -1
    #   1  |  0,2    |  up   ;  up   ;   0   >  |  U-J      | 1  | 1    -1
    #   2  |  0,3    |  up   ; down  ;   0   >  |  U	    | 2  | 1       -1
    #   3  |  0,4    |  up   ;   0   ;  up   >  |  U-J      | 3  | 1          -1
    #   4  |  0,5    |  up   ;   0   ; down  >  |  U	    | 4  | 1             -1
    #   5  |  1,2    | down  ;  up   ;   0   >  |  U	    | 5  |    1 -1
    #   6  |  1,3    | down  ; down  ;   0   >  |  U-J      | 6  |    1    -1
    #   7  |  1,4    | down  ;   0   ;  up   >  |  U	    | 7  |    1       -1
    #   8  |  1,5    | down  ;   0   ; down  >  |  U-J      | 8  |    1          -1
    #   9  |  2,3    |   0   ;up,down;   0   >  |  U+J      | 9  |       1 -1
    #   10 |  2,4    |   0   ;  up   ;  up   >  |  U-J      | 10 |       1    -1
    #   11 |  2,5    |   0   ;  up   ; down  >  |  U	    | 11 |       1       -1
    #   12 |  3,4    |   0   ; down  ;  up   >  |  U	    | 12 |          1 -1
    #   13 |  3,5    |   0   ; down  ; down  >  |  U-J      | 13 |          1    -1
    #   14 |  4,5    |   0   ;   0   ;up,down>  |  U+J      | 14 |             1 -1
    # --------------------------------------------------------------------------

    # number of possible combination spins in the orbital
    # nf = nb_C_2
    nf = Int( nb * (nb - 1) / 2 )
    # number of kind interaction in every pair state.
    Ui = zeros(nf)
    # index of pair state
    pair = zeros(Int,nf,2)
    fs = zeros(Int,nb,nf)

    ij = 0
    for i in 1:(nb-1), j in (i+1):nb
        ij += 1
        Ui[ij] = U
        #what kind spin in state i and j, (up or dw)
        S_i = 2*(i%2)-1; S_j = 2*(j%2)-1
        # if there are |up,down> states, the interaction is U+J
        # elseif there are same spin and they are in nearest neigbour sites
        # then the interaction is U-J
        if (j==(i+1) && (i%2)==1) Ui[ij] += J
        elseif (S_i*S_j>0) Ui[ij] -= J end

        pair[ij,1] = i; pair[ij,2] = j
        fs[i,ij] = 1; fs[j,ij] =-1
    end
    return Ui,pair,fs
end

function startIsing(Ui, beta, L)
    nf = length(Ui)
    λ = zeros(nf); V = zeros(nf,L)
    # lambda parameter in trotter decomposition
    # λ = acosh(e^{-1/2*Δtau*U})
    for i in 1:nf λ[i] = acosh(exp(0.5*(beta/L)*Ui[i])) end
    # V parameter in trotter decompisition
    # V = λs
    for i in 1:nf
        V[i,:] = 2*rand(Bool,L) .- 1
        V[i,:] .*= λ[i]
    end
    return V
end

function cleanUpdate(vn,fs,gtau)
    # calculate interacting green function

    # allocation
    nf = size(vn,1); L  = size(vn,2);  nb = size(gtau,1)
    A = zeros(L,L); a = zeros(L)
    g = copy(gtau)

    # calculate A matrix
    # where A_{ij} = δ_{ij}*e^V_j - g0_{ij}*(e^V_j - 1)
    # numerically calculate this first a = (e^V-1)
    for i in 1:nb
        for l in 1:L
            sum = 0.0
            for j in 1:nf sum += fs[i,j]*vn[j,l] end
            a[l] = exp(sum) - 1.0
        end
        for l1 in 1:L
            for l2 in 1:L A[l1,l2] = -gtau[i,i,l1,l2] * a[l2] end
            A[l1,l1] += 1 + a[l1]
        end

        # now calculate g by solve A*g = g0, using LAPACK
        # in Julia it seems little faster than computing inverse g = A^{-1}g0
        g[i,i,:,:],_,_ = lapack.gesv!(A,real(gtau[i,i,:,:]))
    end
    return g
end

function detRatio(z,l,g,vn,pair)
    a    = zeros(2)
    a[1] = exp(-2*vn[z,l]) - 1
    a[2] = exp( 2*vn[z,l]) - 1
    pair_up = pair[z,1]
    pair_dw = pair[z,2]
    Det_up = 1 + (1 - g[pair_up,pair_up,l,l]) * a[1]
    Det_dw = 1 + (1 - g[pair_dw,pair_dw,l,l]) * a[2]
    return Det_up * Det_dw, a
end

function acceptMove!(g,vn,a,fs,pair, n_accept,ndirty, iz,il)
    vn[iz,il] *= -1 # flip-at state (z,l)

    x0 = zeros(size(vn,2)) # allocate x0 with size L # for blas.ger!
    x1 = zeros(size(vn,2)) # allocate x1 with size L # for blas.ger!
    n_accept += 1
    if (n_accept % ndirty == 0)
        g = cleanUpdate(vn,fs,g)
    else # dirty update
        for ip in 1:2
            p = pair[iz,ip]
            #prefactor
            b = a[ip] / (1 + a[ip]*(1-g[p,p,il,il]))

            # (g-1)_{l,il}
            x0 = g[p,p,:,il]
            x0[il] -= 1

            # g_{l2}
            x1 = g[p,p,il,:]

            # g[l1,l2] = g[l1,l2] + b * x0 * x1'
            g[p,p,:,:] = blas.ger!(b, x0,x1,g[p,p,:,:])
        end
    end
end



p = Dict("U" => 2.0,
         "J" => 0.5,
         "beta" => 16,
         "L_slices" => 64,
         "filename" => "test",
         "niter" => 1,
         "nsweep" => 10_000,
         "ntherm" => 20,
         "measurement" => 200,
         "ndirty" => 10)
g0iwn = setGiwn(("sup","sdw"),1024,16)
g = HF_Solve(g0iwn,p);
