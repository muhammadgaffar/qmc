using LinearAlgebra
using Dierckx
const lapack = LinearAlgebra.LAPACK
const blas   = LinearAlgebra.BLAS

import Dates

include("gf.jl")

function HF_Solve(G0iwn::GfimFreq,params::Dict)
    # PARAMETERS INITIALIZATION
    U,J,L,beta,file,niter,ntherm,nbins,nsweep,ndirty,binsize,mix = checkParams(params)
    if (beta == -1) beta = π / G0iwn.mesh[1] end # if beta is not set in parameter
    nb = length(G0iwn.orbs) # number of orbitals
    ## get interaction matrix, pair matrix, and spin location
    Ui,pair,fs = getisingParams(U,J, size(G0iwn.data,1))
    ## total of ising state
    nIsing = length(Ui)*L
    ## number Monte Carlo steps, warming up first then real step
    nwarmup = ntherm * nIsing
    nstep = nsweep * nbins * binsize * nIsing

    # initializes measurement
    Gtau  = zeros(nb,nb, L) # final Gtau after all Sweep
    Gtave = zeros(nb,nb, L) # G average over all measurement that are collected in bins
    occ   = zeros(nb)       # occupancy of each state
    db_occ = zeros(length(Ui)) # double occupancy
    giwn = copy(G0iwn) # for interacting green function in matsubara
    sig_iwn = copy(G0iwn)

    # PRINTOUT INTRODUCTION OF PROGRAMS
    printIntro(U,J,L,beta,nb,niter, ntherm,nstep+nwarmup,nsweep,ndirty,nIsing,nbins,binsize,mix)

    ## get random ising quantities over L slices and N orbs
    ## This quantitiy is V_{il} = λ_i * s_l, as result of hubbard stranovich
    Vs = startIsing(Ui,beta,L)

    # monte carlo loop
    ## warming up iteration first, we do not collect measurement in warming up
    for iter in 1:niter
        ## Fourier transform Giwn and spline
        Gtau = invFourier(G0iwn)
        Gtau = spline2L(Gtau, L)

        ## (Re)-initialize samplings
        G_ave = zeros(nb,nb, L)
        G_sqr = zeros(nb,nb, L)
        stored = 0
        nbins_stored = 0
        naccept = 0

        ## Make LxL matrix out of Gtau
        ## using spline interpolation for new tau
        Gt = get_Gt_mat(Gtau, L,beta)

        ## Clean update first, calculate g from A*g_new = g0
        g = cleanUpdate(Vs,fs,Gt)

        # print progress
        printProgress(iter)

        naccept = 0
        # start sweep iteration
        for istep in 1:(nstep+nwarmup)
            #visited_avg = istep / (L*nIsing)
            # choose random site in ising state
            r_site = floor(Int, rand()*nIsing)
            l = ceil(Int, (r_site % L)); if (l == 0) l = 1 end # which time slices
            z = floor(Int, (r_site / L) ) + 1 #which state

            P,a = detRatio(z,l,g,Vs,pair) # metropolis min[ρ'/ρ,1]

            #accept Move?
            if (P > rand())
                naccept += 1
                acceptMove!(g,Vs,naccept, a,fs,pair,ndirty,z, l)
            end

            # collect measurement after nsweep flipping
            if ((istep - nwarmup) % (nsweep * nIsing)) == 0
                Gtau, Gtave, G_ave, G_sqr,
                db_occ, stored, nbins_stored = saveMeasurement(g,Gtau,
                                nb,length(Ui),L,pair,
                                Gtave,G_ave,G_sqr,
                                nbins_stored,stored,binsize,
                                db_occ)
            end
        end
        # get Result
        Gtau, Gt_deviation, occ, db_occ = getResult(Gtau, G_ave,G_sqr,db_occ, stored,nbins_stored)

        #print result
        printResult(occ,db_occ,naccept,nstep,nwarmup)

        # fourier transform to matsubara
        Fourier!(Gtau,giwn)

        # get Self energy
        sig_iwn = getSelfEnergy(G0iwn, giwn)

        # self-consistency
        G0_new = inv(iwn - 0.25*giwn)
        G0iwn = (1-mix) * G0iwn + mix * G0_new
    end

    printEnd()

    # only return interacting giwn and sigma_iwn, rest in hdf5
    return inv(inv(G0iwn) - sig_iwn), sig_iwn, Gtau
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
    if typeof(beta) <: Real 1 else throw("Input beta is not a real number") end

    L = try params["L_slices"] catch; throw("Input L_slices is missing") end
    if typeof(L) <: Int 1 else throw("Input L_slices is not an integer") end

    file = try params["filename"] catch; throw("Input filename is missing") end
    if typeof(file) <: String 1 else throw("Input filename is not a string") end

    niter = try params["niter"] catch; throw("Input niter is missing") end
    if typeof(niter) <: Int 1 else throw("Input niter is not an integer") end

    ntherm = try params["ntherm"] catch; throw("Input ntherm is missing") end
    if typeof(ntherm) <: Int 1 else throw("Input ntherm is not an integer") end

    nbins = try params["nbins"] catch; throw("Input nbins is missing") end
    if typeof(nbins) <: Int 1 else throw("Input nbins is not an integer") end

    nsweep = try params["nsweep"] catch; throw("Input nsweep is missing") end
    if typeof(nsweep) <: Int 1 else throw("Input nweep is not an integer") end

    ndirty = try params["ndirty"] catch;; throw("Input ndirty is missing") end
    if typeof(ndirty) <: Int 1 else throw("Input ndirty is not an integer") end

    binsize = try params["binsize"] catch;; throw("Input binsize is missing") end
    if typeof(binsize) <: Int 1 else throw("Input binsize is not an integer") end

    mix = try params["mixing"] catch;; throw("Input mixing is missing") end
    if typeof(mix) <: Real 1 else throw("Input binsize is not a real number") end

    return U,J,L,beta,file,niter,ntherm,nbins,nsweep,ndirty,binsize,mix
end

function printIntro(U,J,L,beta,nb,niter, ntherm,nstep,nsweep,ndirty,nIsing,nbins,binsize,mix)
    println("
    ======================================
    HIRCSH-FYE QUANTUM MONTE CARLO SOLVER
    ======================================
    Copyright © Muhammad Gaffar, 19.02.2020

    Start Running at = $(Dates.now())

    Physical Parameters:
    β   (Inverse Temperature         ) = $beta eV
    U   (Density-density Interaction ) = $U eV
    J   (Exchange coupling           ) = $J eV
    Nb  (Number of Orbital SU(2)     ) = $nb

    Numerical Parameters:
    L       (Number of Time slices   ) = $L
    nIsing  (Number of Ising states  ) = $nIsing
    niter   (Number of MC iteration  ) = $niter
    ntherm  (Number of Thermalization) = $ntherm
    nsweep  (Number of Ising Sweep   ) = $nsweep
    nstep   (Total of MC move        ) = $nstep
    ndirty  (Number of Dirty update  ) = $ndirty
    nbins   (Bins to collect         ) = $nbins
    binsize (Grouped measurement     ) = $binsize
    mix     (Mixing Parameter        ) = $mix
    ")
end

function printProgress(iter)
    println("
    # Iteration #$iter =============================
    Monte Carlo Sweep Starting...
    ")
end

function printResult(occ,db_occ,naccept,nstep,nwarmup)
    println("
    Number of Occupation = $occ
    Double Occupancy     = $db_occ
    Acceptance Rate      = $(naccept / (nstep + nwarmup))

    Saving measurement to file hdf5.
    ")
end

function printEnd()
    println("
    End Running at = $(Dates.now())
    ")
end


function spline2L(Gt::GfimTime, L)
    beta = Gt.mesh[end]
    tau = LinRange(0,beta,L)
    gt = zeros(length(Gt.orbs),length(Gt.orbs), L)

    for iorb in 1:length(Gt.orbs), jorb in 1:length(Gt.orbs)
        spl = Spline1D(Gt.mesh,real(Gt.data[iorb,jorb,:]))
        gt[iorb,jorb,:]  = spl.(tau)
    end

    GfimTime{length(Gt.orbs)}(Gt.orbs,tau,gt)
end

function get_Gt_mat(Gt::GfimTime, L,beta)
    # this function read input Gtau, and make matrix LxL out of it
    # g0tau = g_{i,j} = Gtau(τi - τj), and its antiperodicity
    # Gtau(τ + β) = -Gtau(τ)

    # spline for new L
    g0tau = zeros(length(Gt.orbs),length(Gt.orbs), L,L)
    for iorb in 1:length(Gt.orbs), jorb in 1:length(Gt.orbs)
        g0 = Gt.data[iorb,jorb,:]
        # get G[l1,l2]
        for i in 1:L, j in 1:L
            g0tau[iorb,jorb,i,j] = (i-j >= 0 ? -g0[i-j+1] : g0[L+i-j+1])
        end
    end
    return g0tau
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
    V = zeros(nf,L)
    # parameter in hubbard stranovich
    for i in 1:nf
        λ = acosh(exp(0.5*(beta/L)*Ui[i]))
        V[i,:] = 2*rand(Bool,L) .- 1
        V[i,:] .*= λ
    end
    return V
end

function cleanUpdate(vn,fs,gtau)
    # calculate interacting green function

    # allocation
    nf = size(vn,1); nb = size(gtau,1); L  = size(vn,2);
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

        # now calculate g_new by solve A*g_new = g_old, using LAPACK linear equation solver
        # in Julia it seems little faster than computing inverse g = A^{-1}g0
        g[i,i,:,:],_,_ = lapack.gesv!(A,real(gtau[i,i,:,:]))
    end
    return g
end

function detRatio(z,l,g,vn,pair)
    a    = zeros(2)
    # check probability of flipping at (z,l) s_new = -s_old
    a[1] = exp(-2*vn[z,l]) - 1 # e^{ λ(s_new - s_old)} - 1 = e^{-2λs_old} - 1 = e^{-2V} - 1
    a[2] = exp( 2*vn[z,l]) - 1 # e^{-λ(s_new - s_old)} - 1 = e^{ 2λs_old} - 1 = e^{ 2V} - 1
    pair_up = pair[z,1] # spin up orbital posisition
    pair_dw = pair[z,2] # spin down orbital position
    # now calculate acceptance probability ρ' / ρ
    Det_up = 1 + (1 - g[pair_up,pair_up,l,l]) * a[1]
    Det_dw = 1 + (1 - g[pair_dw,pair_dw,l,l]) * a[2]
    return Det_up * Det_dw, a
end

function acceptMove!(g,vn,n_accept, a,fs,pair,ndirty,iz,il)
    vn[iz,il] *= -1 # flip-at state (z,l)

    x0 = zeros(size(vn,2)) # allocate x0 with size L # for blas.ger!
    x1 = zeros(size(vn,2)) # allocate x1 with size L # for blas.ger!
    if (n_accept % ndirty == 0)
        g = cleanUpdate(vn,fs,g)
    else # dirty update
        for ip in 1:2
            p = pair[iz,ip]
            #prefactor (scalar)
            b = a[ip] / (1 + a[ip]*(1-g[p,p,il,il]))

            # g_{l,il} - δ_{il} (vector over l1)
            x0 = g[p,p,:,il]
            x0[il] -= 1

            # g_{il,l} (vector over l2)
            x1 = g[p,p,il,:]

            # g[l1,l2] = g[l1,l2] + b * x0 * x1' (outer product operation)
            g[p,p,:,:] = blas.ger!(b, x0,x1,g[p,p,:,:])
        end
    end
end

function saveMeasurement(g,Gtau, nb,nf,L,pair, Gtave,G_ave,G_sqr,
                        nbins_stored,stored,binsize, db_occ)
    # save how many measured G in bins already
    stored += 1
    # measure new G
    Gtau.data = zeros(size(Gtau.data)...)
    for i in 1:nb
        for l1 in 1:L, l2 in 1:L
            if (l1>=l2) Gtau.data[i,i,l1-l2+1] += -g[i,i,l1,l2] # antiperodic boundary condition
            else Gtau.data[i,i,L+l1-l2+1] += g[i,i,l1,l2] end # -sign convention
        end
    end
    Gtau.data .*= (1/L) # normalization because there are L^2 pairs
    Gtau.data[:,:,end] = -Gtau.data[:,:,end] .- 1

    # store measured G in bin => Gtave
    for i in 1:nb, l in 1:L Gtave[i,i,l] += Gtau.data[i,i,l] end

    # if bin is full, give result by averaging G in bins
    if (stored % binsize == 0)
        for i in 1:nb, l in 1:L
            G_ave[i,i,l] += Gtave[i,i,l] / binsize
            G_sqr[i,i,l] += G_ave[i,i,l]^2
        end
        Gtave .= 0 # go to next bin
        nbins_stored += 1 # go to next bin
    end

    # also, store double occupancy
    # double occupancy is = <n↑*n↓>
    nnt = zeros(nf)
    for i in 1:nf, l in 1:L
        p_up = pair[i,1]
        p_dw = pair[i,2]
        nnt[i] += (1-g[p_up,p_up,l,l])*(1-g[p_dw,p_dw,l,l])
    end
    nnt *= (1 / L)
    db_occ += nnt

    return Gtau, Gtave, G_ave, G_sqr, db_occ, stored, nbins_stored
end

function getResult(Gt, G_ave,G_sqr,db_occ, stored, nbins_stored)
    # now full average over all measurement in stored bins
    # get Gt, deviation of Gt, occupation, and double occupancy
    occ = zeros(size(Gt.data,1))
    Gt_deviation = zeros(size(Gt.data)...)

    # Gt and deviation of Gt
    for i in 1:size(Gt.data,1), l in 1:size(Gt.data,3)
        Gf  = G_ave[i,i,l] / nbins_stored
        G2f = G_sqr[i,i,l] / nbins_stored
        Gt_deviation[i,i,l] = sqrt( (G2f - Gf^2) / (nbins_stored-1) )
        Gt.data[i,i,l] = Gf
    end
    # occupation and double occupancy
    for i in 1:size(Gt.data,1)
        occ[i] = G_ave[i,i,1] / nbins_stored + 1.0
    end
    db_occ = db_occ / stored
    return Gt, Gt_deviation, occ, db_occ
end

function getSelfEnergy(G0iwn,Giwn)
    Sigma = inv(G0iwn) - inv(Giwn)

    # at infinity , sigma must go to zero
    for iorb in 1:length(G0iwn.orbs), jorb in 1:length(G0iwn.orbs)
        Sigma.data[iorb,jorb,:] -= 1im*(imag(Sigma.data[iorb,jorb,end]) / Sigma.mesh[end]) .* Sigma.mesh
    end
    return Sigma
end

p = Dict("U" => 0.2,
         "J" => 0.0,
         "beta" => 16,
         "L_slices" => 64,
         "filename" => "test",
         "niter" => 1,
         "nsweep" => 2,
         "ntherm" => 100,
         "nbins" => 10,
         "ndirty" => 100,
         "binsize" => 1000,
         "mixing" => 0.5)
g0iwn = setGiwn((1,2),1024,16);
setfromToy!(g0iwn,:bethe);
giwn, sigwn, gt = HF_Solve(g0iwn,p);


plot(gt.mesh,real(gt.data[1,1,:]))
plot(g0iwn.mesh,imag(g0iwn.data[1,1,:]))
plot(giwn.mesh,imag(giwn.data[1,1,:]))
xlims!(0,50)

giw = setfromPade(giwn,nw=500,wrange=(-5,5),npoints=200)
plot(giw.mesh,-imag(giw.data[2,2,:]))
