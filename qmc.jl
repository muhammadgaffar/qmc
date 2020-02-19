using LinearAlgebra
using Dierckx
const lapack = LinearAlgebra.LAPACK

include("gf.jl")

function HF_Solve(G0iwn::GfimFreq,params::Dict)
    # check parameters, and initialization
    U,J,L,beta,file = checkParams(params)
    if (beta == -1) beta = π / G0iwn.mesh[1] end # if beta is not set in parameter
    Ui,pair,fs = getisingParams(U,J,size(G0iwn.data,1))

    # PRINTOUT INTRODUCTION OF PROGRAMS
    printIntro()

    # INTIALIZATION
    ## Fourier transform G0iwn
    G0tau = invFourier(G0iwn)
    ## Make LxL matrix out of Gtau
    ## using spline interpolation for new tau
    tau, g0tau = get_G0tau(G0tau, L,beta)
    ## get random ising quantities over L slices and N orbs
    ## This quantities in is V in e^{-V}, this is result of hubbard-stranovich
    Vs = startIsing(U,beta,L)
    ## make clean update first, calculate g from A*g = g0
    g = cleanUpdate(Vs,fs,g0tau)

    # monte carlo loop
    ## warming up iteration first, we do not collect measurement in warming up

    # fitting

    # measurement

    # print result to HDF5

    # only return interacting giwn and sigma_iwn, rest in hdf5
    return pair,fs
end

function checkParams(params::Dict)
    U = try params["U"] catch; throw("Input U is missing") end
    if typeof(U) <: Real 1 else throw("Input U is not a real number") end

    J = try params["J"] catch; throw("Input J is missing") end
    if typeof(J) <: Real 1 else throw("Input J is not a real number") end

    beta = try params["beta"]
        catch
             @warn("Input beta is missing, program will use beta from G0(iωn)")
             -1
        end
    if typeof(beta) <: Int 1 else throw("Input beta is not a real number") end

    L = try params["L_slices"] catch; throw("Input L_slices is missing") end
    if typeof(L) <: Int 1 else throw("Input L_slices is not an integer") end

    file = try params["filename"] catch; throw("Input filename is missing") end
    if typeof(file) <: String 1 else throw("Input filename is not a string") end

    return U,J,L,beta,file
end

function printIntro()
    println("
    ======================================
    HIRCSH-FYE QUANTUM MONTE CARLO SOLVER
    ======================================
    by: Muhammad Gaffar
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

        # now calculate g by solve A*g = g0
        # in Julia it seems little faster than computing inverse g = A^{-1}g0
        g[i,i,:,:],_,_ = lapack.gesv!(A,real(gtau[i,i,:,:]))
    end
    return g
end


p = Dict("U" => 2.0,
         "J" => 0.5,
         "beta" => 100,
         "L_slices" => 400,
         "filename" => "test")
g0iwn = setGiwn(("sup","sdw","pup","pdw","2","3"),1024,100)
p,f = HF_Solve(g0iwn,p);
