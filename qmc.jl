function HF_Solve(G0iwn::GfimFreq,params::Dict)
    # check parameters, and initialization
    U,J,L,beta,file = checkParams(params)
    if (beta == -1) beta = π / Giwn.mesh[1] end # if beta is not set in parameter
    Ui,pair,fs = getisingParams(U,J,size(Giwn.data,1))

    # PRINTOUT INTRODUCTION OF PROGRAMS
    printIntro()

    # INTIALIZATION
    ## Fourier transform G0iwn
    G0tau = invFourier(G0iwn)
    ## Make LxL matrix out of Gtau
    ## using spline interpolation for new tau
    G0tau = get_G0tau(G0tau)
    ## get random ising quantities over L slices and N orbs
    ## This quantities in is V in e^{-V}, this is result of hubbard-stranovich
    Vs = startIsing(U,beta,L)

    # warming up

    # monte carlo loop

    # fitting

    # measurement

    # print result to HDF5

    # only return interacting giwn and sigma_iwn, rest in hdf5
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
end

function get_G0tau_LL(Gt::GfimTime)
    L = length(Gt.mesh)
    G0tau_LL = zeros(eltype(Gt.data), length(Gt.orbs),length(Gt.orbs), L,L)
    for iorb in 1:length(Gt.orbs), jorb in 1:length(Gt.orbs)
        for i in 1:L, j in 1:L
            G0tau_LL[iorb,jorb,i,j] = (i-j>=0 ? -Gt.data[iorb,jorb,i-j+1] : Gt.data[iorb,jorb,L+i-j])
        end
    end
    return G0tau_LL
end

function getisingParams(U,J,nb)
    nf = Int( nb * (nb - 1) / 2 )
    Ui = zeros(nf)
    pair = zeros(Int,nf,2)
    fs = zeros(Int,nb,nf)

    ij = 0
    for i in 1:(nb-1), j in (i+1):nb
        ij += 1
        Ui[ij] = U
        S_i = 2*(i%2)-1; S_j = 2*(j%2)-1
        if (j==(i+1) && (i%2)==1) Ui[ij] += J
        elseif (S_i*S_j>0) Ui[ij] -= J end
        pair[ij,1] = i; pair[ij,2] = j
        fs[i,ij] = 1; fs[j,ij] =-1
    end
    return Ui,pair,fs
end

function startIsing(U, beta, L)
    Nf = 2*Nband
    Nf = Int( Nf * (Nf - 1) / 2 )

    λ = zeros(Nf)
    for i in 1:Nf
        λ[i] = acosh(exp(0.5*(beta/L)*Ui[i]))
    end

    vn = zeros(Nf,L)
    for i in 1:Nf
        vn[i,:] = 2*rand(Bool,L) - 1
        vn[i,:] .*= λ[i]
    end

    return vn
end
