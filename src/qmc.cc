#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <list>
#include "sutil.h"
#include "sfunction.h"
#include "random.h"
#include "complex.h"

using namespace std;

bool ReadG0(const string& filename, function2D<dcomplex>& G0, const vector<double>& iom)
{
  int Nd = G0.size_N();
  int Niom = G0.size_Nd();
  if (!Nd || !Niom) {cerr<<"G0 not allocated!"<<endl; return false;}

  ifstream inputf(filename.c_str());
  // Computes number of columns in g0 file
  istream input(inputf.rdbuf());
  input.seekg(0,ios::beg);
  if (!input) {
    cerr << "Can't open input file: " << filename << endl;
    return false;
  }
  string str;
  int n=0;
  getline(input,str); n++;
  istringstream oneline(str);
  int m=0; double t;
  while (oneline){oneline>>t; m++;}
  m--;
  while (input){ getline(input,str); n++;}
  n--;
  clog << filename << ": Number of rows:    "<< n <<endl;
  clog << filename << ": Number of columns: "<< m <<endl;
  if (m<2){cerr<<"Not enough rows in "<<filename<<endl; return false;}

  inputf.seekg(0,ios::beg);

  function2D<dcomplex> g0s(Nd,n);
  int in=0; double io;
  while (inputf>>io){
    for (int im=0; im<min((m-1)/2,Nd); im++) inputf>>g0s[im][in];
    for (int im=(m-1)/2; im<Nd; im++) g0s[im][in] = g0s[im-1][in];
    in++;
  }

  for (int l=0; l<iom.size(); l++){// Linear interpolation to the "new mesh" in imaginary frequency
    double a = static_cast<double>(g0s.size_Nd())/(iom.size());
    int i0 = static_cast<int>(l*a);
    for (int i=0; i<Nd; i++)
      G0(i,l) = (i0<g0s.size_Nd()-1) ?  g0s[i][i0]+(g0s[i][i0+1]-g0s[i][i0])*(l*a-i0) : g0s[i][g0s.size_Nd()-1];	// linear interpolation
  }
  ofstream outg0("g0.start");
  for (int l=0; l<iom.size(); l++){
    outg0<<setw(3)<<iom[l]<<" ";
    for (int i=0; i<Nd; i++) outg0<<setw(12)<<G0(i,l)<<" ";
    outg0<<endl;
  }
  return true;
}


class QMC{
  int Nd, Nf, L; // Number of GF, number of ising spins and number of time-slices
  double dtau;
  function2D<int> pair, fs;    // matrices for ising update (check comments in "GetIsingQuantities")
  function1D<double> xlam;     // ising lambdas
  function2D<double> vn;       // lambda_i * s_{il} where s_{il} is ising spin for time slices l and band i
  vector<function2D<double> > g0; // g0 which depend on ising configuration
  int ndirty, accepted;                 // number of accepted moves
  function1D<double> x0, x1;   // temporary arrays for faster rank1 update
  function2D<double> Gt, Gtave, G_ave, G_sqr; // average g, g^2 and standard deviation
  function2D<double> G_ave_test;
  function1D<double> nnt, nn;
  int stored, nbins_stored;
public:
  QMC(int Nd_, int L_, double dtau_, double U, double J, int ndirty_);
  void SetIsingQuantities(double U, double J, function1D<double>& Ui);
  template <class Rand>
  bool GetStartIsingConfiguration(const string& filename, Rand& rand);
  void SetG0(const function2D<double>& G0);
  void CleanUpdate(vector<function2D<double> >& g);
  void AcceptMove(int iz, int il, vector<function2D<double> >& g, const function1D<double>& a, int isweep);
  double DetRatio(int iz, int il, const vector<function2D<double> >& g, function1D<double>& a);
  void SaveMeasurement(const vector<function2D<double> >& g, int binsize);
  void PrintResult(ostream& out);
  void PrintIsingConfiguration(const string& filename);
  void GiveResult(function2D<double>& Gtau, function2D<double>& dGtau, function1D<double>& nf);

  int NumIsingSpins(){return Nf;}
  int Accepted(){return accepted;}
  int Pair(int iz, int ip){return pair[iz][ip];}
};

// Construct initializes memory and g0
QMC::QMC(int Nd_, int L_, double dtau_, double U, double J, int ndirty_) :
  Nd(Nd_), L(L_), dtau(dtau_), g0(Nd), ndirty(ndirty_), accepted(0), x0(L), x1(L),
  Gt(Nd,L+1), Gtave(Nd,L+1), G_ave(Nd,L+1), G_sqr(Nd,L+1), G_ave_test(Nd,L+1), stored(0), nbins_stored(0)
{
  function1D<double> Ui;        // Coulomb interaction for ising fields
  SetIsingQuantities(U, J, Ui); // Calculates ising U and f_ij and pairs
  Nf = Ui.size();               // Number of ising spins
  xlam.resize(Nf);              // Ising lambdas
  for (int i=0; i<xlam.size(); i++) xlam[i] = acosh(exp(0.5*dtau*Ui[i])); // ising lambdas are set
  vn.resize(Nf,L);             // ising spin storage is allocated but not jet set. It will be later read from file if it exists.

  G_ave=0; G_sqr=0; Gtave=0; G_ave_test=0;
  nn.resize(Nf); nn=0;
  for (int i=0; i<g0.size(); i++) g0[i].resize(L,L);
}

void QMC::SetG0(const function2D<double>& G0)
{
  // Initialization of ising g0(tau,tau') from G0(tau-tau'). Must be antiperiodic in time.
  for (int k=0; k<Nd; k++)
    for (int i=0; i<L; i++)
      for (int j=0; j<L; j++)
	g0[k](i,j) = (i-j>=0)?  -G0(k,i-j) : G0(k,L+i-j); // antiperiodic boundary conditions for fermions

  // reinitializes sampling
  nbins_stored=0;
  stored=0;
  G_ave=0;
  G_sqr=0;
  accepted=0;
}

void QMC::SetIsingQuantities(double U, double J, function1D<double>& Ui)
{ // pair gives first and second member of the pair that corresponds to each ising spin
  // f_ij is used to write Trotter decomposition simply as sum over all densities
  // and f_ij takes care of grouping of the pairs and relative sign.
  // Troter decomposition for interaction
  //             \sum_i U_i*(n[pair_i^0]*n[pair_i^1]-0.5(n[pair_i^0]+n[pair_i^1]))
  // is simply   \sum_j \lambda_i*S_i*n_j*f_ij
  //--------------------------------------------------------------------------------
  // Example 3band model:
  //  ij  |pair(0,1)|        pair state        |  Ui       |f_ji| 0  1  2  3  4  5
  //-------------------------------------------------------|-------------------------
  //   0  |  0,1    |up,down;   0   ;   0   >  |  U+J      | 0  | 1 -1
  //   1  |  0,2    |  up   ;  up   ;   0   >  |  U-J      | 1  | 1    -1
  //   2  |  0,3    |  up   ; down  ;   0   >  |  U	   | 2  | 1       -1
  //   3  |  0,4    |  up   ;   0   ;  up   >  |  U-J      | 3  | 1          -1
  //   4  |  0,5    |  up   ;   0   ; down  >  |  U	   | 4  | 1             -1
  //   5  |  1,2    | down  ;  up   ;   0   >  |  U	   | 5  |    1 -1
  //   6  |  1,3    | down  ; down  ;   0   >  |  U-J      | 6  |    1    -1
  //   7  |  1,4    | down  ;   0   ;  up   >  |  U	   | 7  |    1       -1
  //   8  |  1,5    | down  ;   0   ; down  >  |  U-J      | 8  |    1          -1
  //   9  |  2,3    |   0   ;up,down;   0   >  |  U+J      | 9  |       1 -1
  //   10 |  2,4    |   0   ;  up   ;  up   >  |  U-J      | 10 |       1    -1
  //   11 |  2,5    |   0   ;  up   ; down  >  |  U	   | 11 |       1       -1
  //   12 |  3,4    |   0   ; down  ;  up   >  |  U	   | 12 |          1 -1
  //   13 |  3,5    |   0   ; down  ; down  >  |  U-J      | 13 |          1    -1
  //   14 |  4,5    |   0   ;   0   ;up,down>  |  U+J      | 14 |             1 -1
  // --------------------------------------------------------------------------
  int Nf = static_cast<int>(Nd*(Nd-1)/2.);// Number of pairs
  Ui.resize(Nf);
  pair.resize(Nf,2);
  fs.resize(Nd,Nf);
  int ij=0; // index of the pair (i,j)
  for (int i=0; i<Nd-1; i++){
    for (int j=i+1; j<Nd; j++){
      Ui[ij] = U;
      int S_i = 1-2*(i%2); // z component of spin
      int S_j = 1-2*(j%2);
      if (j==i+1 && i%2==0) Ui[ij] += J;   // doubly occupied site
      else if (S_i*S_j>0)   Ui[ij] -= J;   // Hund's coupling prefers aligned spins
      pair(ij,0) = i;
      pair(ij,1) = j;
      fs(i,ij) = 1;
      fs(j,ij) =-1;
      ij++;
    }
  }
}


inline double QMC::DetRatio(int iz, int il, const vector<function2D<double> >& g, function1D<double>& a)
{// One of the most important subroutines in QMC.
  // Computes the probability for the MC step - the ration of determinants between ising configurations.
  // il -- time slice index
  // iz -- band index
  a[0] = exp(-2*vn(iz,il)) - 1;
  a[1] = exp( 2*vn(iz,il)) - 1;
  double Det_up = 1+(1-g[pair[iz][0]](il,il))*a[0];
  double Det_do = 1+(1-g[pair[iz][1]](il,il))*a[1];
  double Det = Det_up*Det_do;
  return Det;
}

void QMC::AcceptMove(int iz, int il, vector<function2D<double> >& g, const function1D<double>& a, int isweep)
{// Most important subroutine in QMC.
 // Updates the quantities (green's function) since the new ising configuration is accepted.
 // This function needs most of the time and should be heavily optimized! Most of the time is currently spend in blas
 // routine daxpy when doing rank1 update.
  vn(iz,il) *= -1; // perform spin-flip
  accepted++;      // accepted move
  if (accepted%ndirty==0) CleanUpdate(g);
  else{ // dirty update
    for (int ip=0; ip<2; ip++){ // both members of pair need to be updated
      int p = pair[iz][ip];     // member of the pair
      // prefactor b = a/(1+a(1-g_pp))
      double b = a[ip]/(1+a[ip]*(1-g[p](il,il)));

      for (int l=0; l<L; l++) // this computes (g-1)_{l,il}
	x0[l] = g[p](l,il);   // first we set x0[l] = g_{l,il}
      x0[il] -= 1;            // now be sutract delta_{l,il} to get x0[l] = g_{l,il}-delta_{l,il}

      x1 = g[p][il];
      // Here we calculate b*(g-1) x g where x is tensor product
      g[p].Add_rank_update(x0,x1,b);// tensor product (g-1)_{l,il} x g_{il,l'}
    }
  }
}

void QMC::CleanUpdate(vector<function2D<double> >& g)
{// Goes over all ising spins and recalculates g from g0 according to the current ising configuration.
  int Nf = vn.size_N();
  int L  = vn.size_Nd();
  int Nd = g0.size();
  static function2D<double> A(L,L), B(L,L);
  static function1D<double> a(L);
  static function1D<int> ipiv(L);

  for (int i=0; i<Nd; i++){
    // a = e^V-1
    for (int l=0; l<L; l++){
      double sum=0;
      for (int j=0; j<Nf; j++) sum += fs(i,j)*vn(j,l);
      a[l] = exp(sum)-1;
    }
    // Matrix A = 1 + (1-g)(e^V-1) == (1+a)*I - g*a
    for (int l0=0; l0<L; l0++){
      for (int l1=0; l1<L; l1++) A(l0,l1) = -g0[i](l0,l1)*a[l1];
      A(l0,l0) += 1 + a[l0];
    }
    // Solves system of linear equations A*g0[i]=g[i]. This is faster than computing inverse.
    for (int l0=0; l0<L; l0++) for (int l1=0; l1<L; l1++) B(l0,l1) = g0[i](l1,l0); // Needs to be tranposed due to Fortran convention
    SolveSOLA(A,B,ipiv);
    for (int l0=0; l0<L; l0++) for (int l1=0; l1<L; l1++) g[i](l0,l1) = B(l1,l0); // Needs to be tranposed due to Fortran convention
  }
}

template <class Rand>
bool QMC::GetStartIsingConfiguration(const string& filename, Rand& rand)
{// If input file exists, continue from a "good" ising configuration. Otherwise start from scratch.
  bool scratch=true;
  ifstream inpIs(filename.c_str());
  if (inpIs){ // file exists
    int iws;
    list<int> lising;
    while(inpIs>>iws) lising.push_back(iws);
    clog<<"Found starting ising configuration from file "<<filename<<" Number of ising spins is "<<lising.size()<<endl;
    if (lising.size()==L*Nf){// If number of ising spings does not fit, they are probabbly useless
      int ii=0;
      for (list<int>::const_iterator it=lising.begin(); it!=lising.end(); it++,ii++) vn(ii/L, ii%L) = (*it)*xlam[ii/L];
      clog<<"Successfully loaded ising configuration. "<<endl;
      scratch=false;
    }
  }
  if (scratch){ //Otherwise start from scratch.
    for (int i=0; i<Nf; i++)
      for (int l=0; l<L; l++)
	vn(i,l) = (rand()>0.5) ? -xlam[i] : xlam[i];
  }
  return !scratch;
}

void QMC::PrintIsingConfiguration(const string& filename)
{
  ofstream outIs(filename.c_str());
  for (int i=0; i<Nf; i++) for (int l=0; l<L; l++) outIs<<setw(4)<<((vn(i,l)>0) ? 1 : -1);
}

void QMC::SaveMeasurement(const vector<function2D<double> >& g, int binsize)
{
  static int first_time=true;
  stored++;
  // Saving Green's function
  Gt=0;
  for(int i=0; i<g.size(); i++){
    for (int l0=0; l0<g[i].size_N(); l0++){
      for (int l1=0; l1<g[i].size_Nd(); l1++){
	if (l0>=l1) Gt[i][l0-l1] += -g[i](l0,l1); // antiperiodic boundary conditions and
	else Gt[i][L+l0-l1] += g[i](l0,l1);       // -isgn convention in QMC community
      }
    }
  }
  Gt*=(1./L); // normalization because there were L^2 pairs for L time slices.
  for (int i=0; i<Gt.size_N(); i++) Gt[i][L] = -Gt[i][0]-1.; // Sets the G(beta) due to analytical knowledge

  // stores into current bin
  for(int i=0; i<Gtave.size_N(); i++) for (int l=0; l<Gtave.size_Nd(); l++) Gtave[i][l] += Gt[i][l];
  for(int i=0; i<Gtave.size_N(); i++) for (int l=0; l<Gtave.size_Nd(); l++) G_ave_test[i][l] += Gt[i][l];

  if (stored%binsize==0){
    // bin is full. Results should be stored
    for(int i=0; i<G_ave.size_N(); i++) for (int l=0; l<G_ave.size_Nd(); l++) G_ave[i][l] += Gtave[i][l]/binsize;
    for(int i=0; i<G_sqr.size_N(); i++) for (int l=0; l<G_sqr.size_Nd(); l++) G_sqr[i][l] += sqr(Gtave[i][l]/binsize);
    Gtave = 0;
    nbins_stored++;
    if (first_time) { clog<<"The bin is completed and number of bins is currently "<<nbins_stored<<" "; first_time=false;}
    else clog<<nbins_stored<<" ";
  }

  // Saving double occupancy
  nnt.resize(Nf);
  nnt=0; // double occupancy can be calculated just like average of n*n over ising configurations
  for (int i=0; i<Nf; i++)// the reason is that the action is quadratic in the representation of ising spins (non-interacting particles)
    for (int l=0; l<L; l++)
      nnt[i] += (1-g[pair[i][0]](l,l))*(1-g[pair[i][1]](l,l));
  nnt*=(1./L);
  nn += nnt;
}

void QMC::GiveResult(function2D<double>& Gtau, function2D<double>& dGtau, function1D<double>& nf)
{
  for(int i=0; i<G_ave.size_N(); i++){
    for (int l=0; l<G_ave.size_Nd(); l++){
      double Gf = G_ave[i][l]/nbins_stored;
      double G2f = G_sqr[i][l]/nbins_stored;
      double sigma = sqrt((G2f-sqr(Gf))/(nbins_stored-1));
      Gtau[i][l] = Gf;
      dGtau[i][l] = sigma;
    }
  }
  for (int i=0; i<Nd; i++) nf[i] = G_ave[i][0]/nbins_stored+1.0;
}

void QMC::PrintResult(ostream& out)
{
  out.precision(12);
  out<<"# nf=";
  for (int i=0; i<Nd; i++) out<<G_ave[i][0]/nbins_stored+1.0<<" ";
  out<<endl;
  out<<"# double-occupancy=";
  for (int i=0; i<Nf; i++) out<<"("<<pair[i][0]<<","<<pair[i][1]<<")="<<nn[i]/stored<<" ";
  out<<endl;
  for (int l=0; l<G_ave.size_Nd(); l++){
    out<<setw(12)<<l*dtau<<" ";
    for(int i=0; i<G_ave.size_N(); i++){
      double Gf = G_ave[i][l]/nbins_stored;
      double G2f = G_sqr[i][l]/nbins_stored;
      double sigma = sqrt((G2f-sqr(Gf))/(nbins_stored-1));
      out<<setw(20)<<Gf<<" "<<setw(20)<<sigma<<"  ";
    }
    out<<endl;
  }
}

class InclMore{
public:
  bool Q;
  double fct;
  InclMore(bool Q_, double fct_) : Q(Q_), fct(fct_){};
};

class SIVEDE{
  vector<double> tau, om, iom;
  int mMax;
  function2D<double> Ap, Gp;
  function2D<double> A, U, Vt;
  function1D<double> S, work;
  function1D<int> iwork;
  double dh;
  int L;
  double beta;
public:
  SIVEDE(double om_max, int Nom, int Niom, int mMax_, int Ntau, double dtau, double beta_) :
    tau(Ntau), om(Nom), iom(Niom), mMax(mMax_), Ap(mMax,Nom), Gp(mMax,Ntau),
    A(Ntau,Nom), U(Nom,Nom), Vt(Ntau,Ntau), S(min(Ntau,Nom)), beta(beta_)
  {
    dh = 2*om_max/(Nom-1);
    for (int i=0; i<Nom; i++) om[i] = -om_max + dh*i;
    for (int i=0; i<Ntau; i++) tau[i] = i*dtau;
    for (int i=0; i<Niom; i++) iom[i] = (2*i+1)*M_PI/beta;

    L = min(tau.size(),om.size());
    int LWORK = 3*min(om.size(),tau.size())*min(om.size(),tau.size())+max(max(om.size(),tau.size()),4*min(om.size(),tau.size())*min(om.size(),tau.size())+4*min(om.size(),tau.size())) + 10;
    int IWORK = 8*min(om.size(),tau.size());
    work.resize(LWORK);  iwork.resize(IWORK);

    for (int i=0; i<tau.size(); i++)
      for (int j=0; j<om.size(); j++)
	A(i,j)=-exp(-om[j]*tau[i])/(1+exp(-beta*om[j]));

    clog<<"....... Calculating SIVEDE, wait a moment!"<<endl;
    SiVeDe(true, A, S, U, Vt, work, iwork);

    //    clog<<"S="<<endl<<S<<endl<<endl;
  }
  void Transform(int m0, const function<double>& Gtau, const function<double>& dGtau, function<dcomplex>& Giom,	bool ph_symmetry, bool Qrenormalize, double nf, const string& add, const InclMore& incl_more);
  int iom_size(){return iom.size();}
  const vector<double>& iome() const{return iom;}
  void InverseFourier(const function<dcomplex>& Giom, function<double>& Gtau);
  void LinearFourier(const  function<double>& Gtau, function<dcomplex>& Giom);
};

void renormalize(function<double>& g, const vector<double>& om, double D0, double n, double beta)
{
  double dh = om[1]-om[0];
  double D1 = om[om.size()-1];
  int il = static_cast<int>((D1-D0)/(2*D1)*(om.size()-1));
  int ir = static_cast<int>((D1+D0)/(2*D1)*(om.size()-1))+1;
  // central region will be unchanged
  double sum0 = 0, sumn=0;
  for (int i=il+1; i<ir; i++){
    double c = (i==0 || i==om.size()-1) ? 0.5 : 1;
    sum0 += g[i]*dh*c; // norm
    sumn += g[i]*dh*c*ferm_f(om[i]*beta); // density
  }
  double dIl = n-sumn;    // integral of the left region must be
  double dIr = 1-sum0-dIl;// integral of the right region must be

  // left region needs change
  double suml0=0, suml1=0, suml2=0, suml3=0; // integral, first and second moment
  for (int i=0; i<=il; i++){
    double c = (i==0) ? 0.5 : 1;
    suml0 += g[i]*dh*c;
    suml1 += g[i]*dh*c*(-om[i]-D0);
    suml2 += g[i]*dh*c*sqr(-om[i]-D0);
    suml3 += g[i]*dh*c*sqr(-om[i]-D0)*(-om[i]-D0);
  }
  double alphal = (suml0-suml2/sqr(D1-D0)-dIl)/(suml2/(D1-D0)-suml1);
  double betal = -(1+alphal*(D1-D0))/sqr(D1-D0);
  double gammal=0;
  if (fabs(betal)>1){
    alphal=0;
    betal = (suml3+pow(D1-D0,3)*(dIl-suml0))/(sqr(D1-D0)*(suml2*(D1-D0)-suml3));
    gammal = (dIl-suml0-betal*suml2)/suml3;
  }
  // right region needs change
  double sumr0=0, sumr1=0, sumr2=0, sumr3=0; // integral, first and second moment
  for (int i=ir; i<om.size(); i++){
    double c = (i==om.size()-1) ? 0.5 : 1;
    sumr0 += g[i]*dh*c;
    sumr1 += g[i]*dh*c*(om[i]-D0);
    sumr2 += g[i]*dh*c*sqr(om[i]-D0);
    sumr3 += g[i]*dh*c*sqr(om[i]-D0)*(om[i]-D0);
  }
  double alphar = (sumr0-sumr2/sqr(D1-D0)-dIr)/(sumr2/(D1-D0)-sumr1);
  double betar = -(1+alphar*(D1-D0))/sqr(D1-D0);
  double gammar=0;
  if (fabs(betar)>1){
    alphar=0;
    betar = (sumr3+pow(D1-D0,3)*(dIr-sumr0))/(sqr(D1-D0)*(sumr2*(D1-D0)-sumr3));
    gammar = (dIr-sumr0-betar*sumr2)/sumr3;
  }
  // finally, normalizing left and right region to get right normalization and right doping
  for (int i=0; i<=il; i++)        g[i] *= (1+alphal*(-om[i]-D0)+betal*sqr(-om[i]-D0)+gammal*pow(-om[i]-D0,3));
  for (int i=ir; i<om.size(); i++) g[i] *= (1+alphar*(om[i]-D0)+betar*sqr(om[i]-D0)+gammar*pow(om[i]-D0,3));
}

void SIVEDE::Transform(int m0, const function<double>& Gtau, const function<double>& dGtau, function<dcomplex>& Giom,
		       bool ph_symmetry, bool Qrenormalize, double nf, const string& add, const InclMore& incl_more)
{
  ofstream ocoef((string("mcoef")+add).c_str());
  Ap=0; Gp=0;
  //  Giom.resize(iom.size());
  if (mMax>L) mMax = L;
  if (m0>mMax) m0 = mMax;
  double sumerr=0;
  double mcoeff=0;
  for (int i=0; i<tau.size(); i++) mcoeff += Vt(i,0)*Gtau[i];
  for (int i=0; i<tau.size(); i++) Gp[0][i] = mcoeff*Vt(i,0);
  for (int i=0; i<om.size(); i++)  Ap[0][i] = mcoeff*U(0,i)/S[0];
  double error=0;
  for (int i=0; i<tau.size(); i++) error += Vt(i,0)*dGtau[i];
  for (int i=0; i<om.size(); i++) sumerr += fabs(error*U(0,i)/S[0]);

  for (int m=1; m<mMax; m++){
    double mcoeff = 0;
    for (int i=0; i<tau.size(); i++) mcoeff += Vt(i,m)*Gtau[i];
    if (ph_symmetry && m%2==1) mcoeff=0;
    if (incl_more.Q && m>m0) mcoeff *=  exp(-incl_more.fct*(m-m0));

    double error=0;
    for (int i=0; i<tau.size(); i++) error += Vt(i,m)*dGtau[i];
    for (int i=0; i<om.size(); i++)  sumerr += fabs(error*U(m,i)/S[m]);
    if (sumerr>1.) mcoeff=0;

    for (int i=0; i<tau.size(); i++) Gp[m][i] = Gp[m-1][i] + mcoeff*Vt(i,m);
    for (int i=0; i<om.size(); i++)  Ap[m][i] = Ap[m-1][i] + mcoeff*U(m,i)/S[m];

    double norm=0;
    for (int i=0; i<Ap.size_Nd(); i++) norm+= Ap[m][i];

    ocoef<<setw(3)<<m<<" "<<setw(12)<<S[m]<<" "<<setw(12)<<mcoeff/S[m]<<" "<<setw(12)<<norm<<" "<<setw(12)<<error<<" "<<setw(12)<<sumerr<<endl;
  }
  Ap *= (1./dh);

  if (incl_more.Q) m0 = mMax-1;
  /*
  // slightly correcting spectral function because we have two constrains
  // A is positive and normalized : A_i>0 and \sum_i A_i=1
  for (int i=0; i<Ap.size_Nd(); i++) if (Ap[m0][i]<0) Ap[m0][i]=0;
  */

  double norm=0;
  for (int i=0; i<Ap.size_Nd(); i++) norm += Ap[m0][i]*dh;
  norm -= 0.5*dh*Ap[m0][0] + 0.5*dh*Ap[m0][om.size()-1];
  clog<<"should renormalize by "<<norm<<endl;

  double D0=3;
  if (Qrenormalize) renormalize(Ap[m0], om, D0, nf, beta);
  else Ap[m0] *= (1/norm);


  // Going to imaginary axis!
  for (int j=0; j<iom.size(); j++){
    dcomplex iomc = dcomplex(0,iom[j]);
    dcomplex sum=0;
    for (int i=0; i<om.size()-1; i++){
      double A0 = Ap[m0][i];
      double A1 = Ap[m0][i+1];
      double dA = (A1-A0);
      sum += -dA+(A0+dA/dh*(iomc-om[i]))*(log(om[i]-iomc)-log(om[i+1]-iomc));
    }
    Giom[j]=sum;
  }

  static function1D<double> Gtau_temp(Gtau.size());
  InverseFourier(Giom, Gtau_temp); // to see how good the fit is

  // Some debugging printing
  ofstream out((string("Awp")+add).c_str());
  for (int i=0; i<om.size(); i++){
    out<<setw(25)<<om[i]<<" ";
    for (int j=0; j<mMax; j++) out<<setw(25)<<Ap[j][i]<<" ";
    out<<endl;
  }
  ofstream outg((string("Gtp")+add).c_str());
  for (int i=0; i<tau.size(); i++){
    outg<<setw(25)<<tau[i]<<" ";
    for (int j=0; j<mMax; j++) outg<<setw(25)<<Gp[j][i]<<" ";
    outg<<endl;
  }
  ofstream outm((string("Aw")+add).c_str());
  for (int i=0; i<om.size(); i++) outm<<setw(25)<<om[i]<<" "<<Ap[m0][i]<<endl;
  ofstream outi((string("Giom")+add).c_str());
  for (int i=0; i<iom.size(); i++) outi<<setw(25)<<iom[i]<<" "<<Giom[i]<<endl;
  ofstream outt((string("Gtau")+add).c_str());
  for (int i=0; i<tau.size(); i++) outt<<setw(25)<<tau[i]<<" "<<Gtau_temp[i]<<" "<<Gtau[i]<<endl;
}

void SIVEDE::InverseFourier(const function<dcomplex>& Giom, function<double>& Gtau)
{
  for (int t=0; t<tau.size(); t++){
    double sum=0;
    for (int n=0; n<iom.size(); n++)
      sum += cos(iom[n]*tau[t])*Giom[n].real() + sin(iom[n]*tau[t])*(Giom[n].imag()+1/iom[n]);
    Gtau[t] = 2*sum/beta-0.5;
  }
}
void SIVEDE::LinearFourier(const  function<double>& Gtau, function<dcomplex>& Giom)
{
  for (int n=0; n<iom.size(); n++){
    double sumre=0, sumim=0;
    for (int t=0; t<tau.size()-1; t++){
      double c0 = cos(tau[t]*iom[n]) , c1 = cos(tau[t+1]*iom[n]);
      double s0 = sin(tau[t]*iom[n]),  s1 = sin(tau[t+1]*iom[n]);
      double G0 = Gtau[t], G1 = Gtau[t+1];
      double dG = (G1-G0)/(tau[t+1]-tau[t]);
      sumim += (c0*G0-c1*G1 + dG*(s1-s0)/iom[n])/iom[n];
      sumre += (s1*G1-s0*G0 + dG*(c1-c0)/iom[n])/iom[n];
    }
    Giom[n] = dcomplex(sumre,sumim);
  }
}

int main(int argc, char *argv[], char *env[])
{
  double beta=16;  // inverse temperature
  double U=2;      // Coulomb U
  double Ed=-1;
  double J=0;      // Hunds J
  int Nd = 2;      // one-band == SU(2)
  int L = 64;      // number of time slices

  double ncor= 3.0;  // Number of sweeps between measurements recorded (should be as uncorrelated as possible)
  int binsize = 1000; // number of measurements grouped together to one bin
  int nbin = 50;   // number of total bins computed

  int ndirty = 100;   // After "ndirty" spin flips calculated by the fast method, numerical accuracy is recovered by a "clean" update
  int nwarm0 = 100;   // Number of sweeps thrown away at the beginning when input ising configuration does not exist
  int nwarm1 = 10;    // Number of sweeps thrown away at the beginning when input ising configuration exists
  bool random_site=true; // Is ising spin choosen randomly or in a sequence?
  bool Metropolis=false;  // Metropolis or heat-bath
  bool svd_inverse_fourier = true; // svd or linear Fourier
  bool SUN=true;
  bool Bethe=true;
  bool ph_symmetry=true;
  bool cure=true;
  double incf=-1;
  int niter = 20;
  double mix = 0.5;

  int m0=8;
  double real_w_cutof = 5;
  int N_real_w = 500;
  int Niom = 3000;

  bool print_help=false;
  ifstream ftry("g0iom.input");
  if (!ftry) print_help=true;

  int i=0;
  while (++i<argc){
    std::string str(argv[i]);
    if (str=="-beta" && i<argc-1)    beta        = atof(argv[++i]);
    if (str=="-U" && i<argc-1)       U           = atof(argv[++i]);
    if (str=="-J" && i<argc-1)       J           = atof(argv[++i]);
    if (str=="-Ed" && i<argc-1)      Ed          = atof(argv[++i]);
    if (str=="-Nd" && i<argc-1)      Nd          = atoi(argv[++i]);
    if (str=="-L" && i<argc-1)       L           = atoi(argv[++i]);
    if (str=="-ncor" && i<argc-1)    ncor        = atof(argv[++i]);
    if (str=="-binsize" && i<argc-1) binsize     = atoi(argv[++i]);
    if (str=="-nbin" && i<argc-1)    nbin        = atoi(argv[++i]);
    if (str=="-ndirty" && i<argc-1)  ndirty      = atoi(argv[++i]);
    if (str=="-nwarm0" && i<argc-1)  nwarm0      = atoi(argv[++i]);
    if (str=="-nwarm1" && i<argc-1)  nwarm1      = atoi(argv[++i]);
    if (str=="-no_randoms")          random_site = false;
    if (str=="-Metropolis")          Metropolis  = true;
    if (str=="-linearF")     svd_inverse_fourier = false;
    if (str=="-m0" && i<argc-1)      m0          = atoi(argv[++i]);
    if (str=="-cutf" && i<argc-1)    real_w_cutof= atof(argv[++i]);
    if (str=="-Nrw" && i<argc-1)     N_real_w    = atoi(argv[++i]);
    if (str=="-Niw" && i<argc-1)     Niom        = atoi(argv[++i]);
    if (str=="-no_SUN")              SUN         = false;
    if (str=="-no_Bethe")            Bethe       = false;
    if (str=="-no_phsym")            ph_symmetry = false;
    if (str=="-niter" && i<argc-1)   niter       = atoi(argv[++i]);
    if (str=="-mix" && i<argc-1)     mix         = atof(argv[++i]);
    if (str=="-no_cure")             cure        = false;
    if (str=="-incf")                incf        = atof(argv[++i]);
    if (str=="-h" || str=="--help" || print_help){
      std::clog<<"**************** QMC program for Anderson impurity model *********\n";
      std::clog<<"**                  using Hirsh-Fye algorithm                   **\n";
      std::clog<<"**              Copyright Kristjan Haule, 29.11.2005            **\n";
      std::clog<<"******************************************************************\n";
      std::clog<<" Program requires file g0iom.input as input for G0 on imaginary axis.\n";
      std::clog<<"                       ising.dat might be used if guess for ising configuration exists\n";
      std::clog<<"qmc [options]\n" ;
      std::clog<<"Options:   -beta        inverse temperature  ("<<beta<<")\n";
      std::clog<<"           -U           Coulomb U ("<<U<<")\n";
      std::clog<<"           -J           Hunds J ("<<J<<")\n";
      std::clog<<"           -Ed          Impurity level for SU(N) ("<<Ed<<")\n";
      std::clog<<"           -Nd          Number of bans (1band=SU(2)->Nd=2) ("<<Nd<<")\n";
      std::clog<<"           -L           number of time slices ("<<L<<")\n";
      std::clog<<"           -ncor        Number of sweeps between measurements recorded ("<<ncor<<")\n";
      std::clog<<"           -binsize     Number of measurements grouped together to one bin ("<<binsize<<")\n";
      std::clog<<"           -nbin        Total number of bins computed ("<<nbin<<")\n";
      std::clog<<"           -ndirty      After 'ndirty' spin flips follows clean update ("<<ndirty<<")\n";
      std::clog<<"           -nwarm0      Number of warmup sweeps if no starting ising configuration ("<<nwarm0<<")\n";
      std::clog<<"           -nwarm1      Number of warmup sweeps if starting ising configuration exists ("<<nwarm1<<")\n";
      std::clog<<"           -no_randoms  Ising spin are choosen in a sequence (sweeps) ("<<!random_site<<")\n";
      std::clog<<"           -Metropolis  Metropolis is used instead of Heat-bath algorithm ("<<Metropolis<<")\n";
      std::clog<<"           -linearF     SVD or linear Fourier("<<!svd_inverse_fourier <<")\n";
      std::clog<<"           -m0          Cutoff for singular values. The last which is taken into account is ("<<m0 <<")\n";
      std::clog<<"           -cutf        Cutoff frequnecy on real axis when doing SVD ("<<real_w_cutof <<")\n";
      std::clog<<"           -Nrw         Number of real frequnecy points when doing SVD ("<<N_real_w <<")\n";
      std::clog<<"           -Niw         Number of imaginary frequnecy points ("<<Niom  <<")\n";
      std::clog<<"           -no_SUN      Wheather is SU(N) problem or not (if SU(N) we can average over bands)("<< !SUN <<")\n";
      std::clog<<"           -no_Bethe    In niter>0 we can treat only Bether lattice ("<< !Bethe<<")\n";
      std::clog<<"           -no_phsym    If there is no particle-hole symmetry ("<< !ph_symmetry<<")\n";
      std::clog<<"           -niter       Number of DMFT iterations in case of Bethe lattice ("<< niter<<")\n";
      std::clog<<"           -mix         Linear mixing parameter ("<< mix<<")\n";
      std::clog<<"           -no_cure     Do not care about spectral function being not exactly normalized("<<!cure<<")\n";
      std::clog<<"           -incf        Includes all singular values but they are cut-off by exponent of this factor ("<<incf<<")\n";
      std::clog<<"           -h           Print this help message \n";
      std::clog<<"*****************************************************\n";
      return 0;
    }
  }
  {
    ofstream history("qmc.history",ios::app);
    if (!history) cerr<<" Didn't suceeded to open history file!"<<endl;
    for (int i=0; i<argc; i++) history << argv[i] << " ";
    history << endl;
  }
  clog.precision(10);
  cout.precision(12);
  if (ph_symmetry) Ed = -0.5*U;

  SIVEDE svd(real_w_cutof, N_real_w, Niom, 20, L+1, beta/L, beta); // class for Fourier transformation

  QMC qmc(Nd, L, beta/L, U, J, ndirty); // QMC simulation class

  int iseed = time(0); // initialization of random number generator
  RanGSL rand(iseed);
  cout<<"iseed="<<iseed<<endl;

  function2D<dcomplex> G0iom(Nd,svd.iom_size()); // input G(iom)
  function2D<double> G0(Nd,L+1), G(Nd,L);        // input G0(tau) and output G(tau)
  vector<function2D<double> > g(Nd); // g is a function of ising configuration
  for (int i=0; i<g.size(); i++) g[i].resize(L,L);

  InclMore incl_more(false, 1.);
  if (incf>0) {incl_more.Q=true; incl_more.fct=incf;}

  if (!ReadG0("g0iom.input", G0iom, svd.iome())) {cerr<<"Can not read G0 input file!"<<endl; return 1;}

  int nIsingSpins = L*qmc.NumIsingSpins();              // number of all ising spins

  // Tries to read starting ising configureation from file. If it does not exist, initializes it randomly and puts nwarmup to nwarm0
  int nwarmup = nwarm0*nIsingSpins;
  if (qmc.GetStartIsingConfiguration("ising.dat", rand)) nwarmup = nwarm1*nIsingSpins;

  int measurement = static_cast<int>(ncor*nIsingSpins); // state recorded every "measurement" MC steps
  int nsteps = measurement*nbin*binsize + nwarmup + 1;
  double nsweep = nsteps/nIsingSpins;  // Number of flips that are tried is nsweeps*L

  clog<<"Number of all ising spins is "<<nIsingSpins<<endl;
  clog<<"Measurement recorded every "<<measurement<<" steps"<<endl;
  clog<<"One bin contains "<<binsize<<" measurements and requires "<<measurement*binsize<<" steps"<<endl;
  clog<<"Number of warmup steps is "<<nwarmup<<endl;
  clog<<"Number of bins that will be collected is "<<nbin<<"; requires total number of "<<nsteps<<" MC steps; or "<<nsweep<<" sweeps."<<endl;
  function1D<double> a(2);// temporary data
  function2D<double> Gtau(Nd,L+1), dGtau(Nd,L+1); // result of QMC simulation G(tau) and its error
  function2D<dcomplex> Giom(Nd,svd.iom_size());   // imaginary frequency analog of G(tau)=G(iom)
  function2D<dcomplex> Sigma(Nd,svd.iom_size());  // self-energy energy on imaginary axis
  function1D<double>  nf(Nd);

  for (int iter=0; iter<niter; iter++){ // over DMFT iterations if SCC is very simple (Bethe lattice)

    for (int i=0; i<Nd; i++) svd.InverseFourier(G0iom[i], G0[i]); // We read G0(iom). We need G0(tau)
    qmc.SetG0(G0);    // We set G0(tau) on discrete mesh. We have to make sure that it is antiperodic function.

    qmc.CleanUpdate(g); // We start with clean update

    for (long istep=0; istep<nsteps; istep++){

      int isweep = istep/nIsingSpins;     // How many times all ising sites were visited on average
      // ising spin which is going ot be flipped
      int jj = (random_site) ?  static_cast<int>(rand()*L*qmc.NumIsingSpins()) : istep % nIsingSpins;
      int l = jj / qmc.NumIsingSpins();  // which time slice
      int z = jj % qmc.NumIsingSpins();  // and which band this ising spin correspond to

      // Fast update - computes Det-ratio of the new and previous configuration
      double Det = qmc.DetRatio(z, l, g, a);

      // The acceptance probability
      double P = (Metropolis) ? Det : Det/(1+Det); //Metropolis or Heat-bath algorithm

      // Accept the move with probability P
      // This step takes all cpu time. Needs to be heavily optimized.
      if (P>rand()) qmc.AcceptMove(z, l, g, a, isweep);

      // If warmup time is over we can record meassurements.
      // We put binsize measurements into each bin
      if (istep > nwarmup && (istep-nwarmup)%measurement==0) qmc.SaveMeasurement(g,binsize);
    }
    qmc.GiveResult(Gtau, dGtau,nf); // This qmc run finished. Print results!

    // Some printing of results
    clog<<endl;
    qmc.PrintResult(cout); ofstream out("results.dat");
    qmc.PrintResult(out);  qmc.PrintIsingConfiguration("ising.dat");


    if (SUN){ // averaging over all bands
      for (int l=0; l<=L; l++){
				double sum=0;
				for (int i=0; i<Nd; i++) sum += Gtau[i][l];
				for (int i=0; i<Nd; i++) Gtau[i][l] = sum/Nd;
      }
    }
    // Fourier transformation can be with SVD decomposition or linear interpolation of G(tau)
    for (int id=0; id<Nd; id++){
      if (svd_inverse_fourier){
				if (ph_symmetry) nf[id]=0.5;
				stringstream c1; c1<<"."<<iter<<ends;
				svd.Transform(m0, Gtau[id], dGtau[id], Giom[id], ph_symmetry, cure, nf[id], c1.str(),incl_more);
      }else
				svd.LinearFourier(Gtau[id],Giom[id]);

      for (int i=0; i<svd.iom_size(); i++) Sigma[id][i] = 1/G0iom[id][i]-1/Giom[id][i];
      double zero = Sigma[id].last().imag()/svd.iome()[svd.iom_size()-1]; // sigma''(infinity) must go to zero
      for (int i=0; i<svd.iom_size(); i++) Sigma[id][i].imag() -= zero*svd.iome()[i];
      stringstream c2; c2<<"sigma."<<iter<<ends;
      ofstream outs(c2.str().c_str());
      for (int i=0; i<svd.iom_size(); i++) outs<<setw(25)<<svd.iome()[i]<<" "<<setw(25)<<Sigma[id][i]<<endl;
    }

    if (Bethe){// very primitive SCC for bethe lattice
      // only for Bethe lattice can continue. Otherwise user needs to restart the program with new G0
      // which is obtained by some other program (for example in LDA+DMFT with ksum)
      double t2 = 0.25;   // t^2 determines the half-bandwidth. Here D is equal to unity
      //      double Ed = -0.5*U; // impurity level : half-filling
      double mu_QMC = -(Ed+(Nd-1)*U/2.);
      for (int id=0; id<Nd; id++){
				for (int i=0; i<svd.iom_size(); i++){
				  dcomplex G0old = G0iom[id][i];
				  dcomplex G0new = 1/(dcomplex(0,svd.iome()[i]) + mu_QMC - t2*Giom[id][i]);
				  G0iom[id][i] = G0old*(1-mix) + mix*G0new;
				}
      }
      // some debug printing
      stringstream c3; c3<<"Gb_out."<<iter<<ends;
      ofstream out1(c3.str().c_str());
      for (int l=0; l<=L; l++) out1<<setw(25)<<l*beta/L<<" "<<setw(25)<<Gtau[0][l]<<" "<<G0[0][l]<<endl;
      stringstream c4; c4<<"G0_out."<<iter<<ends;
      ofstream out2(c4.str().c_str());
      for (int i=0; i<svd.iom_size(); i++){
				out2<<setw(25)<<svd.iome()[i]<<" ";
				for (int id=0; id<Nd; id++) out2<<setw(25)<<G0iom[id][i]<<" ";
				out2<<endl;
      }
    } else{ break;}

  }

  return 0;
}
