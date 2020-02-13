#include <cstdio>
#include <cstdlib>
#include <gsl/gsl_rng.h>
#include <map>

class RanGSL{
  const gsl_rng_type *T;
  gsl_rng *r;
public:
  RanGSL(int idum)
  {
    gsl_rng_env_setup();
    //    T = gsl_rng_default;
    //    T = gsl_rng_taus;
    T = gsl_rng_ranlux389;
    r = gsl_rng_alloc (T);
    gsl_rng_set(r, idum);
  }
  double operator()()
  { return gsl_rng_uniform (r);}
  ~RanGSL()
  {gsl_rng_free (r);}
};

class dRand48{
public:
  dRand48(int idum){srand48(idum);}
  double operator()(){
    return drand48();
  }
};

class Ran0{
  long idum;
public:
  Ran0(long idum_): idum(idum_){};
  double operator()(){
    // Minimal random number generator of Park and Miller. Returns a uniform random deviate
    // between 0.0 and 1.0. Set or reset idum to any integer value (except the unlikely value MASK)
    // to initialize the sequence; idum must not be altered between calls for successive deviates in
    // a sequence.
    static const int IA=16807;
    static const int IM=2147483647;
    static const double AM=(1.0/IM);
    static const int IQ=127773;
    static const int IR=2836;
    static const int MASK=123459876;
    static long idum;
    long k;
    float ans;
    idum ^= MASK; //XORing with MASK allows use of zero and other simple bit patterns for idum.
    k=idum/IQ; 
    idum = IA*(idum-k*IQ)-IR*k; // Compute idum=(IA*idum) % IM without over
    if (idum < 0) idum += IM; // flows by Schrages method.
    ans=AM*idum; // Convert idum to a floating result.
    idum ^= MASK; // Unmask before return.
    return ans;
  }
};

class Ran4{
  static const int NITER=4;
  long idum;
public:
  Ran4(long idum_): idum(idum_){};
  float operator()()
    // Returns a uniform random deviate in the range 0.0 to 1.0,gener ated by pseudo-DES (DESlike)
    // hashing of the 64-bit word (idums,idum),where idums was set by a previous call with
    // negative idum. Also increments idum. Routine can be used to generate a random sequence
    // by successive calls,le aving idum unaltered between calls; or it can randomly access the nth
    // deviate in a sequence by calling with idum = n. Different sequences are initialized by calls with
    // differing negative values of idum.
  {
    unsigned long irword, itemp, lword;
    static long idums = 0;
    //  The hexadecimal constants jflone and jflmsk below are used to produce a floating number
    //    between 1. and 2. by bitwise masking. They are machine-dependent.
    static unsigned long jflone = 0x3f800000;
    static unsigned long jflmsk = 0x007fffff;
    if (idum < 0) { // Reset idums and prepare to return the first deviate in its sequence.
      idums = -idum;
      idum=1;
    }
    irword=idum;
    lword=idums;
    psdes(lword,irword); //(B!HPseudo-DES(B!I encode the words.
    itemp=jflone | (jflmsk & irword); //Mask to a floating number between 1 and 2
    ++idum;
    return (*reinterpret_cast<float*>(&itemp))-1.0; //Subtraction moves range to 0. to 1.
  }
private:
  void psdes(unsigned long& lword, unsigned long& irword)
    //Pseudo hashing of the 64-bit word (lword,irword). Both 32-bit arguments are returnedA
    // hashed on all bits.
  {
    unsigned long i,ia,ib,iswap,itmph=0,itmpl=0;
    static unsigned long c1[NITER]={0xbaa96887L, 0x1e17d32cL, 0x03bcdc3cL, 0x0f33d1b2L};
    static unsigned long c2[NITER]={0x4b0f3b58L, 0xe874f0c3L, 0x6955c5a6L, 0x55a7ca46L};
    for (i=0;i<NITER;i++) {
      //    Perform niter iterations of DES logic,us ing a simpler (non-cryptographic) nonlinear function
      //    instead of 
      ia=(iswap=(irword))^c1[i]; //The bit-rich constants c1 and (below) c2 guarantee lots of nonlinear mixing.
      itmpl = ia & 0xffff;
      itmph = ia >> 16;
      ib=itmpl*itmpl+ ~(itmph*itmph);
      irword=(lword)^(((ia = (ib >> 16) | ((ib & 0xffff) << 16)) ^ c2[i])+itmpl*itmph);
      lword=iswap;
    }
  }
};

