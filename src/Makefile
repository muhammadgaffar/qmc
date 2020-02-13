C++ = g++
LIBS = -L/opt/intel/mkl/8.1/lib/32 -lmkl_lapack -lmkl -lguide -lgsl
objects = qmc.o
exe = qmc
CFLAGS = -O3

$(exe) : $(objects)
	g++ $(CFLAGS) -o $@ $(objects) $(LIBS)

clean :
	rm -f $(objects) $(exe)


.SUFFIXES : .cc
.cc.o:
	$(C++) $(CFLAGS) -c $<

.SUFFIXES : .f
.f.o:
	$(F77) $(FFLAGS) -c $<
