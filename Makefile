CC = gcc -O3 -Wall -g
#CC = g++ -ansi -Wall -pedantic
#CFLAGS = -g -Wall -O3 -ffast-math -DHAVE_INLINE -DGSL_RANGE_CHECK_OFF
# CFLAGS = -g -Wall
LDFLAGS = -lgsl -lgslcblas -lm

GSL_INCLUDE = /usr/global/gsl/1.16/include/
GSL_LIB = /usr/global/gsl/1.16/lib


LSOURCE = main.c
LHEADER = hdplda.h main.h

all: $(LSOURCE) $(HEADER)
	  $(CC) -I$(GSL_INCLUDE) -L$(GSL_LIB) $(LSOURCE) $(LDFLAGS) -o hdplda 

clean:
	-rm -f hdplda


