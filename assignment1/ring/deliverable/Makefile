.DEFAULT_GOAL := compile

compile: ring.cc sum3Dmatrix.cc
	mpic++ -D MEAN -O3 -o ring ring.cc
	mpic++ -D MEAN -O3 -o sum3Dmatrix sum3Dmatrix.cc

debug: ring.cc sum3Dmatrix.cc
	mpic++ -D MEAN -D DEBUG -O3 -o ring ring.cc
	mpic++ -D MEAN -O3 -o sum3Dmatrix sum3Dmatrix.cc
