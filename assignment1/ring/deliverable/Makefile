.DEFAULT_GOAL := compile

compile: ring.cc
	mpic++ -D MEAN -O3 -o ring ring.cc

debug: ring.cc
	mpic++ -D MEAN -D DEBUG -O3 -o ring ring.cc
