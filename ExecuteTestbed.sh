#!/bin/bash


for r in `seq 1 1 15`
do
	for a in `seq 0 10 100`
	do
	   for b in `seq 0 10 100`
	   do
	      for c in {1..5}
	      do
		 for d in {1..5}
		 do
		    python Main_Testbed.py $a $b $c $d $r
		 done
	      done
	   done
	done
done
