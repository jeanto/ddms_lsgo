#!/bin/bash

# MIT License
# 
# Copyright (c) 2022 Jean Nunes Ribeiro Araujo
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

###
### This script runs the same experiments described in the PhD qualification.
###

# number of islands
if [ -z "$1" ]
  then
    islands = 4
else
	islands = "$1"
fi

# function
if [ -z "$2" ]
  then
    fun = 4
else
	fun = "$2"
fi

# number of runs
if [ -z "$3" ]
  then
    run = "$(seq 1 1)"
else
	run = "$(seq 1 "$3")"
fi


if [ $islands = '-h' ]
	then
		echo "./run arg1 arg2 arg3"
		echo "arg1: Number of islands."
		echo "arg2: Function: benchmark function to be solved. Valid values are: 1-15."
		echo "arg3: Number of runs: number of times that function is called." 

else
	# call main function with mpirun
	echo "number of islands: $islands; function: $fun; number of runs: $runi;" 
	cd build
	make
	cd experiments
	mpirun --use-hwthread-cpus -np $islands experiments_de_cc $fun $run
fi
