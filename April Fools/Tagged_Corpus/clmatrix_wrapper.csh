#!/bin/csh -f

foreach i(*.txt *.raw)
	mkdir `basename $i .txt`
	mv $i `basename $i .txt`
	cd `basename $i .txt`
	/home/ed/cl_matrix_all/bin/clmatrixtagwizard.csh
	cd ..
end

