gcc -c -fPIC -o sample_betas_julia.o sample_betas_julia.c
gcc -fPIC -shared -o sample_betas_julia.so sample_betas_julia.o
