using BGLR
using Gadfly


#Test 1
#Gaussian process with simulated data
y=rand(10)
TMP=Dict("mrk"=>RKHS(K=eye(10)))

fm=bglr(y=y,ETA=TMP,R2=.9);

fm.yHat
fm.y

plot(x=fm.y,y=fm.yHat)

#Test 2
#Gaussian process with real data
#Load data into the environment
X = readcsv("/Users/paulino/Documents/Documentos\ Paulino/Estancia\ USA-Michigan/julia/X.csv");
y = vec(readdlm("/Users/paulino/Documents/Documentos\ Paulino/Estancia\ USA-Michigan/julia/y.txt"));

G=X*(X')/size(X)[2];

TMP=Dict("Gmatrix"=>RKHS(K=G))

fm=bglr(y=y,ETA=TMP);

plot(x=fm.y,
     y=fm.yHat,
     Guide.ylabel("yHat"),
     Guide.xlabel("y"),
     Guide.title("Observed vs predicted"))
     
#Test 3
#Bayesian Ridge Regression
TMP=Dict("XMatrix"=>BRR(X))


@elapsed bglr(y=y,ETA=TMP)

fm=bglr(y=y,ETA=TMP);



plot(x=fm.y,
     y=fm.yHat,
     Guide.ylabel("yHat"),
     Guide.xlabel("y"),
     Guide.title("Observed vs predicted"))


ccall((:sample_beta,"/Users/paulino/Documents/Documentos Paulino/Estancia USA-Michigan/julia/sample_betas_julia.so"),
      Void,(Int32, Int32, Ptr{Float64},Ptr{Float64},Ptr{Float64},Ptr{Float64},Float64,Float64,Float64),
      Int32(599),Int32(1279),X,fm.ETA["XMatrix"].x2,fm.ETA["XMatrix"].effects,fm.error,fm.ETA["XMatrix"].var,fm.varE,Float64(1e-7)
      )
      
#int n,  int p, double *X,  double *x2, double *b, double *e, double varBj, 
#double varE, double minAbsBeta


pX=pointer(X)

tic();
for in in 1:1500
	for j in 1:1279
		address=pX+599*(j-1)*sizeof(Float64)
	end
end
toc()

tic();
for in in 1:1500
	for j in 1:1279
		xj=slice(X,:,j)
	end
end
toc()


using ArrayViews
tic();
for in in 1:1500
	for j in 1:1279
		xj=unsafe_view(X, :, j)
	end
end
toc()