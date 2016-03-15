
## Parameters
 h2=.9
 blockSize=10
 nQTL=1000
 nIter=500
 burnIn=100
 n=1000
 p=1000
##

## Requires
 using NumericExtensions
 using Base.LinAlg.BLAS 
 using Distributions
 using Gadfly
 if compu=="home"
 	include("/Users/gdeloscampos/Dropbox/julia/misc.jl")
 else
 	include("/Users/gustavodeloscampos/Dropbox/julia/misc.jl")
 end
 #
##


# Simulation
 X=rand(Normal(),(n,p))
 X=scaleX(X)
 
 X=X./sqrt(p)
 G=X*(X')
 
 K=sumVar(X)
 SDb=sqrt(h2/K)
 b0=rand(Normal(0,SDb),p) 
 signal=X*b0
 var(signal)
 error0=rand(Normal(0,sqrt(1-h2)),n)
 y=signal+error0
 var(signal)
 var(error0)
## End of simulation
