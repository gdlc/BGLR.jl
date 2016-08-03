## BGLR for the Julia Language

This Julia package implements **Bayesian shrinkage and variable selection methods for high-dimensional regressions**.

The design is inspired on the BGLR R-package (BGLR-R). Over time we aim to implement a similar set of methods than the ones implelented in the R version. The R version is highly optimized with use of compiled C code; this gives BGLR-R a computational speed much higher than the one that can be obtained using R code only. 

By developing BGLR for Julia (BGLR-Julia) wee seek to: (i) reach the comunity of Julia users, (ii) achieve a similar or better performance than the one achieved with BGLR-R (this is challenging because BGLR-R is highly optimized and makes intensive use of C and BLAS routines), (iii) enable users to use BGLR with memory-mapped arrays as well as RAM arrays, (iv) capitalize on some of multi-core computing capabilities offered by Julia.

Funding of BGLR-R and BGLR-Julia was provided by NIH (R01 GM101219).

Authors:  Gustavo de los Campos (gustavoc@msu.edu) and Paulino Perez-Rodriguez (perpdgo@gmail.com)

- [BGLR-R in CRAN](https://cran.r-project.org/web/packages/BGLR/index.html)  
- [BGLR-R in GitHub](https://github.com/gdlc/BGLR-R)
- [BGLR-R Publication](http://www.genetics.org/content/early/2014/07/06/genetics.114.164442)


#### Installing BGLR-Julia

```Julia
  Pkg.rm("BGLR")
  Pkg.clone("https://github.com/gdlc/BGLR.jl")
```

#### Examples
  * [Genomic BLUP](#GBLUP)
  * [Parametric Shrinkage and Variable Selection](#BRR)
  * [Integrating fixed effects, regression on markers and pedigrees](#FMP)
  * [Reproducing Kernel Hilbert Spaces Regression with single Kernel methods](#RKHS)
  * [Reproducing Kernel Hilbert Spaces Regression with Kernel Averaging](#RKHS-KA)
  * [Prediction in testing data sets]()
  * [Modeling heterogeneous error variances](#HV)
  * [Modeling genetic by environment interactions](#GxE)
  * [BGLR-J Utils (a collection of utilitary functions)]()

### Genomic BLUP
<div id="GBLUP" />
```julia
 using BGLR
 
# Reading Data 
 #Markers
  X=readcsv(joinpath(Pkg.dir(),"BGLR/data/wheat.X.csv");header=true)[1];
 #Phenotypes
  y=readcsv(joinpath(Pkg.dir(),"BGLR/data/wheat.Y.csv");header=true)[1][:,1];
  
# Computing G-Matrix
  n,p=size(X);
  X=scale(X);
  G=X*X';
  G=G./p;

# Fitting the model
  # Readme: y is the phenotype vector, 
  #         ETA is a dictionary used to specify terms to be used in the regression,
  #         In this case ETA has only one term. 
  #         RKHS(K=G) is used to define a random effect with covariance matrix G.
  
  fm=bglr( y=y, ETA=Dict("mrk"=>RKHS(K=G)));
  
## Retrieving estimates and predictions
  fm.varE # posterior mean of error variance
  fm.yHat # predictions
  fm.ETA["mrk"].var # variance of the random effect
```

### Bayesian Ridge Regression
<div id="BRR" />

```julia
 using BGLR
 
 # Reading Data 
   X=readcsv(joinpath(Pkg.dir(),"BGLR/data/wheat.X.csv");header=true)[1];
   y=readcsv(joinpath(Pkg.dir(),"BGLR/data/wheat.Y.csv");header=true)[1][:,1];
  
  # Bayesian Ridge Regression
   ETA=Dict("mrk"=>BRR(X))
   fm=bglr(y=y,ETA=ETA);
  
  ## Retrieving estimates and predictions
  fm.varE # posterior mean of error variance
  fm.yHat # predictions
  fm.ETA["mrk"].var # variance of the random effect associated to markers
```

### Integrating fixed effects, random regression on markers and pedigrees data
<div id="FMP" />

```julia
##########################################################################################
# Example xxx of 
##########################################################################################

using BGLR
using Gadfly

# Reading data (markers are in BED binary format, http://pngu.mgh.harvard.edu/~purcell/plink/binary.shtml).
  X=read_bed(joinpath(Pkg.dir(),"BGLR/data/mice.X.bed"),1814,10346);
  pheno=readcsv(joinpath(Pkg.dir(),"BGLR/data/mice.pheno.csv");header=true);
  varnames=vec(pheno[2]); pheno=pheno[1]
  y=pheno[:,varnames.=="Obesity.BMI"] #column for BMI
  y=convert(Array{Float64,1}, vec(y))
  
  
# Incidence matrix for sex and litter size using a dummy variable for each level
  male=model_matrix(pheno[:,varnames.=="GENDER"];intercept=false)
  litterSize=model_matrix(pheno[:,varnames.=="Litter"];intercept=false)
  W=hcat(male, litterSize)
  

# Incidence matrix for cage, using a dummy variable for each level
  Z=model_matrix(pheno[:,varnames.=="cage"];intercept=false)


#Relationship matrix derived from pedigree
 A=readcsv(joinpath(Pkg.dir(),"BGLR/data/mice.A.csv");header=true);
 A=A[1]; #The first component of the tuple has the data

ETA=Dict("Fixed"=>FixEff(W),
	 "Cage"=>BRR(Z),
	 "Mrk"=>BRR(X),
	 "Ped"=>RKHS(K=A))


fm=bglr(y=y,ETA=ETA);

plot(x=fm.y,
     y=fm.yHat,
     Guide.ylabel("yHat"),
     Guide.xlabel("y"),
     Guide.title("Observed vs predicted"))
```

### Reproducing Kernel Hilbert Spaces Regression with single Kernel methods
<div id="RKHS" />
```julia
#Reproducing Kernel Hilbert Spaces
#Single kernel methods
#Example 4 of BGLR paper
#Box 10


using BGLR
using Distances
using Gadfly

# Reading Data 
 #Markers
  X=readcsv(joinpath(Pkg.dir(),"BGLR/data/wheat.X.csv");header=true)[1];
 #Phenotypes
  y=readcsv(joinpath(Pkg.dir(),"BGLR/data/wheat.Y.csv");header=true)[1][:,1];
  
#Computing the distance matrix and then the kernel
#pairwise function computes distance between columns, so we transpose
#the matrix to get distance between rows of X
  n,p=size(X);
  X=scale(X);
  D=pairwise(Euclidean(),X');
  D=(D.^2)./p;
  h=0.25;
  K1=exp(-h.*D);
  
# Kernel regression
  ETA=Dict("mrk"=>RKHS(K=K1));
  fm=bglr(y=y,ETA=ETA);
  
#Plots
	plot(x=fm.y,
     	 y=fm.yHat,
         Guide.ylabel("yHat"),
         Guide.xlabel("y"),
         Guide.title("Observed vs predicted"))

## Retrieving estimates and predictions
  fm.varE # posterior mean of error variance
  fm.yHat # predictions
  fm.ETA["mrk"].var # variance of the random effect
```

### Reproducing Kernel Hilbert Spaces Regression with Kernel Averaging
<div id="RKHS-KA">
```julia
#Reproducing Kernel Hilbert Spaces
#Multi kernel methods (Kernel Averaging)
#Example 5 of BGLR paper
#Box 11

using BGLR
using Distances
using Gadfly

# Reading Data 
 #Markers
  X=readcsv(joinpath(Pkg.dir(),"BGLR/data/wheat.X.csv");header=true)[1];
 #Phenotypes
  y=readcsv(joinpath(Pkg.dir(),"BGLR/data/wheat.Y.csv");header=true)[1][:,1];
  
#Computing the distance matrix and then the kernel
#pairwise function computes distance between columns, so we transpose
#the matrix to get distance between rows of X
  n,p=size(X);
  X=scale(X);
  D=pairwise(Euclidean(),X');
  D=(D.^2)./p;
  
  d=reshape(tril(D),n*n,1);
  d=d[d.>0];
  h=1/median(d);
  h=h.*[1/5,1,5];
  
  K1=exp(-h[1].*D);
  K2=exp(-h[2].*D);
  K3=exp(-h[3].*D);
  
# Kernel regression
  ETA=Dict("Kernel1"=>RKHS(K=K1),
           "Kernel2"=>RKHS(K=K2),
           "Kernel3"=>RKHS(K=K3));
  fm=bglr(y=y,ETA=ETA);
  
#Plots
	plot(x=fm.y,
     	 y=fm.yHat,
         Guide.ylabel("yHat"),
         Guide.xlabel("y"),
         Guide.title("Observed vs predicted"))

## Retrieving estimates and predictions
  fm.varE # posterior mean of error variance
  fm.yHat # predictions
  fm.ETA["Kernel1"].var # variance of the random effect
  fm.ETA["Kernel2"].var # variance of the random effect
  fm.ETA["Kernel3"].var # variance of the random effect
```


### Modeling heterogeneous error variances
<div id="HV"/>

```julia

#Heterogeneous variances

using BGLR
using Gadfly

# Reading Data 
   X=readcsv(joinpath(Pkg.dir(),"BGLR/data/wheat.X.csv");header=true)[1];
   y=readcsv(joinpath(Pkg.dir(),"BGLR/data/wheat.Y.csv");header=true)[1][:,1];

# Bayesian Ridge Regression
   ETA=Dict("mrk"=>BRR(X))
   
#Grouping
  groups=vec(readdlm(joinpath(Pkg.dir(),"BGLR/data/wheat.groups.csv")));
  groups=convert(Array{Int64,1},groups)

  fm=bglr(y=y,ETA=ETA;groups=groups);
  
  ## Retrieving estimates and predictions
  fm.varE # posterior mean of error variance
  fm.yHat # predictions
  fm.ETA["mrk"].var # variance of the random effect

  plot(x=fm.y,
       y=fm.yHat,
       Guide.ylabel("yHat"),
       Guide.xlabel("y"),
       Guide.title("Observed vs predicted"))

```

### Modeling genetic by environment interactions
<div id="GxE"/>
```julia

#Genotype x Environment interaction
#Rection norm model based on markers
using BGLR
using Gadfly

# Reading Data
 #Markers
  X=readcsv(joinpath(Pkg.dir(),"BGLR/data/wheat.X.csv");header=true)[1];
 #Phenotypes
  Y=readcsv(joinpath(Pkg.dir(),"BGLR/data/wheat.Y.csv");header=true)[1];

  n,p=size(X);

 #response vector
  y=vec(Y);

 #Environments
  Env=[rep(1;times=n);rep(2;times=n);rep(3;times=n);rep(4,times=599)];

 #genotypes
  g=[1:n;1:n;1:n;1:n]

 #Genomic relationship matrix
  X=scale(X);
  G=X*X';
  G=G./p;

 #Model matrix for environments
  Ze=model_matrix(Env;intercept=false);

 #Model matrix for genotypes
 #in this case is identity because the data is already ordered
  Zg=model_matrix(g;intercept=false);

 #Basic reaction norm model
 #y=Ze*beta_Env+X*beta_markers+u, where u~N(0,(Zg*G*Zg')#ZeZe')

 #Variance covariance matrix for the interaction
 K1=(Zg*G*Zg').*(Ze*Ze');

 #Linear predictor
  ETA=Dict("Env"=>BRR(Ze),
           "Mrk"=>BRR(X),
           "GxE"=>RKHS(K=K1));

 #Fitting the model
  fm=bglr(y=y,ETA=ETA);


 plot(x=fm.y,
     y=fm.yHat,
     Guide.ylabel("yHat"),
     Guide.xlabel("y"),
     Guide.title("Observed vs predicted"))

```
