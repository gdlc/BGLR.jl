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
  * [Genomic BLUP using BLGR-Julia](#GBLUP)
  * [Parametric Shrinkage and Variable Selection](#BRR)
  * [Integrating fixed effects, regression on markers and pedigrees](#FMP)
  * [Reproducing Kernel Hilbert Spaces Regression using BLGR-J]()
  * [Prediction in testing data sets]()
  * [Modeling heterogeneous error variances]()
  * [Modeling genetic by environment interactions using BGLR-J]()
  * [BGLR-J Utils (a collection of utilitary functions)]()

### Genomic BLUP using BGLR-julia
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
  
  
# Incidence matrix for sex (Male=1) and litter size
  male=(pheno[:,varnames.=="GENDER"].=="M").*1.0 ## dummy variable for male
  litterSize=pheno[:,varnames.=="Litter"]
  W=hcat(male, litterSize)
  

# Incidence matrix for cage
  Z=model_matrix(pheno[:,varnames.=="cage"])


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
