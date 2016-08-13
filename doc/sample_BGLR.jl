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


#@elapsed bglr(y=y,ETA=TMP)

fm=bglr(y=y,ETA=TMP);


plot(x=fm.y,
     y=fm.yHat,
     Guide.ylabel("yHat"),
     Guide.xlabel("y"),
     Guide.title("Observed vs predicted"))
     
#Test 4
XFixed=X[:,1:50];
TMP=Dict("XFixed"=>FixEff(XFixed))
fm=bglr(y=y,ETA=TMP);

plot(x=fm.y,
     y=fm.yHat,
     Guide.ylabel("yHat"),
     Guide.xlabel("y"),
     Guide.title("Observed vs predicted"))


#Test 5, two sets of predictors

X1=X[:,1:50];
X2=X[:,51:1279];

ETA=Dict("XFixed"=>FixEff(X1),
		 "Ridge"=>BRR(X2))
		 
fm=bglr(y=y,ETA=ETA);


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
elapsed=toq();


using ArrayViews
tic();
for in in 1:1500
	for j in 1:1279
		xj=unsafe_view(X, :, j)
	end
end
toc()


using BGLR
using Gadfly

#Wheat data
#Markers
X=readcsv(joinpath(Pkg.dir(),"BGLR/data/wheat.X.csv");header=true);
X=X[1];  #The first component of the Tuple

#Phenotypes
Y=readcsv(joinpath(Pkg.dir(),"BGLR/data/wheat.Y.csv");header=true);
Y=Y[1]; #The first component of the Tuple
y=Y[:,1];

#Relationship matrix derived from pedigree
A=readcsv(joinpath(Pkg.dir(),"BGLR/data/wheat.A.csv");header=true);
A=A[1]; #The first component of the Tuple

#Sets for cross-validations
sets=vec(readdlm(joinpath(Pkg.dir(),"BGLR/data/wheat.sets.csv")));


#Test 1: RKHS

G=X*(X')/size(X)[2];

predictor1=Dict("Gmatrix"=>RKHS(K=G))

fm=bglr(y=y,ETA=predictor1);

plot(x=fm.y,
     y=fm.yHat,
     Guide.ylabel("yHat"),
     Guide.xlabel("y"),
     Guide.title("Observed vs predicted"))

#Test 2
#Bayesian Ridge Regression
predictor2=Dict("XMatrix"=>BRR(X))

#@elapsed bglr(y=y,ETA=predictor2)

fm=bglr(y=y,ETA=predictor2);

plot(x=fm.y,
     y=fm.yHat,
     Guide.ylabel("yHat"),
     Guide.xlabel("y"),
     Guide.title("Observed vs predicted"))
     
#Test 3
XFixed=X[:,1:50];
predictor3=Dict("XFixed"=>FixEff(XFixed))
fm=bglr(y=y,ETA=predictor3);

plot(x=fm.y,
     y=fm.yHat,
     Guide.ylabel("yHat"),
     Guide.xlabel("y"),
     Guide.title("Observed vs predicted"))


#Test 4, two sets of predictors

X1=X[:,1:50];
X2=X[:,51:1279];

predictor4=Dict("XFixed"=>FixEff(X1),
		 "Ridge"=>BRR(X2))
		 
fm=bglr(y=y,ETA=predictor4);

plot(x=fm.y,
     y=fm.yHat,
     Guide.ylabel("yHat"),
     Guide.xlabel("y"),
     Guide.title("Observed vs predicted"))


#=
 Bed format,
  
 http://pngu.mgh.harvard.edu/~purcell/plink/binary.shtml

 The first 3 bytes have a special meaning. The first two are fixed, a 
 'magic number' that enables PLINK to confirm that a BED file is really 
 a BED file. That is, BED files should always start 01101100 00011011. 
 The third byte indicates whether the BED file is in SNP-major or 
 individual-major mode: a value of 00000001 indicates SNP-major (i.e. list 
 all individuals for first SNP, all individuals for second SNP, etc) 
 whereas a value of 00000000 indicates individual-major 
 (i.e. list all SNPs for the first individual, list all SNPs for 
 the second individual, etc)

 01101100=0x6c
 00011011=0x1b
 00000001=0x01

 use the xxd command to view the binary file
 
 Example:

 $xxd sample.bed
 
 0000000: 6c1b 01fa ff57 bfab ffff effe bffe ffff  l....W..........
 
 $xxd -b sample.bed
 
 0000000: 01101100 00011011 00000001 11111010 11111111 01010111  l....W


For the genotype data, each byte encodes up to four genotypes (2 bits per genoytpe). The coding is
     00  Homozygote "1"/"1"
     01  Heterozygote
     11  Homozygote "2"/"2"
     10  Missing genotype

The only slightly confusing wrinkle is that each byte is effectively read backwards. That is, if we label each of the 8 position as A to H, we would label backwards:

     01101100
     HGFEDCBA

and so the first four genotypes are read as follows:

     01101100
     HGFEDCBA

           AB   00  -- homozygote (first)
         CD     11  -- other homozygote (second)
       EF       01  -- heterozygote (third)
     GH         10  -- missing genotype (fourth)

Finally, when we reach the end of a SNP (or if in individual-mode, the end of an individual) we skip to the start of a new byte (i.e. skip any remaining bits in that byte).
It is important to remember that the files test.bim and test.fam will already have been read in, so PLINK knows how many SNPs and individuals to expect.

The routine will return a vector of dimension n*p, with the snps stacked. The vector contains integer codes:

Int code	Genotype	
0		00
1		01
2		10
3		11

Recode snp to 0,1,2 Format using allele "1" as reference

0 --> 0
1 --> 1
2 --> NA
3 --> 2

=#

function read_bed(bed_file::ASCIIString,n::Int64,p::Int64)

	s = open(bed_file)
	read(s,UInt8) == 0x6c && read(s,UInt8) == 0x1b || error("Unknown file format")
	read(s,UInt8) == 1 || error("Only snp and individual major order are supported")
	m = div(n+3,4)
	bb=Mmap.mmap(s,Array{UInt8,2},(m,p))
	close(s)
	
	mask=0x03
	recode = [0,2,1,3]

	out=zeros(n*p)
    out

	#Loop over snps
	for j in 0:(p-1)
    	l=-1
    	buffer=bb[:,(j+1)]
    	
    	#loop over individuals
    	for i in 0:(m-1)
    		c=buffer[(i+1)]
    		for k in 0:3
    			l=l+1
    			code = c & mask
    			c=c>>2;  #Right shift (two bits)
    		
    			#Some pieces of information are meaningless if the number of individuals IS NOT a multiple of 4
            	#at the end of the snp
            
            	if(l<n)    
            		#Add one because in julia we begin counting in 1
                	out[(l+(j*n))+1]=recode[1+code]
            	end                          
    		end
    	end
	end
	X=reshape(out,(n,p))
	X
end

X=read_bed("mice.X.bed",1814,10346);

##########################################################################################
#mice
##########################################################################################

using BGLR
using Gadfly

#model matrix for a factor using p-1 Dummy variables
#where p is the number of levels

function model_matrix(x)
	
	levels=sort(unique(x))
	n=size(x)[1]
	p=size(levels)[1]
	
	if(p<2) 
		error("The factor should have at least 2 levels")
	end
		
	X=zeros(n,p-1)
	
	for j in 2:p
		index=(x.==levels[j])
		X[index,j-1]=1
	end
	
	X
end


#Markers in plink format
X=read_bed(joinpath(Pkg.dir(),"BGLR/data/mice.X.bed"),1814,10346);

pheno=readcsv(joinpath(Pkg.dir(),"BGLR/data/mice.pheno.csv");header=true);

#pheno contains two Tuples, the first one is the data without header and 
#the second tuple the headers

pheno[1]  #First tuple
pheno[2]  #Second tuple

col=vec(find(pheno[2].=="Obesity.BMI"))[1] #column for BMI
y=pheno[1][:,col]
y=convert(Array{Float64,1}, y)  #Be sure that y is Array{Float64,1}

#Gender
col=vec(find(pheno[2].=="GENDER"))[1] #column for GENDER
GENDER=pheno[1][:,col]
X1=model_matrix(GENDER)

#Litter
col=vec(find(pheno[2].=="Litter"))[1] #column for Litter
Litter=pheno[1][:,col]
X2=model_matrix(Litter)

Fixed=hcat(X1,X2)

#Gage
col=vec(find(pheno[2].=="cage"))[1] #column for cage
cage=pheno[1][:,col]
X3=model_matrix(cage)


#Relationship matrix derived from pedigree
A=readcsv(joinpath(Pkg.dir(),"BGLR/data/mice.A.csv");header=true);
A=A[1]; #The first component of the Tuple

ETA=Dict("Fixed"=>FixEff(Fixed),
	     "Cage"=>BRR(X3),
	     "Mrk"=>BRR(X),
	     "Ped"=>RKHS(K=A))


fm=bglr(y=y,ETA=ETA);

plot(x=fm.y,
     y=fm.yHat,
     Guide.ylabel("yHat"),
     Guide.xlabel("y"),
     Guide.title("Observed vs predicted"))
     
     
       
#Heterogeneous variances

using BGLR
using Gadfly

# Reading Data 
   X=readcsv(joinpath(Pkg.dir(),"BGLR/data/wheat.X.csv");header=true)[1];
   y=readcsv(joinpath(Pkg.dir(),"BGLR/data/wheat.Y.csv");header=true)[1][:,1];

# Bayesian Ridge Regression
   ETA=Dict("mrk"=>BRR(X))
   
#Sets for cross-validations
groups=vec(readdlm(joinpath(Pkg.dir(),"BGLR/data/wheat.groups.csv")));
groups=convert(Array{Int64,1},groups)

fm=bglr(y=y,ETA=ETA;groups=groups);

plot(x=fm.y,
     y=fm.yHat,
     Guide.ylabel("yHat"),
     Guide.xlabel("y"),
     Guide.title("Observed vs predicted"))


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
  

library(BGLR)
data(wheat)
set.seed(123)
Y=wheat.Y
X=wheat.X
n=nrow(X)
p=ncol(X)
y=Y[,1]
X=scale(X,center=TRUE,scale=TRUE)
D=(as.matrix(dist(X,method="euclidean"))^2)/p
h=0.25
K=exp(-h*D)


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


#Assesment of prediction accuracy using a single training-testing partition
#Box 12 in BGLR package
using BGLR
using StatsBase
using Gadfly

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

#Creating a Testing set
 yNA=deepcopy(y)
 srand(456);
 tst=sample([1:n],100;replace=false)
 yNA[tst]=-999

#Fitting the model
 ETA=Dict("mrk"=>RKHS(K=G))
 
 fm=bglr(y=yNA,ETA=ETA;nIter=5000,burnIn=1000);

#Correlation in training and testing sets
 trn=(yNA.!=-999);
 rtst=cor(fm.yHat[tst],y[tst]);
 rtrn=cor(fm.yHat[trn],y[trn]);
 rtst
 rtrn

 plot(layer(x=y[trn],
            y=fm.yHat[trn],Geom.point,Theme(default_color=color("blue"))),
      layer(x=y[tst],y=fm.yHat[tst],Geom.point,Theme(default_color=color("red"))),
      Guide.ylabel("yHat"),
      Guide.xlabel("y"),
      Guide.title("Observed vs predicted"))


#Assesment of prediction accuracy using multiple training-testing partition
#Box 13 in BGLR package
using BGLR
using StatsBase
using Gadfly

# Reading Data
 #Markers
  X=readcsv(joinpath(Pkg.dir(),"BGLR/data/wheat.X.csv");header=true)[1];
 #Phenotypes
  y=readcsv(joinpath(Pkg.dir(),"BGLR/data/wheat.Y.csv");header=true)[1][:,1];

 #Relationship matrix derived from pedigree
  A=readcsv(joinpath(Pkg.dir(),"BGLR/data/wheat.A.csv");header=true);
  A=A[1]; #The first component of the Tuple

# Simulation parameters
  srand(123);
  nTST=150;
  nRep=100;
  nIter=12000;
  burnIn=2000; 

# Computing G-Matrix
  n,p=size(X);
  X=scale(X);
  G=X*X';
  G=G./p;

# Setting the linear predictors
#Very weird, if you run the model several times with different 
#missing value patterns and the same Dictionaries, the objects 
#become some how corrupted and then the variances are very high and the 
#predictions very bad!, but if you define the dictionary inside the call to
#bglr function it works
#  H0=Dict("PED"=>RKHS(K=A));
#  HA=Dict("PED"=>RKHS(K=A),
#          "MRK"=>RKHS(K=G));

  COR=zeros(nRep,2);

# Loop over TRN-TST partitions
  for i in 1:nRep
    println("i=",i)
    tst=sample([1:n],nTST;replace=false)
    yNA=deepcopy(y)
    yNA[tst]=-999
    fm=bglr(y=yNA,ETA=Dict("PED"=>RKHS(K=A));nIter=nIter,burnIn=burnIn);
    COR[i,1]=cor(fm.yHat[tst],y[tst]);
    fm=bglr(y=yNA,ETA=Dict("PED"=>RKHS(K=A),"MRK"=>RKHS(K=G));nIter=nIter,burnIn=burnIn);
    COR[i,2]=cor(fm.yHat[tst],y[tst]);
  end

#Plots
plot(layer(x=COR[:,1],y=COR[:,2],Geom.point,Theme(default_color=color("red"))),
     layer(x=[0,0.6],y=[0,0.6],Geom.line,Theme(default_color=color("black"))),
     Guide.xlabel("Pedigree"),
     Guide.ylabel("Pedigree+Markers"),
     Guide.title("E1"))


#Bayesian LASSO
 using BGLR

#Reading Data 
 X=readcsv(joinpath(Pkg.dir(),"BGLR/data/wheat.X.csv");header=true)[1];
 y=readcsv(joinpath(Pkg.dir(),"BGLR/data/wheat.Y.csv");header=true)[1][:,1];

# Bayesian Ridge LASSO
 ETA=Dict("mrk"=>BL(X))
 fm=bglr(y=y,ETA=ETA);



#Bayesian LASSO
 using BGLR
 using Gadfly

#Reading Data
 X=readcsv(joinpath(Pkg.dir(),"BGLR/data/wheat.X.csv");header=true)[1];
 y=readcsv(joinpath(Pkg.dir(),"BGLR/data/wheat.Y.csv");header=true)[1][:,1];

# Bayesian Ridge Regression
 ETA=Dict("mrk"=>BL(X))
 fm=bglr(y=y,ETA=ETA);

#Plots
 plot(x=fm.y,y=fm.yHat)

#BayesA
  using BGLR
  using Gadfly

# Reading Data 
  X=readcsv(joinpath(Pkg.dir(),"BGLR/data/wheat.X.csv");header=true)[1];
  y=readcsv(joinpath(Pkg.dir(),"BGLR/data/wheat.Y.csv");header=true)[1][:,1];

# BayesA
  ETA=Dict("mrk"=>BayesA(X))
  fm=bglr(y=y,ETA=ETA);
   
#Plots
  plot(x=fm.y,y=fm.yHat)

#BayesB
  using BGLR
  using Gadfly

# Reading Data
  X=readcsv(joinpath(Pkg.dir(),"BGLR/data/wheat.X.csv");header=true)[1];
  y=readcsv(joinpath(Pkg.dir(),"BGLR/data/wheat.Y.csv");header=true)[1][:,1];

# BayesB
  ETA=Dict("mrk"=>BayesB(X))
  fm=bglr(y=y,ETA=ETA);

#Plots
  plot(x=fm.y,y=fm.yHat)


############################################################################################################
#Example 1
#Box 6

using BGLR
using Distributions
using Gadfly


# Reading data (markers are in BED binary format, http://pngu.mgh.harvard.edu/~purcell/plink/binary.shtml).
  X=read_bed(joinpath(Pkg.dir(),"BGLR/data/mice.X.bed"),1814,10346);

# Simulation

  srand(456);
  n,p=size(X);
  X=scale(X);

  h2=0.5;
  nQTL=10;
  whichQTL=[517,1551,2585,3619,4653,5687,6721,7755,8789,9823];
  b0=zeros(p);
  b0[whichQTL]=rand(Normal(0,sqrt(h2/nQTL)),10);
  signal=X*b0;
  error=rand(Normal(0,sqrt(1-h2)),n);
  y=signal+error;

#Bayesian Ridge Regression
  ETA_BRR=Dict("BRR"=>BRR(X))

#BayesA
  ETA_BA=Dict("BA"=>BayesA(X))

#BayesB
  ETA_BB=Dict("BB"=>BayesB(X))
 
#Fitting models
  fmBRR=bglr(y=y,ETA=ETA_BRR,nIter=10000,burnIn=5000);
  fmBA=bglr(y=y,ETA=ETA_BA,nIter=10000,burnIn=5000);
  fmBB=bglr(y=y,ETA=ETA_BB,nIter=10000,burnIn=5000);

#Plots
  plot(x=fmBB.y,y=fmBB.yHat)

  plot(layer(x=[1:p],y=abs(fmBB.ETA["BB"].post_effects),Geom.point,Theme(default_color=color("red"))),
       layer(x=whichQTL,y=abs(b0[whichQTL]),Geom.point,Theme(default_color=color("blue"))))

  p1=plot(x=[1:p],y=abs(fmBB.ETA["BB"].post_effects),
          xintercept=whichQTL,
	  Geom.point,
          Geom.vline(color=color("blue")),
          Theme(default_color=color("red"),grid_color=color("white"),
                grid_color_focused=color("white"),highlight_width=0mm,
                default_point_size=1.5pt,
                panel_stroke=color("black")),
          Guide.xticks(ticks=[0,2000,4000,6000,8000,10000]),
          Guide.xlabel("Marker position (order)"),
          Guide.ylabel("|&beta;<sub>j</sub>|"))
  q1=layer(x=whichQTL,y=abs(b0[whichQTL]),Geom.point,Theme(default_color=color("blue"))); 
  append!(p1.layers,q1);

  display(p1)
