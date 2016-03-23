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
