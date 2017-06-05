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


#FIXME:
#This is a preeliminary version, it is not memory efficient
#check also coding for missing values

"""
read_bed(bed_file::String,n::Int64,p::Int64)
Function that reads genotype information stored in binary BED files used in plink. 

# Arguments:

* `bed_file::String`: path to BED file.
* `n::Int64`: Number of individuals.
* `p::Int64`: Number of SNPs.

# Returns:

* A matrix with the genotypic information.

"""

function read_bed(bed_file::String,n::Int64,p::Int64)

	s = open(bed_file)
	read(s,UInt8) == 0x6c && read(s,UInt8) == 0x1b || error("Unknown file format")
	read(s,UInt8) == 1 || error("Only snp and individual major order are supported")
	m = div(n+3,4)
	bb=Mmap.mmap(s,Array{UInt8,2},(m,p))
	close(s)
	
	mask=0x03
	recode = [0,2,1,3]

	X=zeros(n*p)

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
                			X[(l+(j*n))+1]=recode[1+code]
            			end                          
    			end
    		end
	end
	
	X=reshape(X,(n,p))

	return(X)
end

#Testing
#X=read_bed("mice.X.bed",1814,10346);

##########################################################################################
#model matrix for a factor using dummy variables
#if intercept=true, use p-1 dummy variables
#if intercept=false, use p dummy variables
#where p is the number of levels

"""
model_matrix(x;intercept=true)
Function that creates a design (or model) matrix, by expanding factors to a set of dummary variables.

# Arguments:
* `x`: a vector that contains the information for the levels.
* `intecept::Bool`: if true, the routine generates the matrix using p-1 dummy variables. If false then the 
                  routine generates the matrix using p dummy variables.

#Returns:

* The design matrix.

"""

function model_matrix(x;intercept=true)
        levels=sort(unique(x))
        n=size(x)[1]
        p=size(levels)[1]

        if(p<2)
                error("The factor should have at least 2 levels")
        end

        X=zeros(n,p)

	if(intercept)
		X[:,1]=1
        	for j in 2:p
                	index=(x.==levels[j])
                	X[index,j]=1
        	end
	else
		for j in 1:p
			index=(x.==levels[j])
			X[index,j]=1
		end
	end
        X
end

##########################################################################################
#This routine appends a textline to
#to a file

function writeln(con, x, delim)
 n=length(x)
 if n>1
   for i in 1:(n-1)
     write(con,string(x[i],delim))
   end
   write(con,string(x[n]))
 else
    write(con,string(x[1]))
 end
 write(con,"\n")
 flush(con)
end

##########################################################################################
#function to get the levels of a factor
function levels(x::Array{Int64,1})
        sort(unique(x))
end

##########################################################################################
#function to get the number of levels of a factor
function nlevels(x::Array{Int64,1})
        size(levels(x))[1]
end

##########################################################################################
function renumber(x)
     counts=table(x)
     levels=counts[1]
     nLevels=size(counts[1])[1]
     n=size(x)[1]
     z=int(zeros(n))
     for i in 1:n
        for j in 1:nLevels
           if x[i]==levels[j]
              z[i]=j
           end
        end 
     end
     return z
end
##########################################################################################

"""
rep(x;each=0,times=0)
replicates the values in x.

# Arguments:

* `x`: a vector.
* `each::Int64`: An integer. Each element of x is repeated each times.
* `times::Int64`:An integer giving the number of times to repeat each element if of length(x) 
                 or to repeat the whole vector if of length 1.

# Returns:

* An object of the same type as x.

"""

function rep(x;each=0,times=0)

        tmp=((each>0)&(times<=0))|((each<=0)&(times>0))
        @assert  tmp "One and only one of \{'each'} or 'times' must be positive"

        nX=length(x)

        if(times>0)
                nZ=nX*times
                z=Array(typeof(x[1]),nZ)
                for i in 1:times
                        z[((i-1)*nX+1):(i*nX)]=x
                end
        else
                nZ=nX*each
                z=Array(typeof(x[1]),nZ)
                for i in 1:length(x)
                        z[((i-1)*each+1):(i*each)]=x[i]

                end
        end

        return z
end

##########################################################################################
#### Frequency table #####################################################################
#The function returns the counts
#The first component of the output corresponds to the levels of x,
#The second component of the output corresponds to the actual counts

function table(x::Array{Int64})
   n=length(x)
   counts=cell(2)
   labels=sort(unique(x))
   counts[1]=labels
   nGroups=length(labels)
   counts[2]=fill(0,nGroups)
   for i in 1:nGroups
         for j in 1:n
            counts[2][i]=counts[2][i]+ifelse(x[j]==labels[i],1,0)
         end
   end
   return counts
end

##########################################################################################
#Function to compute the sum of squares of the entries of a vector

function sumsq(x::Vector{Float64});
        return(sum(x.^2))
end

##########################################################################################

#Sum of squares by column and by groups
#It takes as argument a matrix
function sumsq_group(X::Array{Float64,2}, groups::Array{Int64,1})
        lev=levels(groups)
        nGroups=size(lev)[1]
        p=size(X)[2]
        x2=zeros(nGroups,p)

        for k in 1:nGroups
                #Temporary matrix with the rows that
                #belong to group_k
                tmp=X[groups.==lev[k],:]
                for j in 1:p
                        x2[k,j]=sum(tmp[:,j].^2)
                end
        end

        return(x2)
end

##########################################################################################

function innersimd(x, y,n)
    s = 0.0
    @simd for i=1:n
        @inbounds s += x[i]*y[i]
    end
    s
end

##########################################################################################

function my_axpy!(a,x,y,n)
    @simd for i=1:n
        @inbounds y[i]=a*x[i]+y[i]
    end
end

##########################################################################################

function scale(X::Array{Float64,2};center=true,scale=true)
    n,p=size(X)
    for j in 1:p
        xj=X[:,j]
        mu=mean(xj)
        SD=std(xj)
        X[:,j]=(xj-mu)/SD
    end
        X
end
