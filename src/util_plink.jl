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

function read_bed(bed_file::UTF8String,n::Int64,p::Int64)

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
                			out[(l+(j*n))+1]=recode[1+code]
            			end                          
    			end
    		end
	end
	
	X=reshape(X,(n,p))

	return(X)
end

#Testing
#X=read_bed("mice.X.bed",1814,10346);
