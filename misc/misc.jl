######################

#function sample(x;n=size(x)[1],replace=true)
#  nX=size(x)[1]
  
#  if (n > size(x)[1])&(!replace)
#     error("If replace=false, n must be <= size(x)[1].")
#  end
  
 # if replace
 #   myIndex=rand(1:nX,n)
 # else
 #   myIndex=sortperm(rand(nX))[1:n]
 # end
  
  # return x[myIndex]
#end

######################
function mean_narm(x,naCode)
   sums=0
   counts=0
   for i in 1:length(x)
     if x[i]!=naCode
       sums=sums+x[i]
       counts=counts+1
     end
   end
   mu=sums/counts
   return mu
end
#######################
function var_narm(x,naCode)
   sums=0
   sums2=0
   counts=0
   for i in 1:length(x)
     if x[i]!=naCode
       sums=sums+x[i]
       sums2=sums2+x[i]^2
       counts=counts+1
     end
   end
   tmp=(sums2/counts)-(sums/counts)^2
   return tmp
end

#### scale ################
function scaleX(X;scale=true, center=true)
    n,p=size(X)
    if (center&scale)
       for i in 1:p
          X[:,i]=(X[:,i]-mean(X[:,i]))/std(X[:,i])
       end
    end

    if (center&(!scale))
       for i in 1:p
          X[:,i]=X[:,i]-mean(X[:,i])
       end
    end

    if (!center)&scale
       for i in 1:p
          X[:,i]=X[:,i]/std(X[:,i])
       end
    end   
    return(X) 
end

#### Frequency table ###
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
##########################
function vech (x::Array{Float64})
	p=size(x)[1]
	out=zeros(Float64,convert(Int64,p*(p+1)/2))
	pos=1
	for i in 1:p
	  for j in i:p
	     out[pos]=x[i,j]
	     pos=pos+1
	   end
	end
	return out
end
############################
function writeln (con, x, delim)
 n=length(x)
 if n>1
   for i in 1:(n-1)
     write(con,string(x[i],delim))
   end
   write(con,string(x[n]))
 else
    write(con,string(x))
 end
 write(con,"\n") 
 flush(con)
end
############################
function sumVar (x)
    K=0
    n,p=size(x)
    for i in 1:p
    	K=K+var(x[:,i])*(n-1)/n
    end
   return K
end
############################

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
###########################

 function rep(x;each=0,times=0)

	tmp=( (each>0)&(times<=0))|((each<=0)&(times>0))  
	
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
 