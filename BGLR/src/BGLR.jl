#Define the module BGLR
#Last update March/15/2016

module BGLR

export
	bglr,
	GP,
	BRR

import
	Distributions.Normal,
	Distributions.Chisq,
        Distributions.Gamma,
	Base.LinAlg.BLAS.axpy!
	


#This routine was adapted from rinvGauss function from S-Plus
# Random variates from inverse Gaussian distribution
# Reference:
#      Chhikara and Folks, The Inverse Gaussian Distribution,
#      Marcel Dekker, 1989, page 53.
# GKS  15 Jan 98

function rinvGauss(nu::Float64, lambda::Float64)
        tmp = randn(1);
        y2 = tmp[1]*tmp[1];
        u = rand(1);
        u=u[1];
        r1 = nu/(2*lambda) * (2*lambda + nu*y2 - sqrt(4*lambda*nu*y2 + nu*nu*y2*y2));
        r2 = nu*nu/r1;
        if(u < nu/(nu+r1))
                return(r1)
        else
                return(r2)
        end
end


#=
 * This is a generic function to sample betas in various models, including 
 * Bayesian LASSO, BayesA, Bayesian Ridge Regression, etc.
 
 * For example, in the Bayesian LASSO, we wish to draw samples from the full 
 * conditional distribution of each of the elements in the vector bL. The full conditional 
 * distribution is normal with mean and variance equal to the solution (inverse of the coefficient of the left hand side)
 * of the following equation (See suplementary materials in de los Campos et al., 2009 for details),
   
    (1/varE x_j' x_j + 1/(varE tau_j^2)) bL_j = 1/varE x_j' e
 
    or equivalently, 
    
    mean= (1/varE x_j' e)/ (1/varE x_j' x_j + 1/(varE tau_j^2))
    variance= 1/ (1/varE x_j' x_j + 1/(varE tau_j^2))
    
    xj= the jth column of the incidence matrix
    
 *The notation in the routine is as follows:
 
 n: Number of rows in X
 pL: Number of columns in X
 XL: the matrix X stacked by columns
 XL2: vector with x_j' x_j, j=1,...,p
 bL: vector of regression coefficients
 e: vector with residuals, e=y-yHat, yHat= predicted values
 varBj: vector with variances, 
	For Bayesian LASSO, varBj=tau_j^2 * varE, j=1,...,p
	For Ridge regression, varBj=varB, j=1,...,p, varB is the variance common to all betas.
	For BayesA, varBj=varB_j, j=1,...,p
	For BayesCpi, varBj=varB, j=1,...,p, varB is the variance common to all betas
	
 varE: residual variance
 minAbsBeta: in some cases values of betas near to zero can lead to numerical problems in BL, 
             so, instead of using this tiny number we assingn them minAbsBeta
 
=#

function sample_beta(n::Int64, p::Int64, X::Array{Float64,2},x2::Array{Float64,1},
		     b::Array{Float64,1},error::Array{Float64,1},varBj::Array{Float64,1},
		     varE::Float64;minAbsBeta=1e-9)

	for j in 1:p
                bj=b[j]
                rhs=dot(X[:,j],error)/varE
                rhs+=x2[j]*b/fm.varE
                c=x2[j]/varE + 1.0/varBj
                b[j]=rhs/c+sqrt(1/c)*rand(Normal(0,1))
                bj=bj-b[j]
                axpy!(bj,X[:,j],error)
        end 

end

############################
function writeln(con, x, delim)
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


streamOrASCIIString=Union{ASCIIString,IOStream}

## Linear Term: INTercept
type INT
  name::ASCIIString
  mu::Float64
  post_mu::Float64
  post_mu2::Float64
  post_SD_mu::Float64
  fname::ASCIIString
  con::streamOrASCIIString
  nSums::Int64
  k::Float64
end

function INT(y)
   name="intercept"
   return INT(name,mean(y),0,0,0,"","",0,0)
end
# Example: tmp=INT(rand(10))

## Linear Term: Fixed Effects
type FixEff
  name::ASCIIString
  X::Array{Float64}
  p::Int64
  effects::Array{Float64}
  post_effects::Array{Float64}
  post_effects2::Array{Float64}
  post_SD_effects::Array{Float64}
  fname::ASCIIString
  con::streamOrASCIIString
  nSums::Int64
  k::Float64
end

function  FixEff(X::Array{Float64};name="fix")
   n,p=size(X)
   return FixEff(name,X,p,zeros(p),zeros(p),zeros(p),zeros(p),"","",0,0)
end

#Example: FixEff(rand(4,3))


## Linear Term: Gaussian Process
type GP # Gaussian Process
  name::ASCIIString
  n::Int64 # number or individuals
  p::Int64 # number of vectors
  vectors::Array{Float64,2} # eigenvectors (V)
  values::Array{Float64,1} # eigenvalues (d)
  effects::Array{Float64,1} # effects of eigen-vectors (b)
  eta::Array{Float64,1} # V*b
  R2::Float64
  df0::Float64 #prior degree of freedom
  S0::Float64  #prior scale
  df::Float64  #degree of freedom of the conditional distribution
  var::Float64 # variance of effects
  post_var::Float64 # posterior mean
  post_var2::Float64 # posterior mean of the squared of the variance
  post_SD_var::Float64 # posterior standard deviation
  post_effects::Array{Float64,1}
  post_effects2::Array{Float64,1}
  post_SD_effects::Array{Float64,1}
  post_eta::Array{Float64,1} #1 posterior mean of linear term
  post_eta2::Array{Float64,1} # posterior mean of the linear term squared
  post_SD_eta::Array{Float64,1} # posterior SD of the linear term
  fname::ASCIIString
  con::streamOrASCIIString # a connection where samples will be saved
  nSums::Int64
  k::Float64
end

function GP(;K="null",EVD="null",R2=-Inf,df0= -Inf,S0=-Inf,minEigValue=1e-7,name="") 
  if(EVD=="null")
     if(K=="null")
        error("Please provide either K (symmetric positive semi-definite matrix) or its eigen-value decomposition (EVD=eigfact(K)).")
       else
          EVD=eigfact(K)
       end
     end
  
  keepVector=EVD[:values].>minEigValue
  n= size(EVD[:vectors])[1]
  p=sum(keepVector)
  V=EVD[:vectors][:,keepVector]
  d=EVD[:values][keepVector]
  for i in 1:p
     V[:,i]*=sqrt(d[i])
  end
  return GP(name,n,p,V,d,zeros(p),zeros(n),R2,df0,S0,df0+p,0.0,0.0,0.0,0.0,zeros(p),zeros(p),zeros(p),zeros(n),zeros(n),zeros(n),"","",0,0)

end
#Example: tmp=GP(K=eye(3))


## Linear Term: RandReg
type RandReg
  prior::ASCIIString #Prior Distribution for effects, it can be "BL", "BRR", "BayesA", "BayesB", "BayesCpi"
  name::ASCIIString
  n::Int64 # number or individuals
  p::Int64 # number of vectors
  X::Array{Float64,2} # incidence matrix
  effects::Array{Float64,1} # effects of eigen-vectors (b)
  var::Array{Float64,1} # variance of effects
  d::Array{Bool,1} # is the marker in the model?
  x2::Array{Float64,1} # sum of squares of columns of X
  eta::Array{Float64,1} # linear predictor: X*b
  probIn::Float64 # average prob of marker being in the model
  scaleCol::Bool # scale columns?
  centerCol::Bool # center columns?
  R2::Float64 # R-squared of the term (used to set hyperparameters)
  
  df0::Float64 #prior degree of freedom
  S0::Float64  #prior scale
  countsIn::Float64 # hyper-parameters of the beta prior
  countsOut::Float64
  df::Float64  #degree of freedom of the conditional distribution

  post_effects::Array{Float64,1}
  post_effects2::Array{Float64,1}
  post_eta::Array{Float64,1}
  post_eta2::Array{Float64,1}
  
  post_d::Array{Float64,1}
  post_d2::Array{Float64,1}
 
  post_probIn::Float64
  post_probIn2::Float64
  post_var::Array{Float64,1}
  post_var2::Array{Float64,1}
end
##

#function BayesB
#function BayesA
#function BRR
#function BayesCP
#function BRR_groups
#function BL


## Linear Term: BRR
type RandRegBRR # Bayesian Ridge Regression
  name::ASCIIString
  n::Int64 # number or individuals
  p::Int64 # number of vectors
  X::Array{Float64,2} # incidence matrix
  x2::Array{Float64,1} # sum of squares of columns of X
  effects::Array{Float64,1} # b
  eta::Array{Float64,1} # V*b
  R2::Float64
  df0::Float64 #prior degree of freedom
  S0::Float64  #prior scale
  df::Float64  #degree of freedom of the conditional distribution
  var::Float64 # variance of effects
  post_var::Float64 # posterior mean
  post_var2::Float64 # posterior mean of the squared of the variance
  post_SD_var::Float64 # posterior standard deviation
  post_effects::Array{Float64,1}
  post_effects2::Array{Float64,1}
  post_SD_effects::Array{Float64,1}
  post_eta::Array{Float64,1} #1 posterior mean of linear term
  post_eta2::Array{Float64,1} # posterior mean of the linear term squared
  post_SD_eta::Array{Float64,1} # posterior SD of the linear term
  fname::ASCIIString
  con::streamOrASCIIString # a connection where samples will be saved
  nSums::Int64
  k::Float64
end


#Function to setup RandReg
#When the prior for the coefficients is BRR
#FIXME: Set the S0

function BRR(X::Array{Float64,2};R2=0.5,df0= 5,S0=0.015,name="")

	n,p=size(X);  #sample size and number of predictors
	
	#The sum of squares of columns of X
	x2=zeros(Float64,p);
    
    	for j in 1:p
    		x2[j]=sum(X[:,j].^2);
    	end

	#sumMaeanXSq 
	sumMeanXSq=0.0
	for j in 1:p
		sumMeanXSq+=(mean(X[:,j]))^2
	end

	#FIXME: In the R version we need y in order to compute the 
	#Default scale parameter

	if(S0<0)
		MSx=sum(x2)/n-sumMeanXSq
	end

	return RandRegBRR(name,n,p,X,x2,zeros(p),zeros(n),R2,df0,S0,df0+p,0.0,0.0,0.0,0.0,zeros(p),zeros(p),zeros(p),zeros(n),zeros(n),zeros(n),"","",0,0)

end

#Example: BRR(rand(4,3))

type BGLRt
  y::Array{Float64}
  yStar::Array{Float64}
  yHat::Array{Float64}
  error::Array{Float64}
  post_yHat::Array{Float64}
  post_yHat2::Array{Float64}
  post_SD_yHat::Array{Float64}

  naCode::Float64
  hasNA::Bool
  nNA::Int64
  isNA::Array{Bool}

  ETA::Dict
  nIter::Int64
  burnIn::Int64
  thin::Int64
  R2::Float64
  verbose::Bool
  path::ASCIIString
  n::Int64
  varE::Float64
  df0::Float64
  S0::Float64
  df::Float64
  post_varE::Float64
  post_varE2::Float64
  post_SDVarE::Float64
  updateMeans::Bool
  saveSamples::Bool
  conVarE::IOStream
end

#Function to compute the sum of squares of the entries of a vector

function sumsq(x::Vector{Float64});
	return(sum(x.^2))
end


function bglr(;y="null",ETA=Dict(),nIter=1500,R2=.5,burnIn=500,thin=5,saveAt=string(pwd(),"/"),verbose=true,df0=1,S0=-Inf,naCode= -999)
   #y=rand(10);ETA=Dict();nIter=-1;R2=.5;burnIn=500;thin=5;path="";verbose=true;df0=0;S0=0;saveAt=pwd()*"/"

   if(y=="null")
      error("Provide the response (y).")
   end
   
   # initializing yStar
    yStar=deepcopy(y)
    isNA= (y.== naCode)
    hasNA=any(isNA)
    nNA=sum(isNA)
    yStar[isNA]=mean(y[!isNA])
   
   ## error variance
   Vy=var(yStar)
   if (S0<0)
      S0=(1-R2)*(df0+2)*Vy
   end
   
   ### Initializing the linear predictor
   LP=Dict()
   LP["INT"]=INT(yStar)
  
   #term[2] has information related to a type
   #term[1] has information related to a key in the dictionary
     
   if(length(ETA)>0)
     for term in ETA
        if(typeof(term[2])==GP || 
           typeof(term[2])==FixEff ||
	   typeof(term[2])==RandRegBRR)
        	
        	if(typeof(term[2])==GP)

            	   LP[term[1]]=term[2]
               	   
		   # Setting default values
            	   if(LP[term[1]].df0<0)
               		LP[term[1]].df0=df0
               		LP[term[1]].df=df0+LP[term[1]].p
            	   end
            	   
		   if(LP[term[1]].R2<0)
               		LP[term[1]].R2=R2/length(ETA)
            	   end   
            	
		   if(LP[term[1]].S0<0)
               		LP[term[1]].S0=Vy*(LP[term[1]].df0+2)*LP[term[1]].R2/mean(LP[term[1]].values)
            	   end  
           	
		   if(LP[term[1]].var==0)
               	        LP[term[1]].var=LP[term[1]].S0/(LP[term[1]].df0+2)
            	   end 

         	end #end of if for GP

		if(typeof(term[2])==FixEff)
			#Add your magic code here
		end

		if(typeof(term[2])==RandRegBRR)
			#Add your magic code here
			LP[term[1]]=term[2]
		end

              else 
        	error("The elements of ETA must of type GP, FixEff or RandRegBRR")
	      end
     end #end for    
   end #end if length(ETA)

   ## Opening connections
   for term in LP
   	  term[2].name=term[1]

   	  if(typeof(term[2])==INT)
   	  	term[2].fname=string(saveAt,term[2].name,"_mu.dat")
   	  end

   	  if(typeof(term[2])==GP)
   	  	term[2].fname=string(saveAt,term[2].name,"_var.dat")
   	  end

   	  if(typeof(term[2])==FixEff)
   	  	term[2].fname=string(saveAt,term[2].name,"_b.dat")
   	  end
	
	  if(typeof(term[2])==RandRegBRR)
		#Add your magic code here
		term[2].fname=string(saveAt,term[2].name,"_var.dat")
	  end
	

	  term[2].con=open(term[2].fname,"w+")
   end
   
   mu=mean(yStar)
   n=length(y)
   yHat=ones(n).*mu
   resid=yStar-yHat

   post_yHat=zeros(n)
   post_yHat2=zeros(n)

   nSums=0
   k=0.0
   

   fm=BGLRt(y,yStar,yHat,resid,zeros(n),zeros(n),zeros(n),
   	    naCode,hasNA,nNA,isNA,
   	    LP,nIter,burnIn,thin,R2,verbose,
            saveAt,n,Vy*(1-R2),df0,S0,df0+n,0,0,0,false,false,open(saveAt*"varE.dat","w+"))
              
   if (nIter>0)
   	for i in 1:nIter ## Sampler

   		## determining whether samples or post. means need to be updated
   		
		fm.saveSamples=(i%thin)==0
   		
		fm.updateMeans=fm.saveSamples&&(i>burnIn)
  		
		if fm.updateMeans
  		  	nSums+=1
  		  	k=(nSums-1)/nSums
  		end
  		
  		## Sampling effects and other parameters of the LP
  		for ETA in fm.ETA    ## Loop over terms in the linear predictor
     			
			if(typeof(ETA[2])==INT)
       				fm=updateInt(fm,ETA[1],fm.updateMeans,fm.saveSamples,nSums,k)
       			end     		  

			if(typeof(ETA[2])==GP)
     		  	 	fm=updateGP(fm,ETA[1],fm.updateMeans,fm.saveSamples,nSums,k)
    			end

			if(typeof(ETA[2])==FixEff)
				#Add your magic code here
			end

			if(typeof(ETA[2])==RandRegBRR)
				fm=updateRandRegBRR(fm,ETA[1],fm.updateMeans,fm.saveSamples,nSums,k)
			end	
  		end

  		## Updating error variance
  		
		SS=sumsq(fm.error)+fm.S0
  		fm.varE= SS/rand(Chisq(fm.df),1)[]
  		
  		
  		## Updating error, yHat & yStar
  		fm.yHat=fm.yStar-fm.error
  		  		
  		if(hasNA)
	  		fm.error[fm.isNA]=rand(Normal(0,sqrt(fm.varE)),fm.nNA)
  			fm.yStar[fm.isNA]=fm.yHat[fm.isNA]+fm.error[fm.isNA]
		end
  		
  		if(fm.updateMeans)
  			fm.post_varE=fm.post_varE*k+fm.varE/nSums
  			fm.post_varE2=fm.post_varE2*k+(fm.varE^2)/nSums

  			fm.post_yHat=fm.post_yHat*k+fm.yHat/nSums
			fm.post_yHat2=fm.post_yHat2*k+(fm.yHat.^2)/nSums

  		end
  		if verbose 
  			println("Iter: ",i," VarE=",round(fm.varE,4)) 
  		end
	 end # end of sampler
    end # end of nIter>0
	
    ## Closing connections
    for term in LP
   	   close(term[2].con)
    end
   
    ## Compute posterior SDs
   	fm.post_SD_yHat=sqrt(fm.post_yHat2-fm.post_yHat.^2)
	
	for ETA in fm.ETA 
	  if(typeof(ETA[2])==INT)
	     ETA[2].post_SD_mu=sqrt(ETA[2].post_mu2-ETA[2].post_mu^2)
	  end


	  if(typeof(ETA[2])==GP)
	      ETA[2].post_SD_effects=sqrt(ETA[2].post_effects2-ETA[2].post_effects.^2)
	      ETA[2].post_SD_eta=sqrt(ETA[2].post_eta2-ETA[2].post_eta.^2)
	      ETA[2].post_SD_var=sqrt(ETA[2].post_var2-ETA[2].post_var^2)
	  end
	  
	  if(typeof(ETA[2])==FixEff)
	     #Add your magic code here 	
	  end

	  if(typeof(ETA[2])==RandRegBRR)
             #Add your magic code here
	  end
  
	end #end of for
	
   	return fm
end

## Test
 #y=rand(10)
 #TMP=["mrk"=>GP(K=eye(10))]
 #fm=BGLR(y=y,ETA=TMP,R2=.9)
 
# test


##Update methods

#Update Intercept

function updateInt(fm::BGLRt,label::ASCIIString,updateMeans::Bool,saveSamples::Bool,nSums::Int,k::Float64)
    
	fm.error+=fm.ETA[label].mu
	fm.ETA[label].mu=rand(Normal(mean(fm.error),sqrt(fm.varE/fm.n)))
    	fm.error-=fm.ETA[label].mu

   	if(saveSamples) 

   		writeln(fm.ETA[label].con,fm.ETA[label].mu,"") 
   		if(updateMeans)
   			fm.ETA[label].post_mu=fm.ETA[label].post_mu*k+fm.ETA[label].mu/nSums
   			fm.ETA[label].post_mu2=fm.ETA[label].post_mu2*k+(fm.ETA[label].mu^2)/nSums
   		end
	end

	return fm
end 
 
function updateGP(fm::BGLRt,label::ASCIIString,updateMeans::Bool,saveSamples::Bool,nSums::Int,k::Float64)
	
	axpy!(1,fm.ETA[label].eta ,fm.error)# updating errors
	rhs=fm.ETA[label].vectors'fm.error
	lambda=fm.varE/fm.ETA[label].var
	lhs=fm.ETA[label].values+lambda
	CInv=1./lhs
	sol=CInv.*rhs
	SD=sqrt(CInv)
	fm.ETA[label].effects=sol+rand(Normal(0,sqrt(fm.varE)),fm.ETA[label].p).*SD
	fm.ETA[label].eta=fm.ETA[label].vectors*fm.ETA[label].effects
	axpy!(-1,fm.ETA[label].eta ,fm.error)# updating errors
	     
	SS=sumsq(fm.ETA[label].effects)+fm.ETA[label].S0
	fm.ETA[label].var=SS/rand(Chisq(fm.ETA[label].df),1)[]	
    
    	if(saveSamples)

	    writeln(fm.ETA[label].con,fm.ETA[label].var,"") 
	    
	    if(updateMeans)
   			fm.ETA[label].post_effects=fm.ETA[label].post_effects*k+fm.ETA[label].effects/nSums
			fm.ETA[label].post_effects2=fm.ETA[label].post_effects2*k+(fm.ETA[label].effects.^2)/nSums

   			fm.ETA[label].post_eta =fm.ETA[label].post_eta*k+fm.ETA[label].eta/nSums
   			fm.ETA[label].post_eta2=fm.ETA[label].post_eta2*k+(fm.ETA[label].eta.^2)/nSums

   			fm.ETA[label].post_var=fm.ETA[label].post_var*k+fm.ETA[label].var/nSums
   			fm.ETA[label].post_var2=fm.ETA[label].post_var2*k+(fm.ETA[label].var^2)/nSums

	    end
	end
	
	return fm
end

function innersimd(x, y)
    s = zero(eltype(x))
    @simd for i=1:length(x)
        @inbounds s += x[i]*y[i]
    end
    s
end

function my_axpy(a,x,y)
    @simd for i=1:length(x)
	@inbounds y[i]=a*x[i]+y[i]	
    end
end

#Update RandRegBRR
function updateRandRegBRR(fm::BGLRt, label::ASCIIString, updateMeans::Bool, saveSamples::Bool, nSums::Int, k::Float64)
	
	p=fm.ETA[label].p
	n=fm.ETA[label].n
	
	#Sample beta, julia native code
	#Just the same function in BGLR-R rewritten	

	for j in 1:p
		b=fm.ETA[label].effects[j]
		xj=fm.ETA[label].X[:,j]
		#rhs=dot(xj,fm.error)/fm.varE
		rhs=innersimd(xj,fm.error)/fm.varE
		rhs+=fm.ETA[label].x2[j]*b/fm.varE
		c=fm.ETA[label].x2[j]/fm.varE + 1.0/fm.ETA[label].var
		fm.ETA[label].effects[j]=rhs/c+sqrt(1/c)*rand(Normal(0,1));
		b=b-fm.ETA[label].effects[j]
		#axpy!(b,xj,fm.error)
		my_axpy(b,xj,fm.error)
	end

	#Calling a function in C

	#ccall((:sample_beta,"/Users/paulino/Documents/Documentos Paulino/Estancia USA-Michigan/julia/sample_betas_julia.so"),
      	#	Void,(Int32, Int32, Ptr{Float64},Ptr{Float64},Ptr{Float64},Ptr{Float64},Float64,Float64,Float64),
      	#	Int32(n),Int32(p),fm.ETA[label].X,fm.ETA[label].x2,fm.ETA[label].effects,fm.error,fm.ETA[label].var,fm.varE,Float64(1e-7)
      	#     )

	SS=sumsq(fm.ETA[label].effects)+fm.ETA[label].S0
	fm.ETA[label].var=SS/rand(Chisq(fm.ETA[label].df),1)[]
	
	
	if(saveSamples)
            writeln(fm.ETA[label].con,fm.ETA[label].var,"")

            if(updateMeans)
                        fm.ETA[label].post_effects=fm.ETA[label].post_effects*k+fm.ETA[label].effects/nSums
                        fm.ETA[label].post_effects2=fm.ETA[label].post_effects2*k+(fm.ETA[label].effects.^2)/nSums

			#Do we need eta?
                        fm.ETA[label].post_eta =fm.ETA[label].post_eta*k+fm.ETA[label].eta/nSums
                        fm.ETA[label].post_eta2=fm.ETA[label].post_eta2*k+(fm.ETA[label].eta.^2)/nSums

                        fm.ETA[label].post_var=fm.ETA[label].post_var*k+fm.ETA[label].var/nSums
                        fm.ETA[label].post_var2=fm.ETA[label].post_var2*k+(fm.ETA[label].var^2)/nSums

            end
        end
	return fm
end

end #module end
