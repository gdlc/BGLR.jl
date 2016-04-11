#Define the module BGLR
#Last update April/8/2016

module BGLR

export
	bglr,
	RKHS,
	BRR,
	FixEff,
	read_bed,
	model_matrix

import
	Distributions.Normal,
	Distributions.Chisq,
        Distributions.Gamma,
	Base.LinAlg.BLAS.axpy!,
	ArrayViews.unsafe_view,
	Base.LinAlg.scale 

include("utils.jl")
include("samplers.jl")

##################################################################################################
#Just the welcome function that will appear every time that your run the program
function welcome()

  print("\n");
  print("#--------------------------------------------------------------------#\n");
  print("#        _\\\\|//_                                                     #\n");
  print("#       (` o-o ')      BGLR-J v0.01                                  #\n");
  print("#------ooO-(_)-Ooo---------------------------------------------------#\n");
  print("#                      Bayesian Generalized Linear Regression        #\n");
  print("#                      Gustavo de los Campos, gustavoc@msu.edu       #\n");
  print("#    .oooO     Oooo.   Paulino Perez Rodriguez, perpdgo@gmail.com    #\n");
  print("#    (   )     (   )   April, 2016                                   #\n");
  print("#_____\\ (_______) /_________________________________________________ #\n");
  print("#      \\_)     (_/                                                   #\n");
  print("#                                                                    #\n");
  print("#------------------------------------------------------------------- #\n");
  print("\n")

end
##################################################################################################

streamOrASCIIString=Union{ASCIIString,IOStream}

###################################################################################################################
#Begin BGLRt
###################################################################################################################

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
  varE::Array{Float64,1}  #It is an array because we can have different variances by group
  df0::Float64
  S0::Float64
  df::Float64
  #post_varE::Float64
  post_varE::Array{Float64,1} #It is an array because we can have different variances by group
  #post_varE2::Float64
  post_varE2::Array{Float64,1} #It is an array because we can have different variances by group
  #post_SDVarE::Float64
  post_SDVarE::Array{Float64,1} #It is an array because we can have different variances by group
  updateMeans::Bool
  saveSamples::Bool
  conVarE::IOStream
end

###################################################################################################################
#End BGLRt
###################################################################################################################

###################################################################################################################
#Begin INTercept
###################################################################################################################

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
   return INT("Intercept",mean(y),0,0,0,"","",0,0)
end

# Example: tmp=INT(rand(10))

#Update Intercept
function updateInt(fm::BGLRt,label::ASCIIString,updateMeans::Bool,saveSamples::Bool,nSums::Int,k::Float64, hasGroups::Bool, groups::Array{Int64,1})
    
	fm.error+=fm.ETA[label].mu

	if(!hasGroups)
		fm.ETA[label].mu=rand(Normal(mean(fm.error),sqrt(fm.varE[1]/fm.n)))
	else
		varEexpanded=fm.varE[groups]
		tmp=fm.error./varEexpanded
		rhs=sum(tmp)
		C=sum(fm.n./fm.varE)
		sol=rhs/C
		fm.ETA[label].mu=sol+sqrt(1/C)*rand(Normal(0,1))
	end

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


###################################################################################################################
#End INTercept
###################################################################################################################

###################################################################################################################
#Begin BRR
###################################################################################################################

## Linear Term: BRR
type RandRegBRR # Bayesian Ridge Regression
  name::ASCIIString
  n::Int64 # number or individuals
  p::Int64 # number of vectors
  X::Array{Float64,2} # incidence matrix
  x2::Array{Float64,2} # sum of squares of columns of X, it is a matrix to support heterogeneus variances
  effects::Array{Float64,1} # b
  eta::Array{Float64,1} # X*b
  R2::Float64
  df0::Float64 #prior degree of freedom
  S0::Float64  #prior scale
  df::Float64  #degrees of freedom of the conditional distribution
  var::Float64 # variance of effects
  update_var::Bool #Update the variance?, This is useful for FixedEffects
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
#When the prior for the coefficients is N(0,\sigma^2_beta*I)

function BRR(X::Array{Float64,2};R2=-Inf,df0=-Inf,S0=-Inf)

	n,p=size(X);  #sample size and number of predictors
	return RandRegBRR("BRR",n,p,X,zeros(1,p),zeros(p),zeros(n),R2,df0,S0,0.0,0.0,true,0.0,0.0,0.0,zeros(p),zeros(p),zeros(p),zeros(n),zeros(n),zeros(n),"","",0,0)
end

function BRR_post_init(LT::RandRegBRR, Vy::Float64, nLT::Int64, R2::Float64, hasGroups::Bool, groups::Array{Int64,1})

	#The sum of squares of columns of X
	if(!hasGroups)
		for j in 1:LT.p
                	LT.x2[1,j]=sum(LT.X[:,j].^2);
        	end
	else
		print("Groups is not null, computing SS by Group and by Column\n")
		LT.x2=sumsq_group(LT.X,groups)
	end	
	
	
	if(LT.df0<0)
		warn("Degrees of freedom of LP set to default value 5")
		LT.df0=5
		LT.df=LT.df0+LT.p
	end
	
	if(LT.R2<0)
		LT.R2=R2/nLT
	end

	if(LT.S0<0)
		 #sumMaeanXSq 
        	 sumMeanXSq=0.0
        	 for j in 1:LT.p
                	sumMeanXSq+=(mean(LT.X[:,j]))^2
        	 end

		 MSx=sum(LT.x2)/LT.n-sumMeanXSq 

		 LT.S0=((Vy*LT.R2)/(MSx))*(LT.df0+2) 
		 warn("Scale parameter of LT set to default value ", LT.S0)
	end
end


#Update RandRegBRR
function updateRandRegBRR(fm::BGLRt, label::ASCIIString, updateMeans::Bool, saveSamples::Bool, nSums::Int, k::Float64)
	
	p=fm.ETA[label].p
	n=fm.ETA[label].n
	
	#Implementation, Calling C

	#=
	ccall((:sample_beta,"sample_betas_julia.so"),
      		Void,(Int32, Int32, Ptr{Float64},Ptr{Float64},Ptr{Float64},Ptr{Float64},Float64,Float64,Float64),
      		Int32(n),Int32(p),fm.ETA[label].X,fm.ETA[label].x2,fm.ETA[label].effects,fm.error,fm.ETA[label].var,fm.varE,Float64(1e-7)
      	     )

	=#

	# Implementation using unsafe_view, @inbounds and @simd,  with a few improvements
     
	z=rand(Normal(0,sqrt(fm.varE[1])),fm.ETA[label].p)
    	lambda=fm.varE[1]/fm.ETA[label].var
    	rhs=0.0
    	for j in 1:p         
    		b=fm.ETA[label].effects[j] 
    		SSX=fm.ETA[label].x2[1,j]
		xj=unsafe_view(fm.ETA[label].X, :, j)
		rhs=innersimd(xj,fm.error,fm.n)+SSX*b
		CInv=1/(SSX + lambda)
		fm.ETA[label].effects[j]=rhs*CInv+sqrt(CInv)*z[j]
		tmp=b-fm.ETA[label].effects[j]
		my_axpy!(tmp,xj,fm.error,fm.n)		
	end	
    
	#Update the variance?, it will be true for BRR, but not for FixedEffects
	if(fm.ETA[label].update_var)
	
		SS=sumsq(fm.ETA[label].effects)+fm.ETA[label].S0
		fm.ETA[label].var=SS/rand(Chisq(fm.ETA[label].df),1)[]
	end
	
	if(saveSamples)
            writeln(fm.ETA[label].con,fm.ETA[label].var,"")

            if(updateMeans)
                        fm.ETA[label].post_effects=fm.ETA[label].post_effects*k+fm.ETA[label].effects/nSums
                        fm.ETA[label].post_effects2=fm.ETA[label].post_effects2*k+(fm.ETA[label].effects.^2)/nSums

			#Do we need eta?
                        #fm.ETA[label].post_eta =fm.ETA[label].post_eta*k+fm.ETA[label].eta/nSums
                        #fm.ETA[label].post_eta2=fm.ETA[label].post_eta2*k+(fm.ETA[label].eta.^2)/nSums

                        fm.ETA[label].post_var=fm.ETA[label].post_var*k+fm.ETA[label].var/nSums
                        fm.ETA[label].post_var2=fm.ETA[label].post_var2*k+(fm.ETA[label].var^2)/nSums

            end
        end
	return fm
end

#Example: BRR(rand(4,3))

###################################################################################################################
#End BRR
###################################################################################################################

###################################################################################################################
#Begin FixEff
###################################################################################################################

function  FixEff(X::Array{Float64})
   n,p=size(X)
   return RandRegBRR("FIXED",n,p,X,zeros(1,p),zeros(p),zeros(n),-Inf,-Inf,-Inf,-Inf,0.0,false,0.0,0.0,0.0,zeros(p),zeros(p),zeros(p),zeros(n),zeros(n),zeros(n),"","",0,0)
end

#Example: FixEff(rand(4,3))

###################################################################################################################
#EndFixEff
###################################################################################################################

###################################################################################################################
#Begin RKHS
###################################################################################################################

function RKHS(;K="null",EVD="null",R2=-Inf,df0= -Inf,S0=-Inf,minEigValue=1e-7) 
 
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

  return RandRegBRR("RKHS",n,p,V,zeros(1,p),zeros(p),zeros(n),R2,df0,S0,0.0,0.0,true,0.0,0.0,0.0,zeros(p),zeros(p),zeros(p),zeros(n),zeros(n),zeros(n),"","",0,0)

end


function bglr(;y="null",ETA=Dict(),nIter=1500,R2=.5,burnIn=500,thin=5,saveAt=string(pwd(),"/"),verbose=true,df0=1,S0=-Inf,naCode= -999, groups="null")
   #y=rand(10);ETA=Dict();nIter=-1;R2=.5;burnIn=500;thin=5;path="";verbose=true;df0=0;S0=0;saveAt=pwd()*"/"

   welcome()

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
   nGroups=1
   hasGroups=false

   Vy=var(yStar)
   if (S0<0)
      S0=(1-R2)*(df0+2)*Vy
   end 

   if(groups!="null")
	hasGroups=true
	freqTable=table(groups)
	nGroups=size(freqTable[1])[1]

	if(nGroups<2) 
		error("The groups vector has only 1 group")
	end

	frequencies=freqTable[2]
   end
  
   VError=rep(Vy,each=nGroups)
   
   ### Initializing the linear predictor
   ETA=merge(ETA,Dict("INT"=>INT(yStar)))
  
   #term[2] has information related to a type
   #term[1] has information related to a key in the dictionary
     
   
   for term in ETA
        if(typeof(term[2])==INT ||
	   typeof(term[2])==RandRegBRR)

		#Ridge Regression, RKHS, FIXED effects
		if(typeof(term[2])==RandRegBRR)

			if(nGroups>1)
			  BRR_post_init(term[2], Vy, length(ETA)-1, R2, hasGroups,groups)
			else
			  BRR_post_init(term[2], Vy, length(ETA)-1, R2, hasGroups,[0])
			end
			
			if(term[2].name=="FIXED")
				term[2].var=1e10
			end
		end
			
              else 
        	error("The elements of ETA must of type RandRegBRR or INT")
	      end
   end #end for    

   ## Opening connections
   for term in ETA
   	  term[2].name=term[1]

   	  if(typeof(term[2])==INT)
   	  	term[2].fname=string(saveAt,term[2].name,"_mu.dat")
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
   	    ETA,nIter,burnIn,thin,R2,verbose,
            saveAt,n,VError.*(1-R2),df0,S0,df0+n,zeros(nGroups),zeros(nGroups),zeros(nGroups),false,false,open(saveAt*"varE.dat","w+"))
              
   if (nIter>0)
   	for i in 1:nIter ## Sampler
		
		tic(); 	#Timer
       		
		## determining whether samples or post. means need to be updated
   		
		fm.saveSamples=(i%thin)==0
   		
		fm.updateMeans=fm.saveSamples&&(i>burnIn)
  		
		if fm.updateMeans
  		  	nSums+=1
  		  	k=(nSums-1)/nSums
  		end
  		
  		## Sampling effects and other parameters of the LP
  		for term in ETA    ## Loop over terms in the linear predictor
     			
			if(typeof(term[2])==INT)
				if(nGroups>1)
					fm=updateInt(fm,term[1],fm.updateMeans,fm.saveSamples,nSums,k,hasGroups,groups)
				else
					fm=updateInt(fm,term[1],fm.updateMeans,fm.saveSamples,nSums,k,hasGroups,[0])
				end
       			end     		  

			if(typeof(term[2])==RandRegBRR)

				if(nGroups>1)
					
					sample_beta_groups(fm.ETA[term[1]].n, 
							   fm.ETA[term[1]].p,
							   fm.ETA[term[1]].X,
							   fm.ETA[term[1]].x2,
							   fm.ETA[term[1]].effects,
						           fm.error,
							   fm.ETA[term[1]].var,
							   fm.varE,
							   groups,
							   nGroups)

					SS=sumsq(fm.ETA[term[1]].effects)+fm.ETA[term[1]].S0
                			fm.ETA[term[1]].var=SS/rand(Chisq(fm.ETA[term[1]].df),1)[]

				else

					fm=updateRandRegBRR(fm,term[1],fm.updateMeans,fm.saveSamples,nSums,k)

				end
			end	
  		end

  		## Updating error variance

		if(nGroups>1)
			for g in 1:nGroups
				SS=sumsq(fm.error[groups.==g])
				df=fm.df0+frequencies[g]
				fm.varE[g]=SS/rand(Chisq(df),1)[]
			end
		else
			SS=sumsq(fm.error)+fm.S0
                        fm.varE[1]= SS/rand(Chisq(fm.df),1)[]
		end

		if(fm.saveSamples)
			writeln(fm.conVarE,fm.varE, "\t") 		
		end
  			
  		## Updating error, yHat & yStar
  		fm.yHat=fm.yStar-fm.error
  		  		
  		if(hasNA)
	  		fm.error[fm.isNA]=rand(Normal(0,sqrt(fm.varE)),fm.nNA)
  			fm.yStar[fm.isNA]=fm.yHat[fm.isNA]+fm.error[fm.isNA]
		end

		
  		if(fm.updateMeans)
			
  			fm.post_varE=fm.post_varE*k+fm.varE/nSums
  			fm.post_varE2=fm.post_varE2*k+(fm.varE.^2)/nSums

  			fm.post_yHat=fm.post_yHat*k+fm.yHat/nSums
			fm.post_yHat2=fm.post_yHat2*k+(fm.yHat.^2)/nSums

  		end
		
		elapsed=toq();
		
  		if verbose 
  			println("Iter: ",i," VarE=",round(fm.varE,4),"  Time/Iter=",round(elapsed,4)) 
  		end
  		
	 end # end of sampler
    end # end of nIter>0
	
    ## Closing connections
    for term in ETA
   	   close(term[2].con)
    end
   
    ## Compute posterior SDs
   	fm.post_SD_yHat=sqrt(fm.post_yHat2-fm.post_yHat.^2)
	
	for term in ETA 
	  if(typeof(term[2])==INT)
	     term[2].post_SD_mu=sqrt(term[2].post_mu2-term[2].post_mu^2)
	  end
	  
	  if(typeof(term[2])==RandRegBRR)
             #Add your magic code here
	  end
  
	end #end of for
	
   	return fm
end


end #module end

