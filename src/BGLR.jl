#Define the module BGLR
#Last update Agust/4/2016

module BGLR

export
	bglr,
	RKHS,
	BRR,
        BL,
	BayesA,
	BayesB,
	FixEff,
	read_bed,
	model_matrix,
	rep

import
	Distributions.Normal,
	Distributions.Chisq,
        Distributions.Gamma,
	Distributions.Bernoulli,
	Distributions.Uniform,
	Distributions.Beta,
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
  print("#    (   )     (   )   Agust, 2016                                   #\n");
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
		 #sumMeanXSq 
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
	#z=rand(Normal(0,sqrt(fm.varE[1])),p)
    	#lambda=fm.varE[1]/fm.ETA[label].var
	#x2=vec(fm.ETA[label].x2[1,:])
    	#for j in 1:p         
    	#	b=fm.ETA[label].effects[j] 
	#	SSX=x2[j]
	#	xj=unsafe_view(fm.ETA[label].X, :, j)
	#	rhs=innersimd(xj,fm.error,n)+SSX*b
	#	CInv=1/(SSX + lambda)
	#	fm.ETA[label].effects[j]=rhs*CInv+sqrt(CInv)*z[j]
	#	tmp=b-fm.ETA[label].effects[j]
	#	my_axpy!(tmp,xj,fm.error,n)		
	#end	

	#Actually this implementation is faster that the one shown in previous lines!

	sample_beta(n, p, fm.ETA[label].X, vec(fm.ETA[label].x2[1,:]),
                    fm.ETA[label].effects,fm.error,rep(fm.ETA[label].var,each=p),
		    fm.varE[1])

    
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

##Linear Term: BL
#The well known Bayesian LASSO (Park and Casella, 2008) and 
#de los Campos et al (2009)

type RandRegBL  #Bayesian LASSO
  name::ASCIIString
  n::Int64 #Number of individuals
  p::Int64 #Number of covariates
  X::Array{Float64,2} #Incidence matrix
  x2::Array{Float64,1} #Sum of the squares of the columns of X
  effects::Array{Float64,1} #b
  eta::Array{Float64,1} #X*b
  R2::Float64
  lambda::Float64
  lambda2::Float64
  lambda_type::ASCIIString #Possible values are "gamma", "beta", "FIXED"
  shape::Float64
  rate::Float64
  tau2::Array{Float64,1} #tau^2
  post_effects::Array{Float64,1}
  post_effects2::Array{Float64,1}
  post_SD_effects::Array{Float64,1}
  post_eta::Array{Float64,1} #posterior mean of linear term
  post_eta2::Array{Float64,1} #posterior mean of the linear term squared
  post_SD_eta::Array{Float64,1} #posterior SD of the linear term
  post_lambda::Float64
  post_tau2::Array{Float64,1} 
  fname::ASCIIString
  con::streamOrASCIIString #a connection where samples will be saved
  nSums::Int64
  k::Float64	
end

#Function to setup RandReg
#when the prior for the coefficients is Double Exponential or Laplace

function BL(X::Array{Float64,2};R2=-Inf, lambda=-Inf,lambda_type="gamma", shape=-Inf, rate=-Inf)
	n,p=size(X)  #sample size and number of predictors
	return RandRegBL("BL",n,p,X,zeros(p),zeros(p),zeros(n),R2,lambda,lambda^2,lambda_type,shape,rate,zeros(p),zeros(p),zeros(p),zeros(p),zeros(n),zeros(n),zeros(n),0.0,zeros(p),"","",0,0)
end

#Example
#BL(X)

function BL_post_init(LT::RandRegBL, Vy::Float64, nLT::Int64, R2::Float64)

	#The sum of squares of columns of X
	for j in 1:LT.p
	    LT.x2[j]=sum(LT.X[:,j].^2)
	end

	#Prior
	
	if(LT.R2<0)
	  	LT.R2=R2/nLT 
	end

        #sumMeanXSq		
	sumMeanXSq=0.0
	for j in 1:LT.p
		sumMeanXSq+=(mean(LT.X[:,j]))^2
	end

	MSx=sum(LT.x2)/LT.n-sumMeanXSq

	warn("By default, the prior density of lambda^2 in the LP was set to ", LT.lambda_type,"\n")

	if(LT.lambda<0)
	   LT.lambda2=2*(1-R2)/(LT.R2)*MSx
	   LT.lambda=sqrt(LT.lambda2)
	   warn("Initial value of lambda in LP was set to default value ", LT.lambda,"\n")
	else
	  LT.lambda2=LT.lambda^2	
	end	

	if(LT.lambda_type=="gamma")
		if(LT.shape<0)	
			LT.shape=1.1
			warn("shape parameter in LP was missing and was set to ",LT.shape,"\n")
		end

		if(LT.rate<0)
			LT.rate=(LT.shape-1)/LT.lambda2
			warn("rate parameter in LP was missing and was set to ",LT.rate,"\n")
		end
	end

	if(LT.lambda_type=="beta")
		#Add your magic code here
	end

	#Initial values for tau^2
        LT.tau2=rep((Vy*R2/nLT)/MSx,each=LT.p)
end

function updateRandRegBL(fm::BGLRt,label::ASCIIString, updateMeans::Bool, saveSamples::Bool, nSums::Int, k::Float64)
	p=fm.ETA[label].p
	n=fm.ETA[label].n

	varBj=fm.ETA[label].tau2*fm.varE[1]

	sample_beta(n, p, fm.ETA[label].X, fm.ETA[label].x2,
                    fm.ETA[label].effects,fm.error, varBj,
		    fm.varE[1])

	if(saveSamples)

		writeln(fm.ETA[label].con,fm.ETA[label].lambda,"")
                
		if(updateMeans)
			fm.ETA[label].post_effects=fm.ETA[label].post_effects*k+fm.ETA[label].effects/nSums
                        fm.ETA[label].post_effects2=fm.ETA[label].post_effects2*k+(fm.ETA[label].effects.^2)/nSums

			fm.ETA[label].post_lambda=fm.ETA[label].post_lambda*k+fm.ETA[label].lambda/nSums
                        fm.ETA[label].post_tau2=fm.ETA[label].post_tau2*k+fm.ETA[label].tau2/nSums
		end
	end
	
	return fm
	
end

#Bayes A, Mewissen et al. (2001).
#Prediction of Total Genetic Value Using Genome-Wide Dense Marker Maps
#Genetics 157: 1819-1829, Modified so that the Scale parameter 
#is estimated from data (a gamma prior is assigned)

## Linear Term: BayesA
type RandRegBayesA # BayesA
  name::ASCIIString
  n::Int64 # number or individuals
  p::Int64 # number of vectors
  X::Array{Float64,2} # incidence matrix
  x2::Array{Float64,1} # sum of squares of columns of X
  effects::Array{Float64,1} # b
  eta::Array{Float64,1} # X*b
  R2::Float64
  df0::Float64 #prior degree of freedom
  S0::Float64  #prior scale
  df::Float64  #degrees of freedom of the conditional distribution
  shape0::Float64 #shape parameter for the gamma prior assigned to Scale
  rate0::Float64  #rate parameter for the gamma prior assigned to Scale
  S::Float64 #Scale parameter
  var::Array{Float64,1} # variance of effects
  post_var::Array{Float64,1}  # posterior mean
  post_var2::Array{Float64,1} # posterior mean of the squared of the variance
  post_SD_var::Array{Float64,1} # posterior standard deviation
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
#when the prior for the coefficients is distributed according to BayesA model

function BayesA(X::Array{Float64,2};R2=-Inf,df0=-Inf,S0=-Inf,shape0=-Inf,rate0=-Inf)
        n,p=size(X)  #sample size and number of predictors
        return RandRegBayesA("BayesA",n,p,X,zeros(p),zeros(p),zeros(n),R2,df0,S0,0.0,shape0,rate0,0.0,zeros(p),zeros(p),zeros(p),zeros(p),zeros(p),zeros(p),zeros(p),zeros(n),zeros(n),zeros(n),"","",0,0)
end

#Example
#BayesA(X)

function BayesA_post_init(LT::RandRegBayesA, Vy::Float64, nLT::Int64, R2::Float64)

        #The sum of squares of columns of X
        for j in 1:LT.p
            LT.x2[j]=sum(LT.X[:,j].^2)
        end

	#sumMeanXSq
        sumMeanXSq=0.0
        for j in 1:LT.p
                sumMeanXSq+=(mean(LT.X[:,j]))^2
        end

        MSx=sum(LT.x2)/LT.n-sumMeanXSq

        #Default degrees of freedom for the prior assigned to the variance of the markers
        if(LT.df0<0)
		LT.df0=5
		warn("DF in LP was missing and was set to ",LT.df0,"\n")
		LT.df=LT.df0+1
        end

        if(LT.R2<0)
                LT.R2=R2/nLT
		warn("R2 in LP was missing and was set to ", LT.R2,"\n")
        end

	#Default scale parameter for the prior assigned to the variance of markers
	if(LT.S0<0)
		LT.S0=Vy*LT.R2/MSx*(LT.df0+2)
		warn("Scale in LP was missing and was set to ", LT.S0,"\n")
	end

	#Improvement: Treat Scale as random, assign a gamma density
	if(LT.shape0<0)
		LT.shape0=1.1
	end
	
	if(LT.rate0<0)
		LT.rate0=(LT.shape0-1)/LT.S0
	end

        #Initial value for S
	LT.S=LT.S0

	#Initial value for variances of regression coefficients
	LT.var=rep(LT.S0/(LT.df0+2),each=LT.p)
end

function updateRandRegBayesA(fm::BGLRt,label::ASCIIString, updateMeans::Bool, saveSamples::Bool, nSums::Int, k::Float64)
        p=fm.ETA[label].p
        n=fm.ETA[label].n

        varBj=fm.ETA[label].var

        sample_beta(n, p, fm.ETA[label].X, fm.ETA[label].x2,
                    fm.ETA[label].effects,fm.error, varBj,
                    fm.varE[1])

        if(saveSamples)

                writeln(fm.ETA[label].con,fm.ETA[label].S,"")

                if(updateMeans)
                	fm.ETA[label].post_effects=fm.ETA[label].post_effects*k+fm.ETA[label].effects/nSums
                        fm.ETA[label].post_effects2=fm.ETA[label].post_effects2*k+(fm.ETA[label].effects.^2)/nSums

			fm.ETA[label].post_var=fm.ETA[label].post_var*k+fm.ETA[label].var/nSums
                        fm.ETA[label].post_var2=fm.ETA[label].post_var2*k+(fm.ETA[label].var).^2/nSums
                end
        end

        return fm

end

#Pseudo BayesB with random scale and random proportion of markers "in" the model
#See Variable selection for regression models, 
#Lynn Kuo and Bani Mallic, 1998. 

##Linear Term: BayesB
type RandRegBayesB #BayesB
  name::ASCIIString
  n::Int64  #Number of individuals
  p::Int64  #Number of markers 
  X::Array{Float64,2} #incidence matrix
  x2::Array{Float64,1} #Sum of squares of columns of X
  effects::Array{Float64,1} #b
  eta::Array{Float64,1} #X*b
  probIn::Float64 #Prob of a marker being in the model
  counts::Int64 #Prior counts 
  countsIn::Float64
  countsOut::Float64
  d::Array{Int64,1}
  R2::Float64
  df0::Float64 #prior degree of freedom
  S0::Float64  #prior scale
  df::Float64  #degrees of freedom of the conditional distribution
  shape0::Float64 #shape parameter for the gamma prior assigned to Scale
  rate0::Float64  #rate paramter for the gamma prior assigned to Scale
  S::Float64 #Scale parameter
  var::Array{Float64,1} #Variance of effects
  post_var::Array{Float64,1} #posterior mean
  post_var2::Array{Float64,1} #posterior mean of the squared of the variance
  post_SD_var::Array{Float64,1} #posterior standard deviation
  post_effects::Array{Float64,1} 
  post_effects2::Array{Float64,1}
  post_SD_effects::Array{Float64,1}
  post_eta::Array{Float64,1} #posterior mean of linear term
  post_eta2::Array{Float64,1} #posterior mean of the linear term squared
  post_SD_eta::Array{Float64,1} #posterior SD of the linear term
  post_probIn::Float64
  post_probIn2::Float64 
  fname::ASCIIString
  con::streamOrASCIIString #connection where the samples will be saved
  nSums::Int64
  k::Float64
end


#Function to setup RandReg
#when the prior of the coefficients is a mixture as defined in BayesB

function BayesB(X::Array{Float64,2}; R2=-Inf, df0=-Inf, S0=-Inf, shape0=-Inf, rate0=-Inf,probIn=0.5,counts=10)
	n,p=size(X) #Sample size and number of predictors
        return RandRegBayesB("BayesB",n,p,X,zeros(p),zeros(p),zeros(n),probIn,counts,0.0,0.0,zeros(Int64,p),R2,df0,S0,0.0,shape0,rate0,0.0,zeros(p),zeros(p),zeros(p),zeros(p),zeros(p),zeros(p),zeros(p),zeros(n),zeros(n),zeros(n),0.0,0.0,"","",0,0)
end

#Example
#BayesB(X)

function BayesB_post_init(LT::RandRegBayesB, Vy::Float64, nLT::Int64, R2::Float64)

        #The sum of squares of columns of X
        for j in 1:LT.p
            LT.x2[j]=sum(LT.X[:,j].^2)
        end

        #sumMeanXSq
        sumMeanXSq=0.0
        for j in 1:LT.p
                sumMeanXSq+=(mean(LT.X[:,j]))^2
        end

        MSx=sum(LT.x2)/LT.n-sumMeanXSq

        if(LT.R2<0)
                LT.R2=R2/nLT
                warn("R2 in LP was missing and was set to ", LT.R2,"\n")
        end

	#Default value for the degrees of freedom associated with the distribution assigned to the variance
	#of marker effects
	if(LT.df0<0)
		LT.df0=5
		warn("DF in LP was missing and was set to ", LT.df0,"\n")
		LT.df=LT.df0+1
	end

	#Default value for a marker being "in" the model
        if(LT.probIn<0)
		LT.probIn=0.5
		warn("ProbIn in LT was missing and was set to ", LT.probIn,"\n")
	end

	#Default value for prior counts
	if(LT.counts<0)
		LT.counts=10
		warn("Counts in LP was missing and was set to ", LT.counts,"\n")
	end
	
	LT.countsIn=LT.counts*LT.probIn
	LT.countsOut=LT.counts-LT.countsIn
	
	#Default value for the scale parameter associated with the distribution assigned to the variance of 
	#marker effects
	if(LT.S0<0)
		LT.S0=Vy*LT.R2/MSx*(LT.df0+2)/LT.probIn
		warn("Scale parameter in LP was missing and was set to ",LT.S0,"\n")
	end

	if(LT.shape0<0)
		LT.shape0=1.1
	end

	if(LT.rate0<0)
		LT.rate0=(LT.shape0-1)/LT.S0
	end

	#Initial value for S
	LT.S=LT.S0

	#Initial value for variances of regression coefficients
        LT.var=rep(LT.S0/(LT.df0+2),each=LT.p)

	#Initial value for d
        d=rand(Bernoulli(LT.probIn),LT.p)
	
end

function updateRandRegBayesB(fm::BGLRt,label::ASCIIString, updateMeans::Bool, saveSamples::Bool, nSums::Int, k::Float64)
        p=fm.ETA[label].p
        n=fm.ETA[label].n

        varBj=fm.ETA[label].var

        sample_beta_BB_BCp(n, p, fm.ETA[label].X, fm.ETA[label].x2, fm.ETA[label].effects,
                           fm.ETA[label].d,fm.error,varBj,fm.varE[1],
                           fm.ETA[label].probIn)

        if(saveSamples)

                writeln(fm.ETA[label].con,[fm.ETA[label].probIn, fm.ETA[label].S],"\t")  #probIn and Scale parameter

                if(updateMeans)

			fm.ETA[label].post_effects=fm.ETA[label].post_effects*k+fm.ETA[label].effects/nSums
                        fm.ETA[label].post_effects2=fm.ETA[label].post_effects2*k+(fm.ETA[label].effects.^2)/nSums

			fm.ETA[label].post_var=fm.ETA[label].post_var*k+fm.ETA[label].var/nSums
                        fm.ETA[label].post_var2=fm.ETA[label].post_var2*k+(fm.ETA[label].var).^2/nSums

			fm.ETA[label].post_probIn=fm.ETA[label].post_probIn*k+fm.ETA[label].probIn/nSums
                        fm.ETA[label].post_probIn2=fm.ETA[label].post_probIn2*k+(fm.ETA[label].probIn^2)/nSums
                end
        end

        return fm

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
	   typeof(term[2])==RandRegBRR ||
           typeof(term[2])==RandRegBL || 
           typeof(term[2])==RandRegBayesA || 
           typeof(term[2])==RandRegBayesB)

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
		
		if(typeof(term[2])==RandRegBL)
			if(nGroups>1)
			   	error("Groups not supported for BL")
			else
				BL_post_init(term[2], Vy, length(ETA)-1, R2)
			end
		end
	
		if(typeof(term[2])==RandRegBayesA)
			if(nGroups>1)
				error("Groups not supported for BayesA")
			else
				BayesA_post_init(term[2], Vy, length(ETA)-1,R2)
			end
		end

		if(typeof(term[2])==RandRegBayesB)
			if(nGroups>1)
				error("Groups not supported for BayesB")
			else
				BayesB_post_init(term[2], Vy, length(ETA)-1,R2)
			end
		end
			
              else 
        	error("The elements of ETA must of type RandRegBRR, RandRegBL, RandRegBayesA,RandRegBayesB or INT")
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
	
	  if(typeof(term[2])==RandRegBL)
	        #Add your magic code here
		term[2].fname=string(saveAt,term[2].name,"_lambda.dat")
	  end

	  if(typeof(term[2])==RandRegBayesA)
		#Add your magic code here
		term[2].fname=string(saveAt,term[2].name,"_ScaleBayesA.dat")
	  end

	  if(typeof(term[2])==RandRegBayesB)
		#Add your magic code here
		term[2].fname=string(saveAt,term[2].name,"_parBayesB.dat") 
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

		#deltaSS and deltadf for updating varE
		deltaSS=0
		deltadf=0
  		
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
			
			if(typeof(term[2])==RandRegBL)

				#Groups are not allowed for BL

                                #Update regression coefficients
				fm=updateRandRegBL(fm, term[1], fm.updateMeans, fm.saveSamples, nSums, k)

				#Update tau^2
                                #FIXME: This is slow, can we do faster?

				nu=(sqrt(fm.varE[1])*fm.ETA[term[1]].lambda)./abs(fm.ETA[term[1]].effects)
				for j=1:fm.ETA[term[1]].p
					tmp=1/rinvGauss(nu[j], fm.ETA[term[1]].lambda2)
                                        if(isfinite(tmp))
						fm.ETA[term[1]].tau2[j]=tmp
					else
						warn("tau^2[",j,"] not updated due to numerical problems\n")
					end
				end

				#Update lambda
			        if(fm.ETA[term[1]].lambda_type=="gamma")
				  	#warn("Not updating lambda")
			          	shape=fm.ETA[term[1]].p+fm.ETA[term[1]].shape
                                  	rate=sum(fm.ETA[term[1]].tau2)/2+fm.ETA[term[1]].rate
				  	fm.ETA[term[1]].lambda2=rand(Gamma(shape,1/rate))
				  	fm.ETA[term[1]].lambda=sqrt(fm.ETA[term[1]].lambda2)
				  	println("lambda=",round(fm.ETA[term[1]].lambda,2))
				end
				
				deltaSS=deltaSS+sum(((fm.ETA[term[1]].effects)./sqrt(fm.ETA[term[1]].tau2)).^2)
				deltadf=deltadf+fm.ETA[term[1]].p
                       
			end

			if(typeof(term[2])==RandRegBayesA)

				#Groups are not allowed for BayesA

				#Update regression coefficients
				fm=updateRandRegBayesA(fm, term[1], fm.updateMeans, fm.saveSamples, nSums, k)

				#Update variances
				for j=1:fm.ETA[term[1]].p
					SS=fm.ETA[term[1]].S+ETA[term[1]].effects[j]^2
					fm.ETA[term[1]].var[j]=SS/rand(Chisq(fm.ETA[term[1]].df),1)[]
				end

				#Update scale parameter
				#FIXME, this is constant, so we can move to the initialization of the 
				#linear term
				tmpShape=fm.ETA[term[1]].p*fm.ETA[term[1]].df0/2+fm.ETA[term[1]].shape0
				tmpRate=sum(1./fm.ETA[term[1]].var)/2+fm.ETA[term[1]].rate0
				fm.ETA[term[1]].S=rand(Gamma(tmpShape,1/tmpRate))
			end

			if(typeof(term[2])==RandRegBayesB)
				#Groups are not allowed for BayesB

				#Update regression coefficients

				fm=updateRandRegBayesB(fm, term[1], fm.updateMeans, fm.saveSamples, nSums, k)
				
				#Update variances
                                for j=1:fm.ETA[term[1]].p
                                        SS=fm.ETA[term[1]].S+ETA[term[1]].effects[j]^2
                                        fm.ETA[term[1]].var[j]=SS/rand(Chisq(fm.ETA[term[1]].df),1)[]
                                end

                                #Update scale parameter
                                #FIXME, this is constant, so we can move to the initialization of the
                                #linear term
                                tmpShape=fm.ETA[term[1]].p*fm.ETA[term[1]].df0/2+fm.ETA[term[1]].shape0
                                tmpRate=sum(1./fm.ETA[term[1]].var)/2+fm.ETA[term[1]].rate0
                                fm.ETA[term[1]].S=rand(Gamma(tmpShape,1/tmpRate))

				#Update probIn
				mrkIn=sum(fm.ETA[term[1]].d)
				shape1=mrkIn+fm.ETA[term[1]].countsIn+1
				shape2=fm.ETA[term[1]].p-mrkIn+fm.ETA[term[1]].countsOut+1
                                fm.ETA[term[1]].probIn=rand(Beta(shape1,shape2),1)[]
			end
	
  		end

  		## Updating error variance

		if(nGroups>1)
			for g in 1:nGroups
				SS=sumsq(fm.error[groups.==g])+fm.S0+deltaSS
				df=fm.df0+frequencies[g]+deltadf
				fm.varE[g]=SS/rand(Chisq(df),1)[]
			end
		else
			SS=sumsq(fm.error)+fm.S0+deltaSS
                        fm.varE[1]= SS/rand(Chisq(fm.df+deltadf),1)[]
		end

		if(fm.saveSamples)
			writeln(fm.conVarE,fm.varE, "\t") 		
		end
  			
  		## Updating error, yHat & yStar
  		fm.yHat=fm.yStar-fm.error
  		  		
  		if(hasNA)
			if(nGroups>1)
				println("Missing values for groups not supported yet!")
			else
	  			fm.error[fm.isNA]=rand(Normal(0,sqrt(fm.varE[1])),fm.nNA)
  				fm.yStar[fm.isNA]=fm.yHat[fm.isNA]+fm.error[fm.isNA]
			end
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

