
## To-do list
# Implement missing values **
# Implement methods predict()  effects() variances()
# summary()
##

streamOrASCIIString=Union(ASCIIString,IOStream)

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

##
type BGLR
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


function BGLR(;y="null",ETA=Dict(),nIter=1500,R2=.5,burnIn=500,thin=5,saveAt=string(pwd(),"/"),verbose=true,df0=1,S0=-Inf,naCode= -999)
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
   
   if(length(ETA)>0)
     k=1
     for term in ETA
        if( typeof(term[2])==GP ||typeof(term[2])==FixEff )
        	
        	if (typeof(term[2])==GP)
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
         	end       
        	k+=k
        else 
        	error("The elements of ETA must of type GP.")
        end
     end     
   end 

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
   

   fm=BGLR(	y,yStar,yHat,resid,zeros(n),zeros(n),zeros(n),
   			naCode,hasNA,nNA,isNA,
   			LP,nIter,burnIn,thin,R2,verbose,
          	saveAt,n,Vy*(1-R2),df0,S0,df0+n,0,0,0,false,false,open(saveAt*"varE.dat","w+"))
              
   if nIter>0
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
    		## FixEff
  		end

  		## Updating error variance
  		SS=sumsq(fm.error)+fm.S0
  		fm.varE= SS/rand(Chisq(fm.df),1)[]
  		
  		
  		## Updating error, yHat & yStar
  		fm.yHat=fm.yStar-fm.error
  		  		
  		if hasNA
	  		fm.error[fm.isNA]=rand(Normal(0,sqrt(fm.varE)),fm.nNA)
  			fm.yStar[fm.isNA]=fm.yHat[fm.isNA]+fm.error[fm.isNA]
		end
  		
  		if fm.updateMeans
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
	  if typeof(ETA[2])==INT
	     ETA[2].post_SD_mu=sqrt(ETA[2].post_mu2-ETA[2].post_mu^2)
	  end


	  if typeof(ETA[2])==GP
	      ETA[2].post_SD_effects=sqrt(ETA[2].post_effects2-ETA[2].post_effects.^2)
	      ETA[2].post_SD_eta=sqrt(ETA[2].post_eta2-ETA[2].post_eta.^2)
	      ETA[2].post_SD_var=sqrt(ETA[2].post_var2-ETA[2].post_var^2)
	  
	  
	  end
	  
	  if typeof(ETA[1])==FixEff
	  
	  end
	  
	  
	end
	
   	return fm
end

## Test
 #y=rand(10)
 #TMP=["mrk"=>GP(K=eye(10))]
 #fm=BGLR(y=y,ETA=TMP,R2=.9)
 
# test


## update methods

function updateInt(fm::BGLR,label::ASCIIString,updateMeans::Bool,saveSamples::Bool,nSums::Int,k::Float64)
    fm.error+=fm.ETA[label].mu
	  fm.ETA[label].mu=rand(Normal(mean(fm.error),sqrt(fm.varE/fm.n)))
    fm.error-=fm.ETA[label].mu
   	if saveSamples 
   		writeln(fm.ETA[label].con,fm.ETA[label].mu,"") 
   		
   		if updateMeans
   			fm.ETA[label].post_mu=fm.ETA[label].post_mu*k+fm.ETA[label].mu/nSums
   			fm.ETA[label].post_mu2=fm.ETA[label].post_mu2*k+(fm.ETA[label].mu^2)/nSums
   		end
	end
	return fm
end 
 
function updateGP(fm::BGLR,label::ASCIIString,updateMeans::Bool,saveSamples::Bool,nSums::Int,k::Float64)
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
    
    if saveSamples
	    writeln(fm.ETA[label].con,fm.ETA[label].var,"") 
	    
	    if updateMeans
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

# compu="home"
 #include("/Users/gustavodeloscampos/Dropbox/julia/RKHSV4.jl")
#@elapsed fm=RKHS(y=y+100,ETA=["mrk"=>GP(K=G)],nIter=1000,burnIn=100)




