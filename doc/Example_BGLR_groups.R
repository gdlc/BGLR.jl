##########################################################################################
#wheat dataset, BRR
##########################################################################################

library(BGLR)
library(cluster)

data(wheat)
y=wheat.Y[,1]
X=wheat.X
X=scale(X,center=TRUE,scale=TRUE)/sqrt(ncol(X))
G=tcrossprod(X)
out=eigen(G)
plot(out$vectors[,1],out$vectors[,2])
V=out$vectors[,1:2]

groups=pam(V,k=2)$clustering


groups[groups==1]="z"
groups[groups==2]="a"

ETA=list(list(X=X,model="BRR"))

setwd("/tmp")

#fm=BGLR(y=y,ETA=ETA)

#unlink("*.dat")

#Groups
groups=factor(groups)

#y[groups==2]=y[groups==2]+rnorm(n=sum(groups==2),mean=0,sd=sqrt(0.5))
y[as.character(groups)=="a"]=y[as.character(groups)=="a"]+rnorm(n=sum(as.character(groups)=="a"),mean=0,sd=sqrt(0.5))

#fm=BGLR(y=y,ETA=ETA,groups=groups,nIter=20,burnIn=10)

y[c(1,200,400,500)]=NA

system.time(BGLR(y=y,ETA=ETA,groups=groups,nIter=10000,burnIn=5000))

plot(fm)
unlink("*.dat")


##########################################################################################
#Mouse dataset BRR
##########################################################################################

  rm(list=ls())
  library(cluster) 
  library(BGLR);  set.seed(12345); data(mice);  
  n<-nrow(mice.X); p<-ncol(mice.X);  
  X=scale(mice.X,scale=TRUE,center=TRUE)
  
  G=tcrossprod(X)
  out=eigen(G)
  plot(out$vectors[,1],out$vectors[,2])
  V=out$vectors[,1:2]

  groups=pam(V,k=2)$clustering

 
## Toy simulation example 
  nQTL<-1000; p<-ncol(X); n<-nrow(X); h2<-0.5 
  whichQTL<-seq(from=floor(p/nQTL/2),by=floor(p/nQTL),length=nQTL) 
  b0<-rep(0,p) 
  b0[whichQTL]<-rnorm(n=nQTL,sd=sqrt(h2/nQTL)) 
  signal<-as.vector(X%*%b0) 
  error<-rnorm(n,sd=sqrt(1-h2)) 
  y<-signal+error 
  
  groups[groups==1]="z"
  groups[groups==2]="a"

  ETA=list(list(X=X,model="BRR"))

  setwd("/tmp")

  #Groups
  groups=factor(groups)

  y[as.character(groups)=="a"]=y[as.character(groups)=="a"]+rnorm(n=sum(as.character(groups)=="a"),mean=0,sd=sqrt(0.5))
  
  Rprof(tmp2 <- tempfile())
  BGLR(y=y,ETA=ETA,groups=groups,nIter=1000,burnIn=500)
  Rprof()
  
  Rprof(tmp1 <- tempfile())
  BGLR(y=y,ETA=ETA,nIter=1000,burnIn=500)
  Rprof()
  
  summaryRprof(tmp1)
  summaryRprof(tmp2)
  
  
##############################
#Mouse dataset Fixed effects
##############################

  rm(list=ls())
  library(cluster) 
  library(BGLR);  set.seed(12345); data(mice);  
  n<-nrow(mice.X); p<-ncol(mice.X);  
  X=scale(mice.X,scale=TRUE,center=TRUE)
  
  G=tcrossprod(X)
  out=eigen(G)
  plot(out$vectors[,1],out$vectors[,2])
  V=out$vectors[,1:2]

  groups=pam(V,k=2)$clustering
  
  groups[groups==1]="z"
  groups[groups==2]="a"
  
  groups=factor(groups)

  ETA=list(FIXED=list(~factor(GENDER)+factor(Litter),                    
                   data=mice.pheno,model="FIXED"))
                   
  setwd("/tmp/")

  fm=BGLR(y=mice.pheno$Obesity.BMI,ETA=ETA,groups=groups)
  
  
##########################################################################################
#BayesB 
##########################################################################################

rm(list=ls())

library(BGLR)
library(cluster)

data(wheat)

n<-599   # should be <= 599
p<-1279   # should be <= than 1279=ncol(X)
nQTL<-30 # should be <= than p
X<-wheat.X[1:n,1:p]

## Centering and standarization
for(i in 1:p)
{
        X[,i]<-(X[,i]-mean(X[,i]))/sd(X[,i])
}

# Simulation
b0<-rep(0,p)
whichQTL<-sample(1:p,size=nQTL,replace=FALSE)
b0[whichQTL]<-rnorm(length(whichQTL),
                    sd=sqrt(1/length(whichQTL)))
signal<-as.vector(X%*%b0)
error<-rnorm(n=n,sd=sqrt(0.5))
y<-signal +error


G=tcrossprod(X)
out=eigen(G)
plot(out$vectors[,1],out$vectors[,2])
V=out$vectors[,1:2]

groups=pam(V,k=2)$clustering


groups[groups==1]="z"
groups[groups==2]="a"

ETA=list(list(X=X,model="BayesB",probIn=0.05))
ETA=

setwd("/tmp")

#Groups
groups=factor(groups)

#y[groups==2]=y[groups==2]+rnorm(n=sum(groups==2),mean=0,sd=sqrt(0.5))
y[as.character(groups)=="a"]=y[as.character(groups)=="a"]+rnorm(n=sum(as.character(groups)=="a"),mean=0,sd=sqrt(0.5))

nIter=5000;
burnIn=2500;
thin=10;
saveAt='';
S0=NULL;
weights=NULL;
R2=0.5;

fit_BB=BGLR(y=y,ETA=ETA,nIter=nIter,burnIn=burnIn,thin=thin,saveAt=saveAt,df0=5,S0=S0,weights=weights,R2=R2,groups=groups)

plot(fit_BB$yHat,y)


##########################################################################################
#BayesC 
##########################################################################################

rm(list=ls())

library(BGLR)
library(cluster)

data(wheat)

n<-599   # should be <= 599
p<-1279   # should be <= than 1279=ncol(X)
nQTL<-30 # should be <= than p
X<-wheat.X[1:n,1:p]

## Centering and standarization
for(i in 1:p)
{
        X[,i]<-(X[,i]-mean(X[,i]))/sd(X[,i])
}

# Simulation
b0<-rep(0,p)
whichQTL<-sample(1:p,size=nQTL,replace=FALSE)
b0[whichQTL]<-rnorm(length(whichQTL),
                    sd=sqrt(1/length(whichQTL)))
signal<-as.vector(X%*%b0)
error<-rnorm(n=n,sd=sqrt(0.5))
y<-signal +error


G=tcrossprod(X)
out=eigen(G)
plot(out$vectors[,1],out$vectors[,2])
V=out$vectors[,1:2]

groups=pam(V,k=2)$clustering


groups[groups==1]="z"
groups[groups==2]="a"

ETA=list(list(X=X,model="BayesC"))

setwd("/tmp")

#Groups
groups=factor(groups)

#y[groups==2]=y[groups==2]+rnorm(n=sum(groups==2),mean=0,sd=sqrt(0.5))
y[as.character(groups)=="a"]=y[as.character(groups)=="a"]+rnorm(n=sum(as.character(groups)=="a"),mean=0,sd=sqrt(0.5))

nIter=5000;
burnIn=2500;
thin=10;
saveAt='';
S0=NULL;
weights=NULL;
R2=0.5;

fit_BC=BGLR(y=y,ETA=ETA,nIter=nIter,burnIn=burnIn,thin=thin,saveAt=saveAt,df0=5,S0=S0,weights=weights,R2=R2,groups=groups)

plot(fit_BC$yHat,y)


