using BGLR
using Gadfly


#Test 1
#Gaussian process with simulated data
y=rand(10)
TMP=Dict("mrk"=>GP(K=eye(10)))

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

TMP=Dict("Gmatrix"=>GP(K=G))

fm=bglr(y=y,ETA=TMP);

plot(x=fm.y,
     y=fm.yHat,
     Guide.ylabel("yHat"),
     Guide.xlabel("y"),
     Guide.title("Observed vs predicted"))
     
#Test 3
#Bayesian Ridge Regression
TMP=Dict("XMatrix"=>BRR(X))

fm=bglr(y=y,ETA=TMP);

plot(x=fm.y,
     y=fm.yHat,
     Guide.ylabel("yHat"),
     Guide.xlabel("y"),
     Guide.title("Observed vs predicted"))
     
     