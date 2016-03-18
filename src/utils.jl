function scale(Array{Number,2};center=true,scale=true)
    n,p=size(X)
    for j in 1:p
        xj=X[:,j]
        mu=mean(xj)
        SD=std(xj)
    	X[:,j]=(xj-mu)/SD
    end
	X
end
