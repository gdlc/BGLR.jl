#Functions to implement samplers

#=
 * This is a generic function to sample betas  when we have 
 * groups of individuals with different variances.
 
 * For example, in the Bayesian Ridge Regression, we wish to draw samples from the full 
 * conditional distribution of each of the elements in the vector b. The full conditional 
 * distribution is normal with mean and variance equal to the solution (inverse of the coefficient of the left hand side)
 * of the following equation (See suplementary materials in de los Campos et al., 2009 for details),
   
    (1/varE x_j' x_j + 1/varB) b_j = 1/varE x_j' e
 
    or equivalently, 
    
    mean= (1/varE x_j' e)/ (1/varE x_j' x_j + 1/varB)
    variance= 1/ (1/varE x_j' x_j + 1/varB)
    
    xj= the jth column of the incidence matrix
    
 *The notation in the routine is as follows:
 
 n: Number of rows in X
 p: Number of columns in X
 X: the matrix X 
 x2: matrix with x_jk' x_jk, j=1,...,p, k=1,...,number of Groups
 b: vector of regression coefficients
 e: vector with residuals, e=y-yHat, yHat= predicted values
 varBj: vector with variances
	For Ridge regression, varBj=varB, j=1,...,p, varB is the variance common to all betas.

 varE: vector with residual variances

=# 

function sample_beta_groups(n::Int64, p::Int64, X::Array{Float64,2}, x2::Array{Float64,2},
                     	    b::Array{Float64,1}, error::Array{Float64,1}, varBj::Array{Float64,1},
                            varE::Array{Float64,1}, groups::Array{Int64,1}, nGroups::Int64)

	rhs=zeros(nGroups)

	for j in 1:p
		bj=b[j]
		xj=X[:,j]

		c=0
		sum_rhs=0

		#rhs by group
		for i in 1:n
			rhs[groups[i]]+=xj[i]*e[i]
		end
		
		for k in 1:nGroups
			sum_rhs+=(rhs[k]+bj*x2[k,j])/varE[k]
			c+=x2[k,j]/varE[k]
		end
		
		c+=1.0/varBj[j]
		b[j]=sum_rhs/c + sqrt(1/c)*rand(Normal(0,1))

		bj=bj-b[j]

		axpy!(bj,X[:,j],error)

	end
end
