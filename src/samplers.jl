#Functions to implement samplers

# This routine was adapted from rinvGauss function from S-Plus
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
