## BGLR for the Julia Language

This Julia package implements **Bayesian shrinkage and variable selection methods for high-dimensional regressions**.

The design is inspired on the BGLR R-package (BGLR-R). Over time we aim to implement a similar set of methods than the ones implelented in the R version. The R version is highly optimized with use of compiled C code; this gives BGLR-R a computational speed much higher than the one that can be obtained using R code only. 

By developing BGLR for Julia (BGLR-Julia) wee seek to: (i) reach the comunity of Julia users, (ii) achieve a similar or better performance than the one achieved with BGLR-R (this is challenging because BGLR-R is highly optimized and makes intensive use of C and BLAS routines), (iii) enable users to use BGLR with memmory-mapped arrays as well as RAM arrays, (iv) capitalize on some of multi-core computing capabilities offered by Julia.

Funding of BGLR-R and BGLR-Julia was provided by NIH (R01 GM101219).

Authors:  Gustavo de los Campos (gustavoc@msu.edu) and Paulino Perez-Rodriguez (perpdgo@gmail.com)
    - [BGLR-R in CRAN](https://cran.r-project.org/web/packages/BGLR/index.html)          
    - [BGLR-R in GitHub](https://github.com/gdlc/BGLR-R)
    - [BGLR-R Publication](http://www.genetics.org/content/early/2014/07/06/genetics.114.164442)


#### Installing BGLR-Julia

```Julia
Pkg.add('BGLR..')
```
#### Examples
  * [Genomic BLUP using BLGR-Julia]()
  * [Parametric Shrinkage and Variable Selection]()
  * [Integrating fixed effects, regression on markers and pedigrees]()
  * [Reproducing Kernel Hilbert Spaces Regression using BLGR-J]()
  * [Prediction in testing data sets]()
  * [Modeling heterogeneous error variances]()
  * [Modeling genetic by environment interactions using BGLR-J]()



