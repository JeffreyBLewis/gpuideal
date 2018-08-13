# gpuideal
## R package implementing MCMC estimation of the two-parameter IRT model on GPUs

Requires CUDA support to estimate [Clinton, Jackman, and River's (2004)](https://www.cs.princeton.edu/courses/archive/fall09/cos597A/papers/ClintonJackmanRivers2004.pdf) IDEAL model.  

This package only supports NVIDIA GPUs. To install the package, you must first install Nvidia's CUDA Toolkit available from

http://developer.nvidia.com/cuda-downloads

Install with 

```{r}
> devtools::install_github("jeffreyblewis/gpuideal")
```

Fit to simulated data:

```{r}
> library(gpuideal)
> test_ideal(nrc=200, nmem=100, samples=50000, thin=10)
```

Fit all rollcalls from 114th US Senate:

```{r}
> library(gpuideal)
> library(coda)
> library(pscl)
> rcdat <- readKH("https://voteview.com/static/data/out/votes/S114_votes.ord")

> res <- gpuideal(rcdat, samples=5000, burnin=5000, thin=5,
                  abprior=matrix(c(25,0,0,25),2,2),
                  x = ifelse(rcdat$legis.data$party=="D",-0.5, 0.5))
> scale_dir <- as.integer(rcdat$legis.data$party=="R") 
> rr <- rescaleIdeal(res, scale_dir)  
> summary(rr)[[1]][1:length(scale_dir),] 
```

