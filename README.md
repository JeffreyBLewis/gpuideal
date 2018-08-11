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
