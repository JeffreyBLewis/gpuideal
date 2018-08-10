#' Create a MC data set for testing
#'
#' Generates simulated roll matrix for testing.
#' 
#' @param nrc Number of roll calls
#' @param nmem Number of legislators
#'
#' @return rollcall dataset.
#' @export
#'
#' @examples
simrc <- function(nrc=100,nmem=100) {
      a <- rnorm(nrc)
      b <- rnorm(nrc)
      x <- rnorm(nmem)
      ystar <- cbind(1,x) %*% rbind(a,b) + matrix(rnorm(nrc*nmem),ncol=nrc)
      y <- 2*(ystar>0) - 1
      list(x=x,votes=y,ystar=ystar,a=a,b=b,codes=list('yea'=c(1),'nay'=c(-1)))
}


#' Run a test of the gpuideal point routine
#' 
#' Run a test of the CUDA gpu-based ideal point routine using simulated data.
#'
#' @param nrc Number of roll calls
#' @param nmem Number of legislators
#' @param ... Additional arguments to `gpuideal`
#'
#' @return MCMC object containing posterior draws
#' @export 
#' @imports coda
#'
#' @examples
test_ideal <- function(nrc=100,nmem=100,...) {
       dat <- simrc(nrc=nrc,nmem=nmem)
       for (i in 1:5) {
          print( lm(dat$votes[,i]~dat$x) )
       }
       res <- gpuideal(dat,...)
       rr <- rescaleIdeal(res,dat$x) 
       rrr <- summary(rr)
       print(coda::effectiveSize(rr)) 
       par.old <- par(ask=T)
       
       plot(dat$x, rrr[[1]][1:nmem,1]) 
       abline(0,1)
       
       plot(dat$a, rrr[[1]][(nmem+1):(nmem+nrc),1])
       abline(0,1)

       plot(dat$b, rrr[[1]][(nmem+nrc+1):(nmem+2*nrc),1])
       abline(0,1)

       par(par.old)
       rr
}

#res <- testideal(500, 100, thin=500, burnin=5000, samples=10000, threads=512)




