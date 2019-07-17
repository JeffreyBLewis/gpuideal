#' Recode votes and delete non-votes
#'
#' @param dat roll call object dataset
#' @param codes yea and nay values
#'
#' @return
#' @export
#' @importFrom pscl rollcall
#' @examples
rc_recode <- function(dat,codes) {
    newy <- rep(NA,length(dat$y))
    newy[dat$y %in% codes$yea] <- 1
    newy[dat$y %in% codes$nay] <- -1
    dat$y <- newy
    dat[!is.na(dat$y),]
}

#
#  Rescale posterior draws so that the ideal pt distributons have mean 0 and std dev 1 in 
#  each iteration.
#

#' Rescale posterior draws
#'
#' Rescale posterior draws so that the ideal pt distributons have mean 0 and std dev 1 in 
#' each iteration.
#' 
#' @param mcmcres MCMC object containing draws from IRT model
#' @param dir  A vector that should be positively correlated with the xs -- this sets the direction
#'
#' @return rescaled MCMC object
#' @export
#'
#' @examples
rescaleIdeal <- function(mcmcres,dir) {
     idx <- list()
     for (v in c("x","b","a")) idx[[v]] <- grep(v,names(mcmcres[1,]))
     meansd <- t(apply(mcmcres[,idx[['x']]],1,function(x) c(mean(x),sd(x),sign(cor(x,dir)))))
     mcmcres[,idx[['x']]] <- apply(mcmcres[,idx[['x']]],2, function(x) (x-meansd[,1])/meansd[,2])*meansd[,3]  
     mcmcres[,idx[['a']]] <- mcmcres[,idx[['a']]] + apply(mcmcres[,idx[['b']]],2, function(b) b*meansd[,1])     
     mcmcres[,idx[['b']]] <- apply(mcmcres[,idx[['b']]],2, function(b) b*meansd[,2]*meansd[,3])
     mcmcres
}

#' Run the gpu_ideal sampler
#'
#' Run the gpu ideal sampler
#' @param rcdata Roll call object
#' @param columnwise Logical flag for column-wise roll call data
#' @param samples Number of MCMC samples (after burnin)
#' @param burnin Number of burn in samples
#' @param thin Keep every `thin` sample
#' @param x Start values for ideal points
#' @param abprior Prior variance of bill parameters
#' @param xprior Prior variance of ideal points
#' @param blocks Number of GPU blocks
#' @param threads Number of GPU threads
#'
#' @return MCMC object containing posterior draws
#' @export
#' @importFrom coda mcmc
#' @importFrom pscl rollcall
#' @useDynLib idealcu
#'
#' @examples
gpu_ideal <- function(rcdata,columnwise=F,samples,burnin=0,thin=1,x=NULL,abprior=matrix(c(20,0,0,20),2,2),xprior=1,
	             blocks=0, threads=0) {
    # Format the the roll call data
    if (! columnwise){
        nmem <- dim(rcdata$votes)[1]
        nrc <- dim(rcdata$votes)[2] 
        # Set up column-wise (rollcall-wise data set)
        cw <- rc_recode( data.frame( y = as.vector(rcdata$votes),
                                     rcidx = rep(1:nrc,each=nmem),
                                     memidx = rep(1:nmem,nrc) ),
                         rcdata$codes )
    }
    if (columnwise){
        cw <- rcdata
        nmem <- length(unique(cw$memidx))
        nrc <- length(unique(cw$rcidx))
    }
    n  <- dim(cw)[1]
    rcstart <- tapply(1:n,cw$rcidx,min) 
    rclen <- as.vector(table(cw$rcidx))
    cw$rw2cw = 1:n  # Variable to map row in the rw datset
                    # (created below) back to the cw data

    # Set up row-wise (member-wise data set)
    rw <- cw[order(cw$memidx,cw$rcidx),] 
    memstart <- tapply(1:n,rw$memidx,min) 
    memlen <- as.vector(table(rw$memidx))
    cw$rw2cw <- NULL # This variable is not used in the cw dataset.
    
    # Include start values for ideal points if provided
    xx <- rep(0.0, samples*nmem/thin)
    if (is.null(x)) {
       xx[1:nmem] <- rnorm(nmem)
    }
    else {
       xx[1:nmem] <- x
    }

    cat("Starting GPU-based ideal estimator...\n\n")
    cat("Data summary...\n");
    cat(sprintf("Number of members:   %i\n", nmem))
    cat(sprintf("Number of rollcalls: %i\n", nrc))
    cat(sprintf("Number of choices:   %i\n", dim(rw)[1])) 
    
    # Run the MCMC chain
    timing <- system.time(
    res <- .C("gibbs_ideal",
			# parameters coming back
			x=as.single(xx),
			a=single(samples*nrc/thin),
			b=single(samples*nrc/thin),
			
			# Priors
			iabprior = as.single(solve(abprior)),
			ixprior = as.single(1/xprior),

			# Member-wise data
			rw.y = as.integer(rw$y),
			rw.ystar = as.single(rw$y),
			rw.rcidx = as.integer(rw$rcidx-1),
			rw.memidx = as.integer(rw$memidx-1),
			memstart = as.integer(memstart-1),
			memlen = as.integer(memlen),
			
			# Vote-wise data
			cw.y = as.integer(cw$y),
			cw.ystar = as.single(cw$y),
			cw.memidx = as.integer(cw$memidx-1),
			rcstart = as.integer(rcstart-1),
			rclen = as.integer(rclen),
			
			rw2cw = as.integer(rw$rw2cw-1),

			# Control parameters
			burnin=as.integer(burnin),
			samples=as.integer(samples),
			n=as.integer(n),
			n.rc = as.integer(nrc),
			n.mem = as.integer(nmem), 
			thin = as.integer(thin),
                        blocks = as.integer(blocks),
                        threads = as.integer(threads)

           ) 
    )
    cat("\nTime required:\n") 
    print(timing)

    # Rearrange things to pass back the results as a Coda MCMC object
    res <- cbind( matrix(res$x,ncol=nmem,byrow=T),
    	          matrix(res$a,ncol=nrc,byrow=T),
		  matrix(res$b,ncol=nrc,byrow=T) )

    colnames(res) <- c( sprintf("x[%i]",1:nmem),
    		        sprintf("a[%i]",1:nrc),
			sprintf("b[%i]",1:nrc) )
 
    coda::mcmc(data=res, start = burnin+1, end = burnin+samples, thin = thin)
}



