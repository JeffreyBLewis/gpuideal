#' Run the EM ideal estimator on the GPU
#'
#' Run the EM ideal estimator on the GPU
#' @param rcdata Roll call object
#' @param steps Number of MCMC step
#' @param thin Keep every `thin` step
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
gpu_em_ideal <- function(rcdata,steps,burnin=0,thin=1,x=NULL,abprior=matrix(c(20,0,0,20),2,2),xprior=1,
	             blocks=0, threads=0) {
    # Format the the roll call data
    nmem <- dim(rcdata$votes)[1]
    nrc <- dim(rcdata$votes)[2] 
    cat("Starting GPU-based EM ideal estimator...\n\n")
    cat("Data summary...\n");
    cat(sprintf("Number of members:   %i\n", nmem))
    cat(sprintf("Number of rollcalls: %i\n", nrc))

    # Set up column-wise (rollcall-wise data set)
    cw <- rc_recode( data.frame( y = as.vector(rcdata$votes),
       	                         rcidx = rep(1:nrc,each=nmem),
		                 memidx = rep(1:nmem,nrc) ),
                     rcdata$codes )
    check_rollcall_data(cw) # Sanity check the data.
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
    xx <- rep(0.0, steps*nmem/thin)
    if (is.null(x)) {
       xx[1:nmem] <- rnorm(nmem)
    }
    else {
       xx[1:nmem] <- x
    }

    cat(sprintf("Number of choices:   %i\n", dim(rw)[1]))

    # Inverse mills ratio table construction
    im_tbl_length <- 1001
    im_tbl_min <- -37.5
    im_tbl_range <- 75.0
    z <- seq(im_tbl_min, im_tbl_min+im_tbl_range, length=im_tbl_length)
    im_tbl <- dnorm(z)/pnorm(-z)

    # Run EM 
    timing <- system.time(
    res <- .C("em_ideal",
			# parameters coming back
			x=as.single(xx),
			a=single(steps*nrc/thin),
			b=single(steps*nrc/thin),
			
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
			steps=as.integer(steps),
			n=as.integer(n),
			n.rc = as.integer(nrc),
			n.mem = as.integer(nmem), 
			thin = as.integer(thin),
                        blocks = as.integer(blocks),
                        threads = as.integer(threads),

			# Mills ratio table
			im_tbl = as.single(im_tbl),
			im_tbl_min = as.single(im_tbl_min),
		  	im_tbl_range = as.single(im_tbl_range),
			im_tbl_length = as.integer(im_tbl_length)
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
    res
}


