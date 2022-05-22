sigmoid <- function( x ) {
  return( 1 / (1 + exp( -x ) ) )
}

dsigmoid <- function( x ) {
  return( x * (1 - x) )
}

ReLu <- function( x ) {
  return( ifelse( x <= 0, 0, x ) )
}

dReLu <- function( x ) {
  return( ifelse( x <= 0, 0, 1 ) )
}

tnh <- function(x) {
  return (tanh(x))
}

dtanh <- function(x) {
  return (1 - (tanh(x)^2))
}

lossSS <- function( y_tar, y_hat ) {
  return( 1/2 * sum( ( y_tar - y_hat )^2 ) )
}

SoftMax <- function(x) {
  exp( x ) / sum( exp( x ) )
}

MinMaxOdwrot <- function( x, y_min, y_max ) {
  return(  x * (y_max - y_min) + y_min )
}  

wprzod <- function(X, wout, type, f, df) {
  
  #h1 <- cbind( matrix( 1, nrow = nrow(X) ), sigmoid( X %*% W1 )  )
  #h2 <- cbind( matrix( 1, nrow = nrow(X) ), sigmoid( h1 %*% W2 )  )
  
  hout <- list()
  
  for (i in 1:(length(wout)-1)) {
    
    # h1 <- cbind(matrix(1, nrow = nrow(X)), sigmoid(X %*% wout[[1]]))
    # h2 <- cbind(matrix(1, nrow = nrow(h1)), sigmoid(h1 %*% wout[[2]]))
    # h3 <- cbind(matrix(1, nrow = nrow(h2)), sigmoid(h2 %*% wout[[3]]))
    # h4 <- cbind(matrix(1, nrow = nrow(h3)), sigmoid(h3 %*% wout[[4]]))
    # h5 <- cbind(matrix(1, nrow = nrow(h4)), sigmoid(h4 %*% wout[[5]]))
    
    if (i == 1) {
      
      hout[[i]] <- cbind(matrix(1, nrow = nrow(X)), f(X %*% wout[[i]]))
      
    }
    
    else {
      
      hout[[i]] <- cbind(matrix(1, nrow = nrow(hout[[i-1]])), f(hout[[i-1]] %*% wout[[i]]))
      
    }
    
  }
  
  if (type == "bin") {
    y_hat <- f( hout[[length(hout)]] %*% wout[[length(wout)]] ) # klasyfikacja binarna
  }
  else if (type == "multiclass") {
    y_hat <- matrix(t(apply(hout[[length(hout)]] %*% wout[[length(wout)]], 1, SoftMax)), nrow = nrow(X)) # klasyfikacja wieloklasowa
  }
  else {
    y_hat <- hout[[length(hout)]] %*% wout[[length(wout)]] # regresja
  }
  
  return( list( y = y_hat, H = hout) )
}

wstecz <- function(X, y_tar, y_hat, wout, hout, lr, type, f, df) {
  
  if (type == "bin") {
    dy_hat <- (y_tar - y_hat) * df( y_hat ) # klasyfikacja binarna
  }
  else if (type == "multiclass") {
    dy_hat <- (y_tar - y_hat) / nrow( X ) # klasyfikacja wieloklasowa
  }
  else {
    dy_hat <- (y_tar - y_hat) # regresja
  }
  
  wOut <- list()
  dhout <- list()
  
  for (i in (length(hout)):1) {
    if (i == length(hout)) {
      dw <- t(hout[[i]]) %*% dy_hat
      dh <- dy_hat %*% t(wout[[i+1]]) * df(hout[[i]])
      dhout[[i]] <- dh
      wOut[[i]] <- wout[[i+1]] + lr * dw
    }
    else {
      # dW3 <- t(H2) %*% dy_hat
      # dH2<- dy_hat %*% t(W3) * dsigmoid( H2 )
      
      # dW2 <- t(H1) %*% dH2[,-1]
      # dH1<- dH2[,-1] %*% t(W2) * dsigmoid( H1 )
      
      # dW1 <- t(X) %*% dH1[,-1]
      
      # W1 <- W1 + lr * dW1
      # W2 <- W2 + lr * dW2
      # W3 <- W3 + lr * dW3
      dw <- t(hout[[i]]) %*% dh[,-1]
      dh <- dh[,-1] %*% t(wout[[i+1]])  * df(hout[[i]])
      wOut[[i]] <- wout[[i+1]] + lr *  dw
    }
  }
  
  wOut <- rev(wOut)
  
  dw <- t(X) %*% dh[,-1]
  w <- wout[[1]] + lr * dw
  rownames(w) <- NULL
  
  wOut[[length(wOut) + 1]] <- w
  
  return(W = rev(wOut))
}

pasteNumeric <- function(x) {
  return (as.numeric(paste(x)))
}

MinMaxRELU <- function(x, new_min = 0, new_max = 10) {
  return(((x - min(x)) / (max(x) - min(x))) * (new_max - new_min) + new_min)
}

MinMaxSIG <- function(x, new_min = 0, new_max = 1) {
  return(((x - min(x)) / (max(x) - min(x))) * (new_max - new_min) + new_min)
}

MinMaxTNH <- function(x, new_min = -1, new_max = 1) {
  return(((x - min(x)) / (max(x) - min(x))) * (new_max - new_min) + new_min)
}

trainNN <- function(Yname, Xnames, data, h = c(5,5), lr = 0.01, iter = 10000, seed = 123, method = "sigmoid", plt = 0) {

  if (method == "sigmoid") {
    f <- sigmoid
    df <- dsigmoid
    minN <- 0
    maxN <- 1
    norm <- MinMaxSIG
  }
  
  else if (method == "tanh") {
    f <- tnh
    df <- dtanh
    minN <- -1
    maxN <- 1
    norm <- MinMaxTNH
  }
  
  else if (method == "relu") {
    f <- ReLu
    df <- dReLu
    minN <- 0
    maxN <- 10
    norm <- MinMaxRELU
  }
  
  else {
    stop("\nwrong activation f.!")
  }
  
  if (is.factor(data[,Yname])) {
    if (length(unique(data[,Yname])) > 2) {
      type <- "multiclass"
      print("multiclass")
      
      d <- data.frame(y = as.factor(paste(data[,Yname])))
      y_tar <- model.matrix(~ y - 1, d)
      
      #x <- as.matrix(data[,Xnames])
      #y_tar <- (model.matrix( ~ Yname - 1, data ))
      if (length(Xnames) > 1) {
        x <- as.data.frame(lapply(data[,Xnames], function(i) {
          return (as.numeric(paste(i)))
        }))
        x <- sapply(x, norm)
        lth <- nrow(x)
      }
      else {
        x <- norm(as.numeric(paste(data[,Xnames])))
        lth <- length(x)
      }
    }
    else {
      type <- "bin"
      print("bin")
      
      y_tar <- as.matrix(as.numeric(paste(data[,Yname])))
      
      if (length(Xnames) > 1) {
        x <- as.data.frame(lapply(data[,Xnames], function(i) {
          return (as.numeric(paste(i)))
        }))
        x <- sapply(x, norm)
        lth <- nrow(x)
      }
      else {
        x <- norm(as.numeric(paste(data[,Xnames])))
        lth <- length(x)
      }
    }
  }
  
  else {
    type <- "reg"
    print("reg")
    
    y_max <- max(data[,Yname])
    y_min <- min(data[,Yname])
    
    y_tar <- norm(data[,Yname])
    y_tar <- as.matrix(y_tar)
    
    if (length(Xnames) > 1) {
      x <- as.data.frame(lapply(data[,Xnames], function(i) {
        return (as.numeric(paste(i)))
      }))
      x <- sapply(x, norm)
      lth <- nrow(x)
    }
    else {
      x <- norm(as.numeric(paste(data[,Xnames])))
      lth <- length(x)
    }
  }
  
  set.seed(seed)
  
  X <- cbind(rep(1,lth),x)
  # W1, .... W_liczba_warst_ukrytych+1
  # assign( paste0( "W", 1:W_liczba_warst_ukrytych+1), matrix( rnorm( h + 1 ) ) )
  # W1 <- matrix( runif( ncol(X) * h[1], -1, 1 ), nrow = ncol(X) )
  # W2 <- matrix( runif( (h[1]+1) * h[2], -1, 1 ), nrow = h[1] + 1 )
  # W3 <- matrix( runif( (h[2]+1) * ncol(y_tar), -1, 1 ), nrow = h[2] + 1 )
  #list for layers
  wout <- list()
  
  #input layer
  wout[[1]] <- matrix(rnorm(ncol(X) * h[1]), nrow = ncol(X))
  
  #hidden layers
  for (i in 1:length(h)) {
    if (i == length(h)) {
      wout[[i+1]] <- matrix(rnorm((h[i]+1) * ncol(y_tar)), nrow = h[i] + 1)
    }
    else {
      wout[[i+1]] <- matrix(rnorm((h[i]+1) * h[i+1]), nrow = h[i] + 1)
    }
  }
  
  #output layer = wout[[length(h)]]
  error <- double( iter )
  
  for( i in 1:iter ) {
    sygnalwprzod <- wprzod(X, wout, type, f, df)# W = list( W1, W2, W3 )
    sygnalwtyl <- wstecz(X, y_tar, y_hat = sygnalwprzod$y, wout, sygnalwprzod$H, lr, type, f, df)
    
    wout <- sygnalwtyl
    
    cat( paste0( "\rIteracja: ", i ) )
    error[i] <- lossSS( y_tar, sygnalwprzod$y_hat)
  }
  
  xwartosci <- seq( 1, iter, length = 1000 )
  
  if (plt == 1) {
    print( qplot( xwartosci, error[xwartosci], geom = "line", main = "Error", xlab = "Iteracje" ) )
  }
  
  if (type == "reg") {
    y <- MinMaxOdwrot(sygnalwprzod$y, y_min, y_max)
  }
  
  else {
    y <- sygnalwprzod$y
  }
  
  return(list(y_hat = y, W = wout))
}
