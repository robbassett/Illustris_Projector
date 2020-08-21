########################################################
#Using ProFound to fit ellipse profiles to SAMI galaxies
#Author: Caroline Foster
#Date started: 18 August 2017
########################################################
#Loading libraries:
library(ProFound)
library(ProFit)

#Change directory:
setwd('/Users/rbassett/Desktop/Illustris/')
########################################################
fname = commandArgs(trailingOnly=TRUE)

#Running ellipses on fake galaxy:
  if (file.exists(fname)) {
    #Loading galaxy image.
    image=readFITS(file=fname)$imDat
    t=which(image == 0)
    image[t]=1.e-9
    pixelscale=0.3
    #Creating segmented map:
    seg_im = profoundMakeSegim(image,skycut=2.5,verbose=FALSE, plot=TRUE)
    
    ellipses_box = profoundGetEllipses(image=image,segim=seg_im$segim,levels=12,dobox=FALSE,pixscale=1,plot=TRUE)
    
    tmp = which(ellipses_box$ellipse$fluxfrac > .45 & ellipses_box$ellipse$fluxfrac < 0.55)
    tm2 = which(ellipses_box$ellipse$fluxfrac > .35 & ellipses_box$ellipse$fluxfrac < .65)
      
    elout= mean(ellipses_box$ellipse$axrat[tmp])
    eleo = sd(ellipses_box$ellipse$axrat[tm2]) 
    paout= mean(ellipses_box$ellipse$ang[tmp])
    paeo = sd(ellipses_box$ellipse$ang[tm2])
    reout= mean(ellipses_box$ellipse$radav[tmp])
    text(10,1,round(elout*1000.)/1000.)
    
  }
cat(c(1-elout,paout,reout,eleo,paeo))
    
