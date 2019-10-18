import matplotlib.pyplot as plt
import numpy as np
import lambda_eval
import os
import bindata
from astropy.convolution import convolve, Box1DKernel
from astropy.io import fits
import boxsmoothwitherr
import imp
from scipy.interpolate import interp1d
imp.reload(boxsmoothwitherr)
from astropy.stats import sigma_clip
from matplotlib.image import NonUniformImage
from matplotlib.colors import Normalize

gnprefix='/Volumes/FantomHD/halo7d_data/specbyid_goodsn/'
gsprefix='/Volumes/FantomHD/halo7d_data/specbyid_goodss/'
cosprefix='/Volumes/FantomHD/halo7d_data/specbyid_cosmos/'
egsprefix='/Volumes/FantomHD/halo7d_data/specbyid_egs/'
udsprefix='/Volumes/FantomHD/halo7d_data/specbyid_uds/'

gndirs=os.listdir(gnprefix)
gsdirs=os.listdir(gsprefix)
cosdirs=os.listdir(cosprefix)
egsdirs=os.listdir(egsprefix)
udsdirs=os.listdir(udsprefix)

lines=np.genfromtxt('linelist.csv',dtype=[('names','<U10'),('waveair','<f8'),('wavevac','<f8'),('which','<U10'),('comment','<U15')],delimiter=',')

#                 'Ca II K', 'Ca II H', '[OII]3726', '[OII]3729', '[NeIII]','[NeIII]', 'H12', 'H11', 'H10', 'H9', 'H8',

lineypos=np.array([.4,         .5,       .7,          .6,           .6,      .6,        .4,   .5,     .4,   .5,   .7,

                      #'hepsilon', 'hdelta','hgamma', 'hbeta', 'halpha', '[OIII]5007', '[OIII]4959','[OIII]4363',
                       .7,          .6,      .5,       .7,       .85,      .5,           .6,          .6,
                       
                      #'[NII]6549', '[NII]6583' '[NII]5755', '[SII]6717','[SII]6731', 'Na I', 'Na I', 'He I', 'He I',
                       .4,          .6,          .4,          .7,         .6,          .6,     .7,    .5,     .5,

                      #'He I', '[O I]6300','[SIII]', '[NeII]', '[NeII]', '[NeII]', '[OI]5577', 'Mg b', 'Mg b','Mg b',
                       .5,      .6,         .5,       .7,      .5,       .5,        .7,        .6,     .5,     .7,

                      #'[ArIII]', '[ArIII]', 'Ca II', 'Ca II', 'Ca II', '[SIII]', '[SIII]', 'P11', 'P10', 'P9', 'P8',
                       .7,        .6,         .5,      .7,      .6,      .5,       .7,       .6,   .5,    .7,   .8,

                      #'P7', 'palpha', 'pbeta','palpha', '[Fe II]', 'brgamma', 'H2 S(1)', 'H2 S(0)'
                       .7,   .6,       .5,     .7,       .6,        .7,        .6,         .5])


skylines=np.loadtxt('mk_telluric.txt')
skyspec=np.loadtxt('mk_skyspec2.txt')
plt.rcParams["figure.figsize"]=[36,12]

plt.rcParams['image.cmap'] = 'Greys'
plt.rcParams['axes.linewidth']=8
plt.rcParams['axes.labelsize']=40
plt.rcParams['font.size']=40
plt.rcParams['xtick.major.size']=40
plt.rcParams['xtick.minor.visible']=True
plt.rcParams['xtick.minor.width']=8
plt.rcParams['xtick.minor.size']=20
plt.rcParams['xtick.major.width']=10
plt.rcParams['xtick.top']=True
plt.rcParams['xtick.color']='black'
plt.rcParams['axes.labelweight'] = 'bold'



plt.rcParams['lines.linewidth']=4
plt.rcParams['lines.markersize']=10
plt.rcParams['errorbar.capsize']=0
plt.rcParams['hatch.linewidth']=2

def getbestz(wave,flux,continuumflux,fluxerr,linese,linesa,linesb,object,ztst=np.arange(-.01,3,.0005),continuumorder=3,fitcontinuum=False,savedir='/Users/carletont/halo7d/cutouts/'):

    if fitcontinuum:
        fitcont=np.polyfit(wave-min(wave),continuumflux,continuumorder)
        fluxminuscont=np.zeros_like(flux)+flux
        for i in np.arange(continuumorder):
            fluxminuscont-=fitcont[i]*(wave-min(wave))**(continuumorder-i)
    else:
        fluxminuscont=flux-continuumflux
    
    sigmaclipflux=sigma_clip(np.zeros_like(fluxminuscont)+fluxminuscont,sigma=3, maxiters=1,cenfunc=np.nanmean,stdfunc=np.nanstd).compressed()
    np.savetxt(savedir+'/'+str(object)+'/noise.txt',[np.nanstd(sigmaclipflux)])
    
    #snr=np.append(0,(flux-np.mean(sigmaclipflux))/np.std(sigmaclipflux))
    #snr=np.append(0,(fluxminuscont)/np.std(sigmaclipflux))
    skyinterp=np.interp(wave,skyspec[:,0],skyspec[:,1])
    #wsky=np.where(~np.isfinite(skyinterp))[0]
    #fluxerr[wsky]*=5
    snr=np.append(0,fluxminuscont/fluxerr+skyinterp)
    #snr=np.append(0,fluxminuscont/fluxerr)

    #coef=np.polyfit(wave,np.arange(len(wave)),1)
    #pixfromlambda=lambda lam:pixfromlambda_coef[0]*lam+pixfromlambda_coef[1]
    pixfromlambda=interp1d(wave,np.arange(len(wave)),bounds_error=False, fill_value=0)

    #emission lines
    if len(linese)>0:
        linesalle=np.repeat(linese,len(ztst)).reshape(len(linese),len(ztst))
        ztstalle=np.tile(ztst,len(linese)).reshape(len(linese),len(ztst))
        
        redshiftedlinesalle=linesalle*(1+ztstalle)


        redshiftedpixalle=np.round(pixfromlambda(redshiftedlinesalle)).astype(np.int)

        woff=np.where((redshiftedpixalle<1) | (redshiftedpixalle>=len(flux)+1))
    
        redshiftedpixalle[woff]=0

        ztote=snr[redshiftedpixalle]**2*np.sign(snr[redshiftedpixalle])
        zalle=np.nansum(ztote,axis=0)
    else:
        zalle=0

    #absorption lines
    if len(linesa)>0:
        linesalla=np.repeat(linesa,len(ztst)).reshape(len(linesa),len(ztst))
        ztstalla=np.tile(ztst,len(linesa)).reshape(len(linesa),len(ztst))
    
        redshiftedlinesalla=linesalla*(1+ztstalla)

        redshiftedpixalla=np.round(pixfromlambda(redshiftedlinesalla)).astype(np.int)

        woff=np.where((redshiftedpixalla<1) | (redshiftedpixalla>=len(flux)+1))
    
        redshiftedpixalla[woff]=0

        ztota=-snr[redshiftedpixalla]**2*np.sign(snr[redshiftedpixalla])
        zalla=np.nansum(ztota,axis=0)
    else:
        zalla=0
        
    #both lines
    if len(linesb)>0:
        linesallb=np.repeat(linesb,len(ztst)).reshape(len(linesb),len(ztst))
        ztstallb=np.tile(ztst,len(linesb)).reshape(len(linesb),len(ztst))
        
        redshiftedlinesallb=linesallb*(1+ztstallb)
        
        redshiftedpixallb=np.round(pixfromlambda(redshiftedlinesallb)).astype(np.int)
        
        woff=np.where((redshiftedpixallb<1) | (redshiftedpixallb>=len(flux)+1))
        
        redshiftedpixallb[woff]=0

        ztotb=snr[redshiftedpixallb]**2
        zallb=np.nansum(ztotb,axis=0)
    else:
        zallb=0

    zall=zalle+zalla+zallb
    wn1=np.where(np.isfinite(zall))[0]
    wm=np.argmax(zall[wn1])
    np.savetxt(savedir+'/'+str(object)+'/zbest.txt',[ztst[wn1[np.argmax(zall[wn1])]]])

    return ztst[wn1[np.argmax(zall[wn1])]],fluxminuscont,sigmaclipflux

def makecutoutcoaddnocoadd(ax,smooths,xrng,minortickfrequency=100,majortickfrequency=500):
    ax[-1].set_title(r'No Coadd',y=.9,color='blue')
    ax[-1].set_xlim(xrng)
    ax[-1].set_ylim([0,200])
    minorxticks=np.arange(np.ceil(xrng[0]/minortickfrequency)*minortickfrequency,np.floor(xrng[1]/minortickfrequency)*minortickfrequency+1,minortickfrequency)
    majorxticks=np.arange(np.ceil(xrng[0]/majortickfrequency)*majortickfrequency,np.floor(xrng[1]/majortickfrequency)*majortickfrequency+1,majortickfrequency)
    ax[-1].set_xticks(minorxticks.tolist(),minor=True)
    ax[-1].set_xticks(majorxticks.tolist(),minor=False)
    ax[-1].axes.tick_params(direction='inout',which='both')

    for skline in range(len(skylines)):
        ax[-1].fill_between([skylines[skline,0]-skylines[skline,1]/2,skylines[skline,0]+skylines[skline,1]/2],0,200,color='k',alpha=.3)
    for sm in np.arange(len(smooths)-1,-1,-1):
        ax[sm].set_ylabel(r'Flux')
        ax[sm].set_ylim([0,200])
        ax[sm].set_xlim(xrng)
        ax[sm].set_title('Smooth '+str(smooths[sm]),y=0.9)
        minorxticks=np.arange(np.ceil(xrng[0]/minortickfrequency)*minortickfrequency,np.floor(xrng[1]/minortickfrequency)*minortickfrequency+1,minortickfrequency)
        majorxticks=np.arange(np.ceil(xrng[0]/majortickfrequency)*majortickfrequency,np.floor(xrng[1]/majortickfrequency)*majortickfrequency+1,majortickfrequency)
        ax[sm].set_xticks(minorxticks.tolist(),minor=True)
        ax[sm].set_xticks(majorxticks.tolist(),minor=False)
        ax[sm].axes.tick_params(direction='inout',which='both')
        for skline in range(len(skylines)):
            ax[sm].fill_between([skylines[skline,0]-skylines[skline,1]/2,skylines[skline,0]+skylines[skline,1]/2],0,200,color='k',alpha=.3)

    ax[-1].set_xlabel(r'$\lambda (\AA)$')
    ax[0].set_xlabel(r'$\lambda (\AA)$')
    ax[0].xaxis.set_label_position('top')
    ax[0].xaxis.tick_top()
    

def makecutoutcoadd(ax,object,prefix,savedir='/Users/carletont/halo7d/cutouts/',smooths=[1,2,5,10],minortickfrequency=100,majortickfrequency=500,continuumsmooth=300):

    spectra=fits.open(prefix+str(object)+'/coadd.GN16all.'+str(object)+'.fits')

    smoothedsignal,smoothederr=boxsmoothwitherr.boxsmoothwitherror(spectra[1].data.FLUX[0],1/np.sqrt(spectra[1].data.IVAR[0]),smooths[-1])
    wsignal=np.where(np.isfinite(smoothedsignal))[0]
    if len(wsignal)==0:
        wg0=np.where(spectra[1].data.LAMBDA[0]>0)[0]
        ax[-1].set_ylabel(r'Flux')
        ax[-1].set_ylim([0,200])

        xrng=[min(spectra[1].data.LAMBDA[0][wg0]),max(spectra[1].data.LAMBDA[0][wg0])]
        if xrng[1]<9000:
            xrng[1]=11000
        if xrng[0]>6000:
            xrng[0]=5000
        
    
        makecutoutcoaddnocoadd(ax,smooths,xrng,minortickfrequency=minortickfrequency,majortickfrequency=majortickfrequency)
        np.savetxt(savedir+'/'+str(object)+'/zbest.txt',[-1])
        np.savetxt(savedir+'/'+str(object)+'/noise.txt',[0])
        return
    else:
        buff=int(np.min([500/2,len(wsignal)/4]))

    binnedsignal=bindata.bindata(spectra[1].data.LAMBDA[0][wsignal[0]+buff:wsignal[-1]-buff],spectra[1].data.FLUX[0][wsignal[0]+buff:wsignal[-1]-buff],np.arange(min(spectra[1].data.LAMBDA[0][wsignal[0]+buff:wsignal[-1]-buff]),max(spectra[1].data.LAMBDA[0][wsignal[0]+buff:wsignal[-1]-buff]),smooths[-1]))
    #plt.clf()
    verysmoothsignal,verysmootherr=boxsmoothwitherr.boxsmoothwitherror(spectra[1].data.FLUX[0][wsignal[0]+buff:wsignal[-1]-buff],1/np.sqrt(spectra[1].data.IVAR[0][wsignal[0]+buff:wsignal[-1]-buff]),continuumsmooth,func=np.nanmedian)
    
    #plt.plot(spectra[1].data.LAMBDA[0][wsignal[0]+buff:wsignal[-1]-buff],spectra[1].data.FLUX[0][wsignal[0]+buff:wsignal[-1]-buff])
    #plt.plot(spectra[1].data.LAMBDA[0][wsignal[0]+buff:wsignal[-1]-buff],verysmoothsignal)
    
    sigmaclipflux=sigma_clip(spectra[1].data.FLUX[0][wsignal[0]+buff:wsignal[-1]-buff]-verysmoothsignal,sigma=3, maxiters=5,cenfunc=np.nanmedian,stdfunc=np.nanstd)
    wnotclipped=np.where(~sigmaclipflux.mask)[0]
    verysmoothsignal=np.interp(spectra[1].data.LAMBDA[0][wsignal[0]+buff:wsignal[-1]-buff],spectra[1].data.LAMBDA[0][wsignal[0]+buff:wsignal[-1]-buff][wnotclipped],verysmoothsignal[wnotclipped])

    #plt.plot(spectra[1].data.LAMBDA[0][wsignal[0]+buff:wsignal[-1]-buff][wnotclipped],verysmoothsignal[wnotclipped],'o')
    resmoothed,resmoothederr=boxsmoothwitherr.boxsmoothwitherror(spectra[1].data.FLUX[0][wsignal[0]+buff:wsignal[-1]-buff][wnotclipped],1/np.sqrt(spectra[1].data.IVAR[0][wsignal[0]+buff:wsignal[-1]-buff][wnotclipped]),continuumsmooth,func=np.nanmedian)
    #plt.plot(spectra[1].data.LAMBDA[0][wsignal[0]+buff:wsignal[-1]-buff][wnotclipped],resmoothed,'o')
    #print(np.std(spectra[1].data.FLUX[0][wsignal[0]+buff:wsignal[-1]-buff][wnotclipped]-verysmoothsignal[wnotclipped]))
    #plt.plot(spectra[1].data.LAMBDA[0][wsignal[0]+buff:wsignal[-1]-buff][wnotclipped],verysmoothsignal[wnotclipped])
    #plt.xlim(7000,8000)
    #plt.show()
#    verysmoothsignal,verysmootherr=boxsmoothwitherr.boxsmoothwitherror(sigmaclipflux,1/np.sqrt(spectra[1].data.IVAR[0][wsignal[0]+buff:wsignal[-1]-buff]),continuumsmooth)
    #binnedcontinuum=bindata.bindata(spectra[1].data.LAMBDA[0][wsignal[0]+buff:wsignal[-1]-buff],spectra[1].data.FLUX[0][wsignal[0]+buff:wsignal[-1]-buff],np.arange(min(spectra[1].data.LAMBDA[0][wsignal[0]+buff:wsignal[-1]-buff]),max(spectra[1].data.LAMBDA[0][wsignal[0]+buff:wsignal[-1]-buff]),continuumsmooth))
    #print(binnedsignal[2],binnedcontinuum[2],binnedcontinuum[0])
    
    #binnedcontinuum=np.interp(binnedsignal[2],spectra[1].data.LAMBDA[0][wsignal[0]+buff:wsignal[-1]-buff],verysmoothsignal)
    #zbest,fluxminuscont=getbestz(spectra[1].data.LAMBDA[0][wsignal[0]+buff:wsignal[-1]-buff],smoothedsignal[wsignal[0]+buff:wsignal[-1]-buff],lines['waves'])
    #zbest,fluxminuscont=getbestz(spectra[1].data.LAMBDA[0][wsignal[0]+buff:wsignal[-1]-buff],spectra[1].data.FLUX[0][wsignal[0]+buff:wsignal[-1]-buff],1/np.sqrt(spectra[1].data.IVAR[0][wsignal[0]+buff:wsignal[-1]-buff]),lines['waves'])

    we=np.where(lines['which']=='emission')[0]
    wa=np.where(lines['which']=='absorption')[0]
    wb=np.where(lines['which']=='both')[0]

    #wetofit=np.where((lines['names']=='[OII]3726') | (lines['names']=='[OII]3729') | (lines['names']=='[OIII]4959') | (lines['names']=='[OIII]5007') | (lines['names']=='halpha') | (lines['names']=='hbeta'))[0]
    #wetofit=np.where((lines['names']=='[OII]3729') | (lines['names']=='[OIII]4959') | (lines['names']=='[OIII]5007') | (lines['names']=='halpha') | (lines['names']=='hbeta'))[0]
    wetofit=np.where((lines['names']=='[OII]3729') | (lines['names']=='[OIII]4959') | (lines['names']=='[OIII]5007') |  (lines['names']=='hbeta'))[0]
    watofit=np.where((lines['names']=='Ca II H') | (lines['names']=='Ca II K') | (lines['names']=='Mg b 5167') | (lines['names']=='Na I 5890') | (lines['names']=='Ca IIa') | (lines['names']=='Ca IIb') | (lines['names']=='Ca IIc') | (lines['names']=='TiO'))[0]
    #watofit=np.where(lines['names']=='')[0]
    #wbtofit=np.array([],dtype=np.int)
    wbtofit=np.where(lines['names']=='halpha')[0]
    
    #zbest,fluxminuscont,sigmaclipflux=getbestz(binnedsignal[2],binnedsignal[0],binnedcontinuum,lines['waveair'][we],lines['waveair'][wa],lines['waveair'][wb])
    zbest,fluxminuscont,sigmaclipflux=getbestz(spectra[1].data.LAMBDA[0][wsignal[0]+buff:wsignal[-1]-buff],smoothedsignal[wsignal[0]+buff:wsignal[-1]-buff],verysmoothsignal,smoothederr[wsignal[0]+buff:wsignal[-1]-buff],lines['waveair'][wetofit],lines['waveair'][watofit],lines['waveair'][wbtofit],object,savedir=savedir)
    #zbest=getbestz(spectra[1].data.LAMBDA[0][5710:5760],smoothedsignal[5710:5760],lines['waves'][1:2])
    print(zbest)

    #ylim=[min([0,np.nanmin(smoothedsignal[200:-200])]),np.nanmax(smoothedsignal[200:-200])]
    wg0=np.where(spectra[1].data.LAMBDA[0]>0)[0]
    ylim=[max([np.nanmedian(fluxminuscont)-3*np.nanstd(fluxminuscont[200:-200]),-2000]),min([np.nanmax(smoothedsignal[200:-200]),10000])]
    
    ax[-1].plot(spectra[1].data.LAMBDA[0][wg0],smoothedsignal[wg0],linewidth=1,color='black')
    ax[-1].plot(spectra[1].data.LAMBDA[0][wsignal[0]+buff:wsignal[-1]-buff],fluxminuscont,linewidth=1,color='blue')
    ax[-1].plot(spectra[1].data.LAMBDA[0][wg0],smoothederr[wg0],linewidth=1,color='red',linestyle='-.')
    ax[-1].plot(spectra[1].data.LAMBDA[0][wg0],np.zeros_like(spectra[1].data.LAMBDA[0][wg0]),'k--',alpha=.5)
    for skline in range(len(skylines)):
        ax[-1].fill_between([skylines[skline,0]-skylines[skline,1]/2,skylines[skline,0]+skylines[skline,1]/2],ylim[0],ylim[1],color='k',alpha=.3)

    for i in range(len(lines['waveair'])):
        if lines['waveair'][i]*(1+zbest)>np.nanmin(spectra[1].data.LAMBDA[0][wg0]) and lines['waveair'][i]*(1+zbest)<np.nanmax(spectra[1].data.LAMBDA[0][wg0]):
            if lines['which'][i]=='emission':
                ax[-1].text(lines['waveair'][i]*(1+zbest),lineypos[i]*ylim[1],lines['names'][i],color='blue',ha='center')
                ax[-1].plot([lines['waveair'][i]*(1+zbest),lines['waveair'][i]*(1+zbest)],[ylim[0],ylim[1]],'--',color='blue',alpha=.3)
            elif lines['which'][i]=='absorption':
                ax[-1].text(lines['waveair'][i]*(1+zbest),lineypos[i]*ylim[1],lines['names'][i],color='red',ha='center')
                ax[-1].plot([lines['waveair'][i]*(1+zbest),lines['waveair'][i]*(1+zbest)],[ylim[0],ylim[1]],'--',color='red',alpha=.3)
            elif lines['which'][i]=='both':
                ax[-1].text(lines['waveair'][i]*(1+zbest),lineypos[i]*ylim[1],lines['names'][i],color='green',ha='center')
                ax[-1].plot([lines['waveair'][i]*(1+zbest),lines['waveair'][i]*(1+zbest)],[ylim[0],ylim[1]],'--',color='green',alpha=.3)

    ax[-1].set_ylabel(r'Flux')
    ax[-1].set_ylim(ylim)

    xrng=[min(spectra[1].data.LAMBDA[0][wg0]),max(spectra[1].data.LAMBDA[0][wg0])]
    if xrng[1]<9000:
        xrng[1]=11000
    if xrng[0]>6000:
        xrng[0]=5000
    ax[-1].set_xlim(xrng)
    
    ax[-1].set_title(r'$z=%1.3f$' % zbest,y=.9,color='blue')
    minorxticks=np.arange(np.ceil(xrng[0]/minortickfrequency)*minortickfrequency,np.floor(xrng[1]/minortickfrequency)*minortickfrequency+1,minortickfrequency)
    majorxticks=np.arange(np.ceil(xrng[0]/majortickfrequency)*majortickfrequency,np.floor(xrng[1]/majortickfrequency)*majortickfrequency+1,majortickfrequency)
    ax[-1].set_xticks(minorxticks.tolist(),minor=True)
    ax[-1].set_xticks(majorxticks.tolist(),minor=False)
    ax[-1].axes.tick_params(direction='inout',which='both')
    
    for sm in np.arange(len(smooths)-1,-1,-1):

        smoothedsignal,smoothederr=boxsmoothwitherr.boxsmoothwitherror(spectra[1].data.FLUX[0],1/np.sqrt(spectra[1].data.IVAR[0]),smooths[sm])
        if sm==len(smooths)-1:
            
            ylim=[min([0,np.nanpercentile(smoothedsignal[wsignal[0]+buff:wsignal[-1]-buff],5)]),min([np.nanmax(smoothedsignal[wsignal[0]+buff:wsignal[-1]-buff]),10000])]
        #ax[sm].errorbar(spectra[1].data.LAMBDA[0],smoothedsignal,yerr=smoothederr,linewidth=.5,color='black',alpha=.5)
        ax[sm].plot(spectra[1].data.LAMBDA[0][wg0],smoothedsignal[wg0],linewidth=1,color='black')
        ax[sm].plot(spectra[1].data.LAMBDA[0][wg0],smoothederr[wg0],linewidth=1,color='red',linestyle='-.')
        ax[sm].plot(spectra[1].data.LAMBDA[0][wg0],np.zeros_like(spectra[1].data.LAMBDA[0][wg0]),'k--',alpha=.5)

        ax[sm].set_ylabel(r'Flux')
        ax[sm].set_ylim(ylim)
        ax[sm].set_xlim(xrng)
        ax[sm].set_title('Smooth '+str(smooths[sm]),y=0.9)
        minorxticks=np.arange(np.ceil(xrng[0]/minortickfrequency)*minortickfrequency,np.floor(xrng[1]/minortickfrequency)*minortickfrequency+1,minortickfrequency)
        majorxticks=np.arange(np.ceil(xrng[0]/majortickfrequency)*majortickfrequency,np.floor(xrng[1]/majortickfrequency)*majortickfrequency+1,majortickfrequency)
        ax[sm].set_xticks(minorxticks.tolist(),minor=True)
        ax[sm].set_xticks(majorxticks.tolist(),minor=False)
        ax[sm].set_xticklabels([''])
        ax[sm].axes.tick_params(direction='inout',which='both')
        for skline in range(len(skylines)):
            ax[sm].fill_between([skylines[skline,0]-skylines[skline,1]/2,skylines[skline,0]+skylines[skline,1]/2],ylim[0],ylim[1],color='k',alpha=.3)

    ax[-1].set_xlabel(r'$\lambda (\AA)$')
    ax[0].set_xlabel(r'$\lambda (\AA)$')
    ax[0].xaxis.set_label_position('top')
    ax[0].xaxis.tick_top()

        

def makecutoutoned(ax,object,file,spectra,savedir='/Users/carletont/halo7d/cutouts/',smooths=[1,2,5,10],branch='b',ymin=0,ymax=350,minortickfrequency=100,majortickfrequency=500,xrng=[0,0]):

    

    if branch=='b':
        for sm in np.arange(len(smooths)-1,-1,-1):
            

            smoothedsignal,smoothederr=boxsmoothwitherr.boxsmoothwitherror(spectra[1].data.SPEC[0],1/np.sqrt(spectra[1].data.IVAR[0]),smooths[sm])
            if sm==len(smooths)-1:
                #ylim=[0,max(smoothedsignal[200:-200])*1.5]
                ylim=[np.min([0,np.nanpercentile(np.append(spectra[1].data.SPEC[0],spectra[2].data.SPEC[0]),5)]),np.nanpercentile(np.append(spectra[1].data.SPEC[0],spectra[2].data.SPEC[0]),95)]

            wg0=np.where(spectra[1].data.LAMBDA[0]>0)[0]
            #ax[sm,0].errorbar(spectra[1].data.LAMBDA[0],smoothedsignal,yerr=smoothederr,linewidth=.5,color='black',alpha=.2)
            ax[sm,0].plot(spectra[1].data.LAMBDA[0][wg0],smoothedsignal[wg0],linewidth=1,color='black')
            ax[sm,0].plot(spectra[1].data.LAMBDA[0][wg0],smoothederr[wg0],linewidth=1,color='red',linestyle='-.')

            ax[sm,0].set_ylabel(r'Flux')
            ax[sm,0].set_ylim(np.min([0,np.nanpercentile(np.append(spectra[1].data.SPEC[0],spectra[2].data.SPEC[0]),5)]),np.nanpercentile(np.append(spectra[1].data.SPEC[0],spectra[2].data.SPEC[0]),95))
            ax[sm,0].set_title('Smoothed '+str(smooths[sm]),position=[.5,.9])

            if xrng[0]==xrng[1]:
                xrng=[min(spectra[1].data.LAMBDA[0][wg0]),max(spectra[1].data.LAMBDA[0][wg0])]
            ax[sm,0].set_xlim(xrng)

            ax[sm,0].plot(spectra[1].data.LAMBDA[0][wg0],np.zeros_like(spectra[1].data.LAMBDA[0][wg0]),'k--',alpha=.5)
            ax[sm,0].set_xticks(np.arange(np.ceil(xrng[0]/minortickfrequency)*minortickfrequency,np.floor(xrng[1]/minortickfrequency)*minortickfrequency+1,minortickfrequency),minor=True)
            ax[sm,0].set_xticks(np.arange(np.ceil(xrng[0]/majortickfrequency)*majortickfrequency,np.floor(xrng[1]/majortickfrequency)*majortickfrequency+1,majortickfrequency),minor=False)
            if sm==len(smooths)-1:
                ax[sm,0].set_xlabel(r'$\lambda (\AA)$')
            else:
                ax[sm,0].set_xticklabels([''])

            if sm==1:
                ax[sm,0].axes.tick_params(direction='in',which='both')
            else:
                ax[sm,0].axes.tick_params(direction='inout',which='both')
            for skline in range(len(skylines)):
                ax[sm,0].fill_between([skylines[skline,0]-skylines[skline,1]/2,skylines[skline,0]+skylines[skline,1]/2],ylim[0],ylim[1],color='k',alpha=.3)
    if branch=='r':
        for sm in np.arange(len(smooths)-1,-1,-1):
            if sm==1:
                ax[sm,1].axes.tick_params(top=False)
            else:
                ax[sm,1].axes.tick_params(top=True)

            wg0=np.where(spectra[2].data.LAMBDA[0]>0)[0]

            smoothedsignal,smoothederr=boxsmoothwitherr.boxsmoothwitherror(spectra[2].data.SPEC[0],1/np.sqrt(spectra[2].data.IVAR[0]),smooths[sm])
            if sm==len(smooths)-1:
                #ylim=[0,max(smoothedsignal[200:-200])]
                ylim=[np.min([0,np.nanpercentile(np.append(spectra[1].data.SPEC[0],spectra[2].data.SPEC[0]),5)]),np.nanpercentile(np.append(spectra[1].data.SPEC[0],spectra[2].data.SPEC[0]),95)]
            #ax[sm,1].errorbar(spectra[2].data.LAMBDA[0],smoothedsignal,yerr=smoothederr,linewidth=.5,color='black',alpha=.2)
            ax[sm,1].plot(spectra[2].data.LAMBDA[0][wg0],smoothedsignal[wg0],linewidth=1,color='black')
            ax[sm,1].plot(spectra[2].data.LAMBDA[0][wg0],smoothederr[wg0],linewidth=1,color='red',linestyle='-.')

            #ax[sm,1].set_ylabel(r'Flux')
            ax[sm,1].set_yticklabels([''])
            ax[sm,1].set_ylim(np.min([0,np.nanpercentile(np.append(spectra[1].data.SPEC[0],spectra[2].data.SPEC[0]),5)]),np.nanpercentile(np.append(spectra[1].data.SPEC[0],spectra[2].data.SPEC[0]),95))

            if xrng[0]==xrng[1]:
                xrng=[min(spectra[1].data.LAMBDA[0][wg0]),max(spectra[1].data.LAMBDA[0][wg0])]
            ax[sm,1].set_xlim(xrng)
            ax[sm,1].plot(spectra[2].data.LAMBDA[0][wg0],np.zeros_like(spectra[2].data.LAMBDA[0][wg0]),'k--',alpha=.5)
            ax[sm,1].set_title('Smoothed '+str(smooths[sm]),position=[.5,.9])
            ax[sm,1].set_xticks(np.arange(np.ceil(xrng[0]/minortickfrequency)*minortickfrequency,np.floor(xrng[1]/minortickfrequency)*minortickfrequency+1,minortickfrequency),minor=True)
            ax[sm,1].set_xticks(np.arange(np.ceil(xrng[0]/majortickfrequency)*majortickfrequency,np.floor(xrng[1]/majortickfrequency)*majortickfrequency+1,majortickfrequency),minor=False)
            if sm==len(smooths)-1:
                ax[sm,1].set_xlabel(r'$\lambda (\AA)$')
            else:
                ax[sm,1].set_xticklabels([''])

            if sm==1:
                ax[sm,1].axes.tick_params(direction='in',which='both')
            else:
                ax[sm,1].axes.tick_params(direction='inout',which='both')
                
            for skline in range(len(skylines)):
                ax[sm,1].fill_between([skylines[skline,0]-skylines[skline,1]/2,skylines[skline,0]+skylines[skline,1]/2],ylim[0],ylim[1],color='k',alpha=.3)

    if branch=='nb':
        for sm in np.arange(len(smooths)-1,-1,-1):

            ax[sm,0].set_ylabel(r'Flux')
            ax[sm,0].set_ylim(ymin,ymax)
            ax[sm,0].set_title('Smoothed '+str(smooths[sm]),position=[.5,.9])

            ax[sm,0].set_xlim(xrng)

            ax[sm,0].set_xticks(np.arange(np.ceil(xrng[0]/minortickfrequency)*minortickfrequency,np.floor(xrng[1]/minortickfrequency)*minortickfrequency+1,minortickfrequency),minor=True)
            ax[sm,0].set_xticks(np.arange(np.ceil(xrng[0]/majortickfrequency)*majortickfrequency,np.floor(xrng[1]/majortickfrequency)*majortickfrequency+1,majortickfrequency),minor=False)
            if sm==len(smooths)-1:
                ax[sm,0].set_xlabel(r'$\lambda (\AA)$')
            else:
                ax[sm,0].set_xticklabels([''])

            if sm==1:
                ax[sm,0].axes.tick_params(direction='in',which='both')
            else:
                ax[sm,0].axes.tick_params(direction='inout',which='both')
            for skline in range(len(skylines)):
                ax[sm,0].fill_between([skylines[skline,0]-skylines[skline,1]/2,skylines[skline,0]+skylines[skline,1]/2],ylim[0],ylim[1],color='k',alpha=.3)

    if branch=='nr':
        for sm in np.arange(len(smooths)-1,-1,-1):
            if sm==1:
                ax[sm,1].axes.tick_params(top=False)
            else:
                ax[sm,1].axes.tick_params(top=True)

            
            #ax[sm,1].set_ylabel(r'Flux')
            ax[sm,1].set_yticklabels([''])
            ax[sm,1].set_ylim(ymin,ymax)

            ax[sm,1].set_xlim(xrng)

            ax[sm,1].set_title('Smoothed '+str(smooths[sm]),position=[.5,.9])
            ax[sm,1].set_xticks(np.arange(np.ceil(xrng[0]/minortickfrequency)*minortickfrequency,np.floor(xrng[1]/minortickfrequency)*minortickfrequency+1,minortickfrequency),minor=True)
            ax[sm,1].set_xticks(np.arange(np.ceil(xrng[0]/majortickfrequency)*majortickfrequency,np.floor(xrng[1]/majortickfrequency)*majortickfrequency+1,majortickfrequency),minor=False)
            if sm==len(smooths)-1:
                ax[sm,1].set_xlabel(r'$\lambda (\AA)$')
            else:
                ax[sm,1].set_xticklabels([''])

            if sm==1:
                ax[sm,1].axes.tick_params(direction='in',which='both')
            else:
                ax[sm,1].axes.tick_params(direction='inout',which='both')
            for skline in range(len(skylines)):
                ax[sm,1].fill_between([skylines[skline,0]-skylines[skline,1]/2,skylines[skline,0]+skylines[skline,1]/2],ymin,ymax,color='k',alpha=.3)
        
                
def makecutouttwod(ax,object,file,twodspec,r1,r2,vmin,vmax,datmin,xmin,xmax,savedir='/Users/carletont/halo7d/cutouts/',labelwave=False,labelpix=False,minortickfrequency=100,majortickfrequency=500):
    
    shape=np.shape(twodspec[1].data.FLUX[0])

    waves=np.mean(lambda_eval.lambda_eval(twodspec[1].data.LAMBDAX[0],twodspec[1].data.TILTX[0],twodspec[1].data.DLAM[0][0],xsize=shape[1],ysize=shape[0]),axis=1)
    pixtowave=interp1d(np.arange(shape[1]),waves,bounds_error=False,fill_value='extrapolate')
    wavetopix=interp1d(waves,np.arange(shape[1]),bounds_error=False,fill_value='extrapolate')


    norm = Normalize(vmin=vmin,vmax=vmax,clip=True)
    if 'B.fits' in file:
        labelwaves=np.array([4500,5000,5500,6000,6500,7000,7500,8000,8500])
        newdat=np.log10(twodspec[1].data.FLUX[0]-datmin)[::-1,:]
        wnotfinite=np.where(~np.isfinite(newdat))
        newdat[wnotfinite]=vmin
        im=NonUniformImage(ax[0],cmap='Greys_r',extent=(0,shape[1],0,shape[0]))
        im.set_data(pixtowave(np.arange(4096)),np.arange(shape[0]),norm(newdat))
        ax[0].images.append(im)
        ax[0].set_ylim(0,shape[0])
        ax[0].set_xlim([xmin, xmax])

        
        ax[0].plot([xmin,xmin+100],[shape[0]-r1,shape[0]-r1],color='g',linewidth=4)
        ax[0].plot([xmax-100,shape[1]],[shape[0]-r1,shape[0]-r1],color='g',linewidth=4)

        ax[0].plot([xmin,xmin+100],[shape[0]-r2,shape[0]-r2],color='g',linewidth=4)
        ax[0].plot([xmax-100,shape[1]],[shape[0]-r2,shape[0]-r2],color='g',linewidth=4)

        ax[0].plot([xmin,xmax],[shape[0]-r1,shape[0]-r1],color='g',linewidth=4)
        ax[0].plot([xmin,xmax],[shape[0]-r2,shape[0]-r2],color='g',linewidth=4)
        origticks=ax[0].get_yticks()

        ax[0].set_yticks(np.array([shape[0]-i-1 for i in origticks if (i>=0 and i<shape[0])])[::-1])
        ax[0].set_yticklabels(np.array([str(int(i)) for i in origticks if (i>=0 and i<shape[0])])[::-1])
    else:
        labelwaves=np.array([7000,7500,8000,8500,9000,9500,10000,10500,11000])
        newdat=np.log10(twodspec[1].data.FLUX[0]-datmin)[::-1,:]
        im=NonUniformImage(ax[1],cmap='Greys_r',extent=(0,shape[1],0,shape[0]))
        im.set_data(pixtowave(np.arange(4096)),np.arange(shape[0]),norm(newdat))
        ax[1].images.append(im)
        ax[1].set_ylim(0,shape[0])
        ax[1].set_xlim([xmin, xmax])
        
        ax[1].set_yticklabels([''])
        
        ax[1].plot([xmin,xmin+100],[shape[0]-r1,shape[0]-r1],color='g',linewidth=4)
        ax[1].plot([xmax-100,shape[1]],[shape[0]-r1,shape[0]-r1],color='g',linewidth=4)

        ax[1].plot([xmin,xmin+100],[shape[0]-r2,shape[0]-r2],color='g',linewidth=4)
        ax[1].plot([xmax-100,shape[1]],[shape[0]-r2,shape[0]-r2],color='g',linewidth=4)

        ax[1].plot([xmin,xmax],[shape[0]-r1,shape[0]-r1],color='g',linewidth=4)
        ax[1].plot([xmin,xmax],[shape[0]-r2,shape[0]-r2],color='g',linewidth=4)

    labelwavesmajor=np.arange(np.ceil(xmin/majortickfrequency)*majortickfrequency,np.floor(xmax/majortickfrequency)*majortickfrequency+1,majortickfrequency)
    labelwavesminor=np.arange(np.ceil(xmin/minortickfrequency)*minortickfrequency,np.floor(xmax/minortickfrequency)*minortickfrequency+1,minortickfrequency)

    
    xlim=[xmin,xmax]
    


    if 'B.fits' in file:
        ax[0].set_xticks(labelwavesmajor,minor=False)
        ax[0].set_xticks(labelwavesminor,minor=True)
        
        if labelwave:
            ax[0].set_xticklabels(['%i' % i for i in labelwavesmajor],minor=False)
        else:
            ax[0].set_xticklabels(['' for i in range(len(labelwavesmajor))],minor=False)
    else:
        ax[1].set_xticks(labelwavesmajor,minor=False)
        ax[1].set_xticks(labelwavesminor,minor=True)
        if labelwave:
            ax[1].set_xticklabels(['%i' % i for i in labelwavesmajor],minor=False)
        else:
            ax[1].set_xticklabels(['' for i in range(len(labelwavesmajor))],minor=False)

    newx1=ax[0].twiny()
    newx1.set_xticks(np.arange(0,len(waves),500))
    newx2=ax[1].twiny()
    newx2.set_xticks(np.arange(0,len(waves),500))

    if labelpix:
        newx1.set_xticklabels([str(i) for i in np.arange(0,len(waves),500)],minor=False)
        newx2.set_xticklabels([str(i) for i in np.arange(0,len(waves),500)],minor=False)
    else:
        newx1.set_xticklabels(['' for i in np.arange(0,len(waves),500)],minor=False)
        newx2.set_xticklabels(['' for i in np.arange(0,len(waves),500)],minor=False)

    ax[0].axes.tick_params(direction='in',axis='x',width=5,length=30,which='major')
    ax[0].axes.tick_params(direction='in',axis='x',width=2,length=15,which='minor')

    ax[1].axes.tick_params(direction='in',axis='x',width=5,length=30,which='major')
    ax[1].axes.tick_params(direction='in',axis='x',width=2,length=15,which='minor')

    newx1.axes.tick_params(direction='in',axis='x',width=5,length=30,which='major')
    newx1.axes.tick_params(direction='in',axis='x',width=2,length=15,which='minor')

    newx2.axes.tick_params(direction='in',axis='x',width=5,length=30,which='major')
    newx2.axes.tick_params(direction='in',axis='x',width=2,length=15,which='minor')

    return waves
    #plt.xlim(0,shape[1])
    #plt.savefig(savedir+str(object)+'/twod_'+file[7:-8]+'.png',dpi=100)
    
def makecutouts(object,savedir='/Users/carletont/halo7d/cutouts/',smooths=[1,5,10,20],field='goodss'):
    if field=='goodss':
        dirs=gsdirs
        prefix=gsprefix
    elif field=='goodsn':
        dirs=gndirs
        prefix=gnprefix
    elif field=='cosmos':
        dirs=cosdirs
        prefix=cosprefix
    elif field=='egs':
        dirs=egsdirs
        prefix=egsprefix
    elif field=='uds':
        dirs=udsdirs
        prefix=udsprefix
        
    if str(object) not in dirs:
        print('No observations with that id')
        return

    if str(object) not in os.listdir(savedir):
        os.system('mkdir '+savedir+str(object))

    onedobs=os.listdir(prefix+str(object)+'/1D/')
    twodobs=os.listdir(prefix+str(object)+'/2D/')

    #ymin,ymax=getylim.getylim(object)

    ax1bounds=[]
    fig1,ax1=plt.subplots(nrows=len(smooths)+1,ncols=1,figsize=[36*2,12*(len(smooths)+1)])
    if os.path.isfile(prefix+str(object)+'/coadd.GN16all.'+str(object)+'.fits'):

        makecutoutcoadd(ax1,object,prefix,savedir=savedir,smooths=smooths)
        fig1.tight_layout()
        fig1.subplots_adjust(hspace=0)
        for i in range(len(ax1)):
            ax1bounds.append(np.append(ax1[i].get_position().bounds,np.array([ax1[i].get_xlim()[0],ax1[i].get_ylim()[0],ax1[i].get_xlim()[1]-ax1[i].get_xlim()[0],ax1[i].get_ylim()[1]-ax1[i].get_ylim()[0]])))
            xrng=[ax1[i].get_xlim()[0],ax1[i].get_xlim()[1]]
                
        fig1.savefig(savedir+str(object)+'/coadd_smooth_'+str(object)+'.png')
    else:
        np.savetxt(savedir+'/'+str(object)+'/zbest.txt',[-1])
        np.savetxt(savedir+'/'+str(object)+'/noise.txt',[0])
        xrng=[-1,-1]

    
    chipmin=[]
    chipmax=[]
    
    for i in range(len(onedobs)):
        onedspectra=fits.open(prefix+str(object)+'/1D/'+onedobs[i])

        if i==0 and xrng[0]==-1:
            wg0=np.where(onedspectra[1].data.LAMBDA[0]>0)[0]
            xrng[0]=np.nanmin(onedspectra[1].data.LAMBDA[0][wg0])

            wg0=np.where(onedspectra[2].data.LAMBDA[0]>0)[0]
            xrng[1]=np.nanmax(onedspectra[2].data.LAMBDA[0][wg0])

            if xrng[1]<9000:
                xrng[1]=11000
            if xrng[0]>6000:
                xrng[0]=5000
            makecutoutcoaddnocoadd(ax1,smooths,xrng)
            fig1.tight_layout()
            fig1.subplots_adjust(hspace=0)
            
            fig1.savefig(savedir+str(object)+'/coadd_smooth_'+str(object)+'.png')

            for j in range(len(ax1)):
                ax1bounds.append(np.append(ax1[j].get_position().bounds,np.array([ax1[j].get_xlim()[0],ax1[j].get_ylim()[0],ax1[j].get_xlim()[1]-ax1[j].get_xlim()[0],ax1[j].get_ylim()[1]-ax1[j].get_ylim()[0]])))
            
        if os.path.isfile(prefix+str(object)+'/2D/'+'slit.'+onedobs[i][7:-6-len(str(object))]+'R'+'.fits.gz'):
            twodspecr=fits.open(prefix+str(object)+'/2D/'+'slit.'+onedobs[i][7:-6-len(str(object))]+'R'+'.fits.gz')
            datr=twodspecr[1].data.FLUX[0][onedspectra[2].data.R1[0]:onedspectra[2].data.R2[0]+1,:]
            sigmadatr=sigma_clip(datr,sigma=3, maxiters=2,cenfunc=np.nanmedian,stdfunc=np.nanstd)
            sigmadatr=sigmadatr.compressed()
            wfinite=np.where(np.isfinite(np.log10(sigmadatr-np.nanmin(sigmadatr))))[0]
            
            vminr=np.nanmedian(np.log10(sigmadatr[wfinite]-np.nanmin(sigmadatr)))-3*np.nanstd(np.log10(sigmadatr[wfinite]-np.nanmin(sigmadatr)))
            vmaxr=np.nanmedian(np.log10(sigmadatr[wfinite]-np.nanmin(sigmadatr)))+3*np.nanstd(np.log10(sigmadatr[wfinite]-np.nanmin(sigmadatr)))
        else:
            vminr=np.inf
            vmaxr=-np.inf
            sigmadatr=np.inf

        if os.path.isfile(prefix+str(object)+'/2D/'+'slit.'+onedobs[i][7:-6-len(str(object))]+'B'+'.fits.gz'):
            twodspecb=fits.open(prefix+str(object)+'/2D/'+'slit.'+onedobs[i][7:-6-len(str(object))]+'B'+'.fits.gz')
            datb=twodspecr[1].data.FLUX[0][onedspectra[1].data.R1[0]:onedspectra[1].data.R2[0]+1,:]
            sigmadatb=sigma_clip(datb,sigma=3, maxiters=2,cenfunc=np.nanmedian,stdfunc=np.nanstd)
            sigmadatb=sigmadatb.compressed()
            wfinite=np.where(np.isfinite(np.log10(sigmadatb-np.nanmin(sigmadatb))))

            vminb=np.nanmedian(np.log10(sigmadatb[wfinite]-np.nanmin(sigmadatb)))-3*np.nanstd(np.log10(sigmadatb[wfinite]-np.nanmin(sigmadatb)))
            vmaxb=np.nanmedian(np.log10(sigmadatb[wfinite]-np.nanmin(sigmadatb)))+3*np.nanstd(np.log10(sigmadatb[wfinite]-np.nanmin(sigmadatb)))
        else:
            vminb=np.inf
            vmaxb=-np.inf
            sigmadatb=np.inf

        vmin=min([vminr,vminb])
        vmax=max([vmaxr,vmaxb])
        datmin=min([np.nanmin(sigmadatr),np.nanmin(sigmadatb)])

        #fig2,ax2=plt.subplots(nrows=len(smooths)+1,ncols=2,figsize=[36*2,12*(len(smooths)+1)],sharex=False)
        fig21,ax21=plt.subplots(nrows=len(smooths),ncols=2,figsize=[36*2,12*(len(smooths))],sharex=False)
        fig21.subplots_adjust(hspace=0,wspace=0.01)
        fig22,ax22=plt.subplots(nrows=1,ncols=2,figsize=[36*2,5],sharex=False)
        fig22.subplots_adjust(hspace=0,wspace=0.01)
        if os.path.isfile(prefix+str(object)+'/2D/'+'slit.'+onedobs[i][7:-6-len(str(object))]+'B'+'.fits.gz'):
            makecutouttwod(ax22,object,'slit.'+onedobs[i][7:-6-len(str(object))]+'B'+'.fits.gz',twodspecb,onedspectra[1].data.R1[0],onedspectra[1].data.R2[0]+1,vmin,vmax,datmin,xrng[0],np.nanmax(onedspectra[1].data.LAMBDA[0]),savedir=savedir,labelwave=False,labelpix=False)
            makecutoutoned(ax21,object,onedobs[i],onedspectra,savedir=savedir,smooths=smooths,branch='b',ymin=ax1[0].get_ylim()[0],ymax=ax1[0].get_ylim()[1],xrng=[xrng[0],np.nanmax(onedspectra[1].data.LAMBDA[0])])
            chipmin.append(np.nanmax(onedspectra[1].data.LAMBDA[0]))
        else:
            ax22[0].imshow(np.zeros_like(twodspecr[1].data.FLUX[0])-vmin,aspect='auto',vmin=vmin,vmax=vmax)
            ax22[0].set_yticklabels(['' for i in range(len(ax22[0].get_yticks()))],minor=False)
            ax22[0].set_xticklabels(['' for i in range(len(ax22[0].get_xticks()))],minor=False)
            makecutoutoned(ax21,object,onedobs[i],onedspectra,savedir=savedir,smooths=smooths,branch='nb',ymin=ax1[0].get_ylim()[0],ymax=ax1[0].get_ylim()[1],xrng=[xrng[0],np.nanmin(onedspectra[2].data.LAMBDA[0])-100])

        if os.path.isfile(prefix+str(object)+'/2D/'+'slit.'+onedobs[i][7:-6-len(str(object))]+'R'+'.fits.gz'):
            makecutouttwod(ax22,object,'slit.'+onedobs[i][7:-6-len(str(object))]+'R'+'.fits.gz',twodspecr,onedspectra[2].data.R1[0],onedspectra[2].data.R2[0]+1,vmin,vmax,datmin,np.nanmin(onedspectra[2].data.LAMBDA[0]),xrng[1],savedir=savedir,labelwave=False,labelpix=False)
            makecutoutoned(ax21,object,onedobs[i],onedspectra,savedir=savedir,smooths=smooths,branch='r',ymin=ax1[0].get_ylim()[0],ymax=ax1[0].get_ylim()[1],xrng=[np.nanmin(onedspectra[2].data.LAMBDA[0]),xrng[1]])
            chipmax.append(np.nanmin(onedspectra[2].data.LAMBDA[0]))
        else:
            ax22[1].imshow(np.zeros_like(twodspecb[1].data.FLUX[0])-vmin,aspect='auto',vmin=vmin,vmax=vmax)
            ax22[1].set_yticklabels(['' for i in range(len(ax22[0].get_yticks()))],minor=False)
            ax22[1].set_xticklabels(['' for i in range(len(ax22[0].get_xticks()))],minor=False)
            makecutoutoned(ax21,object,onedobs[i],onedspectra,savedir=savedir,smooths=smooths,branch='nr',ymin=ax1[0].get_ylim()[0],ymax=ax1[0].get_ylim()[1],xrng=[np.nanmax(onedspectra[1].data.LAMBDA[0])+100,xrng[1]])
        
        fig21.tight_layout()
        fig21.subplots_adjust(hspace=0,wspace=0.01)

        #fig22.tight_layout()
        deltax0=ax21[0,0].get_xlim()[1]-ax21[0,0].get_xlim()[0]
        deltax1=ax21[0,1].get_xlim()[1]-ax21[0,1].get_xlim()[0]
        deltaxc=ax1[0].get_xlim()[1]-ax1[0].get_xlim()[0]

        for j in range(len(smooths)):
            newwidth1=deltax0/deltaxc*ax1[0].get_position().bounds[2]
            newwidth2=deltax1/deltaxc*ax1[0].get_position().bounds[2]
            ax21[j,0].set_position([ax1[j].get_position().bounds[0],ax21[j,0].get_position().bounds[1],newwidth1,ax21[j,0].get_position().bounds[3]])
            ax21[j,1].set_position([ax1[j].get_position().bounds[0]+ax1[j].get_position().bounds[2]-newwidth2,ax21[j,0].get_position().bounds[1],newwidth2,ax21[j,0].get_position().bounds[3]])
        ax22[0].set_position([ax1[0].get_position().bounds[0],ax22[0].get_position().bounds[1],newwidth1,ax22[0].get_position().bounds[3]])
        ax22[1].set_position([ax21[0,1].get_position().bounds[0],ax22[1].get_position().bounds[1],newwidth2,ax22[1].get_position().bounds[3]])

        fig21.savefig(savedir+str(object)+'/'+'cutout1_'+onedobs[i][7:-6-len(str(object))]+'.'+str(object)+'.png',dpi=20)
        fig22.savefig(savedir+str(object)+'/'+'cutout2_'+onedobs[i][7:-6-len(str(object))]+'.'+str(object)+'.png',dpi=100)

        if i==0:
            for j in range(len(smooths)):
                ax1bounds.append(np.append(ax21[j,0].get_position().bounds,np.array([ax21[j,0].get_xlim()[0],ax21[j,0].get_ylim()[0],ax21[j,0].get_xlim()[1]-ax21[j,0].get_xlim()[0],ax21[j,0].get_ylim()[1]-ax21[j,0].get_ylim()[0]])))
                ax1bounds.append(np.append(ax21[j,1].get_position().bounds,np.array([ax21[j,1].get_xlim()[0],ax21[j,1].get_ylim()[0],ax21[j,1].get_xlim()[1]-ax21[j,1].get_xlim()[0],ax21[j,1].get_ylim()[1]-ax21[j,1].get_ylim()[0]])))
            ax1bounds.append(np.append(ax22[0].get_position().bounds,np.array([ax22[0].get_xlim()[0],ax22[0].get_ylim()[0],ax22[0].get_xlim()[1]-ax22[0].get_xlim()[0],ax22[0].get_ylim()[1]-ax22[0].get_ylim()[0]])))
            ax1bounds.append(np.append(ax22[1].get_position().bounds,np.array([ax22[1].get_xlim()[0],ax22[1].get_ylim()[0],ax22[1].get_xlim()[1]-ax22[1].get_xlim()[0],ax22[1].get_ylim()[1]-ax22[1].get_ylim()[0]])))
        fig21.clf()
        fig22.clf()
        plt.close(fig=fig22)
        plt.close(fig=fig21)

    for sax in range(len(ax1)):
        ax1[sax].fill_between([np.nanmin(chipmin)-10,np.nanmax(chipmax)+10],ax1[sax].get_ylim()[0],ax1[sax].get_ylim()[1],color='r',alpha=.2)

    fig1.savefig(savedir+str(object)+'/coadd_smooth_'+str(object)+'.png')
    np.savetxt(savedir+str(object)+'/bounds.txt',np.array(ax1bounds))
    fig1.clf()
    plt.close(fig=fig1)
    plt.close("all")

    #plt.figure()
