#!/usr/bin/python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from scipy import stats

@ticker.FuncFormatter
def major_formatter(x, pos):
    label = str(int(-x)) if x < 0 else str(int(x))
    return label

def manhattanPlotOneChromosome(assocLinearFile,chrNum,col):

    # 8 for UKB genotyped (plink1.9); 11 for imputed (plink2.0); 10 for BOLT-LMM
    pvalColumn = 11
    # 2 for UKB genotyped (plink1.9); 1 for imputed (plink2.0); 2 for BOLT-LMM
    posColumn = 1

    plt.style.use('default')
    positions = []
    pvals = []
    with open(assocLinearFile,'r') as f:
        header = f.readline()
        for line in f:
            if 'NA' not in line:
                stuff = line.rstrip().split()
                positions.append(int(stuff[posColumn]))
                p = float(stuff[pvalColumn])
                pvals.append(-np.log10(p))

    f,a = plt.subplots(1,1,figsize=(15,4))
    positions = np.array(positions)
    x = positions/1000000
    plt.scatter(x,pvals,s=3,color=col)
    sigVal = -np.log10(5e-8)
    plt.plot([np.min(x),np.max(x)],[sigVal,sigVal],'--r')
    suggVal = -np.log10(1e-5)
    plt.plot([np.min(x),np.max(x)],[suggVal,suggVal],'--k')
    plt.ylabel('-log(p)',fontsize=16)
    plt.xlabel('Chr ' + str(chrNum) + ' Position (Mb)',fontsize=16)
    #plt.ylim([0,50])
    #plt.xlim([25.4,27.4])

    return positions, pvals, f

def pvalCorrPlot(toPlot,labs,cols):
    f,a = plt.subplots(1,1, figsize = (5,5))
    overallMax = np.max([np.max(toPlot[0]),np.max(toPlot[1])])
    a.plot([0,overallMax],[0,overallMax],'--m')
    a.scatter(toPlot[0],toPlot[1],color = 'k')
    a.set_xlabel(labs[0],fontsize=16)
    a.set_ylabel(labs[1],fontsize=16)
    a.set_title('-log10(p)')
    sigVal = -np.log10(1e-8)
    a.plot([0,sigVal],[sigVal,sigVal],'--r')
    a.plot([sigVal,sigVal],[0,sigVal],'--r')
    a.xaxis.label.set_color(cols[0])
    a.yaxis.label.set_color(cols[1])
    plt.show()

def pvalRatioPlot(toPlot,chrPos,chrNum):
    f,a = plt.subplots(1,1,figsize=(15,4))
    posRatio = np.log10(np.array(toPlot[0]) / np.array(toPlot[1]))
    posRatio = posRatio * max(toPlot) # scaled to significant p-values
    xvals = chrPos/1000000
    a.scatter(xvals,posRatio,color='k')
    a.plot([min(xvals),max(xvals)],[1.5,1.5],'--r')
    a.plot([min(xvals),max(xvals)],[-1.5,-1.5],'--r')
    plt.ylabel('log10(p value ratio)',fontsize=16)
    plt.xlabel('Chr ' + str(chrNum) + ' Position (Mb)',fontsize=16)
    plt.show()

def load_chromosomes(fstem,fdir):

    betaColumn = 8 # 6 for UKB genotyped (plink1.9); 8 for imputed (plink2.0)
    pvalColumn = 11 # 8 for UKB genotyped (plink1.9); 11 for imputed (plink2.0)
    positionColumn = 1 # 2 for UKB genotyped (plink1.9); 1 for imputed (plink2.0)
    variantColumn = 2 # 1 for UKB genotyped (plink1.9); 1 for imputed (plink2.0)
    seColumn = 9

    fsearch = fdir + fstem + '*all.glm*'
    files = glob.glob(fsearch)
    if len(files) == 0:
        print('no files!')
        return
    else:
        print(files[0])

    variants = {}  # key = chromosome numbers; values = list of SNPs
    positions = {} # key = chromosome numbers; values = list of positions
    pvals = {}     # key = chromosome numbers; values = list of pvals
    betas = {}     # key = chromosome numbers; values = list of betas
    errors = {}

    for file in files:
        chrnum = int(file.split('/')[1].split(fstem)[1].split('.')[0])
        positions[chrnum] = []
        pvals[chrnum] = []
        variants[chrnum] = []
        betas[chrnum] = []
        errors[chrnum] = []

        #print('Loading chromosome ' + str(chrnum))
        with open(file,'r') as f:
            header = f.readline()
            for line in f:
                stuff = line.rstrip().split()
                if len(stuff) > 8:
                    variants[chrnum].append(stuff[variantColumn])
                    positions[chrnum].append(int(stuff[positionColumn]))

                    if 'NA' in stuff[pvalColumn]:
                        p = float('nan')
                        beta = p
                        error = p
                    else:
                        p = float(stuff[pvalColumn])
                        beta = float(stuff[betaColumn])
                        error = float(stuff[seColumn])
                    pvals[chrnum].append(-np.log10(p))
                    betas[chrnum].append(beta)
                    errors[chrnum].append(error)
                else:
                    print(str(chrnum) + ' appears to be truncated')
    print('done loading chromosomes')
    return positions, variants, pvals, betas, errors

def manhattanPlotGenome(positions, pvals, max_y=0, snp_locs={}, flip=False, axcolor = 'w'):

    plt.style.use('default')
    f,a = plt.subplots(1,1,figsize=(9,2.5), dpi = 300)

    #color_strategy = 'tab10'
    color_strategy = 'bw'

    if color_strategy == 'bw':
        cols = [[0.04,0.04,0.04], [0.45,0.45,0.45]] * 11
        colorval = 0
    else:
        cols = cm.get_cmap(color_strategy)
        colorval = 0

    xscale = 1000
    x_start_chrom = 0
    previous_chrom_length = 0
    x_chrom_labels = []

    for chrom in sorted(positions.keys()):

        # where should this chromosome start?
        x_start_chrom = x_start_chrom + previous_chrom_length

        # show highlighted snps if present
        if len(snp_locs) > 0 and chrom in snp_locs.keys():
            for pos in snp_locs[chrom]:
                pos = pos / xscale
                pos = pos + x_start_chrom
                a.plot([pos,pos],[0,max_y],':',linewidth=1, color='darkgrey')

        # get color for scatter
        if color_strategy == 'bw':
            col = cols[colorval]
            colorval += 1
        else:
            if colorval < 1:
                col = cols(colorval)
                colorval += 0.11
            else:
                col = cols(0)
                colorval = 0.11

        # how long should this chromosome go?
        bp = np.array(positions[chrom])
        previous_chrom_length = max(bp) / xscale

        # where should points go?
        xpos = bp / xscale
        xpos = xpos + x_start_chrom

        # where should chromosome label go?
        x_chrom_labels.append(x_start_chrom + previous_chrom_length/2)

        # get pvals
        y = np.array(pvals[chrom])

        # show genome wide significance
        sigVal = -np.log10(5e-8)

        # plot
        scatterDotSize = 15
        if flip == True:
            y = -y
            sigVal = -sigVal
        a.scatter(xpos, y , c = [col], s = scatterDotSize)
        plt.plot([0,np.max(xpos)],[sigVal,sigVal],'--r',linewidth=1)

    # label axes
    xlabs = [str(x) for x in sorted(positions.keys())]
    xtickPos = [x-0.5 for x in sorted(positions.keys())]

    if flip == False:
        plt.xticks(x_chrom_labels,xlabs,fontsize=10,rotation = 90)
        #plt.xlabel('Chromosome', fontsize = 16)
    else:
        plt.xticks([])

    plt.ylabel('-log10(p)', fontsize = 16)

    # set max y axis, if specified
    if max_y > 0 and flip == False:
        plt.ylim([0,max_y])

    elif max_y > 0 and flip == True:
        plt.ylim([-max_y,0])
        a.yaxis.set_major_formatter(major_formatter)

    # span the length of the x axis
    plt.xlim([0,x_start_chrom + previous_chrom_length])

    # change axis color
    a.set_facecolor(axcolor)

    plt.show()

def get_locations(pvalthreshold,positionlist,snplist,pvallist):

    #(from each chromosome)
    maxLocus = 1000000
    loci = []
    locus = []

    #get indices of snps meeting threshold
    pvalArray = np.array(pvallist)
    pvalArray = np.nan_to_num(pvalArray)
    sigInd = np.where(pvalArray >= pvalthreshold)

    # if there are snps that meet threshold . . .
    if len(sigInd[0]) > 0:

        #get positions from these indices
        positionArray = np.array(positionlist)
        positions = positionArray[sigInd]
        positions = np.append(positions,positions[-1]+1)

        grabLocus = False

        #step through positions to find locations and ranges
        for i,p in enumerate(positions[:-1]):

            # do we need to collect this position as part of an ongoing locus?
            if grabLocus == True:
                locus.append(p)

                # look at next position - should it be part of this locus?
                if positions[i+1] - p > maxLocus:
                    grabLocus = False

            # we do not need to grab this locus, but we do need to save the old locus
            # and start a new locus
            else:

                # save the old locus
                if len(locus) > 0:
                    loci.append(locus)

                # start a new locus
                locus = [p]

                # look at next position - should it be part of this locus?
                if positions[i+1] - p < maxLocus:
                    grabLocus = True
                else:
                    grabLocus = False

        # done searching, add last locus
        loci.append(locus)

        loci = [[min(x),max(x)] for x in loci]
        #[f(x) if condition else g(x) for x in sequence]
        loci = [x[0] if x[0] == x[1] else x for x in loci]
    return loci # list of locations or ranges

def merge_locations(list1,list2):

    keepers = list1

    for locus2 in list2:
        foundOverlap = False
        if type(locus2) == list:
            start2 = min(locus2)
            end2 = max(locus2)
        else:
            start2 = locus2
            end2 = locus2+1

        # does this locus overlap with anything on list 1?
        for i, locus1 in enumerate(list1):
            if type(locus1) == list:
                start1 = min(locus1)
                end1 = max(locus1)
            else:
                start1 = locus1
                end1 = locus1+1

            # IF overlap
            loc1set = set(range(start1,end1))
            loc2set = set(range(start2,end2))
            if len(loc1set & loc2set) > 0:
                foundOverlap = True
                if start2 < start1:
                    if type(keepers[i]) == list:
                        keepers[i][0] = start2
                    else:
                        keepers[i] = [start2,keepers[i]]
                if end2 > end1:
                    if type(keepers[i]) == list:
                        keepers[i][1] = end2
                    else:
                        keepers[i] = [keepers[i],end2]
                break

        if foundOverlap == False:
            keepers.append(locus2)

    return keepers

def max_snp_in_range(boundaries,positionlist,snplist,pvallist,betalist,errlist):

    if type(boundaries) == list:
        start = boundaries[0]
        end = boundaries[1]
    else:
        start = boundaries
        end = start

    positionArray = np.array(positionlist)
    startInd = np.where(positionArray == start)[0][0]
    endInd = np.where(positionArray == end)[0][0]

    pos = positionArray[startInd:endInd+1]

    snpArray = np.array(snplist)
    snps = snpArray[startInd:endInd+1]

    pvalArray = np.array(pvallist)
    pvals = pvalArray[startInd:endInd+1]

    betaArray = np.array(betalist)
    betas = betaArray[startInd:endInd+1]

    errorArray = np.array(errlist)
    errors = errorArray[startInd:endInd+1]

    maxInd = np.nanargmax(pvals)
    return snps[maxInd], pos[maxInd], pvals[maxInd], betas[maxInd], errors[maxInd]

# make a dictionary of snps => information (pvals or betas or positions)
# input dictionaries are from load_chromosomes()
def getSnpDict(chromSnpDict,chromInfoDict):
    snpDict = {}
    for chrom in chromSnpDict.keys():
        chromDict = dict(zip(chromSnpDict[chrom],chromInfoDict[chrom]))
        snpDict.update(chromDict)
    return snpDict

# dictionary where key = snp, values = [position, pval, beta]
def snpList_to_info(snpList,positionDict,variantDict,pvalDict,betaDict):
    # positionDict,variantDict,pvalDict,betaDict are from load_chromosomes
    # from a particular GWAS

    snp_info = {}

    for chrom in np.arange(1,23):
        variants = variantDict[chrom]
        positions = positionDict[chrom]
        pvals = pvalDict[chrom]
        betas = betaDict[chrom]

        snps_on_chrom = list(set(snpList) & set(variants))
        for snp in snps_on_chrom:
            ind = variants.index(snp)
            pval = pvals[ind]
            beta = betas[ind]
            position = positions[ind]
            snp_info[snp] = [position,pval,beta]
    return snp_info

def decreasers_increasers_plot(snplist, snp_info1, snp_info2,
                               gwas1 = 'gwas one', gwas2 = 'gwas two'):

    plt.style.use('ggplot') # i like this!
    betas1 = np.array([snp_info1[x][2] for x in sorted(snplist)])
    betas2 = np.array([snp_info2[x][2] for x in sorted(snplist)])

    #diffs = np.absolute(betas1) - np.absolute(betas2)
    diffs = betas2 - betas1

    sames = 0
    decreasers = 0
    increasers = 0

    sign_switchers = 0

    for i,b in enumerate(betas1):
        if diffs[i] == 0:
            sames += 1
        elif b > 0 and betas2[i] > 0:
            if diffs[i] > 0:
                increasers += 1
            else:
                decreasers += 1
        elif b < 0 and betas2[i] < 0:
            if diffs[i] < 0:
                increasers += 1
            else:
                decreasers += 1
        else:
            sign_switchers += 1

    print('same: ', sames, '; Decreasers: ', decreasers, '; Increasers: ', increasers)
    print('sign switchers: ', sign_switchers)

    percentSame = "{:2.1f}%".format(sames/float(len(betas1)) * 100)
    percentDecreasers = "{:2.1f}%".format(decreasers/float(len(betas1)) * 100)
    percentIncreasers = "{:2.1f}%".format(increasers/float(len(betas1)) * 100)

    binomial_p = stats.binom_test(decreasers, n=decreasers+increasers, p=0.5, alternative='greater')
    print('binomial p-value for decreasers: ', '{:1.2e}'.format(binomial_p))
    binomial_p = stats.binom_test(decreasers, n=decreasers+increasers, p=0.5, alternative='less')
    print('binomial p-value for increasers: ', '{:1.2e}'.format(binomial_p))

    plt.style.use('ggplot') # i like this!
    plt.figure(figsize=(6,4), dpi=150)
    plt.bar([1,2,3],[sames,decreasers,increasers])
    plt.ylabel('# lead SNPs in \n' + gwas1 + ' vs ' + gwas2, fontsize=12)
    plt.xticks([1,2,3],['Equal effects','Less effect in/\n ' + gwas2,
                        'Greater effect in\n' + gwas2])
    plt.text(0.9,sames+2,percentSame,fontsize=12)
    plt.text(1.9,decreasers+2,percentDecreasers,fontsize=12)
    plt.text(2.9,increasers+2,percentIncreasers,fontsize=12)
    plt.ylim([0,np.max([sames,decreasers,increasers])+10])
    plt.show()

def beta_comp_plot(snplist, snp_info1, snp_info2, plotLims = 0.05, xlab = 'gwas one', ylab = 'gwas two'):
    # snp_info from snpList_to_info

    plt.style.use('ggplot') # i like this!
    betas1 = np.array([snp_info1[x][2] for x in sorted(snplist)])
    betas2 = np.array([snp_info2[x][2] for x in sorted(snplist)])

    abs_betas1 = np.absolute(betas1)
    abs_betas2 = np.absolute(betas2)

    pdiffs = np.zeros(len(betas1))
    for i,x in enumerate(betas1):
        pdiffs[i] = ( (betas2[i] - x) / abs(float(x)) ) * 100

    print('Number of loci: ' + str(len(betas1)))
    print('Average percentage difference in betas: ' + str('{:2.3f}'.format(np.mean(pdiffs))) )

    f,(ax1,ax2) = plt.subplots(1,2,figsize = (6.8,3.3), dpi=150)

    ax1.scatter(betas1,betas2,s=10)
    ax1.plot([-plotLims,plotLims],[-plotLims,plotLims],'--g')

    ax1.set_xlabel('betas: ' + xlab, fontsize=12)
    ax1.set_ylabel('betas: ' + ylab, fontsize=12)
    ax1.set_xlim([-plotLims,plotLims])
    ax1.set_ylim([-plotLims,plotLims])

    print('sum([' + ylab + ' Betas]) - sum([' + xlab + ' Betas]): ',
          '{:2.3f}'.format(np.sum(np.absolute(betas2)) - np.sum(np.absolute(betas1))))
    diffs = np.absolute(betas2) - np.absolute(betas1)

    decreasers = len(diffs[np.where(diffs < 0)])
    increasers = len(diffs[np.where(diffs > 0)])
    sames = len(diffs[np.where(diffs == 0)])
    print('same: ', sames, '; Decreasers: ', decreasers, '; Increasers: ', increasers)

    t,p = stats.ttest_1samp(diffs,0)
    print('one sample t-test val: ', '{:2.2e}'.format(p))

    ax2.boxplot(diffs)
    ax2.set_ylabel(ylab + ' betas - ' + xlab + ' betas',fontsize=12)

    plt.tight_layout()
    plt.show()

def pval_comp_plot(snplist, snp_info1, snp_info2, gwas1 = 'gwas 1', gwas2 = 'gwas 2'):
    # snp_info from snpList_to_info
    plt.style.use('ggplot') # i like this!
    pvals1 = np.array([snp_info1[x][1] for x in sorted(snplist)])
    pvals2 = np.array([snp_info2[x][1] for x in sorted(snplist)])

    overallMax = max([max(pvals1), max(pvals2)])
    overallMax = 1.1 * overallMax
    overallMin = min([min(pvals1), min(pvals2)])
    overallMin = 0.9 * overallMin

    diffs = pvals2 - pvals1

    t,p = stats.ttest_1samp(diffs,0)
    print('one sample t-test val: ', '{:2.2e}'.format(p))
    plt.style.use('ggplot') # i like this!
    f,(ax1,ax2) = plt.subplots(1,2,figsize = (6.8,3.3), dpi=150)

    ax1.scatter(pvals1,pvals2,s=5,c='k')
    ax1.set_xlim([overallMin,overallMax])
    ax1.set_ylim([overallMin,overallMax])
    ax1.plot([overallMin,overallMax],[overallMin,overallMax],'--g')
    ax1.set_xlabel('log10(p-values)\n' + gwas1,fontsize=12)
    ax1.set_ylabel('log10(p-values)\n' + gwas2,fontsize=12)

    ax2.set_ylabel(gwas2 + ' pVals - ' + gwas1 + ' pVals',fontsize=12)
    ax2.boxplot(diffs)
    plt.tight_layout()
    plt.show()


def get_pvals(dir):
    # return dictionary of snps => pvals
    pvalDict = {}
    for chrom in np.arange(1,23):
        fname = dir + '/chr' + str(chrom) + '.all.glm.linear'
        #print(fname)

        # load info
        chromDf = pd.read_csv(fname,delimiter = '\t')

        # pvals
        pvalDict.update(dict(zip(chromDf.ID.values, chromDf.P.values)))

    return pvalDict

def get_snp_info(dir,linear_or_logistic = 'linear'):

    # return dictionaries of
    # snps => alleles
    # snps => pvals
    # snps => betas
    # snps => errors

    betaDict = {}
    errorDict = {}
    alleleDict = {}
    pvalDict = {}

    if 'log' in linear_or_logistic:
        fileStem = '.all.glm.logistic'
    else:
        fileStem = '.all.glm.linear'

    for chrom in np.arange(1,23):
        fname = dir + '/chr' + str(chrom) + fileStem
        #print(fname)

        # load info
        chromDf = pd.read_csv(fname,delimiter = '\t')

        # get tested allele
        alleleDict.update(dict(zip(chromDf.ID.values, chromDf.A1.values)))

        if 'log' in linear_or_logistic:
            # betas
            betaDict.update(dict(zip(chromDf.ID.values, chromDf.OR.values))) # BETA or OR
            # errors
            errorDict.update(dict(zip(chromDf.ID.values, chromDf['LOG(OR)_SE'].values))) # SE or LOG(OR)_SE
            # pvals
            pvalDict.update(dict(zip(chromDf.ID.values, chromDf['P'].values))) # P

        else:
            # betas
            betaDict.update(dict(zip(chromDf.ID.values, chromDf.BETA.values))) # BETA or OR
            # errors
            errorDict.update(dict(zip(chromDf.ID.values, chromDf['SE'].values))) # SE or LOG(OR)_SE
            # pvals
            pvalDict.update(dict(zip(chromDf.ID.values, chromDf['P'].values))) # P

    return betaDict, errorDict, alleleDict, pvalDict


def get_weighted_avg_beta(betas,errors):

    # weighted betas
    ses_sqrd = np.square(errors)
    weights = 1 / ses_sqrd
    sum_weights = np.nansum(weights)

    weighted_betas = weights * betas
    w_avg_beta = np.nansum(weighted_betas) / sum_weights

    # variance
    var = 1 / sum_weights

    return w_avg_beta, var, weighted_betas / sum_weights * len(weighted_betas)

# use this for RAW betas, not weighted betas!
def calc_z_statistic(betas1, betas2, errors1, errors2, gwas1 = 'gwas1', gwas2 = 'gwas2'):
    avg_eff1, var1, wb1 = get_weighted_avg_beta(betas1,errors1)
    avg_eff2, var2, wb2 = get_weighted_avg_beta(betas2,errors2)

    numerator = avg_eff1 - avg_eff2
    denominator = np.sqrt(var1 + var2)

    print(gwas1 + ':  avg_eff ' + '{:1.4f}'.format(avg_eff1) + '  variance: ' + '{:1.3e}'.format(var1))
    print(gwas2 + ':  avg_eff ' + '{:1.4f}'.format(avg_eff2) + '  variance: ' + '{:1.3e}'.format(var2))

    zstat = numerator/denominator
    print('\ndifference in effect: ', '{:1.4f}'.format(numerator))
    print('\nz-statistic: ', '{:1.4f}'.format(zstat))

    return zstat, avg_eff1, avg_eff2

def qqplot(lists_of_pvals, labels, fs = (4,4)):
    plt.style.use('fivethirtyeight')
    cols = ['tab:blue', 'tab:green', 'tab:orange', 'tab:cyan', 'tab:red', 'tab:olive', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

    # which list is longest?
    max_len = max([len(x) for x in lists_of_pvals])

    # figure out how to downsample before plotting

    # plot y = x
    yxvals = np.linspace(0,1,max_len)
    yxvals = yxvals[1:]
    log_yx = sorted(-np.log10(yxvals))

    f,a = plt.subplots(1,1,figsize = fs, dpi = 150)
    a.scatter(log_yx, log_yx, c = 'k', label = 'y=x')

    for i, pvals in enumerate(lists_of_pvals):
        sorted_pvals = sorted(pvals)
        exp_pvals = np.linspace(0,1,len(sorted_pvals))

        obs_pvals = sorted_pvals[1:]
        exp_pvals = exp_pvals[1:]

        log_obs = sorted(-np.log10(obs_pvals))
        log_exp = sorted(-np.log10(exp_pvals))

        a.scatter(log_exp, log_obs, c = cols[i], label = labels[i])

    plt.legend(loc='best',fontsize=16)
    plt.ylabel('Observed -log10(p)')
    plt.xlabel('Expected -log10(p)')

    plt.show()
