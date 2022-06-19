#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from IPython.display import Image
import math
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# calculating angle between points and vertical
def angle_from_vertical(xy1,xy2): # coordinates as tuples
    x1, y1 = xy1
    x2, y2 = xy2

    p1 = [x1,y1]
    p0 = [x1,y1+5]
    p2 = [x2,y2]

    # p0 = [3.5, 6.7]
    # p1 = [7.9, 8.4]
    # p2 = [10.8, 4.8]

    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))

    angle = np.degrees(angle)
    if angle > 0: # left from vertical
        direction = 360-angle
    else:
        direction = 0-angle

    return direction

# function to get distances between coordinates
def dist_between_points(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    return math.hypot(x2-x1, y2-y1)

def show_angles(angles,comparison=''):
    numBins = 36
    edges = np.linspace(10,360,numBins)
    counts = np.zeros(numBins)
    angles = np.array(angles)

    # stupid counting for bins because I couldn't figure out how to make histogram have same number of classes
    for a in angles:
        needCount = True
        for i, e in enumerate(edges):
            if a <= e and needCount == True:
                counts[i] += 1
                needCount = False

    fig, ax = plt.subplots(1,1,figsize=(10,5))
    bars = ax.bar(edges, counts, width=8)

    cmap = plt.cm.get_cmap('viridis')
    cols = [cmap(x) for x in np.arange(len(edges)) / float(len(edges))]

    # Use custom colors and opacity
    for i, bar in enumerate(bars):
        bar.set_facecolor(cols[i])
        bar.set_alpha(0.8)
    plt.xlabel('Angle of movement (0 = north)',fontsize=16)
    plt.ylabel('Number of ' + comparison + ' pairs',fontsize=16)

    return fig

def show_angles_circle(angles, fig_size, tick_size, axis_label_size, y_ticks):

    numBins = 36
    edges = np.linspace(10,360,numBins)
    #edges = edges-10
    counts = np.zeros(numBins)
    angles = np.array(angles)

    w = (1.5*np.pi) / len(edges)

    # stupid counting for bins because I couldn't figure out how to make histogram have same number of classes
    for a in angles:
        needCount = True
        for i, e in enumerate(edges):
            if a <= e and needCount == True:
                counts[i] += 1
                needCount = False

    # convert degrees to radians
    edges = np.deg2rad(edges)

    f = plt.figure(figsize = fig_size)
    ax = plt.subplot(111, polar=True)
    bars = ax.bar(edges, counts, width=w)

    # # set the label go clockwise and start from the top
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    cmap = plt.cm.get_cmap('viridis')
    cols = [cmap(x) for x in np.arange(len(edges)) / float(len(edges))]

    # Use custom colors and opacity
    for i, bar in enumerate(bars):
        bar.set_facecolor(cols[i])
        bar.set_alpha(0.8)

    ax.set_yticks(y_ticks)
    plt.locator_params(axis='y', nbins = 3)
    plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=tick_size)
    plt.setp(ax.get_yticklabels(), fontsize=tick_size)

    return f, counts

def get_ncolors_from_cmap(num_colors,colormap='viridis'):
    cmap = cm.get_cmap(colormap)
    cols = [cmap(x) for x in np.linspace(0,1,num_colors)]
    return cols

def color_coded_pheno_legend(codes,colList,text_size,ax):
    for i,c in enumerate(colList):
        codeText = codes[i]
        rect = Rectangle((1,i),0.8,0.8,color=c)
        ax.text(1.9,i+0.3,codeText,fontsize = text_size)
        ax.add_patch(rect)
    ax.set_xlim([0,5])
    ax.set_ylim([-1,len(colList)])
    ax.set_axis_off()
    return ax

def get_pheno_vector(phe, pheno_dfs, stayers, movers, showPlot = 'big'):
    # from a phenotype, and a list of mover / stayer eids
    # [stayer_phenotypes,mover_phenotypes]

    [pheno_cat, pheno_cont, pheno_int, pheno_strat] = pheno_dfs
    pheno_type = get_pheno_type(pheno_dfs)
    plotType, df = get_pheno_df(pheno_type[phe], pheno_dfs)

    # remove nan
    stayer_phenotypes = [x for x in df[df['eid'].isin(stayers)][phe].tolist() if x >= 0 ]
    mover_phenotypes =  [x for x in df[df['eid'].isin(movers) ][phe].tolist() if x >= 0 ]

    m_to_plot = [stayer_phenotypes, mover_phenotypes]

    # plot
    cols = []
    if showPlot == 'big' or showPlot == 'small':

        grouplabels = ['stayers','movers']
        plt.style.use('fivethirtyeight')
        #cls = ['tab:blue','tab:green']
        cls = [[0.705,0,0.411],[0.24,0.51,0.1]]

        if plotType == 'stacked':

            rect_colors = cls * int(len(m_to_plot)/2)

            if showPlot == 'big':
                fig_size = (6,8)
                axis_text_size = 24
                tick_text_size = 24
            else:
                fig_size = (2,1.8)
                axis_text_size = 8
                tick_text_size = 8

            #fig,(ax,ax2) = plt.subplots(1,2,figsize=fig_size)
            fig,ax = plt.subplots(1,1,figsize=fig_size,facecolor = 'w')

            # total number of categories
            n = np.unique(df[phe])
            n = n[~np.isnan(n)]
            n = n[np.where(n>=0)]
            numCats = len(n)

            plotVals = []
            cols = get_ncolors_from_cmap(numCats,colormap='gray')

            table_for_stats = []

            for i,m in enumerate(m_to_plot):

                labels, counts = np.unique(m, return_counts=True)
                percentages = [x/np.sum(counts)*100 for x in counts]
                print(grouplabels[i],percentages)
                table_for_stats.append(counts)

                # add background rectangles
                rect = Rectangle((i+0.5,0),1,105,color=rect_colors[i],alpha=0.4)
                ax.add_patch(rect)

                bot = 0
                for j,p in enumerate(percentages):
                    ax.bar(i+1, p , align='center', bottom=bot, width = 0.9, color=cols[j])
                    bot = bot + p



            ax.set_ylabel('percentage of samples',fontsize=axis_text_size)
            ax.set_xticks([1,2])
            ax.set_xticklabels(grouplabels,fontsize=tick_text_size)
            ax.set_facecolor('w')

            # color movers stayers?
            # different colors for different xtick labels:
            # https://stackoverflow.com/questions/21936014/set-color-for-xticklabels-individually-in-matplotlib
            cls = [[0.705,0,0.411],[0.24,0.51,0.1]]
            #cls = ['tab:blue','tab:green']
            [t.set_color(i) for (i,t) in zip(cls, ax.xaxis.get_ticklabels())]

            yt = [0,25,50,75,100]
            ax.set_yticks(yt)
            ax.set_ylim([0,105])
            ax.set_yticklabels([str(x) for x in yt],fontsize=tick_text_size)
            ax.set_title(phe,fontsize=axis_text_size)

            print('numbers:\t','\t'.join(str(y) for y in [len(x) for x in m_to_plot]))

            # pc = get_pheno_codes()
            # if phe in pc.keys():
            #     # make an image of a key
            #     codes = pc[phe]
            #     color_coded_pheno_legend(codes,cols,axis_text_size,ax2)
            #
            # else:
            #     # get a saved image of the key
            #     fname = phe + '.png'
            #     img=mpimg.imread('../05_pheno_codes/' + fname)
            #     ax2.imshow(img)
            #
            # ax2.set_axis_off()

        elif plotType == 'boxes':

            if showPlot == 'big':
                fig_size = (2,6)
                axis_text_size = 24
                tick_text_size = 24
            else:
                fig_size = (2,2.57)
                axis_text_size = 8
                tick_text_size = 8

            # integer or continuous data: plot as boxplots (or violinplots)
            fig, ax = plt.subplots(1,1,figsize=fig_size,facecolor = 'w')
            cls = [[0.705,0,0.411],[0.24,0.51,0.1]]
            #cls = ['tab:blue','tab:green']

            if len(m_to_plot) == 2:

                bp = plt.boxplot(m_to_plot[0], positions=[1], patch_artist=True,
                    showfliers = False)
                for element in ['boxes', 'whiskers', 'fliers', 'means', 'caps']:
                    plt.setp(bp[element], color=cls[0] , linewidth = 4)
                plt.setp(bp['medians'], color = 'y', linewidth = 5)

                bp = plt.boxplot(m_to_plot[1], positions=[2], patch_artist=True,
                    showfliers = False)
                for element in ['boxes', 'whiskers', 'fliers', 'means', 'caps']:
                    plt.setp(bp[element], color=cls[1], linewidth = 4)
                plt.setp(bp['medians'], color = 'y', linewidth = 5)

            else:
                bp = ax.boxplot(m_to_plot,showfliers=False)

            #ax.boxplot(m_to_plot)
            #ax.violinplot(m_to_plot)
            ax.set_xticks([1,2])
            ax.set_xticklabels(grouplabels,fontsize=tick_text_size)
            ax.set_facecolor('w')
            [t.set_color(i) for (i,t) in zip(cls, ax.xaxis.get_ticklabels())]

            plt.yticks(fontsize = tick_text_size)
            ax.set_ylabel(phe,fontsize=axis_text_size)

            if len(m_to_plot[0]) > 0:
                print('numbers: ',' '.join(str(y) for y in [len(x) for x in m_to_plot]))
                print('means: ', '  '.join(['{:01.2f}'.format(y) for y in [np.mean(x) for x in m_to_plot]]))

        ax.set_xlim([0.5,len(m_to_plot)+0.5])
        plt.tight_layout()
        plt.show()
    else:
        fig=''

    return m_to_plot, fig

# make dictionary: phenotypes => what file they are from (and what type they are)
def get_pheno_type(dfs):

    [pheno_cat, pheno_cont, pheno_int, pheno_strat] = dfs
    pheno_type = {}

    cats = pheno_cat.columns.tolist()[1:]
    for c in cats:
        pheno_type[c] = 'cat'
    strats = pheno_strat.columns.tolist()[1:]
    for s in strats:
        pheno_type[s] = 'strat'
    ints = pheno_int.columns.tolist()[1:]
    for i in ints:
        pheno_type[i] = 'int'
    conts = pheno_cont.columns.tolist()[1:]
    for co in conts:
        pheno_type[co] = 'cont'
    return pheno_type

def get_pheno_df(dftype, dfs):
    [pheno_cat, pheno_cont, pheno_int, pheno_strat] = dfs
    if dftype == 'cat':
        return 'stacked', pheno_cat
    elif dftype == 'strat':
        return 'stacked', pheno_strat
    elif dftype == 'int':
        return 'boxes', pheno_int
    else:
        return 'boxes', pheno_cont


# list of eids of movers and stayers: from a pob_por_migration dataframe (or derivative)
def get_stayers_movers(pob_por_migration_df, movement_threshold):
    stayers = pob_por_migration_df[pob_por_migration_df.distances < movement_threshold].eid.tolist()
    movers = pob_por_migration_df[pob_por_migration_df.distances >= movement_threshold].eid.tolist()
    return stayers, movers

# get dictionary of phenotype => codes
def get_pheno_codes():
    pheno_codes = {}
    currentPhenotype = ''
    phenoList = []
    with open('../05_pheno_codes/all_phenocodes.csv') as f:

        for line in f:

            line = line.rstrip()

            if line.startswith('#'): # a phenotype
                if len(currentPhenotype) > 0: # deal with old stuff
                    pheno_codes[currentPhenotype] = phenoList
                    phenoList = [] # reset phenotype code list
                currentPhenotype = line[1:]
            else:
                phenoList.append(line)

        # done going through lines
        pheno_codes[currentPhenotype] = phenoList
    return pheno_codes

# make a legend for a particular phenotype
def pheno_legend(phe,fig_size,text_size):
    f,ax = plt.subplots(figsize = fig_size, facecolor = 'w')
    pheno_codes = get_pheno_codes()
    codes = pheno_codes[phe]
    numCats = len(codes)
    cols = get_ncolors_from_cmap(numCats,colormap='gray')
    for i,c in enumerate(cols):
        codeText = codes[i]
        rect = Rectangle((1,i),0.4,0.8,facecolor=c,edgecolor='k')
        ax.text(1.5,i+0.3,codeText,fontsize = text_size)
        ax.add_patch(rect)
    ax.set_xlim([0,5])
    ax.set_ylim([-1,len(cols)])
    ax.set_axis_off()
    plt.tight_layout()
    return f

def get_summary_stats(stayer_mover_phenotypes_list):
    [stayer_phenotypes,mover_phenotypes] = stayer_mover_phenotypes_list
    n_stayers = len(stayer_phenotypes)
    n_movers =  len(mover_phenotypes)

    if n_stayers > 0 and n_movers > 0:
        mean_stayers = np.mean(stayer_phenotypes)
        mean_movers  = np.mean(mover_phenotypes)
        std_stayers = np.std(stayer_phenotypes)
        fracStd = (mean_movers - mean_stayers) / float(std_stayers)
        percDiff = (mean_movers - mean_stayers) / float(mean_stayers)
        percDiff = percDiff * 100

    else:
        mean_stayers = float('nan')
        mean_movers = float('nan')
        fracStd = float('nan')
        percDiff = float('nan')

    return mean_stayers, mean_movers, percDiff, fracStd, n_stayers, n_movers

def plot_phenos(m_to_plot, phe, pheno_dfs, grouplabels, showPlot = 'big'):

    [pheno_cat, pheno_cont, pheno_int, pheno_strat] = pheno_dfs
    pheno_type = get_pheno_type(pheno_dfs)
    plotType, df = get_pheno_df(pheno_type[phe], pheno_dfs)

    if plotType == 'stacked':

        if showPlot == 'big':
            fig_size = (8,8)
            axis_text_size = 16
            tick_text_size = 16
        else:
            fig_size = (3,2.57)
            axis_text_size = 8
            tick_text_size = 8

        # total number of categories
        n = np.unique(df[phe])
        n = n[~np.isnan(n)]
        n = n[np.where(n>=0)]
        numCats = len(n)

        plotVals = []
        cols = get_ncolors_from_cmap(numCats,colormap='inferno')

        fig, ax = plt.subplots(1,1,figsize=fig_size)

        for i,m in enumerate(m_to_plot):
            labels, counts = np.unique(m, return_counts=True)
            percentages = [x/np.sum(counts)*100 for x in counts]
            print(grouplabels[i],percentages)

            bot = 0
            for j,p in enumerate(percentages):
                ax.bar(i+1, p , align='center', bottom=bot, color=cols[j])
                bot = bot + p

        ax.set_ylabel('percentage of samples',fontsize=axis_text_size)

        print('numbers:\t','\t'.join(str(y) for y in [len(x) for x in m_to_plot]))

    else: # boxes

        if showPlot == 'big':
            fig_size = (5,8)
            axis_text_size = 16
            tick_text_size = 16
        else:
            fig_size = (2,2.57)
            axis_text_size = 8
            tick_text_size = 8

        fig, ax = plt.subplots(1,1,figsize=fig_size)

        ax.boxplot(m_to_plot,showfliers=False)
        #ax.boxplot(m_to_plot)
        #ax.violinplot(m_to_plot)

        ax.set_ylabel(phe)

        if len(m_to_plot[0]) > 0:
            print('numbers: ',' '.join(str(y) for y in [len(x) for x in m_to_plot]))
            print('means: ', '  '.join(['{:01.2f}'.format(y) for y in [np.mean(x) for x in m_to_plot]]))

    xlab_pos = np.arange(len(m_to_plot))+1
    plt.title(phe,fontsize=axis_text_size)
    plt.xticks(xlab_pos, grouplabels, rotation=90, fontsize = tick_text_size)
    plt.tight_layout()

    if plotType == 'stacked':
        pheno_legend(phe,fig_size,tick_text_size)

    return fig

# x and y are arrays of xcoords, ycoords
def get_centroid(x,y):
    return np.mean(x), np.mean(y)

# should also feed this eids (or feed it eids instead, and get coordinates from the eids.)
# want to return eids that are in clusters and outliers
def slidingWindowDisplacement(eidList, pobE, pobN, windowIncreaseFactor = 2, showPlot=False):

    x = [pobE[eid] for eid in eidList if eid in pobE.keys()]
    y = [pobN[eid] for eid in eidList if eid in pobE.keys()]

    centroid = get_centroid(x,y)
    distancesToCentroid = [math.hypot(val-centroid[0],y[i]-centroid[1]) for i, val in enumerate(x)]
    meanD = np.mean(distancesToCentroid)

    stepSize = np.floor(meanD + windowIncreaseFactor * np.std(distancesToCentroid))
    stepFraction = 0.5
    windowBuffer = stepFraction*stepSize

    lowerLeftCorner = (min(x)-windowBuffer,min(y)-windowBuffer)
    upperRightCorner = (max(x)+windowBuffer,max(y)+windowBuffer)
    xSpan = upperRightCorner[0] - lowerLeftCorner[0]
    ySpan = upperRightCorner[1] - lowerLeftCorner[1]

    if xSpan > 0 or ySpan > 0:
        numCols = int(np.ceil(xSpan/windowBuffer))
        numRows = int(np.ceil(ySpan/windowBuffer))
    else:
        numCols,numRows = 1,1

    pointCounts = np.zeros([numRows,numCols])

    maxCount = 0
    maxPol = (0,0)
    maxCentroid = (0,0)
    centroidIn = centroid
    centroidOut = centroid

    insideCluster = eidList
    outsideCluster = []

    for r in np.arange(numRows):
        for c in np.arange(numCols):
            pointList = []
            excludedPoints = []

            clusterEids = []
            outlierEids = []

            ll = (lowerLeftCorner[0] + c * windowBuffer, lowerLeftCorner[0] + r * windowBuffer)
            ur = (ll[0]+stepSize,ll[1] + stepSize)
            polygon = Polygon([ (ll[0],ll[1]), (ll[0],ur[1]), (ur[0],ur[1]), (ur[0],ll[1]) ])
            for i,xc in enumerate(x):
                point = Point(xc,y[i])
                if polygon.contains(point):
                    pointList.append((xc,y[i]))
                    clusterEids.append(eidList[i])
                else:
                    excludedPoints.append((xc,y[i]))
                    outlierEids.append(eidList[i])

            if len(pointList) > maxCount:
                maxCount = len(pointList)
                xPoints = np.array([p[0] for p in pointList])
                yPoints = np.array([p[1] for p in pointList])

                if len(excludedPoints) > 0 and len(pointList) >= 2*len(excludedPoints):
                    centroidIn = get_centroid(xPoints,yPoints)
                    xPoints = np.array([p[0] for p in excludedPoints])
                    yPoints = np.array([p[1] for p in excludedPoints])
                    centroidOut = get_centroid(xPoints,yPoints)
                    insideCluster = clusterEids
                    outsideCluster = outlierEids

                maxPol = (ll,ur)

    if showPlot == True:
        # setup figure
        fig=plt.figure(dpi= 80, facecolor='w', edgecolor='k')#, figsize=(5, 5))
        ax = plt.subplot(111)

        # plot all points
        ax.plot(x,y,'bo')

        # plot centroid of points
        ax.plot(centroid[0],centroid[1], 'go', markersize = 16)

        # if cluster & outlier(s) . . .
        if centroidIn != centroid:
            # plot line connecting centroids
            ax.plot((centroidIn[0],centroidOut[0]),(centroidIn[1],centroidOut[1]),'k-', linewidth=3)

            # plot centroid of points WITHIN max region
            ax.plot(centroidIn[0],centroidIn[1],'r*', markersize = 16)

            # plot centroid of points NOT WITHIN max region
            ax.plot(centroidOut[0],centroidOut[1],'m*', markersize = 16)

        # equal scale
        ax.set_aspect('equal')

        # buffer axes
        if lowerLeftCorner[0] != upperRightCorner[0]:
            ax.set_xlim([lowerLeftCorner[0],upperRightCorner[0]])
        if lowerLeftCorner[1] != upperRightCorner[1]:
            ax.set_ylim([lowerLeftCorner[1],upperRightCorner[1]])

        plt.show()

    return meanD, insideCluster, outsideCluster, centroidIn, centroidOut

def slidingWindows2qgis(eidList,POBE,POBN,windowIncreaseFactor=2,showPlot=False):

    foundOutlier = False
    angle = 0
    displacement = 0

#     xcoords = np.zeros(len(eidList))
#     ycoords = np.zeros(len(eidList))

#     for j, eid in enumerate(eidList):
#         if eid in POBE.keys():
#             xcoords[j] = POBE[eid]
#             ycoords[j] = POBN[eid]

    # slidingWindowDisplacement(x,y,windowIncreaseFactor = 2, showPlot=False)
    #centroidIn, centroidOut = slidingWindowDisplacement(xcoords,ycoords,windowIncreaseFactor,showPlot)
    j1,j2,j3,centroidIn, centroidOut = slidingWindowDisplacement(eidList, POBE, POBN, windowIncreaseFactor, showPlot)
    #return meanD, insideCluster, outsideCluster, centroidIn, centroidOut

    if centroidIn != centroidOut:
        foundOutlier = True
        angle = ukb.angle_from_vertical(centroidIn,centroidOut)
        displacement = dist_between_points(centroidIn,centroidOut)

    lineToPrint = ','.join([str(val) for val in [centroidIn[0], centroidIn[1], centroidOut[0], centroidOut[1], displacement, angle]])

    return foundOutlier, lineToPrint


def kindf_to_kindict(df):
    id1 = df.ID1.values.tolist()
    id2 = df.ID2.values.tolist()
    kinDict = {}

    for i,id in enumerate(id1):
        kin = id2[i]
        if id in kinDict.keys():
            kinDict[id].append(kin)
        else:
            kinDict[id] = [kin]

    for i,id in enumerate(id2):
        kin = id1[i]
        if id in kinDict.keys():
            kinDict[id].append(kin)
        else:
            kinDict[id] = [kin]

    return kinDict

# cumulative frequency histogram for distances
def cumulative_frequency_histogram_distance(toPlot,fig_size,labs,cols,maxX=500):
    f,a = plt.subplots(1,1,figsize=fig_size,dpi=150)
    xlab = 'Distance moved (km)'

    for i, d in enumerate(toPlot):
        d = d[~np.isnan(d)] / 1000
        n,b,p = a.hist(d, 1000, density=True, histtype='step', cumulative=True,
               color=cols[i], linewidth = 1, label = labs[i])
        p[0].set_xy(p[0].get_xy()[:-1])

    a.set_xticks(np.arange(0,maxX,100))
    a.legend(prop={'size': 8}, loc = 'lower right')
    a.set_xlim([-10,maxX])
    a.set_xlabel(xlab,fontsize=8)
    a.set_ylabel('Proportion of samples',fontsize=8)
    a.tick_params(axis='both', which='major', labelsize=8)
    a.plot([20,20],[0,1],'--r')
    return f


# make three dictionaries of distance from kinship:
# eid_kin[eid]=[list,of,kin]
# eid_kin_POBdist[eid]=[list,of,distances,in,same,order,as,list,of,kin]
# eid_kin_PORdist[eid]=[list,of,distances,in,same,order,as,list,of,kin]
def get_eid_kin(kindf):

  eid_kin = {}
  eid_kin_POBdist = {}
  eid_kin_PORdist = {}

  id1s = kindf.ID1.values.tolist()
  id2s = kindf.ID2.values.tolist()
  pobD = kindf.POB_distance.values.tolist()
  porD = kindf.POR_distance.values.tolist()

  # go through id1 list
  for i,id1 in enumerate(id1s):
      id2 = id2s[i]
      bd = pobD[i]
      rd = porD[i]
      if id1 in eid_kin.keys():
          eid_kin[id1].append(id2)
          eid_kin_POBdist[id1].append(bd)
          eid_kin_PORdist[id1].append(rd)
      else:
          eid_kin[id1]= [id2]
          eid_kin_POBdist[id1] = [bd]
          eid_kin_PORdist[id1] = [rd]

  # go through id2 list
  for i,id2 in enumerate(id2s):
      id1 = id1s[i]
      bd = pobD[i]
      rd = porD[i]
      if id2 in eid_kin.keys():
          eid_kin[id2].append(id1)
          eid_kin_POBdist[id2].append(bd)
          eid_kin_PORdist[id2].append(rd)
      else:
          eid_kin[id2]= [id1]
          eid_kin_POBdist[id2] = [bd]
          eid_kin_PORdist[id2] = [rd]

  return eid_kin, eid_kin_POBdist, eid_kin_PORdist

# function to get dictionary of weighted distances for eids (keys) in eid_kin dictionary
# eid_kin = dictionary eid=>[list,of,kin,for,that,eid] from get_eid_kin(kindf)
# weights_dict = dictionary nuts3=>sample weighting for that nuts3 region
# place_dict = dictionary eid=>nuts3 region of pob or por
# dist_dict = dictionary eid=>[list,of,pob,or,por,distances,for,each,kin,eid]
def get_weighted_distances(eid_kin,weights_dict,place_dict,dist_dict):

    dist_weighted = {}

    for eid in eid_kin.keys():

        # get list of kin and list of distances
        kins = eid_kin[eid]
        ds = dist_dict[eid]

        if eid in place_dict.keys(): # some eid's are in kinship but not eid_data

            # weight for this eid in this location
            weight1 = weights_dict[place_dict[eid]]

            # weights for kin in their locations
            # [f(x) for x in sequence if condition]
            kin_weights = [weights_dict[place_dict[k]] for k in kins if k in place_dict.keys()]

            # factors = weight1 * each weight in kin_weights
            factors = [weight1 * w for w in kin_weights]

            # convert each factor to an integer
            factors = [int(np.round(f)) for f in factors]

            # make factor instances of each distance
            tt = sum([f * [ds[i]] for i,f in enumerate(factors)], [])

            dist_weighted[eid] = tt

    return dist_weighted

# specify nuts colors
def get_nuts1_colors():
    nuts1 = ['UKC', 'UKD', 'UKE', 'UKF', 'UKG', 'UKH', 'UKI', 'UKJ', 'UKK', 'UKL', 'UKM', 'UKN']
    nuts1_color_dict = {}
    nuts1_color_list = []
    f = open('../02_maps_qgis/12cols.txt','r')
    needCols = True
    i = 0
    for line in f:
        line = line.rstrip()
        if line.startswith('#'):
            needCols = True
        elif needCols == True:
            rgb = [int(x) for x in line.split()[:3]]
            rgb = [x/255 for x in rgb]
            nuts1_color_dict[nuts1[i]] = rgb
            nuts1_color_list.append(rgb)
            i += 1
    f.close()
    nuts1_color_dict = dict(zip(nuts1,nuts1_color_list))
    return nuts1,nuts1_color_list, nuts1_color_dict
