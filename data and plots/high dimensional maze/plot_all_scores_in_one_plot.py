import numpy as np
import joblib
import matplotlib.pyplot as plt
import argparse
import glob 
import statistics
import math
from collections import defaultdict
# import colorsys
# from colour import Color


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


class Dataline(object):
    pass

if __name__ == "__main__":
    #example usage: $ python3 plot_all_scores_in_one_plot.py -i tempscores -o testcolors -g lr
    parser = argparse.ArgumentParser()
    #e.g. dump file name: test004_discount0.9_seed22347_lr0.0002_meta_hard.jldump
    parser.add_argument('-i', '--input', default="/scores/", help="the dumpfile names should contain the name of the input and it should start with 'test'")
    parser.add_argument('-o', '--output', default="allscores")
    parser.add_argument('-g', '--groupby', default="name")
    parser.add_argument('-r', '--reverse', default=0)
    parser.add_argument('-s', '--stderror', default=0)
    args = parser.parse_args()
    basename = args.input
    outputplot = args.output
    groupby = args.groupby
    reverse = args.reverse
    stderror = args.stderror
    
    print("stderror:" + stderror)
    
    alphas=0.5
    if stderror=="1":
        alphas=0.3

    if groupby == "name":
        outputplot += "_grouped_by_name"
    elif groupby == "lr":
        outputplot += "_grouped_by_lr"
    elif groupby == "d":
        outputplot += "_grouped_by_discount"
    elif groupby == "seed":
        outputplot += "_grouped_by_seed"

    scoresFiles = glob.glob("./" + basename + "/*.jldump")
    print("found score dumps: ", scoresFiles)

    uniqueinputnames = []
    sampleSizePerGroup = [0]*30 #max 30 categories
    colors = ['b','g','r','c','m','y','k','w']

    datalines = []
    for scorefile in scoresFiles:
        dataline = Dataline()
        dataline.scores = joblib.load(scorefile)['vs']

        scorefile = scorefile[scorefile.rfind("/"):-1] #filter directory stuff
        inputname = scorefile[scorefile.find("test"):scorefile.find("_")]
        dataline.inputname = inputname

        ind = scorefile.find("discount")
        if ind != -1:
            dataline.d = "d" + scorefile[ind+8:scorefile.find("_",ind)]

        # dataline.lr = scorefile[scorefile.find("lr"):scorefile.find("_",scorefile.find("lr"))]
        dataline.lr = scorefile[scorefile.find("lr")+2:scorefile.find("_",scorefile.find("lr"))]
        dataline.seed = scorefile[scorefile.find("seed"):scorefile.find("_",scorefile.find("seed"))]

        ind = scorefile.find("meta")
        if ind != -1:
            dataline.meta ="meta"
        ind = scorefile.find("hard")
        if ind != -1:
            dataline.difficulty ="hard"
        datalines.append(dataline)

        #group by name/lr/..:
        if groupby == "name":
            key = dataline.inputname
        elif groupby == "lr":
            key = dataline.lr
        elif groupby == "d":
            key = dataline.d
        elif groupby == "seed":
            key = dataline.seed
        if key not in uniqueinputnames:
            uniqueinputnames.append(key) #remember new group name if file got a different key

        #count sample size per group
        if sampleSizePerGroup[uniqueinputnames.index(key)] != 0:
            sampleSizePerGroup[uniqueinputnames.index(key)] += 1
        else:
            sampleSizePerGroup[uniqueinputnames.index(key)] = 1
             
    if groupby == "lr":
        # uniqueinputnames = uniqueinputnames[2:-1]
        uniqueinputnames = list(map(float,uniqueinputnames))
        uniqueinputnames.sort()
        print(uniqueinputnames)
        uniqueinputnames = list(map(str,uniqueinputnames))
        # red = Color("red")
        # colors = list(red.range_to(Color("green"),6))

        c1='b' #blue
        c2='r' #green
        colors = [colorFader(c1,c2,1), colorFader(c1,c2,0.8), colorFader(c1,c2,0.6),colorFader(c1,c2,0.4), colorFader(c1,c2,0.2), colorFader(c1,c2,0)]
        # colors = ['#FF0000','#BB0000','#770000','#220000','#000022','#000077','#0000BB','#0000FF']
        # colors = [colorsys.hsv_to_rgb(1,1,1),colorsys.hsv_to_rgb(0.92,1,1),colorsys.hsv_to_rgb(0.86,1,1),colorsys.hsv_to_rgb(0.82,1,1),colorsys.hsv_to_rgb(0.76,1,1),colorsys.hsv_to_rgb(0.7,1,1)]
        # colors = ['b','g','r','c','m','y','k','w']



    if groupby == "name":
        uniqueinputnames.sort()
        if reverse=="1":
            uniqueinputnames.sort(reverse=True)
    
                
    #grouping
    groups = []
    for uniquename in uniqueinputnames:
        groups.append([])
    for dataline in datalines:
        if groupby == "name":
            key = dataline.inputname
        elif groupby == "lr":
            key = dataline.lr
        elif groupby == "d":
            key = dataline.d
        elif groupby == "seed":
            key = dataline.seed
        ind = uniqueinputnames.index(key)
        groups[ind].append(dataline.scores)

    print("Grouped into: ", uniqueinputnames)


    

    #calculate mean and std for each timepoint for each group
    groupsDatapoints = []
    groupsDatapointsStd = []
    for group in groups:
        group.sort(key=lambda x: len(x), reverse=True) #needed bcz some samples have smaller epoch counts
        listGroupDatapoints = []
        listGroupDatapointsStd = []
        epochrange=100
        # epochrange=len(group[0])
        for timepoint in range(0, epochrange):
            datapointsForTimepoint = []
            for dataline in group:
                if timepoint < len(dataline):
                    datapointsForTimepoint.append(dataline[timepoint])
            if len(datapointsForTimepoint) > 1:
                std = statistics.stdev(datapointsForTimepoint)
                if stderror == "1":
                    std= std/math.sqrt(6)
                avg = statistics.mean(datapointsForTimepoint)
                listGroupDatapoints.append(avg)
                listGroupDatapointsStd.append(std)
            else:
                # print("EACH GROUP ONLY HAS ONE SAMPLE")
                listGroupDatapoints.append(datapointsForTimepoint[0])
                listGroupDatapointsStd.append(0)
            # print("avg= ", avg, ", std= ", std)
        groupsDatapoints.append(listGroupDatapoints)
        groupsDatapointsStd.append(listGroupDatapointsStd)
        

    #plot
    for i in range(0, len(groupsDatapoints)):
        plt.plot(range(1, len(groupsDatapoints[i])+1), groupsDatapoints[i], label=uniqueinputnames[i] + ", n=" + str(sampleSizePerGroup[i]), color=colors[i])
        plusSigma = np.asarray(groupsDatapoints[i])+np.asarray(groupsDatapointsStd[i])
        minusSigma = np.asarray(groupsDatapoints[i])-np.asarray(groupsDatapointsStd[i])
        plt.fill_between(range(1, len(groupsDatapoints[i])+1), plusSigma, minusSigma, facecolor=colors[i], alpha=alphas) #alpha0.3
    plt.legend()
    plt.xlabel("Number of epochs")
    plt.ylabel("Score")
    plt.savefig(outputplot + ".pdf")
    plt.show()
    plt.close()


# for vraints use opactiy 0.5, and reverse = True
