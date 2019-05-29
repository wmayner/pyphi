
        #The state(s) at which divergence was maximal
        sar = np.argwhere(flatten(divs)==np.max(divs))
        maxIndsL.append([x[0] for x in sar])
    #Across partitions, the same state might be maximal - we don't care about these ties
    #So we just take the unique indices and return this list of uniques
    maxInds = list(set(x for l in maxIndsL for x in l))