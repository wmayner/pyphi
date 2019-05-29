import pyphi
from pyphi.distribution import flatten, marginal_zero
from functools import reduce
import numpy as np
from pyphi.partition import Tripartition, Part, mip_partitions
past, future = pyphi.Direction.CAUSE, pyphi.Direction.EFFECT


#This function returns an index (or indices) to the 'Maximally Irreducible State' of a cause/effect
#The index is re a flattened array
def getMaxState(mip,subsystem,dispoutput = False):
    eps = np.finfo(float).eps
    #maxIndsL will be a list of lists
    maxIndsL = []
    #For each partition that yielded MinPhi for the maximally-irreducible purview:
    for partition in mip._partlist:
        p = mip.repertoire + eps
        q = subsystem.partitioned_repertoire(mip.direction,partition) + eps
        #divergences at each state in mip's repertoire
        divs = np.round(np.abs(p*np.nan_to_num(np.log(p/q))),pyphi.config.PRECISION)

        #The state(s) at which divergence was maximal
        maxIndsL.append(np.argwhere(flatten(divs)==np.max(divs)))
    #Across partitions, the same state might be maximal - we don't care about these ties
    #So we just take the unique indices and return this list of uniques
    maxInds = np.unique(maxIndsL)

    ####Printing result
    if dispovtput:
        for partition in mip._partlist:
            p = mip.repertoire + eps
            q = subsystem.partitioned_repertoire(mip.direction,partition) + eps

            divs = np.round(np.abs(p*np.nan_to_num(np.log(p/q))),pyphi.config.PRECISION)

            Ms = np.argwhere(divs== np.amax(divs))
            s = p.shape
            purview = [x for x in range(0,len(s)) if s[x]==2]
            for M in Ms:
                pslabs = [str(M[purview[n]]) for n in range(0,len(purview))]
                plab = ''.join([subsystem.node_labels.labels[ind] for ind in purview])

                print('**')
                print('     phi(' + str(np.round(np.max(divs),pyphi.config.PRECISION)) + ')' + plab + ':' + ''.join(pslabs))
    ###
    return maxInds

def getAllRelations(subsystem,const,dispoutput = False):
    #This function takes a list of CONCEPTS (const) and returns the set of all relations, at all orders,
    #between all the concepts' causes and effects.
    #Can treat as a usage example.
    #Beware: combinatorial eternity is at your fingers.
    ####
    CElist = []
    for n in range(0,len(const)):
        CElist.append(const[n].cause)
        CElist.append(const[n].effect)

    pset = list(pyphi.utils.powerset(range(0,len(CElist))))

    rps = []
    for x in range(1,len(pset)):
        if len(pset[x])>1:
            print('working on rel ' + str(x))
            clist = [CElist[n] for n in pset[x]]

            rphi, rpurview = getRelationExt(subsystem,clist,dispoutput)
            if np.round(rphi,6)>0:
                rps.append([pset[x],rphi,rpurview])
    # HERE: find out what the relata are
    return rps

#Function for finding a relation
def getRelationExt(subsystem,clist, dispoutput = False):
    eps = np.finfo(float).eps

    if len(clist)==1:
        if dispoutput:
            print('\n-------\n-------')
            print('Singletons do not relate')
        phiMax = -1
        rpurview = ()
        return phiMax, rpurview

    #List of lists of states for each cause/effect
    mosIndices = []
    for C in clist:
        #Get the 'state' or 'states' for each mip
        mosIndices.append(getMaxState(C,subsystem,dispoutput=False))

    purvlist = []
    for x in range(0,len(clist)):
        #all purviews
        purvlist.append(clist[x].purview)

    #get the powerset of intersection of purviews, as candidate relation purviews
    #Any candidate purview with elements that are not in one of the cause/effect purviews
    #will be reducible (phi=0), so we only consider subsets of the purview intersection.
    subpurvs = list(pyphi.utils.powerset(reduce(np.intersect1d,purvlist)))
    candidate_jpurviews = [subpurvs[x] for x in range(1,len(subpurvs))]

    if candidate_jpurviews==[]:
        if dispoutput:
            print('\n-------\n-------')
            print('\nConstraints:')
            for x in range(0,len(clist)):
                print('M:' + str(clist[x].mechanism))
                print(str(clist[x].direction) + ' P:' + str(clist[x].purview))
            print('\n')
            print('Empty purview intersection (no relation)')
        phiMax = 0
        rpurview = ()
        return phiMax, rpurview

    # This is a tuple of mechanism meta-labels, e.g. (0,1,2) for the 1st, 2nd,
    # and 3rd involved mechanisms
    metamechanism = tuple([x for x in range(0,len(clist))])

    # List of the MIP phis for each candidate joint purview (jpurview)
    phiMIP_over_jpurviews = []
    for candidate_jpurview in candidate_jpurviews:

        # For each involved cause/effect, get the relevant mechanism's
        # repertoire only over the candidate purview
        # Make a list of these repertoires
        j_ps = []
        for C in clist:
            rep = subsystem.repertoire(C.direction,C.mechanism,candidate_jpurview)
            j_ps.append(eps+rep)

        # for all cuts
        phi_over_partitions = []
        # for the list of partitions, we invert the wedge cut with relationsV
        for metapartition in mip_partitions(metamechanism,candidate_jpurview,relationsV=True):

            # For each involved cause/effect, project the metapartition over
            # the relevant mechanism and the candidate purview
            # Make a list of these partitioned repertoires
            j_qs = []
            for M in metamechanism:
                # Need to create a partition for each mip; this is the
                # intersection of each mip with metapartition, Just remember
                # that the .mechanism part of metapartition is not elements but
                # mechanisms (0th, 1th, etc)
                partlist = []
                # For each 'metapart' in the metapartition, translate the
                # metapart into a part for the Mth cause/effect
                for p in [0,1,2]:
                    mint = tuple(np.intersect1d(metapartition[p].mechanism,
                                                tuple([M])))
                    pint = tuple(np.intersect1d(metapartition[p].purview,
                                                candidate_jpurview))
                    # if mpartition[p] includes the xth mechanism
                    if len(mint) > 0:
                        partlist.append(Part(clist[M].mechanism, pint))
                    else:
                        partlist.append(Part((), pint))

                partition = Tripartition(partlist[0],
                                         partlist[1],
                                         partlist[2])

                prep = subsystem.partitioned_repertoire(clist[M].direction,
                                                        partition)

                j_qs.append(eps+prep)

            # This is a list of 'divergence repertoires' (p log p/q) for each
            # involved cause/effect, using the meta-partitioned repertoires.
            # In a sense this method assumes the use of KLM(whatever)
            # divergence as a phi measure - the method is Not Defined for other
            # divergences (EMD, KLD, etc) - maybe should assert
            # config.MEASURE='KLM' at the outset
            divslist = []
            for M in metamechanism:
                divslist.append(abs(j_ps[M]*np.nan_to_num(np.log(j_ps[M]/j_qs[M]))))

            # The candidate_jpurview divergence product is taken over all C/Es
            # - this is the repertoire of "joint divergences"
            joint_divergence = np.prod(np.array(divslist),axis=0)

            # Each cause/effect 'sees' the joint divergence from the
            # perspective of its own state (states if there is a state tie) We
            # will list these cause/effect-wise divergences:
            CEwise_divs = np.zeros(len(clist))
            for M in metamechanism:
                # Expand the divergence product from candidate_jpurview to
                # clist[x]'s purview
                non_candidate_jpurview_indices = tuple(
                    set(clist[M].purview) - set(candidate_jpurview)
                )
                umat = subsystem.unconstrained_repertoire(
                    clist[M].direction,
                    non_candidate_jpurview_indices
                )
                # The joint divergence is "spread out" over the possible states
                expanded_joint_div = flatten(joint_divergence*umat)

                mIs = mosIndices[M]
                divergenceAtState = []
                # Here we consider the possibility that a cause/effect has a
                # 'state tie'

                # Record the joint divergence at each tied state
                for mI in mIs:
                    divergenceAtState.append(np.abs(expanded_joint_div[mI]))
                # The 'CEwise' joint divergence is the maximum over
                # divergenceAtState
                # Thus a state tie may be (but is not necessarily) resolved as
                # that state that is maximally irreducible
                # It might be nice to record the 'max state' but don't bother
                # for now (can handle that externally)
                CEwise_divs[M] = max(divergenceAtState)
            # The irreducibility over the 'metapartition' is given by the
            # cause/effect whose state is least divergent
            minphi = np.round(min(CEwise_divs),6)
            phi_over_partitions.append(minphi) # Should use the config precision here
            if(minphi==0):
                # If a purview is reducible, stop checking partitions
                break
        # phi MIP for
        phiMIP_over_jpurviews.append(min(phi_over_partitions))
        # TODO:HERE

    # The phi and the purview of the relation:
    phiMax = max(phiMIP_over_jpurviews)
    rpurview = [
        candidate_jpurviews[x] for x in range(0,len(candidate_jpurviews))
        if phiMIP_over_jpurviews[x]==phiMax
    ]

    ###
    if dispoutput:
        print('\n-------\n-------')
        print('\nConstraints:')
        for x in range(0,len(clist)):
            print('M:' + str(clist[x].mechanism))
            print(str(clist[x].direction) + ' P:' + str(clist[x].purview))
        print('\n')

        print('distance:' + str(pyphi.config.MEASURE))
        for x in range(0,len(candidate_jpurviews)):
            if phiMIP_over_jpurviews[x]==phiMax:
                print('***' + str(candidate_jpurviews[x]) + ':' + '{:1.5f}'.format(phiMIP_over_jpurviews[x]))
            else:
                print('   ' + str(candidate_jpurviews[x]) + ':' + '{:1.5f}'.format(phiMIP_over_jpurviews[x]))

        print('\n******************')
        print('\nPhi: ' + str(phiMax))
        if phiMax>0:
            print('purview:' + str(rpurview[0]))


    return phiMax, rpurview
