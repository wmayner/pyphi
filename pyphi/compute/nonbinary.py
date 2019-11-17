
import numpy as np
from .parallel import MapReduce
from ..models import CauseEffectStructure
from ..models import cuts
from itertools import combinations
import functools
from .. import config
from ..partition import directed_bipartition
from ..partition import bipartition
from ..subsystem_nb import Subsystem_nb
from .distance import ces_distance

class ComputeCauseEffectStructure(MapReduce):#avoid
    """Engine for computing a |CauseEffectStructure|."""
    # pylint: disable=unused-argument,arguments-differ

    description = 'Computing concepts'

    @property
    def subsystem(self):
        return self.context[0]

    def empty_result(self, *args):
        return []

    @staticmethod
    def compute(mechanism, subsystem, purviews, cause_purviews,
                effect_purviews):
        """Compute a |Concept| for a mechanism, in this |Subsystem| with the
        provided purviews.
        """
        concept = subsystem.concept(mechanism,
                                    purviews=purviews,
                                    cause_purviews=cause_purviews,
                                    effect_purviews=effect_purviews)

        concept.subsystem = None
        return concept

    def process_result(self, new_concept, concepts):
        """Save all concepts with non-zero |small_phi| to the
        |CauseEffectStructure|.
        """
        if new_concept.phi > 0:
            # Replace the subsystem
            new_concept.subsystem = self.subsystem
            concepts.append(new_concept)
        return concepts
    

def ces_nb(subsystem, mechanisms=False, purviews=False, cause_purviews=False,
        effect_purviews=False, parallel=False):
   
    if mechanisms is False:
        mechanisms=[]
        mechanism=tuple(subsystem._label2index(subsystem.nodes))
        for i in  range(1,len(mechanism)+1):
            mechanisms=mechanisms+sorted(combinations(mechanism, i)) 
        

    if subsystem.base.count(2)==len(subsystem.base):
        sb=subsystem.dummy_subsys
    else:
        sb=subsystem
    
    engine = ComputeCauseEffectStructure(mechanisms, sb, purviews,
                                         cause_purviews, effect_purviews)
    
   
    return CauseEffectStructure(engine.run(parallel or  
                                           config.PARALLEL_CONCEPT_EVALUATION),
                                subsystem=sb) #avoid

def format_sia(obj):
    return 'System irreducibility analysis: Φ = {self.phi} \n\
{self.cut}\n{self.subsystem}\n{self.ces}'.format(self=obj)

class SystemIrreducibilityAnalysis():#avoid
    def __init__(self, ces, partitioned_ces, cut, phi, subsystem):        

        self.subsystem=subsystem
        self.ces=ces
        self.partitioned_ces=partitioned_ces
        #self.cut=cut
        self.phi=phi
        self.cut = cuts.Cut(tuple(cut[0]), tuple(cut[1]))
    
    def ces(self):
        return self.subsystem
    def ces(self):
        return self.ces
    
    def partitioned_ces(self):
        return self.partitioned_ces
    
    def cut(self):
        #create a cut 
        return cut
    
    def phi(self):
        return phi
    def __str__(self):
        return format_sia(self)
    
    def __repr__(self):
        return format_sia(self)
    
    def __gt__(self, other):
        return self.phi > other.phi

def tensor(a,b):

    return functools.reduce(lambda a,b: np.concatenate((a,b),axis=1), 
                        [np.transpose(np.multiply(np.transpose(a),b[:,c])) for c in range(b.shape[-1])] ) 
def tpm_cut(subsystem,cut1,cut2):
    
    v=subsystem.node_tpm_expanded(subsystem.node_tpm(cut1,cut2))
    v=[x.values for x in v]
    return functools.reduce(lambda x,y: tensor(x,y),v)

def cut_ces(sb):

    cuts=directed_bipartition(sb.nodes, nontrivial=True)
    cuts=[[list(x[0]), list(x[1])] for x in cuts]
        
    ans=[]
    for cut in cuts:
        cut_tpm=tpm_cut(sb,cut[0],cut[1])
        sb.apply_cut(cut_tpm)
        ans+=[ces_nb(sb)]
        sb.remove_cut()
    
    return ans, cuts
    
def compare_ces(subsystem,uncut_ces, cut_ces):
    
    if subsystem.base.count(2)==len(subsystem.base): #use EMD-based extended distance from pyphi
        return ces_distance(uncut_ces,cut_ces)
    else: #USE_SMALL_PHI_DIFFERENCE_FOR_CES_DISTANCE for non binary systems
        return sum(c.phi for c in uncut_ces.concepts) - sum(c.phi for c in cut_ces.concepts)

def compute_sia(subsystem):
    
    all_cut_ces, cuts=cut_ces(subsystem)
    uncut_ces=ces_nb(subsystem)
    
    return uncut_ces, [ [ x,compare_ces(subsystem,uncut_ces, x)] for x in all_cut_ces], cuts

def sia_nb(subsystem):
    ces, all_partitioned_ces, cuts=compute_sia(subsystem)
    phis=[x[1] for x in all_partitioned_ces]
    Φ=min(phis)
    Φ_id= phis.index(Φ)
    
    return SystemIrreducibilityAnalysis(ces,all_partitioned_ces[Φ_id][0],cuts[Φ_id], Φ, subsystem)

def calculate_all_sia(network,state,nodes):
    
    if len(nodes)<2:
        return 
    try:
        subsystem=Subsystem_nb(network,state,list(nodes))
        return sia_nb(subsystem)        
    except:
        return


def major_complex_nb(network, state):
    network.reachable(state)
    
    a=[]
    for p in bipartition(network.node_labels):
        a+=[calculate_all_sia(network,state,p[0])] + [calculate_all_sia(network,state,p[1])]
        
    return max(list(filter(None, a)))

	