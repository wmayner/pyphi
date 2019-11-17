import numpy as np
import os
import logging
from itertools import product
from itertools import combinations
import pandas as pd
import functools
from pyemd import emd
from scipy.spatial.distance import cdist
from scipy.stats import entropy

from .partition import mip_partitions
from . import Direction
from .distribution import flatten
from .labels import NodeLabels
from .convert import (to_2dimensional,state_by_state2state_by_node)
from .network import Network
from .subsystem import Subsystem
from .models import (Concept, MaximallyIrreducibleCause,
                     MaximallyIrreducibleEffect, NullCut,
                     RepertoireIrreducibilityAnalysis, _null_ria)

log = logging.getLogger(__name__)


class Subsystem_nb:

    def __init__(self, network, state, nodes, cm=None):
    	
    	self.network=network

    	self.nodes=self._validate_nodes(nodes)

    	self.state=self._validate_state(state)

    	self.network.reachable(state)

    	self.base=self._validate_base()

    	self.cm=self._validate_cm(cm)

    	self.net_state=list(state)

    	condition=sorted(list(set(self.network.node_labels)-set(self.nodes)))

    	self.tpm=self.network.conditioned_tpm(self.net_state, condition)

    	self.states_node=[list(range(b)) for b in self.base]

    	self.sates_all_nodes=[list(x[::-1])  for x in  list(product(*self.states_node[::-1])) ]
    	self.sates_all_nodes=np.transpose(self.sates_all_nodes).tolist()

    	self.tpm=self._tpm2df(self.tpm)

    	self.indexes=dict(zip(self.nodes, [x for x in range(len(self.nodes))] ))

    	self.node_labels=NodeLabels(self.nodes,list(range(len(self.nodes))))

    	root=os.path.abspath(os.path.dirname(__file__))

    	path=os.path.join(root, 'data', 'distance_matrices.npy')

    	self.distance_matrices=np.load(path,allow_pickle=True)

    	self.dummy_subsys=self._dummy()

    	self.cut_indices=tuple(self._label2index(self.nodes))

    	self.node_indices=tuple(self._label2index(self.nodes))

    def _validate_base(self):
        
        return [self.network.base[self.network.node_labels.index(n)]    for n in self.nodes]
        
     
    def _validate_cm(self,cm):
        
        condition=sorted(list(set(self.network.node_labels)-set(self.nodes)))
        if cm is None:
            cm=self.network.conditioned_cm( condition )
        if not np.array_equal(cm,self.network.conditioned_cm(condition)):
            
            raise NameError("cm does not match network's")
        
        return cm
    
    def _validate_state(self,state):
        
        if type(state)!=type(tuple()) and type(state)!=type([]):
            raise NameError('Sate is expected to be a tuple or list')            
        state=list(state)   
        self.network._validate_state(state)
        
        state=[state[self.network.node_labels.index(n)]    for n in self.nodes]
                
        return dict(zip(self.nodes, state))
    
    def _validate_nodes(self,nodes):
        
        if type(nodes)!=type(tuple()) and type(nodes)!=type([]):
            raise NameError('Nodes are expected to be a tuple or list')            
        nodes=list(nodes)       
        
        for n in nodes: 
            if n not in self.network.node_labels:
                raise NameError('Nodes foreign to the network' )
        if len(nodes)>len(self.network.node_labels):
                raise NameError('Exceeding number of nodes' )
        
        return nodes
    
    
    def _dummy(self):
        
        if self.base.count(2)==len(self.base):
            tpm_dummy=to_2dimensional(state_by_state2state_by_node(self.tpm.values))
            network = Network(tpm_dummy, node_labels=self.nodes)
            nodes = tuple(self.nodes)
            st=list(self.state.values())
            subsystem= Subsystem(network, tuple(st), nodes)        
            return subsystem
        
        network = Network(np.zeros(( np.power(2,len(self.nodes)) ,  len(self.nodes) ))   , 
                                  np.zeros(( len(self.nodes) , len(self.nodes)))   , node_labels=self.nodes)
        nodes = tuple(self.nodes)
        state = tuple([0]*len(self.nodes))
        subsystem = Subsystem(network, state, nodes)
        return subsystem
    
    def _tpm2df(self, tpm):

        index = pd.MultiIndex.from_arrays(self.sates_all_nodes, names=self.nodes)
        columns = pd.MultiIndex.from_arrays(self.sates_all_nodes, names=self.nodes)
        
        return pd.DataFrame(tpm,columns=columns, index=index)
    
    def _label2index(self,nodes):
             
        return [self.indexes[x] for x in nodes] if type(nodes)!=type(0) else self.indexes[nodes]
    
    def _index2label(self,nodes):
        
        return [self.nodes[x] for x in nodes] if type(nodes)!=type(0) else self.nodes[nodes]
    
    def _shape(self, purview):
        
        purview=self._label2index(purview)
        return [self.base[i] if i in purview else 1 for i in range(len(self.nodes))] 
    
    def _factor(self,set_of_nodes):
        
        return np.power(self.tpm.shape[0],1/len(self.nodes))**(len(self.nodes)-len(set_of_nodes))
    
  
    def _validate_inputs(self,units):
        
        if type(units)==type(0):
            return tuple()
        elif len(units)==0:
            return tuple()
        else:
            return tuple(self._index2label(units))

    @functools.lru_cache(maxsize=None)
    def _single_node_effect_repertoire(self,mechanism,purview):
        
        factor=self._factor(mechanism)  
       
        tpm=(self.tpm.transpose().groupby(list(purview)).sum()).transpose() 
        
        if len(mechanism)>0:
            tpm=(tpm.groupby(list(mechanism)).sum())*(1/factor)
            row=[self.state[i] for i in mechanism]
            return list(tpm.loc[tuple(row), :]) if len(mechanism)>1  else list(tpm.loc[row[0], :])
        else:
            return list(tpm.sum(axis=0)/tpm.shape[0])
    
    @functools.lru_cache(maxsize=None)   
    def effect_repertoire(self, mechanism, purview):
        
        purview=self._validate_inputs(purview)
        mechanism=self._validate_inputs(mechanism)

     
        if not purview: return np.array([1])        
       
        joint=np.ones(self._shape(purview))

        return joint *functools.reduce(np.multiply,  
                                       [np.array(
                                           self._single_node_effect_repertoire(mechanism,tuple(p))).reshape(self._shape([p])) 
                                        for p in purview]   ) 
    
    @functools.lru_cache(maxsize=None)
    def _single_node_cause_repertoire(self,mechanism,purview):
    
        factor=self._factor(purview)           
        factor2=np.prod([self.base[i] for i in self._label2index(purview)])


        #group in revrse [::-1] just to preserve paper's notation
        #i.e. the fisrt variable varies faster [A, B] A: 0101 B:0011(slower)
        tpm=(self.tpm.groupby(list(purview[::-1])).sum())*(1/factor)

        if len(mechanism)>0:
            tpm=(tpm.transpose().groupby(list(mechanism)).sum()).transpose()
            col=[self.state[i] for i in mechanism]
            return list(tpm.loc[:, col[0]]/sum(list(tpm.loc[:, col[0]])))
        else:
            #the unconstraind is just the normal distribution (equal probability for all)
            return [1/factor2]*int(factor2)

    @functools.lru_cache(maxsize=None)   
    def cause_repertoire(self, mechanism, purview):
        
        purview=self._validate_inputs(purview)
        mechanism=self._validate_inputs(mechanism)
     
        if not purview: return np.array([1])
        
        if not mechanism: return np.array(self._single_node_cause_repertoire((),purview))\
        .reshape(self._shape(purview),order='F')
       
        cause=functools.reduce(np.multiply, [ np.array(self._single_node_cause_repertoire(tuple(m),purview)) 
                                             for m in mechanism])
        
        
        cause=cause.reshape(self._shape(purview),order='F')
        return cause/cause.sum() if cause.sum()!=0  else cause
    
    def _null_ria(self,direction, mechanism, purview, repertoire=None, phi=0.0):
        return RepertoireIrreducibilityAnalysis(  #AVOID PYPHINB!!!!
            direction=direction,
            mechanism=mechanism,
            purview=purview,
            partition=None,
            repertoire=repertoire,
            partitioned_repertoire=None,
            phi=phi,
            net_labels=self.network.node_labels,
            net_bases=self.network.states_node
        )
    
    def repertoire_distance(self,a,b):
        
        a=flatten(a)
        b=flatten(b)
       
        if self.base.count(2)==len(self.base): #dist_type=='emd'        
            N=len(a)
            if N>9:
                states=[list(i[::-1])  for i in     list(product([0, 1], repeat=N))     ]
                distance_matrix=cdist(states, states, 'hamming')*N
                return emd(a, b, distance_matrix)
            else:
                return emd(a, b, self.distance_matrices[N-1])
        else: #dist_type=='kld':
            return entropy(a,b)
        
    def repertoire(self,direction, mechanism, purview):
        
        if direction == Direction.CAUSE: #avoid!!!
            return self.cause_repertoire(mechanism, purview)
        elif direction == Direction.EFFECT:#avoid!!!
            return self.effect_repertoire(mechanism, purview)

    
    def partitioned_repertoire(self, direction,partition):
        repertoires = [
            self.repertoire(direction, part.mechanism, part.purview)
            for part in partition
        ]
        return functools.reduce(np.multiply, repertoires)
    
    def evaluate_partition(self,direction, mechanism, purview, partition, repertoire):
        
        if repertoire is None:
            repertoire = self.repertoire(direction, mechanism, purview)

        partitioned_repertoire = self.partitioned_repertoire(direction,
                                                                 partition)
        
        phi = self.repertoire_distance(repertoire, partitioned_repertoire)

        return (phi, partitioned_repertoire) 
        
    def find_mip(self,direction,mechanism,purview):
        
        if not purview:
            return self._null_ria(direction, mechanism, purview)

        repertoire = self.repertoire(direction, mechanism, purview)

        def _mip(phi, partition, partitioned_repertoire):

            return RepertoireIrreducibilityAnalysis( #AVOID PYPHINB!!!!
            phi=phi,
            direction=direction,
            mechanism=mechanism,
            purview=purview,
            partition=partition,
            repertoire=repertoire,
            partitioned_repertoire=partitioned_repertoire,
            node_labels=self.node_labels,
            net_labels=self.network.node_labels,
            net_bases=self.network.states_node
            )
        
        
        if (direction == Direction.CAUSE and #avoid
                np.all(repertoire == 0)):
            return _mip(0, None, None)

        mip = self._null_ria(direction, mechanism, purview, phi=float('inf'))

        for partition in mip_partitions(mechanism, purview, self.node_labels):#AVOID PYPHINB!!!!
            phi, partitioned_repertoire = self.evaluate_partition(
                direction, mechanism, purview, partition,
                repertoire=repertoire)

            phi=np.round(phi,6)
            if phi < 0e-6:
                return _mip(0.0, partition, partitioned_repertoire)

            if phi < mip.phi:
                mip = _mip(phi, partition, partitioned_repertoire)

        return mip

    
    def cause_mip(self,mechanism,purview):
        
        
        return self.find_mip(Direction.CAUSE,mechanism,purview) #avoid!!!
        
    def effect_mip(self,mechanism,purview):
        

        return self.find_mip(Direction.EFFECT,mechanism,purview) #avoid
    
    @functools.lru_cache(maxsize=None)   
    def find_mice(self, direction, mechanism, purview=False):

        
        if purview==False:
            purview=tuple(self._label2index(self.nodes))
                        
        purviews=[]
        for i in  range(1,len(purview)+1):
            purviews=purviews+sorted(combinations(purview, i)) 
    

        if not purviews:
            max_mip = self._null_ria(direction, mechanism, ())
        else:
            max_mip = max(self.find_mip(direction, mechanism, purview)
                          for purview in purviews)

            
        if direction == Direction.CAUSE:#avoid!!
            
            return MaximallyIrreducibleCause(max_mip)
        elif direction == Direction.EFFECT:#avoid!!!
            
            return MaximallyIrreducibleEffect(max_mip)
        

    def mic(self, mechanism, purview=False):

        return self.find_mice(Direction.CAUSE, mechanism, purview)#avoid

    def mie(self, mechanism, purview=False):

        return self.find_mice(Direction.EFFECT, mechanism, purview)#avoid
    
    def null_concept(self):
        
        # Unconstrained cause repertoire.
        cause_repertoire = self.cause_repertoire((), ())
        # Unconstrained effect repertoire.
        effect_repertoire = self.effect_repertoire((), ())
        
        # Null cause.
        cause = MaximallyIrreducibleCause(
            self._null_ria(Direction.CAUSE, (), (), cause_repertoire))#avoid
        # Null effect.
        effect = MaximallyIrreducibleEffect(
            self._null_ria(Direction.EFFECT, (), (), effect_repertoire))#avoid

        # All together now...
        return Concept(mechanism=(),#avoid
                       cause=cause,
                       effect=effect,
                       subsystem=self.dummy_subsys)


    def concept(self, mechanism, purviews=False, cause_purviews=False,
                effect_purviews=False):
        
        log.debug('Computing concept %s...', mechanism)

        # If the mechanism is empty, there is no concept.
        if not mechanism:
            log.debug('Empty concept; returning null concept')
            return self.null_concept

        # Calculate the maximally irreducible cause repertoire.
        cause = self.mic(mechanism)

        # Calculate the maximally irreducible effect repertoire.
        effect = self.mie(mechanism)

        log.debug('Found concept %s')

        return Concept(mechanism=mechanism, cause=cause, effect=effect,
                       subsystem=self.dummy_subsys)
    
    def connections(self):
        
        inputs= [[self._index2label(i) for i in range(self.cm.shape[0]) if self.cm[i,node]==1 ] 
                 for node in self._label2index(self.nodes) ]
        
        return dict(zip(self.nodes, inputs))
    
    def node_tpm(self, cut1, cut2):

        #cut1=self._index2label(cut1)
        #cut2=self._index2label(cut2)
        
        connections = self.connections() #returns the connectivity map of the system
        list_elem_tpm=[]
                
        for element in self.nodes: #this creates the tpm for each element in the subsystem
            inputs=connections[element]
            remain_connections=list(set(inputs)-set(cut1)) if element in cut2 else inputs
            if remain_connections==self.nodes: # if the elment is connected to everyone (including itself), no marg. of any input
                list_elem_tpm+=[self.tpm.groupby(element,axis=1).sum()]
            else:
                if not remain_connections:#element got disconnected, so it depends on itself 
                    remain_connections=[element]
                factor=sum([self.base[i] for i in self._label2index(tuple(set(self.nodes)-set(remain_connections)))  ])
                list_elem_tpm+=[(self.tpm.groupby(element,axis=1).sum().groupby(remain_connections[::-1]).sum())*\
                (1/factor)]
          
        return list_elem_tpm 
    
    def node_tpm_expanded(self,list_elem_tpm):
        
        subsystem=self.nodes
        list_elem_tpm_exp=list()
        a=self.tpm.reset_index()[subsystem]
        a.columns=self.nodes
        for df in list_elem_tpm:#now we expand each element tpm 
            b=df.reset_index()
            b.columns=list(b.columns)
            inter=list(set(a.columns).intersection(set(b.columns)))
            expanded=pd.merge(a,b, on=inter).sort_values(by=subsystem[::-1])
            expanded=expanded[list(set(expanded.columns)-set(subsystem))].values
            list_elem_tpm_exp.append(pd.DataFrame(expanded,columns=df.columns, index=self.tpm.index)) 

        return list_elem_tpm_exp
    
    def cut_cm(self, cut1, cut2):
        
        cut1=self._label2index(cut1)
        cut2=self._label2index(cut2)

        new_cm=self.cm
        new_cm[cut1,cut2]=0        
        return new_cm
    
    def expand(self,repertoire,purview, direction):
        
        purview_complement= tuple(set(self._label2index(self.nodes))-set(list(purview)))
        if len(purview_complement)==0:
            return repertoire
        
        if direction=='c':
            complement=self.cause_repertoire((), purview_complement  )
        else:
            complement=self.effect_repertoire((), purview_complement )
            
        return np.multiply(repertoire, complement)
    
    def clear_cache(self):
        self._single_node_effect_repertoire.cache_clear()
        self._single_node_cause_repertoire.cache_clear()
        self.effect_repertoire.cache_clear()
        self.cause_repertoire.cache_clear()
        self.find_mice.cache_clear()
        
        
    def apply_cut(self,cut_tpm):
        
        self.tpm=self._tpm2df(cut_tpm)
        self.clear_cache()
        if self.base.count(2)==len(self.base): self.dummy_subsys=self._dummy()
        
        
    def remove_cut(self):
        
        condition=sorted(list(set(self.network.node_labels)-set(self.nodes)))
        self.tpm=self._tpm2df(self.network.conditioned_tpm(self.net_state, condition))
        self.clear_cache()
        if self.base.count(2)==len(self.base): self.dummy_subsys=self._dummy()
        
    def __str__(self):
        return 'Subsystem({self.nodes})'.format(self=self)
    
    def __repr__(self):
        return 'Subsystem({self.nodes})'.format(self=self)

