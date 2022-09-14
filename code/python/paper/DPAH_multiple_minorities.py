################################################################
# Systems' dependencies
################################################################
import time
import powerlaw
import numpy as np
import networkx as nx
from collections import Counter

################################################################
# Constants
################################################################
CLASS = 'm'
LABELS = [0,1,2] # 0 for majority, minority 1, minority 2
GROUPS = ['M', 'm1', 'm2']
EPSILON = 0.00001

################################################################
# Functions
################################################################

def DPAH(N, d, fm1, fm2, plo_M, plo_m1, plo_m2, h, verbose=False, seed=None):
    '''
    Generates a Directed Barabasi-Albert Homophilic network.
    - param N: number of nodes
    - param d: max edge density
    - param fm: fraction of minorities
    - param plo_M: power-law outdegree distribution majority class
    - param plo_m: power-law outdegree distribution minority class
    - h_MM: homophily among majorities
    - h_mm: homophily among minorities
    - verbose: if True prints every steps in detail.
    - seed: randommness seed for reproducibility
    '''
    np.random.seed(seed)
    start_time = time.time()

    # 1. Init nodes
    nodes, labels, NM, Nm1, Nm2 = _init_nodes(N,fm1,fm2)

    # 2. Init Directed Graph
    G = nx.DiGraph()
    G.graph = {'name':'DPAH', 'label':CLASS, 'groups': GROUPS}
    G.add_nodes_from([(n, {CLASS:l}) for n,l in zip(*[nodes,labels])])

    # 3. Init edges and indegrees
    E = int(round(d * N * (N-1)))
    indegrees = np.zeros(N)
    outdegrees = np.zeros(N)

    # 4. Init Activity (out-degree)
    act_M = powerlaw.Power_Law(parameters=[plo_M], discrete=True).generate_random(NM)
    act_m1 = powerlaw.Power_Law(parameters=[plo_m1], discrete=True).generate_random(Nm1)
    act_m2 = powerlaw.Power_Law(parameters=[plo_m2], discrete=True).generate_random(Nm2)
    activity = np.concatenate([act_M, act_m1, act_m2])
    activity /= activity.sum()

    # 5. Init homophily
    homophily = np.array([[EPSILON if h_val == 0 else h_val for h_val in h[i]] for i in range(len(h))])

    # INIT SUMMARY
    if verbose:
        print("Directed Graph:")
        print("N={} (NM={}, Nm1={}, Nm2={})".format(N, NM, Nm1, Nm2))
        print("E={} (d={})".format(E, d))
        print("Activity Power-Law outdegree: M={}, m1={}, m2={}".format(plo_M, plo_m1, plo_m2))
        print("Homophily (M, m1, m2 x M, m1, m2):\n{}".format(homophily))
        print('')

    # 5. Generative process
    tries = 0
    while G.number_of_edges() < E:
        tries += 1
        source = _pick_source(N, activity)
        ns = nodes[source]
        target = _pick_target(source, N, labels, indegrees, outdegrees, homophily)

        if target is None:
            tries = 0
            continue

        nt = nodes[target]

        if not G.has_edge(ns, nt):
            G.add_edge(ns, nt)
            indegrees[target] += 1
            outdegrees[source] += 1
            tries = 0

        if verbose:
            ls = labels[source]
            lt = labels[target]
            print("{}->{} ({}{}): {}".format(ns, nt, str(ls) if ls>0 else 'M', str(lt) if lt>0 else 'M',
                                             G.number_of_edges()))

        if tries > G.number_of_nodes():
            # it does not find any more new connections
            print("\nEdge density ({}) might differ from {}. N{} fm1{} fm2() seed{}\n".format(round(nx.density(G),5),
                                                                                              round(d,5),N,fm1, fm2,
                                                                                              seed))
            break

    duration = time.time() - start_time
    if verbose:
        print()
        print(G.graph)
        print(nx.info(G))
        degrees = [d for n,d in G.out_degree()]
        print("min degree={}, max degree={}".format(min(degrees), max(degrees)))
        print(Counter(degrees))
        print(Counter([data[1][CLASS] for data in G.nodes(data=True)]))
        print()
        for k in [0,1]:
            fit = powerlaw.Fit(data=[d for n,d in G.out_degree() if G.nodes[n][CLASS]==k], discrete=True)
            print("{}: alpha={}, sigma={}, min={}, max={}".format('m' if k else 'M',
                                                                  fit.power_law.alpha,
                                                                  fit.power_law.sigma,
                                                                  fit.power_law.xmin,
                                                                  fit.power_law.xmax))
        print()
        print("--- %s seconds ---" % (duration))

    return G

def _init_nodes(N, fm1, fm2):
    '''
    Generates random nodes, and assigns them a binary label.
    param N: number of nodes
    param fm1: fraction of minority 1
    param fm2: fraction of minority 2
    '''

    nodes = np.arange(N)
    np.random.shuffle(nodes)
    Nm1 = int(round(N*(fm1)))
    Nm2 = int(round(N*(fm2)))
    NM = N - Nm1 - Nm2
    labels = np.empty((N), dtype=int)
    for i in range(NM):
        labels[i] = LABELS[0]
    for i in range(NM, NM+Nm1):
        labels[i] = LABELS[1]
    for i in range(NM+Nm1, N):
        labels[i] = LABELS[2]

    return nodes, labels, NM, Nm1, Nm2

def _pick_source(N,activity):
    '''
    Picks 1 (index) node as source (edge from) based on activity score.
    '''
    return np.random.choice(a=np.arange(N),size=1,replace=True,p=activity)[0]

def _pick_target(source, N, labels, indegrees, outdegrees, homophily):
    '''
    Given a (index) source node, it returns 1 (index) target node based on homophily and pref. attachment (indegree).
    The target node must have out_degree > 0 (the older the node in the network, the more likely to get more links)
    '''
    one_percent = N * 1/100.
    targets = [n for n in np.arange(N) if n!=source and (outdegrees[n]>0 if outdegrees.sum()>one_percent else True)]

    if len(targets) == 0:
        return None

    probs = np.array([ homophily[labels[source],labels[n]] * (indegrees[n]+1) for n in targets])

    probs /= probs.sum()
    
    if sum(1 for i in probs if i < 0)>0:
        print('This is a negative probability')
        print(probs)
    return np.random.choice(a=targets,size=1,replace=True,p=probs)[0]

################################################################
# Main
################################################################

if __name__ == "__main__":

    G = DPAH(N=100,
             fm1=0.25,
             fm2=0.25,
             d=0.01,
             plo_M=2.5,
             plo_m1=2.5,
             plo_m2=2.5,
             h=np.ones((3,3)),
             verbose=True,
             seed=1)

