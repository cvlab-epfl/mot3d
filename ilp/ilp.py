import networkx as nx
from pulp import *
import time
import ast

def solve_graph_ilp(g, verbose=True):

    if verbose:
        start_time = time.time()
    
    edges = list(g.edges())
    costs = [g.get_edge_data(*edge)['weight'] for edge in edges]
    edge_names = ["{}".format(edge) for edge in edges]

    prob = LpProblem("Min-cost flow solver", LpMinimize)
    flows = LpVariable.dicts("edge", edge_names, 0, 1, LpInteger)

    cost_fun = []
    for edge_name, cost in zip(edge_names, costs):
        w = cost*flows[edge_name]
        cost_fun.append(w)
    prob += lpSum(cost_fun), "Objective function"

    SOURCE = 1
    SINK = list(g.nodes())[-1]

    for node in g.nodes():

        if node==SOURCE or node==SINK:
            continue

        in_edge_names = ["{}".format(edge) for edge in g.in_edges(node)]
        out_edge_names = ["{}".format(edge) for edge in g.out_edges(node)]

        in_flow = sum([flows[edge_name] for edge_name in in_edge_names])
        out_flow = sum([flows[edge_name] for edge_name in out_edge_names])

        constraint = in_flow-out_flow

        prob += lpSum(constraint) == 0, ""

    prob.solve()    
    
    if verbose:
        elapsed_time = time.time()-start_time
        final_cost = value(prob.objective)
        print("status:{} final cost:{} elapsed_time:{:0.2}s".format(LpStatus[prob.status], final_cost, elapsed_time))


    g_copy = g.copy()
    for edge_name, val in flows.items():
        if val.value()==0:
            g_copy.remove_edge(*ast.literal_eval(edge_name))
    SOURCE = list(g.nodes())[0]
    SINK = list(g.nodes())[-1]
    tracks_nodes = [ track[::2][1:] for track in list(nx.all_simple_paths(g_copy, SOURCE, SINK))] 
    
    return tracks_nodes
