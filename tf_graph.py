import tensorflow as tf
import copy
import networkx as nx
import queue

def pb2nx(graph_def):
    """Convert graph_def to NetworkX
    """
    # graph_def to hash
    pb_hash = {}
    for n in graph_def.node:
        pb_hash[n.name] = n

    G = nx.DiGraph()
    G.add_nodes_from(pb_hash.keys())
    for node in graph_def.node:
        v = node.name
        for i in node.input:
            x = i.split(':')
            u = x[0]
            j = 0 if len(x) == 1 else int(x[-1])
            G.add_edge(u, v)

    #@print G.out_edges()
    return G, pb_hash

def remove_input_node(input_graph_def, input_node_name, keep_switch=0):
    """Remove an input node with tree from input_graph_def
    
    Note:
    Fixed branch of switch will be kept (keep_switch)
    There should be no cycle in input_graph_def
    """
    G, pb_hash = pb2nx(input_graph_def)

    hard_rm_list = []
    done_list = []
    # find nodes from leaf
    q = queue.Queue()
    q.put(input_node_name)
    while not q.empty():
        x = q.get()
        if x in done_list:
            continue
        done_list.append(x)
        op = pb_hash[x].op
        outs = G.out_edges(x)
        # TODO Add
        if op == 'Switch':
            hard_rm_list.append(x)
            ins = pb_hash[x].input
            assert len(ins) == 2
            # input_data
            if ins[0].split(':')[0] in hard_rm_list:
                for o in outs:
                    y = o[1]
                    q.put(y)
            else:
                for o in outs:
                    y = o[1]
                    # check all inputs, keep keep_switch
                    yins = pb_hash[y].input
                    for j, iii in enumerate(yins):
                        iiii = iii.split(':')
                        yi = 0 if len(iiii) == 1 else int(iiii[-1])
                        if iiii[0] == x and yi != keep_switch:
                            q.put(y)
                            break
                        elif iiii[0] == x:
                            yins[j] = ins[0]
        elif op == 'Merge':
            mi = pb_hash[x].input
            if len(mi) == 1:
                for o in outs:
                    y = o[1]
                    q.put(y)
            ri = []
            for mm in mi:
                if mm.split(':')[0] in hard_rm_list:
                    ri.append(mm)
            for rr in ri:
                pb_hash[x].input.remove(rr)
                pb_hash[x].attr['N'].i -= 1
            if len(pb_hash[x].input) == 0:
                hard_rm_list.append(x)
                for o in outs:
                    y = o[1]
                    q.put(y)
        else:
            hard_rm_list.append(x)
            for o in outs:
                y = o[1]
                q.put(y)

    # from pb_hash to output_graph_def
    output_graph_def = tf.GraphDef()
    for node_name in pb_hash.keys():
        if node_name not in hard_rm_list:
            output_graph_def.node.extend([copy.deepcopy(pb_hash[node_name])])

    return output_graph_def

def rm_const_input(graph_def):
    """In place remove Const op inputs
    """
    for n in graph_def.node:
        if n.op == 'Const':
            if len(n.input):
                n.ClearField('input')
