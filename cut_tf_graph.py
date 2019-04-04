import tensorflow as tf
from tensorflow.contrib.graph_editor import SubGraphView
from tensorflow.contrib.graph_editor import make_list_of_op
from tensorflow.contrib.graph_editor import make_placeholder_from_tensor
from tensorflow.contrib.graph_editor import copy_with_input_replacements

def cut_graph_def(graph_def, cut_nodes):
    """Cut groph_def to two parts by cut_nodes. All ancesters  of cut_nodes are put
    into back and the rest are put into head.

    Args:
        graph_def: input tf.GraphDef
        cut_nodes: a list of node names to cut
    """
    # back
    back = tf.graph_util.extract_sub_graph(graph_def, cut_nodes)

    # head
    head_node_names = [n.name for n in graph_def.node if n not in back.node]
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        all_ops = make_list_of_op(graph)
    head_ops = [o for o in all_ops if o.name in head_node_names]
    head_subgraph = SubGraphView(inside_ops=head_ops)

    head_graph = tf.Graph()
    replace_ts = {}
    for i in head_subgraph.inputs:
        k = i.name
        replace_ts[k] = make_placeholder_from_tensor(i)
    copy_with_input_replacements(
        head_subgraph,
        replace_ts,
        dst_graph=head_graph
    )

    # return
    return back, head_graph.as_graph_def()
