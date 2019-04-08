#!/usr/bin/python
import tensorflow as tf

def freeze_model(cp_path, output_node_names, output_file='frozen.pb'):
    """freeze tf models

    Args:
        cp_path: tf checkpoint path with checkpoint file
        output_node_names: model output nodes
        output_file: frozen graph protobuf serialize file
    """
    sess = tf.Session()
    ckp_state = tf.train.get_checkpoint_state(cp_path)
    saver = tf.train.import_meta_graph(
        ckp_state.model_checkpoint_path + '.meta'
    )
    saver.restore(sess, ckp_state.model_checkpoint_path)

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def(add_shapes=True)

    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        input_graph_def,
        output_node_names
    )
    
    if output_file:
        with tf.gfile.GFile(output_file, 'wb') as f:
            f.write(output_graph_def.SerializeToString())

    return output_graph_def
