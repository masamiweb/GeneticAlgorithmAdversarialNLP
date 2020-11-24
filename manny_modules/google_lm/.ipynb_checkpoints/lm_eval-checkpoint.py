# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
adapted from - Eval pre-trained 1 billion word language model.
"""

import sys
import tensorflow as tf
from google.protobuf import text_format


def load_model(sess, graph, gd_file, ckpt_file):
    """Load the model from GraphDef and Checkpoint.

  Args:
    gd_file: GraphDef proto text file.
    ckpt_file: TensorFlow Checkpoint file.

  Returns:
    TensorFlow session and tensors dict.
    :param ckpt_file:
    :param gd_file:
    :param graph:
    :param sess:
  """
    with graph.as_default():
        sys.stderr.write('Recovering graph.\n') 
        with tf.io.gfile.GFile(gd_file, 'r') as f:
            s = f.read()
            gd = tf.compat.v1.GraphDef()
            text_format.Merge(s, gd)

        tf.compat.v1.logging.info('Recovering Graph %s', gd_file)
        t = {}
        [t['states_init'], t['lstm/lstm_0/control_dependency'],
         t['lstm/lstm_1/control_dependency'], t['softmax_out'], t['class_ids_out'],
         t['class_weights_out'], t['log_perplexity_out'], t['inputs_in'],
         t['targets_in'], t['target_weights_in'], t['char_inputs_in'],
         t['all_embs'], t['softmax_weights'], t['global_step']
         ] = tf.import_graph_def(gd, {}, ['states_init',
                                          'lstm/lstm_0/control_dependency:0',
                                          'lstm/lstm_1/control_dependency:0',
                                          'softmax_out:0',
                                          'class_ids_out:0',
                                          'class_weights_out:0',
                                          'log_perplexity_out:0',
                                          'inputs_in:0',
                                          'targets_in:0',
                                          'target_weights_in:0',
                                          'char_inputs_in:0',
                                          'all_embs_out:0',
                                          'Reshape_3:0',
                                          'global_step:0'], name='')

        sys.stderr.write('Recovering checkpoint %s\n' % ckpt_file)
        sess.run('save/restore_all', {'save/Const:0': ckpt_file})
        sess.run(t['states_init'])

    return t
