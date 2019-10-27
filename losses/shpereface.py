import tensorflow as tf
import math

def Angular_Softmax_Loss(embeddings, labels, margin=4):
    """
    Note:(about the value of margin)
    as for binary-class case, the minimal value of margin is 2+sqrt(3)
    as for multi-class  case, the minimal value of margin is 3
    the value of margin proposed by the author of paper is 4.
    here the margin value is 4.
    """
    l = 0.
    embeddings_norm = tf.norm(embeddings, axis=1)

    with tf.variable_scope("softmax"):
        weights = tf.get_variable(name='embedding_weights',
                                    shape=[embeddings.get_shape().as_list()[-1], 10],
                                    initializer=tf.contrib.layers.xavier_initializer())
        weights = tf.nn.l2_normalize(weights, axis=0)
        # cacualting the cos value of angles between embeddings and weights
        orgina_logits = tf.matmul(embeddings, weights)
        N = embeddings.get_shape()[0] # get batch_size
        single_sample_label_index = tf.stack([tf.constant(list(range(N)), tf.int64), labels], axis=1)
        # N = 128, labels = [1,0,...,9]
        # single_sample_label_index:
        # [ [0,1],
        #   [1,0],
        #   ....
        #   [128,9]]
        selected_logits = tf.gather_nd(orgina_logits, single_sample_label_index)
        cos_theta = tf.div(selected_logits, embeddings_norm)
        cos_theta_power = tf.square(cos_theta)
        cos_theta_biq = tf.pow(cos_theta, 4)
        sign0 = tf.sign(cos_theta)
        sign3 = tf.multiply(tf.sign(2*cos_theta_power-1), sign0)
        sign4 = 2*sign0 + sign3 -3
        result=sign3*(8*cos_theta_biq-8*cos_theta_power+1) + sign4

        margin_logits = tf.multiply(result, embeddings_norm)
        f = 1.0/(1.0+l)
        ff = 1.0 - f
        combined_logits = tf.add(orgina_logits, tf.scatter_nd(single_sample_label_index,
                                                        tf.subtract(margin_logits, selected_logits),
                                                        orgina_logits.get_shape()))
        updated_logits = ff*orgina_logits + f*combined_logits
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=updated_logits))
        pred_prob = tf.nn.softmax(logits=updated_logits)
        return pred_prob, loss