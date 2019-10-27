import tensorflow as tf
import math

def Modified_Softmax_Loss(embeddings, labels):
    """
    This kind of loss is slightly different from the orginal softmax loss. the main difference
    lies in that the L2-norm of the weights are constrained  to 1, then the
    decision boundary will only depends on the angle between weights and embeddings.
    """
    # # normalize embeddings
    # embeddings_norm = tf.norm(embeddings, axis=1, keepdims=True)
    # embeddings = tf.div(embeddings, embeddings_norm, name="normalize_embedding")
    """
    the abovel commented-out code would lead loss to divergence, maybe you can try it.
    """
    with tf.variable_scope("softmax"):
        weights = tf.get_variable(name='embedding_weights',
                                    shape=[embeddings.get_shape().as_list()[-1], 10],
                                    initializer=tf.contrib.layers.xavier_initializer())
        # normalize weights
        weights_norm = tf.norm(weights, axis=0, keepdims=True)
        weights = tf.div(weights, weights_norm, name="normalize_weights")
        logits = tf.matmul(embeddings, weights)
        pred_prob = tf.nn.softmax(logits=logits)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        return pred_prob, loss