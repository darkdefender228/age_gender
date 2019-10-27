import tensorflow as tf
import math


def combine_loss(embedding, labels, batch_size, out_num, w_init, margin_a=1., margin_m=0.3, margin_b=0.2, s=64.):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param batch_size: input batch size
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    with tf.variable_scope('combine_loss'):
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)
        weights_unit = tf.nn.l2_normalize(weights, axis=0)
        embedding_unit = tf.nn.l2_normalize(embedding, axis=1)
        cos_t = tf.matmul(embedding_unit, weights_unit)
        ordinal = tf.constant(list(range(0, batch_size)), tf.int64)
        ordinal_y = tf.stack([ordinal, labels], axis=1)
        zy = cos_t * s
        sel_cos_t = tf.gather_nd(zy, ordinal_y)
        if margin_a != 1.0 or margin_m != 0.0 or margin_b != 0.0:
            if margin_a == 1.0 and margin_m == 0.0:
                s_m = s * margin_b
                new_zy = sel_cos_t - s_m
            else:
                cos_value = sel_cos_t / s
                t = tf.acos(cos_value)
                if margin_a != 1.0:
                    t = t * margin_a
                if margin_m > 0.0:
                    t = t + margin_m
                body = tf.cos(t)
                if margin_b > 0.0:
                    body = body - margin_b
                new_zy = body * s
        updated_logits = tf.add(zy, tf.scatter_nd(ordinal_y, tf.subtract(new_zy, sel_cos_t), (batch_size, out_num)))
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=updated_logits))
        # predict_cls = tf.argmax(updated_logits, 1)
        # accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(predict_cls, tf.int64), tf.cast(labels, tf.int64)), 'float'))
        # predict_cls_s = tf.argmax(zy, 1)
        # accuracy_s = tf.reduce_mean(tf.cast(tf.equal(tf.cast(predict_cls_s, tf.int64), tf.cast(labels, tf.int64)), 'float'))
        # return zy, loss, accuracy, accuracy_s, predict_cls_s

    return loss, updated_logits