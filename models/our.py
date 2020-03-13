import tensorflow as tf

FLAGS = tf.flags.FLAGS


def our_fusion(sticker_encoded, sticker_pix_feature, context_encoded_states, context_final_state, context_number):
    """

    :param sticker_encoded: [batch, candidate_num, 2*hidden]
    :param sticker_pix_feature: inception (batch*candidate, 6, 6, 768)
    :param context_encoded_states: [batch, context_len, max_enc_len, 2*hidden_dim]
    :param context_final_state: [batch, context_len, 2*hidden_dim]
    :param context_number: [batch]
    :return:
    """
    trans_sticker_pix_feature = tf.layers.dense(sticker_pix_feature, 2 * FLAGS.hidden_dim, use_bias=False)
    # inception (batch*candidate, 6, 6, 2*hidden_dim), cifar (batch*candidate, 32, 32, 2*hidden_dim)
    trans_sticker = tf.layers.dense(sticker_encoded, 2 * FLAGS.hidden_dim, use_bias=False)
    pix_size = trans_sticker_pix_feature.get_shape()[-2]
    pix_dim = trans_sticker_pix_feature.get_shape()[-1]
    # 扩展图片向量维度
    tiled_sticker_pix_feature = tf.expand_dims(trans_sticker_pix_feature, 1)  # (batch*candidate, 1, 32, 32, 2*FLAGS.hidden_dim)
    tiled_sticker_pix_feature = tf.tile(tiled_sticker_pix_feature, [1, FLAGS.context_len, 1, 1, 1])
    # (batch*candidate, context_len, 32, 32, 2*FLAGS.hidden_dim)
    flatten_sticker_pix_feature = tf.reshape(tiled_sticker_pix_feature,
            [FLAGS.batch_size * FLAGS.sticker_candidates * FLAGS.context_len, -1, pix_dim])
    # (batch*candidate*context_len, pix_len, 2*FLAGS.hidden_dim)

    # 扩展context向量维度
    tiled_context_encoded_states = tf.expand_dims(context_encoded_states, 1)  # [batch, 1, context_len, max_enc_len, 2*hidden_dim]
    tiled_context_encoded_states = tf.tile(tiled_context_encoded_states, [1, FLAGS.sticker_candidates, 1, 1, 1])
    # [batch, candidate, context_len, max_enc_len, 2*hidden_dim]
    flatten_context_encoded_states = tf.reshape(tiled_context_encoded_states,
                                                [FLAGS.batch_size * FLAGS.sticker_candidates * FLAGS.context_len, -1, 2*FLAGS.hidden_dim])
    # (batch*candidate*context_len, max_enc_len, 2*FLAGS.hidden_dim)

    # 算图和context之间的attention权重
    attention = tf.matmul(flatten_sticker_pix_feature, flatten_context_encoded_states, transpose_b=True)
    # (batch*candidate*context_len, pix_len, max_enc_len)
    attention = tf.reshape(attention, [FLAGS.batch_size, FLAGS.sticker_candidates, FLAGS.context_len, -1, FLAGS.max_enc_steps])
    attention = attention / tf.sqrt(tf.cast(2*FLAGS.hidden_dim, tf.float32))

    # 计算图片对于context的context vector
    pix2ctx_attention = tf.nn.softmax(attention, axis=4)  # (batch, candidate, context_len, pix_len, max_enc_len)
    pix2ctx_attention = tf.reduce_max(pix2ctx_attention, 3)

    # (batch, candidate, context_len, max_enc_len)
    utterance_explanation_weight = pix2ctx_attention
    fused_context_repr = tf.reduce_mean(tf.expand_dims(pix2ctx_attention, -1) * tiled_context_encoded_states, 3)
    # [batch, candidate, context_len, 2*hidden_dim]

    # 计算context对于图片的context vector
    ctx2pix_attention = tf.nn.softmax(attention, axis=3)
    ctx2pix_attention = tf.reduce_max(ctx2pix_attention, 4)
    # (batch, candidate, context_len, pix_len)

    explanation_weight = tf.reshape(ctx2pix_attention, [-1, FLAGS.sticker_candidates, FLAGS.context_len, pix_size, pix_size])
    tf.summary.image('complex_attention', tf.reshape(ctx2pix_attention, [-1, FLAGS.context_len, pix_size, pix_size, 1])[:, 0, :, :, :], max_outputs=6)

    tiled_sticker_pix_feature = tf.reshape(tiled_sticker_pix_feature,
                            [-1, FLAGS.sticker_candidates, FLAGS.context_len, pix_size*pix_size, pix_dim])
    complex_fused_sticker_repr = fused_sticker_repr = tf.reduce_mean(tf.expand_dims(ctx2pix_attention, -1) * tiled_sticker_pix_feature, 3)
    # [batch, candidate, context_len, 2*FLAGS.hidden_dim]
    trans_sticker = tf.expand_dims(trans_sticker, 2)  # [batch, candidate_num, 1, 2*hidden]
    fused_trans_sticker_repr = tf.tile(trans_sticker, [1, 1, FLAGS.context_len, 1])
    # [batch, sticker_candidates, context_len, 2 * hidden_dim]
    fused_sticker_repr = fused_trans_sticker_repr

    fusion = tf.concat([
        fused_context_repr + fused_sticker_repr,
        fused_context_repr * fused_sticker_repr,
        fused_context_repr,
        fused_sticker_repr
    ], axis=-1)  # [batch, sticker_candidates, context_len, 4 * 2 * hidden_dim]
    fusion = tf.reshape(fusion, [-1, FLAGS.context_len, 4*2*FLAGS.hidden_dim])
    fusion = tf.concat([fusion, tf.reshape(complex_fused_sticker_repr, [-1, FLAGS.context_len, 2*FLAGS.hidden_dim])], axis=2)

    context_number = tf.expand_dims(context_number, axis=1)
    context_number = tf.tile(context_number, [1, FLAGS.sticker_candidates])
    context_number = tf.reshape(context_number, [-1])

    # fusion GRU
    cell = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_dim)
    (gru_encoder_outputs, state) = tf.nn.dynamic_rnn(cell, fusion, dtype=tf.float32, sequence_length=context_number,
                                                                        swap_memory=True)

    # fusion transformer
    import opennmt as onmt
    fusion_self_encoder = onmt.encoders.SelfAttentionEncoder(num_layers=2, num_heads=4, num_units=FLAGS.hidden_dim)
    trans_encoder_outputs, state, _ = fusion_self_encoder.encode(fusion, context_number)

    mask = tf.sequence_mask(context_number, maxlen=FLAGS.context_len, dtype=tf.float32)
    submulti = batch_coattention_nnsubmulti(gru_encoder_outputs, trans_encoder_outputs, mask)
    mix_cell = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_dim, name='turn_fusion_gru')
    (_, state) = tf.nn.dynamic_rnn(mix_cell, submulti, dtype=tf.float32, sequence_length=context_number, swap_memory=True)

    logits = tf.layers.dense(state, 1, kernel_initializer=tf.contrib.layers.xavier_initializer())
    logits = tf.reshape(logits, [-1, FLAGS.sticker_candidates])
    return logits, explanation_weight, utterance_explanation_weight


def batch_coattention_nnsubmulti(utterance, response, utterance_mask, scope="co_attention", reuse=None):
    '''Point-wise interaction. (NNSUBMULTI)
    Args:
      utterance: [batch*turns, len_utt, dim]
      response: [batch*turns, len_res, dim]
      scope: Optional scope for `variable_scope`.

    Returns:
      A 3d tensor with the same shape and dtype as response
    '''

    with tf.variable_scope(scope, reuse=reuse):
        dim = utterance.get_shape().as_list()[-1]

        weight = tf.get_variable('Weight', shape=[dim, dim], dtype=tf.float32)
        e_utterance = tf.einsum('aij,jk->aik', utterance, weight)
        # [batch, len_res, dim] * [batch, len_utterance, dim]T -> [batch, len_res, len_utterance]
        a_matrix = tf.matmul(response, tf.transpose(e_utterance, perm=[0, 2, 1]))  # [batch, len_res, len_utterance]

        reponse_atten = tf.matmul(rx_masked_softmax(a_matrix, utterance_mask), utterance)  #

        feature_mul = tf.multiply(reponse_atten, response)
        feature_sub = tf.subtract(reponse_atten, response)
        feature_last = tf.layers.dense(tf.concat([feature_mul, feature_sub], axis=-1), dim, use_bias=True,
                                       activation=tf.nn.relu, reuse=reuse)  # [batch*turn, len, dim]
    return feature_last


def rx_masked_softmax(scores, mask):
    """
    Used to calculcate a softmax score with true sequence length (without padding), rather than max-sequence length.
    Input shape: (batch_size, len_res, len_utt).
    mask parameter: Tensor of shape (batch_size, len_utt). Such a mask is given by the length() function.
    return shape: [batch_size, len_res, len_utt]
    """
    numerator = tf.exp(tf.subtract(scores, tf.reduce_max(scores, 2, keep_dims=True))) * tf.expand_dims(mask, axis=1)
    denominator = tf.reduce_sum(numerator, 2, keep_dims=True)

    weights = tf.div(numerator + 1e-5 / mask.get_shape()[-1].value, denominator + 1e-5)
    return weights
