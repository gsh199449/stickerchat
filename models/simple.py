import tensorflow as tf

FLAGS = tf.flags.FLAGS


def context_encoder(enc_context_batch, enc_context_lens):
    """

    :param enc_context_batch: [batch, context_len, max_enc_len, emb]
    :param enc_context_lens: [batch, context_len]
    :return: encoder_outputs [batch, context_len, max_enc_len, 2*hidden_dim], final_state [batch, context_len, 2*hidden_dim]
    """
    with tf.variable_scope('context_word_encoder'):
        if FLAGS.context_encoder == 'bigru':
            cell_fw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_dim)
            cell_bw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_dim)
            enc_context_batch = tf.reshape(enc_context_batch, [-1, FLAGS.max_enc_steps, FLAGS.emb_dim])
            enc_context_lens = tf.reshape(enc_context_lens, [-1])
            (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, enc_context_batch,
                                                                                dtype=tf.float32,
                                                                                sequence_length=enc_context_lens,
                                                                                swap_memory=True)
            encoder_outputs = tf.concat(axis=2, values=encoder_outputs)  # concatenate the forwards and backwards states
            encoder_outputs = tf.reshape(encoder_outputs, [-1, FLAGS.context_len, FLAGS.max_enc_steps, 2*FLAGS.hidden_dim])
            final_state = tf.concat([fw_st, bw_st], axis=-1)
            final_state = tf.reshape(final_state, [-1, FLAGS.context_len, 2*FLAGS.hidden_dim])
        elif FLAGS.context_encoder == 'transformer':
            import opennmt as onmt
            enc_context_batch = tf.reshape(enc_context_batch, [-1, FLAGS.max_enc_steps, FLAGS.emb_dim])
            enc_context_lens = tf.reshape(enc_context_lens, [-1])
            fusion_self_encoder = onmt.encoders.SelfAttentionEncoder(2, num_units=2*FLAGS.hidden_dim, num_heads=5)
            encoder_outputs, final_state, _ = fusion_self_encoder.encode(enc_context_batch, enc_context_lens)
            encoder_outputs = tf.reshape(encoder_outputs, [-1, FLAGS.context_len, FLAGS.max_enc_steps, encoder_outputs.get_shape()[-1]])
            final_state = final_state[-1]
            final_state = tf.reshape(final_state, [-1, FLAGS.context_len, final_state.get_shape()[-1]])
        else:
            raise NotImplementedError()
    return encoder_outputs, final_state
