import tensorflow as tf

SRC_TRAIN_PATH = 'train_en'
TRG_TRAIN_PATH = 'train_zh'
CHECKPOINT_PATH = 'seq2seq_ckpt'
HIDDEN_SIZE = 50
NUM_LAYERS = 2
SRC_VOCAB_SIZE = 174
TRG_VOCAB_SIZE = 183
NUM_EPOCH = 5
KEEP_PROB = 0.8
MAX_GRAG_NROM = 5
SHARR_EMB_AND_SOFTMAX = True

class Model(object):
    def __init__(self):
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])

        self.src_embedding = tf.get_variable('src_emb', [SRC_VOCAB_SIZE, HIDDEN_SIZE])
        self.trg_embedding = tf.get_variable('trg_emb', [TRG_VOCAB_SIZE, HIDDEN_SIZE])

        if SHARR_EMB_AND_SOFTMAX:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight = tf.get_variable('weight', [HIDDEN_SIZE, TRG_VOCAB_SIZE])
        self.softmax_bias = tf.get_variable('softmax_bias', [TRG_VOCAB_SIZE])

        def forward(self, src_input, src_size, trg_input, trg_label, trg_size):
            batch_size = tf.shape(src_input)[0]

            src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)
            trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)

            src_emb = tf.nn.dropout(src_emb, KEEP_PROB)
            trg_emb = tf.nn.dropout(trg_emb, KEEP_PROB)

            with tf.variable_scope('encoder'):
                # enc_outputs : (batch_size, max_time, HIDDEN_SIZE)
                # enc_state : ((c, h), (c, h))
                enc_outputs, enc_state = tf.nn.dynamic_rnn(self.enc_cell, src_emb, src_size, dtype=tf.float32)

            initial_state = enc_state
            with tf.variable_scope('decoder'):
                dec_outputs, _ = tf.nn.dynamic_rnn(self.dec_cell, trg_emb, trg_size, initial_state=enc_state)
            
            output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
            logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(trg_label, [-1]), logits=logits)
            
            label_weights = tf.sequence_mask(trg_size, maxlen=tf.shape(trg_label)[1], dtype=tf.float32)
            label_weights = tf.reshape(label_weights, [-1])
            cost = tf.reduce_sum(loss * label_weights)
            cost_per_token = cost / tf.reduce_sum(label_weights)

            trainable_variables = tf.trainable_variables()

            grads = tf.gradients(cost / tf.to_float(batch_size), trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, MAX_GRAG_NROM)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
            train_op = optimizer.apply_gradients(zip(grads, trainable_variables))
            
            return cost_per_token, train_op





















