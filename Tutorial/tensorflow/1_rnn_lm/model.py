import numpy as np
import tensorflow as tf

from batch_the_data import read_data, make_batches

############################ define hyper-parameters ############################ 
TRAIN_DATA = 'ptb.train'
EVAL_DATA = 'ptb.valid'
TEST_DATA = 'ptb.test'

HIDDEN_SIZE = 300
NUM_LAYERS = 2
VOCAB_SIZE = 10000
TRAIN_BATCH_SIZE = 20
TRAIN_NUM_STEP = 35

EVAL_BATCH_SIZE = 1
EVAL_NUM_STEP = 1
NUM_EPOCH = 5
LSTM_KEEP_PROB = 0.9
EMBEDDING_KEEP_PROB = 0.9
MAX_GRAD_NORM = 5
SHARE_EMB_AND_SOFTMAX = True

############################ define the model ############################
class Model(object):
    '''
        initialize the model operation in the graph,
        remember to run it in order to process them actually
    '''
    def __init__(self, is_training, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps

        # define the placeholder, in order to feed the data
        self.input_data = tf.placeholder(tf.int32, shape=(batch_size, num_steps))
        self.targets = tf.placeholder(tf.int32, shape=(batch_size, num_steps))

        # define the lstm layer
        dropout_keep_prob = LSTM_KEEP_PROB if is_training else 1.0
        lstm_cells = [tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE), 
                            output_keep_prob=dropout_keep_prob) 
                                for _ in range(NUM_LAYERS)]
        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

        ########################### graph ###########################
        # initial_state contains (h, c)
        self.initial_state = cell.zero_state(batch_size, tf.float32)
        #############################################################

        # define the embedding layer
        # if not initializer passed, the defalut initializer passed in the variable scope will be used,
        # if that one is None too, a gloror_uniform_initializer will be used
        embedding = tf.get_variable('embedding', shape=(VOCAB_SIZE, HIDDEN_SIZE))

        # map the input to the embedding
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        if is_training:
            inputs = tf.nn.dropout(inputs, EMBEDDING_KEEP_PROB)

        # define the output layer
        outputs = []
        state = self.initial_state
        with tf.variable_scope('RNN'):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time_step, :], state)  
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(outputs, 1), (-1, HIDDEN_SIZE))

        if SHARE_EMB_AND_SOFTMAX:
            weight = tf.transpose(embedding)
        else:
            weight = tf.get_variable('weight', [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable('bias', [VOCAB_SIZE])
        logits = tf.matmul(output, weight) + bias

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = tf.reshape(self.targets, [-1]),
                logits = logits
        )

        ########################### graph ###########################
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state
        ############################################################

        if not is_training : return

        train_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, train_variables), MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        self.train_op = optimizer.apply_gradients(zip(grads, train_variables))

def run_epoch(session, model, batches, train_op, output_log, step):
    total_cost = 0.0
    iters = 0
    state = session.run(model.initial_state)

    for x, y in batches:    # x -> (batch_size, num_steps)
        cost, state, _ = session.run([model.cost, model.final_state, train_op], 
                                     {model.input_data : x, model.targets : y, model.initial_state : state})

        total_cost += cost
        iters += model.num_steps

        if output_log and step % 100 == 0:
            print('After %d steps, perplexity is %.3f'%(step, np.exp(total_cost / iters)))
        
        step += 1
    
    return step, np.exp(total_cost / iters)

def main():
    initializer = tf.initializers.random_uniform(-0.05, 0.05)

    with tf.variable_scope('language_model', reuse=None, initializer=initializer):
        train_model = Model(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)
    
    with tf.variable_scope('language_model', reuse=True, initializer=initializer):
        eval_model = Model(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)
    
    with tf.Session() as session:
        tf.global_variables_initializer().run()

        train_batches = make_batches(read_data(TRAIN_DATA), TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)
        eval_batches = make_batches(read_data(EVAL_DATA), EVAL_BATCH_SIZE, EVAL_NUM_STEP)
        test_batches = make_batches(read_data(TEST_DATA), EVAL_BATCH_SIZE, EVAL_NUM_STEP)
        
        step = 0

        for i in range(NUM_EPOCH):
            print('In iteration : %d'%(i+1))
            step, train_pplx = run_epoch(session, train_model, train_batches, train_model.train_op, True, step)
            print('STEP : %d'%(step))
            print('Epoch %d Train Perplexity : %.3f'%(i+1, train_pplx))

            _, eval_pplx = run_epoch(session, eval_model, eval_batches, tf.no_op(), False, 0)
            print('Epoch : %d Eval Perplexity : %.3f'%(i+1, eval_pplx))
        
        _, test_pplx = run_epoch(session, eval_model, test_batches, tf.np_op(), False, 0)
        print('Test Perplexity %.3f'%(test_pplx))

if __name__ == '__main__':
    # model = Model(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)
    main()