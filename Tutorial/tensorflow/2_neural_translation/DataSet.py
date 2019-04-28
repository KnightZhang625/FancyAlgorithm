MAX_LEN = 50
SOS_ID = 180
import tensorflow as tf
import codecs
import pickle

# with codecs.open('word_idx_en.pickle', 'rb') as file:
#     word_idx = pickle.load(file)
# print(word_idx)
# sys.exit()

def MakeDataset(file_path):
    dataset = tf.data.TextLineDataset(file_path)
    dataset = dataset.map(lambda string : tf.string_split([string]).values)
    dataset = dataset.map(lambda string : tf.string_to_number(string, tf.int32))
    dataset = dataset.map(lambda x : (x, tf.size(x)))
    return dataset

def MakeSrcTrgDataset(src_path, trg_path, batch_size):
    src_data = MakeDataset(src_path)
    trg_data = MakeDataset(trg_path)

    dataset = tf.data.Dataset.zip((src_data, trg_data))
    
    def FilterLength(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        src_len_ok = tf.logical_and(tf.greater(src_len, 1), tf.less_equal(src_len, MAX_LEN))
        trg_len_ok = tf.logical_and(tf.greater(trg_len, 1), tf.less_equal(trg_len, MAX_LEN))
        
        return tf.logical_and(src_len_ok, trg_len_ok)
    
    dataset = dataset.filter(FilterLength)

    def MakeTrgInput(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        trg_input = tf.concat([[SOS_ID], trg_label[:-1]], axis=0)
        return ((src_input, src_len), (trg_input, trg_label, trg_len))
    
    dataset = dataset.map(MakeSrcTrgDataset)

    dataset = dataset.shuffle(20)

    padded_shapes = (
                        (tf.TensorShape([None]),
                         tf.TensorShape([])),
                        (tf.TensorShape([None]),
                         tf.TensorShape([None]),
                         tf.TensorShape([]))
                    )
    batched_dataset = dataset.padded_batch(batch_size, padded_shapes)
    return batched_datasetN

if __name__ == '__main__':
    MakeSrcTrgDataset('train_en', 'train_zh', 5)
    