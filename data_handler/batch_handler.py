import tensorflow as tf


def batch_producer(ids, batch_size):
    with tf.name_scope(None, "BatchProducer", [ids, batch_size]):
        raw_data = tf.convert_to_tensor(ids, dtype=tf.int32, name="raw_data")
        raw_data = tf.reshape(raw_data, [-1])

        data_len = len(raw_data)
        batch_len = (data_len // batch_size) // 3
        data = tf.reshape(raw_data[0: batch_size * batch_len * 3], shape=[batch_size, -1])

        assertion = tf.assert_positive(batch_len, message="batch_size过大，数据不够了，需要调小batch_size或减少num_steps")
        with tf.control_dependencies([ assertion ]):
            batch_len = tf.identity(batch_len, name="epoch_size")

        index = tf.train.range_input_producer(batch_len, shuffle=False).dequeue()

        head = tf.strided_slice(data, [0, index * 3], [batch_len, index * 3 + 1])
        head = tf.reshape(head, [-1])

        relation = tf.strided_slice(data, [0, index * 3 + 1], [batch_len, index * 3 + 2])
        relation = tf.reshape(relation, [-1])

        tail = tf.strided_slice(data, [0, index * 3 + 2], [batch_len, index * 3 + 3])
        tail = tf.reshape(tail, [-1])

        return head, relation, tail



