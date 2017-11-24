from data_handler.constant import *
from data_handler.batch_handler import *
from data_handler.data_reader import *
from model.RealTransModel import *

import tensorflow as tf

if __name__ == '__main__':
    config = RealTransModelConfig()
    entity_voc_list, relation_voc_list = read_voc_list()
    config.VOC_CNT = len(entity_voc_list)
    config.REL_CNT = len(relation_voc_list)
    train_ids, test_ids = get_train_and_test_ids(percent=config.TRAIN_TEST_RATIO)
    print('ids准备完毕')

    train_h_batch, train_r_batch, train_t_batch = batch_producer(train_ids, config.BATCH_SIZE)
    test_h_batch, test_r_batch, test_t_batch = batch_producer(test_ids, config.BATCH_SIZE)
    print('data batch准备完毕')

    initializer = tf.random_uniform_initializer(-config.INIT_SCALE, config.INIT_SCALE)

    with tf.name_scope("Train"):
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            model = TransNSigmoidModel(config, is_training=True)

    with tf.name_scope("Test"):
        with tf.variable_scope("Model", reuse=True):
            test_model = TransNSigmoidModel(config, is_training=False)

    print("模型准备完毕")

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        tf.global_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(200000):
        head, relation, tail = sess.run([train_h_batch, train_r_batch, train_t_batch])

        _, softmax_loss, _, nce_loss = sess.run([model.softmax_train_op, model.softmax_loss, model.nce_train_op,
                                                 model.nce_loss],
                                                {model.heads: head, model.relations: relation,
                                                 model.tails: tail})

        if step % 400 == 0:
            softmax_pred, softmax_acc, softmax_top10_acc, softmax_top20_acc, softmax_top100_acc = sess.run(
                [model.softmax_pred, model.softmax_acc, model.top10_softmax_acc, model.top20_softmax_acc, model.top100_softmax_acc],
                {model.heads: head, model.relations: relation, model.tails: tail}
            )

            test_head, test_relation, test_tail = sess.run([test_h_batch, test_r_batch, test_t_batch])
            test_softmax_loss, test_nce_loss = sess.run([test_model.softmax_loss, test_model.nce_loss],
                                                        {test_h_batch: test_head, test_r_batch: test_relation,
                                                         test_t_batch: test_tail})

            test_softmax_pred, test_softmax_acc, test_softmax_top10_acc, test_softmax_top20_acc, test_softmax_top100_acc = sess.run(
                [test_model.softmax_pred, test_model.softmax_acc, test_model.top10_softmax_acc, test_model.top20_softmax_acc,
                 test_model.top100_softmax_acc],
                {test_model.heads: test_head, test_model.relations: test_relation, test_model.tails: test_tail}
            )

            print('==============[%d] %.4f %.4f\t %.4f %.4f==============' % (step, nce_loss, softmax_loss,
                                                                              test_nce_loss, test_softmax_loss))

            print('[softmax acc: %.3f]\t[top10 acc: %.3f]\t[top20 acc: %.3f]\t[top100 acc: %.3f]' % (
                softmax_acc, softmax_top10_acc, softmax_top20_acc, softmax_top100_acc))

            print('[t-softmax acc: %.3f]\t[t-top10 acc: %.3f]\t[t-top20 acc: %.3f]\t[t-top100 acc: %.3f]' % (
                test_softmax_acc, test_softmax_top10_acc, test_softmax_top20_acc, test_softmax_top100_acc))

        coord.request_stop()
        coord.join(threads)
