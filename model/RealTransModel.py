import tensorflow as tf


class RealTransModel:
    def __init__(self, config, is_training=True):
        voc_cnt = config.VOC_CNT
        rel_cnt = config.REL_CNT
        batch_size = config.BATCH_SIZE
        embedding_size = config.EMBEDDING_SIZE
        offset = config.OFFSET

        self.heads = tf.placeholder(dtype=tf.int32, shape=[batch_size], name="heads in a batch")
        self.relations = tf.placeholder(dtype=tf.int32, shape=[batch_size], name="relations in a batch")
        self.tails = tf.placeholder(dtype=tf.int32, shape=[batch_size], name="tails in a batch")

        self.bi_direction = tf.placeholder(dtype=bool)
        self.include_rel = tf.placeholder(dtype=bool)

        self.neg_heads = tf.placeholder(dtype=tf.int32, shape=[batch_size], name="neg_heads in corr batch entries")
        self.neg_relations = tf.placeholder(dtype=tf.int32, shape=[batch_size], name="neg_rels in corr batch entries")
        self.neg_tails = tf.placeholder(dtype=tf.int32, shape=[batch_size], name="neg_tails in corr batch entries")

        with tf.device('/cpu:0'):
            self.voc_embeddings = tf.get_variable(name="voc emb", shape=[voc_cnt, embedding_size], dtype=tf.float32)
            self.rel_embeddings = tf.get_variable(name="rel emb", shape=[rel_cnt, embedding_size], dtype=tf.float32)

            emb_heads = tf.nn.embedding_lookup(self.voc_embeddings, self.heads, name="heads emb in batch")
            emb_rels = tf.nn.embedding_lookup(self.rel_embeddings, self.relations, name="rels emb in batch")
            emb_tails = tf.nn.embedding_lookup(self.voc_embeddings, self.tails, name="tails emb in batch")

            # if self.bi_direction:
            #     emb_neg_heads = tf.nn.embedding_lookup(self.voc_embeddings, self.neg_heads, name="negheads emb in batch")
            # if self.include_rel:
            #     emb_neg_rels = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_relations, name="negrels emb in batch")
            emb_neg_tails = tf.nn.embedding_lookup(self.voc_embeddings, self.neg_tails, name="negtails emb in batch")

        trans_weights = tf.get_variable(name="trans_weight", shape=[embedding_size, embedding_size], dtype=tf.float32, trainable=True)
        trans_biases = tf.get_variable(name="trans_biases", shape=[embedding_size], dtype=tf.float32, trainable=True)

        trans_tail = tf.add(
            tf.matmul(
                tf.add(emb_heads, emb_rels),
                trans_weights
            ),
            trans_biases
        )

        pos = tf.reduce_sum((trans_tail - emb_tails) ** 2, 1, keep_dims=True)
        neg = tf.reduce_sum((trans_tail - emb_neg_tails) ** 2, 1, keep_dims=True)

        #
        # if not self.bi_direction and not self.include_rel:
        loss = tf.reduce_sum(tf.maximum(pos - neg + offset, 0))

        self.loss = loss

        if is_training:
            global_step =tf.contrib.framework.get_or_create_global_step()
            learning_rate = tf.train.exponential_decay(
                learning_rate=config.base_learning_rate,
                global_step=global_step,
                decay_steps=300,
                decay_rate=0.98
            )
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.loss)


class TransNSigmoidModel:
    def __init__(self, config, is_training=True):
        voc_cnt = config.VOC_CNT
        rel_cnt = config.REL_CNT
        batch_size = config.BATCH_SIZE
        embedding_size = config.EMBEDDING_SIZE
        num_sampled = config.NUM_SAMPLED

        self.heads = tf.placeholder(dtype=tf.int32, shape=[batch_size], name="heads in a batch")
        self.relations = tf.placeholder(dtype=tf.int32, shape=[batch_size], name="relations in a batch")
        self.tails = tf.placeholder(dtype=tf.int32, shape=[batch_size], name="tails in a batch")

        # self.bi_direction = tf.placeholder(dtype=bool)
        # self.include_rel = tf.placeholder(dtype=bool)

        with tf.device('/cpu:0'):
            self.voc_embeddings = tf.get_variable(name="voc emb", shape=[voc_cnt, embedding_size], dtype=tf.float32)
            self.rel_embeddings = tf.get_variable(name="rel emb", shape=[rel_cnt, embedding_size], dtype=tf.float32)

            emb_heads = tf.nn.embedding_lookup(params=self.voc_embeddings, ids=self.heads, name="heads emb in batch")
            emb_rels = tf.nn.embedding_lookup(params=self.rel_embeddings, ids=self.relations, name="rels emb in batch")

        trans_weights = tf.get_variable(name="trans_weight", shape=[embedding_size, embedding_size], dtype=tf.float32,
                                        trainable=True)
        trans_biases = tf.get_variable(name="trans_biases", shape=[embedding_size], dtype=tf.float32, trainable=True)

        trans_tail = tf.add(
            tf.matmul(
                tf.add(emb_heads, emb_rels),
                trans_weights
            ),
            trans_biases
        )

        # emb_tails_should_be = tf.add(emb_heads, emb_rels)

        nce_weights = tf.get_variable(name="nce_weight", shape=[voc_cnt, embedding_size], dtype=tf.float32)
        nce_biases = tf.get_variable(name="nce_biases", shape=[voc_cnt], dtype=tf.float32)

        softmax_weights = tf.get_variable(name="softmax_weight", shape=[embedding_size, voc_cnt], dtype=tf.float32)
        softmax_biases = tf.get_variable(name="softmax_biase", shape=[voc_cnt], dtype=tf.float32)
        softmax_logits = tf.add(tf.matmul(trans_tail, softmax_weights), softmax_biases)

        self.softmax_pred = tf.argmax(tf.nn.softmax(softmax_logits), axis=-1)
        softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(tensor=self.tails, shape=[batch_size, 1]),
            logits=softmax_logits,
        )

        self.softmax_loss = tf.reduce_mean(softmax_loss)

        self.nce_loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=tf.reshape(tensor=self.tails, shape=[batch_size, 1]),
                inputs=trans_tail,
                num_sampled=num_sampled,
                num_classes=voc_cnt
            )
        )

        correct_prediction = tf.equal(x=tf.cast(tf.argmax(input=softmax_logits, axis=-1), tf.int32),
                                      y=self.tails)
        top10_correct_prediction = tf.nn.in_top_k(predictions=softmax_logits, targets=self.tails, k=10)
        top20_correct_prediction = tf.nn.in_top_k(predictions=softmax_logits, targets=self.tails, k=20)
        top100_correct_prediction = tf.nn.in_top_k(predictions=softmax_logits, targets=self.tails, k=100)

        self.softmax_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.top10_softmax_acc = tf.reduce_mean(tf.cast(top10_correct_prediction, tf.float32))
        self.top20_softmax_acc = tf.reduce_mean(tf.cast(top20_correct_prediction, tf.float32))
        self.top100_softmax_acc = tf.reduce_mean(tf.cast(top100_correct_prediction, tf.float32))

        if is_training:
            global_step = tf.contrib.framework.get_or_create_global_step()
            learning_rate = tf.train.exponential_decay(
                config.BASE_LEARNING_RATE,
                global_step,
                300,
                0.98
            )
            self.nce_train_op = tf.train.GradientDescentOptimizer(self.nce_loss).minimize(self.nce_loss)
            self.softmax_train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.softmax_loss)


