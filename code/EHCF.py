import numpy as np
import tensorflow as tf
import os
import pandas as pd
import scipy.sparse
import time
import sys

DATA_ROOT = '../data/Beibei'
f1 = open(os.path.join(DATA_ROOT, 'EHCF.txt'), 'w')


def load_data(csv_file):
    tp = pd.read_csv(csv_file, sep='\t')
    return tp


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


tp_test = load_data(os.path.join(DATA_ROOT, 'buy.test.txt'))
tp_train = load_data(os.path.join(DATA_ROOT, 'buy.train.txt'))
tp_view = load_data(os.path.join(DATA_ROOT, 'pv.csv'))
tp_cart = load_data(os.path.join(DATA_ROOT, 'cart.csv'))
tp_all = tp_train.append(tp_test)

# print tp_all

usercount, itemcount = get_count(tp_all, 'uid'), get_count(tp_all, 'sid')

n_users, n_items = usercount.shape[0], itemcount.shape[0]

print n_users, n_items


def _writeline_and_time(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    return time.time()


class EHCF:
    def __init__(self, user_num, item_num, embedding_size, max_item_view, max_item_cart, max_item_buy, b_num):
        self.user_num = user_num
        self.item_num = item_num
        self.embedding_size = embedding_size
        self.max_item_view = max_item_view
        self.max_item_cart = max_item_cart
        self.max_item_buy = max_item_buy
        self.weight = [0.1, 0.1, 0.1]
        self.lambda_bilinear = [1e-4, 0.0]
        self.b_num = b_num
        self.coefficient = [1.0/6, 4.0/6, 1.0/6]

    def _create_placeholders(self):
        self.input_u = tf.placeholder(tf.int32, [None, 1], name="input_uid")

        self.lable_view = tf.placeholder(tf.int32, [None, self.max_item_view], name="lable_view")
        self.lable_cart = tf.placeholder(tf.int32, [None, self.max_item_cart], name="lable_cart")
        self.lable_buy = tf.placeholder(tf.int32, [None, self.max_item_buy], name="lable_buy")

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def _create_variables(self):
        self.uidW = tf.Variable(tf.truncated_normal(shape=[self.user_num, self.embedding_size], mean=0.0,
                                                    stddev=0.01), dtype=tf.float32, name="uidWg")
        self.iidW = tf.Variable(tf.truncated_normal(shape=[self.item_num + 1, self.embedding_size], mean=0.0,
                                                    stddev=0.01), dtype=tf.float32, name="iidW")

        # view_para
        self.H_1 = tf.Variable(
            tf.random_uniform([self.embedding_size, 1],
                              minval=-tf.sqrt(3.0 / self.embedding_size), maxval=tf.sqrt(3.0 / self.embedding_size),
                              name='h_1', dtype=tf.float32))

        # view->cart
        self.W_vc = tf.Variable(
            tf.random_uniform(shape=[self.embedding_size, self.embedding_size],
                              minval=-tf.sqrt(3.0 / self.embedding_size), maxval=tf.sqrt(3.0 / self.embedding_size),
                              name='W_vc', dtype=tf.float32))
        self.R_vc = tf.Variable(
            tf.random_uniform([self.embedding_size, 1],
                              minval=-tf.sqrt(3.0 / self.embedding_size), maxval=tf.sqrt(3.0 / self.embedding_size),
                              name='R_vc', dtype=tf.float32))

        # view->buy
        self.W_vb = tf.Variable(
            tf.random_uniform(shape=[self.embedding_size, self.embedding_size],
                              minval=-tf.sqrt(3.0 / self.embedding_size), maxval=tf.sqrt(3.0 / self.embedding_size),
                              name='W_vb', dtype=tf.float32))
        self.R_vb = tf.Variable(
            tf.random_uniform([self.embedding_size, 1],
                              minval=-tf.sqrt(3.0 / self.embedding_size), maxval=tf.sqrt(3.0 / self.embedding_size),
                              name='R_vb', dtype=tf.float32))

        # cart->buy
        self.W_cb = tf.Variable(
            tf.random_uniform(shape=[self.embedding_size, self.embedding_size],
                              minval=-tf.sqrt(3.0 / self.embedding_size), maxval=tf.sqrt(3.0 / self.embedding_size),
                              name='W_cb', dtype=tf.float32))
        self.R_cb = tf.Variable(
            tf.random_uniform([self.embedding_size, 1],
                              minval=-tf.sqrt(3.0 / self.embedding_size), maxval=tf.sqrt(3.0 / self.embedding_size),
                              name='R_cb', dtype=tf.float32))

    def _create_inference(self):
        self.uid = tf.nn.embedding_lookup(self.uidW, self.input_u)
        self.uid = tf.reshape(self.uid, [-1, self.embedding_size])

        self.uid = tf.nn.dropout(self.uid, self.dropout_keep_prob)

        if self.b_num == 3:

            # predict view
            self.pos_view = tf.nn.embedding_lookup(self.iidW, self.lable_view)
            self.pos_num_view = tf.cast(tf.not_equal(self.lable_view, self.item_num), 'float32')
            self.pos_view = tf.einsum('ab,abc->abc', self.pos_num_view, self.pos_view)
            self.pos_rv = tf.einsum('ac,abc->abc', self.uid, self.pos_view)
            self.pos_rv = tf.einsum('ajk,kl->ajl', self.pos_rv, self.H_1)
            self.pos_rv = tf.reshape(self.pos_rv, [-1, self.max_item_view])

            # predict cart
            self.pos_cart = tf.nn.embedding_lookup(self.iidW, self.lable_cart)
            self.pos_num_cart = tf.cast(tf.not_equal(self.lable_cart, self.item_num), 'float32')
            self.pos_cart = tf.einsum('ab,abc->abc', self.pos_num_cart, self.pos_cart)
            self.pos_rc = tf.einsum('ac,abc->abc', self.uid, self.pos_cart)

            self.H_2 = tf.matmul(self.W_vc, self.H_1) + self.R_vc

            self.pos_rc = tf.einsum('ajk,kl->ajl', self.pos_rc, self.H_2)
            self.pos_rc = tf.reshape(self.pos_rc, [-1, self.max_item_cart])

            # predict buy
            self.pos_buy = tf.nn.embedding_lookup(self.iidW, self.lable_buy)
            self.pos_num_buy = tf.cast(tf.not_equal(self.lable_buy, self.item_num), 'float32')
            self.pos_buy = tf.einsum('ab,abc->abc', self.pos_num_buy, self.pos_buy)
            self.pos_rb = tf.einsum('ac,abc->abc', self.uid, self.pos_buy)

            self.H_3 = tf.matmul(self.W_vb, self.H_1) + self.R_vb \
                       + tf.matmul(self.W_cb, self.H_2) + self.R_cb

            self.pos_rb = tf.einsum('ajk,kl->ajl', self.pos_rb, self.H_3)
            self.pos_rb = tf.reshape(self.pos_rb, [-1, self.max_item_buy])

        elif self.b_num == 2:

            # predict view
            self.pos_view = tf.nn.embedding_lookup(self.iidW, self.lable_view)
            self.pos_num_view = tf.cast(tf.not_equal(self.lable_view, self.item_num), 'float32')
            self.pos_view = tf.einsum('ab,abc->abc', self.pos_num_view, self.pos_view)
            self.pos_rv = tf.einsum('ac,abc->abc', self.uid, self.pos_view)
            self.pos_rv = tf.einsum('ajk,kl->ajl', self.pos_rv, self.H_1)
            self.pos_rv = tf.reshape(self.pos_rv, [-1, self.max_item_view])

            # predict buy
            self.pos_buy = tf.nn.embedding_lookup(self.iidW, self.lable_buy)
            self.pos_num_buy = tf.cast(tf.not_equal(self.lable_buy, self.item_num), 'float32')
            self.pos_buy = tf.einsum('ab,abc->abc', self.pos_num_buy, self.pos_buy)
            self.pos_rb = tf.einsum('ac,abc->abc', self.uid, self.pos_buy)

            self.H_3 = tf.matmul(self.W_vb, self.H_1) + self.R_vb

            self.pos_rb = tf.einsum('ajk,kl->ajl', self.pos_rb, self.H_3)
            self.pos_rb = tf.reshape(self.pos_rb, [-1, self.max_item_buy])

    def _pre(self):
        dot = tf.einsum('ac,bc->abc', self.uid, self.iidW)
        pre = tf.einsum('ajk,kl->ajl', dot, self.H_3)
        pre = tf.reshape(pre, [-1, self.item_num + 1])
        return pre

    def _create_loss(self):

        temp = tf.reduce_sum(tf.einsum('ab,ac->abc', self.iidW, self.iidW), 0) \
               * tf.reduce_sum(tf.einsum('ab,ac->abc', self.uid, self.uid), 0)

        self.l2_loss0 = tf.nn.l2_loss(self.W_vc) + tf.nn.l2_loss(self.W_vb) + tf.nn.l2_loss(self.W_cb)
        self.l2_loss1 = tf.nn.l2_loss(self.iidW)

        if self.b_num == 3:
            self.loss1 = self.weight[0] * tf.reduce_sum(
                tf.reduce_sum(temp * tf.matmul(self.H_1, self.H_1, transpose_b=True), 0), 0)
            self.loss1 += tf.reduce_sum((1.0 - self.weight[0]) * tf.square(self.pos_rv) - 2.0 * self.pos_rv)

            self.loss2 = self.weight[1] * tf.reduce_sum(
                tf.reduce_sum(temp * tf.matmul(self.H_2, self.H_2, transpose_b=True), 0), 0)
            self.loss2 += tf.reduce_sum((1.0 - self.weight[1]) * tf.square(self.pos_rc) - 2.0 * self.pos_rc)

            self.loss3 = self.weight[2] * tf.reduce_sum(
                tf.reduce_sum(temp * tf.matmul(self.H_3, self.H_3, transpose_b=True), 0), 0)
            self.loss3 += tf.reduce_sum((1.0 - self.weight[2]) * tf.square(self.pos_rb) - 2.0 * self.pos_rb)

            self.loss = self.coefficient[0] * self.loss1 + self.coefficient[1] * self.loss2 + self.coefficient[
                                                                                                  2] * self.loss3 \
                        + self.lambda_bilinear[0] * self.l2_loss0 \
                        + self.lambda_bilinear[1] * self.l2_loss1
        elif self.b_num == 2:
            self.loss1 = self.weight[0] * tf.reduce_sum(
                tf.reduce_sum(temp * tf.matmul(self.H_1, self.H_1, transpose_b=True), 0), 0)
            self.loss1 += tf.reduce_sum((1.0 - self.weight[0]) * tf.square(self.pos_rv) - 2.0 * self.pos_rv)

            self.loss3 = self.weight[2] * tf.reduce_sum(
                tf.reduce_sum(temp * tf.matmul(self.H_3, self.H_3, transpose_b=True), 0), 0)
            self.loss3 += tf.reduce_sum((1.0 - self.weight[2]) * tf.square(self.pos_rb) - 2.0 * self.pos_rb)

            self.loss = self.coefficient[0] * self.loss1 + self.coefficient[2] * self.loss3 \
                        + self.lambda_bilinear[0] * self.l2_loss0 \
                        + self.lambda_bilinear[1] * self.l2_loss1

        self.reg_loss = self.lambda_bilinear[0] * self.l2_loss0 \
                        + self.lambda_bilinear[1] * self.l2_loss1

    def _build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self.pre = self._pre()


def train_step1(u_batch, view_batch, cart_batch, buy_batch):
    """
    A single training step
    """
    feed_dict = {
        deep.input_u: u_batch,
        deep.lable_view: view_batch,
        deep.lable_cart: cart_batch,
        deep.lable_buy: buy_batch,
        deep.dropout_keep_prob: 0.5,
    }
    _, loss, loss1, loss2 = sess.run(
        [train_op1, deep.loss, deep.loss1, deep.reg_loss],
        feed_dict)
    return loss, loss1, loss2

def dev_cold(u_train,i_train,tset, train_m, test_m):

    recall100=[[],[],[],[],[]]

    ndcg100=[[],[],[],[],[]]

    user_te=[[],[],[],[],[]]

    trset = {}
    for i in range(len(u_train)):
        if trset.has_key(u_train[i]):
            trset[u_train[i]].append(i_train[i])
        else:
            trset[u_train[i]] = [i_train[i]]

    for i in tset.keys():
        if len(trset[i])<9:
            user_te[0].append(i)
        elif len(trset[i])<13:
            user_te[1].append(i)
        elif len(trset[i])<17:
            user_te[2].append(i)
        elif len(trset[i])<20:
            user_te[3].append(i)
        else:
            user_te[4].append(i)
    for l in range(len(user_te)):
        u=np.array(user_te[l])
        user_te2=u[:,np.newaxis]
        ll = int(len(u) / 128) + 1

        for batch_num in range(ll):
            start_index = batch_num * 128
            end_index = min((batch_num + 1) * 128, len(u))
            u_batch = user_te2[start_index:end_index]

            batch_users = end_index - start_index

            feed_dict = {
                deep.input_u: u_batch,
                deep.dropout_keep_prob: 1.0,
            }

            pre = sess.run(
                deep.pre, feed_dict)

            u_b = u[start_index:end_index]

            pre = np.array(pre)
            pre = np.delete(pre, -1, axis=1)

            idx = np.zeros_like(pre, dtype=bool)
            idx[train_m[u_b].nonzero()] = True
            pre[idx] = -np.inf

            idx_topk_part = np.argpartition(-pre, 100, 1)
            pre_bin = np.zeros_like(pre, dtype=bool)
            pre_bin[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :100]] = True
            true_bin = np.zeros_like(pre, dtype=bool)
            true_bin[test_m[u_b].nonzero()] = True

            tmp = (np.logical_and(true_bin, pre_bin).sum(axis=1)).astype(np.float32)
            recall=tmp / np.minimum(100, true_bin.sum(axis=1))


            idx_topk_part = np.argpartition(-pre, 100, 1)

            topk_part = pre[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :100]]
            idx_part = np.argsort(-topk_part, axis=1)
            idx_topk = idx_topk_part[np.arange(end_index - start_index)[:, np.newaxis], idx_part]

            tp = np.log(2) / np.log(np.arange(2, 100 + 2))

            test_batch = test_m[u_b]
            DCG = (test_batch[np.arange(batch_users)[:, np.newaxis],
                              idx_topk].toarray() * tp).sum(axis=1)

            IDCG = np.array([(tp[:min(n, 100)]).sum()
                             for n in test_batch.getnnz(axis=1)])
            ndcg=DCG / IDCG
            recall100[l].append(recall)
            ndcg100[l].append(ndcg)

    for l in range(len(recall100)):
        recall100[l]=np.hstack(recall100[l])
        ndcg100[l]=np.hstack(ndcg100[l])

    for l in range(len(recall100)):
        print np.mean(recall100[l]),np.mean(ndcg100[l])


def dev_step(tset, train_m, test_m):
    """
    Evaluates model on a dev set

    """
    user_te = np.array(tset.keys())
    user_te2 = user_te[:, np.newaxis]

    ll = int(len(user_te) / 128) + 1

    recall50 = []
    recall100 = []
    recall200 = []
    ndcg50 = []
    ndcg100 = []
    ndcg200 = []

    for batch_num in range(ll):

        start_index = batch_num * 128
        end_index = min((batch_num + 1) * 128, len(user_te))
        u_batch = user_te2[start_index:end_index]

        batch_users = end_index - start_index

        feed_dict = {
            deep.input_u: u_batch,
            deep.dropout_keep_prob: 1.0,
        }

        pre = sess.run(
            deep.pre, feed_dict)

        u_b = user_te[start_index:end_index]

        pre = np.array(pre)
        pre = np.delete(pre, -1, axis=1)

        idx = np.zeros_like(pre, dtype=bool)
        idx[train_m[u_b].nonzero()] = True
        pre[idx] = -np.inf

        # recall

        recall = []

        for kj in [10,50, 100]:
            idx_topk_part = np.argpartition(-pre, kj, 1)

            # print pre[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :kj]]

            # print idx_topk_part
            pre_bin = np.zeros_like(pre, dtype=bool)
            pre_bin[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :kj]] = True

            # print pre_bin

            true_bin = np.zeros_like(pre, dtype=bool)
            true_bin[test_m[u_b].nonzero()] = True

            tmp = (np.logical_and(true_bin, pre_bin).sum(axis=1)).astype(np.float32)
            recall.append(tmp / np.minimum(kj, true_bin.sum(axis=1)))
            # print tmp

        # ndcg10
        ndcg = []

        for kj in [10, 50, 100]:
            idx_topk_part = np.argpartition(-pre, kj, 1)

            topk_part = pre[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :kj]]
            idx_part = np.argsort(-topk_part, axis=1)
            idx_topk = idx_topk_part[np.arange(end_index - start_index)[:, np.newaxis], idx_part]

            tp = np.log(2) / np.log(np.arange(2, kj + 2))

            test_batch = test_m[u_b]
            DCG = (test_batch[np.arange(batch_users)[:, np.newaxis],
                              idx_topk].toarray() * tp).sum(axis=1)

            IDCG = np.array([(tp[:min(n, kj)]).sum()
                             for n in test_batch.getnnz(axis=1)])
            ndcg.append(DCG / IDCG)

        recall50.append(recall[0])
        recall100.append(recall[1])
        recall200.append(recall[2])
        ndcg50.append(ndcg[0])
        ndcg100.append(ndcg[1])
        ndcg200.append(ndcg[2])

    recall50 = np.hstack(recall50)
    recall100 = np.hstack(recall100)
    recall200 = np.hstack(recall200)
    ndcg50 = np.hstack(ndcg50)
    ndcg100 = np.hstack(ndcg100)
    ndcg200 = np.hstack(ndcg200)

    print np.mean(recall50), np.mean(ndcg50)
    print np.mean(recall100), np.mean(ndcg100)
    print np.mean(recall200), np.mean(ndcg200)
    f1.write(str(np.mean(recall100)) + ' ' + str(np.mean(ndcg100)) + '\n')
    f1.flush()

    return loss


def get_train_instances1(view_lable, cart_lable, buy_lable):
    user_train, view_item, cart_item, buy_item = [], [], [], []

    for i in buy_lable.keys():
        user_train.append(i)
        buy_item.append(buy_lable[i])
        if not view_lable.has_key(i):
            view_item.append([n_items] * max_item_view)
        else:
            view_item.append(view_lable[i])

        if not cart_lable.has_key(i):
            cart_item.append([n_items] * max_item_cart)
        else:
            cart_item.append(cart_lable[i])

    user_train = np.array(user_train)
    view_item = np.array(view_item)
    cart_item = np.array(cart_item)
    buy_item = np.array(buy_item)
    user_train = user_train[:, np.newaxis]
    return user_train, view_item, cart_item, buy_item


def get_lables(u_temp, i_temp, k=1.0):
    lable_set = {}
    max_item = 0
    for i in range(len(u_temp)):
        if lable_set.has_key(u_temp[i]):
            lable_set[u_temp[i]].append(i_temp[i])
        else:
            lable_set[u_temp[i]] = [i_temp[i]]
    for i in lable_set.keys():
        if len(lable_set[i]) > max_item:
            max_item = len(lable_set[i])

    max_item = int(max_item * k)

    for i in lable_set.keys():
        while len(lable_set[i]) < max_item:
            lable_set[i].append(n_items)
        if len(lable_set[i]) > max_item:
            lable_set[i] = lable_set[i][0:max_item]
    return max_item, lable_set


if __name__ == '__main__':
    np.random.seed(2019)
    random_seed = 2019

    u_train = np.array(tp_train['uid'], dtype=np.int32)
    i_train = np.array(tp_train['sid'], dtype=np.int32)
    u_test = np.array(tp_test['uid'], dtype=np.int32)
    i_test = np.array(tp_test['sid'], dtype=np.int32)

    u_view = np.array(tp_view['uid'], dtype=np.int32)
    i_view = np.array(tp_view['sid'], dtype=np.int32)

    u_cart = np.array(tp_cart['uid'], dtype=np.int32)
    i_cart = np.array(tp_cart['sid'], dtype=np.int32)

    tset = {}

    count = np.ones(len(u_train))
    train_m = scipy.sparse.csr_matrix((count, (u_train, i_train)), dtype=np.int16, shape=(n_users, n_items))
    count = np.ones(len(u_test))
    test_m = scipy.sparse.csr_matrix((count, (u_test, i_test)), dtype=np.int16, shape=(n_users, n_items))

    for i in range(len(u_test)):
        if tset.has_key(u_test[i]):
            tset[u_test[i]].append(i_test[i])
        else:
            tset[u_test[i]] = [i_test[i]]

    max_item_view, view_lable = get_lables(u_view, i_view)
    max_item_cart, cart_lable = get_lables(u_cart, i_cart)
    max_item_buy, buy_lable = get_lables(u_train, i_train)

    batch_size = 256
    with tf.Graph().as_default():
        tf.set_random_seed(random_seed)
        session_conf = tf.ConfigProto()
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            deep = EHCF(n_users, n_items, 64, max_item_view, max_item_cart, max_item_buy, b_num=3)
            deep._build_graph()
            # optimizer1 = tf.train.AdamOptimizer(learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(deep.loss)
            optimizer1 = tf.train.AdagradOptimizer(learning_rate=0.05, initial_accumulator_value=1e-8).minimize(
                deep.loss)
            # optimizer1=tf.train.MomentumOptimizer(learning_rate=0.02, momentum=0.95).minimize(deep.loss1)
            train_op1 = optimizer1  # .apply_gradients(grads_and_vars, global_step=global_step)

            sess.run(tf.global_variables_initializer())

            user_train1, view_item1, cart_item1, buy_item1 = get_train_instances1(view_lable, cart_lable, buy_lable)

            for epoch in range(505):
                print epoch
                start_t = _writeline_and_time('\tUpdating...')

                shuffle_indices = np.random.permutation(np.arange(len(user_train1)))
                user_train1 = user_train1[shuffle_indices]
                view_item1 = view_item1[shuffle_indices]
                cart_item1 = cart_item1[shuffle_indices]
                buy_item1 = buy_item1[shuffle_indices]

                ll = int(len(user_train1) / batch_size)
                loss = [0.0, 0.0, 0.0]

                for batch_num in range(ll):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, len(user_train1))

                    u_batch = user_train1[start_index:end_index]
                    v_batch = view_item1[start_index:end_index]
                    c_batch = cart_item1[start_index:end_index]
                    b_batch = buy_item1[start_index:end_index]

                    loss1, loss2, loss3 = train_step1(u_batch, v_batch, c_batch, b_batch)
                    loss[0] += loss1
                    loss[1] += loss2
                    loss[2] += loss3
                print('\r\tUpdating: time=%.2f'
                      % (time.time() - start_t))
                print 'loss,loss_no_reg,loss_reg ', loss[0] / ll, loss[1] / ll, loss[2] / ll

                if epoch < 500:
                    if epoch % 10 == 0:
                        dev_step(tset, train_m, test_m)
                        dev_cold(u_train,i_train, tset, train_m, test_m)
                if epoch >= 500:
                    dev_step(tset, train_m, test_m)
                    dev_cold(u_train,i_train, tset, train_m, test_m)























