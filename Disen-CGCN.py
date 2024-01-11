'''
Disen-CGCN
'''

import os
import sys
import random
import threading
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from utility.helper import *
from utility.batch_test import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

cpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'CPU']


class DisenCGCN(object):
    def __init__(self, data_config, pretrain_data, pretrain_data2, pretrain_data3):
        # argument settings
        self.model_type = 'Disen-CGCN'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.dataset = args.data_path
        self.pretrain_data = pretrain_data
        self.pretrain_data2 = pretrain_data2
        self.pretrain_data3 = pretrain_data3

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_fold = 100
        self.norm_adj = data_config['norm_adj']
        self.norm_adj2 = data_config['norm_adj2']
        self.norm_adj3 = data_config['norm_adj3']

        self.n_nonzero_elems = self.norm_adj.count_nonzero()
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.meta_emb_dim = args.meta_embed_size
        self.batch_size = args.batch_size
        self.attention_enlarge_num = args.attention_enlarge_num
        self.weight_size = eval(args.layer_size2)
        self.weight_size2 = eval(args.layer_size3)
        self.weight_size3 = eval(args.layer_size4)

        self.n_layers = len(self.weight_size)
        self.n_layers2 = len(self.weight_size2)
        self.n_layers3 = len(self.weight_size3)


        self.cor_flag3 = args.cor_flag3
        self.T_items = 11903
        self.B_items = self.n_items
        self.n_factors = args.n_factors

        self.reg_c = eval(args.reg_c)
        self.decay_c = self.reg_c[0]

        self.reg_W_l2 = eval(args.reg_W_l2)
        self.decay_w = self.reg_W_l2[0]

        self.reg_W_PFT_l2 = eval(args.reg_W_PFT_l2)
        self.decay_w_pft = self.reg_W_PFT_l2[0]

        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.log_dir = self.create_model_str()
        self.verbose = args.verbose
        self.Ks = eval(args.Ks)

        '''
        *********************************************************
        Create Placeholder for Input Tmall & Dropout.
        '''
        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])
        with tf.name_scope('TRAIN_LOSS'):
            self.train_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_loss', self.train_loss)
            self.train_mf_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_mf_loss', self.train_mf_loss)
            self.train_emb_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_emb_loss', self.train_emb_loss)
            self.train_reg_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_reg_loss', self.train_reg_loss)
            self.train_cor_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_cor_loss', self.train_cor_loss)
        self.merged_train_loss = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TRAIN_LOSS'))

        with tf.name_scope('TRAIN_ACC'):
            self.train_rec_first = tf.placeholder(tf.float32)
            # record for top(Ks[0])
            tf.summary.scalar('train_rec_first', self.train_rec_first)  # recall
            self.train_rec_last = tf.placeholder(tf.float32)
            # record for top(Ks[-1])
            tf.summary.scalar('train_rec_last', self.train_rec_last)
            self.train_ndcg_first = tf.placeholder(tf.float32)
            tf.summary.scalar('train_ndcg_first', self.train_ndcg_first)
            self.train_ndcg_last = tf.placeholder(tf.float32)
            tf.summary.scalar('train_ndcg_last', self.train_ndcg_last)
        self.merged_train_acc = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TRAIN_ACC'))

        with tf.name_scope('TEST_LOSS'):
            self.test_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_loss', self.test_loss)
            self.test_mf_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_mf_loss', self.test_mf_loss)
            self.test_emb_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_emb_loss', self.test_emb_loss)
            self.test_reg_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_reg_loss', self.test_reg_loss)
            self.test_cor_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_cor_loss', self.test_cor_loss)
        self.merged_test_loss = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TEST_LOSS'))

        with tf.name_scope('TEST_ACC'):
            self.test_rec_first = tf.placeholder(tf.float32)
            tf.summary.scalar('test_rec_first', self.test_rec_first)
            self.test_rec_last = tf.placeholder(tf.float32)
            tf.summary.scalar('test_rec_last', self.test_rec_last)
            self.test_ndcg_first = tf.placeholder(tf.float32)
            tf.summary.scalar('test_ndcg_first', self.test_ndcg_first)
            self.test_ndcg_last = tf.placeholder(tf.float32)
            tf.summary.scalar('test_ndcg_last', self.test_ndcg_last)
        self.merged_test_acc = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TEST_ACC'))
        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights_one = self._init_weights()

        '''
        *********************************************************
        cascading LightGCN blocks
        '''

        if self.alg_type in ['Disen-CGCN']:
            'The first behavior'
            self.cor_loss03 = self.create_cor_loss3(self.weights_one['user_embedding3'], self.weights_one['item_embedding3'], self.users, self.pos_items)

            self.ua_embeddings3, self.ia_embeddings3, self.one_layer_gcn_users3, self.one_layer_gcn_items3 = self._create_mbcgcn_embed3(self.weights_one['user_embedding3'], self.weights_one['item_embedding3'])  # The first behavior

            self.cor_loss3 = self.create_cor_loss3(self.ua_embeddings3, self.ia_embeddings3, self.users, self.pos_items)

            self.ua_embeddings11, self.ia_embeddings11 = self.personal_fea_trans3(self.one_layer_gcn_users3, self.one_layer_gcn_items3, self.ua_embeddings3, self.ia_embeddings3)

            'The second behavior'
            self.ua_embeddings2, self.ia_embeddings2, self.one_layer_gcn_users2, self.one_layer_gcn_items2 = self._create_mbcgcn_embed2(self.ua_embeddings11, self.ia_embeddings11)

            self.ua_embeddings22, self.ia_embeddings22 = self.personal_fea_trans2(self.one_layer_gcn_users2, self.one_layer_gcn_items2, self.ua_embeddings2, self.ia_embeddings2)

            'The last behavior'
            self.ua_embeddings1, self.ia_embeddings1 = self._create_mbcgcn_embed(self.ua_embeddings22, self.ia_embeddings22)


            self.pos_scores = self.compute_score(self.ua_embeddings3, self.ia_embeddings3,
                                                                         self.ua_embeddings2, self.ia_embeddings2,
                                                                         self.ua_embeddings1, self.ia_embeddings1,
                                                                         self.users, self.pos_items)

            self.neg_scores = self.compute_score(self.ua_embeddings3, self.ia_embeddings3,
                                                                         self.ua_embeddings2, self.ia_embeddings2,
                                                                         self.ua_embeddings1, self.ia_embeddings1,
                                                                         self.users, self.neg_items)

        """
        *********************************************************
        regularizer
        """
        self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights_one['user_embedding3'], self.users)
        self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights_one['item_embedding3'], self.pos_items)
        self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights_one['item_embedding3'], self.neg_items)


        """
        *********************************************************
        Inference for the testing phase.
        """
        if self.dataset == 'Tmall/':
            self.batch_ratings = tf.reshape(self.pos_scores, [-1, self.T_items])
        else:
            self.batch_ratings = tf.reshape(self.pos_scores, [-1, self.B_items])
        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """

        regularizer = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(
            self.pos_i_g_embeddings_pre) + tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
        regularizer = regularizer / self.batch_size
        self.mf_loss = tf.reduce_mean(tf.nn.softplus(-(self.pos_scores - self.neg_scores)))  # BPR
        self.emb_loss = self.decay * regularizer
        self.reg_loss = tf.constant(0.0, tf.float32, [1])

        """
        *********************************************************
        Cor loss.
        """
        self.cor_loss = self.decay_c * (self.cor_loss03 + self.cor_loss3)


        self.loss = self.mf_loss + self.emb_loss + self.cor_loss



        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def create_model_str(self):
        log_dir = '/' + self.alg_type + '/layers_' + str(self.n_layers) + '/dim_' + str(
            self.emb_dim)  # /MBCGCN/layers_3/dim_64
        log_dir += '/' + args.dataset + '/lr_' + str(self.lr) + '/reg_' + str(self.decay)
        return log_dir

    def _init_weights(self):
        all_weights_one = dict()
        initializer = tf.random_normal_initializer(stddev=0.01)  # tf.contrib.layers.xavier_initializer()
        if self.pretrain_data is None:
            all_weights_one['user_embedding1'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                             name='user_embedding1')
            all_weights_one['item_embedding1'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                             name='item_embedding1')
            all_weights_one['user_embedding2'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                             name='user_embedding2')
            all_weights_one['item_embedding2'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                             name='item_embedding2')
            all_weights_one['user_embedding3'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                             name='user_embedding3')
            all_weights_one['item_embedding3'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                             name='item_embedding3')
            print('using random initialization')  # print('using xavier initialization')
        else:
            all_weights_one['user_embedding1'] = tf.Variable(initial_value=self.pretrain_data3['user_embed'],
                                                             trainable=True,
                                                             name='user_embedding1', dtype=tf.float32)
            all_weights_one['item_embedding1'] = tf.Variable(initial_value=self.pretrain_data3['item_embed'],
                                                             trainable=True,
                                                             name='item_embedding1', dtype=tf.float32)
            all_weights_one['user_embedding2'] = tf.Variable(initial_value=self.pretrain_data3['user_embed'],
                                                             trainable=True,
                                                             name='user_embedding2', dtype=tf.float32)
            all_weights_one['item_embedding2'] = tf.Variable(initial_value=self.pretrain_data3['item_embed'],
                                                             trainable=True,
                                                             name='item_embedding2', dtype=tf.float32)
            all_weights_one['user_embedding3'] = tf.Variable(initial_value=self.pretrain_data3['user_embed'],
                                                             trainable=True,
                                                             name='user_embedding3', dtype=tf.float32)
            all_weights_one['item_embedding3'] = tf.Variable(initial_value=self.pretrain_data3['item_embed'],
                                                             trainable=True,
                                                             name='item_embedding3', dtype=tf.float32)
            print(all_weights_one)
            print('using pretrained initialization')


        return all_weights_one



    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat  # [[len],[],...,[]]

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _create_mbcgcn_embed(self, user_embedding, item_embedding):  # The last behavior
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)
        ego_embeddings1 = tf.concat([user_embedding, item_embedding], axis=0)
        all_embeddings = [ego_embeddings1]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings1))

            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings1 = side_embeddings
            all_embeddings += [ego_embeddings1]
        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def compute_score(self, ub3, i3, ub2, i2, ub1, i1, user, item):

        #look_up_table
        user3 = tf.nn.embedding_lookup(ub3, user)
        item3 = tf.nn.embedding_lookup(i3, item)
        user2 = tf.nn.embedding_lookup(ub2, user)
        item2 = tf.nn.embedding_lookup(i2, item)
        user1 = tf.nn.embedding_lookup(ub1, user)
        item1 = tf.nn.embedding_lookup(i1, item)

        #split
        u3_factor = tf.split(user3, self.n_factors, 1)
        i3_factor = tf.split(item3, self.n_factors, 1)
        u2_factor = tf.split(user2, self.n_factors, 1)
        i2_factor = tf.split(item2, self.n_factors, 1)
        u1_factor = tf.split(user1, self.n_factors, 1)
        i1_factor = tf.split(item1, self.n_factors, 1)


        #U3
        attention_weight3 = []
        for i in range(0, self.n_factors):
            input = tf.nn.l2_normalize(tf.concat([u3_factor[i], i3_factor[i]], 1), 1)
            output_h_u3 = tf.layers.dense(inputs=input, units=self.emb_dim, activation=tf.nn.tanh,
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(self.decay_w),
                                          name="attention3", reuse=tf.AUTO_REUSE)
            output_u3 = tf.layers.dense(inputs=output_h_u3, units=1, activation=None,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.decay_w),
                                        use_bias=None, name="attention_h3", reuse=tf.AUTO_REUSE)
            attention_weight3.append(output_u3)

        attention_3 = tf.concat(attention_weight3, 1)

        atte_3 = self.attention_enlarge_num *  tf.nn.softmax(attention_3, 1)

        u3_user_embbedding = []
        for i in range(0, self.n_factors):

            u3_factors = tf.expand_dims(atte_3[:, i], 1) * u3_factor[i]
            u3_user_embbedding.append(u3_factors)

        u3_user_embbedding = tf.concat(u3_user_embbedding, 1)


        #u2
        attention_weight2 = []
        for i in range(0, self.n_factors):
            input = tf.nn.l2_normalize(tf.concat([u2_factor[i], i2_factor[i]], 1), 1)
            output_h_u2 = tf.layers.dense(inputs=input, units=self.emb_dim, activation=tf.nn.tanh,
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(self.decay_w),
                                          name="attention2", reuse=tf.AUTO_REUSE)
            output_u2 = tf.layers.dense(inputs=output_h_u2, units=1, activation=None,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.decay_w),
                                        use_bias=None, name="attention_h2", reuse=tf.AUTO_REUSE)
            attention_weight2.append(output_u2)

        attention_2 = tf.concat(attention_weight2, 1)

        atte_2 = self.attention_enlarge_num * tf.nn.softmax(attention_2, 1)

        u2_user_embbedding = []
        for i in range(0, self.n_factors):
            u2_factors = tf.expand_dims(atte_2[:, i], 1) * u2_factor[i]
            u2_user_embbedding.append(u2_factors)

        u2_user_embbedding = tf.concat(u2_user_embbedding, 1)

        # u1
        attention_weight1 = []
        for i in range(0, self.n_factors):
            input = tf.nn.l2_normalize(tf.concat([u1_factor[i], i1_factor[i]], 1), 1)
            output_h_u1 = tf.layers.dense(inputs=input, units=self.emb_dim, activation=tf.nn.tanh,
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(self.decay_w),
                                          name="attention1", reuse=tf.AUTO_REUSE)
            output_u1 = tf.layers.dense(inputs=output_h_u1, units=1, activation=None,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.decay_w),
                                        use_bias=None, name="attention_h1", reuse=tf.AUTO_REUSE)
            attention_weight1.append(output_u1)

        attention_1 = tf.concat(attention_weight1, 1)

        atte_1 = self.attention_enlarge_num *  tf.nn.softmax(attention_1, 1)

        u1_user_embbedding = []
        for i in range(0, self.n_factors):
            u1_factors = tf.expand_dims(atte_1[:, i], 1) * u1_factor[i]
            u1_user_embbedding.append(u1_factors)

        u1_user_embbedding = tf.concat(u1_user_embbedding, 1)

        user = u3_user_embbedding + u2_user_embbedding + u1_user_embbedding
        item = item3 + item2 + item1

        score = tf.reduce_sum(tf.multiply(user, item ), 1)

        return score


    def _create_mbcgcn_embed2(self, user_embedding, item_embedding):  # The second behavior
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj2)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj2)

        ego_embeddings2 = tf.concat([user_embedding, item_embedding], axis=0)
        all_embeddings = [ego_embeddings2]
        one_layer_gcn = 0
        for k in range(0, self.n_layers2):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings2))

            side_embeddings2 = tf.concat(temp_embed, 0)
            if k == 0:
                one_layer_gcn = side_embeddings2
            ego_embeddings2 = side_embeddings2
            all_embeddings += [ego_embeddings2]
        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        u_g_embeddings2, i_g_embeddings2 = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        one_layer_gcn_users, one_layer_gcn_items = tf.split(one_layer_gcn, [self.n_users, self.n_items], 0)
        return u_g_embeddings2, i_g_embeddings2, one_layer_gcn_users, one_layer_gcn_items


    def personal_fea_trans2(self, one_layer_gcn_users2, one_layer_gcn_items2, ua_embeddings2, ia_embeddings2):  # The second behavior meta network

        # user personal feature transformation
        u_emb = []

        user_one_gcn_factor = tf.split(one_layer_gcn_users2, self.n_factors, 1)
        u_embeddings_factor = tf.split(ua_embeddings2, self.n_factors, 1)

        for i in range(0, self.n_factors):
            input = tf.concat([u_embeddings_factor[i], user_one_gcn_factor[i]], axis=1)
            meta_u2 = tf.layers.dense(inputs=input, units=self.emb_dim, activation=tf.nn.relu,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(self.decay_w_pft),
                                      name="meta_u2", reuse=tf.AUTO_REUSE)
            meta_o_u2 = tf.layers.dense(inputs=meta_u2,
                                        units=int(self.emb_dim / self.n_factors) * int(self.emb_dim / self.n_factors),
                                        activation=None,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.decay_w_pft),
                                        name="meta_o_u2", reuse=tf.AUTO_REUSE)
            meta_o_u2 = tf.nn.l2_normalize(meta_o_u2, 1)
            meta_o_u2 = tf.reshape(meta_o_u2, [-1, int(self.emb_dim / self.n_factors), int(self.emb_dim / self.n_factors)])

            # user personal feature transformations
            gcn_ua_2_factor = tf.expand_dims(u_embeddings_factor[i], axis=1)
            ua_embeddings1_0 = tf.matmul(gcn_ua_2_factor, meta_o_u2)
            ua_embeddings1_0 = tf.squeeze(ua_embeddings1_0, [1])
            u_emb.append(ua_embeddings1_0)

        u_emb_next = tf.concat(u_emb, 1)

        # item personal feature transformation
        i_emb = []
        item_one_gcn_factor = tf.split(one_layer_gcn_items2, self.n_factors, 1)
        i_embeddings_factor = tf.split(ia_embeddings2, self.n_factors, 1)

        for i in range(0, self.n_factors):
            input = tf.concat([i_embeddings_factor[i], item_one_gcn_factor[i]], axis=1)
            meta_i2 = tf.layers.dense(inputs=input, units=self.emb_dim, activation=tf.nn.relu,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(self.decay_w_pft),
                                      name="meta_i2", reuse=tf.AUTO_REUSE)
            meta_o_i2 = tf.layers.dense(inputs=meta_i2,
                                        units=int(self.emb_dim / self.n_factors) * int(self.emb_dim / self.n_factors),
                                        activation=None,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.decay_w_pft),
                                        name="meta_o_i2", reuse=tf.AUTO_REUSE)
            meta_o_i2 = tf.nn.l2_normalize(meta_o_i2, 1)
            meta_o_i2 = tf.reshape(meta_o_i2, [-1, int(self.emb_dim / self.n_factors), int(self.emb_dim / self.n_factors)])

            # item personal feature transformations
            gcn_ia_2_factor = tf.expand_dims(i_embeddings_factor[i], axis=1)
            ia_embeddings1_0 = tf.matmul(gcn_ia_2_factor, meta_o_i2)
            ia_embeddings1_0 = tf.squeeze(ia_embeddings1_0, [1])
            i_emb.append(ia_embeddings1_0)

        i_emb_next = tf.concat(i_emb, 1)

        return u_emb_next, i_emb_next

    def _create_mbcgcn_embed3(self, user_embedding, item_embedding):  # The first behavior
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj3)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj3)

        ego_embeddings3 = tf.concat([user_embedding, item_embedding], axis=0)
        all_embeddings = [ego_embeddings3]
        one_layer_gcn = 0
        for k in range(0, self.n_layers3):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings3))

            side_embeddings3 = tf.concat(temp_embed, 0)
            if k == 0:
                one_layer_gcn = side_embeddings3
            ego_embeddings3 = side_embeddings3
            all_embeddings += [ego_embeddings3]
        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        u_g_embeddings3, i_g_embeddings3 = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        one_layer_gcn_users, one_layer_gcn_items = tf.split(one_layer_gcn, [self.n_users, self.n_items], 0)
        return u_g_embeddings3, i_g_embeddings3, one_layer_gcn_users, one_layer_gcn_items


    def create_cor_loss3(self, ua_embeddings3, ia_embeddings3, users, pos_items):  # Disentangle the first behavior
        """
        :return: the first behavoir distance metric loss
        """
        cor_loss = tf.constant(0.0, tf.float32)

        if self.cor_flag3 == 0:
            return cor_loss

        self.cor_u_g_embeddings = tf.nn.embedding_lookup(ua_embeddings3, users)
        self.cor_i_g_embeddings = tf.nn.embedding_lookup(ia_embeddings3, pos_items)

        ui_embeddings = tf.concat([self.cor_u_g_embeddings, self.cor_i_g_embeddings], axis=0)
        ui_factor_embeddings = tf.split(ui_embeddings, self.n_factors, 1)

        for i in range(0, self.n_factors - 1):
            x = ui_factor_embeddings[i]
            y = ui_factor_embeddings[i + 1]
            cor_loss += self._create_distance_correlation(x, y)

        cor_loss /= ((self.n_factors + 1.0) * self.n_factors / 2)

        return cor_loss

    def personal_fea_trans3(self, one_layer_gcn_users3, one_layer_gcn_items3, ua_embeddings3, ia_embeddings3):  # The first behavior meta network

        #user personal feature transformation
        u_emb = []

        user_one_gcn_factor = tf.split(one_layer_gcn_users3, self.n_factors, 1)
        u_embeddings_factor = tf.split(ua_embeddings3, self.n_factors, 1)

        for i in range(0, self.n_factors):
            input = tf.concat([u_embeddings_factor[i] ,user_one_gcn_factor[i]], axis=1)
            meta_u3 = tf.layers.dense(inputs=input, units=self.emb_dim, activation=tf.nn.relu,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(self.decay_w_pft),
                                      name="meta_u3", reuse=tf.AUTO_REUSE)
            meta_o_u3 = tf.layers.dense(inputs=meta_u3, units=int(self.emb_dim/self.n_factors) * int(self.emb_dim/self.n_factors), activation=None,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.decay_w_pft),
                                        name="meta_o_u3", reuse=tf.AUTO_REUSE)
            meta_o_u3 = tf.nn.l2_normalize(meta_o_u3, 1)
            meta_o_u3 = tf.reshape(meta_o_u3, [-1, int(self.emb_dim/self.n_factors), int(self.emb_dim/self.n_factors)])

            # user personal feature transformations
            gcn_ua_3_factor = tf.expand_dims(u_embeddings_factor[i], axis=1)
            ua_embeddings2_0 = tf.matmul(gcn_ua_3_factor, meta_o_u3)
            ua_embeddings2_0 = tf.squeeze(ua_embeddings2_0, [1])
            u_emb.append(ua_embeddings2_0)

        u_emb_next = tf.concat(u_emb, 1)

        # item personal feature transformation
        i_emb = []
        item_one_gcn_factor = tf.split(one_layer_gcn_items3, self.n_factors, 1)
        i_embeddings_factor = tf.split(ia_embeddings3, self.n_factors, 1)

        for i in range(0, self.n_factors):
            input = tf.concat([i_embeddings_factor[i], item_one_gcn_factor[i]], axis=1)
            meta_i3 = tf.layers.dense(inputs=input, units=self.emb_dim, activation=tf.nn.relu,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(self.decay_w_pft),
                                      name="meta_i3", reuse=tf.AUTO_REUSE)
            meta_o_i3 = tf.layers.dense(inputs=meta_i3, units=int(self.emb_dim/self.n_factors) * int(self.emb_dim/self.n_factors), activation=None,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.decay_w_pft),
                                        name="meta_o_i3", reuse=tf.AUTO_REUSE)
            meta_o_i3 = tf.nn.l2_normalize(meta_o_i3, 1)
            meta_o_i3 = tf.reshape(meta_o_i3, [-1, int(self.emb_dim/self.n_factors), int(self.emb_dim/self.n_factors)])

            # item personal feature transformations
            gcn_ia_3_factor = tf.expand_dims(i_embeddings_factor[i], axis=1)
            ia_embeddings2_0 = tf.matmul(gcn_ia_3_factor, meta_o_i3)
            ia_embeddings2_0 = tf.squeeze(ia_embeddings2_0, [1])
            i_emb.append(ia_embeddings2_0)

        i_emb_next = tf.concat(i_emb, 1)

        return u_emb_next, i_emb_next


    def _create_distance_correlation(self, X1, X2):
        def _create_centered_distance(X):
            '''
                Used to calculate the distance matrix of N samples.
                (However how could tf store a HUGE matrix with the shape like 70000*70000*4 Bytes????)
            '''
            # calculate the pairwise distance of X
            # .... A with the size of [batch_size, embed_size/n_factors]
            # .... D with the size of [batch_size, batch_size]
            # X = tf.math.l2_normalize(XX, axis=1)
            r = tf.reduce_sum(tf.square(X), 1, keepdims=True)
            D = tf.sqrt(tf.maximum(r - 2 * tf.matmul(a=X, b=X, transpose_b=True) + tf.transpose(r), 0.0) + 1e-8)

            # # calculate the centered distance of X
            # # .... D with the size of [batch_size, batch_size]
            D = D - tf.reduce_mean(D, axis=0, keepdims=True) - tf.reduce_mean(D, axis=1, keepdims=True) \
                + tf.reduce_mean(D)
            return D

        def _create_distance_covariance(D1, D2):
            # calculate distance covariance between D1 and D2
            n_samples = tf.cast(tf.shape(D1)[0], tf.float32)
            dcov = tf.sqrt(tf.maximum(tf.reduce_sum(D1 * D2) / (n_samples * n_samples), 0.0) + 1e-8)
            # dcov = tf.sqrt(tf.maximum(tf.reduce_sum(D1 * D2)) / n_samples
            return dcov

        D1 = _create_centered_distance(X1)
        D2 = _create_centered_distance(X2)

        dcov_12 = _create_distance_covariance(D1, D2)
        dcov_11 = _create_distance_covariance(D1, D1)
        dcov_22 = _create_distance_covariance(D2, D2)

        # calculate the distance correlation
        dcor = dcov_12 / (tf.sqrt(tf.maximum(dcov_11 * dcov_22, 0.0)) + 1e-10)
        # return tf.reduce_sum(D1) + tf.reduce_sum(D2)
        return dcor

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)  # 转换成稀疏矩阵   #源代码少一行
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)


def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    pretrain_path2 = '%spretrain/%s/%s.npz' % (args.proj_path2, args.dataset, 'embedding')
    pretrain_path3 = '%spretrain/%s/%s.npz' % (args.proj_path3, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        pretrain_data2 = np.load(pretrain_path2)
        pretrain_data3 = np.load(pretrain_path3)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
        pretrain_data2 = None
        pretrain_data3 = None
    return pretrain_data, pretrain_data2, pretrain_data3


# parallelized sampling on CPU
class sample_thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        with tf.device(cpus[0]):
            self.data = data_generator.sample()


class sample_thread_test(threading.Thread):  # <user,pos,neg> pair-wise
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        with tf.device(cpus[0]):
            self.data = data_generator.sample_test()

        # training on GPU


class train_thread(threading.Thread):
    def __init__(self, model, sess, sample):
        threading.Thread.__init__(self)
        self.model = model
        self.sess = sess
        self.sample = sample

    def run(self):
        users, pos_items, neg_items = self.sample.data
        self.data = sess.run(
            [self.model.opt, self.model.loss, self.model.mf_loss, self.model.emb_loss, self.model.reg_loss, self.model.cor_loss],
            feed_dict={model.users: users, model.pos_items: pos_items,
                       model.node_dropout: eval(args.node_dropout),
                       model.mess_dropout: eval(args.mess_dropout),
                       model.neg_items: neg_items})


class train_thread_test(threading.Thread):
    def __init__(self, model, sess, sample):
        threading.Thread.__init__(self)
        self.model = model
        self.sess = sess
        self.sample = sample

    def run(self):
        users, pos_items, neg_items = self.sample.data
        self.data = sess.run([self.model.loss, self.model.mf_loss, self.model.emb_loss, self.model.cor_loss],
                             feed_dict={model.users: users, model.pos_items: pos_items,
                                        model.neg_items: neg_items,
                                        model.node_dropout: eval(args.node_dropout),
                                        model.mess_dropout: eval(args.mess_dropout)})


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    f0 = time()


    config = dict()
    n_item = max(data_generator.n_items, data_generator2.n_items, data_generator3.n_items)
    config['n_users'] = data_generator.n_users
    config['n_items'] = n_item
    # sess = tf.Session(config=config)
    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    plain_adj, norm_adj, mean_adj, pre_adj = data_generator.get_adj_mat()
    plain_adj2, norm_adj2, mean_adj2, pre_adj2 = data_generator2.get_adj_mat()
    plain_adj3, norm_adj3, mean_adj3, pre_adj3 = data_generator3.get_adj_mat()

    if args.adj_type == 'plain':
        config['norm_adj'] = plain_adj
        print('use the plain adjacency matrix')
    elif args.adj_type == 'norm':
        config['norm_adj'] = norm_adj
        print('use the normalized adjacency matrix')
    elif args.adj_type == 'gcmc':
        config['norm_adj'] = mean_adj
        print('use the gcmc adjacency matrix')
    elif args.adj_type == 'pre':
        config['norm_adj'] = pre_adj
        config['norm_adj2'] = pre_adj2
        config['norm_adj3'] = pre_adj3
        print('use the pre adjcency matrix')
    else:
        config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
        print('use the mean adjacency matrix')
    t0 = time()
    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None
        pretrain_data2 = None
        pretrain_data3 = None
    """
    *********************************************************
    Save the model parameters.
    """

    """
    *********************************************************
    Reload the pretrained model parameters.
    """
    if args.pretrain == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        model_type = 'LightGCN'
        pretrain_path = '%s/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset1, model_type, layer,
                                                 str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))

        pretrain_path2 = '%s/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset2, model_type, layer,
                                                  str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))

        pretrain_path3 = '%s/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset3, model_type, layer,
                                                  str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))
        print(pretrain_path)
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        print(ckpt)
        ckpt2 = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path2 + '/checkpoint'))
        ckpt3 = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path3 + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            with tf.Session() as sess:
                saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')

                saver.restore(sess, tf.train.latest_checkpoint(pretrain_path))
                graph = tf.get_default_graph()
                print(graph.get_tensor_by_name("user_embedding:0"))
                print(graph.get_tensor_by_name("item_embedding:0"))
                user_embedding = sess.run('user_embedding:0')
                item_embedding = sess.run('item_embedding:0')
                print('load the pretrained model parameters from: ', pretrain_path)
                pretrain_data = dict()
                pretrain_data = {'user_embed': user_embedding, 'item_embed': item_embedding}
                print('load the pretrained data ')
            tf.reset_default_graph()

        if ckpt2 and ckpt2.model_checkpoint_path:
            with tf.Session() as sess:
                saver = tf.train.import_meta_graph(ckpt2.model_checkpoint_path + '.meta')

                saver.restore(sess, tf.train.latest_checkpoint(pretrain_path2))
                graph = tf.get_default_graph()
                print(graph.get_tensor_by_name("user_embedding:0"))
                print(graph.get_tensor_by_name("item_embedding:0"))
                user_embedding2 = sess.run('user_embedding:0')
                item_embedding2 = sess.run('item_embedding:0')

                print('load the pretrained model parameters from: ', pretrain_path2)
                pretrain_data2 = dict()
                pretrain_data2 = {'user_embed': user_embedding2, 'item_embed': item_embedding2}
                print('load the pretrained data2 ')
            tf.reset_default_graph()

        if ckpt3 and ckpt3.model_checkpoint_path:
            with tf.Session() as sess:
                saver = tf.train.import_meta_graph(ckpt3.model_checkpoint_path + '.meta')
                saver.restore(sess, tf.train.latest_checkpoint(pretrain_path3))
                graph = tf.get_default_graph()
                print(graph.get_tensor_by_name("user_embedding:0"))
                print(graph.get_tensor_by_name("item_embedding:0"))
                user_embedding3 = sess.run('user_embedding:0')
                item_embedding3 = sess.run('item_embedding:0')
                print(user_embedding3)
                print(item_embedding3)
                print('load the pretrained model parameters from: ', pretrain_path3)
                pretrain_data3 = dict()
                pretrain_data3 = {'user_embed': user_embedding3, 'item_embed': item_embedding3}

                print('load the pretrained data3 ')
            tf.reset_default_graph()

        else:
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            cur_best_pre_0 = 0.
            print('without pretraining.')
        pretrain_data['u1_u1_W1'] = pretrain_data['user_embed']
        cur_best_pre_0 = 0.
    else:
        cur_best_pre_0 = 0.
        print('without pretraining.')
    model = DisenCGCN(data_config=config, pretrain_data=pretrain_data, pretrain_data2=pretrain_data2,
                   pretrain_data3=pretrain_data3)  # 创建模型类的对象

    'Save'
    # saver = tf.train.Saver()
    if args.save_flag == 1:
        model_type = 'Disen-CGCN'
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        weights_save_path = '%sweights_one/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model_type, layer,
                                                                str(args.lr),
                                                                '-'.join([str(r) for r in eval(args.regs)]))
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    """
    *********************************************************
    Get the performance w.r.t. different sparsity levels.
    """
    if args.report == 1:
        assert args.test_flag == 'full'
        users_to_test_list, split_state = data_generator.get_sparsity_split()
        users_to_test_list.append(list(data_generator.test_set.keys()))
        split_state.append('all')

        report_path = '%sreport/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
        ensureDir(report_path)
        f = open(report_path, 'w')
        f.write(
            'embed_size=%d, lr=%.4f, layer_size=%s, keep_prob=%s, regs=%s, loss_type=%s, adj_type=%s\n'
            % (args.embed_size, args.lr, args.layer_size, args.keep_prob, args.regs, args.loss_type, args.adj_type))

        for i, users_to_test in enumerate(users_to_test_list):
            ret = test(sess, model, users_to_test, drop_flag=True)

            final_perf = "recall=[%s], precision=[%s], ndcg=[%s]" % \
                         (', '.join(['%.5f' % r for r in ret['recall']]),
                          ', '.join(['%.5f' % r for r in ret['precision']]),
                          ', '.join(['%.5f' % r for r in ret['ndcg']]))

            f.write('\t%s\n\t%s\n' % (split_state[i], final_perf))
        f.close()
        exit()

    """
    *********************************************************
    Train.
    """
    tensorboard_model_path = 'tensorboard/'
    if not os.path.exists(tensorboard_model_path):
        os.makedirs(tensorboard_model_path)
    run_time = 1
    while (True):
        if os.path.exists(tensorboard_model_path + model.log_dir + '/run_' + str(run_time)):
            run_time += 1
        else:
            break
    train_writer = tf.summary.FileWriter(tensorboard_model_path + model.log_dir + '/run_' + str(run_time), sess.graph)

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False

    # sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    # print('i_here')
    # print(sess.run(model.weights_one))
    for epoch in range(1, args.epoch + 1):
        t1 = time()  # time
        loss, mf_loss, emb_loss, reg_loss, cor_loss = 0., 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1
        loss_test, mf_loss_test, emb_loss_test, reg_loss_test, cor_loss_test = 0., 0., 0., 0., 0.
        '''
        *********************************************************
        parallelized sampling
        '''
        sample_last = sample_thread()
        sample_last.start()
        sample_last.join()
        for idx in range(n_batch):
            train_cur = train_thread(model, sess, sample_last)
            sample_next = sample_thread()

            train_cur.start()
            sample_next.start()

            sample_next.join()
            train_cur.join()

            users, pos_items, neg_items = sample_last.data
            _, batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss, batch_cor_loss = train_cur.data
            sample_last = sample_next

            loss += batch_loss / n_batch
            mf_loss += batch_mf_loss / n_batch
            emb_loss += batch_emb_loss / n_batch
            cor_loss += batch_cor_loss / n_batch

        summary_train_loss = sess.run(model.merged_train_loss,
                                      feed_dict={model.train_loss: loss, model.train_mf_loss: mf_loss,
                                                 model.train_emb_loss: emb_loss, model.train_reg_loss: reg_loss, model.train_cor_loss: cor_loss})  # 总
        train_writer.add_summary(summary_train_loss, epoch)
        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        if (epoch % 5) != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss, cor_loss)
                print(perf_str)
            continue

        users_to_test = list(data_generator.train_items.keys())
        ret = test(sess, model, users_to_test, drop_flag=True, train_set_flag=1)
        perf_str = 'Epoch %d: train==[%.5f=%.5f + %.5f + %.5f], recall=[%s], precision=[%s], ndcg=[%s]' % \
                   (epoch, loss, mf_loss, emb_loss, reg_loss,
                    ', '.join(['%.5f' % r for r in ret['recall']]),
                    ', '.join(['%.5f' % r for r in ret['precision']]),
                    ', '.join(['%.5f' % r for r in ret['ndcg']]))
        print(perf_str)
        summary_train_acc = sess.run(model.merged_train_acc, feed_dict={model.train_rec_first: ret['recall'][0],
                                                                        model.train_rec_last: ret['recall'][-1],
                                                                        model.train_ndcg_first: ret['ndcg'][0],
                                                                        model.train_ndcg_last: ret['ndcg'][-1]})
        train_writer.add_summary(summary_train_acc, epoch // 5)

        '''
        *********************************************************
        parallelized sampling 
        '''
        sample_last = sample_thread_test()
        sample_last.start()
        sample_last.join()
        for idx in range(n_batch):
            train_cur = train_thread_test(model, sess, sample_last)
            sample_next = sample_thread_test()

            train_cur.start()
            sample_next.start()

            sample_next.join()
            train_cur.join()

            users, pos_items, neg_items = sample_last.data
            batch_loss_test, batch_mf_loss_test, batch_emb_loss_test, batch_cor_loss_test= train_cur.data
            sample_last = sample_next

            loss_test += batch_loss_test / n_batch
            mf_loss_test += batch_mf_loss_test / n_batch
            emb_loss_test += batch_emb_loss_test / n_batch
            cor_loss_test += batch_cor_loss_test / n_batch


        summary_test_loss = sess.run(model.merged_test_loss,
                                     feed_dict={model.test_loss: loss_test, model.test_mf_loss: mf_loss_test,
                                                model.test_emb_loss: emb_loss_test, model.test_reg_loss: reg_loss_test, model.test_cor_loss: cor_loss_test})
        train_writer.add_summary(summary_test_loss, epoch // 5)
        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(sess, model, users_to_test, drop_flag=True)
        summary_test_acc = sess.run(model.merged_test_acc,
                                    feed_dict={model.test_rec_first: ret['recall'][0],
                                               model.test_rec_last: ret['recall'][-1],
                                               model.test_ndcg_first: ret['ndcg'][0],
                                               model.test_ndcg_last: ret['ndcg'][-1]})  # 在测试集的训练指标
        train_writer.add_summary(summary_test_acc, epoch // 5)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: test==[%.5f=%.5f + %.5f + %.5f], recall=[%s], ' \
                       'precision=[%s], ndcg=[%s]' % \
                       (epoch, t2 - t1, t3 - t2, loss_test, mf_loss_test, emb_loss_test, cor_loss_test,
                        ', '.join(['%.5f' % r for r in ret['recall']]),
                        ', '.join(['%.5f' % r for r in ret['precision']]),
                        ', '.join(['%.5f' % r for r in ret['ndcg']]))
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
            print('save the weights in path: ', weights_save_path)

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)

    save_path = '%soutput/%s/%s/%s.result2' % (args.proj_path, args.dataset, args.data_path, model.model_type)
    ensureDir(save_path)
    f = open(save_path, 'a')

    f.write(
        'dataset:%s, embed_size=%d, lr=%.4f, layer_size=%s, b1_layer_size=%s, b2_layer_size=%s, b3_layer_size=%s, n_factors=%s, cor_flag3=%s, node_dropout=%s, mess_dropout=%s, regs=%s, reg_c=%s, reg_W_l2=%s, reg_W_PFT_l2=%s, attention_enlarge_num=%s, adj_type=%s\n\t%s\n'
        % (args.weights_path, args.embed_size, args.lr, args.layer_size, args.layer_size4, args.layer_size3, args.layer_size2, args.n_factors, args.cor_flag3,
           args.node_dropout, args.mess_dropout, args.regs, args.reg_c, args.reg_W_l2,  args.reg_W_PFT_l2, args.attention_enlarge_num,args.adj_type, final_perf))
    f.close()
