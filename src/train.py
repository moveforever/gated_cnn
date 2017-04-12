#!/usr/bin/env python
import sys
import numpy as np
import json
import codecs

sys.path.insert(0, "./src/lib")

import mxnet as mx
from loss   import LogLoss
from io_iter   import CNABucketIter

def build_net(sentence_size=40, num_embed=40, vocab_size = 2000, filter_list=[3, 4, 5], num_filter=100, dropout=0.):
    #########CNN_TITLE
    input_x = mx.sym.Variable('title_data') # placeholder for input
    input_y = mx.sym.Variable('softmax_label') # placeholder for output

    # embedding layer
    embed_layer = mx.sym.Embedding(data=input_x, input_dim=vocab_size, output_dim=num_embed, name='vocab_embed')
    embed_layer = mx.sym.Reshape(data=embed_layer, shape=(-1, 1, sentence_size, num_embed))

    # create convolution + (max) pooling layer for each filter operation
    pooled_outputs = []
    for i, filter_size in enumerate(filter_list):
        conv_w = mx.sym.Convolution(data=embed_layer, kernel=(filter_size, num_embed), num_filter=num_filter, name = "linear_%d" % i)
        conv_w = mx.sym.Activation(data=conv_w, act_type='sigmoid')
        conv_v = mx.sym.Convolution(data=embed_layer, kernel=(filter_size, num_embed), num_filter=num_filter, name = "gated_%d" % i)
	conv = conv_w*conv_v
        pooli = mx.sym.Pooling(data=conv, pool_type='max', kernel=(sentence_size - filter_size + 1, 1), stride=(1,1))
        pooled_outputs.append(pooli)

    # combine all pooled outputs
    total_filters = num_filter * len(filter_list)
    concat = mx.sym.Concat(*pooled_outputs, dim=1)
    h_pool = mx.sym.Reshape(data=concat, shape=(-1, total_filters))

    # dropout layer
    if dropout > 0.0:
        h_drop = mx.sym.Dropout(data=h_pool, p=dropout)
    else:
        h_drop = h_pool

    title = mx.sym.FullyConnected(data=h_drop, num_hidden=100, name="fc")
    title = mx.symbol.Activation(data=title, act_type="relu")
    title = mx.sym.FullyConnected(data=title, num_hidden=1, name="title_feature")

    net = mx.symbol.LogisticRegressionOutput(data=title, label=input_y, name='softmax')
    return net 

if __name__ == "__main__":
    batch_size = 256
    #parameter parser
    vocab_size = 5143
    data_dir   = 'data' 
    epoch_num = 20
    model_dir = 'model'
    sentence_size  = 30

    buckets = None

    train  = CNABucketIter(
				    max_len   = sentence_size,
				    min_len   = 5,
				    batch_size = batch_size ,
				    title_file=data_dir + "/title_sample" ,
				    label_file=data_dir + "/label_sample",
				    is_shuffle = True,
				    buckets = buckets)

    test  = CNABucketIter(
				    max_len   = sentence_size,
				    min_len   = 5,
				    batch_size = batch_size ,
				    title_file=data_dir + "/title_sample" ,
				    label_file=data_dir + "/label_sample",
				    buckets = train.buckets)


    devs = [mx.gpu(i) for i in range(1)]
    def norm_stat(d):
       return mx.nd.norm(d)/np.sqrt(d.size)
    mon = mx.mon.Monitor(
        100,                 # Print every 100 batches\n",
        norm_stat)         # The statistics function defined above\n",
        #pattern='.*weight')       # A regular expression. Only arrays with name matching this pattern will be included.\n",
          
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    def sym_gen(seq_len):
    	sym = build_net(sentence_size=seq_len, num_embed=64, vocab_size=vocab_size, filter_list=[3, 4, 5], num_filter=100, dropout=0)
    	data_names = ['title_data']
    	label_names = ['softmax_label']
    	return (sym, data_names, label_names)

    mod = mx.mod.BucketingModule(sym_gen,
                                 default_bucket_key=train.default_bucket_key,
                                 context=devs)
  
  # initialization
    optimizer_params = {'wd': 0.00002,
                      'learning_rate': 0.0000001}
          		#'lr_scheduler': mx.lr_scheduler.FactorScheduler(298000,0.81)}

    epoch_end_callback = mx.callback.do_checkpoint(model_dir + '/cnn_title', period=1)
    batch_end_callback = mx.callback.Speedometer(batch_size, 10)
    eval_metric = mx.metric.np(LogLoss)
  
    init = mx.init.Mixed([".*embed_weight", ".*"], [mx.init.MSRAPrelu(), mx.init.Xavier(factor_type="in", magnitude=2)])
    #init = mx.init.Mixed([".*embed_weight", ".*"], [mx.init.Xavier(rnd_type="gaussian", factor_type="in", magnitude=2.34), mx.init.Xavier(factor_type="in", magnitude=2.34)])
    #init = mx.init.Mixed(["*embed_weight", ".*"], [mx.init.Xavier(rnd_type="uniform", factor_type="", magnitude=2.34), mx.init.Xavier(factor_type="in", magnitude=2.34)])
  # start train
    print "start training..." 
    mod.fit(train, test, eval_metric=eval_metric,
          num_epoch = epoch_num,
          epoch_end_callback=epoch_end_callback,
          batch_end_callback=batch_end_callback,
          optimizer='sgd', optimizer_params=optimizer_params,
          initializer=init,
	  monitor=mon)
 
