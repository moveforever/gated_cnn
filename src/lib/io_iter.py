# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
from __future__ import print_function
import sys
import numpy as np
import mxnet as mx
import threading

# The interface of a data iter that works for bucketing
#
# DataIter
#   - default_bucket_key: the bucket key for the default symbol.
#
# DataBatch
#   - provide_data: same as DataIter, but specific to this batch
#   - provide_label: same as DataIter, but specific to this batch
#   - bucket_key: the key for the bucket that should be used for this batch


def default_gen_buckets(sentences, batch_size, max_len, min_len):
    len_dict = {}
    for sentence in sentences:
        lw = len(sentence)
        if lw < min_len:
            continue
        if lw in len_dict:
            len_dict[lw] += 1
        else:
            len_dict[lw] = 1

    tl = 0
    buckets = []
    for l, n in len_dict.items(): # TODO: There are better heuristic ways to do this
        if n + tl >= batch_size:
            buckets.append(l)
            tl = 0
        else:
            tl += n
    if tl > 0:
        buckets.append(max_len)
    return buckets


class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label, bucket_key):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key

        self.pad = 0
        self.index = None # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

class CNABucketIter(mx.io.DataIter):
    def __init__(self, max_len, min_len, batch_size, title_file, label_file, is_shuffle = False,
                 data_names=['title_data'], label_name=['softmax_label'], buckets = None):
        super(CNABucketIter, self).__init__()

	class LoadThread (threading.Thread):
    	    def __init__(self, data_file, ismmap):
        	threading.Thread.__init__(self)
		self.data_file = data_file
		self.ismmap = ismmap
		self.arr = np.array([])
    	    def run(self):
	    	print("Load data %s" % self.data_file)
	    	if self.ismmap:
            	    self.arr = np.load(self.data_file + ".npy", mmap_mode='r')
	    	else:
            	    self.arr = np.load(self.data_file + ".npy")
	
	t2 = LoadThread(label_file, True)
	t3 = LoadThread(title_file, False)
	t2.start()
	t3.start()

	t2.join()
	t3.join()

	labels = t2.arr
	sentences = t3.arr

        self.data_names = data_names
        self.label_name = label_name
	self.is_shuffle = is_shuffle

        if buckets is None:
            buckets = default_gen_buckets(sentences, batch_size, max_len, min_len)
        print(buckets)

        buckets.sort()
        self.buckets = buckets
        self.indices = [[] for _ in buckets]

        # pre-allocate with the largest bucket for better memory sharing
        self.default_bucket_key = max(buckets)

        ind = -1
        for sentence in sentences:
            ind += 1
            if len(sentence) < min_len:
                continue
            for i, bkt in enumerate(buckets):
                if bkt >= len(sentence):
                    self.indices[i].append(ind)
                    break

            # we just ignore the sentence it is longer than the maximum
            # bucket size here

        # convert data into ndarrays for better speed during training
        title_data = [np.zeros((len(x), buckets[i])) for i, x in enumerate(self.indices)]
        for i_bucket in range(len(self.buckets)):
            for j in range(len(self.indices[i_bucket])):
                sentence = sentences[self.indices[i_bucket][j]]
                title_data[i_bucket][j, :len(sentence)] = sentence

        self.data = title_data
        self.labels = [ labels[arr] for arr in self.indices]

        # Get the size of each bucket, so that we could sample
        # uniformly from the bucket
        bucket_sizes = [len(x) for x in self.data]

        print("Summary of dataset ==================")
        for bkt, size in zip(buckets, bucket_sizes):
            print("bucket of len %3d : %d samples" % (bkt, size))

        self.batch_size = int(batch_size)
        self.make_data_iter_plan()

        self.provide_data = [('title_data', (self.batch_size, self.default_bucket_key)) ] 
        self.provide_label = [('softmax_label', (self.batch_size, 1))]

    def make_data_iter_plan(self):
        "make a random data iteration plan"
        # truncate each bucket into multiple of batch-size
        bucket_n_batches = []
        for i in range(len(self.data)):
            bucket_n_batches.append(len(self.data[i]) / self.batch_size)
	    print("bucket i = %d, missing = %d" % (i, len(self.data[i]) - bucket_n_batches[i]*self.batch_size))
            self.data[i] = self.data[i][:int(bucket_n_batches[i]*self.batch_size)]
            self.labels[i] = self.labels[i][:int(bucket_n_batches[i]*self.batch_size)]

        bucket_plan = np.hstack([np.zeros(n, int)+i for i, n in enumerate(bucket_n_batches)])
        np.random.shuffle(bucket_plan)

	if self.is_shuffle:
            self.bucket_idx_all = [np.random.permutation(len(x)) for x in self.data]
	else:
            self.bucket_idx_all = [range(len(x)) for x in self.data]

        self.bucket_plan = bucket_plan
        self.bucket_curr_idx = [0 for x in self.data]

        self.data_buffer = []
        self.label_buffer = np.zeros((self.batch_size, 1))
        for i_bucket in range(len(self.data)):
            data = np.zeros((self.batch_size, self.buckets[i_bucket]))
	    self.data_buffer.append(data)

    def __iter__(self):

        for i_bucket in self.bucket_plan:
            data = self.data_buffer[i_bucket]
            label = self.label_buffer

            i_idx = self.bucket_curr_idx[i_bucket]
            idx = self.bucket_idx_all[i_bucket][i_idx:i_idx+self.batch_size]
            self.bucket_curr_idx[i_bucket] += self.batch_size
            data[:] = self.data[i_bucket][idx]
            label[:] = self.labels[i_bucket][idx]

            data_all = [mx.nd.array(data)]
            label_all = [mx.nd.array(label)]

            data_batch = SimpleBatch(self.data_names, data_all, self.label_name, label_all,
                                     self.buckets[i_bucket])

            yield data_batch

    def reset(self):
        self.bucket_curr_idx = [0 for x in self.data]
	if self.is_shuffle:
            self.bucket_idx_all[:] = [np.random.permutation(len(x)) for x in self.data]
