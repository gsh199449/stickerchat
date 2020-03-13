# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to process data into batches"""
import os
import queue as Queue
import re
import time
from random import shuffle, randint, seed
from subprocess import check_output
from threading import Thread
from glob import glob
from typing import List

import numpy as np
import tensorflow as tf


import data
import traceback
FLAGS = tf.app.flags.FLAGS
seed(111)

def wc(filename):
    return int(check_output(["wc", "-l", filename]).split()[0])

class Example(object):
    """Class representing a train/val/test example for text summarization."""

    def __init__(self, context_text, reply_text, sticker_set_id, sticker_id, sticker_alt, vocab, emoji_vocab, all_stickers):
        self.FLAGS = FLAGS

        self.context_lens = []
        self.context_input = []
        for c in context_text:
            c_text = c[1].strip()
            if c_text == '':
                continue
            c_words = c_text.split()
            if len(c_words) > FLAGS.max_enc_steps:
                c_words = c_words[:FLAGS.max_enc_steps]
            self.context_lens.append(len(c_words))
            self.context_input.append([vocab.word2id(w) for w in c_words])
        self.context_input = self.context_input[-FLAGS.context_len:]
        self.context_lens = self.context_lens[-FLAGS.context_len:]

        img = np.load(os.path.join(FLAGS.sticker_path, str(sticker_set_id), str(sticker_id)+'.npy'))
        self.sticker_pix = img
        self.sticker_alt_id = emoji_vocab.word2id(sticker_alt)
        self.sticker_id = sticker_id
        self.sticker_set_id = sticker_set_id

        stickers = all_stickers[str(sticker_set_id)]
        self.negative_sticker_ids = list(stickers.keys())
        self.negative_sticker_ids.remove(str(sticker_id))
        self.negative_sticker_ids = self.negative_sticker_ids[:FLAGS.sticker_candidates-1]
        negative_sticker_alts = [stickers[i] for i in self.negative_sticker_ids]
        self.negative_sticker_pixs = [
            np.load(os.path.join(FLAGS.sticker_path, str(sticker_set_id), i + '.npy')) for i in self.negative_sticker_ids
        ]
        self.negative_sticker_alt_ids = [emoji_vocab.word2id(i) for i in negative_sticker_alts]


        self.original_contexts = [c[1] for c in context_text if c[1] != ''][-FLAGS.context_len:]
        self.original_reply_text = reply_text
        self.original_sticker_alt = sticker_alt
        self.original_negative_sticker_alts = negative_sticker_alts

    def pad_encoder_input(self, max_len, pad_id):

        for i in range(len(self.context_input)):
            while len(self.context_input[i]) < max_len:
                self.context_input[i].append(pad_id)


class Batch(object):
    """Class representing a minibatch of train/val/test examples for text summarization."""

    def __init__(self, example_list, vocab):
        self.pad_id = vocab.word2id(data.PAD_TOKEN)  # id of the PAD token used to pad sequences
        self.init_encoder_seq(example_list)  # initialize the input to the encoder
        self.store_orig_strings(example_list)  # store the original strings

    def init_encoder_seq(self, example_list: List[Example]):
        max_enc_seq_len = FLAGS.max_enc_steps

        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

        self.enc_context_batch = np.zeros((FLAGS.batch_size, FLAGS.context_len, max_enc_seq_len), dtype=np.int32)
        self.enc_context_lens = np.zeros((FLAGS.batch_size, FLAGS.context_len), dtype=np.int32)
        self.enc_context_number = np.zeros((FLAGS.batch_size), dtype=np.int32)
        self.enc_context_padding_mask = np.zeros((FLAGS.batch_size, FLAGS.context_len, max_enc_seq_len), dtype=np.float32)

        self.sticker_pix = np.zeros((FLAGS.batch_size, FLAGS.sticker_candidates, FLAGS.sticker_height, FLAGS.sticker_weight, 3), dtype=np.float32)
        self.sticker_alt = np.zeros((FLAGS.batch_size, FLAGS.sticker_candidates), dtype=np.int32)
        self.original_sticker_alt = np.zeros((FLAGS.batch_size, FLAGS.sticker_candidates), dtype=object)
        self.original_sticker_alt[:, :] = '[PAD]'
        self.sticker_selection_label = np.zeros((FLAGS.batch_size), dtype=np.int32)
        self.ground_truth_sticker_alt = np.zeros((FLAGS.batch_size), dtype=np.int32)
        self.candidate_sticker_ids = []
        self.candidate_sticker_set_ids = []

        for i, ex in enumerate(example_list):

            all_samples = [(p, a, oa, False, sticker_id) for p, a, oa, sticker_id in zip(ex.negative_sticker_pixs, ex.negative_sticker_alt_ids, ex.original_negative_sticker_alts, ex.negative_sticker_ids)]
            all_samples.append((ex.sticker_pix, ex.sticker_alt_id, ex.original_sticker_alt, True, ex.sticker_id))
            shuffle(all_samples)
            self.ground_truth_sticker_alt[i] = ex.sticker_alt_id
            self.candidate_sticker_ids.append([])
            self.candidate_sticker_set_ids.append(ex.sticker_set_id)
            for index, (p, a, oa, positive_flag, sid) in enumerate(all_samples):
                if positive_flag:
                    self.sticker_selection_label[i] = index
                self.sticker_pix[i, index, :len(p[0]), :len(p[1]), :] = p
                self.sticker_alt[i, index] = a
                self.original_sticker_alt[i, index] = oa
                self.candidate_sticker_ids[-1].append(sid)

            self.enc_context_number[i] = min(len(ex.context_input), FLAGS.context_len)
            for k in range(min(len(ex.context_input), FLAGS.context_len)):
                self.enc_context_batch[i, k, :] = ex.context_input[k]
                self.enc_context_lens[i, k] = ex.context_lens[k]
                for m in range(ex.context_lens[k]):
                    self.enc_context_padding_mask[i][k][m] = 1

    def store_orig_strings(self, example_list: List[Example]):
        """Store the original article and abstract strings in the Batch object"""
        self.original_contexts = [ex.original_contexts for ex in example_list]  # list of lists
        self.original_reply_text = [ex.original_reply_text for ex in example_list]  # list of list of lists
        self.original_sticker_alt = np.reshape(self.original_sticker_alt, [-1]).tolist()
        self.original_negative_sticker_ids = [ex.negative_sticker_ids for ex in example_list]  # list of lists


class Batcher(object):
    """A class to generate minibatches of data. Buckets examples together based on length of the encoder sequence."""

    BATCH_QUEUE_MAX = 100  # max number of batches the batch_queue can hold

    def __init__(self, data_path, vocab, emoji_vocab, single_pass):
        """Initialize the batcher. Start threads that process the data into batches.

        Args:
          data_path: tf.Example filepattern.
          vocab: Vocabulary object
          FLAGS: hyperparameters
          single_pass: If True, run through the dataset exactly once (useful for when you want to run evaluation on the dev or test set). Otherwise generate random batches indefinitely (useful for training).
        """
        self._data_path = data_path
        self._vocab = vocab
        self._emoji_vocab = emoji_vocab
        self._single_pass = single_pass

        if FLAGS.dataset_size is None or FLAGS.dataset_size < 0:
            tf.logging.info('counting file lines')
            lines = 0
            for f in glob(data_path):
                lines += wc(f)
            self._total_lines = lines
        else:
            tf.logging.info('using FLAGS.dataset_size as _total_lines')
            self._total_lines = FLAGS.dataset_size
        self._batch_num = 0

        # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
        self._batch_queue = Queue.Queue(self.BATCH_QUEUE_MAX)
        self._example_queue = Queue.Queue(self.BATCH_QUEUE_MAX * FLAGS.batch_size)

        # Different settings depending on whether we're in single_pass mode or not
        if single_pass:
            self._num_example_q_threads = 1  # just one thread, so we read through the dataset just once
            self._num_batch_q_threads = 1  # just one thread to batch examples
            self._bucketing_cache_size = 1  # only load one batch's worth of examples before bucketing; this essentially means no bucketing
            self._finished_reading = False  # this will tell us when we're finished reading the dataset
        else:
            self._num_example_q_threads = 64  # num threads to fill example queue
            self._num_batch_q_threads = 10  # num threads to fill batch queue
            self._bucketing_cache_size = 5  # how many batches-worth of examples to load into cache before bucketing

        # Start the threads that load the queues
        self._example_q_threads = []
        for _ in range(self._num_example_q_threads):
            self._example_q_threads.append(Thread(target=self.fill_example_queue))
            self._example_q_threads[-1].daemon = True
            self._example_q_threads[-1].start()
        self._batch_q_threads = []
        for _ in range(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()

        # Start a thread that watches the other threads and restarts them if they're dead
        if not single_pass:  # We don't want a watcher in single_pass mode because the threads shouldn't run forever
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()

    def next_batch(self) -> Batch:
        """Return a Batch from the batch queue.

        If mode='decode' then each batch contains a single example repeated beam_size-many times; this is necessary for beam search.

        Returns:
          batch: a Batch object, or None if we're in single_pass mode and we've exhausted the dataset.
        """
        # If the batch queue is empty, print a warning
        if self._batch_queue.qsize() == 0:
            tf.logging.warning(
                'Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i',
                self._batch_queue.qsize(), self._example_queue.qsize())
            if self._single_pass and self._finished_reading:
                tf.logging.info("Finished reading dataset in single_pass mode.")
                return None

        batch = self._batch_queue.get()  # get the next Batch

        self._batch_num += 1
        if self._batch_num * FLAGS.batch_size > self._total_lines:
            self._batch_num = 0

        return batch

    def progress(self):
        return self._batch_num * FLAGS.batch_size / self._total_lines

    def total_data(self):
        return self._total_lines

    def fill_example_queue(self):
        """Reads data from file and processes into Examples which are then placed into the example queue."""
        input_gen = self.text_generator(data.json_generator(self._data_path, self._single_pass))
        all_stickers = loading_stickers()
        while True:
            try:
                (context_text, reply_text, sticker_set_id, sticker_id, sticker_alt) = next(input_gen)
            except (RuntimeError, StopIteration) as e:  # if there are no more examples:
                tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
                if self._single_pass:
                    tf.logging.info(
                        "single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
                    self._finished_reading = True
                    break
                else:
                    raise Exception("single_pass mode is off but the example generator is out of data; error.")

            if str(sticker_id) not in all_stickers[str(sticker_set_id)]:
                pass
                # tf.logging.warning('sticker %d not in sticker set %d, alt is %s', sticker_id, sticker_set_id, sticker_alt)
                continue
            else:
                example = Example(context_text, reply_text, sticker_set_id, sticker_id, sticker_alt,
                                  self._vocab, self._emoji_vocab, all_stickers)

            if example.sticker_alt_id != 0:  #
                self._example_queue.put(example)  # place the Example in the example queue.

    def fill_batch_queue(self):
        """Takes Examples out of example queue, sorts them by encoder sequence length, processes into Batches and places them in the batch queue.

        In decode mode, makes batches that each contain a single example repeated.
        """
        while True:
            # Get bucketing_cache_size-many batches of Examples into a list, then sort
            inputs = []
            for _ in range(FLAGS.batch_size * self._bucketing_cache_size):
                inputs.append(self._example_queue.get())
            inputs = sorted(inputs, key=lambda inp: sum(inp.context_lens))  # sort by length of encoder sequence

            # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
            batches = []
            for i in range(0, len(inputs), FLAGS.batch_size):
                batches.append(inputs[i:i + FLAGS.batch_size])
            if not self._single_pass:
                shuffle(batches)
            for b in batches:  # each b is a list of Example objects
                self._batch_queue.put(Batch(b, self._vocab))

    def watch_threads(self):
        """Watch example queue and batch queue threads and restart if dead."""
        while True:
            time.sleep(60)
            for idx, t in enumerate(self._example_q_threads):
                if not t.is_alive():  # if the thread is dead
                    tf.logging.error('Found example queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_example_queue)
                    self._example_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()
            for idx, t in enumerate(self._batch_q_threads):
                if not t.is_alive():  # if the thread is dead
                    tf.logging.error('Found batch queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_batch_queue)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()

    def text_generator(self, example_generator):
        """Generates article and abstract text from tf.Example.

        Args:
          example_generator: a generator of tf.Examples from file. See data.example_generator"""
        while True:
            e = next(example_generator)
            try:
                context_text = []
                id_text = {}
                for c in e['context']:
                    context_text.append((c['id'], c['text']))
                    id_text[c['id']] = c['text']
                # current_text = e['current']['text']
                reply_id = e['current']['reply_to_msg_id']
                reply_text = id_text[reply_id] if reply_id is not None and reply_id in id_text else None
                sticker_set_id = e['current']['sticker_set_id']
                sticker_id = e['current']['sticker_id']
                sticker_alt = e['current']['sticker_alt']
            except (ValueError, KeyError, FileNotFoundError) as err:
                traceback.print_exc()
                tf.logging.error('Failed to get article or abstract from example %s', err)
                continue
            if len(context_text) == 0:
                pass
                # tf.logging.warning('Found an example with empty article text. Skipping it.')
            else:
                yield (context_text, reply_text, sticker_set_id, sticker_id, sticker_alt)


def loading_stickers():
    pattern = re.compile('stickers/(\d+)')
    all_srickers = {}
    for sset in glob(os.path.join(FLAGS.sticker_path, '*')):
        r = pattern.findall(sset)
        set_id = r[0]
        f = open(os.path.join(sset, 'emoji_mapping.txt'), encoding='utf8')
        emojis = {}
        for l in f:
            ee = l.strip().split('\t')
            emojis[ee[0]] = ee[1]
        f.close()
        all_srickers[set_id] = emojis
    return all_srickers

