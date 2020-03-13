import glob
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime

import os
import time

import numpy as np
import tensorflow as tf

import data
from data import Vocab, get_config
from gpu_cluster import get_available_gpu, bcolors, get_free_gpu
from models.simple import context_encoder
from text_batcher import Batch, Batcher
from evaluation import evaluation
from slim.inception_model import inception_v3
from slim.inception_model import loss as inception_losses
from models.our import our_fusion
from models import pairwise_losses
FLAGS = tf.flags.FLAGS

# Where to find data
tf.flags.DEFINE_string('base_path', None, 'base_path')
tf.flags.DEFINE_string('sticker_path', 'npy_stickers', 'sticker_path.')
tf.flags.DEFINE_string('data_path', 'release_train.json', 'sticker_path.')
tf.flags.DEFINE_string('test_path', 'release_val.json', 'sticker_path.')
tf.flags.DEFINE_string('vocab_path', 'vocab', 'sticker_path.')
tf.flags.DEFINE_string('emoji_vocab_path', 'emoji_vocab', 'sticker_path.')
tf.flags.DEFINE_string('inception_ckpt', 'inception_v3_new.ckpt', 'sticker_path.')
tf.flags.DEFINE_integer('dataset_size', 320168, 'minibatch size')

# Where to save output
tf.flags.DEFINE_string('proj_name', 'sticker_classify_2', 'name of project.')
tf.flags.DEFINE_string('log_root', 'logs_2', 'Root directory for all logging.')
tf.flags.DEFINE_string('exp_name', None, 'Name for experiment. ')
tf.flags.DEFINE_enum('mode', 'train', ['train', 'decode', 'eval', 'auto_decode'], '手工运行auto deocde不会有输出')
tf.flags.DEFINE_boolean('single_pass', False, '')
tf.flags.DEFINE_string('current_source_code_zip', None, "current_source_code_zip")

# Hyperparameters
tf.flags.DEFINE_enum('optimizer', 'adam', ['adam', 'rms'], '手工运行auto deocde不会有输出')
tf.flags.DEFINE_enum('inception_endpoint', 'mixed_17x17x768e', ['mixed_8x8x2048b', 'mixed_17x17x768e'], '手工运行auto deocde不会有输出')
tf.flags.DEFINE_enum('context_encoder', 'transformer', ['bigru', 'transformer'], '手工运行auto deocde不会有输出')

tf.flags.DEFINE_integer('emb_dim', 100, 'dimension of word embeddings')
tf.flags.DEFINE_integer('hidden_dim', 100, 'dimension of word embeddings')

tf.flags.DEFINE_integer('batch_size', 32, 'minibatch size')
tf.flags.DEFINE_integer('sticker_height', 128, 'minibatch size')
tf.flags.DEFINE_integer('sticker_weight', 128, 'minibatch size')
tf.flags.DEFINE_integer('context_len', 15, 'minibatch size')
tf.flags.DEFINE_integer('sticker_candidates', 10, 'minibatch size')
tf.flags.DEFINE_integer('max_enc_steps', 30, 'max timesteps of encoder (max source text tokens)')
tf.flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary.')
tf.flags.DEFINE_integer('emoji_vocab_size', 300, 'Size of vocabulary.')
tf.flags.DEFINE_integer('emoji_classify_train_steps', 1000, 'Size of vocabulary.')
tf.flags.DEFINE_integer('auto_test_step', 800, 'Size of vocabulary.')
tf.flags.DEFINE_float('lr', 0.0001, 'learning rate')
tf.flags.DEFINE_float('dropout', 0.1, 'The probability to drop units from the outputs')
tf.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')
tf.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.flags.DEFINE_boolean('add_emoji_classify_loss', True, "plot the gradients on tensorboard")


tf.flags.DEFINE_integer('device', None, '')


tf.flags.mark_flag_as_required("data_path")
tf.flags.mark_flag_as_required("exp_name")
tf.flags.mark_flag_as_required("dataset_size")
tf.flags.mark_flag_as_required("emoji_vocab_path")
tf.flags.mark_flag_as_required("vocab_path")


class StickerClassify:

    def _add_placeholders(self):
        """Add placeholders to the graph. These are entry points for any input data."""
        self._sticker_pix = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.sticker_candidates, FLAGS.sticker_height, FLAGS.sticker_weight, 3],
                                           name='sticker_pix')
        self._sticker_alt = tf.placeholder(tf.int32, [FLAGS.batch_size, FLAGS.sticker_candidates], name='sticker_alt')
        self._sticker_selection_label = tf.placeholder(tf.int32, [FLAGS.batch_size], name='sticker_selection_label')
        self._ground_truth_sticker_alt = tf.placeholder(tf.int32, [FLAGS.batch_size], name='ground_truth_sticker_alt')

        self._enc_context_batch = tf.placeholder(tf.int32, [FLAGS.batch_size, FLAGS.context_len, FLAGS.max_enc_steps], name='enc_context_batch')
        self._enc_context_lens = tf.placeholder(tf.int32, [FLAGS.batch_size, FLAGS.context_len], name='enc_context_lens')
        self._enc_context_number = tf.placeholder(tf.int32, [FLAGS.batch_size], name='enc_context_number')
        self._enc_context_padding_mask = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.context_len, FLAGS.max_enc_steps],
                                                name='enc_context_padding_mask')

    def _make_feed_dict(self, batch: Batch):
        """Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

        Args:
          batch: Batch object
          just_enc: Boolean. If True, only feed the parts needed for the encoder.
        """
        feed_dict = {}
        feed_dict[self._sticker_pix] = batch.sticker_pix
        feed_dict[self._sticker_alt] = batch.sticker_alt
        feed_dict[self._sticker_selection_label] = batch.sticker_selection_label
        feed_dict[self._ground_truth_sticker_alt] = batch.ground_truth_sticker_alt
        feed_dict[self._enc_context_batch] = batch.enc_context_batch
        feed_dict[self._enc_context_lens] = batch.enc_context_lens
        feed_dict[self._enc_context_padding_mask] = batch.enc_context_padding_mask
        feed_dict[self._enc_context_number] = batch.enc_context_number
        return feed_dict

    def _add_train_op(self):
        """Sets self._train_op, the op to run for training."""
        loss_to_minimize = self._loss
        tvars = tf.trainable_variables()

        tf.summary.scalar('loss/minimize_loss', loss_to_minimize)
        tf.summary.scalar('loss/sticker_selection_loss', self.sticker_selection_loss)

        if FLAGS.add_emoji_classify_loss:
            grads = tf.gradients(loss_to_minimize, tvars)
            emoji_grads = tf.gradients(self.emoji_classification_loss, tvars)
        else:
            grads = tf.gradients(self.sticker_selection_loss, tvars)
            emoji_grads = None

        learning_rate = tf.train.polynomial_decay(FLAGS.lr, self.global_step,
                                                  FLAGS.dataset_size / FLAGS.batch_size * 5,
                                                  FLAGS.lr / 10)
        tf.summary.scalar('loss/learning_rate', learning_rate)
        if FLAGS.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif FLAGS.optimizer == 'rms':
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
        else:
            raise NotImplementedError('%s not implement' % FLAGS.optimizer)
        if FLAGS.add_emoji_classify_loss:
            self._emoji_classify_stage = optimizer.apply_gradients(
                [(g, v) for g, v in zip(emoji_grads, tvars) if 'image_encoder' in v.name], global_step=self.global_step)
        else:
            self._emoji_classify_stage = tf.no_op()
        self._merge_train_stage = optimizer.apply_gradients(
            [(g, v) for g, v in zip(grads, tvars)], global_step=self.global_step)

    def build_graph(self):
        tf.logging.info('Building graph...')
        t0 = time.time()
        self._add_placeholders()
        self.global_epoch = tf.get_variable('epoch_num', [], initializer=tf.constant_initializer(1, tf.int32),
                                            trainable=False, dtype=tf.int32)
        self.add_epoch_op = tf.assign_add(self.global_epoch, 1)
        with tf.device("/gpu:%d" % FLAGS.device if FLAGS.device != '' and FLAGS.device >= 0 else "/cpu:0"):
            self._add_model()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if FLAGS.mode == 'train':
            self._add_train_op()
        self._summaries = tf.summary.merge_all()
        t1 = time.time()
        tf.logging.info('Time to build graph: %i seconds', t1 - t0)

    def run_train_step(self, sess, batch, merge_train=False, summary=False):
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'train_op': self._merge_train_stage if merge_train else self._emoji_classify_stage,
            'loss': self._loss,
            'sticker_logits': self.norm_sticker_logits,
            'global_step': self.global_step,
            'global_epoch': self.global_epoch,
        }
        if summary:
            to_return['summaries'] = self._summaries
        result = sess.run(to_return, feed_dict)
        return result

    def run_decode_step(self, sess, batch):
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'loss': self._loss,
            'sticker_predicts': self.sticker_predicts,
            'sticker_logits': self.norm_sticker_logits,
            'global_step': self.global_step,
            'global_epoch': self.global_epoch,
        }
        result = sess.run(to_return, feed_dict)
        return result

    def _add_model(self):
        with tf.variable_scope('image_encoder') as scope:
            self.reshaped_pix = tf.reshape(self._sticker_pix, [-1, FLAGS.sticker_height, FLAGS.sticker_weight, 3])
            tf.logging.info('start building image encoder')
            emoji_logits, endpoints, sticker_encoded = inception_v3(self.reshaped_pix, num_classes=FLAGS.emoji_vocab_size, is_training=FLAGS.mode == 'train')
            sticker_pix_feature = endpoints[FLAGS.inception_endpoint]  # (batch*candidate, 6, 6, 768)

            # calculate loss with regularization
            auxiliary_logits = endpoints['aux_logits']
            inception_losses((emoji_logits, auxiliary_logits), tf.reshape(self._sticker_alt, [-1]))
            losses = tf.get_collection('_losses')

            # Calculate the total loss for the current tower.
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_loss = tf.add_n(losses + regularization_losses, name='total_loss')

            # Compute the moving average of all individual losses and the total loss.
            loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
            loss_averages_op = loss_averages.apply(losses + [total_loss])

            with tf.control_dependencies([loss_averages_op]):
                total_loss = tf.identity(total_loss)
            self.emoji_classification_loss = total_loss
            sticker_encoded = tf.reshape(sticker_encoded, [FLAGS.batch_size, FLAGS.sticker_candidates, -1])
            tf.logging.info('image encoder finish')
        with tf.variable_scope('embedding'):
            embedding = tf.get_variable('embedding', [FLAGS.vocab_size, FLAGS.emb_dim])
            emb_context = tf.nn.embedding_lookup(embedding, self._enc_context_batch)
        with tf.variable_scope('context_encoder'):
            tf.logging.info('start building context encoder')
            context_encoded_states, context_final_state = context_encoder(emb_context, self._enc_context_lens)
        with tf.variable_scope('image_context_fusion'):
            tf.logging.info('start building image context fusion')
            sticker_logits, self.explanation_weight, self.utterance_explanation_weight = our_fusion(sticker_encoded, sticker_pix_feature,
                                        context_encoded_states, context_final_state, self._enc_context_number)

        tf.logging.info('calculating losses')
        self.sticker_selection_loss = pairwise_losses.pairwise_hinge_loss(
            labels=tf.one_hot(self._sticker_selection_label, FLAGS.sticker_candidates),
            logits=sticker_logits)
        self._loss = self.emoji_classification_loss + self.sticker_selection_loss

        tf.logging.info('calculating predictions')

        self.norm_sticker_logits = tf.nn.softmax(sticker_logits)
        self.sticker_predicts = tf.cast(tf.argmax(self.norm_sticker_logits, -1), tf.int32)


def setup_training(model, batcher, emoji_vocab):
    """Does setup before starting training (run_training)"""
    train_dir = os.path.join(FLAGS.log_root, "train")
    if not os.path.exists(train_dir): os.makedirs(train_dir)

    epoch_ckpt_dir = os.path.join(FLAGS.log_root, "epoch_ckpt")
    if not os.path.exists(epoch_ckpt_dir): os.makedirs(epoch_ckpt_dir)

    model.build_graph()  # build the graph
    saver = tf.train.Saver(max_to_keep=3)  # keep 3 checkpoints at a time
    epoch_saver = tf.train.Saver(max_to_keep=99)  # keep 3 checkpoints at a time
    pretrain_saver = tf.train.Saver(var_list={v.name[:-2]: v for v in tf.trainable_variables() if v.name.startswith('image_encoder') and 'logits' not in v.name})  # keep 3 checkpoints at a time
    sv = tf.train.Supervisor(logdir=train_dir,
                             is_chief=True,
                             saver=saver,
                             summary_op=None,
                             save_summaries_secs=60,
                             save_model_secs=600,
                             global_step=model.global_step)
    summary_writer = sv.summary_writer
    tf.logging.info("Preparing or waiting for session...")
    sess_context_manager = sv.prepare_or_wait_for_session(config=get_config())
    tf.logging.info("Created session.")
    tf.logging.info("Loading pretrained parameters.")
    pretrain_saver.restore(sess_context_manager, FLAGS.inception_ckpt)
    try:
        run_training(model, batcher, sess_context_manager, summary_writer, epoch_saver, emoji_vocab)
    except KeyboardInterrupt:
        tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
        sv.stop()


def start_auto_decode_proc(epoch_num=None):
    def run_command(command, stdout=None):
        if stdout is None:
            with open(os.devnull, 'w') as devnull:
                child = subprocess.Popen(command, shell=True, stdout=devnull)
                return child
        else:
            child = subprocess.Popen(command, shell=True, stdout=stdout)
            return child
    # 创建decode所需flags
    flag_str = ''
    except_key = ['mode', 'data_path', 'log_root', 'h', 'help', 'helpfull', 'helpshort', 'device', 'vocab_path', 'emoji_vocab_path',
                  'test_path']
    for key, val in FLAGS.__flags.items():
        val = val._value
        if key not in except_key and val is not None:
            flag_str += '--%s=%s ' % (key, val)
        elif key == 'mode':
            flag_str += '--mode=decode '
        elif key == 'data_path':
            flag_str += '--data_path=%s ' % os.path.abspath(FLAGS.test_path)
        elif key == 'test_path':
            flag_str += '--test_path=%s ' % os.path.abspath(FLAGS.test_path)
        elif key == 'vocab_path':
            flag_str += '--vocab_path=%s ' % os.path.abspath(FLAGS.vocab_path)
        elif key == 'emoji_vocab_path':
            flag_str += '--emoji_vocab_path=%s ' % os.path.abspath(FLAGS.emoji_vocab_path)
        elif key == 'log_root':
            flag_str += '--log_root=%s ' % os.path.abspath(os.path.join(FLAGS.log_root, '../'))
        elif key == 'device':
            flag_str += '--device=%d ' % get_free_gpu()
    if epoch_num is not None:
        flag_str += '--auto_decode_epoch_num=%d ' % epoch_num

    # 解压train code压缩包
    source_code_path = os.path.join(os.path.abspath(os.path.dirname(FLAGS.current_source_code_zip)), 'train_code')
    if os.path.exists(source_code_path):
        shutil.rmtree(source_code_path)
    zip_ref = zipfile.ZipFile(FLAGS.current_source_code_zip, 'r')
    zip_ref.extractall(source_code_path)
    zip_ref.close()
    tf.logging.info('unzip source code finish!')

    run_file_path = os.path.join(source_code_path, 'sticker_classification.py')
    tf.logging.debug(' '.join([sys.executable, run_file_path, flag_str]))
    child = run_command(' '.join([sys.executable, run_file_path, flag_str]))
    sys.stderr.write(' '.join([sys.executable, run_file_path, flag_str]) + '\n')
    sys.stderr.flush()


def run_training(model, batcher, sess_context_manager, summary_writer, epoch_saver, emoji_vocab):
    """Repeatedly runs training iterations, logging loss to screen and writing summaries"""
    tf.logging.info("starting run_training")
    tf.get_default_graph().finalize()
    with sess_context_manager as sess:
        train_step = None
        while True:
            batch = batcher.next_batch()
            if batch is None:
                break
            summary_flag = False
            if train_step is not None and train_step % 20 == 0:
                summary_flag = True
            t0 = time.time()
            if FLAGS.add_emoji_classify_loss:  # 只有在multi task时才会涉及两段训练
                results = model.run_train_step(sess, batch, summary=summary_flag, merge_train=train_step is not None and train_step > FLAGS.emoji_classify_train_steps)
            else:  # 不multi task时 merge train的op只有sticker选择的loss
                results = model.run_train_step(sess, batch, summary=summary_flag, merge_train=True)
            t1 = time.time()

            train_step = results['global_step']
            train_epoch = results['global_epoch']

            if 'summaries' in results:
                summaries = results['summaries']
                summary_writer.add_summary(summaries, train_step)
                summary_writer.flush()

            if train_step * FLAGS.batch_size > train_epoch * FLAGS.dataset_size:
                epoch_num = sess.run(model.add_epoch_op)

            if train_step % 20 == 0 and FLAGS.mode == 'train':
                loss = results['loss']
                tf.logging.info('epoch: %d | step: %d | loss: %.3f | time: %.3f',
                                train_epoch, train_step, loss, t1 - t0)
                if not np.isfinite(loss):
                    raise Exception("Loss is not finite. Stopping.")

            if train_step % FLAGS.auto_test_step == 0:
                start_auto_decode_proc()


def run_test(model, batcher, emoji_vocab):
    import numpy as np
    model.build_graph()
    fp = None
    sess = tf.Session(config=data.get_config())
    saver = tf.train.Saver()
    ckpt_path = data.load_ckpt(saver, sess)
    tf.get_default_graph().finalize()
    nn = datetime.now().strftime('%m-%d-%H-%M')
    all_logits = []
    all_one_hot_lables = []
    train_step = train_epoch = 0
    counter = 0
    while True:
        batch = batcher.next_batch()
        if batch is None:
            break
        results = model.run_decode_step(sess, batch)
        train_step = results['global_step']
        train_epoch = results['global_epoch']
        if fp is None:
            fp = open(os.path.join(FLAGS.log_root, "predict-ep%dstep%d-" % (train_epoch, train_step) + nn + '.txt'), 'w', encoding='utf8')
        sticker_logits = results['sticker_logits']
        one_hot_targets = np.eye(FLAGS.sticker_candidates)[batch.sticker_selection_label]
        for i in range(FLAGS.batch_size):
            logits = sticker_logits[i].tolist()
            one_hot = one_hot_targets[i].tolist()
            for l, o in zip(logits, one_hot):
                fp.write('%.3f\t%d\n' % (l, o))
                all_logits.append(l)
                all_one_hot_lables.append(o)

            counter += 1

    fp.close()
    metrics = evaluation(all_logits, all_one_hot_lables, FLAGS.sticker_candidates)
    metrics_string = []
    for k in list(metrics.keys()):
        metrics_string.append('%s:%.3f' % (k, metrics[k]))
    mf = open(os.path.join(FLAGS.log_root, "metric.txt"), 'a+', encoding='utf8')
    mf.write('%s\n' % ' | '.join(metrics_string))
    mf.close()


def main(unused_argv):
    FLAGS.sticker_path = os.path.join(FLAGS.base_path, FLAGS.sticker_path)
    FLAGS.data_path = os.path.join(FLAGS.base_path, FLAGS.data_path)
    FLAGS.test_path = os.path.join(FLAGS.base_path, FLAGS.test_path)
    FLAGS.vocab_path = os.path.join(FLAGS.base_path, FLAGS.vocab_path)
    FLAGS.emoji_vocab_path = os.path.join(FLAGS.base_path, FLAGS.emoji_vocab_path)
    FLAGS.inception_ckpt = os.path.join(FLAGS.base_path, FLAGS.inception_ckpt)
    if 'decode' in FLAGS.mode:
        FLAGS.single_pass = True
        FLAGS.batch_size = 4
        FLAGS.dataset_size = -1
    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)  # create a vocabulary
    emoji_vocab = Vocab(FLAGS.emoji_vocab_path, FLAGS.emoji_vocab_size)  # create a vocabulary
    if 'decode' in FLAGS.mode:
        batcher = Batcher(FLAGS.test_path, vocab, emoji_vocab, single_pass=FLAGS.single_pass)
    else:
        batcher = Batcher(FLAGS.data_path, vocab, emoji_vocab, single_pass=FLAGS.single_pass)

    if 'decode' in FLAGS.mode:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import logging
        log = logging.getLogger('tensorflow')
        log.setLevel(logging.FATAL)
        for h in log.handlers:
            log.removeHandler(h)
        log.addHandler(logging.NullHandler())

    # GPU tricks
    if FLAGS.device is None:
        index_of_gpu = get_available_gpu()
        if index_of_gpu < 0:
            index_of_gpu = ''
        FLAGS.device = index_of_gpu
        tf.logging.info(bcolors.OKGREEN + 'using {}'.format(FLAGS.device) + bcolors.ENDC)
    else:
        index_of_gpu = FLAGS.device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index_of_gpu)
    tf.set_random_seed(5683)  # a seed value for randomness

    if len(unused_argv) != 1:
        raise Exception("Problem with flags: %s" % unused_argv)

    tf.logging.set_verbosity(tf.logging.INFO)  # choose what level of logging you want
    tf.logging.info('Starting sticker classification in %s mode...', (FLAGS.mode))

    # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
    if not os.path.exists(FLAGS.log_root):
        if FLAGS.mode == "train":
            os.makedirs(FLAGS.log_root)
        else:
            raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (FLAGS.log_root))

    # If single_pass=True, check we're in decode mode
    if FLAGS.single_pass and 'decode' not in FLAGS.mode:
        raise Exception("The single_pass flag should only be True in decode mode")

    ######################
    # save parameters and python script
    ######################
    export_json = {}
    for key, val in FLAGS.__flags.items():
        val = val._value
        export_json[key] = val
    # save parameters
    tf.logging.info('saving parameters')
    current_time_str = datetime.now().strftime('%m-%d-%H-%M')
    json_para_file = open(os.path.join(FLAGS.log_root, 'flags-' + current_time_str + '-' + FLAGS.mode + '.json'), 'w')
    json_para_file.write(json.dumps(export_json, indent=4) + '\n')
    json_para_file.close()
    # save python source code
    FLAGS.current_source_code_zip = os.path.abspath(os.path.join(FLAGS.log_root, 'source_code_bak-' + current_time_str + '-' + FLAGS.mode + '.zip'))
    tf.logging.info('saving source code: %s', FLAGS.current_source_code_zip)
    python_list = glob.glob('./*.py')
    zip_file = zipfile.ZipFile(FLAGS.current_source_code_zip, 'w')
    for d in python_list:
        zip_file.write(d)
    for d in glob.glob('slim/*.py'):
        zip_file.write(d)
    for d in glob.glob('models/*.py'):
        zip_file.write(d)
    zip_file.close()

    tf.set_random_seed(111)  # a seed value for randomness

    if FLAGS.mode == 'train':
        tf.logging.info("creating model...")
        model = StickerClassify()
        setup_training(model, batcher, emoji_vocab)
    elif FLAGS.mode == 'decode':
        tf.logging.info("creating model...")
        model = StickerClassify()
        run_test(model, batcher, emoji_vocab)
    else:
        raise ValueError("The 'mode' flag must be one of train/eval/decode/auto_decode")


if __name__ == '__main__':
    tf.app.run()
