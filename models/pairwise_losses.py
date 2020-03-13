from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.losses import losses as core_losses

# The smallest probability that is used to derive smallest logit for invalid or
# padding entries.
_EPSILON = 1e-10


def pairwise_hinge_loss(labels,
                         logits,
                         weights=None,
                         lambda_weight=None,
                         reduction=core_losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
                         name=None):
    """Computes the pairwise hinge loss for a list.
    The hinge loss is defined as Hinge(l_i > l_j) = max(0, 1 - (s_i - s_j)). So a
    correctly ordered pair has 0 loss if (s_i - s_j >= 1). Otherwise the loss
    increases linearly with s_i - s_j. When the list_size is 2, this reduces to
    the standard hinge loss.
    Args:
      labels: A `Tensor` of the same shape as `logits` representing graded
        relevance.
      logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
        ranking score of the corresponding item.
      weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
        weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
        weights.
      lambda_weight: A `_LambdaWeight` object.
      reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
        reduce training loss over batch.
      name: A string used as the name for this loss.
    Returns:
      An op for the pairwise hinge loss.
    """

    def _loss(logits):
        """The loss of pairwise logits with l_i > l_j."""
        # TODO(xuanhui, pointer-team): Consider pass params object into the loss and
        # put a margin here.
        return nn_ops.relu(1. - logits)

    with ops.name_scope(name, 'pairwise_hinge_loss', (labels, logits, weights)):
        return _pairwise_loss(
            _loss, labels, logits, weights, lambda_weight, reduction=reduction)



def _sort_and_normalize(labels, logits, weights=None):
    """Sorts `labels` and `logits` and normalize `weights`.
    Args:
      labels: A `Tensor` of the same shape as `logits` representing graded
        relevance.
      logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
        ranking score of the corresponding item.
      weights: A scalar, a `Tensor` with shape [batch_size, 1], or a `Tensor` with
        the same shape as `labels`.
    Returns:
      A tuple of (sorted_labels, sorted_logits, sorted_weights).
    """
    labels = ops.convert_to_tensor(labels)
    logits = ops.convert_to_tensor(logits)
    logits.get_shape().assert_has_rank(2)
    logits.get_shape().assert_is_compatible_with(labels.get_shape())
    weights = 1.0 if weights is None else ops.convert_to_tensor(weights)
    weights = array_ops.ones_like(labels) * weights
    _, topn = array_ops.unstack(array_ops.shape(logits))

    # Only sort entries with valid labels that are >= 0.
    scores = array_ops.where(
        math_ops.greater_equal(labels, 0.), logits,
        -1e-6 * array_ops.ones_like(logits) + math_ops.reduce_min(
            logits, axis=1, keepdims=True))
    sorted_labels, sorted_logits, sorted_weights = sort_by_scores(
        scores, [labels, logits, weights], topn=topn)
    return sorted_labels, sorted_logits, sorted_weights


def _pairwise_comparison(sorted_labels,
                         sorted_logits,
                         sorted_weights,
                         lambda_weight=None):
    r"""Returns pairwise comparison `Tensor`s.
    Given a list of n items, the labels of graded relevance l_i and the logits
    s_i, we sort the items in a list based on s_i and obtain ranks r_i. We form
    n^2 pairs of items. For each pair, we have the following:
                          /
                          | 1   if l_i > l_j
    * `pairwise_labels` = |
                          | 0   if l_i <= l_j
                          \
    * `pairwise_logits` = s_i - s_j
                           /
                           | 0              if l_i <= l_j,
    * `pairwise_weights` = | |l_i - l_j|    if lambda_weight is None,
                           | lambda_weight  otherwise.
                           \
    The `sorted_weights` is item-wise and is applied non-symmetrically to update
    pairwise_weights as
      pairwise_weights(i, j) = w_i * pairwise_weights(i, j).
    This effectively applies to all pairs with l_i > l_j. Note that it is actually
    symmetric when `sorted_weights` are constant per list, i.e., listwise weights.
    Args:
      sorted_labels: A `Tensor` with shape [batch_size, list_size] of labels
        sorted.
      sorted_logits: A `Tensor` with shape [batch_size, list_size] of logits
        sorted.
      sorted_weights: A `Tensor` with shape [batch_size, list_size] of item-wise
        weights sorted.
      lambda_weight: A `_LambdaWeight` object.
    Returns:
      A tuple of (pairwise_labels, pairwise_logits, pairwise_weights) with each
      having the shape [batch_size, list_size, list_size].
    """
    # Compute the difference for all pairs in a list. The output is a Tensor with
    # shape [batch_size, list_size, list_size] where the entry [-1, i, j] stores
    # the information for pair (i, j).
    pairwise_label_diff = array_ops.expand_dims(
        sorted_labels, 2) - array_ops.expand_dims(sorted_labels, 1)
    pairwise_logits = array_ops.expand_dims(
        sorted_logits, 2) - array_ops.expand_dims(sorted_logits, 1)
    pairwise_labels = math_ops.to_float(math_ops.greater(pairwise_label_diff, 0))
    is_label_valid = is_label_valid_func(sorted_labels)
    valid_pair = math_ops.logical_and(
        array_ops.expand_dims(is_label_valid, 2),
        array_ops.expand_dims(is_label_valid, 1))
    # Only keep the case when l_i > l_j.
    pairwise_weights = pairwise_labels * math_ops.to_float(valid_pair)
    # Apply the item-wise weights along l_i.
    pairwise_weights *= array_ops.expand_dims(sorted_weights, 2)
    if lambda_weight is not None:
        pairwise_weights *= lambda_weight.pair_weights(sorted_labels)
    else:
        pairwise_weights *= math_ops.abs(pairwise_label_diff)
    pairwise_weights = array_ops.stop_gradient(
        pairwise_weights, name='weights_stop_gradient')
    return pairwise_labels, pairwise_logits, pairwise_weights


def _pairwise_loss(loss_fn,
                   labels,
                   logits,
                   weights=None,
                   lambda_weight=None,
                   reduction=core_losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
    """Template to compute pairwise loss.
    Args:
      loss_fn: A function that computes loss from the pairwise logits with l_i >
        l_j.
      labels: A `Tensor` of the same shape as `logits` representing graded
        relevance.
      logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
        ranking score of the corresponding item.
      weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
        weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
        weights.
      lambda_weight: A `_LambdaWeight` object.
      reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
        reduce training loss over batch.
    Returns:
      An op for the pairwise loss.
    """
    sorted_labels, sorted_logits, sorted_weights = _sort_and_normalize(
        labels, logits, weights)
    _, pairwise_logits, pairwise_weights = _pairwise_comparison(
        sorted_labels, sorted_logits, sorted_weights, lambda_weight)
    if lambda_weight is not None:
        # For LambdaLoss with relative rank difference, the scale of loss becomes
        # much smaller when applying LambdaWeight. This affects the training can
        # make the optimal learning rate become much larger. We use a heuristic to
        # scale it up to the same magnitude as standard pairwise loss.
        pairwise_weights *= math_ops.to_float(array_ops.shape(sorted_labels)[1])
    return core_losses.compute_weighted_loss(
        loss_fn(pairwise_logits), weights=pairwise_weights, reduction=reduction)


def is_label_valid_func(labels):
    """Returns a boolean `Tensor` for label validity."""
    labels = ops.convert_to_tensor(labels)
    return math_ops.greater_equal(labels, 0.)


def sort_by_scores(scores, features_list, topn=None):
    """Sorts example features according to per-example scores.
    Args:
      scores: A `Tensor` of shape [batch_size, list_size] representing the
        per-example scores.
      features_list: A list of `Tensor`s with the same shape as scores to be
        sorted.
      topn: An integer as the cutoff of examples in the sorted list.
    Returns:
      A list of `Tensor`s as the list of sorted features by `scores`.
    """
    scores = ops.convert_to_tensor(scores)
    scores.get_shape().assert_has_rank(2)
    batch_size, list_size = array_ops.unstack(array_ops.shape(scores))
    if topn is None:
        topn = list_size
    topn = math_ops.minimum(topn, list_size)
    _, indices = nn_ops.top_k(scores, topn, sorted=True)
    list_offsets = array_ops.expand_dims(
        math_ops.range(batch_size) * list_size, 1)
    # The shape of `indices` is [batch_size, topn] and the shape of
    # `list_offsets` is [batch_size, 1]. Broadcasting is used here.
    gather_indices = array_ops.reshape(indices + list_offsets, [-1])
    output_shape = array_ops.stack([batch_size, topn])
    # Each feature is first flattened to a 1-D vector and then gathered by the
    # indices from sorted scores and then re-shaped.
    return [
        array_ops.reshape(
            array_ops.gather(array_ops.reshape(feature, [-1]), gather_indices),
            output_shape) for feature in features_list
    ]
