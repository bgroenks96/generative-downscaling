import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp
import numpy as np

def scaled_rmse_metric(scale=1.0, axis=0):
    @tf.function
    def scaled_rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square((y_true - y_pred)*scale), axis=axis))
    return scaled_rmse

def scaled_mae_metric(scale=1.0, axis=0):
    @tf.function
    def scaled_mae(y_true, y_pred):
        return K.mean(K.abs(y_true - y_pred)*scale, axis=axis)
    return scaled_mae

def sparse_scaled_mae_metric(offset=0.0, scale=1.0, epsilon=1.0E-3):
    eps = K.constant(np.array(epsilon))
    s = K.constant(scale)
    @tf.function
    def sparse_scaled_mae(y_true, y_pred):
        y_true = s*y_true + offset
        y_pred = s*y_pred + offset
        # absolute error over all points
        diff = K.abs(y_true - y_pred)
        # compute mask over points we want to ignore
        y_zero = K.cast(K.abs(y_true) < eps, dtype='float32')
        d_zero = K.cast(diff < eps, dtype='float32')
        # combine masks with logical and/multiply op
        mask = y_zero*d_zero
        # sum over masks to count the number of negligble values
        n_m = K.sum(mask)
        # compute the total number of values as the product of shape
        n = K.cast(K.prod(K.shape(diff)), dtype='float32')
        # return the adjusted mean ignoring zero values with low error
        return K.sum(diff) / (n-n_m)
    return sparse_scaled_mae

def bias_metric(offset=0.0, scale=1.0, axis=0):
    @tf.function
    def bias(y_true, y_pred):
        y_true = scale*y_true + offset
        y_pred = scale*y_pred + offset
        return K.mean(y_pred - y_true, axis=axis)
    return bias

def sparse_bias_metric(offset=0.0, scale=1.0, epsilon=1.0E-3):
    eps = K.constant(np.array(epsilon))
    s = K.constant(scale)
    @tf.function
    def sparse_scaled_mae(y_true, y_pred):
        y_true = s*y_true + offset
        y_pred = s*y_pred + offset
        # absolute error over all points
        diff = y_pred - y_true
        # compute mask over points we want to ignore
        y_zero = K.cast(K.abs(y_true) < eps, dtype='float32')
        d_zero = K.cast(K.abs(diff) < eps, dtype='float32')
        # combine masks with logical and/multiply op
        mask = y_zero*d_zero
        # sum over masks to count the number of negligble values
        n_m = K.sum(mask)
        # compute the total number of values as the product of shape
        n = K.cast(K.prod(K.shape(diff)), dtype='float32')
        # return the adjusted mean ignoring zero values with low error
        return K.sum(diff) / (n-n_m)
    return sparse_scaled_mae

def qqrsq_metric(num_quantiles=100, axis=0):
    @tf.function
    def rsquared(yt, yp):
        unexplained_error = tf.math.reduce_sum((yt - yp)**2, axis=axis)
        total_error = tf.math.reduce_sum((yt - tf.math.reduce_mean(yt, axis=axis))**2, axis=axis)
        return 1.0 - (unexplained_error / total_error)
    @tf.function
    def qqrsq(y_true, y_pred):
        yt_quantiles = tfp.stats.quantiles(y_true, num_quantiles, axis=axis)
        yp_quantiles = tfp.stats.quantiles(y_pred, num_quantiles, axis=axis)
        r2 = rsquared(yt_quantiles, yp_quantiles)
        return r2
    return qqrsq

def stratified_skill_score_metric(class_centers, offset=0.0, scale=1.0, drop_n=0):
    """
    Stratified skill score metric. Assigns bin/class numbers given by 'class_centers' to each
    predicted real value, shifts/scales by offset/scale, and drops the first 'drop_n' classes.
    Computed as a normalized difference between bin counts over all values. Output in range [0,1].
    """
    classes = K.constant(np.array(class_centers))
    indices = K.constant(np.array(range(len(class_centers))), dtype='int32')
    @tf.function
    def skill_score(y_true, y_pred):
        y_true = y_true*scale + offset
        y_pred = y_pred*scale + offset
        c_true = K.cast(K.argmin(K.abs(K.expand_dims(y_true, axis=-1) - classes), axis=-1), dtype='int32')
        c_pred = K.cast(K.argmin(K.abs(K.expand_dims(y_pred, axis=-1) - classes), axis=-1), dtype='int32')
        c_true_per_class = K.cast(K.equal(K.expand_dims(c_true, axis=-1), indices), dtype='float32')
        c_pred_per_class = K.cast(K.equal(K.expand_dims(c_pred, axis=-1), indices), dtype='float32')
        c_true_freqs = tf.math.log1p(K.sum(c_true_per_class, axis=[i for i in range(len(K.int_shape(y_true)))])[drop_n:])
        c_pred_freqs = tf.math.log1p(K.sum(c_pred_per_class, axis=[i for i in range(len(K.int_shape(y_pred)))])[drop_n:])
        freq_total = K.sum(c_true_freqs, axis=None)
        c_true_dist = c_true_freqs / freq_total
        c_pred_dist = c_pred_freqs / freq_total
        return K.sum(K.minimum(c_true_dist, c_pred_dist))
    return skill_score

def stratified_kld_metric(class_centers, offset=0.0, scale=1.0):
    from keras.losses import kld
    classes = K.constant(np.array(class_centers))
    indices = K.constant(np.array(range(len(class_centers))), dtype='int32')
    @tf.function
    def stratified_kld(y_true, y_pred):
        y_true = y_true*scale + offset
        y_pred = y_pred*scale + offset
        c_true = K.cast(K.argmin(K.abs(K.expand_dims(y_true, axis=-1) - classes), axis=-1), dtype='int32')
        c_pred = K.cast(K.argmin(K.abs(K.expand_dims(y_pred, axis=-1) - classes), axis=-1), dtype='int32')
        c_true_per_class = K.cast(K.equal(K.expand_dims(c_true, axis=-1), indices), dtype='float32')
        c_pred_per_class = K.cast(K.equal(K.expand_dims(c_pred, axis=-1), indices), dtype='float32')
        c_true_freqs = K.sum(c_true_per_class, axis=[i for i in range(len(K.int_shape(y_true)))])
        c_pred_freqs = K.sum(c_pred_per_class, axis=[i for i in range(len(K.int_shape(y_pred)))])
        freq_total = K.sum(c_true_freqs, axis=None)
        c_true_dist = c_true_freqs / freq_total
        c_pred_dist = c_pred_freqs / freq_total
        return kld(c_true_dist, c_pred_dist)
    return stratified_kld

def stratified_accuracy_metric(class_centers, offset=0.0, scale=1.0):
    classes = K.constant(np.array(class_centers))
    @tf.function
    def stratified_acc(y_true, y_pred):
        y_true = y_true*scale + offset
        y_pred = y_pred*scale + offset
        c_true = K.cast(K.argmin(K.abs(K.expand_dims(y_true, axis=-1) - classes), axis=-1), dtype='int32')
        c_pred = K.cast(K.argmin(K.abs(K.expand_dims(y_pred, axis=-1) - classes), axis=-1), dtype='int32')
        freq_correct = K.sum(K.cast(K.equal(c_true, c_pred), dtype='float32'))
        freq_total = K.cast(K.prod(K.shape(c_true)), dtype='float32')
        return freq_correct / freq_total
    return stratified_acc
    
def stratified_f1_metric(class_centers, offset=0.0, scale=1.0):
    classes = K.constant(np.array(class_centers))
    indices = K.constant(np.array(range(len(class_centers))), dtype='int32')
    @tf.function
    def stratified_f1(y_true, y_pred):
        y_true = y_true*scale + offset
        y_pred = y_pred*scale + offset
        c_true = K.cast(K.argmin(K.abs(K.expand_dims(y_true, axis=-1) - classes), axis=-1), dtype='int32')
        c_pred = K.cast(K.argmin(K.abs(K.expand_dims(y_pred, axis=-1) - classes), axis=-1), dtype='int32')
        c_true_per_class = K.cast(K.equal(K.expand_dims(c_true, axis=-1), indices), dtype='int32')
        c_pred_per_class = K.cast(K.equal(K.expand_dims(c_pred, axis=-1), indices), dtype='int32')
        c_pred_pos_per_class = K.cast(K.equal(c_pred_per_class, 1), dtype='int32')
        c_pred_neg_per_class = K.cast(K.equal(c_pred_per_class, 0), dtype='int32')
        tp_per_class = c_true_per_class * c_pred_per_class
        tp = K.sum(K.cast(tp_per_class, dtype='float32'), axis=(0,1,2,3))
        fp_per_class = (1-c_true_per_class) * c_pred_pos_per_class
        fp = K.sum(K.cast(fp_per_class, dtype='float32'), axis=(0,1,2,3))
        fn_per_class = c_true_per_class * c_pred_neg_per_class
        fn = K.sum(K.cast(fn_per_class, dtype='float32'), axis=(0,1,2,3))
        f1_per_class = 2*tp / (2*tp + fp + fn)
        return K.mean(f1_per_class)
    return stratified_f1
    
def stratified_mcc_metric(class_centers, offset=0.0, scale=1.0):
    classes = K.constant(np.array(class_centers))
    indices = K.constant(np.array(range(len(class_centers))), dtype='int32')
    @tf.function
    def stratified_mcc(y_true, y_pred):
        import tensorflow as tf
        k = K.shape(classes)[0] # k classes
        y_true = y_true*scale + offset
        y_pred = y_pred*scale + offset
        c_true = K.cast(K.argmin(K.abs(K.expand_dims(y_true, axis=-1) - classes), axis=-1), dtype='int32')
        c_pred = K.cast(K.argmin(K.abs(K.expand_dims(y_pred, axis=-1) - classes), axis=-1), dtype='int32')
        c_true_per_class = K.cast(K.equal(K.expand_dims(c_true, axis=-1), indices), dtype='int32')
        c_pred_per_class = K.cast(K.equal(K.expand_dims(c_pred, axis=-1), indices), dtype='int32')
        c_pred_pos_per_class = c_pred_per_class
        c_pred_neg_per_class = 1 - c_pred_per_class
        N = K.cast(K.prod(K.shape(c_true)), dtype='float32')
        tp_per_class = c_true_per_class * c_pred_pos_per_class
        tp = tf.clip_by_value(K.sum(K.cast(tp_per_class, dtype='float32'), axis=(1,2,3)), 0, N)
        fp_per_class = (1-c_true_per_class) * c_pred_pos_per_class
        fp = tf.clip_by_value(K.sum(K.cast(fp_per_class, dtype='float32'), axis=(1,2,3)), 0, N)
        tn_per_class = (1-c_true_per_class) * c_pred_neg_per_class
        tn = tf.clip_by_value(K.sum(K.cast(tn_per_class, dtype='float32'), axis=(1,2,3)), 0, N)
        fn_per_class = c_true_per_class * c_pred_neg_per_class
        fn = tf.clip_by_value(K.sum(K.cast(fn_per_class, dtype='float32'), axis=(1,2,3)), 0, N)
        # calculate denominator
        d = K.sqrt(tp+fp)*K.sqrt(tp+fn)*K.sqrt(tn+fp)*K.sqrt(tn+fn)
        # add 1 to zero values; do nothing to positive values
        d_zero_mask = K.cast(K.less_equal(0.0, d), dtype='float32')
        d += d_zero_mask
        mcc_per_class = (tp*tn - fp*fn) / d #K.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        mcc_nonzero_mask = K.cast(mcc_per_class > 0, dtype='float32')
        nonzero_class_count = tf.clip_by_value(K.sum(mcc_nonzero_mask, axis=-1), 1.0, K.cast(k, dtype='float32'))
        batch_mcc_scores = K.sum(mcc_per_class, axis=-1) / nonzero_class_count
        nonzero_batch_mask = K.cast(batch_mcc_scores > 0, dtype='float32')
        nonzero_batch_count = tf.clip_by_value(K.sum(nonzero_batch_mask), 1.0, K.cast(k, dtype='float32'))
        return K.sum(batch_mcc_scores) / nonzero_batch_count
    return stratified_mcc
