"""
Downscaling CNNs
"""

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    UpSampling2D,
    Conv2D,
    Conv2DTranspose,
    Dense,
    SpatialDropout2D,
    BatchNormalization,
    Activation,
    Concatenate,
    Reshape,
    Flatten,
    Lambda,
    PReLU
)
from tensorflow.keras.layers import add as add_layers
from tensorflow.keras.losses import mse, mae, logcosh
from tensorflow.keras.initializers import Constant
from tensorflow.keras.regularizers import l2, l1
import tensorflow.keras.backend as K
import numpy as np

def mixed_spatial_mse(input_lr, beta=0.5):
    """
    Returns a loss function that combines pointwise MSE with a "spatial" MSE
    that measures the distance between the high-res spatial averages and the
    low res input.
    """
    kernel = tf.ones((scale+1,scale+1,1,1)) / (scale+1)**2
    def loss(y_true, y_pred):
        mse_pointwise = K.mean(mse(y_true, y_pred), axis=(1,2))
        true_vol = K.conv2d(y_pred, kernel, strides=(scale, scale), padding='same')
        pred_vol = K.conv2d(y_pred, kernel, strides=(scale, scale), padding='same')
        mse_spatial = K.mean(mse(true_vol - input_lr, pred_vol - input_lr), axis=(1,2))
        return beta*mse_pointwise + (1. - beta)*mse_spatial
    return loss

def create_srcnn(wt=None, ht=None, scale=None, c_in = 1, c_out = 1, c_aux=0, f_1 = 64, f_2 = 32, kernels=(9,1,5), output_activity='linear',
                 hidden_activity='elu', loss='mse', optimizer='adam', alpha = 1.0E-5, beta=0.9, dropout_rate=0.1,
                 metrics=[], run_opts=None, run_metadata=None):
    """
    Builds a modified version of SR-CNN (Dong et al. 2015).

    scale           : if not None, upsample the input by factor 'scale' via transposed convolutions
    c_in            : number of primary input channels (variables)
    c_out           : number of primary output channels (variables)
    c_aux           : number of axuiliary input channels (variables)
    f_1             : number of filters in extraction layer
    f_2             : number of filters in mapping layer
    kernels         : 3-tuple of kernels sizes for each layer; defaults to 9-1-5 as specified by Dong et al.
    output_activity : activation function name for the output; defaults to 'linear'
    hidden_activity : activation function name for hidden layers; defautls to exponential linear unit, 'elu'
    loss            : loss function name; defaults to 'mse', use 'spatial_mse' for custom mixed spatial MSE
    optimizer       : optimizer for training; defaults to 'adam'
    alpha           : L2 regularization coeff for all layers
    beta            : mixing fraction for spatial_mse loss; ignored otherwise
    dropout_rate    : fraction of feature maps in hidden layers to randomly "drop out" or ignore per training iteration
    metrics         : an array of Keras-compatible metric functions
    run_opts        : Tensorflow run options
    run_metadata    : Tensorflow run metadata
    """
    assert len(kernels) == 3
    input_0 = Input(shape=(wt, ht, c_in))
    aux_input = Input(shape=(None, None, c_aux))
    if scale is not None:
        x = UpSampling2D(scale, interpolation='bilinear')(input_0)
    else:
        x = input_0
    if c_aux > 0:
        x = Concatenate(axis=-1)([x, aux_input])
    k1, k2, k3 = kernels
    # patch extraction/representation
    patch_conv = Conv2D(f_1, k1, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(alpha), activation=hidden_activity, name='patch_conv')
    # nonlinear mapping
    map_conv = Conv2D(f_2, k2, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(alpha), activation=hidden_activity, name='map_conv')
    # reconstruction
    rec_conv = Conv2D(c_out, k3, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(alpha), activation=output_activity, name='rec_conv')
    # dropout layer
    dropout = SpatialDropout2D(rate=dropout_rate)
    # compose all layers
    output_0 = rec_conv(dropout(map_conv(dropout(patch_conv(x)))))
    # consolidate all inputs
    inputs = [input_0, aux_input] if c_aux > 0 else input_0
    srcnn = Model(inputs=inputs, outputs=output_0)
    loss = mixed_spatial_mse(input_0, beta) if loss == 'spatial_mse' else loss
    srcnn.compile(optimizer=optimizer, loss=loss, metrics=metrics, options=run_opts, run_metadata=run_metadata)
    return srcnn

def create_vdsrcnn(scale=None, c_in=1, c_out=1, c_aux=1, f=64, kernels=3, module_layers=3, n_layers=20, res_net=True,
                   output_activity='linear', hidden_activity='relu', output_init='glorot_uniform',
                   alpha=1.0E-5, beta=0.9, multi_scale_inputs=False, dropout_rate=0.1):
    """
    Builds a modified VDSR-CNN network (Kim et al. 2016) with residual connections per-module.

    scale           : if not None, upsample the input by factor 'scale' via transposed convolutions; must be power of two
    c_in            : number of primary input channels (variables)
    c_out           : number of primary output channels (variables)
    c_aux           : number of auxiliary input channels for multi-scale inputs
    f_1             : number of filters all layers; defaults to 64
    kernels         : kernel size for all layers; defaults to 3
    module_layers   : number of convolutions per module; defaults to 3
    n_layers        : total number of convolutions in the full network; ignored when scale != None
    res_net         : if True, use residual chunks from ResNet architecture; defaults to False, i.e. residual added only at the end
    output_activity : activation function name for the output; defaults to 'linear'
    hidden_activity : activation function name for hidden layers; defautls to exponential linear unit, 'elu'
    loss            : loss function name; defaults to 'mse', use 'spatial_mse' for custom mixed spatial MSE
    optimizer       : optimizer for training; defaults to 'adam'
    alpha           : L2 regularization coeff for all layers
    beta            : mixing fraction for spatial_mse loss; ignored otherwise
    dropout_rate    : fraction of feature maps in hidden layers to randomly "drop out" or ignore per training iteration
    metrics         : an array of Keras-compatible metric functions
    run_opts        : Tensorflow run options
    run_metadata    : Tensorflow run metadata
    """
    log_scale = None if scale is None else np.log2(scale)
    assert log_scale is None or np.isclose(log_scale, np.floor(log_scale))
    log_scale = int(log_scale)
    inputs = [Input(shape=(None,None,c_in))]
    if log_scale is not None and multi_scale_inputs:
        inputs += [Input(shape=(None, None, c_aux)) for i in range(log_scale)]
    k = kernels
    act = lambda x: Activation(hidden_activity)(x)
    dropout = lambda x: SpatialDropout2D(dropout_rate)(x)
    conv_layer = lambda i, f, x: Conv2D(f, k, padding='same', kernel_initializer='he_normal',
                                        kernel_regularizer=l2(alpha), name=f'conv_{i}')(x)
    conv_t_layer = lambda i, f, x: Conv2DTranspose(f, k, strides=2, padding='same', kernel_initializer='he_normal',
                                                   kernel_regularizer=l2(alpha), name=f'conv_transpose_{i}')(x)
    concat = lambda x: Concatenate(axis=-1)(x)
    def vdsrcnn_module(x, i):
        """
        Represents a single chunk of VDSR-CNN layers.

        x     : input tensor
        i     : starting layer index
        f_out : number of filters on the last layer
        """
        # apply scaling, if configured
        if scale is not None:
            y0 = act(conv_t_layer(i, f, x))
        else:
            y0 = x
        if scale is not None and multi_scale_inputs:
            y0 = conv_layer(f'aux_{i}', f, concat([y0, inputs[i]]))
        # apply module layers
        y = y0
        for j in range(module_layers):
            y = dropout(act(conv_layer(i*module_layers+j, f, y)))
        if res_net:
            y = add_layers([y0, y])
        return y
    # build full network
    x = inputs[0]
    # initial conv layer
    y = conv_layer(0, f, x)
    iters = range(1, n_layers - 1, module_layers) if scale is None else range(1, int(log_scale) + 1)
    for i in iters:
        y = vdsrcnn_module(y, i)
    # final conv layer
    y = Conv2D(c_out, k, padding='same', kernel_initializer=output_init, kernel_regularizer=l2(alpha), name=f'conv_final')(y)
    if scale is None:
        # add final residual; skip for multi-scale architecture
        x_ = Lambda(lambda x: x[:,:,:,:1])(x)
        y = add_layers([x_, y])
    vdsrcnn = Model(inputs=inputs, outputs=y)
    return vdsrcnn

def create_fsrcnn(scale, c=1, d=48, s=16, m=3, kernels=(9,1,5), output_activity='linear', hidden_activity='elu',
                 loss='mse', optimizer='adam', alpha = 1.0E-5, beta=0.9, dropout_rate=0.1, metrics=[],
                 run_opts=None, run_metadata=None):
    """
    Builds a modified version of FastSR-CNN (Dong et al. 2016). Adds residual connections and stacked deconvolutions.

    c               : number of primary input/output channels (variables)
    d               : number of filters in patch extraction and expansion layers
    s               : number of filters in shrinking/mapping layers
    m               : number of mapping layers
    kernels         : 3-tuple of kernels sizes for each layer; defaults to 9-1-5 as specified by Dong et al.
    output_activity : activation function name for the output; defaults to 'linear'
    hidden_activity : activation function name for hidden layers; defautls to exponential linear unit, 'elu'
    loss            : loss function name; defaults to 'mse', use 'spatial_mse' for custom mixed spatial MSE
    optimizer       : optimizer for training; defaults to 'adam'
    alpha           : L2 regularization coeff for all layers
    beta            : mixing fraction for spatial_mse loss; ignored otherwise
    dropout_rate    : fraction of feature maps in hidden layers to randomly "drop out" or ignore per training iteration
    metrics         : an array of Keras-compatible metric functions
    run_opts        : Tensorflow run options
    run_metadata    : Tensorflow run metadata
    """
    log_scale = np.log2(scale)
    assert np.isclose(log_scale, np.floor(log_scale)), 'scale must be a power of 2'
    input_0 = Input(shape=(None,None,c))
    patch_k, shrink_k, map_k, deconv_k = kernels
    act = lambda x: Activation('elu')(x)
    batch_norm = lambda x: BatchNormalization()(x)
    dropout = lambda x: SpatialDropout2D(dropout_rate)(x)
    chan_concat = lambda x: Concatenate(axis=-1)(x)
    patch_conv = Conv2D(d, patch_k, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(alpha), name='patch_conv')
    shrink_conv = Conv2D(s, shrink_k, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(alpha), name='shrink_conv')
    map_conv = lambda i, x: Conv2D(s, map_k, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(alpha), name=f'map_conv_{i}')(x)
    expand_conv = Conv2D(d, shrink_k, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(alpha), name='expand_conv')
    deconv = lambda i, f, x: Conv2DTranspose(f, deconv_k, strides=2, padding='same', kernel_initializer='he_normal',
                                             kernel_regularizer=l1(alpha), name=f'deconv_{i}')(x)
    output_conv = Conv2D(1, 1, padding='same', kernel_regularizer=l2(alpha), activation=output_activity, name='linear_conv')
    # low res conv encoder: input -> feature extraction -> shrinking + residual
    def encoder(x):
        x_p = act(patch_conv(x))
        z = act(shrink_conv(x_p))
        return add_layers([z, x]), x_p
    # non-linear mapping units
    def mapping(z_in, i):
        z_out = dropout(act(batch_norm(map_conv(i, z_in))))
        return z_out
    # hi res conv decoder: mapping -> expansion
    def decoder(z):
        y = dropout(act(batch_norm(expand_conv(z))))
        return y
    # compose layers
    z0, x_p = encoder(input_0)
    z = z0
    for i in range(m):
        z = mapping(z, i)
    # residual connection across mapping layers
    z = add_layers([z, z0])
    y = decoder(z)
    # residual connection from patch layer
    y = batch_norm(add_layers([y, x_p]))
    #y = deconv(0, s, act(y))
    for i in range(int(log_scale)):
        y = deconv(i, scale // (2**i), act(y))
    output_0 = output_conv(act(y))
    fsrcnn = Model(inputs=input_0, outputs=output_0)
    loss = mixed_spatial_mse(input_0, beta) if loss == 'spatial_mse' else loss
    fsrcnn.compile(optimizer='adam', loss=loss, metrics=metrics, options=run_opts, run_metadata=run_metadata)
    return fsrcnn

def create_bmg_cnn10(img_wt, img_ht, scale=2, c_in=1, c_out=1, filters=(50,25,10), kernel_sizes=(3,3,3)):
    """
    Creates the CNN10 model from Bano-Medina et al. (2019)
    https://doi.org/10.5194/gmd-2019-278
    """
    input_0 = Input(shape=(img_wt, img_ht, c_in))
    conv_1 = Conv2D(filters[0], kernel_sizes[0], activation='relu', padding='same')
    conv_2 = Conv2D(filters[1], kernel_sizes[1], activation='relu', padding='same')
    conv_3 = Conv2D(filters[2], kernel_sizes[2], activation='relu', padding='same')
    flatten = Flatten()
    dense_out = Dense(img_wt*img_ht*c_out*scale**2, activation='linear', kernel_initializer='zeros')
    reshape_out = Reshape((img_wt*scale, img_ht*scale, c_out))
    output_0 = reshape_out(dense_out(flatten(conv_3(conv_2(conv_1(input_0))))))
    return Model(inputs=input_0, outputs=output_0)
