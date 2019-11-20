import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

method_dict = {'nearest': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
               'bilinear': tf.image.ResizeMethod.BILINEAR,
               'bicubic': tf.image.ResizeMethod.BICUBIC,
               'area': tf.image.ResizeMethod.AREA}

@tf.function
def upsample(x, scale, method):
    x_shape = tf.shape(x)
    return tf.image.resize(x, size=scale*x_shape[1:-1], method=method_dict[method])

@tf.function
def extract_patches(x, k, stride):
    x_shape = tf.shape(x)
    x_patch = tf.image.extract_patches(x, sizes=(1,k,k,1), strides=(1,stride,stride,1), rates=(1,1,1,1), padding='VALID')
    output_shape = tf.shape(x_patch)
    return tf.reshape(x_patch, (output_shape[0]*output_shape[1]*output_shape[2], k, k, x_shape[-1]))

class DeconvNet:
    def __init__(self, conv_model, activation='relu'):
        from keras.layers import Conv2D
        self.activation = activation
        self.conv_model = conv_model
        self.conv_layers = [layer for layer in conv_model.layers if isinstance(layer, Conv2D)]
        self.deconv_layers = self._init_deconv_layers()
        
    def _init_deconv_layers(self):
        from keras.layers import Conv2DTranspose
        deconv_layers = []
        # create a transposed convolution layer for each convolution layer, in reverse order
        for conv in reversed(self.conv_layers):
            conv_t = Conv2DTranspose(conv.filters, conv.kernel_size, strides=conv.strides, padding=conv.padding,
                                     use_bias=False, data_format=conv.data_format, dilation_rate=conv.dilation_rate)
            deconv_layers.append(conv_t)
        deconv_layers[-1].filters = 1
        return deconv_layers
    
    def evaluate_at(self, layer_idx, inputs):
        from keras.layers import Input, Activation
        conv_layer = self.conv_layers[layer_idx]
        conv_inputs = self.conv_model.inputs
        conv_output = conv_layer.output
        conv_fn = K.function(conv_inputs, [conv_output])
        act = Activation(self.activation)
        deconv_input = Input(shape=conv_layer.output_shape[1:])
        deconv_output = deconv_input
        for deconv_layer, conv in zip(self.deconv_layers[-layer_idx-1:], reversed(self.conv_layers[:layer_idx+1])):
            conv_k, conv_b = conv.get_weights()
            deconv_output = deconv_layer(act(deconv_output) - conv_b)
            deconv_layer.set_weights([conv_k])
        deconv_fn = K.function([deconv_input], [deconv_output])
        conv_output_values = conv_fn([inputs])[0]
        def eval_filter_fn(f_idx):
            mask = np.zeros(conv_output_values.shape)
            mask[:,:,:,f_idx] = np.ones(conv_output_values.shape[:-1])
            mask = K.constant(mask)
            return deconv_fn([conv_output_values*mask])[0]
        return eval_filter_fn

def get_activations(X, input_layers, layer):
    inputs = [K.learning_phase()] + input_layers
    layer_fn = K.function(inputs, [layer.output])
    layer_out = layer_fn([0, X])[0]
    z = layer_out
    return z

def optimize_filter_inputs(filter_idx, layer, model, input_shape, aux_inputs=[], grad_step=1, 
                           step_decay=5.0E-2, tol=1.0E-1, verbose=False):
    """
    Maximizes activation of a single filter from a given layer in model via gradient ascent on the input image.
    Useful for visualizing learned filters.
    See: https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
    """
    assert len(input_shape) == len(model.inputs[0].shape) == 4
    wt, ht, c = input_shape[1], input_shape[2], input_shape[3]
    # create loss function to maximize mean activation of filter
    layer_out = layer.output
    loss = K.mean(layer_out[:,:,:,filter_idx])
    # compute gradients w.r.t input tensor
    grads = K.gradients(loss, model.inputs[0])[0]
    # normalize gradients
    grads /= K.sqrt(K.mean(K.square(grads))) + K.epsilon()
    # create gradient update function
    update = K.function(model.inputs, [loss, grads])
    # sample input image from normal distribution at given center/scale
    sample_img = np.random.uniform(0.0, 1.0, size=input_shape)
    step = grad_step
    prev_loss = -float('inf')
    loss_delta = float('inf')
    itr = 0
    while loss_delta > tol:
        itr += 1
        loss, grads = update([sample_img] + aux_inputs)
        deltas = grads*step
        sample_img += deltas
        step -= step_decay
        loss_delta = loss - prev_loss
        prev_loss = loss
        if verbose:
            print(f'[iteration {itr}] loss: {loss}  loss delta: {loss_delta} avg gradient: {np.mean(grads)}')
    return sample_img

def relative_spatial_diff(shape, k):
    """
    Creates a Keras Model that transforms a spatial map using local averaging; i.e. each output
    point y_ij is the difference between the input point X_ij and the local average of its neighborhood
    of size k.
    
    shape : input shape
    k     : kernel window size
    """
    from keras.layers import Input, Conv2D, Subtract
    from keras.initializers import Constant
    assert len(shape) == 3
    input_0 = Input(shape=shape)
    conv_avg = Conv2D(shape[-1], k, padding='same', kernel_initializer=Constant(1.0/k**2), activation='linear', name='avg_conv', trainable=False)
    out_diff = Subtract()([input_0, conv_avg(input_0)])
    model = Model(inputs=input_0, outputs=out_diff)
    return model

def conv_neighborhoods(X, k=3):
    """
    Finds k x k nearest neighbors via a predefined convolution function.
    Deprecated in favor of simpler solutions.
    """
    assert len(X.shape) == 4
    # neighborhood convolution can only be computed on one channel
    # at a time; additional channels would require depthwise convolution
    assert X.shape[-1] == 1
    assert k >= 3
    # neighborhood window must be odd
    assert k % 2 == 1
    output_shape = (*X.shape[:-1], k**2)
    # init tensorflow placeholder for X
    x = tf.placeholder(tf.float32, shape=X.shape, name='x')
    def _build_conv_op():
        with tf.variable_scope('neighbor_conv') as scope:
            one_hot_inds = np.array([[j if j != k**2 // 2 else -1 for j in range(i, i+k)] for i in range(0, k**2, k)])
            filters = tf.one_hot(one_hot_inds, depth=output_shape[-1], axis=-1)
            # add input channel dimension
            filters = tf.expand_dims(filters, axis=-2)
            return tf.nn.conv2d(x, filters, strides=[1,1,1,1], padding='SAME'), filters
    conv_op, filters = _build_conv_op()
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        result = session.run(conv_op, {x: X})
        fs = filters.eval().squeeze()
        print(fs.shape)
        print(fs)
        return result