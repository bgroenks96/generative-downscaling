import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adamax
from normalizing_flows.models import JointFlowLVM, VariationalModel, adversarial
from normalizing_flows.models.losses import spatial_mae
from normalizing_flows.flows import Invert
from normalizing_flows.flows.glow import GlowFlow, coupling_nn_glow

def build_jflvm(input_shape, scale, layers, depth, prior=tfp.distributions.Normal(loc=0.0, scale=1.0),
                min_filters=32, max_filters=256, dnet_layers=3, dnet_filters=64, aux_error_type='average'):
    flow_hr = Invert(GlowFlow(num_layers=layers, depth=depth, name='glow_hr',
                              coupling_nn_ctor=coupling_nn_glow(min_filters=min_filters, max_filters=max_filters)))
    flow_lr = Invert(GlowFlow(num_layers=layers, depth=depth, name='glow_lr',
                              coupling_nn_ctor=coupling_nn_glow(min_filters=min_filters, max_filters=max_filters)))
    dx = adversarial.PatchDiscriminator(input_shape[1:], n_layers=dnet_layers, n_filters=dnet_filters)
    dy = adversarial.PatchDiscriminator(input_shape[1:], n_layers=dnet_layers, n_filters=dnet_filters)
    c = {'average': 1.0, 'sum': scale**2, 'none': 0.0}[aux_error_type]
    model_joint = JointFlowLVM(flow_lr, flow_hr, dx, dy, input_shape=input_shape, prior=prior,
                               Gy_aux_loss=spatial_mae(scale, c=c))
    return model_joint

def build_variational(input_shape, encoder, layers, depth, min_filters=32, max_filters=256, lr=1.0E-4):
    flow = Invert(GlowFlow(num_layers=layers, depth=depth,
                           coupling_nn_ctor=coupling_nn_glow(min_filters=min_filters, max_filters=max_filters)))
    model = VariationalModel(encoder, normal(), flow)
    model.compile(optimizer=Adamax(lr=lr), output_shape=input_shape)
    return model
