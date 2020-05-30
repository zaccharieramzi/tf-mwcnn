import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D,
    Layer,
    BatchNormalization,
    Activation,
)


DEFAULT_N_FILTERS_PER_SCALE = [160, 256, 256]
DEFAULT_N_CONVS_PER_SCALE = [4] * 3


class MWCNNConvBlock(Layer):
    def __init__(self, n_filters=256, kernel_size=3, **kwargs):
        super(MWCNNConvBlock, self).__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.conv = Conv2D(
            self.n_filters,
            self.kernel_size,
            padding='same',
            use_bias=False,
        )
        self.bn = BatchNormalization()
        self.activation = Activation('relu')

    def call(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.bn(outputs)
        outputs = self.activation(outputs)
        return outputs

class DWT(Layer):
    def call(self, inputs):
        # taken from
        # https://github.com/lpj-github-io/MWCNNv2/blob/master/MWCNN_code/model/common.py#L65
        x01 = inputs[:, 0::2] / 2
        x02 = inputs[:, 1::2] / 2
        x1 = x01[:, :, 0::2]
        x2 = x02[:, :, 0::2]
        x3 = x01[:, :, 1::2]
        x4 = x02[:, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return tf.concat((x_LL, x_HL, x_LH, x_HH), axis=-1)

class IWT(Layer):
    def call(self, inputs):
        r = 2
        in_batch, in_channel, in_height, in_width = tf.shape(inputs)
        #print([in_batch, in_channel, in_height, in_width])
        out_batch, out_channel, out_height, out_width = in_batch, int(
            in_channel / (r ** 2)), r * in_height, r * in_width
        x1 = inputs[:, 0:out_channel, :, :] / 2
        x2 = inputs[:, out_channel:out_channel * 2, :, :] / 2
        x3 = inputs[:, out_channel * 2:out_channel * 3, :, :] / 2
        x4 = inputs[:, out_channel * 3:out_channel * 4, :, :] / 2


        h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

        return h


class MWCNN(Model):
    def __init__(
            self,
            n_scales=3,
            kernel_size=3,
            n_filters_per_scale=DEFAULT_N_FILTERS_PER_SCALE,
            n_convs_per_scale=DEFAULT_N_CONVS_PER_SCALE,
            **kwargs,
        ):
        super(MWCNN, self).__init__(**kwargs)
        self.n_scales = n_scales
        self.kernel_size = kernel_size
        self.n_filters_per_scale = n_filters_per_scale
        self.n_convs_per_scale = n_convs_per_scale
        self.conv_blocks_per_scale = [
            [MWCNNConvBlock(
                n_filters=self.n_filters_for_conv_for_scale(i_scale, i_conv),
                kernel_size=self.kernel_size,
            ) for i_conv in range(self.n_convs_per_scale[i_scale] * 2)]
            for i_scale in range(self.n_scales)
        ]
        # the last convolution is without bn and relu, and also has only
        # 4 filters, that's why we treat it separately
        self.conv_blocks_per_scale[0][-1] = Conv2D(
            4,
            self.kernel_size,
            padding='same',
            use_bias=True,
        )
        # TODO: implement these 2 wavelet operators
        self.pooling = DWT()
        self.unpooling = IWT()

    def n_filters_for_conv_for_scale(self, i_scale, i_conv):
        n_filters = self.n_filters_per_scale[i_scale]
        if i_conv == self.n_convs_per_scale[i_scale] * 2 - 1:
            n_filters *= 4
        return n_filters

    def call(self, inputs):
        last_feature_for_scale = []
        current_feature = inputs
        for i_scale in range(self.n_scales):
            current_feature = self.pooling(current_feature)
            for i_conv in range(self.n_convs_per_scale[i_scale]):
                conv = self.conv_blocks_per_scale[i_scale][i_conv]
                current_feature = conv(current_feature)
            last_feature_for_scale.append(current_feature)
        for i_scale in range(self.n_scales - 1, -1, -1):
            if i_scale != self.n_scales - 1:
                current_feature = self.unpooling(current_feature)
                current_feature = current_feature + last_feature_for_scale[i_scale]
            n_convs = self.n_convs_per_scale[i_scale]
            for i_conv in range(n_convs, 2*n_convs):
                conv = self.conv_blocks_per_scale[i_scale][i_conv]
                current_feature = conv(current_feature)
        outputs = inputs + self.unpooling(current_feature)
        return outputs
