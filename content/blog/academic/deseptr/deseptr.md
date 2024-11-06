---
math: true
---

# ViTs for Wearable Sensor Spectrogram Analysis with DeSepTr

This is a blog that I have wrote in conjunction with my paper: [A Data Efficient Vision Transformer for Robust Human Activity Recognition from the Spectrograms of Wearable Sensor Data](https://ieeexplore.ieee.org/document/10208059/citations?tabFilter=papers#citations). If you would like access feel free to reach out! 

In this blog, I am going to provide the code for implementing the Data Efficient Separable Transformer (DeSepTr). This is a modification of the [Separable Transformer (SepTr)](https://arxiv.org/abs/2203.09581) [1] which was developed for analyzing wearable sensor spectrograms with computer vision (CV) techniques. The research application of this paper was for wearable sensor analysis, however, I'm sure someone can reapply this work to audio and compare to the original SepTr.

---

## Background

I presented this work at the 2023 IEEE Statistical Signal Processing Workshop (SSP) in Hanoi, Vietnam. The presentation format was a [poster](/blog/academic/deseptr/poster.pdf) with a short paper published in the proceedings. Visiting Hanoi was an amazing academic, cultural, and historical experience! I had an amazing time meeting collegues from all over the world and discussing novel ideas related to signal procesisng.

<table>
  <tr>
    <td><img src="/blog/academic/deseptr/vietnam1.jpg" alt="Hanoi, Vietnam" width="300"/></td>
    <td><img src="/blog/academic/deseptr/vietnam2.jpg" alt="IEEE SSP 2024" width="300"/></td>
  </tr>
</table>

---

## Introduction

### Vision Transformers

[Vision Transformers (ViTs)](https://arxiv.org/pdf/2010.11929) [2] are a fast emerging framework for computer vision (CV) based on the [transformer](https://arxiv.org/pdf/1706.03762) [3] that leverages multi-head attention (MHA) to weight the importance of each input. This enables the full image-context to be utilised and overcomes issues faced by convolutional neural networks (CNNs). A limitation of ViTs that can cause performance degradations is the weak inductive bias of the model which can cause over-fitting to the training data.

<div style="text-align: center;">
  <img src="/blog/academic/deseptr/vit.png" alt="vit">
  <figcaption>Figure 1. Diagram of the ViT [2]</figcaption>
</div>
<br>

In our experiments, we implemented the ViT in the TensorFlow framework with the following code:

```python
import tensorflow as tf

class ViT(tf.keras.layers.Layer):

    def __init__(
      self, 
      input_dim, 
      patch_size, 
      embed_dim, 
      num_layers, 
      num_heads, 
      mlp_dims, 
      dropout_rate, 
      **kwargs
    ):
        super(ViT, self).__init__(**kwargs)
        self.num_patches = input_dim[0] * input_dim[1]
        self.embed_dim = embed_dim
        self.input_norm = tf.keras.layers.LayerNormalization(
          epsilon=1e-6
        )
        self.patches = PatchProjection(
          self.embed_dim, patch_size
        )
        self.class_emb = self.add_weight(
            "class_emb", 
            shape=(1, 1, self.embed_dim),
            initializer='uniform',
            trainable=True
        )
        self.pos_emb = self.add_weight(
            "position_emb",
            shape=(1, self.num_patches+1, self.embed_dim),
            initializer='uniform',
            trainable=True
        )
        self.encoder = Encoder(
            num_patches=self.num_patches, 
            embed_dim=self.embed_dim,
            input_pos_emb=self.pos_emb,
            input_class_emb=self.class_emb
        )
        self.transformer_block = tf.keras.Sequential([
            TransformerBlock(
                embed_dim=self.embed_dim, 
                num_heads=num_heads, 
                mlp_dims=mlp_dims, 
                dropout_rate=dropout_rate
            )
            for _ in range(0, num_layers)
        ]) 

    def call(self, inputs, training=None):
        input_dim = tf.shape(inputs)
        x = self.input_norm(inputs)
        patch_x = self.patches(x)
        patch_x = tf.reshape(patch_x, (input_dim[0], self.num_patches, self.embed_dim))
        encoded_x = self.encoder(patch_x)
        transformer_x = self.transformer_block(encoded_x, training=training)
        cls = transformer_x[:, 0, :]

        return cls

class Encoder(tf.keras.layers.Layer):
    def __init__(
        self, 
        num_patches,
        embed_dim,
        input_pos_emb,
        input_class_emb  
    ):
        super(Encoder, self).__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.pos_emb = input_pos_emb
        self.class_emb = input_class_emb

    def call(self, x, training=None):
        batch_size = tf.shape(x)[0]
        pos_emb = tf.broadcast_to(
            self.pos_emb, [batch_size, self.num_patches+1, self.embed_dim]
        )
        class_emb = tf.broadcast_to(
            self.class_emb, [batch_size, 1, self.embed_dim]
        )
        encoded = tf.concat([class_emb, x], axis=1)
        encoded = encoded + pos_emb

        return encoded


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(
      self,
      embed_dim, 
      num_heads, 
      mlp_dims, 
      dropout_rate
    ):
        super(TransformerBlock, self).__init__()
        self.norm0 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha0 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            dropout=dropout_rate
        )
        layers = [[tf.keras.layers.Dense(x), tf.keras.layers.Dropout(dropout_rate)]for x in mlp_dims]
        self.mlp = tf.keras.Sequential([x for sublayer in layers for x in sublayer])
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training):
        inputs_norm = self.norm0(inputs)
        attn_output = self.mha0(inputs_norm, inputs_norm, training=training)
        out0 = attn_output + inputs
        x = self.norm1(out0)
        x = self.mlp(x)

        return x + out0


class PatchProjection(tf.keras.layers.Layer):
    def __init__(
      self, 
      embed_dim,
      patch_size
    ):
        super(PatchProjection, self).__init__()
        self.patch_size = patch_size
        self.projection = tf.keras.layers.Dense(units=embed_dim)

    def call(self, inputs, training=None):
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        projected_patches = self.projection(patches)

        return projected_patches
```

### Why Spectrograms?

Accelerometer and Gyroscope data is typically analyzed in the native time-series format. This allows for supervised machine learning (ML) applications using algorithms, such as: 1D-CNN, RNNs, and LSTM. The Short Time Fourier Transform (STFT) can extract the time-frequency components of the time-series data and effectively provide an image representation of the sensor data.

$$
\textrm{STFT}[x[n]] (m, k)  = \sum_{n = -\infty}^{\infty} x[n] \cdot w[n - mR] \cdot e^{-j \frac{2 \pi}{N_{x}} k n}.
$$

<div style="text-align: center;">
  <img src="/blog/academic/deseptr/spec.jpg" alt="vit">
  <figcaption>Figure 2. Spectrogram representation of accelerometer data.</figcaption>
</div>
<br>


This transformation provides the algorithm with spectral information and opens up an arsenal of extremely powerful CV methods (e.g. ResNet).

We implement the spectrogram conversion as a custom TensorFlow layer:

```python
class SpecGeneration(tf.keras.layers.Layer):
    def __init__(
      self, 
      frame_length, 
      frame_step, 
      **kwargs
    ):
        super().__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step = frame_step

    def call(self, inputs, training=None):
        x = tf.transpose(inputs, perm=[0, 2, 1])
        x = tf.abs(
            tf.signal.stft(
                x,
                frame_length=self.frame_length,
                frame_step=self.frame_step,
                fft_length=self.frame_length-1,
                pad_end=True
            )
        )

        return tf.transpose(x, perm=[0, 3, 2, 1])
```

We implement the custom augmentation protocol in Python:

```python
class SpecAugmentation(tf.keras.layers.Layer):
    '''
    class that implements the random augmentation prior to training
    '''

    def __init__(self, num_sensors, **kwargs):
        super().__init__(**kwargs)
        self.num_sensors = num_sensors
        self.gaussian = tf.keras.layers.GaussianNoise(stddev=1e-3)

    def call(self, inputs, training=None):
        _, v, tau, channels = inputs.shape

        if training:
            x = self.gaussian(inputs)
            x = self.shift_spec(x)
            x = self.time_warp(x, warp_param=4)
            x = self.freq_warp(x, warp_param=4)
            x = self.cut_mix(
              x, time_mask_num=4, time_mask_param=2, freq_mask_num=4, freq_mask_param=2, mix_param=1
            )
            return x
        else:
            return inputs

    def cut_mix(
        self, 
        spec, 
        time_mask_num,
        time_mask_param, 
        freq_mask_num,
        freq_mask_param, 
        mix_param=0.5
    ):

        spec_size = tf.shape(spec)
        v, n = spec_size[1], spec_size[2]

        if time_mask_param > n / 4:
            time_mask_param = int(n / 4)
        
        if freq_mask_param > v / 4:
            freq_mask_param = int(v / 4)

        # shuffled for the cut mix
        shuffle_spec = tf.random.shuffle(spec)

        for _ in range(time_mask_num):

            t = tf.random.uniform([], minval=0, maxval=time_mask_param, dtype=tf.int32)
            n = tf.cast(n, dtype=tf.int32)
            t0 = tf.random.uniform([], minval=0, maxval=n-t, dtype=tf.int32)

            cut_mask = tf.concat((
                tf.zeros(shape=(1, v, n-t0-t, 1)),
                tf.ones(shape=(1, v, t, 1)),
                tf.zeros(shape=(1, v, t0, 1))
            ), 2)

            spec_mask = tf.concat((
                tf.ones(shape=(1, v, n-t0-t, 1)),
                tf.zeros(shape=(1, v, t, 1)),
                tf.ones(shape=(1, v, t0, 1))
            ), 2)

            # add the cut spec together
            spec = (mix_param * tf.math.multiply(cut_mask, shuffle_spec)) + tf.math.multiply(spec, spec_mask)

        for _ in range(freq_mask_num):

            f = tf.random.uniform([], minval=0, maxval=freq_mask_param, dtype=tf.int32)
            v = tf.cast(v, dtype=tf.int32)
            f0 = tf.random.uniform([], minval=0, maxval=v-f, dtype=tf.int32)

            cut_mask = tf.concat((
                tf.zeros(shape=(1, v-f0-f, n, 1)),
                tf.ones(shape=(1, f, n, 1)),
                tf.zeros(shape=(1, f0, n, 1))
            ), 1)

            spec_mask = tf.concat((
                tf.ones(shape=(1, v- f0 - f, n, 1)),
                tf.zeros(shape=(1, f, n, 1)),
                tf.ones(shape=(1, f0, n, 1))
            ), 1)

            # add the cut spec together
            spec = (mix_param * tf.math.multiply(cut_mask, shuffle_spec)) + tf.math.multiply(spec, spec_mask)

        return spec

    def shift_spec(self, spec):
        return tf.roll(spec, shift=random.randint(0, spec.shape[2] // 2), axis=2)

    def time_warp(self, spec, warp_param):

        v, tau = spec.shape[1], spec.shape[2]

        # horizontal line through spectrogram centre
        horiz_line = spec[0][v // 2]

        random_point = horiz_line[random.randrange(start=int(tau / 4), stop=int(tau - (tau / 4)))]

        w = tf.random.uniform(shape=[], minval=-warp_param * tau, maxval=warp_param * tau)

        src_points = [[[v // 2, random_point[0]]]]

        dst_points = [[[v // 2, random_point[0] + w]]]

        warped_spec = sparse_image_warp(spec, src_points, dst_points, num_boundary_points=2)

        return warped_spec[0]

    def freq_warp(self, spec, warp_param):

        spec = tf.transpose(spec, perm=[0, 2, 1, 3])

        v, tau = spec.shape[1], spec.shape[2]

        # horizontal line through spectrogram centre
        horiz_line = spec[0][v // 2]

        random_point = horiz_line[random.randrange(start=int(tau / 4), stop=int(tau - (tau / 4)))]

        w = tf.random.uniform(shape=[], minval=-warp_param, maxval=warp_param)

        src_points = [[[v // 2, random_point[0]]]]

        dst_points = [[[v // 2, random_point[0] + w]]]

        warped_spec = sparse_image_warp(spec, src_points, dst_points, num_boundary_points=2)

        return tf.transpose(warped_spec[0], perm=[0, 2, 1, 3])
```

### Separable Transformer

The Separable Transformer (SepTr) [1] was developed for analyzing audio spectrograms with ViTs. The architecture includes two sequential transformer blocks that process sub-samples of the spectrograms' extracted patches, grouped by the time-interval and frequency-band for the first and second transformer blocks, respectively. When grouped together, the positional embedding and class embedding is repeated for each sub-sample, with the class token being initialized prior to the forward-pass through the vertical transformer, and the mean value computated at the output of the vertical transformer. The key contribution here is that SepTr leverages the time-dependent nature of spectrograms within the ViT framework.

<div style="text-align: center;">
  <img src="/blog/academic/deseptr/septr.png" alt="vit">
  <figcaption>Figure 3. Diagram of the SepTr [1]</figcaption>
</div>

We implement SepTr with the following Python code:

```python
class SepTr(tf.keras.layers.Layer):
    def __init__(
      self, 
      input_dim, 
      patch_size, 
      embed_dim, 
      num_layers, 
      num_heads, 
      mlp_dims, 
      dropout_rate, 
      **kwargs
    ):
        super(SepTr, self).__init__(**kwargs)
        self.class_emb = self.add_weight(
            "class_emb", 
            shape=(1, 1, 1, embed_dim),
            initializer='uniform',
            trainable=True
        )
        self.time_pos_emb = self.add_weight(
            "time_position_emb",
            shape=(1, input_dim[0]+1, 1, embed_dim),
            initializer='uniform',
            trainable=True
        )
        self.freq_pos_emb = self.add_weight(
            "freq_position_emb",
            shape=(1, 1, input_dim[1]+1, embed_dim),
            initializer='uniform',
            trainable=True
        )
        self.time_encoder = TimeEncoder(
            spec_dim=input_dim, 
            embed_dim=embed_dim, 
            input_pos_emb=self.time_pos_emb, 
            input_class_emb=self.class_emb,
        )
        self.freq_encoder = FreqEncoder(
            spec_dim=input_dim, 
            embed_dim=embed_dim, 
            input_pos_emb=self.freq_pos_emb, 
            input_class_emb=self.class_emb,
        )
        self.patch_projection = PatchProjection(embed_dim=embed_dim, patch_size=patch_size)
        self.time_transformer = tf.keras.Sequential(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dims=mlp_dims,
                    dropout_rate=dropout_rate
                ) for _ in range(num_layers)
            ],
            name="time_transformer"
        )
        self.freq_transformer = tf.keras.Sequential(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dims=mlp_dims,
                    dropout_rate=dropout_rate
                ) for _ in range(num_layers)
            ], 
            name="freq_transformer"
        )
        self.pool0 = tf.keras.layers.GlobalAveragePooling1D()
        self.pool1 = tf.keras.layers.GlobalAveragePooling1D()

    def call(self, inputs, training=None):
        patch_x = self.patch_projection(inputs)

        # encode the patches in time batches and rearrange input
        time_encoded_x = self.time_encoder(patch_x)
        time_encoded_x = tf.transpose(time_encoded_x, perm=[0, 2, 1, 3])

        # pass the input through the transformer and decouple from the position embedding
        time_transformer_x = self.time_transformer(time_encoded_x, training=training)
        time_transformer_x = tf.transpose(time_transformer_x, perm=[0, 2, 1, 3])

        # get the mean cls token
        time_mean_cls = self.pool0(time_transformer_x[:, 0, :, :])

        # get the output vector
        output_x = time_transformer_x[:, 1:, :, :]

        # rearrange the axis
        output_x = tf.transpose(output_x, perm=[0, 1, 2, 3])

        # encode the frequencies as batches 
        freq_encoded_x = self.freq_encoder(output_x, time_mean_cls)

        # pass the encoded frequency input to the transformer
        freq_transformer_x = self.freq_transformer(freq_encoded_x, training=training)

        # get the mean cls token
        freq_mean_cls = self.pool1(freq_transformer_x[:, :, 0, :])

        return freq_mean_cls

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dims, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.norm0 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha0 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            attention_axes=(2,3),
            dropout=dropout_rate,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer
        )
        layers = [[tf.keras.layers.Dense(x), tf.keras.layers.Dropout(dropout_rate)]for x in mlp_dims]
        self.mlp = tf.keras.Sequential([x for sublayer in layers for x in sublayer])
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=None):
        inputs_norm = self.norm0(inputs)
        attn_output = self.mha0(inputs_norm, inputs_norm, training=training)
        out0 = attn_output + inputs
        x = self.norm1(out0)
        x = self.mlp(x)

        return x + out0

class TimeEncoder(tf.keras.layers.Layer):
    def __init__(
        self, 
        spec_dim,
        embed_dim, 
        input_pos_emb, 
        input_class_emb, 
    ):
        super(TimeEncoder, self).__init__()
        self.time_dim = spec_dim[1]
        self.freq_dim = spec_dim[0]
        self.embed_dim = embed_dim
        self.input_class_emb = input_class_emb
        self.pos_emb = input_pos_emb

    def call(self, x, optional_class_emb=None, training=None):

        batch_size = tf.shape(x)[0]

        # use the class emb if provided
        if optional_class_emb is not None:
            class_emb = optional_class_emb
            class_emb = tf.expand_dims(tf.expand_dims(class_emb, axis=-1), axis=-1)

        pos_emb = tf.broadcast_to(
            self.pos_emb, [batch_size, self.freq_dim+1, self.time_dim, self.embed_dim]
        )

        class_emb = tf.broadcast_to(
            self.input_class_emb, [batch_size, 1, self.time_dim, self.embed_dim]
        )

        encoded = tf.concat([class_emb, x], axis=1)

        encoded = encoded + pos_emb

        return encoded

class FreqEncoder(tf.keras.layers.Layer):
    def __init__(
        self, 
        spec_dim,
        embed_dim, 
        input_pos_emb, 
        input_class_emb,
    ):
        super(FreqEncoder, self).__init__()
        self.time_dim = spec_dim[1]
        self.freq_dim = spec_dim[0]
        self.embed_dim = embed_dim
        self.class_emb = input_class_emb
        self.pos_emb = input_pos_emb

    def call(self, x, optional_class_emb=None, training=None):

        batch_size = tf.shape(x)[0]

        # use the class emb if provided
        if optional_class_emb is not None:
            class_emb = optional_class_emb
            class_emb = tf.expand_dims(tf.expand_dims(class_emb, axis=1), axis=1)

        class_emb = tf.broadcast_to(
            self.class_emb, [batch_size, self.freq_dim, 1, self.embed_dim]
        )

        distillation_token = tf.broadcast_to(
            self.class_emb, [batch_size, self.freq_dim, 1, self.embed_dim]
        )

        pos_emb = tf.broadcast_to(
            self.pos_emb, [batch_size, self.freq_dim, self.time_dim+1, self.embed_dim]
        )

        encoded = tf.concat([class_emb, x], axis=2)

        encoded = encoded + pos_emb

        return encoded

class PatchProjection(tf.keras.layers.Layer):
    def __init__(self, embed_dim, patch_size):
        super(PatchProjection, self).__init__()
        self.patch_size = patch_size
        self.projection = tf.keras.layers.Dense(units=embed_dim)

    def call(self, inputs, training=None):
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        projected_patches = self.projection(patches)

        return projected_patches
```

---

## Data Efficient Separable Transformer

### Theory

We proposed the Data Efficient Separable Transformer (DeSepTr) as a modification of SepTr that incorporates the knowledge distillation (KD) protocol from the [Data Efficient Image Transformer (DeiT)](https://arxiv.org/abs/2012.12877) [4] and utilizes the convolutional 'teacher' network to provide the embedded representation of data to the 'student' transformer.

<div style="text-align: center;">
  <img src="/blog/academic/deseptr/deit.jpg" alt="deit" style="width: 70%; max-width: 600px;">
  <figcaption>Figure 3. Diagram of DeiT [4]</figcaption>
</div>
<br>

A key detail of DeiT is the inclusion of the distillation token alongside the class token. The predictions from the convolutional 'teacher' model, $y_{t}$ interact with the distillation token following processing with the transformer, $y_{dist}$, to transfer knowledge.

$$
L_{s} = \frac{1}{2} L_{CE} (y, y_{cls}) + \frac{1}{2} L_{CE} (y_{dist}, y_{t})
$$

### Solution

We adapt SepTr following the data efficient image transformer (DeiT) framework, adopting [ResNet-18](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) [5] as the 'teacher' model and utilizing the initial layers as the projection layer for the 'student' transformer. 

The code for the convolutional 'teacher':

```python

def create_resnet_teacher(
  window_length, 
  num_sensors, 
  frame_length, 
  frame_step,
  num_classes
):

  # define the input to the model
  input = tf.keras.Input(shape=(window_length, num_sensors * 3))

  # generate the spectrogram
  spec_x = SpecGeneration(
      frame_length, frame_step
  )(input)

  # augment the sensor data
  x = SpecAugmentation(num_sensors=num_wearables * num_sensors)(spec_x)

  x = tf.keras.layers.Conv2D(
      filters=64, 
      kernel_size=(7,7), 
      strides=2, 
      kernel_initializer='he_normal', 
      padding="same",
      kernel_regularizer=regularizer,
      bias_regularizer=regularizer
  )(x)

  x = tf.keras.layers.Dropout(rate=0.3)(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.ReLU()(x)
  x = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2, padding="same")(x)

  x = ResNetBlock(channels=64)(x)
  encoded = ResNetBlock(channels=64)(x)

  x = ResNetBlock(channels=128, down_sample=True)(encoded)
  x = ResNetBlock(channels=128)(x)

  x = ResNetBlock(channels=256, down_sample=True)(x)
  x = ResNetBlock(channels=256)(x)

  x = ResNetBlock(channels=512, down_sample=True)(x)
  x = ResNetBlock(channels=512)(x)

  x = tf.keras.layers.Flatten()(x)
  output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

  model = tf.keras.Model(inputs=input, outputs=[output, encoded])

  return model


class ResNetBlock(tf.keras.layers.Layer):

  def __init__(self, channels, down_sample=False, **kwargs):
      super(ResNetBlock, self).__init__(**kwargs)

      self.__down_sample = down_sample
      self.__strides = [2, 1] if down_sample else [1, 1]

      self.conv0 = tf.keras.layers.Conv2D(
          filters=channels,
          kernel_size=(3,3),
          strides=self.__strides[0],
          padding="same",
          kernel_initializer='he_normal',
          kernel_regularizer=regularizer,
          bias_regularizer=regularizer
      )
      self.d0 = tf.keras.layers.Dropout(rate=0.3)
      self.bn0 = tf.keras.layers.BatchNormalization()
          
      self.conv1 = tf.keras.layers.Conv2D(
          filters=channels,
          kernel_size=(3,3),
          strides=self.__strides[1],
          padding="same",
          kernel_initializer='he_normal',
          kernel_regularizer=regularizer,
          bias_regularizer=regularizer
      )
      self.d1 = tf.keras.layers.Dropout(rate=0.3)
      self.bn1 = tf.keras.layers.BatchNormalization()

    self.merge = tf.keras.layers.Add()

    if self.__down_sample:

        self.res_conv = tf.keras.layers.Conv2D(
            filters=channels, 
            strides=2,
            kernel_size=(1,1),
            kernel_initializer="he_normal",
            padding="same",
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer
        )
        self.res_d = tf.keras.layers.Dropout(rate=0.3)
        self.res_bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None):
        res = inputs

        x = self.conv0(inputs)
        x = self.d0(x, training)
        x = self.bn0(x, training)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.d1(x, training)
        x = self.bn1(x, training)

        if self.__down_sample:
            res = self.res_conv(res)
            res = self.res_d(res, training)
            res = self.res_bn(res, training)

        x = self.merge([x, res])
        out = tf.nn.relu(x)

        return out
```

The code for the 'student' transformer:

```python
def create_septr_encoded_student(
  input_dim,
  patch_size, 
  embed_dim,
  num_layers,
  num_heads,
  mlp_dims,
  dropout_rate,
  num_classes
):

    input = tf.keras.Input(shape=input_dim)

    mean_cls, mean_dist = SepTr(
      input_dim, patch_size, embed_dim, num_layers, num_heads, mlp_dims, dropout_rate
    )(input)

    ce_output = tf.keras.layers.Dense(num_classes, activation='softmax', name="cls_output")(mean_cls)
    t_output = tf.keras.layers.Dense(num_classes, activation='softmax', name="distillation_output")(mean_dist)

    return tf.keras.Model(inputs=input, outputs=[ce_output, t_output])
```

We can visualize the training procedure of DeSepTr in Figure 4.

<div style="text-align: center;">
  <img src="/blog/academic/deseptr/deseptr.pdf" alt="deseptr">
  <figcaption>Figure 4. Diagram of DeSepTr</figcaption>
</div>
<br>

The code for the training procedure:

```python
class CustomDeiT(tf.keras.Model):

    def __init__(self, student, teacher, encoder, **kwargs):
        super(CustomDeiT, self).__init__(**kwargs)

        self.teacher = teacher
        self.student = student
        self.encoder = encoder

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn
    ):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn

        self.distillation_loss_fn = distillation_loss_fn

    def train_step(self, data):

        # unpack the data
        x, y, sample_weight = data

        # get the predictions from the teacher model 
        teacher_predictions = self.teacher(x, training=False)

        encoded_x = self.encoder(x, training=True)

        with tf.GradientTape() as tape:

            # get the class and distillation tokens from the student model 
            student_output = self.student(encoded_x, training=True)

            # get the predicitions from the student loss function
            student_loss = self.student_loss_fn(y, student_output[0], sample_weight)

            # get the distillation loss from the teacher predictions 
            distillation_loss = self.distillation_loss_fn(teacher_predictions, student_output[1], sample_weight)

            # compute the overall loss 
            loss = (0.5 * student_loss) + (0.5 * distillation_loss)

        # compute the gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # update the weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y, student_output[0])

        results = {m.name: m.result() for m in self.metrics}

        results.update(
            {
                "student_loss": student_loss, 
                "distillation_loss": distillation_loss,
                "loss": loss
            }
        )

        return results

    def test_step(self, data):

        x, y = data

        encoded_x = self.encoder(x, training=False)

        teacher_predictions = self.teacher(x, training=False)

        y_prediction = self.student(encoded_x, training=False)

        student_loss = self.student_loss_fn(y, y_prediction[0])

        distillation_loss = self.distillation_loss_fn(teacher_predictions, y_prediction[1])

        loss = (0.5 * student_loss) + (0.5 * distillation_loss)

        self.compiled_metrics.update_state(y, y_prediction[0])

        results = {m.name: m.result() for m in self.metrics}

        results.update(
            {
                "student_loss": student_loss,
                "distillation_loss": distillation_loss,
                "loss": loss
            }
        )

        return results

    def custom_predict(self, X):

        encoded_X = self.encoder(X, training=False)

        y_prediction = self.student(encoded_X, training=False)

        return y_prediction[0]
```

Example for using the training procedure:

```python

teacher_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10,
    mode='min',
    restore_best_weights=True
)
student_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10,
    mode='min',
    restore_best_weights=True
)

# get the class weights
sample_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=list(range(num_classes)),
    y=train.label.to_numpy().tolist()
).tolist()

weight_dict = dict()
for i, weight in enumerate(sample_weights):
    weight_dict[i] = weight

teacher.compile(
    optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=params["training"]["teacher"]["lr"]),
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=[tf.keras.metrics.CategoricalAccuracy()]
)

teacher.fit(
    X_train, 
    Y_train, 
    batch_size=batch_size
    epochs=epochs,, 
    validation_data=(X_val, Y_val), 
    verbose=2, 
    callbacks=[teacher_callback, TeacherLoggingCallback()],
    class_weight=weight_dict
)

teacher_val_output = teacher.evaluate(X_val, Y_val, verbose=2)
teacher_test_output = teacher.evaluate(X_test, Y_test, verbose=2)

distiller = CustomDeiT(student, teacher, encoder)

distiller.compile(
    optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=learning_rate),
    metrics=[tf.keras.metrics.CategoricalAccuracy()],
    student_loss_fn=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    distillation_loss_fn=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
)

student_history = distiller.fit(
    X_train, 
    Y_train, 
    batch_size=batch_size, 
    epochs=epochs, 
    validation_data=(X_val, Y_val),
    verbose=2, 
    callbacks=[student_callback, StudentLoggingCallback()],
    class_weight=weight_dict
)

deseptr_val_output = distiller.evaluate(X_val, Y_val, verbose=2)
deseptr_test_output = distiller.evaluate(X_test, Y_test, verbose=2)

```

### Experiments

We evaluated our proposed framework on the following datasets:

1. [A database of human gait performance on irregular and uneven surfaces collected by wearable sensors](https://www.nature.com/articles/s41597-020-0563-y)
2. [PAMAP2 Physical Activity Monitoring](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)
3. [Daphnet Freezing of Gait](https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait)

The results from the surface condition recognition where the data originated from the right-shank device are presented in Table 1 and Figure 5.

<div style="text-align: center;">
  <img src="/blog/academic/deseptr/tab_surface.png" alt="tab_surface" style="width: 70%; max-width: 600px;">
  <figcaption>Table 1. Results from the surface condition recognition dataset.</figcaption>
</div>
<br>

<div style="text-align: center;">
  <img src="/blog/academic/deseptr/conf_mat.png" alt="conf_mat" style="width: 70%; max-width: 600px;">
  <figcaption>Figure 5. Confusion matrices for DeSepTr from the surface condition recognition dataset</figcaption>
</div>
<br>

The results from the PAMPA2 dataset where the data originated from the ankle device are presented in Table 2.

<div style="text-align: center;">
  <img src="/blog/academic/deseptr/tab_activity.png" alt="tab_activity" style="width: 70%; max-width: 600px;">
  <figcaption>Table 2. Results from the PAMPA2 dataset.</figcaption>
</div>
<br>

The results from the Daphnet dataset where the data originated from the shank device are presented in Table 3.

<div style="text-align: center;">
  <img src="/blog/academic/deseptr/tab_fog.png" alt="tab_fog" style="width: 70%; max-width: 600px;">
  <figcaption>Table 2. Results from the Daphnet dataset.</figcaption>
</div>
<br>

We can see that adopting the DeSepTr framework offers improvements on the original ViT and SepTr architectures for the different wearable sensor applications.

---

## Conclusion

I hope this blog has provided you with the technical code to implement DeSepTr for your own spectrogram implementations. This was my first real deep learning (DL) publication and I had a lot of fun devising my own architecture for analyzing datasets relevant to my PhD. Ultimately, DeSepTr did not find its way into my PhD, however, I've recently published my Data Efficient Sensor Transformer (DesT) and Federated Data Efficient Sensor Transformer (FeDesT) in the special session on [Federated Learning for Big Data](https://www3.cs.stonybrook.edu/~ieeebigdata2024/SpecialSessions.html#SpecialSession8) at the [2024 IEEE Big Data conference](https://www3.cs.stonybrook.edu/~ieeebigdata2024/index.html). I will be writing a blog for DesT and FeDesT for my presentation to help keep the code free and transparent.

I've recently focused on revamping my professional website for my career. This has given me the push to publish the code for my PhD research. Revisiting this paper I have found a few things that needed corrected and have digged out the supporting code. If there's any errors please feel free to [contact me](/contact) and I'll try and rectify any issues.

If you have benefited from any of the content in this blog please drop me a reference wherever appropriate! I would also appreciate shoutouts on Twitter (X), Reddit ([r/MachineLearning](https://www.reddit.com/r/MachineLearning/)), or LinkedIn.

1. McQuire, J., Watson, P., Wright, N., Hiden, H. and Catt, M., 2023, July. A Data Efficient Vision Transformer for Robust Human Activity Recognition from the Spectrograms of Wearable Sensor Data. In 2023 IEEE Statistical Signal Processing Workshop (SSP) (pp. 364-368). IEEE.

---

## References

1. Ristea, N.C., Ionescu, R.T. and Khan, F.S., 2022. Septr: Separable transformer for audio spectrogram processing. arXiv preprint arXiv:2203.09581.
2. Dosovitskiy, A., 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.
3. Vaswani, A., 2017. Attention is all you need. Advances in Neural Information Processing Systems.
4. Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A. and Jégou, H., 2021, July. Training data-efficient image transformers & distillation through attention. In International conference on machine learning (pp. 10347-10357). PMLR.
5. He, K., Zhang, X., Ren, S. and Sun, J., 2016. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
