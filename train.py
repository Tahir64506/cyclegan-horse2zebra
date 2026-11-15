# =============================================================================
# CycleGAN: Horse <-> Zebra (TensorFlow / Keras)
# Run with:  python train.py   (set EPOCHS below; resumes automatically if a checkpoint exists)
# =============================================================================

# %% PIECE 0: Setup ------------------------------------------------------------
import os
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

AUTOTUNE = tf.data.AUTOTUNE

# Hyperparameters (the standard CycleGAN settings)
IMG_SIZE = 256
JITTER_SIZE = 286
BUFFER_SIZE = 1000
BATCH_SIZE = 1
EPOCHS = 100            # use 5-10 first to check everything works
LAMBDA_CYCLE = 10.0
LAMBDA_IDENTITY = 5.0   # = 0.5 * LAMBDA_CYCLE
LR = 2e-4
BETA_1 = 0.5

OUTPUT_DIR = os.environ.get("CYCLEGAN_OUTPUT_DIR", "cyclegan_output")   # where samples and weights are saved
os.makedirs(f"{OUTPUT_DIR}/samples", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/checkpoints", exist_ok=True)

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))


# %% PIECE 1: Data pipeline ----------------------------------------------------
# trainA = horses, trainB = zebras (same for testA / testB)
dataset, info = tfds.load("cycle_gan/horse2zebra", with_info=True, as_supervised=True)
train_horses, train_zebras = dataset["trainA"], dataset["trainB"]
test_horses, test_zebras = dataset["testA"], dataset["testB"]


def normalize(img):
    """uint8 [0,255] -> float32 [-1,1] (matches the generator's tanh output)."""
    img = tf.cast(img, tf.float32)
    return img / 127.5 - 1.0


def random_jitter(img):
    """Resize up to 286x286, random-crop back to 256x256, random horizontal flip."""
    img = tf.image.resize(img, [JITTER_SIZE, JITTER_SIZE],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img = tf.image.random_crop(img, [IMG_SIZE, IMG_SIZE, 3])
    img = tf.image.random_flip_left_right(img)
    return img


def preprocess_train(img, _label):
    return random_jitter(normalize(img))


def preprocess_test(img, _label):
    return normalize(img)


def make_train_ds(ds):
    return (ds.map(preprocess_train, num_parallel_calls=AUTOTUNE)
              .cache()
              .shuffle(BUFFER_SIZE)
              .batch(BATCH_SIZE)
              .prefetch(AUTOTUNE))


def make_test_ds(ds):
    return (ds.map(preprocess_test, num_parallel_calls=AUTOTUNE)
              .batch(BATCH_SIZE)
              .prefetch(AUTOTUNE))


train_horses_ds = make_train_ds(train_horses)
train_zebras_ds = make_train_ds(train_zebras)
test_horses_ds = make_test_ds(test_horses)
test_zebras_ds = make_test_ds(test_zebras)

# Pair the two unpaired domains; zip stops at the shorter one (1067 horses)
train_ds = tf.data.Dataset.zip((train_horses_ds, train_zebras_ds))

# Quick look at one sample from each domain
sample_horse = next(iter(test_horses_ds))
sample_zebra = next(iter(test_zebras_ds))
plt.figure(figsize=(6, 3))
for i, (img, title) in enumerate([(sample_horse, "Horse"), (sample_zebra, "Zebra")]):
    plt.subplot(1, 2, i + 1)
    plt.title(title)
    plt.imshow(img[0] * 0.5 + 0.5)
    plt.axis("off")
plt.show()


# %% PIECE 2: Custom layers ----------------------------------------------------
class InstanceNorm(layers.Layer):
    """Instance normalization (normalizes each image/channel independently).
    Written by hand because tensorflow_addons is deprecated."""

    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale", shape=input_shape[-1:],
            initializer=keras.initializers.RandomNormal(1.0, 0.02), trainable=True)
        self.offset = self.add_weight(
            name="offset", shape=input_shape[-1:],
            initializer="zeros", trainable=True)

    def call(self, x):
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x = (x - mean) * tf.math.rsqrt(var + self.epsilon)
        return self.scale * x + self.offset


class ReflectionPad2D(layers.Layer):
    """Pads borders by mirroring pixels, which avoids edge artifacts."""

    def __init__(self, pad=1, **kwargs):
        super().__init__(**kwargs)
        self.pad = pad

    def call(self, x):
        p = self.pad
        return tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], mode="REFLECT")


# %% PIECE 3: Generator (ResNet-based, 9 residual blocks) ----------------------
INIT = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


def residual_block(x, filters):
    skip = x
    x = ReflectionPad2D(1)(x)
    x = layers.Conv2D(filters, 3, kernel_initializer=INIT, use_bias=False)(x)
    x = InstanceNorm()(x)
    x = layers.ReLU()(x)
    x = ReflectionPad2D(1)(x)
    x = layers.Conv2D(filters, 3, kernel_initializer=INIT, use_bias=False)(x)
    x = InstanceNorm()(x)
    return layers.Add()([skip, x])


def build_generator(name, n_res_blocks=9):
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # c7s1-64
    x = ReflectionPad2D(3)(inp)
    x = layers.Conv2D(64, 7, kernel_initializer=INIT, use_bias=False)(x)
    x = InstanceNorm()(x)
    x = layers.ReLU()(x)

    # Downsample: 128, 256
    for f in (128, 256):
        x = layers.Conv2D(f, 3, strides=2, padding="same",
                          kernel_initializer=INIT, use_bias=False)(x)
        x = InstanceNorm()(x)
        x = layers.ReLU()(x)

    # Residual blocks
    for _ in range(n_res_blocks):
        x = residual_block(x, 256)

    # Upsample: 128, 64
    for f in (128, 64):
        x = layers.Conv2DTranspose(f, 3, strides=2, padding="same",
                                   kernel_initializer=INIT, use_bias=False)(x)
        x = InstanceNorm()(x)
        x = layers.ReLU()(x)

    # Output layer
    x = ReflectionPad2D(3)(x)
    x = layers.Conv2D(3, 7, activation="tanh", kernel_initializer=INIT)(x)
    return keras.Model(inp, x, name=name)


# %% PIECE 4: Discriminator (70x70 PatchGAN) -----------------------------------
def build_discriminator(name):
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # C64 (no normalization on the first layer)
    x = layers.Conv2D(64, 4, strides=2, padding="same", kernel_initializer=INIT)(inp)
    x = layers.LeakyReLU(0.2)(x)

    # C128, C256 (stride 2), C512 (stride 1)
    for f, s in ((128, 2), (256, 2), (512, 1)):
        x = layers.Conv2D(f, 4, strides=s, padding="same",
                          kernel_initializer=INIT, use_bias=False)(x)
        x = InstanceNorm()(x)
        x = layers.LeakyReLU(0.2)(x)

    # 1-channel map of real/fake logits, one per patch (no sigmoid: we use LSGAN)
    x = layers.Conv2D(1, 4, padding="same", kernel_initializer=INIT)(x)
    return keras.Model(inp, x, name=name)


gen_G = build_generator("G_horse_to_zebra")   # X (horse) -> Y (zebra)
gen_F = build_generator("F_zebra_to_horse")   # Y (zebra) -> X (horse)
disc_X = build_discriminator("D_horse")       # judges real vs fake horses
disc_Y = build_discriminator("D_zebra")       # judges real vs fake zebras

gen_G.summary(line_length=100)
disc_X.summary(line_length=100)


# %% PIECE 5: Losses -----------------------------------------------------------
mse = keras.losses.MeanSquaredError()
mae = keras.losses.MeanAbsoluteError()


def generator_adv_loss(fake_logits):
    """Generator wants the discriminator to output 1 for its fakes (LSGAN)."""
    return mse(tf.ones_like(fake_logits), fake_logits)


def discriminator_loss(real_logits, fake_logits):
    real = mse(tf.ones_like(real_logits), real_logits)
    fake = mse(tf.zeros_like(fake_logits), fake_logits)
    return 0.5 * (real + fake)


def cycle_loss(real, cycled):
    return LAMBDA_CYCLE * mae(real, cycled)


def identity_loss(real, same):
    return LAMBDA_IDENTITY * mae(real, same)


# %% PIECE 6: CycleGAN model with custom training step -------------------------
class CycleGAN(keras.Model):
    def __init__(self, gen_G, gen_F, disc_X, disc_Y):
        super().__init__()
        self.gen_G, self.gen_F = gen_G, gen_F
        self.disc_X, self.disc_Y = disc_X, disc_Y

    def compile(self, gen_G_opt, gen_F_opt, disc_X_opt, disc_Y_opt):
        super().compile()
        self.gen_G_opt, self.gen_F_opt = gen_G_opt, gen_F_opt
        self.disc_X_opt, self.disc_Y_opt = disc_X_opt, disc_Y_opt

    def call(self, x):
        return self.gen_G(x)

    def train_step(self, batch):
        real_x, real_y = batch  # horse, zebra

        with tf.GradientTape(persistent=True) as tape:
            # Forward translations and cycles
            fake_y = self.gen_G(real_x, training=True)      # horse -> zebra
            cycled_x = self.gen_F(fake_y, training=True)    # ... -> horse
            fake_x = self.gen_F(real_y, training=True)      # zebra -> horse
            cycled_y = self.gen_G(fake_x, training=True)    # ... -> zebra

            # Identity mapping (G should leave zebras alone, F should leave horses alone)
            same_x = self.gen_F(real_x, training=True)
            same_y = self.gen_G(real_y, training=True)

            # Discriminator outputs
            d_real_x = self.disc_X(real_x, training=True)
            d_fake_x = self.disc_X(fake_x, training=True)
            d_real_y = self.disc_Y(real_y, training=True)
            d_fake_y = self.disc_Y(fake_y, training=True)

            # Generator losses = adversarial + cycle + identity
            total_cycle = cycle_loss(real_x, cycled_x) + cycle_loss(real_y, cycled_y)
            g_loss = (generator_adv_loss(d_fake_y) + total_cycle
                      + identity_loss(real_y, same_y))
            f_loss = (generator_adv_loss(d_fake_x) + total_cycle
                      + identity_loss(real_x, same_x))

            # Discriminator losses
            dx_loss = discriminator_loss(d_real_x, d_fake_x)
            dy_loss = discriminator_loss(d_real_y, d_fake_y)

        # Gradients and updates
        for loss, model, opt in (
            (g_loss, self.gen_G, self.gen_G_opt),
            (f_loss, self.gen_F, self.gen_F_opt),
            (dx_loss, self.disc_X, self.disc_X_opt),
            (dy_loss, self.disc_Y, self.disc_Y_opt),
        ):
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
        del tape

        return {"G_loss": g_loss, "F_loss": f_loss,
                "D_X_loss": dx_loss, "D_Y_loss": dy_loss}


def make_opt():
    return keras.optimizers.Adam(learning_rate=LR, beta_1=BETA_1)


model = CycleGAN(gen_G, gen_F, disc_X, disc_Y)
model.compile(make_opt(), make_opt(), make_opt(), make_opt())


# %% PIECE 7: Callbacks (sample images + checkpoints) --------------------------
class SaveSamples(keras.callbacks.Callback):
    """After every epoch: save a horse->zebra->horse and zebra->horse->zebra preview
    and write the weights to disk."""

    def __init__(self, horses, zebras, n=2):
        super().__init__()
        self.horses = list(horses.take(n))
        self.zebras = list(zebras.take(n))

    def on_epoch_end(self, epoch, logs=None):
        rows = []
        for h in self.horses:
            fake_z = self.model.gen_G(h, training=False)
            rows.append(("Horse", h, "Fake zebra", fake_z,
                         "Cycled horse", self.model.gen_F(fake_z, training=False)))
        for z in self.zebras:
            fake_h = self.model.gen_F(z, training=False)
            rows.append(("Zebra", z, "Fake horse", fake_h,
                         "Cycled zebra", self.model.gen_G(fake_h, training=False)))

        fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
        for r, row in enumerate(rows):
            for c in range(3):
                title, img = row[2 * c], row[2 * c + 1]
                axes[r, c].imshow(img[0] * 0.5 + 0.5)
                axes[r, c].set_title(title, fontsize=9)
                axes[r, c].axis("off")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/samples/epoch_{epoch + 1:03d}.png", dpi=100)
        plt.close(fig)

        self.model.gen_G.save_weights(f"{OUTPUT_DIR}/checkpoints/G.weights.h5")
        self.model.gen_F.save_weights(f"{OUTPUT_DIR}/checkpoints/F.weights.h5")
        self.model.disc_X.save_weights(f"{OUTPUT_DIR}/checkpoints/DX.weights.h5")
        self.model.disc_Y.save_weights(f"{OUTPUT_DIR}/checkpoints/DY.weights.h5")

        # Written last, so it only says "epoch N done" if all weights saved fine
        with open(f"{OUTPUT_DIR}/checkpoints/last_epoch.txt", "w") as f:
            f.write(str(epoch + 1))


# %% PIECE 8: Train ------------------------------------------------------------
# --- Resume from the last completed epoch, if a checkpoint exists ---
ckpt = f"{OUTPUT_DIR}/checkpoints"
initial_epoch = 0
if os.path.exists(f"{ckpt}/last_epoch.txt"):
    with open(f"{ckpt}/last_epoch.txt") as f:
        initial_epoch = int(f.read().strip())
    gen_G.load_weights(f"{ckpt}/G.weights.h5")
    gen_F.load_weights(f"{ckpt}/F.weights.h5")
    disc_X.load_weights(f"{ckpt}/DX.weights.h5")
    disc_Y.load_weights(f"{ckpt}/DY.weights.h5")
    print(f"Resuming from epoch {initial_epoch}")
else:
    print("No checkpoint found, starting from scratch")

# EPOCHS is the TOTAL you want (set it in Piece 0), not "how many more".
start = time.time()
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    initial_epoch=initial_epoch,
    callbacks=[SaveSamples(test_horses_ds, test_zebras_ds)],
)
print(f"Training finished in {(time.time() - start) / 60:.1f} minutes")

# Loss curves
plt.figure(figsize=(8, 4))
for k, v in history.history.items():
    plt.plot(v, label=k)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("CycleGAN training losses")
plt.savefig(f"{OUTPUT_DIR}/loss_curves.png", dpi=100)
plt.show()


# %% PIECE 9: Inference / evaluation on the test set ---------------------------
def show_translations(ds, generator, title_in, title_out, n=5):
    plt.figure(figsize=(3 * n, 6))
    for i, img in enumerate(ds.take(n)):
        out = generator(img, training=False)
        plt.subplot(2, n, i + 1)
        plt.imshow(img[0] * 0.5 + 0.5)
        plt.title(title_in)
        plt.axis("off")
        plt.subplot(2, n, n + i + 1)
        plt.imshow(out[0] * 0.5 + 0.5)
        plt.title(title_out)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


show_translations(test_horses_ds, gen_G, "Real horse", "Fake zebra")
show_translations(test_zebras_ds, gen_F, "Real zebra", "Fake horse")

# To reuse a trained model later without retraining:
#   gen_G = build_generator("G"); gen_G(tf.zeros([1, 256, 256, 3]))
#   gen_G.load_weights(f"{OUTPUT_DIR}/checkpoints/G.weights.h5")
