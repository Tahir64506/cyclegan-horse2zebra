"""CycleGAN Horse <-> Zebra demo (Gradio). Accepts any image size."""
import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import gradio as gr

WEIGHTS_DIR = os.environ.get("CYCLEGAN_WEIGHTS_DIR", "weights")

# --- MODEL START (same networks as training; the generator accepts any size) ---
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
    inp = layers.Input(shape=(None, None, 3))  # any size (multiple of 4)

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
# --- MODEL END ---


def load_generator(name, filename):
    path = os.path.join(WEIGHTS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Put G.weights.h5 and F.weights.h5 in the '{WEIGHTS_DIR}' folder.")
    g = build_generator(name)
    g.load_weights(path)
    return g


gen_G = load_generator("G_horse_to_zebra", "G.weights.h5")   # horse -> zebra
gen_F = load_generator("F_zebra_to_horse", "F.weights.h5")   # zebra -> horse

DIRECTIONS = ["Horse → Zebra", "Zebra → Horse"]
SIZE_MODES = ["Keep my image size", "256 x 256 (trained size)"]


def pick_size(w, h, mode, max_side):
    """Size the model works at. Always a multiple of 4 (the network halves the size twice)."""
    if mode == SIZE_MODES[1]:
        return 256, 256
    scale = min(1.0, max_side / max(w, h))          # shrink big images, never enlarge
    snap = lambda v: max(64, int(round(v * scale / 4)) * 4)
    return snap(w), snap(h)


def translate(img, direction, mode, max_side, back_to_original):
    if img is None:
        return None, "Upload an image to start."
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    gen = gen_G if direction == DIRECTIONS[0] else gen_F
    h, w = img.shape[:2]
    nw, nh = pick_size(w, h, mode, int(max_side))
    x = tf.cast(img[..., :3], tf.float32)
    x = tf.image.resize(x, [nh, nw], antialias=True) / 127.5 - 1.0
    y = gen(x[None, ...], training=False)[0].numpy()
    out = ((y + 1.0) * 127.5).clip(0, 255).astype("uint8")
    note = f"Input {w} x {h} px. The model worked at {nw} x {nh} px."
    if back_to_original and (nw, nh) != (w, h):
        out = np.asarray(Image.fromarray(out).resize((w, h), Image.LANCZOS))
        note += f" Result stretched back to {w} x {h} px (may look softer)."
    else:
        note += f" Result is {out.shape[1]} x {out.shape[0]} px."
    if mode == SIZE_MODES[1] and (w, h) != (256, 256):
        note += " Your image was squashed to a square, so proportions may look different."
    return out, note


def asset(name):
    p = os.path.join("assets", name)
    return p if os.path.exists(p) else None


EXAMPLES = sorted(glob.glob("examples/*.jpg"))

with gr.Blocks(title="CycleGAN Horse ↔ Zebra") as demo:
    gr.Markdown("# 🐎 ↔ 🦓 CycleGAN: Horse ↔ Zebra\n"
                "Upload a photo and watch it change species. Unpaired image-to-image translation, "
                "trained for 100 epochs on the Horse2Zebra dataset.")

    with gr.Tab("Try it"):
        direction = gr.Radio(DIRECTIONS, value=DIRECTIONS[0], label="Direction")
        with gr.Row():
            mode = gr.Radio(SIZE_MODES, value=SIZE_MODES[0], label="Output size")
            max_side = gr.Slider(256, 1024, value=512, step=64, label="Largest side (px) when keeping my size")
        back = gr.Checkbox(value=False, label="Stretch the result back to my original size")
        with gr.Row():
            inp = gr.Image(label="Your image", type="numpy")
            out = gr.Image(label="Translated image")
        info = gr.Textbox(label="Sizes", interactive=False)
        gr.Button("Translate", variant="primary").click(
            translate, [inp, direction, mode, max_side, back], [out, info])
        if EXAMPLES:
            gr.Examples(examples=[[p] for p in EXAMPLES], inputs=[inp], label="Or try an example photo")
        with gr.Accordion("Which image sizes work?", open=False):
            gr.Markdown("""
- The model was **trained on 256 x 256** photos, so it is most reliable near that size.
- **Keep my image size** (default): your image is used as it is, up to the limit you choose. Larger photos are shrunk to that limit to keep things fast. Each side of the result is a multiple of 4.
- **256 x 256**: the safest quality, but the image is squashed to a square.
- Photos much larger than 512 px can get finer, patchier stripes, because the model learned stripe size at 256 px.
- JPG, PNG and WEBP all work. Clear photos of one horse or zebra give the best results.
""")

    with gr.Tab("Results"):
        gr.Markdown("Test photos the model never saw: the original, the result at epoch 66, and the result at epoch 100. "
                    "A later epoch is not always better in a GAN.")
        for title, name in (("Horse to zebra", "compare_horse_to_zebra.jpg"), ("Zebra to horse", "compare_zebra_to_horse.jpg")):
            if asset(name):
                gr.Image(value=asset(name), label=title, interactive=False)
        if asset("loss_curves.png"):
            gr.Image(value=asset("loss_curves.png"), label="Training losses, epochs 1 to 100", interactive=False)

    with gr.Tab("About"):
        gr.Markdown("""
**Model:** CycleGAN with two ResNet generators (9 residual blocks) and two 70x70 PatchGAN discriminators.

**Losses:** least-squares adversarial + cycle-consistency (weight 10) + identity (weight 5).

**Training:** Horse2Zebra (1,067 horse and 1,334 zebra training photos, 256x256), Adam (lr 2e-4, beta1 0.5), batch size 1, 100 epochs on a Kaggle T4 GPU.

**Framework:** TensorFlow / Keras.

**Paper:** Zhu, Park, Isola and Efros, *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*, ICCV 2017.
""")

if __name__ == "__main__":
    demo.launch()
