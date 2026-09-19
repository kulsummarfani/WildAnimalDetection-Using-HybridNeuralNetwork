import cv2
import numpy as np
import tensorflow as tf
from classify import load_classifier, IMG_SIZE, LABELS


def make_gradcam_overlay(crop_bgr):
    """Returns (overlay_image, predicted_species, confidence) for a cropped detection."""
    model = load_classifier()
    vgg = model.get_layer(index=0)  # VGG19 backbone; include_top=False so its own
                                     # output IS the last conv block's feature map

    img = cv2.resize(crop_bgr, (IMG_SIZE, IMG_SIZE)).astype('float32') / 255.0
    img_tensor = tf.convert_to_tensor(np.expand_dims(img, axis=0))

    with tf.GradientTape() as tape:
        conv_out = vgg(img_tensor, training=False)
        tape.watch(conv_out)  # conv_out is an intermediate tensor, not a Variable,
                               # so it must be watched explicitly to get gradients w.r.t. it

        x = conv_out
        for layer in model.layers[1:]:  # replay the rest of the model on top of conv_out,
            x = layer(x, training=False)  # inside the SAME tape context this time
        preds = x

        class_idx = int(tf.argmax(preds[0]))
        class_score = preds[:, class_idx]

    grads = tape.gradient(class_score, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, (crop_bgr.shape[1], crop_bgr.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(crop_bgr, 0.6, heatmap_color, 0.4, 0)

    return overlay, LABELS[class_idx], float(preds[0][class_idx])