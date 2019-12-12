import tensorflow as tf


def new_treechop_state(gymState):
    """Converts a gym state for MineRLTreechop-v0 into a tensor."""
    t = tf.convert_to_tensor(gymState["pov"])
    t = tf.reshape(t, shape=[-1])
    return tf.dtypes.cast(t, dtype=tf.float32)
