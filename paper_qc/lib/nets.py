"""
Heteroscedastic deep ensembles (Lakshminarayanan et al. 2017).

One member is a network of four ReLU layers (256, 128, 64, 32; batch normalisation and dropout
after the first two) with a mean head and a softplus variance head, trained on the Gaussian
negative log-likelihood with Adam and early stopping. Members differ only in their seed; the
ensemble is the equal-weight Gaussian mixture of its members.
"""

import numpy as np

MIN_SIGMA = 1e-3


def configure_gpu():
    """Let TensorFlow grow GPU memory instead of reserving it all at import time."""
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # already initialised; harmless

def build_network(n_in, hidden=(256, 128, 64, 32), dropout=0.1, lr=1e-3):
    import tensorflow as tf
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import BatchNormalization, Dense, Dropout

    inp = Input(shape=(n_in,))
    x = inp
    for i, units in enumerate(hidden):
        x = Dense(units, activation="relu")(x)
        if i < 2:
            x = BatchNormalization()(x)
            x = Dropout(dropout)(x)
    mean = Dense(1, name="mean")(x)
    # Softplus keeps the variance positive; the floor stops the NLL running away on
    # levels the model fits almost exactly (the deep, well-ventilated ones).
    log_var = Dense(1, name="log_var")(x)
    out = tf.keras.layers.Concatenate()([mean, log_var])

    def nll(y_true, y_pred):
        # Reshape is essential: y_true arrives as (batch,) while the heads are (batch, 1),
        # and the subtraction would broadcast to (batch, batch) - a silently wrong loss
        # that still trains, but only to a near-constant prediction.
        y_true = tf.reshape(tf.cast(y_true, y_pred.dtype), (-1, 1))
        mu, log_v = y_pred[:, :1], y_pred[:, 1:]
        var = tf.nn.softplus(log_v) + MIN_SIGMA ** 2
        return tf.reduce_mean(0.5 * (tf.math.log(var) + tf.square(y_true - mu) / var))

    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr, clipnorm=1.0), loss=nll)
    return model

def fit_network(Xtr, ytr, Xva, yva, seed, epochs=60, batch_size=1024, verbose=0, **kwargs):
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    tf.keras.utils.set_random_seed(seed)
    model = build_network(Xtr.shape[1], **kwargs)
    history = model.fit(
        Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=batch_size,
        verbose=verbose,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True,
                          start_from_epoch=5),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
        ],
    )
    return model, len(history.history["loss"])

def predict_network(model, X, batch_size=8192):
    import tensorflow as tf

    raw = model.predict(X, batch_size=batch_size, verbose=0)
    mu = raw[:, 0]
    var = tf.nn.softplus(raw[:, 1]).numpy() + MIN_SIGMA ** 2
    return mu.astype("float32"), var.astype("float32")

def fit_ensemble(Xtr, ytr, Xva, yva, n_members=5, base_seed=0, **kwargs):
    models, epochs = [], []
    for m in range(n_members):
        model, n = fit_network(Xtr, ytr, Xva, yva, seed=base_seed + m, **kwargs)
        models.append(model)
        epochs.append(n)
    return models, epochs

def predict_ensemble(models, X):
    """(mu, sigma, member_mus) of the Gaussian mixture over ensemble members."""
    mus, variances = [], []
    for model in models:
        mu, var = predict_network(model, X)
        mus.append(mu)
        variances.append(var)
    mus = np.stack(mus)
    variances = np.stack(variances)
    mu = mus.mean(axis=0)
    var = (variances + mus ** 2).mean(axis=0) - mu ** 2
    return mu, np.sqrt(np.maximum(var, MIN_SIGMA ** 2)), mus
