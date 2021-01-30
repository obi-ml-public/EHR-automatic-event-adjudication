import tensorflow.keras as keras


class MetricsModelCheckpoint(keras.callbacks.Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    Arguments:
        filepath: string, path to save the model file.
        metrics: dict with metrics for single output or list of dicts with metrics for multiple outputs.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`, the latest best model according
          to the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}. If `save_best_only=True`, the decision to
          overwrite the current save file is made based on either the maximization
          or the minimization of the monitored quantity. For `val_acc`, this
          should be `max`, for `val_loss` this should be `min`, etc. In `auto`
          mode, the direction is automatically inferred from the name of the
          monitored quantity.
        save_weights_only: if True, then only the model's weights will be saved
          (`model.save_weights(filepath)`), else the full model is saved
          (`model.save(filepath)`).
        .
    """

    # TODO add loss to metrics

    def __init__(self,
                 filepath,
                 x,
                 y,
                 x_val,
                 y_val,
                 metrics,
                 monitor,
                 batch_size,
                 base_model=None,
                 verbose=0,
                 save_best_only=True,
                 save_weights_only=True,
                 mode='max',
                 **kwargs,
                 ):

        self.filepath = filepath
        self.x = x
        self.y = y
        self.x_val = x_val
        self.y_val = y_val
        if base_model is None:
            self.base_model = self.model
        else:
            self.base_model = base_model
        self.batch_size = batch_size
        self.metrics = metrics
        self.monitor = monitor
        self.iter = 0
        self.best_score = None
        self.best_iter = None
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.verbose = verbose

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.iter += 1

        y_pred = self.model.predict(self.x, batch_size=self.batch_size, verbose=self.verbose)
        y_pred_val = self.model.predict(self.x_val, batch_size=self.batch_size, verbose=self.verbose)

        outputs = [node.op.name.split('/')[0].split('_')[0] for node in self.base_model.outputs]
        # calculate metrics for single output
        if len(outputs) == 1:
            for metric_name, metric in self.metrics.items():
                logs[F'{metric_name}'] = metric(self.y, y_pred)
                logs[F'val_{metric_name}'] = metric(self.y_val, y_pred_val)
        else:
            # iterate over outputs
            for i in range(len(outputs)):
                _y_pred = y_pred[i]
                _y_pred_val = y_pred_val[i]
                for metric_name, metric in self.metrics[i].items():
                    logs[F'{metric_name}_{outputs[i]}'] = metric(self.y[i], _y_pred)
                    logs[F'val_{metric_name}_{outputs[i]}'] = metric(self.y_val[i], _y_pred_val)

        if self.verbose:
            # print(logs)
            print('|'.join([f'{k}:{v:.2f}' for k, v in logs.items()]))

        self._update_best(logs[self.monitor])

        if not self.save_best_only or self.best_iter == self.iter:
            self._save_model(epoch, logs)

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

    def _save_model(self, epoch, logs={}):

        # include epoch etc in filepath
        filepath = self.filepath.format(epoch=epoch, **logs)

        self.base_model.save_weights(filepath)
        if not self.save_weights_only:
            # TODO this requires .h5 in filepath
            json_string = self.base_model.to_json()
            open(filepath.replace('.h5', '.json'), 'w').write(json_string)
            yaml_string = self.base_model.to_yaml()
            open(filepath.replace('.h5', '.yaml'), 'w').write(yaml_string)

    def _update_best(self, score):
        if (self.best_score is None or
                (self.mode == 'max' and score > self.best_score) or
                (self.mode == 'min' and score < self.best_score)):
            if self.verbose and self.best_score is not None:
                print(F'{self.monitor} improved from {self.best_score:.2f} to {score:.2f}.')
            self.best_score = score
            self.best_iter = self.iter
