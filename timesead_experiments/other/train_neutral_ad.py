import torch

from timesead_experiments.utils import data_ingredient, load_dataset, training_ingredient, train_model, make_experiment, \
    make_experiment_tempfile, serialization_guard, get_dataloader
from timesead_experiments.utils.training_ingredient import instantiate_loss
from timesead.models.other import NeutralAD, NeutralADAnomalyDetector, NeutralADLoss
from timesead.utils.utils import Bunch


experiment = make_experiment(ingredients=[data_ingredient, training_ingredient])


def get_training_pipeline():
    return {
        'window': {'class': 'WindowTransform', 'args': {'window_size': 50}},
        'normal_filter': {'class': 'WindowLabelFilterTransform', 'args': {'keep_normal': True}}
    }


def get_test_pipeline():
    return {
        'window': {'class': 'WindowTransform', 'args': {'window_size': 50}}
    }


def get_batch_dim():
    return 0


@data_ingredient.config
def data_config():
    pipeline = [get_training_pipeline(), get_test_pipeline()]

    ds_args = dict(
        training=True
    )

    split = (0.75, 0.25)


@training_ingredient.config
def training_config():
    loss = NeutralADLoss
    batch_dim = get_batch_dim()
    trainer_hooks = []
    scheduler = {
        'class': torch.optim.lr_scheduler.MultiStepLR,
        'args': dict(milestones=[20], gamma=0.1)
    }


@experiment.config
def config():
    # Model-specific parameters
    model_params = dict(
        num_trans=4,
        trans_type='residual',
        enc_hdim=32,
        enc_nlayers=4,
        trans_nlayers=4,
        latent_dim=32,
        batch_norm=False,
        enc_bias=False
    )

    train_detector = True
    save_detector = True


@experiment.command(unobserved=True)
@serialization_guard
def get_datasets():
    train_ds, val_ds = load_dataset()

    return get_dataloader(train_ds), get_dataloader(val_ds)


@experiment.command(unobserved=True)
@serialization_guard('model', 'val_loader')
def get_anomaly_detector(model, val_loader, training, _run, save_detector=True):
    training = Bunch(training)
    loss = instantiate_loss(training.loss)
    detector = NeutralADAnomalyDetector(model, loss).to(training.device)

    if save_detector:
        with make_experiment_tempfile('final_model.pth', _run, mode='wb') as f:
            torch.save(dict(detector=detector), f)

    return detector


@experiment.automain
@serialization_guard
def main(model_params, dataset, training, _run, train_detector=True):
    train_ds, val_ds = load_dataset()
    model = NeutralAD(train_ds.num_features, train_ds.seq_len, **model_params)

    trainer = train_model(_run, model, train_ds, val_ds)

    if train_detector:
        detector = get_anomaly_detector(model, trainer.val_iter)
    else:
        detector = None

    return dict(detector=detector, model=model)
