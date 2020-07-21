from meantime.models import model_factory
from meantime.dataloaders import dataloader_factory
from meantime.trainers import trainer_factory
from meantime.utils import *
from meantime.config import *


def main(args):
    if args.mode == 'train':
        train(args)
    elif args.mode == 'validate':
        validate(args, mode='val')
    elif args.mode == 'test':
        validate(args, mode='test')
    else:
        raise ValueError


def train(args):
    local_export_root, remote_export_root, communicator = setup_train(args, MACHINE_IS_HOST)
    assert (communicator is None and MACHINE_IS_HOST) or (communicator is not None and not MACHINE_IS_HOST)
    if communicator:
        communicator.close()  # close station because it might lose connection during long training
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, local_export_root)
    status_file = os.path.join(local_export_root, 'status.txt')
    error_log_file = os.path.join(local_export_root, 'error_log.txt')
    open(status_file, 'w').write(STATUS_RUNNING)
    try:
        trainer.train()
        open(status_file, 'w').write(STATUS_FINISHED)
        if not MACHINE_IS_HOST and args.experiment_group != 'test':
            communicator = Communicator(HOST, PORT, USERNAME, PASSWORD)
            communicator.upload_dir(local_export_root, remote_export_root)
            communicator.close()
    except Exception as err:
        # recover
        if args.experiment_group == 'test':
            raise
        if not os.path.exists(os.path.join(local_export_root, 'tables', 'val_log.csv')):
            print('Removing empty local export root')
            shutil.rmtree(local_export_root)
            raise
        open(status_file, 'w').write(STATUS_RECOVERY)
        open(error_log_file, 'w').write(str(err))
        if not MACHINE_IS_HOST and args.experiment_group != 'test':
            print('Uploading recovery file')
            communicator = Communicator(HOST, PORT, USERNAME, PASSWORD)
            communicator.upload_dir(local_export_root, remote_export_root)
            communicator.close()
        raise


def validate(args, mode='val'):
    local_export_root, remote_export_root, communicator = setup_train(args, MACHINE_IS_HOST)
    if communicator:
        communicator.close()
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)
    if args.pretrained_weights is not None:
        model.load(args.pretrained_weights)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, local_export_root)
    trainer.just_validate(mode)
