import os
import argparse
import random
import torch
import numpy as np
from torchmeta.datasets import MiniImagenet, Omniglot
from torchmeta.transforms import ClassSplitter, Categorical, Rotation
from torchmeta.utils.data import BatchMetaDataLoader
from torchvision.transforms import ToTensor, Compose, Resize
from src.abml import ABMLExplicit, ABMLImplicit


def main(args):
    suffix = '-' + str(args.num_way) + 'way' + str(args.num_supp) + 'shot'
    args.model_dir = os.path.join(args.model_dir, args.dataset.lower(), args.algorithm.lower() + suffix)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"   # for CUDA >= 10.2
    torch.use_deterministic_algorithms(True)

    class_splitter_train = ClassSplitter(shuffle=True, num_train_per_class=args.num_supp,
                                         num_test_per_class=args.num_qry)
    class_splitter_eval = ClassSplitter(shuffle=True, num_train_per_class=args.num_supp,
                                        num_test_per_class=args.num_supp)

    dataset = args.dataset.lower()
    if dataset == 'miniimagenet':
        dataset = MiniImagenet
        transform = ToTensor()
        class_aug = None
    elif dataset == 'omniglot':
        dataset = Omniglot
        transform = Compose([Resize(28), ToTensor()])
        class_aug = [Rotation([90, 180, 270])]
    else:
        raise NotImplementedError

    train_dataset = dataset(args.data_dir,
                            num_classes_per_task=args.num_way,
                            transform=transform,
                            target_transform=Categorical(num_classes=args.num_way),
                            dataset_transform=class_splitter_train,
                            class_augmentations=class_aug,
                            meta_train=True,
                            download=args.download)
    val_dataset = dataset(args.data_dir,
                          num_classes_per_task=args.num_way,
                          transform=transform,
                          target_transform=Categorical(num_classes=args.num_way),
                          dataset_transform=class_splitter_eval,
                          class_augmentations=class_aug,
                          meta_val=True,
                          download=args.download)
    test_dataset = dataset(args.data_dir,
                           num_classes_per_task=args.num_way,
                           transform=transform,
                           target_transform=Categorical(num_classes=args.num_way),
                           dataset_transform=class_splitter_eval,
                           class_augmentations=class_aug,
                           meta_test=True,
                           download=args.download)

    train_dataset.seed(args.seed)
    val_dataset.seed(args.seed)
    test_dataset.seed(args.seed)

    train_dataloader = BatchMetaDataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_worker)
    val_dataloader = BatchMetaDataLoader(val_dataset, batch_size=1, num_workers=1)
    test_dataloader = BatchMetaDataLoader(test_dataset, batch_size=1, num_workers=1)

    for k, v in args.__dict__.items():
        print('%s: %s' % (k, v))

    args.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    args.loss_fn = torch.nn.CrossEntropyLoss()

    if args.algorithm.lower() == 'ibaml':
        alg = ABMLImplicit(args)
    elif args.algorithm.lower() == 'abml':
        alg = ABMLExplicit(args)
    else:
        raise NotImplementedError

    alg.train(train_dataloader=train_dataloader, val_dataloader=val_dataloader)
    alg.load_model(args.algorithm.lower() + '_final.ct')
    alg.test(test_dataloader=test_dataloader)


if __name__ == '__main__':
    # --------------------------------------------------
    # SETUP INPUT PARSER
    # --------------------------------------------------
    parser = argparse.ArgumentParser(description='Setup variables')

    # dir
    parser.add_argument('--data-dir', type=str, default='./datasets/', help='Dataset directory')
    parser.add_argument('--model-dir', type=str, default='./models/', help='Save directory')
    parser.add_argument('--log-dir', type=str, default='./logs/', help='Log directory')

    # dataset
    parser.add_argument('--dataset', type=str, default='MiniImageNet', help='Dataset')
    parser.add_argument('--download', type=bool, default=False, help='Whether to download the dataset')
    parser.add_argument('--num-way', type=int, default=5, help='Number of classes per task')
    parser.add_argument('--num-supp', type=int, default=5, help='Number of data per class (aka. shot) in support set')
    parser.add_argument('--num-qry', type=int, default=15, help='Number of data per class in query set')
    parser.add_argument('--num-val-tasks', type=int, default=1000, help='Number of tasks for meta-validation')
    parser.add_argument('--num-ts-tasks', type=int, default=1000, help='Number of tasks for meta-test')
    parser.add_argument('--num-worker', type=int, default=2, help='Number of workers used in dataloader')
    parser.add_argument('--seed', type=int, default=0, help='seed for Reproducibility')

    # algorithm
    parser.add_argument('--algorithm', type=str, default='iBaML', help='Few-shot learning methods')
    parser.add_argument('--cuda', type=bool, default=True, help='Whether to use cuda')

    # meta training params
    parser.add_argument('--meta-iter', type=int, default=40000, help='Number of epochs for meta training')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size of tasks to update meta-parameters')
    parser.add_argument('--log-iter', type=int, default=200, help='Log iter')
    parser.add_argument('--num-log-tasks', type=int, default=100, help='Number of val tasks to for logging')
    parser.add_argument('--save-iter', type=int, default=2000, help='Save iter')
    parser.add_argument('--meta-lr', type=float, default=1e-3, help='Learning rate for meta-updates')

    # task training params
    parser.add_argument('--task-iter', type=int, default=5, help='The number of inner updates for task adaptation')
    parser.add_argument('--task-lr', type=float, default=1e-2, help='Learning rate for task-updates')
    parser.add_argument('--kld-weight', type=float, default=1e-2, help='Weight for prior term')
    parser.add_argument('--cg-iter', type=float, default=5, help='Number of CG iterations')

    args = parser.parse_args()

    main(args)
