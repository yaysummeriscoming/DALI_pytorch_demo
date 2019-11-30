import os, gc, time
import numpy as np
import torch
import importlib

import torchvision.transforms as transforms
import torchvision.datasets as datasets

try:
    from nvidia import dali
    from dali import HybridTrainPipe, HybridValPipe, DaliIteratorCPU, DaliIteratorGPU
except:
    print('Could not import DALI')

def clear_memory(verbose=False):
    stt = time.time()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # https://forums.fast.ai/t/clearing-gpu-memory-pytorch/14637
    gc.collect()

    if verbose:
        print('Cleared memory.  Time taken was %f secs' % (time.time() - stt))


class Dataset():
    """
    Pytorch Dataloader, with torchvision or Nvidia DALI CPU/GPU pipelines.
    This dataloader implements ImageNet style training preprocessing, namely:
    -random resized crop
    -random horizontal flip

    And ImageNet style validation preprocessing, namely:
    -resize to specified size
    -center crop to desired size

    data_dir (str): Directory to dataset.  Format should be the same as torchvision dataloader,
    batch_size (int): how many samples per batch to load
    size (int): Output size (typically 224 for ImageNet)
    val_size (int): Validation pipeline resize size (typically 256 for ImageNet)
    workers (int): how many workers to use for data loading
    world_size (int, optional, default = 1) - Partition the data into this many parts (used for multiGPU training)
    cuda (bool): Output tensors on CUDA, CPU otherwise
    use_dali (bool): Use Nvidia DALI backend, torchvision otherwise
    dali_cpu (bool): Use Nvidia DALI cpu backend, GPU backend otherwise
    fp16 (bool, optional, default = False) - Output the data in fp16 instead of fp32
    mean (tuple): Image mean value for each channel
    std (tuple): Image standard deviation value for each channel
    pin_memory (bool): Transfer CPU tensor to pinned memory before transfer to GPU (torchvision only)
    pin_memory_dali (bool): Transfer CPU tensor to pinned memory before transfer to GPU (dali only)
    """

    def __init__(self,
                 data_dir,
                 batch_size,
                 size=224,
                 val_batch_size=None,
                 val_size=256,
                 min_crop_size=0.08,
                 workers=4,
                 world_size=1,
                 cuda=True,
                 use_dali=False,
                 dali_cpu=True,
                 fp16=False,
                 mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
                 std=(0.229 * 255, 0.224 * 255, 0.225 * 255),
                 pin_memory=True,
                 pin_memory_dali=False,
                 ):

            self.batch_size = batch_size
            self.size = size
            self.val_batch_size = val_batch_size
            self.min_crop_size = min_crop_size
            self.workers = workers
            self.world_size = world_size
            self.cuda = cuda
            self.use_dali = use_dali
            self.dali_cpu = dali_cpu
            self.fp16 = fp16
            self.mean = mean
            self.std = std
            self.pin_memory = pin_memory
            self.pin_memory_dali = pin_memory_dali

            self.val_size = val_size
            if self.val_size is None:
                self.val_size = self.size

            if self.val_batch_size is None:
                self.val_batch_size = self.batch_size

            # Data loading code
            self.traindir = os.path.join(data_dir, 'train')
            self.valdir = os.path.join(data_dir, 'val')

            # DALI Dataloader
            if self.use_dali:
                print('Using Nvidia DALI dataloader')
                assert len(datasets.ImageFolder(self.valdir)) % self.val_batch_size == 0, 'Validation batch size must divide validation dataset size cleanly...  DALI has problems otherwise.'
                self._build_dali_pipeline()

            # Standard torchvision dataloader
            else:
                print('Using torchvision dataloader')
                self._build_torchvision_pipeline()


    def _build_torchvision_pipeline(self):
        preproc_train = [transforms.RandomResizedCrop(self.size, scale=(self.min_crop_size, 1.0)),
                         transforms.RandomHorizontalFlip(),
                         ]

        preproc_val = [transforms.Resize(self.val_size),
                       transforms.CenterCrop(self.size),
                       ]

        train_dataset = datasets.ImageFolder(self.traindir, transforms.Compose(preproc_train))
        val_dataset = datasets.ImageFolder(self.valdir, transforms.Compose(preproc_val))

        self.train_sampler = None
        self.val_sampler = None

        if self.world_size > 1:
            raise NotImplementedError('distributed support not tested yet...')
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            self.val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=(self.train_sampler is None),
            num_workers=self.workers, pin_memory=self.pin_memory, sampler=self.train_sampler, collate_fn=fast_collate)

        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.workers,
            pin_memory=self.pin_memory, sampler=self.val_sampler, collate_fn=fast_collate)

    def _build_dali_pipeline(self, val_on_cpu=True):
        assert self.world_size == 1, 'Distributed support not tested yet'

        iterator_train = DaliIteratorGPU
        if self.dali_cpu:
            iterator_train = DaliIteratorCPU

        self.train_pipe = HybridTrainPipe(batch_size=self.batch_size, num_threads=self.workers, device_id=0,
                                          data_dir=self.traindir, crop=self.size, dali_cpu=self.dali_cpu,
                                          mean=self.mean, std=self.std, local_rank=0,
                                          world_size=self.world_size, shuffle=True, fp16=self.fp16, min_crop_size=self.min_crop_size)

        self.train_pipe.build()
        self.train_loader = iterator_train(pipelines=self.train_pipe, size=self.get_nb_train() / self.world_size, fp16=self.fp16, mean=self.mean, std=self.std, pin_memory=self.pin_memory_dali)

        iterator_val = DaliIteratorGPU
        if val_on_cpu:
            iterator_val = DaliIteratorCPU

        self.val_pipe = HybridValPipe(batch_size=self.val_batch_size, num_threads=self.workers, device_id=0,
                                      data_dir=self.valdir, crop=self.size, size=self.val_size, dali_cpu=val_on_cpu,
                                      mean=self.mean, std=self.std, local_rank=0,
                                      world_size=self.world_size, shuffle=False, fp16=self.fp16)

        self.val_pipe.build()
        self.val_loader = iterator_val(pipelines=self.val_pipe, size=self.get_nb_val() / self.world_size, fp16=self.fp16, mean=self.mean, std=self.std, pin_memory=self.pin_memory_dali)

    def _get_torchvision_loader(self, loader):
        return TorchvisionIterator(loader=loader,
                                   cuda=self.cuda,
                                   fp16=self.fp16,
                                   mean=self.mean,
                                   std=self.std,
                                   )

    def get_train_loader(self):
        """
        Creates & returns an iterator for the training dataset
        :return: Dataset iterator object
        """
        if self.use_dali:
            return self.train_loader
        return self._get_torchvision_loader(loader=self.train_loader)

    def get_val_loader(self):
        """
        Creates & returns an iterator for the training dataset
        :return: Dataset iterator object
        """
        if self.use_dali:
            return self.val_loader
        return self._get_torchvision_loader(loader=self.val_loader)

    def get_nb_train(self):
        """
        :return: Number of training examples
        """
        if self.use_dali:
            return int(self.train_pipe.epoch_size("Reader"))
        return len(datasets.ImageFolder(self.traindir))

    def get_nb_val(self):
        """
        :return: Number of validation examples
        """
        if self.use_dali:
            return int(self.val_pipe.epoch_size("Reader"))
        return len(datasets.ImageFolder(self.valdir))

    def prep_for_val(self):
        self.reset(val_on_cpu=False)

    # This is needed only for DALI
    def reset(self, val_on_cpu=True):
        if self.use_dali:
            clear_memory()

            # Currently we need to delete & rebuild the dali pipeline every epoch,
            # due to a memory leak somewhere in DALI
            print('Recreating DALI dataloaders to reduce memory usage')
            del self.train_loader, self.val_loader, self.train_pipe, self.val_pipe
            clear_memory()

            # taken from: https://stackoverflow.com/questions/1254370/reimport-a-module-in-python-while-interactive
            importlib.reload(dali)
            from dali import HybridTrainPipe, HybridValPipe, DaliIteratorCPU, DaliIteratorGPU

            self._build_dali_pipeline(val_on_cpu=val_on_cpu)

    def set_train_batch_size(self, train_batch_size):
        self.batch_size = train_batch_size
        if self.use_dali:
            del self.train_loader, self.val_loader, self.train_pipe, self.val_pipe
            self._build_dali_pipeline()
        else:
            del self.train_sampler, self.val_sampler, self.train_loader, self.val_loader
            self._build_torchvision_pipeline()

    def get_nb_classes(self):
        """
        :return: The number of classes in the dataset - as indicated by the validation dataset
        """
        return len(datasets.ImageFolder(self.valdir).classes)


def fast_collate(batch):
    """Convert batch into tuple of X and Y tensors."""
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if (nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets


class TorchvisionIterator():
    """
    Iterator to perform final data pre-processing steps:
    -transfer to device (done on 8 bit tensor to reduce bandwidth requirements)
    -convert to fp32/fp16 tensor
    -apply mean/std scaling

    loader (DataLoader): Torchvision Dataloader
    cuda (bool): Transfer tensor to CUDA device
    fp16 (bool): Convert tensor to fp16 instead of fp32
    mean (tuple): Image mean value for each channel
    std (tuple): Image standard deviation value for each channel
    """
    def __init__(self,
                 loader,
                 cuda=False,
                 fp16=False,
                 mean=(0., 0., 0.),
                 std=(1., 1., 1.),
                 ):
        print('Using Torchvision iterator')
        self.loader = iter(loader)
        self.cuda = cuda
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.fp16 = fp16

        if self.cuda:
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()

        if self.fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()

    def __iter__(self):
        return self

    def __next__(self):
        input, target = next(self.loader)

        if self.cuda:
            input = input.cuda()
            target = target.cuda()

        if self.fp16:
            input = input.half()
        else:
            input = input.float()

        input = input.sub_(self.mean).div_(self.std)

        return input, target

    def __len__(self):
        return len(self.loader)
