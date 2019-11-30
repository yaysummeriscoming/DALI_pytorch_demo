import torch, math

import threading
from torch.multiprocessing import Event
from torch._six import queue

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


class HybridTrainPipe(Pipeline):
    """
    DALI Train Pipeline
    Based on the official example: https://github.com/NVIDIA/DALI/blob/master/docs/examples/pytorch/resnet50/main.py
    In comparison to the example, the CPU backend does more computation on CPU, reducing GPU load & memory use.
    This dataloader implements ImageNet style training preprocessing, namely:
    -random resized crop
    -random horizontal flip

    batch_size (int): how many samples per batch to load
    num_threads (int): how many DALI workers to use for data loading.
    device_id (int): GPU device ID
    data_dir (str): Directory to dataset.  Format should be the same as torchvision dataloader,
    containing train & val subdirectories, with image class subfolders
    crop (int): Image output size (typically 224 for ImageNet)
    mean (tuple): Image mean value for each channel
    std (tuple): Image standard deviation value for each channel
    local_rank (int, optional, default = 0) – Id of the part to read
    world_size (int, optional, default = 1) - Partition the data into this many parts (used for multiGPU training)
    dali_cpu (bool, optional, default = False) - Use DALI CPU mode instead of GPU
    shuffle (bool, optional, default = True) - Shuffle the dataset each epoch
    fp16 (bool, optional, default = False) - Output the data in fp16 instead of fp32 (GPU mode only)
    min_crop_size (float, optional, default = 0.08) - Minimum random crop size
    """

    def __init__(self, batch_size, num_threads, device_id, data_dir, crop,
                 mean, std, local_rank=0, world_size=1, dali_cpu=False, shuffle=True, fp16=False,
                 min_crop_size=0.08):

        # As we're recreating the Pipeline every epoch, seed must be -1 (random seed)
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=-1)

        # Tried enabling read_ahead here, but this just slowed down processing ~40%
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size,
                                    random_shuffle=shuffle)

        # let user decide which pipeline works him bets for RN version he runs
        if dali_cpu:
            decode_device = "cpu"
            self.dali_device = "cpu"
            self.flip = ops.Flip(device=self.dali_device)
        else:
            decode_device = "mixed"
            self.dali_device = "gpu"

            output_dtype = types.FLOAT
            if self.dali_device == "gpu" and fp16:
                output_dtype = types.FLOAT16

            self.cmn = ops.CropMirrorNormalize(device="gpu",
                                               output_dtype=output_dtype,
                                               output_layout=types.NCHW,
                                               crop=(crop, crop),
                                               image_type=types.RGB,
                                               mean=mean,
                                               std=std,)

        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decode_device == 'mixed' else 0
        host_memory_padding = 140544512 if decode_device == 'mixed' else 0
        self.decode = ops.ImageDecoderRandomCrop(device=decode_device, output_type=types.RGB,
                                                 device_memory_padding=device_memory_padding,
                                                 host_memory_padding=host_memory_padding,
                                                 random_aspect_ratio=[0.8, 1.25],
                                                 random_area=[min_crop_size, 1.0],
                                                 num_attempts=100)

        # Resize to desired size.  To match torchvision dataloader, use triangular interpolation
        self.res = ops.Resize(device=self.dali_device, resize_x=crop, resize_y=crop,
                              interp_type=types.INTERP_TRIANGULAR)

        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(self.dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")

        # Combined decode & random crop
        images = self.decode(self.jpegs)

        # Resize to desired size
        images = self.res(images)

        if self.dali_device == "gpu":
            output = self.cmn(images, mirror=rng)
        else:
            # CPU backend uses torch to apply mean & std
            output = self.flip(images, horizontal=rng)

        self.labels = self.labels.gpu()
        return [output, self.labels]


class HybridValPipe(Pipeline):
    """
    DALI Validation Pipeline
    Based on the official example: https://github.com/NVIDIA/DALI/blob/master/docs/examples/pytorch/resnet50/main.py
    In comparison to the example, the CPU backend does more computation on CPU, reducing GPU load & memory use.
    This dataloader implements ImageNet style validation preprocessing, namely:
    -resize to specified size
    -center crop to desired size

    batch_size (int): how many samples per batch to load
    num_threads (int): how many DALI workers to use for data loading.
    device_id (int): GPU device ID
    data_dir (str): Directory to dataset.  Format should be the same as torchvision dataloader,
        containing train & val subdirectories, with image class subfolders
    crop (int): Image output size (typically 224 for ImageNet)
    size (int): Resize size (typically 256 for ImageNet)
    mean (tuple): Image mean value for each channel
    std (tuple): Image standard deviation value for each channel
    local_rank (int, optional, default = 0) – Id of the part to read
    world_size (int, optional, default = 1) - Partition the data into this many parts (used for multiGPU training)
    dali_cpu (bool, optional, default = False) - Use DALI CPU mode instead of GPU
    shuffle (bool, optional, default = True) - Shuffle the dataset each epoch
    fp16 (bool, optional, default = False) - Output the data in fp16 instead of fp32 (GPU mode only)
    """

    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size,
                 mean, std, local_rank=0, world_size=1, dali_cpu=False, shuffle=False, fp16=False):

        # As we're recreating the Pipeline every epoch, seed must be -1 (random seed)
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=-1)

        # Tried enabling read_ahead here, but this just slowed down processing ~40%
        # Note: initial_fill is for the shuffle buffer.  As we only want to see every example once, this is set to 1
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=shuffle, initial_fill=1)
        if dali_cpu:
            decode_device = "cpu"
            self.dali_device = "cpu"
            self.crop = ops.Crop(device="cpu", crop=(crop, crop))

        else:
            decode_device = "mixed"
            self.dali_device = "gpu"

            output_dtype = types.FLOAT
            if fp16:
                output_dtype = types.FLOAT16

            self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                                output_dtype=output_dtype,
                                                output_layout=types.NCHW,
                                                crop=(crop, crop),
                                                image_type=types.RGB,
                                                mean=mean,
                                                std=std)

        self.decode = ops.ImageDecoder(device=decode_device, output_type=types.RGB)

        # Resize to desired size.  To match torchvision dataloader, use triangular interpolation
        self.res = ops.Resize(device=self.dali_device, resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        if self.dali_device == 'gpu':
            output = self.cmnp(images)
        else:
            # CPU backend uses torch to apply mean & std
            output = self.crop(images)

        self.labels = self.labels.gpu()
        return [output, self.labels]


class DaliIterator():
    """
    Wrapper class to decode the DALI iterator output & provide iterator that functions the same as torchvision

    pipelines (Pipeline): DALI pipelines
    size (int): Number of examples in set

    Note: allow extra inputs to keep compatibility with CPU iterator
    """

    def __init__(self, pipelines, size, **kwargs):
        self._dali_iterator = DALIClassificationIterator(pipelines=pipelines, size=size)

    def __iter__(self):
        return self

    def __len__(self):
        return int(math.ceil(self._dali_iterator._size / self._dali_iterator.batch_size))


class DaliIteratorGPU(DaliIterator):
    """
    Wrapper class to decode the DALI iterator output & provide iterator that functions the same as torchvision

    pipelines (Pipeline): DALI pipelines
    size (int): Number of examples in set

    Note: allow extra inputs to keep compatibility with CPU iterator
    """

    def __next__(self):
        try:
            data = next(self._dali_iterator)
        except StopIteration:
            print('Resetting DALI loader')
            self._dali_iterator.reset()
            raise StopIteration

        # Decode the data output
        input = data[0]['data']
        target = data[0]['label'].squeeze().long()

        return input, target


def _preproc_worker(dali_iterator, cuda_stream, fp16, mean, std, output_queue, proc_next_input, done_event, pin_memory):
    """
    Worker function to parse DALI output & apply final pre-processing steps
    """

    while not done_event.is_set():
        # Wait until main thread signals to process next input - normally when it's taken the last processed input
        proc_next_input.wait()
        proc_next_input.clear()

        if done_event.is_set():
            print('Shutting down preproc thread')
            break

        try:
            data = next(dali_iterator)

            # Decode the data output
            input_orig = data[0]['data']
            target = data[0]['label'].squeeze().long()  # DALI should already output target on device

            # Copy to GPU & apply final processing in seperate CUDA stream
            with torch.cuda.stream(cuda_stream):
                input = input_orig
                if pin_memory:
                    input = input.pin_memory()
                    del input_orig  # Save memory
                input = input.cuda(non_blocking=True)

                input = input.permute(0, 3, 1, 2)

                # Input tensor is transferred to GPU as 8 bit, to save bandwidth
                if fp16:
                    input = input.half()
                else:
                    input = input.float()

                input = input.sub_(mean).div_(std)

            # Put the result on the queue
            output_queue.put((input, target))

        except StopIteration:
            print('Resetting DALI loader')
            dali_iterator.reset()
            output_queue.put(None)


class DaliIteratorCPU(DaliIterator):
    """
    Wrapper class to decode the DALI iterator output & provide iterator that functions the same as torchvision
    Note that permutation to channels first, converting from 8 bit to float & normalization are all performed on GPU

    pipelines (Pipeline): DALI pipelines
    size (int): Number of examples in set
    fp16 (bool): Use fp16 as output format, f32 otherwise
    mean (tuple): Image mean value for each channel
    std (tuple): Image standard deviation value for each channel
    pin_memory (bool): Transfer input tensor to pinned memory, before moving to GPU
    """
    def __init__(self, fp16=False, mean=(0., 0., 0.), std=(1., 1., 1.), pin_memory=True, **kwargs):
        super().__init__(**kwargs)
        print('Using DALI CPU iterator')
        self.stream = torch.cuda.Stream()

        self.fp16 = fp16
        self.mean = torch.tensor(mean).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor(std).cuda().view(1, 3, 1, 1)
        self.pin_memory = pin_memory

        if self.fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()

        self.proc_next_input = Event()
        self.done_event = Event()
        self.output_queue = queue.Queue(maxsize=5)
        self.preproc_thread = threading.Thread(
            target=_preproc_worker,
            kwargs={'dali_iterator': self._dali_iterator, 'cuda_stream': self.stream, 'fp16': self.fp16, 'mean': self.mean, 'std': self.std, 'proc_next_input': self.proc_next_input, 'done_event': self.done_event, 'output_queue': self.output_queue, 'pin_memory': self.pin_memory})
        self.preproc_thread.daemon = True
        self.preproc_thread.start()

        self.proc_next_input.set()

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.output_queue.get()
        self.proc_next_input.set()
        if data is None:
            raise StopIteration
        return data

    def __del__(self):
        self.done_event.set()
        self.proc_next_input.set()
        torch.cuda.current_stream().wait_stream(self.stream)
        self.preproc_thread.join()


class DaliIteratorCPUNoPrefetch(DaliIterator):
    """
    Wrapper class to decode the DALI iterator output & provide iterator that functions the same as torchvision
    Note that permutation to channels first, converting from 8 bit to float & normalization are all performed on GPU

    pipelines (Pipeline): DALI pipelines
    size (int): Number of examples in set
    fp16 (bool): Use fp16 as output format, f32 otherwise
    mean (tuple): Image mean value for each channel
    std (tuple): Image standard deviation value for each channel
    pin_memory (bool): Transfer input tensor to pinned memory, before moving to GPU
    """
    def __init__(self, fp16, mean, std, pin_memory=True, **kwargs):
        super().__init__(**kwargs)
        print('Using DALI CPU iterator')

        self.stream = torch.cuda.Stream()

        self.fp16 = fp16
        self.mean = torch.tensor(mean).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor(std).cuda().view(1, 3, 1, 1)
        self.pin_memory = pin_memory

        if self.fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()

    def __next__(self):
        data = next(self._dali_iterator)

        # Decode the data output
        input = data[0]['data']
        target = data[0]['label'].squeeze().long()  # DALI should already output target on device

        # Copy to GPU & apply final processing in seperate CUDA stream
        input = input.cuda(non_blocking=True)

        input = input.permute(0, 3, 1, 2)

        # Input tensor is transferred to GPU as 8 bit, to save bandwidth
        if self.fp16:
            input = input.half()
        else:
            input = input.float()

        input = input.sub_(self.mean).div_(self.std)
        return input, target
