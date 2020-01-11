# DALI_pytorch_demo
Example code showing how to use Nvidia DALI in pytorch, with fallback to torchvision.  Contains a few differences to the official Nvidia example:

* Reimport DALI & recreate dataloaders at end of every epoch to reduce long term memory usage
* Move CPU DALI pipeline completely to CPU, freeing up GPU resources
* Keep DALI validation pipeline off GPU during training, reducing GPU memory usage

Compared to the official example, these mods allow for a ~50% increase in max batch size (tested using ResNet18 on a GCloud V100 instance with 10 workers:

Dataloader Type          | Max Batch Size
---                      | ---
DALI GPU reference       | 640
DALI GPU                 | 928 / 45% increase
DALI CPU reference       | 800
DALI CPU                 | 1216 / 52% increase
Torchvision w/ PIL-SIMD  | 1248

Here are some benchmarks on a Google Cloud V100 instance with 12 vCPUs (6 physical cores), 78GB RAM, Apex FP16 training with Shufflenet V2 0.5 & batch size 512:

Dataloader Type                                  | Speed (images/s)
---                                              | ---
DALI GPU                                         | 3910
DALI CPU                                         | 1816
Torchvision w/ PIL-SIMD                          | 1058

You can read the correspondig blog post <here>
