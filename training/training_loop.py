# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc

import random
from torchvision import transforms 
import torchvision.transforms.functional as TF

#----------------------------------------------------------------------------
# Auxiliary functions for data augmentation in patches 

class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

rotation_transform = MyRotationTransform(angles=[0, 90, 180, 270])
hflip_transform = transforms.RandomHorizontalFlip()

#----------------------------------------------------------------------------
# Auxiliary function to ensure that data has range [-1,1] 

def get_data_scaler(data_centered):
  """Data normalizer. Assume data are always in [0, 1]."""
  if data_centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x

#----------------------------------------------------------------------------
# Auxiliary function to sample patches during training 

def sample_patch(data,batch_sz,patch_sz,n_patches,device,augment_pipe):
  n, n_ch, w, h = data.size()
  n_mat = n_ch#//2 #//= 2
  h_start = np.random.choice(np.array(range(0,h-patch_sz)), batch_sz*n_patches)
  w_start = np.random.choice(np.array(range(0,w-patch_sz)), batch_sz*n_patches)
  out = torch.zeros((n*n_patches,n_mat,patch_sz,patch_sz)).to(device)
  k = 0
  for j in range(n):
    for i in range(n_patches):
      idx_h = torch.tensor(range(h_start[k], h_start[k]+patch_sz)).to(device)
      idx_w = torch.tensor(range(w_start[k], w_start[k]+patch_sz)).to(device)
      if augment_pipe is None:
        data[j,:,:,:] = hflip_transform(data[j,:,:,:])
        data[j,:,:,:] = rotation_transform(data[j,:,:,:])
      out[k,:,:,:] = data[j,0:n_mat,:,:].index_select(1, idx_h).index_select(2, idx_w)
      k += 1
  return out

#----------------------------------------------------------------------------
# Auxiliary function to check memory usage of data tensors
def get_info(data):
  print(f'Size: {data.size()}')
  print(f'Dtype: {data.dtype}')
  print(f'Bytes used: {data.element_size()*data.nelement()}')
  print(f'Device: {data.get_device()}')

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    dataset_n_kwargs    = {},       # Options for training noise set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 10000,    # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_kimg         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
    stf                 = False,
    pfgmpp              = False,
    rbatch              = 4096,
    D                   = 128,
    opts                = None,
    patch_sz            = None,     # Patch size for denoising training
    n_patches           = None,     # Number of patches extracted from each image (effecive batch_sz = batch_sz*n_patches)
    weight_n            = 1,        # Weight given to noise patch 
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    if stf:
        batch_gpu_total = rbatch
        batch_gpu = batch_gpu_total
        num_accumulation_rounds = 1
    else:
        batch_gpu_total = batch_size // dist.get_world_size()
        if batch_gpu is None or batch_gpu > batch_gpu_total:
            batch_gpu = batch_gpu_total
        # default is one
        num_accumulation_rounds = batch_gpu_total // batch_gpu
        # print("check num_accumulation_rounds", num_accumulation_rounds)
        assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    if len(dataset_n_kwargs)>0 and dataset_kwargs.path != dataset_n_kwargs.path:
      dataset_n_obj = dnnlib.util.construct_class_by_name(**dataset_n_kwargs) # subclass of training.dataset.Dataset
      dataset_n_sampler = misc.InfiniteSampler(dataset=dataset_n_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
      dataset_n_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_n_obj, sampler=dataset_n_sampler, batch_size=batch_gpu, **data_loader_kwargs))
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))
    scaler = get_data_scaler(True) # hardcoded for now

    # Construct network.
    dist.print0('Constructing network...')
    if patch_sz is not None:    
      interface_kwargs = dict(img_resolution=patch_sz, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim,
                             pfgmpp=pfgmpp, D=D)
    else:
      interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim,
                             pfgmpp=pfgmpp, D=D)
    if len(dataset_n_kwargs) >0:
      if  dataset_kwargs.path == dataset_n_kwargs.path:
        interface_kwargs['img_channels'] = dataset_obj.num_channels // 2
    #print(network_kwargs)
    #print(interface_kwargs)
    #assert 1 == 2
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    #print(net)
    #print(net.img_channels)
    #assert 1 == 2
    net.train().requires_grad_(True).to(device)
    if dist.get_rank() == 0:
        B = batch_size // dist.get_world_size()
        with torch.no_grad():
            if len(dataset_n_kwargs)>0:
              images = torch.zeros([B, net.img_channels*2, net.img_resolution, net.img_resolution], device=device)
            else:
              images = torch.zeros([B, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            sigma = torch.ones([B], device=device)
            labels = torch.zeros([B, net.label_dim], device=device)
            misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)
            del images, sigma, labels
            
    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_kwargs.D = D
    loss_kwargs.N = net.img_channels * net.img_resolution * net.img_resolution
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        optimizer.load_state_dict(data['optimizer_state'])
        cur_nimg = data['step']
        cur_tick = cur_nimg // (1000 * kimg_per_tick)
        del data # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        cur_nimg = data['step']
        cur_tick = cur_nimg // (1000 * kimg_per_tick)+1
        del data # conserve memory

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    while True:

        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                if len(dataset_n_kwargs)>0 and dataset_kwargs.path != dataset_n_kwargs.path:
                  images, labels = next(dataset_iterator)
                  noise, labels = next(dataset_n_iterator) 
                  labels = labels.to(device)
                  images = scaler(torch.cat((images+weight_n*noise,images),dim=1))
                  if patch_sz is not None:
                    images = sample_patch(images,batch_size,patch_sz,n_patches,device='cpu',augment_pipe=augment_pipe)
                else:
                  images, labels = next(dataset_iterator)
                  #images = images.to(device).to(torch.float32) / 127.5 - 1
                  images = scaler(images) #.to(device).to(torch.float32)
                  labels = labels.to(device)
                  if patch_sz is not None: 
                      images = sample_patch(images,batch_size,patch_sz,n_patches,device='cpu',augment_pipe=augment_pipe)
                if stf:
                    # divide the mini-batch by the device number
                    # per-device 128 samples
                    # stf = 1024
                    batch_images = images[:batch_size // dist.get_world_size()].to(device)
                    batch_labels = labels[:batch_size // dist.get_world_size()].to(device)
                else:
                    batch_images = images.to(device)
                    batch_labels = labels.to(device)

                # B * C * H * W
                loss = loss_fn(net=ddp, images=batch_images, labels=batch_labels, augment_pipe=augment_pipe, stf=stf,
                               pfgmpp=pfgmpp, ref_images=images)
                training_stats.report('Loss/loss', loss)
                #dist.print0("loss:", loss.mean().item())
                loss.sum().mul(loss_scaling / (batch_size // dist.get_world_size())).backward()

        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            torch.save(dict(optimizer_state=optimizer.state_dict(), step=cur_nimg, ema=ema, net=net),
                       os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

        # Save full dump for each
        if dist.get_rank() == 0:
          torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-000000.pt'))

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
