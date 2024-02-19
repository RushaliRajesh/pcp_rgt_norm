from __future__ import print_function

import argparse
import os
import sys
import random
import math
import shutil
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import os
import pdb
import pytorch3d.ops
import numpy as np
import torch.nn.init as init_i
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F 
from tqdm import tqdm
from model_4params import CNN_nopos as CNN
# from dataset import PointcloudPatchDataset, RandomPointcloudPatchSampler, SequentialShapeRandomPointcloudPatchSampler
from dataset_posenc import PointcloudPatchDataset, RandomPointcloudPatchSampler, SequentialShapeRandomPointcloudPatchSampler

# from tensorboardX import SummaryWriter 
# # default `log_dir` is "runs" - we'll be more specific here
# writer = SummaryWriter('runs/train_diff')
# writerval = SummaryWriter('runs_val/val_diff')

def parse_arguments():
    parser = argparse.ArgumentParser()

    # naming / file handling
    parser.add_argument('--name', type=str, default='my_single_scale_normal', help='training run name')
    parser.add_argument('--desc', type=str, default='My training run for single-scale normal estimation.', help='description')
    parser.add_argument('--indir', type=str, default='./pclouds', help='input folder (point clouds)')
    parser.add_argument('--outdir', type=str, default='./models', help='output folder (trained models)')
    parser.add_argument('--logdir', type=str, default='./logs', help='training log folder')
    parser.add_argument('--trainset', type=str, default='trainingset_whitenoise.txt', help='training set file name')
    parser.add_argument('--testset', type=str, default='validationset_whitenoise.txt', help='test set file name')
    parser.add_argument('--saveinterval', type=int, default='10', help='save model each n epochs')
    parser.add_argument('--refine', type=str, default='', help='refine model at this path')
    parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU')

    # training parameters
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--patch_radius', type=float, default=[0.05], nargs='+', help='patch radius in multiples of the shape\'s bounding box diagonal, multiple values for multi-scale.')
    parser.add_argument('--patch_center', type=str, default='point', help='center patch at...\n'
                        'point: center point\n'
                        'mean: patch mean')
    parser.add_argument('--patch_point_count_std', type=float, default=0, help='standard deviation of the number of points in a patch')
    parser.add_argument('--patches_per_shape', type=int, default=1000, help='number of patches sampled from each shape in an epoch')
    parser.add_argument('--workers', type=int, default=1, help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=100, help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    parser.add_argument('--seed', type=int, default=3627473, help='manual seed')
    parser.add_argument('--training_order', type=str, default='random', help='order in which the training patches are presented:\n'
                        'random: fully random over the entire dataset (the set of all patches is permuted)\n'
                        'random_shape_consecutive: random over the entire dataset, but patches of a shape remain consecutive (shapes and patches inside a shape are permuted)')
    parser.add_argument('--identical_epochs', type=int, default=False, help='use same patches in each epoch, mainly for debugging')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='gradient descent momentum')
    parser.add_argument('--use_pca', type=int, default=False, help='Give both inputs and ground truth in local PCA coordinate frame')
    parser.add_argument('--normal_loss', type=str, default='ms_euclidean', help='Normal loss type:\n'
                        'ms_euclidean: mean square euclidean distance\n'
                        'ms_oneminuscos: mean square 1-cos(angle error)')

    # model hyperparameters
    parser.add_argument('--outputs', type=str, nargs='+', default=['unoriented_normals'], help='outputs of the network, a list with elements of:\n'
                        'unoriented_normals: unoriented (flip-invariant) point normals\n'
                        'oriented_normals: oriented point normals\n'
                        'max_curvature: maximum curvature\n'
                        'min_curvature: mininum curvature')
    parser.add_argument('--use_point_stn', type=int, default=True, help='use point spatial transformer')
    parser.add_argument('--use_feat_stn', type=int, default=True, help='use feature spatial transformer')
    parser.add_argument('--sym_op', type=str, default='max', help='symmetry operation')
    parser.add_argument('--point_tuple', type=int, default=1, help='use n-tuples of points as input instead of single points')
    parser.add_argument('--points_per_patch', type=int, default=500, help='max. number of points per patch')

    return parser.parse_args()

# def loss_func(norm, init, pred):
#     inter = torch.add(init, pred)
    # pdb.set_trace()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
save_dir = "params_cor_norm_MSE_nodropout"
os.makedirs(save_dir, exist_ok=True) 

def get_angle(a, b):
    """Angle between vectors"""
    # print(a.shape, b.shape)
    a = a / torch.linalg.norm(a, dim = 1).unsqueeze(1)  
    b = b / torch.linalg.norm(b, dim= 1).unsqueeze(1)
    # print(a.device, b.device)
    dot = torch.sum(a * b, dim=1).clamp(-1, 1).to(device)
    # print("dot shape: ",dot.shape)
    return torch.acos(dot)

def param2norm (pred_params , ori_params, init, ori_norms, check_ori = False):
    init = nn.functional.normalize(init, dim=1)
    sin = pred_params[:, :3]
    cos = pred_params[:,3].clamp(-1,1)
    theta = torch.acos(cos)
    sin_theta_norm = torch.sin(theta).reshape(-1,1)
    axis =torch.where(sin_theta_norm > 0, sin / (sin.norm(dim=1).unsqueeze(1)), torch.tensor([[1.0, 0.0, 0.0]]).to(device)) 
    axis_angle = axis * theta.unsqueeze(1) #checked

    rot_mat = pytorch3d.transforms.axis_angle_to_matrix(axis_angle)
    rot_norms = torch.matmul(rot_mat, init.unsqueeze(2)).squeeze(2)
    rot_norms = nn.functional.normalize(rot_norms, dim=1)
    angle_diff = get_angle(ori_norms, rot_norms)

    '''check for the original_params and angle diff:'''
    if(check_ori == True):
        ori_norms = nn.functional.normalize(ori_norms, dim=1)
        sin_ori = ori_params[:, :3]
        cos_ori = ori_params[:,3].clamp(-1,1)
        theta_ori = torch.acos(cos_ori)
        sin_theta_norm_ori = torch.sin(theta_ori).reshape(-1,1)
        axis_ori =torch.where(sin_theta_norm_ori > 0, sin_ori / (sin_ori.norm(dim=1).unsqueeze(1)), torch.tensor([[1.0, 0.0, 0.0]]).to(device))
        axis_angle_ori = axis_ori * theta_ori.unsqueeze(1) 

        rot_mat_ori = pytorch3d.transforms.axis_angle_to_matrix(axis_angle_ori)
        rot_norms_ori = torch.matmul(rot_mat_ori, init.unsqueeze(2)).squeeze(2)
        rot_norms_ori = nn.functional.normalize(rot_norms_ori, dim=1)
        angle_diff_ori = get_angle(ori_norms, rot_norms_ori)
        # print(angle_diff_ori)
        # print(angle_diff_ori.dtype)
        return angle_diff, angle_diff_ori, rot_norms, rot_norms_ori

    return angle_diff, None, rot_norms, None

def train_pcpnet(opt):

    # device = torch.device("cpu" if opt.gpu_idx < 0 else "cuda:%d" % opt.gpu_idx)
    target_features = []
    pred_dim = 0
    for o in opt.outputs:
        if o == 'unoriented_normals' or o == 'oriented_normals':
            if 'normal' not in target_features:
                target_features.append('normal')
            pred_dim += 3
        elif o == 'max_curvature' or o == 'min_curvature':
            if o not in target_features:
                target_features.append(o)

    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)

    print("Random Seed: %d" % (opt.seed))
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    import time
    
    # create train and test dataset loaders
    # pdb.set_trace()
    train_dataset = PointcloudPatchDataset(
        task = 'mse_6_4params_no_pos',
        root=opt.indir,
        shape_list_filename=opt.trainset,
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        patch_features=target_features,
        point_count_std=opt.patch_point_count_std,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        use_pca=opt.use_pca,
        center=opt.patch_center,
        point_tuple=opt.point_tuple,
        cache_capacity=opt.cache_capacity)
    
    print(len(train_dataset))

    train_datasampler = RandomPointcloudPatchSampler(
        train_dataset,
        patches_per_shape=opt.patches_per_shape,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs)
    
    print(len(train_datasampler))        

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_datasampler,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers))
    
    for i in train_dataloader:
        print(len(i[0]))
        break
    
    test_dataset = PointcloudPatchDataset(
        task = 'mse_6_4params_no_pos',
        root=opt.indir,
        shape_list_filename=opt.testset,
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        patch_features=target_features,
        point_count_std=opt.patch_point_count_std,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        use_pca=opt.use_pca,
        center=opt.patch_center,
        point_tuple=opt.point_tuple,
        cache_capacity=opt.cache_capacity)
    test_datasampler = RandomPointcloudPatchSampler(
        test_dataset,
        patches_per_shape=1000,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=test_datasampler,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers))

    print(len(test_datasampler))    
    
    model = CNN()
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    model = model.to(device)
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
    def initialize_weights(model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                init_i.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init_i.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init_i.constant_(m.weight, 1)
                init_i.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init_i.xavier_normal_(m.weight)
                if m.bias is not None:
                    init_i.constant_(m.bias, 0)

    initialize_weights(model)

    # model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    #sgd optim
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=opt.momentum, weight_decay=1e-5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', threshold = 1e-3)
    # optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10,15,20,25,30], gamma=0.5) # milestones in number of optimizer iterations
    tr_loss_per_epoch = []
    val_loss_per_epoch = []

    for epoch in range(opt.nepoch):
        train_loss = []
        val_loss = []
        model.train()
        pbar = tqdm(train_dataloader)
        for i, (data, init, params, norms) in enumerate(pbar, 0):
            inputs = data.float()
            inputs = inputs.permute(0,3,1,2)
            params = params.float()
            init = init.float()
            norms = norms.float()
            inputs, params, init, norms = inputs.to(device), params.to(device), init.to(device), norms.to(device)
            optimizer.zero_grad()
            out = model(inputs, init)
            tr_angle_diff,_,out_norms,_ = param2norm(out, params, init, norms) #pred_params , ori_params, init, ori_norms, check_ori = False
            loss = mse_loss(out, params)
            # pdb.set_trace()

            loss.backward()

            prev_param_values = {name: p.clone().detach() for name, p in model.named_parameters()}

            optimizer.step()

            tolerance = 1e-6  # Define a small tolerance value
            for name, p in model.named_parameters():
                diff = torch.abs(prev_param_values[name] - p.detach())
                max_diff = torch.max(diff)
                if max_diff <= tolerance:
                    print(f"{name} is not updating significantly (max diff: {max_diff.item()})")
            # for name, parameter in model.named_parameters():
            #     if parameter.grad is not None:
            #         grad_norm = parameter.grad.norm(2)
            #         print(f"{name}: gradient norm = {grad_norm}")
            if torch.isnan(data).any() or torch.isinf(data).any():
                print("Data contains NaN or inf values")
                break

            train_loss.append(loss.item())
            pbar.set_postfix(Epoch=epoch, tr_loss=loss.item())
            pbar.set_description('Iter: {}'.format(loss.item()))

        tot_train_loss = np.mean(train_loss)  
        tr_loss_per_epoch.append(tot_train_loss)

        bef_lr = optimizer.param_groups[0]['lr']
        scheduler.step(tot_train_loss)
        aft_lr = optimizer.param_groups[0]['lr']
        if(bef_lr != aft_lr):
            print(f'epoch: {epoch}, learning rate: {bef_lr} -> {aft_lr}')

    
        with torch.no_grad():
            model.eval()
            pbar1 = tqdm(test_dataloader)
            for i, (data, init, params, norms) in enumerate(pbar1, 0):
                inputs = data.float()
                inputs = inputs.permute(0,3,1,2)
                params = params.float()
                init = init.float()
                norms = norms.float()
                inputs, params, init, norms = inputs.to(device), params.to(device), init.to(device), norms.to(device)
                
                out = model(inputs, init)
                val_angle_diff,_,out_norms,_ = param2norm(out, params, init, norms)
                loss = mse_loss(out, params)
                val_loss.append(loss.item())
                pbar1.set_postfix(Epoch=epoch, val_loss=loss.item())
                
        tot_val_loss = np.mean(val_loss)
        val_loss_per_epoch.append(tot_val_loss)

        if epoch % 10 == 0:
            EPOCH = epoch
            PATH = f"{save_dir}/{EPOCH}.pt"
            LOSS = tot_train_loss

            torch.save({
                        'epoch': EPOCH,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': LOSS,
                        'batchsize' : opt.batchSize,
                        'val_losses_so_far' : val_loss_per_epoch,
                        'train_losses_so_far' : tr_loss_per_epoch
                        }, PATH)
            print("Model saved at epoch: ", epoch)

        print(f'epoch: {epoch} training loss: {tot_train_loss}, train_angle_diff (rad): {tr_angle_diff.sum().item() / tr_angle_diff.shape[0]} , {PATH}')
        print(f'epoch: {epoch} val loss: {tot_val_loss} val_angle_diff (rad): {val_angle_diff.sum().item() / val_angle_diff.shape[0]}')

        # writer.add_scalar('train loss',tot_train_loss, epoch)
        # writerval.add_scalar('val loss',
        #                     tot_val_loss,
        #                     epoch)


if __name__ == '__main__':
    train_opt = parse_arguments()
    train_pcpnet(train_opt)


