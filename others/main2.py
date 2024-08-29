'''
Training model
    Save net parameters with the best performance.
    Visualize pictures including: images(before and after training), evaluation data plots.

Updated 2022/3/23 By Aurora Yin
'''
import os
import sys
import argparse
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt
from model import *
from dataPro import *
from utils import *
import args_option
from datasets import *
import torch.distributed as dist
from args_option import *
from warmup_scheduler.scheduler  import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from apex import amp

torch.cuda.empty_cache()

############ Parser ##############
opt = Options().init(argparse.ArgumentParser(description='image denoising')).parse_args()
# print(opt)

############ Set GPUs ##############
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

############ Set Seeds #############
random.seed(123456)
np.random.seed(123456)
torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)

save_fp = '../'  # folder path to save best paras of model

######### Logs and Tensorboard ###########
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, './auxiliary/'))
print(dir_name)
log_dir = os.path.join(dir_name, 'log_denoise_aswhole', opt.arch + '_logs')
# log_dir = os.path.join(dir_name, 'log_pro_4', opt.arch + '_logs') # model -- del FFN
# log_dir = os.path.join(dir_name, 'log_derain', opt.arch + '_logs')

writer = SummaryWriter("../../Plots")  # use Tensorboard

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logname = os.path.join(log_dir, datetime.datetime.now().isoformat() + '.txt')
print("Now time is : ", datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
model_dir = os.path.join(log_dir, 'model_log')

if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

#########################################

channel = 3  # RGB
layers_num = 20
mode = 'S'  # training mode
noise_level = 35

lr_list = []  # learning rate
loss_list = []
psnr_list = []
ssim_list = []
best_psnr = 0  # find net paras with best psnr

############ Model #############
if opt.arch == 'DnCNN_Pro':
    model = DnCNN_Pro(channels=channel, num_of_layers=layers_num)
elif opt.arch == 'AtUNet':
    model = AtUNet(img_size=opt.train_ps, embed_dim=opt.embed_dim, win_size=opt.win_size,
                   token_projection=opt.token_projection, token_mlp=opt.token_mlp)
elif opt.arch == 'UNet':
    model = UNet(dim=opt.embed_dim)
else:
    raise Exception("Arch error!")

with open(logname, 'a') as f:
    f.write(str(opt) + '\n')
    f.write(str(model) + '\n')

########## DataParallel ###########
device = torch.device("cuda")
model = model.to(device)
# device_ids = [i for i in range(torch.cuda.device_count())]
# model = nn.DataParallel(model).cuda()

############# Use DistributedDataParallel
# if opt.local_rank != -1:
#     torch.cuda.set_device(opt.local_rank)
#     device = torch.device("cuda", opt.local_rank)
#     torch.distributed.init_process_group(backend="nccl", init_method='env://')
#
# model.to(device)
torch.cuda.set_device(opt.local_rank)
torch.distributed.init_process_group(
    backend='nccl',
    init_method='env://',
    world_size=4,
    rank=opt.local_rank
)

############ Load Dataset ############
print('------------------------------------------------------------------------------')
print(">>>>>>>>>>>>> Loading datasets ... <<<<<<<<<<<<<")
img_options_train = {'patch_size': opt.train_ps}
assert os.path.exists(opt.train_dir)
train_dataset = TrainDataset(opt.train_dir, img_options_train)
train_sampler = DistributedSampler(train_dataset)  #### add sampler
batch_sampler_train = torch.utils.data.BatchSampler(
    train_sampler,
    batch_size=opt.batch_size,
    drop_last=False,
)
# train_loader = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler_train, batch_size=opt.batch_size, shuffle=True,
#                           num_workers=opt.train_workers, pin_memory=True, drop_last=False)
train_loader = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler_train, num_workers=opt.train_workers,
                          pin_memory=True)

assert os.path.exists(opt.val_dir)
val_dataset = ValDataset(opt.val_dir)
val_sampler = DistributedSampler(val_dataset)  #### add sampler
batch_sampler_val = torch.utils.data.BatchSampler(
    val_sampler,
    batch_size=opt.batch_size,
    drop_last=False,
)
val_loader = DataLoader(dataset=val_dataset, batch_sampler=batch_sampler_val, num_workers=opt.eval_workers,
                        pin_memory=True)
# val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False,
#                         num_workers=opt.eval_workers, pin_memory=True, drop_last=False)

print("\nFinish loading datasets ...")
len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print("Sizeof training set: ", len_trainset, ", sizeof validation set: ", len_valset)
print('------------------------------------------------------------------------------')

########### Loss and Optimizer ###########
if opt.loss == 'Charbonnier':
    criterion = CharbonnierLoss().cuda()
elif opt.loss == 'MSE':
    criterion = nn.MSELoss(size_average=False).cuda()
else:
    raise Exception("Loss error!")

optimizer = optim.AdamW(model.parameters(), lr=opt.lr_init, weight_decay=1e-4)
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")  # 这里是“欧一”，不是“零一”
# model = nn.DataParallel(model).cuda()

# model = nn.DistributedDataParallel(model).cuda()

# num_gpus = torch.cuda.device_count()
# if num_gpus > 1:
#     print('\nuse {} gpus!\n'.format(num_gpus))
#     model = nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank],
#                                                 output_device=opt.local_rank)
if torch.cuda.device_count() > 1:
    model = torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[opt.local_rank])
warmup = True

########### Resume Training ############
start_epoch = 1
opt.resume = True
if opt.resume:
    path_chk_rest = opt.pretrain_weights
    load_checkpoint(model, path_chk_rest)
    start_epoch = load_start_epoch(path_chk_rest) + 1
    lr = load_optim(optimizer, path_chk_rest)

    for p in optimizer.param_groups: p['lr'] = lr
    warmup = False
    new_lr = lr
    print('------------------------------------------------------------------------------')
    print(">>>>>>>> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epoch - start_epoch + 1, eta_min=1e-6)

########### Warmup ###########
if warmup:
    print(">>>>>>>> Warmuping ......")
    warmup_epochs = 5
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epoch - warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()
else:
    # scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.2)  # update lr
    base_lr = 0.0002
    max_lr = 5e-6

    # 设置学习率调节方法
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=500, step_size_down=500,
                                                  mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle',
                                                  cycle_momentum=False, base_momentum=0.8, max_momentum=0.9,
                                                  last_epoch=-1)
    scheduler.step()

######### validation ###########
# with torch.no_grad():
#     psnr_val_rgb = []
#     ssim_val_rgb = []
#     index = 1
#     for ii, data_val in enumerate((val_loader), 0):
#         target = data_val[0].cuda()
#         input_ = data_val[1].cuda()
#         filenames = data_val[2]
#         psnr_val_rgb.append(batch_PSNR(input_, target, 1.).item())
#         ssim_val_rgb.append(batch_SSIM(input_, target, True).item())
#         print(batch_PSNR(input_, target, 1.).item())
#         print(batch_SSIM(input_, target, True).item())
#         index += 1
#     psnr_val_rgb = sum(psnr_val_rgb) / index
#     ssim_val_rgb = sum(ssim_val_rgb) / index
#     print('\nInput & GT (PSNR) -->%.4f dB  (SSIM) -->%.4f' % (psnr_val_rgb, ssim_val_rgb))

############################ Training Model ###############################
print('------------------------------------------------------------------------------')
print(">>>>>>>>>>>>> Start training ... <<<<<<<<<<<<<")

# if os.path.exists('../net_paras_1.pth'):
#     model.load_state_dict(torch.load('../net_paras_1.pth'))

# if os.path.exists('../net_paras_2.pth'):
#     model.load_state_dict(torch.load('../net_paras_2.pth'))

# if os.path.exists('../net_paras_3.pth'):
#     model.load_state_dict(torch.load('../net_paras_3.pth'))
print()
best_psnr = 0
best_epoch = 0
best_iter = 0
eval_now = len(train_loader) // 4
torch.cuda.empty_cache()
for epoch in range(start_epoch, opt.epoch + 1):
    train_sampler.set_epoch(epoch)
    val_sampler.set_epoch(epoch)

    print(">>>>>>>> Current learning rate: %f" % scheduler.get_lr()[0])
    start_t = time.time()
    epochLoss = 0

    for i, imdata in enumerate(train_loader, 0):
        # torch.cuda.empty_cache()
        model.train()
        # model.zero_grad()
        optimizer.zero_grad()

        ground_truth = imdata[0].cuda()
        noisy_img = imdata[1].cuda()

        with torch.cuda.amp.autocast():
            print(noisy_img.size())

            # pad input image to be a multiple of args.win_size
            _, _, h_old, w_old = noisy_img.size()
            h_pad = (h_old // opt.win_size + 1) * opt.win_size - h_old
            w_pad = (w_old // opt.win_size + 1) * opt.win_size - w_old
            noisy_img = torch.cat([noisy_img, torch.flip(noisy_img, [2])], 2)[:, :, :h_old + h_pad, :]
            noisy_img = torch.cat([noisy_img, torch.flip(noisy_img, [3])], 3)[:, :, :, :w_old + w_pad]
            print(noisy_img.shape)


            train_out = model(noisy_img)
            # denoised_img = torch.clamp(train_out, 0., 1.)

            train_out = train_out[..., :h_old, :w_old]
            denoised_img = train_out.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            denoised_img = np.transpose(denoised_img[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
            denoised_img = (denoised_img * 255.0).round().astype(np.uint8)  # float32 to uint8
            print(denoised_img.shape)


            loss = criterion(train_out, ground_truth)  # AtUNet

        ground_truth = ground_truth.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        ground_truth = (ground_truth * 255.0).round().astype(np.uint8)  # float32 to uint8
        ground_truth = ground_truth[:h_old, :w_old, ...]  # crop ground_truth
        ground_truth = np.squeeze(ground_truth)

        psnr_train = batch_PSNR(ground_truth, denoised_img, True)
        # ssim_train = batch_SSIM(ground_truth, denoised_img, True)
        print("[epoch %d/%d][%d/%d] loss: %.4f PSNR_train: %.4f" %
              (epoch, opt.epoch, i + 1, len(train_loader), loss.item(), psnr_train.item()))

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # loss.backward()
        optimizer.step()
        epochLoss += loss.item()

        ## Evaluation (per batch) ##
        # model.eval()
        # denoise_img = torch.clamp(noisy_img-train_out, 0., 1.)   # 网络输出结果为图片中的噪音，原始噪声图像-噪音 = 去噪图像
        # print(denoise_img.size())

        if (i + 1) % eval_now == 0 and i > 0:
            with torch.no_grad():
                model.eval()
                index = 1
                psnr_sum = 0
                ssim_sum = 0
                for j, valdata in enumerate(val_loader, 0):
                    ground_truth = valdata[0].cuda()
                    noisy_val_img = valdata[1].cuda()
                    filenames = valdata[2]

                    with torch.cuda.amp.autocast():
                        val_out = model(noisy_val_img)

                    denoised_val_img = torch.clamp(val_out, 0., 1.)
                    psnr_sum += batch_PSNR(ground_truth, denoised_val_img, False).item()
                    # ssim_sum += batch_SSIM(ground_truth, denoised_val_img, True).item()
                    index += 1
                psnr = psnr_sum / len_valset
                psnr = psnr_sum / (index*8)
                # ssim = ssim_sum/index
                print("psnr_sum=%d, index=%d, val_len=%d" % (psnr_sum, index, val_dataset.__len__()))
                print("[Epoch %d Iteration %d] Val with PSNR:%.4f  time: %.2f\n" % (
                epoch, i + 1, psnr, time.time() - start_t))

                if psnr > best_psnr:
                    best_psnr = psnr
                    torch.save({'epoch': epoch,
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict()
                                }, os.path.join(model_dir, "model_best.pth"))

                    # torch.save(model.state_dict(), os.path.join(save_fp, 'net_paras_1.pth')) # --> DnCNN_Pro
                    # torch.save(model.state_dict(), os.path.join(save_fp, 'net_paras_2.pth'))  # --> AtUNet
                    torch.save(model.state_dict(), os.path.join(save_fp, 'net_paras_derain.pth'))  # --> AtUNet_Pro_3

                model.train()
                torch.cuda.empty_cache()

    ########### Evaluation (per epoch) ############
    sum_of_psnr, sum_of_ssim = 0, 0
    valdata_len = val_dataset.__len__()
    with torch.no_grad():
        model.eval()
        index = 1
        for i, valdata in enumerate(val_loader, 0):
            ground_truth = valdata[0].cuda()
            noisy_val_img = valdata[1].cuda()
            filenames = valdata[2]

            # noise = torch.FloatTensor(valdata.size()).normal_(mean=0, std=noise_level / 255.)
            # noisy_val_img = valdata + noise
            # val_img = Variable(valdata.cuda())
            # noisy_val_img = Variable(noisy_val_img.cuda())
            with torch.cuda.amp.autocast():
                val_out = model(noisy_val_img)
            # denoise_val_img = torch.clamp(noisy_val_img - val_out, 0., 1.) #DnCNN
            # denoise_val_img = val_out

            denoised_val_img = torch.clamp(val_out, 0., 1.)
            sum_of_psnr += batch_PSNR(ground_truth, denoised_val_img, False).item()
            sum_of_ssim += batch_SSIM(ground_truth, denoised_val_img, True).item()
            index += 1

    avg_psnr = sum_of_psnr / (index*8)
    avg_ssim = sum_of_ssim / index
    epochLoss = epochLoss / index
    end_t = time.time()
    print("\n[Epoch %d] Finished with loss:%4f  avg_PSNR: %.4f  avg_SSIM: %.4f  time: %.2f\n   sum_psnr=%d, sum_ssim=%d" % (
    epoch, epochLoss, avg_psnr, avg_ssim, end_t - start_t, sum_of_psnr, sum_of_ssim))
    print('--------------------------------------------------------------------')

    ## Save the best state dict of model (update per epoch) ##
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_best.pth"))

        # torch.save(model.state_dict(), os.path.join(save_fp, 'net_paras_1.pth')) # --> DnCNN_Pro
        torch.save(model.state_dict(), os.path.join(save_fp, 'net_paras_derain.pth')) # --> AtUNet
        # torch.save(model.state_dict(), os.path.join(save_fp, 'net_paras_2.pth'))  # --> UANet

    with open(logname, 'a') as f:
        f.write("[Epoch %d] Finished with loss:%4f  avg_PSNR: %.4f  avg_SSIM: %.4f  time: %.2f\n" % (
        epochLoss, epoch + 1, avg_psnr, avg_ssim, end_t - start_t) + '\n')
    torch.save({'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))

    if epoch % opt.checkpoint == 0:
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_epoch_{}.pth".format(epoch)))

    model.train()
    torch.cuda.empty_cache()

    ## Visualization. add result images to Tensorboard ##
    denoise_img = torch.clamp(noisy_img - model(noisy_img), 0., 1.)
    gt = tv.utils.make_grid(ground_truth.data, nrow=8, normalize=True, scale_each=True)
    n_img = tv.utils.make_grid(noisy_img.data, nrow=8, normalize=True, scale_each=True)
    dn_img = tv.utils.make_grid(denoise_img.data, nrow=8, normalize=True, scale_each=True)
    writer.add_image('ground truth image', gt, epoch)
    writer.add_image('noisy image', n_img, epoch)
    writer.add_image('reconstructed image', dn_img, epoch)

    # record for future plots
    lr_list.append(scheduler.get_last_lr())
    loss_list.append(epochLoss)
    psnr_list.append(avg_psnr)
    ssim_list.append(avg_ssim)
    scheduler.step()

    ############# the end of one epoch ###############

# add training data plots to Tensorboard
for i, lr in enumerate(lr_list):
    writer.add_scalar('Train-lr', lr, i)
for i, loss in enumerate(loss_list):
    writer.add_scalar('Train-loss', loss, i)
for i, p in enumerate(psnr_list):
    writer.add_scalar('Train-psnr', p, i)
for i, s in enumerate(ssim_list):
    writer.add_scalar('Train-ssim', s, i)
#
# def main():
#
#
#
# if __name__ == '__main__':
#     # prepare_data(path='../dataset', patch_size=128, stride=10, aug_times=2)
#     main()
#
