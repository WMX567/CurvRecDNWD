import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import argparse
import utils.log
from PointSegDA.data.dataloader import datareader_curv
from PointSegDA.models_seg import DGCNN_DefRec
from utils import pc_utils
from sklearn.metrics import jaccard_score
from CurvDefRec import DefRec
from utils.nwd import NWD_Seg


NWORKERS=4
MAX_LOSS = 9 * (10**9)

def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
# ==================
# Argparse
# ==================
parser = argparse.ArgumentParser(description='DA on Point Clouds')
parser.add_argument('--exp_name', type=str, default='DefRec_PCM',  help='Name of the experiment')
parser.add_argument('--dataroot', type=str, default='./data/PointSegDAdataset', help='data path')
parser.add_argument('--out_path', type=str, default='./experiments', help='log folder path')
parser.add_argument('--src_dataset', type=str, default='adobe', choices=['adobe', 'faust', 'mit', 'scape'])
parser.add_argument('--trgt_dataset', type=str, default='faust', choices=['adobe', 'faust', 'mit', 'scape'])
parser.add_argument('--epochs', type=int, default=200, help='number of episode to train')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size', help='Size of train batch per domain')
parser.add_argument('--test_batch_size', type=int, default=2, metavar='batch_size', help='Size of test batch per domain')
parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD'])
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--DefRec_dist', type=str, default='volume_based_radius', metavar='N',
                    choices=['volume_based_voxels', 'volume_based_radius'],
                    help='distortion of points')
parser.add_argument('--largest', type=str2bool, default=False, help='Use largest curvature')
parser.add_argument('--num_regions', type=int, default=3, help='number of regions to split shape by')
parser.add_argument('--noise_std', type=float, default=0.1, help='learning rate')
parser.add_argument('--DefRec_weight', type=float, default=0.05, help='weight of the DefRec loss')
parser.add_argument('--mixup_params', type=float, default=1.0, help='a,b in beta distribution')
parser.add_argument('--nwd_weight', type=float, default=1.0, help='skip avg_pool')
parser.add_argument('--nwd', type=str2bool, default=True, help='use nwd')
parser.add_argument('--model_type', type = str, default=None, help = 'model name')
parser.add_argument('--target_weight', type = float, default=0.2, help = 'weight of the target loss')
args = parser.parse_args()

# ==================
# init
# ==================
io = utils.log.IOStream(args)
io.cprint(str(args))

random.seed(1)
np.random.seed(1)  # to get the same point choice in ModelNet and ScanNet leave it fixed
torch.manual_seed(args.seed)
args.cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    io.cprint('Using GPUs')
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    io.cprint('Using CPU')

# ==================
# Read Data
# ==================
src_trainset = datareader_curv(args.dataroot, dataset=args.src_dataset, partition='train', domain='source')
src_valset = datareader_curv(args.dataroot, dataset=args.src_dataset, partition='val', domain='source')
trgt_trainset = datareader_curv(args.dataroot, dataset=args.trgt_dataset, partition='train', domain='target')
trgt_valset = datareader_curv(args.dataroot, dataset=args.trgt_dataset, partition='val', domain='target')
trgt_testset = datareader_curv(args.dataroot, dataset=args.trgt_dataset, partition='test', domain='target')

# dataloaders for source and target
batch_size = min(len(src_trainset), len(trgt_trainset), args.batch_size)
src_train_loader = DataLoader(src_trainset, num_workers=NWORKERS, batch_size=batch_size,
                               shuffle=True, drop_last=True)
src_val_loader = DataLoader(src_valset, num_workers=NWORKERS, batch_size=args.test_batch_size)
trgt_train_loader = DataLoader(trgt_trainset, num_workers=NWORKERS, batch_size=batch_size,
                               shuffle=True, drop_last=True)
trgt_val_loader = DataLoader(trgt_valset, num_workers=NWORKERS, batch_size=args.test_batch_size)
trgt_test_loader = DataLoader(trgt_testset, num_workers=NWORKERS, batch_size=args.test_batch_size)

# ==================
# Init Model
# ==================
num_classes = 8
model = DGCNN_DefRec(args, in_size=3, num_classes=num_classes)
model = model.to(device)
nwd = NWD_Seg(model.seg)

# ==================
# Optimizer
# ==================
opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd) \
    if args.optimizer == "SGD" \
    else optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
t_max = args.epochs
scheduler = CosineAnnealingLR(opt, T_max=t_max, eta_min=0.0)

# ==================
# Loss and Metrics
# ==================
criterion = nn.CrossEntropyLoss()  # return the mean of CE over the batch
sample_criterion = nn.CrossEntropyLoss(reduction='none')  # to get the loss per shape
def seg_metrics(labels, preds):
    batch_size = labels.shape[0]
    mIOU = accuracy = 0
    for b in range(batch_size):
        y_true = labels[b, :].detach().cpu().numpy()
        y_pred = preds[b, :].detach().cpu().numpy()
        # IOU per class and average
        mIOU += jaccard_score(y_true, y_pred, average='macro')
        accuracy += np.mean(y_true == y_pred)
    return mIOU, accuracy

# ==================
# Validation/test
# ==================
def test(test_loader):
    # Run on cpu or gpu
    seg_loss = mIOU = accuracy = 0.0
    batch_idx = num_samples = 0
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(test_loader):
            data, labels = data[0].to(device), data[1].to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.shape[0]
            logits = model(data, make_seg=True, activate_DefRec=False)
            loss = criterion(logits["seg"].permute(0, 2, 1), labels)
            seg_loss += loss.item() * batch_size
            
            # evaluation metrics
            preds = logits["seg"].max(dim=2)[1]
            batch_mIOU, batch_seg_acc = seg_metrics(labels, preds)
            mIOU += batch_mIOU
            accuracy += batch_seg_acc
            num_samples += batch_size
            batch_idx += 1
    seg_loss /= num_samples
    mIOU /= num_samples
    accuracy /= num_samples
    model.train()
    return seg_loss, mIOU, accuracy

# ==================
# Train
# ==================
src_best_val_acc = trgt_best_val_acc = best_val_epoch = 0
src_best_val_mIOU = trgt_best_val_mIOU = 0.0
src_best_val_loss = trgt_best_val_loss = MAX_LOSS
epoch = step = 0
lookup = torch.Tensor(pc_utils.region_mean(args.num_regions)).to(device)

for epoch in range(args.epochs):
    model.train()
    
    # init data structures for saving epoch stats
    src_seg_loss = src_mIOU = src_accuracy = 0.0
    trgt_rec_loss = total_loss = 0.0
    batch_idx = src_count = trgt_count = 0
    
    # init data structures for saving epoch stats
    src_print_losses = {'mIOU': 0.0,'DefRec': 0.0, 'seg': 0.0, 'acc': 0.0}
    trgt_print_losses = {'DefRec': 0.0,'NWD':0.0}
    src_count = trgt_count = 0.0
    for k, data in enumerate(zip(src_train_loader, trgt_train_loader)):
        step += 1
        opt.zero_grad()
        trgt_batch_loss = src_batch_loss = batch_mIOU = batch_seg_acc = 0.0
        
        #### source data ####
        if data[0] is not None:
            src_data, src_labels, src_curv = data[0][0].to(device), data[0][1].to(device), data[0][2].to(device)
            src_data_orig = src_data.clone()
            batch_size = src_data.shape[0]
            num_points = src_data.shape[1]
            num_groups = 40
            top_k = int(num_groups * 0.25)
            src_data_d, src_mask = DefRec.deform_input_curv(src_data, src_curv, args.DefRec_dist, 
                                                           top_k=top_k, num_group=num_groups, 
                                                           group_size=55, largest=args.largest)
            src_cls_logits_deform = model(src_data_d, activate_DefRec=True)
            loss = DefRec.calc_loss(args, src_cls_logits_deform, src_data_orig.permute(0, 2, 1), src_mask)
            src_print_losses['DefRec'] += loss.item() * batch_size
            loss.backward()
            src_data = src_data_orig.clone()
            src_data = src_data.permute(0, 2, 1)
            logits = model(src_data, make_seg=True, activate_DefRec=False)
            loss = criterion(logits['seg'].permute(0, 2, 1), src_labels)
            src_data_deform = src_data_d.clone()
            logits_deform = model(src_data_deform, make_seg=True, activate_DefRec=False)
            loss += criterion(logits_deform['seg'].permute(0, 2, 1), src_labels)
            loss /= 2
            src_print_losses['seg'] += loss.item() * batch_size
            loss.backward()
            
            # evaluation metrics
            preds = logits['seg'].max(dim=2)[1]
            batch_mIOU, batch_seg_acc = seg_metrics(src_labels, preds)
            src_print_losses['mIOU'] += batch_mIOU
            src_print_losses['acc'] += batch_seg_acc
            src_count += batch_size
            trgt_count += batch_size

        #### target data ####
        if data[1] is not None:
            trgt_data, trgt_labels, trgt_curv = data[1][0].to(device), data[1][1].to(device), data[1][2].to(device)
            batch_size = trgt_data.shape[0]
            trgt_data_orig = trgt_data.clone()
            num_points = trgt_data.shape[1]
            num_groups = 40
            top_k = int(num_groups * 0.25)
            trgt_data, trgt_mask = DefRec.deform_input_curv(trgt_data, trgt_curv, args.DefRec_dist, 
                                                            top_k=top_k, num_group=num_groups, 
                                                            group_size=55, largest=args.largest)
            trgt_data_deform = trgt_data.clone()
            trgt_logits = model(trgt_data, activate_DefRec=True)
            loss = DefRec.calc_loss(args, trgt_logits, trgt_data_orig.permute(0, 2, 1), trgt_mask)
            trgt_print_losses['DefRec'] += loss.item() * batch_size
            loss.backward()
            trgt_count += batch_size

        if args.nwd and data[0] is not None and data[1] is not None:
            _, latent_point_src = model(src_data_orig.permute(0,2,1), make_seg=True, nwd=True)
            logits, latent_point_trgt = model(trgt_data_orig.permute(0,2,1), make_seg=True, nwd=True)
            _, latent_point_src_deform = model(src_data_deform, make_seg=True, nwd=True)
            _, latent_point_trgt_deform = model(trgt_data_deform, make_seg=True, nwd=True)
            
            index_t = torch.arange(latent_point_trgt.size(0)).unsqueeze(1).to(device)
            latent_point_trgt = latent_point_trgt[index_t[:, 0]]
            latent_point_trgt_deform = latent_point_trgt_deform[index_t[:, 0]]
            
            if index_t.size(0) >= 2:
                loss_nwd = -args.nwd_weight * nwd(latent_point_src_deform, latent_point_trgt_deform, latent_point_src, latent_point_trgt, args.target_weight)
                trgt_print_losses['NWD'] += loss_nwd.item() * batch_size
                loss_nwd.backward()

        batch_idx += 1
        opt.step()
    scheduler.step()

    # print progress
    src_print_losses = {k: v * 1.0 / src_count for (k, v) in src_print_losses.items()}
    src_acc = io.print_progress("Source", "Trn", epoch, src_print_losses)
    trgt_print_losses = {k: v * 1.0 / trgt_count for (k, v) in trgt_print_losses.items()}
    trgt_acc = io.print_progress("Target", "Trn", epoch, trgt_print_losses)
   
    #===================
    # Validation
    #===================
    src_val_loss, src_val_miou, src_val_acc = test(src_val_loader)
    trgt_val_loss, trgt_val_miou, trgt_val_acc = test(trgt_val_loader)
    
    # save model according to best source model (since we don't have target labels)
    if src_val_loss < src_best_val_loss:
        src_best_val_mIOU = src_val_miou
        src_best_val_acc = src_val_acc
        src_best_val_loss = src_val_loss
        trgt_best_val_mIOU = trgt_val_miou
        trgt_best_val_acc = trgt_val_acc
        trgt_best_val_loss = trgt_val_loss
        best_val_epoch = epoch
        io.save_model(model)

io.cprint("Best model was found at epoch %d\n"
          "source val seg loss: %.4f, source val seg mIOU: %.4f, source val seg accuracy: %.4f\n"
          "target val seg loss: %.4f, target val seg mIOU: %.4f, target val seg accuracy: %.4f\n"
         % (best_val_epoch,
            src_best_val_loss, src_best_val_mIOU, src_best_val_acc,
            trgt_best_val_loss, trgt_best_val_mIOU, trgt_best_val_acc))

#===================
# Test
#===================
path = args.out_path + '/{}_{}_{}'.format(args.src_dataset, 
                                              args.trgt_dataset, 
                                              args.seed) + args.model_type +'.pt'
model.load_state_dict(torch.load(path))
trgt_test_loss, trgt_test_miou, trgt_test_acc = test(trgt_test_loader)
io.cprint("target test seg loss: %.4f, target test seg mIOU: %.4f, target test seg accuracy: %.4f"
          % (trgt_test_loss, trgt_test_miou, trgt_test_acc))