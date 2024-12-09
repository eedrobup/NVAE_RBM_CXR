# In train.py, modify after argument parsing and before calling main():

import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.cuda.amp import autocast, GradScaler

from model import AutoEncoder
from thirdparty.adamax import Adamax
import utils as utils
import pickle
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as T
import re
from NVAE.fid.fid_score import compute_statistics_of_generator, load_statistics, calculate_frechet_distance
from NVAE.fid.inception import InceptionV3

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load RBM and vectorizers
from threelayerRBM import ThreeLayerRBM  # ensure this is accessible
# Suppose your RBM model and vectorizers are saved locally:
rbm = ThreeLayerRBM(n_visible=30, n_hidden_middle=30, n_hidden_top=24)  # fill in actual sizes
rbm.load_model('rbm_model.pkl')
with open('vectorizer_v.pkl', 'rb') as f:
    vectorizer_v = pickle.load(f)
with open('hidden_vectorizers.pkl', 'rb') as f:
    hidden_vectorizers = pickle.load(f)

findings_list = [
    "left", "right", "atelectasis", "bronchiectasis", "bulla", "consolidation", "dextrocardia", "effusion", "emphysema",
    "fracture clavicle", "fracture rib", "groundglass opacity", "interstitial opacification",
    "mass paraspinal", "mass soft tissue", "nodule", "opacity", "pneumomediastinum", "pneumonia",
    "pneumoperitoneum", "pneumothorax", "pleural effusion", "pulmonary edema", "scoliosis",
    "tuberculosis", "volume loss", "rib", "mass", "infiltration", "other findings"
]

hidden_features_dict = {
    'location': [
        "left lung", "right lung", "upper lobe", "lower lobe", "cardiac region",
        "pleural space", "diaphragm", "mediastinum", "thoracic spine", "abdominal region"
    ],
    'organ_system': [
        "respiratory system", "cardiovascular system", "musculoskeletal system", "digestive system"
    ],
    'mode_of_pathology': [
        "congenital", "acquired", "infection", "inflammation", "tumor", "degenerative", "vascular"
    ],
    'severity': [
        "mild", "moderate", "severe",
    ],
}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def get_features_from_text(clean_text, vectorizer_v, hidden_vectorizers):
    X_visible = vectorizer_v.transform([clean_text]).toarray()
    hidden_features = []
    for category, vec in hidden_vectorizers.items():
        X_hidden_cat = vec.transform([clean_text]).toarray()
        hidden_features.append(X_hidden_cat)
    X_hidden = np.concatenate(hidden_features, axis=1) if hidden_features else np.zeros((1,0))
    return X_visible, X_hidden

class ChestXRayDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, root_dir, rbm, vectorizer_v, hidden_vectorizers, transform=None):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.rbm = rbm
        self.vectorizer_v = vectorizer_v
        self.hidden_vectorizers = hidden_vectorizers
        self.transform = transform
        self.samples = []
        for _, row in self.df.iterrows():
            level1 = row['Level1']
            level2 = row['Level2']
            file_ = row['File']
            img_dir = os.path.join(self.root_dir, str(level1), str(level2), str(file_))

            findings = str(row['FINDINGS']) if not pd.isnull(row['FINDINGS']) else ''
            impression = str(row['IMPRESSION']) if not pd.isnull(row['IMPRESSION']) else ''
            text = findings + ' ' + impression
            self.samples.append((img_dir, text))

    def load_image(self, img_dir):
        imgs = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]
        if len(imgs) == 0:
            raise FileNotFoundError(f"No image found in {img_dir}")
        img_path = os.path.join(img_dir, imgs[0])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def text_to_cond_z(self, text):
        clean_text = preprocess_text(text)
        X_visible, X_hidden = get_features_from_text(clean_text, self.vectorizer_v, self.hidden_vectorizers)
        cond_z_np = self.rbm.transform(X_visible)  # np array
        cond_z = torch.tensor(cond_z_np, dtype=torch.float32)
        return cond_z[0]  # [n_hidden_top]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_dir, text = self.samples[idx]
        img = self.load_image(img_dir)
        cond_z = self.text_to_cond_z(text)
        return img, cond_z

def get_loaders_custom(args, rbm, vectorizer_v, hidden_vectorizers):
    # A replacement for datasets.get_loaders(args)
    # Create transforms
    transform = T.Compose([
        T.Resize((64,64)),
        #T.Grayscale(num_output_channels=1),  # Convert to single-channel grayscale
        T.ToTensor(),
    ])
    train_dataset = ChestXRayDataset(csv_path='outP10_500.csv', root_dir='./files-1024',
                                     rbm=rbm, vectorizer_v=vectorizer_v,
                                     hidden_vectorizers=hidden_vectorizers,
                                     transform=transform)
    # If you have a separate validation set, create a valid_dataset similarly.
    # For now, let's assume train_dataset is also used as valid_dataset for demonstration.
    valid_dataset = train_dataset

    train_queue = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    valid_queue = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    num_classes = 0  # or however you define

    return train_queue, valid_queue, num_classes

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logging = utils.Logger(args.global_rank, args.save)
    writer = utils.Writer(args.global_rank, args.save)

    # Instead of the original datasets.get_loaders(args), use our custom loader
    train_queue, valid_queue, num_classes = get_loaders_custom(args, rbm, vectorizer_v, hidden_vectorizers)

    args.num_total_iter = len(train_queue) * args.epochs
    warmup_iters = len(train_queue) * args.warmup_epochs
    swa_start = len(train_queue) * (args.epochs - 1)

    arch_instance = utils.get_arch_cells(args.arch_instance)

    cond_z_dim = 24 # Adjust based on RBM top layer dimension
    model = AutoEncoder(args, writer, arch_instance, cond_z_dim=cond_z_dim)
    model = model.to(DEVICE)

    logging.info('args = %s', args)
    logging.info('param size = %fM ', utils.count_parameters_in_M(model))
    logging.info('groups per scale: %s, total_groups: %d', model.groups_per_scale, sum(model.groups_per_scale))

    if args.fast_adamax:
        cnn_optimizer = Adamax(model.parameters(), args.learning_rate,
                               weight_decay=args.weight_decay, eps=1e-3)
    else:
        cnn_optimizer = torch.optim.Adamax(model.parameters(), args.learning_rate,
                                           weight_decay=args.weight_decay, eps=1e-3)

    cnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        cnn_optimizer, float(args.epochs - args.warmup_epochs - 1), eta_min=args.learning_rate_min)
    grad_scalar = GradScaler(2**10)

    num_output = utils.num_output(args.dataset)
    bpd_coeff = 1. / np.log(2.) / num_output

    checkpoint_file = os.path.join(args.save, 'checkpoint.pt')
    if args.cont_training:
        logging.info('loading the model.')
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        init_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(DEVICE)
        cnn_optimizer.load_state_dict(checkpoint['optimizer'])
        grad_scalar.load_state_dict(checkpoint['grad_scalar'])
        cnn_scheduler.load_state_dict(checkpoint['scheduler'])
        global_step = checkpoint['global_step']
    else:
        global_step, init_epoch = 0, 0

    for epoch in range(init_epoch, args.epochs):
        if args.distributed:
            train_queue.sampler.set_epoch(global_step + args.seed)
            valid_queue.sampler.set_epoch(0)

        if epoch > args.warmup_epochs:
            cnn_scheduler.step()

        logging.info('epoch %d', epoch)

        # Training
        train_nelbo, global_step = train(train_queue, model, cnn_optimizer, grad_scalar, global_step, warmup_iters, writer, logging, args)
        logging.info('train_nelbo %f', train_nelbo)
        writer.add_scalar('train/nelbo', train_nelbo, global_step)

        model.eval()
        save_freq = int(np.ceil(args.epochs / 100))
        if epoch % save_freq == 0 or epoch == (args.epochs - 1):
            if args.global_rank == 0:
                logging.info('saving the model.')
                torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                            'optimizer': cnn_optimizer.state_dict(), 'global_step': global_step,
                            'args': args, 'arch_instance': arch_instance, 'scheduler': cnn_scheduler.state_dict(),
                            'grad_scalar': grad_scalar.state_dict()}, checkpoint_file)
        print("Saved model")
        # generate samples less frequently
        eval_freq = 1 if args.epochs <= 50 else 20
        if epoch % eval_freq == 0 or epoch == (args.epochs - 1):
            with torch.no_grad():
                num_samples = 16
                n = int(np.floor(np.sqrt(num_samples)))
                for t in [0.7, 0.8, 0.9, 1.0]:
                    logits = model.sample(num_samples, t)
                    output = model.decoder_output(logits)
                    output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) else output.sample(t)
                    output_tiled = utils.tile_image(output_img, n)
                    writer.add_image('generated_%0.1f' % t, output_tiled, global_step)
            valid_neg_log_p, valid_nelbo = test(valid_queue, model, num_samples=10, args=args, logging=logging)
            logging.info('valid_nelbo %f', valid_nelbo)
            logging.info('valid neg log p %f', valid_neg_log_p)
            logging.info('valid bpd elbo %f', valid_nelbo * bpd_coeff)
            logging.info('valid bpd log p %f', valid_neg_log_p * bpd_coeff)
            writer.add_scalar('val/neg_log_p', valid_neg_log_p, epoch)
            writer.add_scalar('val/nelbo', valid_nelbo, epoch)
            writer.add_scalar('val/bpd_log_p', valid_neg_log_p * bpd_coeff, epoch)
            writer.add_scalar('val/bpd_elbo', valid_nelbo * bpd_coeff, epoch)


    # Final validation
    valid_neg_log_p, valid_nelbo = test(valid_queue, model, num_samples=1000, args=args, logging=logging)
    logging.info('final valid nelbo %f', valid_nelbo)
    logging.info('final valid neg log p %f', valid_neg_log_p)
    writer.add_scalar('val/neg_log_p', valid_neg_log_p, epoch + 1)
    writer.add_scalar('val/nelbo', valid_nelbo, epoch + 1)
    writer.add_scalar('val/bpd_log_p', valid_neg_log_p * bpd_coeff, epoch + 1)
    writer.add_scalar('val/bpd_elbo', valid_nelbo * bpd_coeff, epoch + 1)
    writer.close()

def get_custom_posterior(rbm_stack, findings):
    with torch.no_grad():
        loc, sys, path = rbm_stack.sample_top(findings)
        # cond_z is a concatenation of loc, sys, path features
        cond_z = torch.cat([loc, sys, path], dim=1).float()  # shape [B, cond_z_dim]
    return cond_z

# The rest of the file (train, test functions) remain the same, except for how you handle inputs:
# In train function, now you get (imgs, cond_z) directly from dataloader:
def train(train_queue, model, cnn_optimizer, grad_scalar, global_step, warmup_iters, writer, logging, args):
    alpha_i = utils.kl_balancer_coeff(num_scales=model.num_latent_scales,
                                      groups_per_scale=model.groups_per_scale, fun='square')
    nelbo = utils.AvgrageMeter()
    model.train()
    for step, (imgs, cond_z) in enumerate(train_queue):
        imgs = imgs.to(DEVICE)
        cond_z = cond_z.to(DEVICE)

        # No findings extraction needed here anymore since cond_z is already provided by dataset
        # Just pre-process imgs if needed:
        imgs = utils.pre_process(imgs, args.num_x_bits)

        if global_step < warmup_iters:
            lr = args.learning_rate * float(global_step) / warmup_iters
            for param_group in cnn_optimizer.param_groups:
                param_group['lr'] = lr

        if step % 100 == 0:
            utils.average_params(model.parameters(), args.distributed)

        cnn_optimizer.zero_grad()
        with autocast():
            logits, log_q, log_p, kl_all, kl_diag = model(imgs, cond_z=cond_z)

            output = model.decoder_output(logits)
            kl_coeff = utils.kl_coeff(global_step, args.kl_anneal_portion * args.num_total_iter,
                                      args.kl_const_portion * args.num_total_iter, args.kl_const_coeff)

            recon_loss = utils.reconstruction_loss(output, imgs, crop=model.crop_output)
            balanced_kl, kl_coeffs, kl_vals = utils.kl_balancer(kl_all, kl_coeff, kl_balance=True, alpha_i=alpha_i)

            nelbo_batch = recon_loss + balanced_kl
            loss = torch.mean(nelbo_batch)
            norm_loss = model.spectral_norm_parallel()
            bn_loss = model.batchnorm_loss()

            if args.weight_decay_norm_anneal:
                assert args.weight_decay_norm_init > 0 and args.weight_decay_norm > 0, 'init and final wdn should be positive.'
                wdn_coeff = (1. - kl_coeff)*np.log(args.weight_decay_norm_init) + kl_coeff*np.log(args.weight_decay_norm)
                wdn_coeff = np.exp(wdn_coeff)
            else:
                wdn_coeff = args.weight_decay_norm

            loss += norm_loss * wdn_coeff + bn_loss * wdn_coeff

        grad_scalar.scale(loss).backward()
        utils.average_gradients(model.parameters(), args.distributed)
        grad_scalar.step(cnn_optimizer)
        grad_scalar.update()
        nelbo.update(loss.data, 1)

        if (global_step + 1) % 100 == 0:
            if (global_step + 1) % 1000 == 0:  # reduced frequency
                n = int(np.floor(np.sqrt(imgs.size(0))))
                x_img = imgs[:n*n]
                output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) else output.sample()
                output_img = output_img[:n*n]
                x_tiled = utils.tile_image(x_img, n)
                output_tiled = utils.tile_image(output_img, n)
                in_out_tiled = torch.cat((x_tiled, output_tiled), dim=2)
                writer.add_image('reconstruction', in_out_tiled, global_step)

            # norm
            writer.add_scalar('train/norm_loss', norm_loss, global_step)
            writer.add_scalar('train/bn_loss', bn_loss, global_step)
            writer.add_scalar('train/norm_coeff', wdn_coeff, global_step)

            utils.average_tensor(nelbo.avg, args.distributed)
            logging.info('train %d %f', global_step, nelbo.avg)
            writer.add_scalar('train/nelbo_avg', nelbo.avg, global_step)
            writer.add_scalar('train/lr', cnn_optimizer.state_dict()['param_groups'][0]['lr'], global_step)
            writer.add_scalar('train/nelbo_iter', loss, global_step)
            writer.add_scalar('train/kl_iter', torch.mean(sum(kl_all)), global_step)
            writer.add_scalar('train/recon_iter', torch.mean(utils.reconstruction_loss(output, imgs, crop=model.crop_output)), global_step)
            writer.add_scalar('kl_coeff/coeff', kl_coeff, global_step)
            total_active = 0
            for i, kl_diag_i in enumerate(kl_diag):
                utils.average_tensor(kl_diag_i, args.distributed)
                num_active = torch.sum(kl_diag_i > 0.1).detach()
                total_active += num_active

                # kl_ceoff
                writer.add_scalar('kl/active_%d' % i, num_active, global_step)
                writer.add_scalar('kl_coeff/layer_%d' % i, kl_coeffs[i], global_step)
                writer.add_scalar('kl_vals/layer_%d' % i, kl_vals[i], global_step)
            writer.add_scalar('kl/total_active', total_active, global_step)


        global_step += 1

    utils.average_tensor(nelbo.avg, args.distributed)
    return nelbo.avg, global_step


def test(valid_queue, model, num_samples, args, logging):
    if args.distributed:
        dist.barrier()
    nelbo_avg = utils.AvgrageMeter()
    neg_log_p_avg = utils.AvgrageMeter()
    model.eval()
    for step, x in enumerate(valid_queue):
        x = x[0] if len(x) > 1 else x
        x = x.to(DEVICE)

        # change bit length
        x = utils.pre_process(x, args.num_x_bits)

        with torch.no_grad():
            nelbo, log_iw = [], []
            for k in range(num_samples):
                logits, log_q, log_p, kl_all, _ = model(x)
                output = model.decoder_output(logits)
                recon_loss = utils.reconstruction_loss(output, x, crop=model.crop_output)
                balanced_kl, _, _ = utils.kl_balancer(kl_all, kl_balance=False)
                nelbo_batch = recon_loss + balanced_kl
                nelbo.append(nelbo_batch)
                log_iw.append(utils.log_iw(output, x, log_q, log_p, crop=model.crop_output))

            nelbo = torch.mean(torch.stack(nelbo, dim=1))
            log_p = torch.mean(torch.logsumexp(torch.stack(log_iw, dim=1), dim=1) - np.log(num_samples))

        nelbo_avg.update(nelbo.data, x.size(0))
        neg_log_p_avg.update(- log_p.data, x.size(0))

    utils.average_tensor(nelbo_avg.avg, args.distributed)
    utils.average_tensor(neg_log_p_avg.avg, args.distributed)
    if args.distributed:
        # block to sync
        dist.barrier()
    logging.info('val, step: %d, NELBO: %f, neg Log p %f', step, nelbo_avg.avg, neg_log_p_avg.avg)
    return neg_log_p_avg.avg, nelbo_avg.avg


def create_generator_vae(model, batch_size, num_total_samples):
    num_iters = int(np.ceil(num_total_samples / batch_size))
    for i in range(num_iters):
        with torch.no_grad():
            logits = model.sample(batch_size, 1.0)
            output = model.decoder_output(logits)
            output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) else output.mean()
        yield output_img.float()


def test_vae_fid(model, args, total_fid_samples):
    dims = 2048
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_gpus = args.num_process_per_node * args.num_proc_node
    num_sample_per_gpu = int(np.ceil(total_fid_samples / num_gpus))

    g = create_generator_vae(model, args.batch_size, num_sample_per_gpu)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], model_dir=args.fid_dir).to(device)
    m, s = compute_statistics_of_generator(g, model, args.batch_size, dims, device, max_samples=num_sample_per_gpu)

    # share m and s
    m = torch.from_numpy(m).to(DEVICE)
    s = torch.from_numpy(s).to(DEVICE)
    # take average across gpus
    utils.average_tensor(m, args.distributed)
    utils.average_tensor(s, args.distributed)

    # convert m, s
    m = m.cpu().numpy()
    s = s.cpu().numpy()

    # load precomputed m, s
    path = os.path.join(args.fid_dir, args.dataset + '.npz')
    m0, s0 = load_statistics(path)

    fid = calculate_frechet_distance(m0, s0, m, s)
    return fid


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '6020'
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
    #dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    # If you're using distributed for some reason (even on CPU):
    dist.init_process_group(backend='gloo', init_method='env://', rank=rank, world_size=size)
    fn(args)
    cleanup()


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('encoder decoder examiner')
    # experimental results
    parser.add_argument('--root', type=str, default='/tmp/nasvae/expr',
                        help='location of the results')
    parser.add_argument('--save', type=str, default='exp',
                        help='id used for storing intermediate results')
    # data
    parser.add_argument('--dataset', type=str, default='imagenet_64',
                        choices=['cifar10', 'mnist', 'omniglot', 'celeba_64', 'celeba_256',
                                 'imagenet_32', 'ffhq', 'lsun_bedroom_128', 'stacked_mnist',
                                 'lsun_church_128', 'lsun_church_64'],
                        help='which dataset to use')
    parser.add_argument('--data', type=str, default='/tmp/nasvae/data',
                        help='location of the data corpus')
    # optimization
    parser.add_argument('--batch_size', type=int, default=200,
                        help='batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=1e-4,
                        help='min learning rate')
    parser.add_argument('--weight_decay', type=float, default=3e-4,
                        help='weight decay')
    parser.add_argument('--weight_decay_norm', type=float, default=0.,
                        help='The lambda parameter for spectral regularization.')
    parser.add_argument('--weight_decay_norm_init', type=float, default=10.,
                        help='The initial lambda parameter')
    parser.add_argument('--weight_decay_norm_anneal', action='store_true', default=False,
                        help='This flag enables annealing the lambda coefficient from '
                             '--weight_decay_norm_init to --weight_decay_norm.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='num of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='num of training epochs in which lr is warmed up')
    parser.add_argument('--fast_adamax', action='store_true', default=False,
                        help='This flag enables using our optimized adamax.')
    parser.add_argument('--arch_instance', type=str, default='res_mbconv',
                        help='path to the architecture instance')
    # KL annealing
    parser.add_argument('--kl_anneal_portion', type=float, default=0.3,
                        help='The portions epochs that KL is annealed')
    parser.add_argument('--kl_const_portion', type=float, default=0.0001,
                        help='The portions epochs that KL is constant at kl_const_coeff')
    parser.add_argument('--kl_const_coeff', type=float, default=0.0001,
                        help='The constant value used for min KL coeff')
    # Flow params
    parser.add_argument('--num_nf', type=int, default=0,
                        help='The number of normalizing flow cells per groups. Set this to zero to disable flows.')
    parser.add_argument('--num_x_bits', type=int, default=8,
                        help='The number of bits used for representing data for colored images.')
    # latent variables
    parser.add_argument('--num_latent_scales', type=int, default=1,
                        help='the number of latent scales')
    parser.add_argument('--num_groups_per_scale', type=int, default=10,
                        help='number of groups of latent variables per scale')
    parser.add_argument('--num_latent_per_group', type=int, default=20,
                        help='number of channels in latent variables per group')
    parser.add_argument('--ada_groups', action='store_true', default=False,
                        help='Settings this to true will set different number of groups per scale.')
    parser.add_argument('--min_groups_per_scale', type=int, default=1,
                        help='the minimum number of groups per scale.')
    # encoder parameters
    parser.add_argument('--num_channels_enc', type=int, default=32,
                        help='number of channels in encoder')
    parser.add_argument('--num_preprocess_blocks', type=int, default=2,
                        help='number of preprocessing blocks')
    parser.add_argument('--num_preprocess_cells', type=int, default=3,
                        help='number of cells per block')
    parser.add_argument('--num_cell_per_cond_enc', type=int, default=1,
                        help='number of cell for each conditional in encoder')
    # decoder parameters
    parser.add_argument('--num_channels_dec', type=int, default=32,
                        help='number of channels in decoder')
    parser.add_argument('--num_postprocess_blocks', type=int, default=2,
                        help='number of postprocessing blocks')
    parser.add_argument('--num_postprocess_cells', type=int, default=3,
                        help='number of cells per block')
    parser.add_argument('--num_cell_per_cond_dec', type=int, default=1,
                        help='number of cell for each conditional in decoder')
    parser.add_argument('--num_mixture_dec', type=int, default=10,
                        help='number of mixture components in decoder. set to 1 for Normal decoder.')
    # NAS
    parser.add_argument('--use_se', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--res_dist', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--cont_training', action='store_true', default=False,
                        help='This flag enables training from an existing checkpoint.')
    # DDP.
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--global_rank', type=int, default=0,
                        help='rank of process among all the processes')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed used for initialization')
    args = parser.parse_args()
    args.save = args.root + '/eval-' + args.save
    utils.create_exp_dir(args.save)

    size = args.num_process_per_node

    if size > 1:
        #args.distributed = True
        args.distributed = False
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, main, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        # for debugging
        print('starting in debug mode')
        args.distributed = False
        #init_processes(0, size, main, args)
        main(args)



