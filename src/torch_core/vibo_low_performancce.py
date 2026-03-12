import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import time
import math
import numpy as np
from tqdm import tqdm

import torch
from torch import optim
import torch.distributions as dist
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score
import sys
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if base_dir not in sys.path:
    sys.path.append(base_dir)
from src.torch_core.models import (
    VIBO_1PL, 
    VIBO_2PL, 
    VIBO_3PL,
)
from src.datasets import load_dataset, artificially_mask_dataset, load_dataset_tvt
from src.utils import AverageMeter, save_checkpoint
from src.config import OUT_DIR, IS_REAL_WORLD


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--generative-model', type=str, default='irt', 
                        choices=['irt', 'link', 'deep', 'residual', 'neural', 'dina'], 
                        help='irt|link|deep|residual|neural (default: irt)')
    parser.add_argument('--irt-model', type=str, default='1pl', 
                        choices=['1pl', '2pl', '3pl'], 
                        help='1pl|2pl|3pl (default: 1pl)')
    parser.add_argument('--dataset', type=str, default='math1', 
                        help='which dataset to run on')
    parser.add_argument('--ability-dim', type=int, default=1, 
                        help='number of ability dimensions (default: 1)')
    parser.add_argument('--ability-merge', type=str, default='product', 
                        choices=['mean, product'], 
                        help='mean|product|transformer (default: product)')
    parser.add_argument('--lr', type=float, default=5e-3, 
                        help='default learning rate: 5e-3')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N', 
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', 
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--gpu-device', type=int, default=0, 
                        help='which CUDA device to use (default: 0)')
    parser.add_argument('--cuda', action='store_true', default=False, 
                        help='enables CUDA training (default: False)')
    args = parser.parse_args()

    run_start_ts = time.time()
    run_start_str = time.strftime("%Y%m%d-%H%M%S", time.localtime(run_start_ts))

    args.response_dist = 'bernoulli'
    args.drop_missing = False
    args.artificial_missing_perc = 0.0
    args.n_norm_flows = 0
    args.no_infer_dict = False
    args.no_marginal = False
    args.no_test = False
    args.no_predictive = False
    args.num_person = 1000
    args.num_item = 100
    args.num_posterior_samples = 400
    args.hidden_dim = 64
    args.max_num_person = None
    args.max_num_item = None
    args.out_dir = OUT_DIR
    args.max_iters = -1
    args.num_workers = 0
    args.anneal_kl = False
    args.beta_kl = 1.0
    args.seed = 42
    args.conditional_posterior = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if IS_REAL_WORLD[args.dataset]:
        # these params are only for IRT simulation datasets
        args.num_person = None
        args.num_item = None

        if args.max_num_person is not None:
            args.max_num_person = int(args.max_num_person)
        
        if args.max_num_item is not None:
            args.max_num_item = int(args.max_num_item)
        
    else:
        args.max_num_person = None
        args.max_num_item = None

    out_file = 'VIBO_{}_{}_{}_{}_{}person_{}item_{}maxperson_{}maxitem_{}maskperc_{}ability_{}_{}_seed{}'.format(
        args.irt_model, 
        args.dataset,
        args.response_dist,
        args.generative_model,
        args.num_person, 
        args.num_item,
        args.max_num_person,
        args.max_num_item,
        args.artificial_missing_perc,
        args.ability_dim, 
        args.ability_merge,
        'conditional_q' if args.conditional_posterior else 'unconditional_q',
        args.seed,
    )
    args.out_dir = os.path.join(args.out_dir, out_file) 
    
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    result_root = os.path.join(os.path.dirname(OUT_DIR), 'result')
    if not os.path.isdir(result_root):
        os.makedirs(result_root)

    device = torch.device("cuda" if args.cuda else "cpu")
    if args.cuda: torch.cuda.set_device(args.gpu_device)

    if args.response_dist == 'bernoulli':
        dataset_name = args.dataset
    else:
        dataset_name = f'{args.dataset}_continuous'

    train_dataset, valid_npy, test_npy = load_dataset_tvt(
        dataset_name, 
        train = True, 
        num_person = args.num_person, 
        num_item = args.num_item,  
        ability_dim = args.ability_dim,
        max_num_person = args.max_num_person,
        max_num_item = args.max_num_item,
    )

    # test_dataset  = load_dataset(
    #     dataset_name, 
    #     train = False, 
    #     num_person = args.num_person, 
    #     num_item = args.num_item, 
    #     ability_dim = args.ability_dim,
    #     max_num_person = args.max_num_person,
    #     max_num_item = args.max_num_item,
    # )

    num_person = train_dataset.num_person
    num_item   = train_dataset.num_item

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size = args.batch_size, 
        shuffle = True,
        num_workers = args.num_workers,
    )

    N_mini_batches = len(train_loader)
    if args.max_iters != -1:
        args.epochs = int(math.ceil(args.max_iters / float(len(train_loader))))
        print(f'Found MAX_ITERS={args.max_iters}, setting EPOCHS={args.epochs}')

    if args.irt_model == '1pl':
        model_class = VIBO_1PL
    elif args.irt_model == '2pl':
        model_class = VIBO_2PL
    elif args.irt_model == '3pl':
        model_class = VIBO_3PL
    else:
        raise Exception(f'model {args.irt_model} not recognized')

    model = model_class(
        args.ability_dim,
        num_item,
        hidden_dim = args.hidden_dim,
        ability_merge = args.ability_merge,
        conditional_posterior = args.conditional_posterior,
        generative_model = args.generative_model,
        response_dist = args.response_dist,
        replace_missing_with_prior = not args.drop_missing,
        n_norm_flows = args.n_norm_flows,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    def get_annealing_factor(epoch, which_mini_batch):
        if args.anneal_kl:
            annealing_factor = \
                (float(which_mini_batch + epoch * N_mini_batches + 1) /
                 float(args.epochs // 2 * N_mini_batches))
        else:
            annealing_factor = args.beta_kl 
        return annealing_factor

    def train(epoch):
        model.train()
        train_loss = AverageMeter()

        for batch_idx, (_, response, _, mask) in enumerate(train_loader):
            mb = response.size(0)
            response = response.to(device)
            mask = mask.long().to(device)
            annealing_factor = get_annealing_factor(epoch, batch_idx)
        
            optimizer.zero_grad()
            if args.n_norm_flows > 0:
                (
                    response, mask, response_mu, 
                    ability_k, ability, 
                    ability_mu, ability_logvar, ability_logabsdetjac, 
                    item_feat_k, item_feat, 
                    item_feat_mu, item_feat_logvar, item_feat_logabsdetjac,
                ) = model(response, mask)
                loss = model.elbo(
                    response, mask, response_mu, 
                    ability, ability_mu, ability_logvar,
                    item_feat, item_feat_mu, item_feat_logvar, 
                    annealing_factor = annealing_factor,
                    use_kl_divergence = False,
                    ability_k = ability_k,
                    item_feat_k = item_feat_k,
                    ability_logabsdetjac = ability_logabsdetjac,
                    item_logabsdetjac = item_feat_logabsdetjac,
                )
            else:
                outputs = model(response, mask)
                loss = model.elbo(*outputs, annealing_factor=annealing_factor,
                                use_kl_divergence=True)
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), mb)

        print('====> Train Epoch: {} Loss: {:.4f}'.format(epoch, train_loss.avg))

        return train_loss.avg

    # def test(epoch):
    #     model.eval()
    #     test_loss = AverageMeter()
    #     pbar = tqdm(total=len(test_loader))

    #     with torch.no_grad():
    #         for _, response, _, mask in test_loader:
    #             mb = response.size(0)
    #             response = response.to(device)
    #             mask = mask.long().to(device)

    #             if args.n_norm_flows > 0:
    #                 (
    #                     response, mask, response_mu, 
    #                     ability_k, ability, 
    #                     ability_mu, ability_logvar, ability_logabsdetjac, 
    #                     item_feat_k, item_feat, 
    #                     item_feat_mu, item_feat_logvar, item_feat_logabsdetjac,
    #                 ) = model(response, mask)
    #                 loss = model.elbo(
    #                     response, mask, response_mu, 
    #                     ability, ability_mu, ability_logvar,
    #                     item_feat, item_feat_mu, item_feat_logvar, 
    #                     use_kl_divergence = False,
    #                     ability_k = ability_k,
    #                     item_feat_k = item_feat_k,
    #                     ability_logabsdetjac = ability_logabsdetjac,
    #                     item_logabsdetjac = item_feat_logabsdetjac,
    #                 )
    #             else:
    #                 outputs = model(response, mask)
    #                 loss = model.elbo(*outputs)
    #             test_loss.update(loss.item(), mb)

    #             pbar.update()
    #             pbar.set_postfix({'Loss': test_loss.avg})

    #     pbar.close()
    #     print('====> Test Epoch: {} Loss: {:.4f}'.format(epoch, test_loss.avg))

    #     return test_loss.avg

    def test_acc(epoch, eval_np, name='Test', infer_dict=None):
        model.eval()
        if infer_dict is None:
            infer_dict = get_infer_dict(train_loader)
        missing_indices = eval_np[:, :2]
        missing_labels = eval_np[:, 2:]

        if np.ndim(missing_labels) == 1:
            missing_labels = missing_labels[:, np.newaxis]

        ability = infer_dict['ability_mu'].to(device)
        item_feat = infer_dict['item_feat_mu'].to(device)
        inferred_response = model.decode(ability, item_feat)

        y_true, y_pred = [], []
        for missing_index, missing_label in zip(missing_indices, missing_labels):
            inferred_label = inferred_response[missing_index[0], missing_index[1]]
            y_pred.append(inferred_label.item())
            y_true.append(round(missing_label[0]))
        missing_imputation_accuracy = accuracy_score(y_true, [round(i) for i in y_pred])
        missing_imputation_f1 = f1_score(y_true, [round(i) for i in y_pred])
        missing_imputation_rmse = mean_squared_error(y_true, y_pred) ** 0.5
        missing_imputation_auc = roc_auc_score(y_true, y_pred).item()
        print(f'====> {name} Epoch: ' + '{} Acc: {:.4f} F1: {:.4f} RMSE: {:.4f} AUC: {:.4f}'.format(epoch, missing_imputation_accuracy, missing_imputation_f1, missing_imputation_rmse, missing_imputation_auc))
        return {
            'acc': missing_imputation_accuracy,
            'f1': missing_imputation_f1,
            'rmse': missing_imputation_rmse,
            'auc': missing_imputation_auc,
        }

    def get_infer_dict(loader):
        model.eval()
        infer_dict = {}

        with torch.no_grad(): 
            ability_mus, item_feat_mus = [], []
            ability_logvars, item_feat_logvars = [], []
            for _, response, _, mask in loader:
                mb = response.size(0)
                response = response.to(device)
                mask = mask.long().to(device)

                _, ability_mu, ability_logvar, _, item_feat_mu, item_feat_logvar = \
                    model.encode(response, mask)

                ability_mus.append(ability_mu.cpu())
                ability_logvars.append(ability_logvar.cpu())

                item_feat_mus.append(item_feat_mu.cpu())
                item_feat_logvars.append(item_feat_logvar.cpu())

            ability_mus = torch.cat(ability_mus, dim=0)
            ability_logvars = torch.cat(ability_logvars, dim=0)

        infer_dict['ability_mu'] = ability_mus
        infer_dict['ability_logvar'] = ability_logvars
        infer_dict['item_feat_mu'] = item_feat_mu
        infer_dict['item_feat_logvar'] = item_feat_logvar

        return infer_dict

    is_best, best_loss = False, np.inf
    train_losses = np.zeros(args.epochs)
    if not args.no_test:
        test_losses  = np.zeros(args.epochs)
    train_times = np.zeros(args.epochs)
    valid_accs = np.zeros(args.epochs)
    valid_f1s = np.zeros(args.epochs)
    valid_rmses = np.zeros(args.epochs)
    valid_aucs = np.zeros(args.epochs)
    test_accs = np.zeros(args.epochs)
    test_f1s = np.zeros(args.epochs)
    test_rmses = np.zeros(args.epochs)
    test_aucs = np.zeros(args.epochs)

    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = train(epoch)
        end_time = time.time()
        train_losses[epoch] = train_loss
        train_times[epoch] = start_time - end_time

        infer_dict = get_infer_dict(train_loader)
        valid_metrics = test_acc(epoch, valid_npy, 'Valid', infer_dict)
        test_metrics = test_acc(epoch, test_npy, 'Test', infer_dict)
        valid_accs[epoch] = valid_metrics['acc']
        valid_f1s[epoch] = valid_metrics['f1']
        valid_rmses[epoch] = valid_metrics['rmse']
        valid_aucs[epoch] = valid_metrics['auc']
        test_accs[epoch] = test_metrics['acc']
        test_f1s[epoch] = test_metrics['f1']
        test_rmses[epoch] = test_metrics['rmse']
        test_aucs[epoch] = test_metrics['auc']

    run_end_ts = time.time()
    run_end_str = time.strftime("%Y%m%d-%H%M%S", time.localtime(run_end_ts))
    duration = run_end_ts - run_start_ts
    method_name = f'vibo_{args.irt_model}_{args.generative_model}'
    result_filename = f'{method_name}_{args.dataset}_{run_start_str}.txt'
    result_path = os.path.join(result_root, result_filename)

    with open(result_path, 'w') as f:
        f.write(f'method={method_name}\n')
        f.write(f'dataset={args.dataset}\n')
        f.write(f'irt_model={args.irt_model}\n')
        f.write(f'generative_model={args.generative_model}\n')
        f.write(f'ability_dim={args.ability_dim}\n')
        f.write(f'ability_merge={args.ability_merge}\n')
        f.write(f'num_person={num_person}\n')
        f.write(f'num_item={num_item}\n')
        f.write(f'lr={args.lr}\n')
        f.write(f'batch_size={args.batch_size}\n')
        f.write(f'epochs={args.epochs}\n')
        f.write(f'seed={args.seed}\n')
        f.write(f'response_dist={args.response_dist}\n')
        f.write(f'start_time={run_start_str}\n')
        f.write(f'end_time={run_end_str}\n')
        f.write(f'duration_seconds={duration}\n')
        f.write('epoch,valid_acc,valid_f1,valid_rmse,valid_auc,test_acc,test_f1,test_rmse,test_auc\n')
        for epoch in range(args.epochs):
            f.write(
                f'{epoch + 1},'
                f'{valid_accs[epoch]:.6f},'
                f'{valid_f1s[epoch]:.6f},'
                f'{valid_rmses[epoch]:.6f},'
                f'{valid_aucs[epoch]:.6f},'
                f'{test_accs[epoch]:.6f},'
                f'{test_f1s[epoch]:.6f},'
                f'{test_rmses[epoch]:.6f},'
                f'{test_aucs[epoch]:.6f}\n'
            )

    print(f'Saved results to {result_path}')
