import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tensorboardX import SummaryWriter
from scipy import sparse
import models
# import dataORIG
import metric
from utils import naive_sparse2tensor,sparse2torch_sparse
from data import *
#TODO: Dataloader is not multi process
#TODO: Dataloader to add embeddings
#TODO: Save mapping of original paper/author ids to indexes.
#TODO: Try to use CNN model - less parameters
#TODO: Last layer softmax!!! normalize input also

class UpdateCount:
    def __init__(self):
        self.count = 0

def train(model, train_loader, args, writer, device, optimizer, criterion, epoch, update_count):
    # Turn on training mode
    model.train()
    train_loss = 0.0
    start_time = time.time()

    for batch_idx, (array, offsets, weights, embs,  sparse_matrix) in enumerate(train_loader):

        # batch = sparse2torch_sparse(batch, model.half_precision).to(device)

        # batch = naive_sparse2tensor(batch, model.half_precision).to(device)
        # batch = naive_sparse2tensor(batch, False).to(device)

        sparse_matrix = sparse2torch_sparse(sparse_matrix).to(device)

        array = array.to(device)
        offsets = offsets.to(device)
        weights = weights.to(device)
        embs = embs.to(device)

        if args.total_anneal_steps > 0:
            anneal = min(args.anneal_cap,
                         1. * update_count.count / args.total_anneal_steps)
        else:
            anneal = args.anneal_cap

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(array, offsets, weights, embs)

        # loss = criterion(recon_batch, batch, mu, logvar, anneal)
        loss = criterion(recon_batch, sparse_matrix, mu, logvar, anneal)
        # torch.cuda.empty_cache()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        update_count.count += 1

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
                  'loss {:4.2f}'.format(
                epoch, batch_idx, len(train_loader),
                elapsed * 1000 / args.log_interval,
                train_loss / args.log_interval))

            # Log loss to tensorboard
            n_iter = (epoch - 1) * len(train_loader) + batch_idx
            writer.add_scalars('data/loss', {'train': train_loss / args.log_interval}, n_iter)

            start_time = time.time()
            train_loss = 0.0

            # TODO: This block is my addition
            with open(args.save, 'wb') as f:
                torch.save(model, f)

def evaluate(model,args, validation_train_loader, validation_test_loader):
    # Turn on evaluation mode
    model.eval()
    total_loss = 0.0
    e_idxlist = list(range(data_tr.shape[0]))
    e_N = data_tr.shape[0]
    n100_list = []
    r20_list = []
    r50_list = []

    with torch.no_grad():
        for start_idx in range(0, e_N, args.batch_size):
            end_idx = min(start_idx + args.batch_size, N)
            data = data_tr[e_idxlist[start_idx:end_idx]]
            heldout_data = data_te[e_idxlist[start_idx:end_idx]]

            data_tensor = naive_sparse2tensor(data).to(device)

            if args.total_anneal_steps > 0:
                anneal = min(args.anneal_cap,
                             1. * update_count.count / args.total_anneal_steps)
            else:
                anneal = args.anneal_cap

            recon_batch, mu, logvar = model(data_tensor)

            loss = criterion(recon_batch, data_tensor, mu, logvar, anneal)
            total_loss += loss.item()

            # Exclude examples from training set
            recon_batch = recon_batch.cpu().numpy()
            recon_batch[data.nonzero()] = -np.inf

            n100 = metric.NDCG_binary_at_k_batch(recon_batch, heldout_data, 100)
            r20 = metric.Recall_at_k_batch(recon_batch, heldout_data, 20)
            r50 = metric.Recall_at_k_batch(recon_batch, heldout_data, 50)

            n100_list.append(n100)
            r20_list.append(r20)
            r50_list.append(r50)

            # TODO: This block is my addition
            with open(args.save, 'wb') as f:
                torch.save(model, f)

    total_loss /= len(range(0, e_N, args.batch_size))
    n100_list = np.concatenate(n100_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)

    return total_loss, np.mean(n100_list), np.mean(r20_list), np.mean(r50_list)

def main():
    parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')
    parser.add_argument('--input_file', type=str, default='data/processed_output.h5',
                        help='Processed input h5 file.')
    parser.add_argument('--embeddings_file', type=str, default='data/embeddings_output.h5',
                        help='Processed input h5 file.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='initial learning rate')
    parser.add_argument('--wd', type=float, default=0.00,
                        help='weight decay coefficient')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=200,
                        help='upper epoch limit')
    parser.add_argument('--total_anneal_steps', type=int, default=200000,
                        help='the total number of gradient updates for annealing')
    parser.add_argument('--anneal_cap', type=float, default=0.2,
                        help='largest annealing parameter')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    parser.add_argument('--workers', type=int, default=2,
                        help='num workers')
    parser.add_argument('--hidden_dim1', type=int, default=150,
                        help='Dimension of first layer in model.')
    parser.add_argument('--hidden_dim2', type=int, default=50,
                        help='Dimension of second layer in model.')

    args = parser.parse_args()

    # Set the random seed manually for reproductibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device("cuda" if args.cuda else "cpu")
    # device = torch.device("cuda")

    # warnings.warn("Using CPU")
    # device = torch.device("cpu")
    # args.batch_size = 16
    ###############################################################################
    # Load data
    ###############################################################################

    train_dataset = H5Dataset(args.input_file, args.embeddings_file,  'train', 'train')
    train_sampler = RangeSampler(list(range(0, len(train_dataset)))) #TODO: See if can be omitted
    train_loader = H5DataLoader(train_dataset, train_sampler, args.batch_size, args.workers)

    validation_train_dataset = H5Dataset(args.input_file,args.embeddings_file, 'validation', 'train')
    validation_train_sampler = RangeSampler(list(range(0, len(validation_train_dataset))))
    validation_train_loader = H5DataLoader( validation_train_dataset, validation_train_sampler, args.batch_size, args.workers)

    validation_test_dataset = H5Dataset(args.input_file,args.embeddings_file, 'validation', 'test')
    validation_test_sampler = RangeSampler(list(range(0, len(validation_test_dataset))))
    validation_test_loader = H5DataLoader( validation_test_dataset, validation_test_sampler, args.batch_size, args.workers)

    ###############################################################################
    # Build the model
    ###############################################################################

    # p_dims = [ 100, 300, train_dataset.authors_n]
    p_dims = [100, 300, train_dataset.authors_n]

    #TODO
    model = models.SparseMultiVAE(train_dataset.authors_n, args.hidden_dim1, args.hidden_dim2).to(device)

    # model = models.MultiVAE(p_dims, half_precision=False).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.wd)
    criterion = models.loss_function
    ###############################################################################
    # Training code
    ###############################################################################

    best_n100 = -np.inf
    update_count = UpdateCount()

    # TensorboardX Writer
    writer = SummaryWriter()

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train(model, train_loader, args, writer, device, optimizer, criterion, epoch, update_count)
            # val_loss, n100, r20, r50 = evaluate(model,args ,validation_train_loader, validation_test_loader)
            # print('-' * 89)
            # print('| end of epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | '
            #       'n100 {:5.3f} | r20 {:5.3f} | r50 {:5.3f}'.format(
            #     epoch, time.time() - epoch_start_time, val_loss,
            #     n100, r20, r50))
            # print('-' * 89)
            #
            # n_iter = epoch * len(range(0, N, args.batch_size))
            # writer.add_scalars('data/loss', {'valid': val_loss}, n_iter)
            # writer.add_scalar('data/n100', n100, n_iter)
            # writer.add_scalar('data/r20', r20, n_iter)
            # writer.add_scalar('data/r50', r50, n_iter)
            #
            # # Save the model if the n100 is the best we've seen so far.
            # if n100 > best_n100:
            #     with open(args.save, 'wb') as f:
            #         torch.save(model, f)
            #     best_n100 = n100

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_loss, n100, r20, r50 = evaluate(test_data_tr, test_data_te)
    print('=' * 89)
    print('| End of training | test loss {:4.2f} | n100 {:4.2f} | r20 {:4.2f} | '
          'r50 {:4.2f}'.format(test_loss, n100, r20, r50))
    print('=' * 89)
if __name__=='__main__':
    profile = False
    if profile:
        # try: os.remove('program.prof')
        # except Exception as e: print(e)
        import cProfile
        cProfile.run("main()",'program.prof')
    else:
        main()
