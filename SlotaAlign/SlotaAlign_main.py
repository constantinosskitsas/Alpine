import json
import time
from SlotaAlign.GWLTorch import *
from SlotaAlign.utils import *
random.seed(123)
torch.random.manual_seed(123)
np.random.seed(123)
def args1(dataset):
    parser = argparse.ArgumentParser()
    parser.add_argument('--step_size', type=float, default=0.1)
    parser.add_argument('--bases', type=int, default=2)
    parser.add_argument('--joint_epoch', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--gw_beta', type=float, default=0.01)
    f = open(f'SlotaAlign/config/{dataset}.json', 'r')
    args = parser.parse_args()
    arg_dict = json.load(f)
    for t in arg_dict:
        args.__dict__[t] = arg_dict[t]
    return args

    

def SlotaA(Aadj1,Badj1,Afeat,Bfeat,dataset):
    args=args1(dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Aadj=nx.to_numpy_array(Aadj1)
    Badj=nx.to_numpy_array(Badj1)
    #Aadj, Badj, Afeat, Bfeat, ground_truth = myload(args.dataset, args.edge_noise)
    Adim, Bdim = Afeat.shape[0], Bfeat.shape[0]
    #print(np.shape(Aadj))
    #print(np.shape(Badj))
    #print(Adim)
    #print(Bdim)
    Ag = dgl.graph(np.nonzero(Aadj), num_nodes=Adim)
    Bg = dgl.graph(np.nonzero(Badj), num_nodes=Bdim)
    Afeat -= Afeat.mean(0)
    Bfeat -= Bfeat.mean(0)

    #if args.truncate:
    #    Afeat = Afeat[:,:100]
    #    Bfeat = Bfeat[:,:100]


    Afeat = torch.tensor(Afeat).float().to(device)
    Bfeat = torch.tensor(Bfeat).float().to(device)

    time_st = time.time()
    layers = args.bases-2
    conv = GraphConv(0, 0, norm='both', weight=False, bias=False)
    Afeats = [torch.tensor(Afeat)]
    Bfeats = [torch.tensor(Bfeat)]
    Ag = Ag.to(device)
    Bg = Bg.to(device)
    for i in range(layers):
        Afeats.append(conv(dgl.add_self_loop(Ag), torch.tensor(Afeats[-1])).detach().clone())
        Bfeats.append(conv(dgl.add_self_loop(Bg), torch.tensor(Bfeats[-1])).detach().clone())

    Asims, Bsims = [Ag.adj().to_dense().to(device)],[Bg.adj().to_dense().to(device)]
    for i in range(len(Afeats)):
        Afeat = Afeats[i]
        Bfeat = Bfeats[i]
        Afeat = Afeat / (Afeat.norm(dim=1)[:, None]+1e-16)
        Bfeat = Bfeat / (Bfeat.norm(dim=1)[:, None]+1e-16)
        Asim = Afeat.mm(Afeat.T)
        Bsim = Bfeat.mm(Bfeat.T)
        Asims.append(Asim)
        Bsims.append(Bsim)


    Adim, Bdim = Afeat.shape[0], Bfeat.shape[0]
    a = torch.ones([Adim,1]).to(device)/Adim
    b = torch.ones([Bdim,1]).to(device)/Bdim
    X = a@b.T
    As = torch.stack(Asims,dim=2)
    Bs = torch.stack(Bsims,dim=2)

    alpha0 = np.ones(layers+2).astype(np.float32)/(layers+2)
    beta0 = np.ones(layers+2).astype(np.float32)/(layers+2)
    for ii in range(args.joint_epoch):
        #print(ii,"outside ", args.joint_epoch)
        alpha = torch.autograd.Variable(torch.tensor(alpha0)).to(device)
        alpha.requires_grad = True
        beta = torch.autograd.Variable(torch.tensor(beta0)).to(device)
        beta.requires_grad = True
        A = (As * alpha).sum(2)
        B = (Bs * beta).sum(2)
        objective = (A ** 2).mean() + (B ** 2).mean() - torch.trace(A @ X @ B @ X.T)
        alpha_grad = torch.autograd.grad(outputs=objective, inputs=alpha, retain_graph=True)[0]
        alpha = alpha - args.step_size * alpha_grad
        alpha0 = alpha.detach().cpu().numpy()
        alpha0 = euclidean_proj_simplex(alpha0)
        beta_grad = torch.autograd.grad(outputs=objective, inputs=beta)[0]
        beta = beta - args.step_size * beta_grad
        beta0 = beta.detach().cpu().numpy()
        beta0 = euclidean_proj_simplex(beta0)
        X = gw_torch(A.clone().detach(), B.clone().detach(), a, b, X.clone().detach(), beta=args.gw_beta,
                    outer_iter=1, inner_iter=50).clone().detach()
        if ii == args.joint_epoch-1:
            print(alpha0, beta0)
            X = gw_torch(A.clone().detach(), B.clone().detach(), a, b, X, beta=args.gw_beta, outer_iter=args.epoch-args.joint_epoch, inner_iter=20, gt=None)
    time_ed = time.time()
    res=X.T.cpu().numpy()
    #a1,a5,a10,a30 = my_check_align(res, ground_truth)
    time_cost = time_ed - time_st
    return res

