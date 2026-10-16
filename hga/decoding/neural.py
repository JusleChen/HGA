"""Final backbone factories, normalization and augmentation helpers."""
from __future__ import annotations
import os,random,hashlib
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from .fusion import probabilities
from ..models.atcnet import ATCNetTorch

def setup(seed):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    torch.set_num_threads(2);torch.set_num_interop_threads(2) if torch.get_num_interop_threads()!=2 else None
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')


def normalize(x,train,family):
    a=x[train];axis=0 if family=='ATCNet' else (0,2);mean=a.mean(axis=axis,keepdims=family!='ATCNet');scale=a.std(axis=axis,keepdims=family!='ATCNet');scale=np.where(scale<1e-20,1.,scale)
    z=((x-mean)/scale).astype(np.float32)
    if family=='ATCNet':z=z[:,None]
    return z,dict(mean=mean,scale=scale,fit_trial_count=len(train),fit_indices=train,scope='training trials only',family=family)


def apply_norm(x,stats):
    z=((x-stats['mean'])/stats['scale']).astype(np.float32)
    return z[:,None] if stats['family']=='ATCNet' else z


def state_hash(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def interaug(x,y):
    # Vectorized faithful official interaug for registered 1125-sample input (9x125).
    n_chunks=9;assert x.shape[-1]==1125
    xx=torch.zeros_like(x);yy=torch.zeros_like(y);current=0
    for cls in torch.unique(y):
        a=x[y==cls];n=len(a);chunks=torch.cat(torch.chunk(a,n_chunks,dim=-1),dim=0);indices=torch.randint(0,n,(n,n_chunks),device=x.device);offset=torch.arange(n_chunks,device=x.device)*n
        xx[current:current+n]=chunks[indices+offset].permute(0,2,1,3).reshape(n,x.shape[1],1125);yy[current:current+n]=cls;current+=n
    assert current==len(x);xx=torch.cat([x,xx]);yy=torch.cat([y,yy]);perm=torch.randperm(len(xx),device=x.device);return xx[perm],yy[perm]


@torch.no_grad()
def infer(model,x,batch=48):
    model.eval();out=[]
    for i in range(0,len(x),batch):out.append(torch.softmax(model(torch.as_tensor(x[i:i+batch],device='cuda:0')),dim=1).double().cpu().numpy())
    p=np.concatenate(out);p/=p.sum(1,keepdims=True);probabilities(p,len(x));return p


def make_decoder(family, pretrained=None):
    if family=='ATCNet':return ATCNetTorch()
    if family=='TCFormer':
        from ..models.tcformer import TCFormer
        return TCFormer(n_channels=22,n_classes=4,F1=32,temp_kernel_lengths=(20,32,64),
            d_group=16,D=2,pool_length_1=8,pool_length_2=7,dropout_conv=.4,
            use_group_attn=True,q_heads=4,kv_heads=2,trans_depth=5,trans_dropout=.4,
            tcn_depth=2,kernel_length_tcn=4,dropout_tcn=.3,lr=.0009,beta_1=.5,
            beta_2=.999,weight_decay=.001,optimizer='adam',scheduler=True,
            max_epochs=1000,warmup_epochs=20)
    if family=='MIRepNet':
        from ..models.mirepnet import mlm_mask
        return mlm_mask(emb_size=256,depth=6,n_classes=4,pretrain=pretrained,pretrainmode=False)
    raise ValueError('unknown backbone')


def mirepnet_input(x):
    from ..models.mirepnet.preprocessing import EA,pad_missing_channels_diff
    from ..models.mirepnet.channel_list import BNCI2014001_chn_names,use_channels_names
    a=EA(x)
    if not np.isrealobj(a) or not np.isfinite(a).all():raise ValueError('nonfinite/complex EA')
    return pad_missing_channels_diff(a.astype(np.float32),use_channels_names,BNCI2014001_chn_names).astype(np.float32)


@torch.no_grad()
def predict_decoder(model,x,family,batch=None):
    model.eval();out=[]
    batch=batch or (8 if family=='MIRepNet' else 64 if family=='ATCNet' else 48)
    for start in range(0,len(x),batch):
        value=model(torch.as_tensor(x[start:start+batch],device='cuda:0'))
        logits=value[1] if family=='MIRepNet' else value
        out.append(logits.softmax(-1).double().cpu().numpy())
    p=np.concatenate(out);p/=p.sum(1,keepdims=True);probabilities(p,len(x));return p


def train_decoder(x,y,train_indices,family,seed,pretrained=None):
    """Fixed final-epoch fit, returning the model and train-only normalization.

    The function receives training labels only. It never selects an epoch
    from held-out accuracy. Restart/checkpoint/report machinery is excluded
    from this compact source release.
    """
    if family=='MIRepNet':
        from ..models.mirepnet.preprocessing import set_seed,train as official_train
        if not pretrained:raise ValueError('official MIRepNet weights must be supplied separately')
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(False)
        set_seed(seed);xx=mirepnet_input(x[train_indices]);stats={'family':family}
    else:
        setup(seed);normalized,stats=normalize(x,train_indices,family);xx=normalized[train_indices]
    model=make_decoder(family,pretrained).to('cuda:0')
    yy=np.asarray(y[train_indices],dtype=np.int64)
    if family=='MIRepNet':
        loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(xx),torch.from_numpy(yy)),
            batch_size=8,shuffle=True,num_workers=0,drop_last=False)
        opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-6)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=10)
        for _ in range(10):official_train(model,loader,torch.nn.CrossEntropyLoss(),opt,'cuda',sched)
    else:
        if family=='ATCNet':
            opt=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-7,weight_decay=0)
            sched=None;epochs=500;batch=64
        else:
            from ..models.tcformer.lr_scheduler import linear_warmup_cosine_decay
            opt=torch.optim.Adam(model.parameters(),lr=.0009,betas=(.5,.999),eps=1e-8,weight_decay=.001)
            sched=torch.optim.lr_scheduler.LambdaLR(opt,linear_warmup_cosine_decay(20,1000));epochs=1000;batch=48
        xt=torch.as_tensor(xx,device='cuda:0');yt=torch.as_tensor(yy,device='cuda:0')
        for _ in range(epochs):
            model.train();perm=torch.randperm(len(yy),device='cuda:0')
            for start in range(0,len(yy),batch):
                ix=perm[start:start+batch];bx,by=xt[ix],yt[ix]
                if family=='TCFormer':bx,by=interaug(bx,by)
                opt.zero_grad(set_to_none=True);logits=model(bx)
                ce=torch.nn.functional.cross_entropy(logits,by)
                loss=ce+model.regularization_loss() if family=='ATCNet' else ce
                if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
                loss.backward();opt.step()
                if family=='ATCNet':model.project_constraints_()
            if sched:sched.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):raise RuntimeError('nonfinite model')
    return model,stats


def prepare_prediction(x,stats):
    return mirepnet_input(x) if stats['family']=='MIRepNet' else apply_norm(x,stats)
