"""Train-only source fusion for one participant; explicit prediction/scoring stages.

This is a compact array interface to the final algorithms, not the archived
multi-process execution/audit system. HGD supports ATCNet only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from hga.decoding.source_features import extract_trials
from hga.decoding.nested import fit_nested,predict
from hga.decoding.classical import calibrate
from hga.decoding.fusion import calibrated,choose_weight,fuse

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save_json(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

def read_features(path,assets,training):
    with np.load(path,allow_pickle=False) as z:
        if not training and any(k in z for k in ('y','labels','class_labels')):
            raise ValueError('prediction input must contain no evaluation labels')
        x=z['X_v'];mi=z['mi_X'] if 'mi_X' in z else None
        y=z['y'] if training else None;runs=z['run_id']
    with np.load(assets,allow_pickle=False) as z:
        arrays=extract_trials(x,z['kernel'],[z[f'roi_{i}'] for i in range(8)])
    return x,mi,y,runs,arrays

def train(args):
    from hga.decoding.neural import train_decoder,prepare_prediction,predict_decoder
    import torch
    if args.dataset=='hgd' and args.model!='ATCNet':raise ValueError('only ATCNet was evaluated on HGD')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if any((out/n).exists() for n in ['training.lock.json','predictions.lock.json','scores.json']):
        raise RuntimeError('output is already locked; use a new directory for another run')
    x,mi,y,runs,arrays=read_features(args.input,args.assets,True)
    cfg=json.loads(Path('configs/'+args.dataset+'.json').read_text())
    seeds=cfg['seeds'];groups=np.unique(runs)
    classical={};oof={}
    for condition in ('source','extra'):
        oof[condition],classical[condition]=fit_nested(arrays,y,runs,condition)
    joblib.dump(classical,out/'classical.joblib')
    neural_x=mi if args.model=='MIRepNet' else x
    if neural_x is None:raise ValueError('MIRepNet requires separately reconstructed mi_X')
    for seed in seeds:
        deep_oof=np.zeros((len(y),4))
        for held in groups:
            tr=np.flatnonzero(runs!=held);va=runs==held
            model,stats=train_decoder(neural_x,y,tr,args.model,seed,args.pretrained)
            deep_oof[va]=predict_decoder(model,prepare_prediction(neural_x[va],stats),args.model)
            del model;torch.cuda.empty_cache()
        temperature=1.
        if args.model!='MIRepNet':
            temperature,info=calibrate(deep_oof,y)
            if not info['success'] or not np.isfinite(temperature) or not .1<=temperature<=10:temperature=1.
        calibrated_oof=calibrated(deep_oof,temperature)
        weights={c:choose_weight(calibrated_oof,oof[c],y,runs)[0] for c in classical}
        model,stats=train_decoder(neural_x,y,np.arange(len(y)),args.model,seed,args.pretrained)
        torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/f'seed{seed}.pt')
        joblib.dump(stats,out/f'seed{seed}_normalization.joblib')
        save_json(out/f'seed{seed}_fusion.json',dict(temperature=temperature,weights=weights))
        del model;torch.cuda.empty_cache()
    files=sorted(p for p in out.iterdir() if p.is_file())
    save_json(out/'training.lock.json',dict(dataset=args.dataset,model=args.model,seeds=seeds,
        training_input_sha256=digest(args.input),assets_sha256=digest(args.assets),
        config_sha256=digest('configs/'+args.dataset+'.json'),files={p.name:digest(p) for p in files}))

def prediction(args):
    from hga.decoding.neural import make_decoder,prepare_prediction,predict_decoder
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for registered decoder inference')
    out=Path(args.output)
    if (out/'predictions.lock.json').exists() or (out/'scores.json').exists():raise RuntimeError('predictions already locked')
    lock=json.loads((out/'training.lock.json').read_text())
    for name,wanted in lock['files'].items():
        if digest(out/name)!=wanted:raise RuntimeError('training artifact changed: '+name)
    if digest(args.assets)!=lock['assets_sha256']:raise RuntimeError('source assets changed')
    x,mi,_,runs,arrays=read_features(args.input,args.assets,False)
    classical=joblib.load(out/'classical.joblib')
    branches={k:predict(v,arrays) for k,v in classical.items()};probabilities={}
    for seed in lock['seeds']:
        family=lock['model'];model=make_decoder(family).to('cuda:0')
        model.load_state_dict(torch.load(out/f'seed{seed}.pt',map_location='cuda:0',weights_only=True))
        stats=joblib.load(out/f'seed{seed}_normalization.joblib')
        policy=json.loads((out/f'seed{seed}_fusion.json').read_text())
        xx=mi if family=='MIRepNet' else x
        raw=predict_decoder(model,prepare_prediction(xx,stats),family)
        baseline=calibrated(raw,policy['temperature'])
        probabilities[f'M0_seed{seed}']=baseline
        probabilities[f'M1_seed{seed}']=fuse(baseline,branches['source'],policy['weights']['source'])
        probabilities[f'M2_seed{seed}']=fuse(baseline,branches['extra'],policy['weights']['extra'])
        del model;torch.cuda.empty_cache()
    np.savez_compressed(out/'predictions.npz',**probabilities)
    save_json(out/'predictions.lock.json',dict(prediction_sha256=digest(out/'predictions.npz'),
        training_lock_sha256=digest(out/'training.lock.json'),labels_read=False))

def score(args):
    from sklearn.metrics import balanced_accuracy_score,f1_score,cohen_kappa_score
    out=Path(args.output);once=out/'label_access_once.json'
    if once.exists():raise RuntimeError('labels already opened; recover saved scores')
    lock=json.loads((out/'predictions.lock.json').read_text())
    if digest(out/'predictions.npz')!=lock['prediction_sha256']:raise RuntimeError('prediction hash mismatch')
    # Record the sole label-access attempt before opening the external label file.
    save_json(once,dict(status='LABEL_ACCESS_ATTEMPTED_AFTER_PREDICTION_LOCK'))
    y=np.load(args.labels,allow_pickle=False)
    rows=[]
    with np.load(out/'predictions.npz') as z:
        for name in z.files:
            pred=z[name].argmax(1)
            rows.append(dict(method_seed=name,BA=float(balanced_accuracy_score(y,pred)),
                MacroF1=float(f1_score(y,pred,average='macro')),Kappa=float(cohen_kappa_score(y,pred))))
    save_json(out/'scores.json',rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['train','predict','score'])
    p.add_argument('--dataset',choices=['bcic','hgd'],default='bcic')
    p.add_argument('--model',choices=['ATCNet','TCFormer','MIRepNet'],default='ATCNet')
    p.add_argument('--input',help='Prepared training or label-free test NPZ')
    p.add_argument('--assets',help='External NPZ: kernel and eight roi_i arrays')
    p.add_argument('--output',required=True)
    p.add_argument('--pretrained',help='External official MIRepNet weights; never uploaded')
    p.add_argument('--labels',help='External evaluation label NPY, score stage only')
    args=p.parse_args()
    if args.stage in ('train','predict') and (not args.input or not args.assets):p.error('--input and --assets required')
    if args.stage=='score' and not args.labels:p.error('--labels required for scoring')
    {'train':train,'predict':prediction,'score':score}[args.stage](args)

if __name__=='__main__':main()
