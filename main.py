import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from utils.dataloader import get_dataloader
from utils.MOOD import get_ood_score, sample_estimator
from utils.MOOD import auroc, fpr95
import argparse

mood_parser = argparse.ArgumentParser()
mood_parser.add_argument('-s', '--score', type=str,
                    default='energy', 
                    help='basic score for MOOD method, choose from: energy, msp, odin, mahalanobis')

mood_parser.add_argument('-f', '--file', type=str,
                    default='trained_model/msdnet_cifar10.pth.tar', 
                    help='model file for MSDNet')
mood_parser.add_argument('-l', '--layer', type=int,
                    default=5, 
                    help='# of exits for MSDNet')

mood_parser.add_argument('-i', '--id', type=str,
                    default='cifar10', 
                    help='in distribution dataset: cifar10 or cifar100')
mood_parser.add_argument('-o', '--od', type=list, 
                    default=['mnist',
                             'kmnist',
                             'fasionmnist',
                             'lsun',
                             'svhn',
                             'dtd',
                             'stl10',
                             'place365',
                             'isun',
                             'lsunR'
                             ],
                    help='all 10 OOD datasets used in experiment')

mood_parser.add_argument('-c', '--compressor', type=str, 
                    default='png',
                    help='compressor for complexity')
mood_parser.add_argument('-t', '--threshold', type=int, 
                    default=[0, 
                             1*2700/5, 
                             2*2700/5,
                             3*2700/5,
                             4*2700/5,
                             9999],
                    
                    help='the complex thresholds for different exits in MSDNet')
mood_parser.add_argument('-a', '--adjusted', type=int, 
                    default=1,
                    help='adjusted energy score: mode 1: minus mean; mode 0: keep as original')

mood_parser.add_argument('-b', '--bs', type=int, 
                    default=64,
                    help='batch size')
mood_args = mood_parser.parse_args()


if 1:#load and test model
    from msd_args import arg_parser
    import models
    from msd_dataloader import msd_get_dataloaders
    args = arg_parser.parse_args()
    args.grFactor = list(map(int, args.grFactor.split('-')))
    args.bnFactor = list(map(int, args.bnFactor.split('-')))
    args.nScales = len(args.grFactor)
    
    if args.use_valid:
        args.splits = ['train', 'val', 'test']
    else:
        args.splits = ['train', 'val']
    args.data = mood_args.id
    if args.data == 'cifar10':
        args.num_classes = 10
    elif args.data == 'cifar100':
        args.num_classes = 100
    else:
        print('dataset not support!')
    
    model = getattr(models, args.arch)(args)
    model = torch.nn.DataParallel(model).cuda()
    
    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    train_loader, val_loader, test_loader = msd_get_dataloaders(args)
    print("*************************************")
    print(args.use_valid, len(train_loader), len(val_loader))
    print("*************************************")
    
    model.load_state_dict(torch.load(mood_args.file)['state_dict'])
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    model.eval()
if 1:
    from utils.msdnet_function import validate
    val_loss, val_err1, val_err5 = validate(test_loader, model, criterion)



if mood_args.id == 'cifar10':
    MEAN=[0.4914, 0.4824, 0.4467]
    STD=[0.2471, 0.2435, 0.2616]
    NM = [MEAN,STD]
elif mood_args.id == 'cifar100':
    MEAN=[0.5071, 0.4867, 0.4408]
    STD=[0.2675, 0.2565, 0.2761]
    NM = [MEAN,STD]
else:
    print('wrong indistribution dataset! use cifar10 or cifar100!')
    
normalizer = transforms.Normalize(mean=MEAN, std=STD)
print('calculating ood scores and complexity takes long time')
print('process ',mood_args.id)

dataloader = get_dataloader(mood_args.id, normalizer, mood_args.bs)
if mood_args.score == 'mahalanobis':
    print('processing mahalanobis parameters')
    if mood_args.id == 'cifar10':
        num_classes = 10
        magnitude = 0.012
    elif mood_args.id == 'cifar100':
        num_classes = 100
        magnitude = 0.006
    else:
        print('did not support this in distribution dataset!')
    # get fake feature list
    model.eval()
    temp_x = torch.rand(2,3,32,32).cuda()
    temp_list = model(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1
    sample_mean, precision = sample_estimator(model, num_classes, feature_list, dataloader)
    data_output = open('mahalanobis_parameters/sample_mean.pkl','wb')
    pickle.dump(sample_mean, data_output)
    data_output.close()
    data_output = open('mahalanobis_parameters/precision.pkl','wb')
    pickle.dump(precision, data_output)
    data_output.close()
    data_output = open('mahalanobis_parameters/num_classes.pkl','wb')
    pickle.dump(num_classes, data_output)
    data_output.close()
    data_output = open('mahalanobis_parameters/magnitude.pkl','wb')
    pickle.dump(magnitude, data_output)
    data_output.close()
    print('processing mahalanobis parameters finished!')
    
i_score, i_adjusted_score, i_complexity = get_ood_score(data_name=mood_args.id,
                           model=model,
                           L=mood_args.layer,
                           dataloader=dataloader,
                           score_type=mood_args.score,
                           threshold=mood_args.threshold,
                           NM=NM,
                           adjusted_mode=0,   
                           mean=None,
                           cal_complexity=True
                           )
mean=[]
for i in range(mood_args.layer):
    mean.append( np.mean(i_score[i]) )

i_score, i_adjusted_score, i_complexity = get_ood_score(data_name=mood_args.id,
                           model=model,
                           L=mood_args.layer,
                           dataloader=dataloader,
                           score_type=mood_args.score,
                           threshold=mood_args.threshold,
                           NM=NM,
                           adjusted_mode=mood_args.adjusted,
                           mean=mean,
                           cal_complexity=True
                           )
auroc_base = []
fpr95_base = []
auroc_mood = []
fpr95_mood = []
auroc_for_barplot = []
complexity_for_arplot = []
for o_name in mood_args.od:
    print('process ',o_name)
    dataloader = get_dataloader(o_name, normalizer, mood_args.bs)
    o_score, o_adjusted_score, o_complexity = get_ood_score(data_name=o_name,
                           model=model,
                           L=mood_args.layer,
                           dataloader=dataloader,
                           score_type=mood_args.score,
                           threshold=mood_args.threshold,
                           NM=NM,
                           adjusted_mode=mood_args.adjusted,
                           mean=mean,
                           cal_complexity=True
                           )
    auroc_base.append(auroc(i_score[-1], o_score[-1]))
    fpr95_base.append(fpr95(i_score[-1], o_score[-1]))
    auroc_mood.append(auroc(i_adjusted_score, o_adjusted_score))
    fpr95_mood.append(fpr95(i_adjusted_score, o_adjusted_score))
    auroc_for_barplot.append([auroc(i_score[i], o_score[i]) for i in range(mood_args.layer)])
    complexity_for_arplot.append(o_complexity)

print('********** auroc result ',mood_args.id,' with ',mood_args.score,' **********')
print('                         auroc                  fpr95    ')
print('OOD dataset      exit@last    MOOD      exit@last    MOOD')
for i in range(len(mood_args.od)):
    data_name=mood_args.od[i]
    data_name = data_name + ' '*(17-len(data_name))
    print(data_name,"%.4f"%auroc_base[i],'   ',"%.4f"%auroc_mood[i],'    ',"%.4f"%fpr95_base[i],'   ',"%.4f"%fpr95_mood[i])
data_name = 'average'
data_name = data_name + ' '*(17-len(data_name))
print(data_name,"%.4f"%np.mean(auroc_base),'   ',"%.4f"%np.mean(auroc_mood),'    ',"%.4f"%np.mean(fpr95_base),'   ',"%.4f"%np.mean(fpr95_mood))


if mood_args.score == 'energy' and mood_args.adjusted == 1 :
    flops = np.array([26621540, 51598536, 68873004, 88417936, 105102580])
    auroc_score = np.array(auroc_for_barplot)
    S=20
    selected_datasets = mood_args.od
    selected_score = np.zeros_like(auroc_score)
    for k, complexity in enumerate(complexity_for_arplot):
        for i in range(mood_args.layer):
            index = (mood_args.threshold[i]<complexity) * (complexity<=mood_args.threshold[i+1])
            selected_score[k,i] = np.sum(index)/complexity.shape[0]
    
    Flops = np.zeros([10])
    for i in range(10):
        Flops[i] = np.sum(selected_score[i,:]*flops)
    Flops2 = np.ones([10])*flops[-1]
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas
    df = pandas.DataFrame({
    'dataset': selected_datasets,
    'Exit@1': selected_score[:,0],
    'Exit@2': selected_score[:,1],
    'Exit@3': selected_score[:,2],
    'Exit@4': selected_score[:,3],
    'Exit@5': selected_score[:,4],
    })
    fig, ax = plt.subplots(figsize=(30,5.5))
    tidy = df.melt(id_vars='dataset').rename(columns={"dataset": "Dataset",
                                                      "variable": "Method",
                                                      "value": "AUROC"})
    sns.barplot(x='Dataset', y='AUROC', hue='Method', data=tidy, ax=ax, palette=['#d9ece0','#a8e9dd','#8bd6f3','#508fed','#544cbd','#909090'])
    plt.setp(ax.get_xticklabels(), fontsize=S)
    plt.setp(ax.get_yticklabels(), fontsize=S)
    plt.xlabel('Dataset', fontsize=S)
    plt.ylabel('Exit Distribution', fontsize=S)
    plt.ylim(0,1)
    ax.legend(bbox_to_anchor=(1.14, 0.90), fontsize=S)
    
    
    ax2 = ax.twinx()
    ax2.plot(selected_datasets, Flops, '--', label = 'MOOD',marker='x', linewidth=2.5)
    ax2.plot(selected_datasets, Flops2, '--', label = 'Exit@5',marker='x', linewidth=2.5)

    ax2.set_ylabel("Computational Cost(Flops)", fontsize=S)
    plt.setp(ax2.get_yticklabels(), fontsize=S)
    ax2.yaxis.get_offset_text().set_fontsize(S-4)
    ax2.legend(bbox_to_anchor=(1.14, 0.30), fontsize=S)

    fig.savefig("Flops.pdf", bbox_inches='tight')
