import torch
import torchvision.transforms as transforms
from torchvision import datasets

def get_dataloader(name,normalizer,bs):
    if name == 'cifar10':
        dataloader = cifar10(normalizer,bs)
    elif name == 'cifar100':
        dataloader = cifar100(normalizer,bs)
    elif name == 'mnist':
        dataloader = mnist(normalizer,bs)
    elif name == 'kmnist':
        dataloader = kmnist(normalizer,bs)
    elif name == 'fasionmnist':
        dataloader = fasionmnist(normalizer,bs)
    elif name == 'svhn':
        dataloader = svhn(normalizer,bs)
    elif name == 'stl10':
        dataloader = stl10(normalizer,bs)
    elif name == 'dtd':
        dataloader = dtd(normalizer,bs)
    elif name == 'place365':
        dataloader = place365(normalizer,bs)
    elif name == 'lsun':
        dataloader = lsun(normalizer,bs)
    elif name == 'lsunR':
        dataloader = lsunR(normalizer,bs)
    elif name == 'isun':
        dataloader = isun(normalizer,bs)
    elif name == 'celebA':
        dataloader = celebA(normalizer,bs)
    else:
        print('the dataset is not used in this project')
        return None
    return dataloader


def cifar10(normalizer,bs):
    transform_cifar10 = transforms.Compose([transforms.ToTensor(),
                                            normalizer
                                            ])
    dataloader = torch.utils.data.DataLoader(
                datasets.CIFAR10('data/cifar10', 
                                train=False, 
                                download=True,
                                transform=transform_cifar10),
                batch_size=bs, 
                shuffle=False, 
                num_workers=1, 
                pin_memory=True)
    return dataloader

def celebA(normalizer,bs):
    transformer = transforms.Compose([transforms.Resize(32),
                                      transforms.ToTensor(),
                                      normalizer
                                      ])
    dataloader = torch.utils.data.DataLoader(
                datasets.CelebA('data/celebA', 
                                split='test', 
                                download=True,
                                transform=transformer),
                batch_size=bs, 
                shuffle=False, 
                num_workers=1, 
                pin_memory=True)
    return dataloader

def cifar100(normalizer,bs):
    transform_cifar100 = transforms.Compose([transforms.ToTensor(),
                                            normalizer
                                            ])
    dataloader = torch.utils.data.DataLoader(
                datasets.CIFAR100('data/cifar100', 
                                 train=False, 
                                 download=True,
                                 transform=transform_cifar100),
                batch_size=bs, 
                shuffle=False, 
                num_workers=1, 
                pin_memory=True)
    return dataloader
def mnist(normalizer,bs):
    transformer = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                      transforms.Pad(padding=2),
                                      transforms.ToTensor(),
                                      normalizer
                                      ])
    dataloader = torch.utils.data.DataLoader(
                datasets.MNIST('data/mnist', 
                                train=False, 
                                download=True,
                                transform=transformer),
                batch_size=bs, 
                shuffle=False, 
                num_workers=1, 
                pin_memory=True)
    return dataloader
def kmnist(normalizer,bs):
    transformer = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                      transforms.Pad(padding=2),
                                      transforms.ToTensor(),
                                      normalizer
                                      ])
    dataloader = torch.utils.data.DataLoader(
                datasets.KMNIST('data/kmnist', 
                                train=False, 
                                download=True,
                                transform=transformer),
                batch_size=bs, 
                shuffle=False, 
                num_workers=1, 
                pin_memory=True)
    return dataloader
def fasionmnist(normalizer,bs):
    transformer = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                      transforms.Pad(padding=2),
                                      transforms.ToTensor(),
                                      normalizer
                                      ])
    dataloader = torch.utils.data.DataLoader(
                datasets.FashionMNIST('data/fasionmnist', 
                                train=False, 
                                download=True,
                                transform=transformer),
                batch_size=bs, 
                shuffle=False, 
                num_workers=1, 
                pin_memory=True)
    return dataloader
'''
def svhn(normalizer,bs):
    transformer = transforms.Compose([transforms.ToTensor(),
                                      normalizer
                                      ])
    dataloader = torch.utils.data.DataLoader(
                datasets.SVHN('data/svhn', 
                              split='test', 
                              download=True,
                              transform=transformer),
                batch_size=bs, 
                shuffle=False, 
                num_workers=1, 
                pin_memory=True)
    return dataloader
'''
def stl10(normalizer,bs):
    transformer = transforms.Compose([transforms.Resize(32),
                                      transforms.ToTensor(),
                                      normalizer
                                      ])
    dataloader = torch.utils.data.DataLoader(
                datasets.STL10('data/STL10',
                                split='test',
                                folds=0,
                                download=(True),
                                transform=transformer),
                batch_size=bs, 
                shuffle=False, 
                num_workers=1, 
                pin_memory=True)
    return dataloader

def svhn(normalizer,bs):
    import utils.svhn_loader as svhn
    transformer = transforms.Compose([transforms.ToTensor(),
                                      normalizer
                                      ])
    info_svhn_dataset = svhn.SVHN('data/svhn', split='test',
                                  transform=transformer, download=True)
    dataloader = torch.utils.data.DataLoader(
                info_svhn_dataset,
                batch_size=bs, 
                shuffle=False, 
                num_workers=1, 
                pin_memory=True)
    return dataloader

def dtd(normalizer,bs):
    import torchvision
    transformer = transforms.Compose([transforms.Resize(32),
                                      transforms.CenterCrop(32),#32*40 exist
                                      transforms.ToTensor(),
                                      normalizer
                                      ])
    info_dtd_dataset = torchvision.datasets.ImageFolder(root="data/dtd/images",
                                                        transform=transformer)
    dataloader = torch.utils.data.DataLoader(
                info_dtd_dataset,
                batch_size=bs, 
                shuffle=False, 
                num_workers=1, 
                pin_memory=True)
    return dataloader
def place365(normalizer,bs):
    import torchvision
    transformer = transforms.Compose([transforms.Resize(32),
                                      #transforms.CenterCrop(32),
                                      transforms.ToTensor(),
                                      normalizer
                                      ])
    info_place365_dataset = torchvision.datasets.ImageFolder(root="data/places365/test_subset",
                                                             transform=transformer)
    dataloader = torch.utils.data.DataLoader(
                info_place365_dataset,
                batch_size=bs, 
                shuffle=False, 
                num_workers=1, 
                pin_memory=True)
    return dataloader
def lsun(normalizer,bs):
    import torchvision
    transformer = transforms.Compose([transforms.Resize(32),
                                      #transforms.CenterCrop(32),
                                      transforms.ToTensor(),
                                      normalizer
                                      ])
    info_lsun_dataset = torchvision.datasets.ImageFolder("data/LSUN",
                                                         transform=transformer)
    dataloader = torch.utils.data.DataLoader(
                info_lsun_dataset,
                batch_size=bs, 
                shuffle=False, 
                num_workers=1, 
                pin_memory=True)
    return dataloader
def lsunR(normalizer,bs):
    import torchvision
    transformer = transforms.Compose([transforms.Resize(32),
                                      #transforms.CenterCrop(32),
                                      transforms.ToTensor(),
                                      normalizer
                                      ])
    info_lsunR_dataset = torchvision.datasets.ImageFolder("data/LSUN_resize",
                                                          transform=transformer)
    dataloader = torch.utils.data.DataLoader(
                info_lsunR_dataset,
                batch_size=bs, 
                shuffle=False, 
                num_workers=1, 
                pin_memory=True)
    return dataloader
def isun(normalizer,bs):
    import torchvision
    transformer = transforms.Compose([transforms.Resize(32),
                                      #transforms.CenterCrop(32),
                                      transforms.ToTensor(),
                                      normalizer
                                      ])
    info_isun_dataset = torchvision.datasets.ImageFolder("data/iSUN",
                                                         transform=transformer)
    dataloader = torch.utils.data.DataLoader(
                info_isun_dataset,
                batch_size=bs, 
                shuffle=False, 
                num_workers=1, 
                pin_memory=True)
    return dataloader
   
