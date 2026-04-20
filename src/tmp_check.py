import torch
import albumentations as A
from dataset import AmodalDataset
from model import AmodalSwinUNet

ds = AmodalDataset('../data/train2014','../data/annotations/COCO_amodal_train2014.json',transform=A.Compose([A.Resize(224,224)]))
print('len', len(ds))
inp,am,oc,cls = ds[0]
print('cls', int(cls), 'vis sum', int(inp[3].sum().item()), 'true sum', int(am.sum().item()))
model=AmodalSwinUNet().cpu()
ckpt=torch.load('../checkpoints/swin_amodal_epoch_30.pth',map_location='cpu')
print('ckpt type', type(ckpt), 'len', len(ckpt))
model.load_state_dict(ckpt)
print('loaded state')
out=model(inp.unsqueeze(0), torch.tensor([cls]))
print('out shape', out.shape, 'min', out.min().item(), 'max', out.max().item())
pm=(torch.sigmoid(out)>0.5).squeeze().numpy()
print('pred sum', int(pm.sum()))
