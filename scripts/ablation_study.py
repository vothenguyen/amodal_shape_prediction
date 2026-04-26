import json
import argparse
import torch
import torch.nn as nn
import albumentations as A
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import AmodalSwinUNet
from dataset import AmodalDataset


class AmodalSwinUNetNoAttention(nn.Module):
    def __init__(self, num_classes=91):
        super().__init__()
        self.base = AmodalSwinUNet(num_classes=num_classes)
        self.base.spatial_attention = nn.Identity()

    def forward(self, x, class_ids):
        return self.base(x, class_ids)


def calculate_iou(pred_logits, target, threshold=0.5):
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou


def eval_model(model, loader, device, checkpoint_path):
    # Load đúng vào base nếu là NoAttention
    target_model = model.base if isinstance(model, AmodalSwinUNetNoAttention) else model
    target_model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    model.eval()

    total_iou = 0
    with torch.no_grad():
        for inputs, targets, _, class_ids in tqdm(loader, desc='Evaluating'):
            inputs = inputs.to(device)
            targets = targets.unsqueeze(1).float().to(device)
            class_ids = class_ids.to(device)
            outputs = model(inputs, class_ids)
            iou = calculate_iou(outputs, targets)
            total_iou += iou.sum().item()
    return total_iou / len(loader.dataset)


def ablation_study(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = A.Compose([A.Resize(224, 224)])
    dataset = AmodalDataset(img_dir=args.img_dir, ann_file=args.ann_file, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print('Ablation Study: Spatial Attention')

    print('WITH attention...')
    model_with = AmodalSwinUNet(num_classes=91).to(device)
    iou_with = eval_model(model_with, loader, device, args.checkpoint)
    print(f'mIoU = {iou_with * 100:.2f}%')

    print('WITHOUT attention...')
    model_without = AmodalSwinUNetNoAttention(num_classes=91).to(device)
    iou_without = eval_model(model_without, loader, device, args.checkpoint)
    print(f'mIoU = {iou_without * 100:.2f}%')

    improvement = ((iou_with - iou_without) / iou_without) * 100 if iou_without > 0 else 0
    print(f'Improvement: {improvement:+.2f}%')

    results = {
        'with_attention': iou_with,
        'without_attention': iou_without,
        'improvement_percent': improvement
    }
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', type=str, default='../data/val2014')
    parser.add_argument('--ann-file', type=str, default='../data/annotations/COCO_amodal_val2014.json')
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/swin_amodal_epoch_30.pth')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--output', type=str, default='')
    args = parser.parse_args()
    ablation_study(args)