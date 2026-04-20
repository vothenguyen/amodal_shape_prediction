import json
import argparse
import matplotlib.pyplot as plt
import torch
import albumentations as A

from model import AmodalSwinUNet
from dataset import AmodalDataset


def qualitative_eval(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(args.eval_results, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    per_sample_metrics = results['per_sample_metrics']
    indexed_metrics = [(i, m['iou']) for i, m in enumerate(per_sample_metrics)]
    indexed_metrics.sort(key=lambda x: x[1], reverse=True)
    
    top_k = args.top_k
    top_indices = [idx for idx, _ in indexed_metrics[:top_k]]
    
    print(f'🎨 Trích xuất {top_k} ảnh tốt nhất (highest IoU)...')
    
    transform = A.Compose([A.Resize(224, 224)])
    dataset = AmodalDataset(img_dir=args.img_dir, ann_file=args.ann_file, transform=transform)
    
    model = AmodalSwinUNet(num_classes=91).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    fig, axes = plt.subplots(top_k, 3, figsize=(15, 5 * top_k))
    if top_k == 1:
        axes = axes.reshape(1, -1)
    
    for plot_idx, sample_idx in enumerate(top_indices):
        input_tensor, target_mask, occluded, class_id = dataset[sample_idx]
        input_batch = input_tensor.unsqueeze(0).to(device)
        class_id_batch = torch.tensor([class_id]).to(device)
        
        with torch.no_grad():
            output_logits = model(input_batch, class_id_batch)
            pred_mask = torch.sigmoid(output_logits)
            pred_mask = (pred_mask > 0.5).squeeze().cpu().numpy()
        
        img_rgb = input_tensor[:3].numpy().transpose(1, 2, 0)
        truth_mask = target_mask.numpy()
        iou_score = per_sample_metrics[sample_idx]['iou']
        
        axes[plot_idx, 0].imshow(img_rgb)
        axes[plot_idx, 0].set_title('Original Image', fontsize=10)
        axes[plot_idx, 0].axis('off')
        
        axes[plot_idx, 1].imshow(truth_mask, cmap='gray')
        axes[plot_idx, 1].set_title('Ground Truth', fontsize=10)
        axes[plot_idx, 1].axis('off')
        
        axes[plot_idx, 2].imshow(pred_mask, cmap='gray')
        axes[plot_idx, 2].set_title(f'Pred (IoU: {iou_score*100:.1f}%)', fontsize=10)
        axes[plot_idx, 2].axis('off')
    
    plt.tight_layout()
    output_path = args.output or 'qualitative_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'✅ Đã lưu {output_path}')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-results', type=str, required=True)
    parser.add_argument('--img-dir', type=str, default='../data/val2014')
    parser.add_argument('--ann-file', type=str, default='../data/annotations/COCO_amodal_val2014.json')
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/swin_amodal_epoch_30.pth')
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument('--output', type=str, default='')
    args = parser.parse_args()
    qualitative_eval(args)
