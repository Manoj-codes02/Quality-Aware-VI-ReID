import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import os
import gc
import numpy as np
import json
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
torch.backends.cudnn.benchmark = True

from models.resnet import ReIDResNet
from models.discriminator import Discriminator
from enhancement.zero_dce import ZeroDCE
from enhancement.sigan import SIGAN
from quality_metrics.iqa import QualityAssessment
from evaluation.metrics import eval_sysu
from evaluation.ecn import ecn_reranking
from dataloaders.sysu_dataloader import get_dataloader
from dataloaders.sampler import CrossModalSampler
from evaluation.loss import TripletLoss, CrossEntropyLabelSmooth
from torch.utils.data import DataLoader

def normalize_batch(batch):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(batch.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(batch.device)
    return (batch - mean) / std

def get_device_config():
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"CUDA Enabled. Detected GPU Memory: {gpu_mem:.2f} GB")
        batch_size = 32 if gpu_mem > 8 else 16
        num_workers = min(4, os.cpu_count())
        pin_memory = True
        return torch.device('cuda'), batch_size, num_workers, pin_memory
    else:
        print("CUDA not available. Using CPU fallback. Warning: Training will be extremely slow.")
        return torch.device('cpu'), 8, 0, False

def train():
    device, batch_size, num_workers, pin_memory = get_device_config()
    
    # Check total identities in train mapping for Label Smooth
    num_train_classes = 171 # Extracted from the dataset split output
    model = ReIDResNet(num_classes=num_train_classes).to(device)
    zero_dce = ZeroDCE().to(device)
    sigan = SIGAN().to(device)
    discriminator = Discriminator().to(device)
    iqa = QualityAssessment(brisque_threshold=40.0)
    
    criterion_ce = CrossEntropyLabelSmooth(num_classes=num_train_classes)
    criterion_triplet = TripletLoss(margin=0.3)
    criterion_gan = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=5e-4)
    optimizer_ZeroDCE = optim.Adam(zero_dce.parameters(), lr=0.0001, weight_decay=1e-4)
    optimizer_SIGAN = optim.Adam(sigan.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Cosine Annealing scheduler with warmup could be added here; keeping simple Cosine
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())
    
    data_dir = 'dataset/SYSU-MM01'
    if not os.path.exists(data_dir):
        data_dir = 'dataset'
        
    print(f"Initializing DataLoader with batch={batch_size}, workers={num_workers}")
    train_dataset = get_dataloader(data_dir, mode='train')
    if train_dataset is None:
        return
    train_dataset = train_dataset.dataset
    
    num_instances = min(4, batch_size // 2)
    if num_instances % 2 != 0: num_instances -= 1
    
    sampler = CrossModalSampler(train_dataset, batch_size=batch_size, num_instances=num_instances)
    loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory)
    
    epochs = 100
    best_mAP = 0.0
    metrics_log = []
    os.makedirs('checkpoints', exist_ok=True)
    
    model.train()
    zero_dce.train()
    sigan.train()
    discriminator.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (imgs, labels, cams, _) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            is_ir = (cams == 3) | (cams == 6)
            
            enhanced_imgs = imgs.clone().detach()
            gan_loss = torch.tensor(0.0).to(device)
            zdce_loss = torch.tensor(0.0).to(device)
            
            if is_ir.any():
                ir_imgs = imgs[is_ir].clone()
                ir_imgs_eval = ir_imgs.clone().detach()
                
                with torch.no_grad():
                    needs_sigan = iqa.evaluate(ir_imgs_eval) 
                
                sigan_indices = torch.where(needs_sigan)[0]
                zdce_indices = torch.where(~needs_sigan)[0]
                
                # Zero-DCE
                if len(zdce_indices) > 0:
                    optimizer_ZeroDCE.zero_grad()
                    out_zdce = zero_dce(ir_imgs[zdce_indices])
                    ir_imgs_eval[zdce_indices] = out_zdce.detach()
                    zdce_loss = torch.mean(torch.abs(out_zdce - ir_imgs[zdce_indices].detach()))
                    if zdce_loss.requires_grad:
                        zdce_loss.backward()
                        optimizer_ZeroDCE.step()
                
                # SIGAN
                if len(sigan_indices) > 0:
                    out_sigan = sigan(ir_imgs[sigan_indices])
                    ir_imgs_eval[sigan_indices] = out_sigan.detach()
                    
                    optimizer_D.zero_grad()
                    real_imgs = imgs[~is_ir][:len(sigan_indices)] if (~is_ir).any() else out_sigan.detach()
                    if len(real_imgs) == 0: real_imgs = out_sigan.detach()
                    
                    pred_real = discriminator(real_imgs.detach())
                    loss_D_real = criterion_gan(pred_real, torch.ones_like(pred_real))
                    pred_fake = discriminator(out_sigan.detach())
                    loss_D_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
                    
                    loss_D = (loss_D_real + loss_D_fake) * 0.5
                    loss_D.backward()
                    optimizer_D.step()
                    
                    optimizer_SIGAN.zero_grad()
                    pred_fake_G = discriminator(out_sigan)
                    gan_loss = criterion_gan(pred_fake_G, torch.ones_like(pred_fake_G))
                    gan_loss.backward()
                    optimizer_SIGAN.step()
                
                enhanced_imgs[is_ir] = ir_imgs_eval
                
            input_imgs = normalize_batch(enhanced_imgs)
            optimizer.zero_grad()
            
            with autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(input_imgs)

                if isinstance(outputs, tuple):
                    cls_score = outputs[0]
                    features = outputs[1]
                else:
                    cls_score = outputs
                    features = outputs

                    
                loss_ce = criterion_ce(cls_score, labels)
                loss_triplet = criterion_triplet(features, labels)
                loss_reid = loss_ce + loss_triplet
                
            scaler.scale(loss_reid).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss_reid.item()

                
        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}] Avg Loss: {avg_loss:.4f}")

        torch.cuda.empty_cache()
        gc.collect()
        
        # Periodic Eval and Checkpointing
        print(f"--- Epoch {epoch+1} Evaluation ---")
        cmc, mAP = evaluate(model, zero_dce, sigan, iqa, data_dir, device)
        
        if cmc is not None and mAP is not None:
            metrics_log.append({"epoch": epoch+1, "Rank-1": float(cmc[0]), "mAP": float(mAP)})
            if mAP > best_mAP:
                best_mAP = mAP

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_mAP': best_mAP
                }, 'checkpoints/best_reid_model.pth')

                print(">> Saved best checkpoint!")
        
    with open('checkpoints/metrics_log.json', 'w') as f:
        json.dump(metrics_log, f)

def evaluate(model, zero_dce, sigan, iqa, data_dir, device):
    model.eval()
    zero_dce.eval()
    sigan.eval()
    
    q_loader = get_dataloader(data_dir, batch_size=32, mode='query')
    g_loader = get_dataloader(data_dir, batch_size=32, mode='gallery')
    
    if not q_loader or not g_loader:
        print("Query/Gallery set missing. Skipping eval.")
        return None, None
        
    def extract_features(loader, is_query=False):
        features = []
        pids = []
        camids = []
        psnrs = []
        ssims = []
        saved_zdce_imgs = None
        saved_sigan_imgs = None
        
        with torch.no_grad():
            for imgs, labels, cams, _ in loader:
                imgs = imgs.to(device, non_blocking=True)
                if is_query:
                    needs_sigan = iqa.evaluate(imgs.cpu()).to(device)

                    for i in range(len(imgs)):
                        img_single = imgs[i:i+1]
                        orig_np = img_single.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                        
                        if needs_sigan[i]:
                            out = sigan(img_single)
                            if saved_sigan_imgs is None:
                                saved_sigan_imgs = (orig_np.copy(), out.squeeze(0).cpu().numpy().transpose(1, 2, 0).copy(), labels[i].item())
                        else:
                            out = zero_dce(img_single)
                            if saved_zdce_imgs is None:
                                saved_zdce_imgs = (orig_np.copy(), out.squeeze(0).cpu().numpy().transpose(1, 2, 0).copy(), labels[i].item())
                                
                        imgs[i] = out.squeeze(0)
                        enh_np = out.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                        
                        orig_norm = np.clip(orig_np, 0, 1) if orig_np.max() <= 1.0 else np.clip(orig_np/255.0, 0, 1)
                        enh_norm = np.clip(enh_np, 0, 1) if enh_np.max() <= 1.0 else np.clip(enh_np/255.0, 0, 1)
                        
                        psnr = peak_signal_noise_ratio(orig_norm, enh_norm, data_range=1.0)
                        ssim = structural_similarity(orig_norm, enh_norm, data_range=1.0, channel_axis=2)
                        psnrs.append(psnr)
                        ssims.append(ssim)

                input_imgs = normalize_batch(imgs)
                outputs = model(input_imgs)

                if isinstance(outputs, tuple):
                    feat = outputs[1]
                else:
                    feat = outputs

                feat = torch.nn.functional.normalize(feat, p=2, dim=1)
                features.append(feat.cpu())
                pids.extend(labels.numpy())
                camids.extend(cams.numpy())
                
        avg_psnr = np.mean(psnrs) if len(psnrs) > 0 else 0.0
        avg_ssim = np.mean(ssims) if len(ssims) > 0 else 0.0
        if len(features) == 0: return None, None, None, 0.0, 0.0, None, None
        return torch.cat(features, dim=0), np.array(pids), np.array(camids), avg_psnr, avg_ssim, saved_zdce_imgs, saved_sigan_imgs
        
    q_feat, q_pids, q_camids, avg_psnr, avg_ssim, saved_zdce, saved_sigan = extract_features(q_loader, is_query=True)
    g_feat, g_pids, g_camids, _, _, _, _ = extract_features(g_loader, is_query=False)
    
    if q_feat is None or g_feat is None:
        return None, None
    
    distmat = ecn_reranking(q_feat.detach(), g_feat.detach())
    cmc, mAP = eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids)
    
    print(f"Results -> Rank-1: {cmc[0]:.2%}, Rank-5: {cmc[4]:.2%}, Rank-10: {cmc[9]:.2%}, mAP: {mAP:.2%}")
    print(f"Metrics -> PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
    
    os.makedirs('results/figures', exist_ok=True)
    
    def generate_fig(saved_data, fig_name, title_prefix):
        if saved_data is None: return
        orig_np, enh_np, q_pid = saved_data
        
        q_idx = np.where(q_pids == q_pid)[0][0]
        indices = np.argsort(distmat[q_idx])
        top_10_idx = indices[:10]
        top_10_g_pids = g_pids[top_10_idx]
        top_10_dists = distmat[q_idx][top_10_idx]
        
        fig, axes = plt.subplots(1, 12, figsize=(24, 4))
        
        ax = axes[0]
        orig_disp = np.clip(orig_np, 0, 1) if orig_np.max() <= 1.0 else orig_np.astype(np.uint8)
        ax.imshow(orig_disp)
        ax.set_title("Original IR")
        ax.axis('off')
        
        ax = axes[1]
        enh_disp = np.clip(enh_np, 0, 1) if enh_np.max() <= 1.0 else enh_np.astype(np.uint8)
        ax.imshow(enh_disp)
        ax.set_title(title_prefix)
        ax.axis('off')
        
        for i, idx in enumerate(top_10_idx):
            ax = axes[i+2]
            g_img_path = g_loader.dataset.images[idx]
            g_img = Image.open(g_img_path).convert('RGB') if 'Image' in globals() else plt.imread(g_img_path)
            ax.imshow(g_img)
            
            sim = 1 - top_10_dists[i]
            is_correct = top_10_g_pids[i] == q_pid
            color = 'green' if is_correct else 'red'
            ax.set_title(f"Sim: {sim:.2f}", color=color)
            
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
            ax.set_xticks([])
            ax.set_yticks([])
            
        plt.tight_layout()
        plt.savefig(f'results/figures/{fig_name}')
        plt.close()
        
    generate_fig(saved_zdce, "Figure_3.png", "Enhanced (Zero-DCE)")
    generate_fig(saved_sigan, "Figure_4.png", "Refined (SIGAN)")
    
    ranks = np.arange(1, 21)
    plt.figure(figsize=(8, 6))
    plt.plot(ranks, cmc[:20], marker='o', label='Proposed Method')
    plt.xlabel('Rank')
    plt.ylabel('Matching Rate')
    plt.title('CMC Curve on SYSU-MM01')
    plt.grid(True)
    plt.legend()
    plt.savefig('results/figures/cmc_curve.png', dpi=300)
    plt.close()
    
    metrics = {
        "Rank1": float(cmc[0]),
        "Rank5": float(cmc[4]),
        "Rank10": float(cmc[9]),
        "mAP": float(mAP),
        "PSNR": float(avg_psnr),
        "SSIM": float(avg_ssim)
    }
    with open('results/final_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    return cmc, mAP

if __name__ == '__main__':
    train()
