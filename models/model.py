import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.layers.weight_init import trunc_normal_
import timm
import numpy as np
from timm.models.layers import to_2tuple
import random
from sklearn.cluster import KMeans


# Override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size[0] == 0 and W % self.patch_size[1] == 0, \
            f"Input size ({H}x{W}) should be divisible by patch size {self.patch_size}"
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class CompassModel(nn.Module):
    def __init__(self, label_dim=527,
                 fshape=128, tshape=2, fstride=128, tstride=2,
                 input_fdim=128, input_tdim=300, model_size='base',
                 pretrain_stage=True, load_pretrained_mdl_path=None):

        super(CompassModel, self).__init__()

        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        self.model_size = model_size
        self.pretrain_stage = pretrain_stage
        self.input_fdim = input_fdim
        self.input_tdim = input_tdim
        self.fshape = fshape
        self.tshape = tshape

        if pretrain_stage == True:
            if load_pretrained_mdl_path is not None:
                raise ValueError('Setting load_pretrained_mdl_path at pretraining stage is useless, '
                                 'pretraining is always from scratch, please change it to None.')
            if fstride != fshape or tstride != tshape:
                raise ValueError('fstride != fshape or tstride != tshape, they must be equal at the '
                                  'pretraining stage, patch split overlapping is not supported.')

            def create_model_safe(model_names_list):
                for model_name in model_names_list:
                    try:
                        return timm.create_model(model_name, pretrained=False)
                    except RuntimeError:
                        continue
                raise RuntimeError(f'None of the model names {model_names_list} are available in current timm version')

            if model_size == 'tiny':
                self.v = create_model_safe(['deit_tiny_distilled_patch16_224', 'vit_deit_tiny_distilled_patch16_224'])
                self.heads, self.depth = 3, 12
                self.cls_token_num = 2
            elif model_size == 'small':
                self.v = create_model_safe(['deit_small_distilled_patch16_224', 'vit_deit_small_distilled_patch16_224'])
                self.heads, self.depth = 6, 12
                self.cls_token_num = 2
            elif model_size == 'base':
                self.v = create_model_safe(['deit_base_distilled_patch16_384', 'vit_deit_base_distilled_patch16_384'])
                self.heads, self.depth = 16, 16
                self.cls_token_num = 2
            elif model_size == 'base_nokd':
                self.v = create_model_safe(['deit_base_patch16_384', 'vit_deit_base_patch16_384'])
                self.heads, self.depth = 12, 12
                self.cls_token_num = 1
            else:
                raise Exception('Model size must be one of tiny, small, base, base_nokd')

            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]

            self.softmax = nn.Softmax(dim=-1)
            self.lsoftmax = nn.LogSoftmax(dim=-1)
            self.fshape, self.tshape = fshape, tshape
            self.fstride, self.tstride = fstride, tstride
            self.input_fdim, self.input_tdim = input_fdim, input_tdim
            self.p_input_fdim = nn.Parameter(torch.tensor(input_fdim), requires_grad=False)
            self.p_input_tdim = nn.Parameter(torch.tensor(input_tdim), requires_grad=False)

            self.p_f_dim, self.p_t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
            num_patches = self.p_f_dim * self.p_t_dim
            self.num_patches = num_patches

            print('pretraining patch split stride: frequency={:d}, time={:d}'.format(fstride, tstride))
            print('pretraining patch shape: frequency={:d}, time={:d}'.format(fshape, tshape))
            print('pretraining patch array dimension: frequency={:d}, time={:d}'.format(self.p_f_dim, self.p_t_dim))
            print('pretraining number of patches={:d}'.format(num_patches))

            audio_patch_embed = PatchEmbed(
                img_size=(input_fdim, input_tdim),
                patch_size=(fshape, tshape),
                in_chans=1,
                embed_dim=self.original_embedding_dim
            )

            if hasattr(self.v.patch_embed, 'proj') and hasattr(self.v.patch_embed.proj, 'weight'):
                original_weight = self.v.patch_embed.proj.weight
                new_weight = torch.mean(original_weight, dim=1, keepdim=True)
                if new_weight.shape[2:] == audio_patch_embed.proj.weight.shape[2:]:
                    audio_patch_embed.proj.weight.data.copy_(new_weight)
                else:
                    trunc_normal_(audio_patch_embed.proj.weight, std=.02)
                if hasattr(self.v.patch_embed.proj, 'bias') and self.v.patch_embed.proj.bias is not None:
                    audio_patch_embed.proj.bias.data.copy_(self.v.patch_embed.proj.bias.data)
            else:
                trunc_normal_(audio_patch_embed.proj.weight, std=.02)

            self.v.patch_embed = audio_patch_embed
            self.v.patch_embed.num_patches = num_patches

            new_pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.cls_token_num, self.original_embedding_dim))
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=.02)

            # Shared components
            self.unfold = torch.nn.Unfold(kernel_size=(fshape, tshape), stride=(fstride, tstride))
            self.mask_embed = nn.Parameter(torch.zeros([1, 1, self.original_embedding_dim]))
            self.mask_embed = torch.nn.init.xavier_normal_(self.mask_embed)

            # PoST head: predicts original position (absolute position classification)
            self.post_position_predictor = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim),
                nn.Linear(self.original_embedding_dim, self.num_patches)
            )

            # CoRe reconstruction head: H_M -> linear -> X_hat_M
            self.core_recon_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim),
                nn.Linear(self.original_embedding_dim, self.fshape * self.tshape)
            )

        elif pretrain_stage == False:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if load_pretrained_mdl_path is None:
                raise ValueError('Please set load_pretrained_mdl_path to load a pretrained model.')
            sd = torch.load(load_pretrained_mdl_path, map_location=device)
            try:
                p_fshape = sd['module.v.patch_embed.proj.weight'].shape[2]
                p_tshape = sd['module.v.patch_embed.proj.weight'].shape[3]
                p_input_fdim = sd['module.p_input_fdim'].item()
                p_input_tdim = sd['module.p_input_tdim'].item()
            except:
                raise ValueError('The model loaded is not from a torch.nn.DataParallel object. '
                                  'Wrap it with torch.nn.DataParallel and try again.')

            print('now load a SSL pretrained model from ' + load_pretrained_mdl_path)
            audio_model = CompassModel(fstride=p_fshape, tstride=p_tshape, fshape=p_fshape, tshape=p_tshape,
                                   input_fdim=p_input_fdim, input_tdim=p_input_tdim,
                                   pretrain_stage=True, model_size=model_size)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)

            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.cls_token_num = audio_model.module.cls_token_num

            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim),
                nn.Linear(self.original_embedding_dim, label_dim)
            )

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
            p_f_dim, p_t_dim = audio_model.module.p_f_dim, audio_model.module.p_t_dim
            num_patches = f_dim * t_dim
            p_num_patches = p_f_dim * p_t_dim
            self.v.patch_embed.num_patches = num_patches

            print('fine-tuning patch split stride: frequency={:d}, time={:d}'.format(fstride, tstride))
            print('fine-tuning number of patches={:d}'.format(num_patches))

            if fshape != p_fshape or tshape != p_tshape:
                raise ValueError('Patch shape mismatch between pretraining and fine-tuning.')

            if fstride != p_fshape or tstride != p_tshape:
                new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape),
                                           stride=(fstride, tstride))
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
                self.v.patch_embed.proj = new_proj

            new_pos_embed = self.v.pos_embed[:, self.cls_token_num:, :].detach().reshape(
                1, p_num_patches, self.original_embedding_dim)
            new_pos_embed = new_pos_embed.transpose(1, 2).reshape(
                1, self.original_embedding_dim, p_f_dim, p_t_dim)

            if t_dim < p_t_dim:
                new_pos_embed = new_pos_embed[:, :, :,
                                int(p_t_dim / 2) - int(t_dim / 2): int(p_t_dim / 2) - int(t_dim / 2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(p_f_dim, t_dim), mode='bilinear')
            if f_dim < p_f_dim:
                new_pos_embed = new_pos_embed[:, :,
                                int(p_f_dim / 2) - int(f_dim / 2): int(p_f_dim / 2) - int(f_dim / 2) + f_dim, :]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')

            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(
                torch.cat([self.v.pos_embed[:, :self.cls_token_num, :].detach(), new_pos_embed], dim=1))

    def get_shape(self, fstride, tstride, input_fdim, input_tdim, fshape, tshape):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def gen_maskid_patch(self, sequence_len=512, mask_size=100, cluster=16, input_patches=None):
        """Structure-Preserving Masking Strategy (SPMS): K-means with fixed K=16 on patch embeddings,
        then masking entire clusters so visible regions remain acoustically coherent."""
        from random import randrange

        mask_id = []

        if input_patches is not None and input_patches.size(0) > 0:
            batch_size, seq_len, embed_dim = input_patches.shape
            patches_for_clustering = input_patches[0].detach().cpu().numpy()

            # Fixed K=16 per paper (SPMS)
            n_clusters = 16
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(patches_for_clustering)

            unique_labels = np.unique(cluster_labels)
            selected_clusters = np.random.choice(
                unique_labels,
                size=min(len(unique_labels), n_clusters // 2),
                replace=False
            )

            for cluster_id in selected_clusters:
                cluster_patches = np.where(cluster_labels == cluster_id)[0]
                mask_id.extend(cluster_patches.tolist())

            if len(mask_id) < mask_size:
                remaining_patches = [i for i in range(sequence_len) if i not in mask_id]
                additional_masks = np.random.choice(
                    remaining_patches,
                    size=min(mask_size - len(mask_id), len(remaining_patches)),
                    replace=False
                )
                mask_id.extend(additional_masks.tolist())

            mask_id = list(set(mask_id))[:mask_size]

        else:
            # Fallback: random contiguous block masking when patch embeddings are unavailable
            cur_clus = randrange(cluster) + 3
            while len(list(set(mask_id))) <= mask_size:
                start_id = randrange(sequence_len)
                cur_mask = []
                for i in range(0, cur_clus):
                    for j in range(0, cur_clus):
                        mask_cand = start_id + self.p_t_dim * i + j
                        if 0 < mask_cand < sequence_len:
                            cur_mask.append(mask_cand)
                mask_id = mask_id + cur_mask
            mask_id = list(set(mask_id))[:mask_size]

        if input_patches is not None:
            device = input_patches.device
        elif hasattr(self, 'mask_embed') and self.mask_embed is not None:
            device = self.mask_embed.device
        else:
            device = 'cpu'

        return torch.tensor(mask_id, device=device)

    def gen_maskid_frame(self, sequence_len=512, mask_size=100):
        mask_id = random.sample(range(0, sequence_len), mask_size)
        return torch.tensor(mask_id)

    def finetuningavgtok(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = torch.mean(x[:, self.cls_token_num:, :], dim=1)
        x = self.mlp_head(x)
        return x

    def finetuningcls(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        if self.cls_token_num == 2:
            cls_feat = (x[:, 0] + x[:, 1]) / 2
        else:
            cls_feat = x[:, 0]
        return self.mlp_head(cls_feat)

    def post(self, x, mask_patch, cluster, show_mask=False):
        """PoST: Position-aware masked prediction.
        Masked patches are replaced with mask_embed token before Transformer.
        Predicts original absolute positions via cross-entropy loss."""
        input_unfolded = self.unfold(x).transpose(1, 2)
        B = x.shape[0]
        x_embed = self.v.patch_embed(x)  # [B, N, D]

        mask_index = torch.empty((B, mask_patch), device=x.device, requires_grad=False).long()
        for i in range(B):
            if cluster:
                mask_index[i] = self.gen_maskid_patch(self.num_patches, mask_patch,
                                                       input_patches=x_embed)
            else:
                mask_index[i] = self.gen_maskid_frame(self.num_patches, mask_patch).to(x.device)

        # Replace masked patch embeddings with mask_embed token (paper: mask tokens before Transformer)
        x_masked = x_embed.clone()
        for i in range(B):
            x_masked[i, mask_index[i], :] = self.mask_embed

        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x_tokens = torch.cat((cls_tokens, dist_token, x_masked), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x_tokens = torch.cat((cls_tokens, x_masked), dim=1)
        x_tokens = x_tokens + self.v.pos_embed
        x_tokens = self.v.pos_drop(x_tokens)
        for blk in self.v.blocks:
            x_tokens = blk(x_tokens)
        x_tokens = self.v.norm(x_tokens)

        # H_M: hidden states at masked positions
        patch_tokens = x_tokens[:, self.cls_token_num:, :]  # [B, N, D]

        total_loss = torch.tensor(0.0, device=x.device)
        total_correct = torch.tensor(0.0, device=x.device)

        for i in range(B):
            masked_feat = patch_tokens[i, mask_index[i], :]   # [M, D]
            logits = self.post_position_predictor(masked_feat) # [M, N]
            targets = mask_index[i]                            # ground-truth absolute positions
            loss_i = F.cross_entropy(logits, targets, reduction='sum')
            total_loss += loss_i
            pred_idx = torch.argmax(logits, dim=-1)
            total_correct += torch.sum(pred_idx == targets)

        avg_loss = total_loss / (B * mask_patch)
        acc = total_correct.float() / (B * mask_patch)

        if not show_mask:
            return acc, avg_loss
        else:
            pred_vis = input_unfolded.clone()
            masked = input_unfolded.clone()
            real_pred = input_unfolded.clone()
            for i in range(B):
                masked[i, mask_index[i], :] = 99.0
                masked_feat = patch_tokens[i, mask_index[i], :]
                logits = self.post_position_predictor(masked_feat)
                probs = F.softmax(logits, dim=-1)
                true_pos_probs = probs[torch.arange(mask_patch, device=x.device), mask_index[i]]
                scores = (true_pos_probs * 99.0).unsqueeze(1).expand(-1, self.fshape * self.tshape)
                pred_vis[i, mask_index[i], :] = scores
                pred_idx = torch.argmax(logits, dim=-1)
                pred_idx_clamped = torch.clamp(pred_idx, 0, self.num_patches - 1)
                real_pred[i, mask_index[i], :] = input_unfolded[i, pred_idx_clamped, :]

            fold = torch.nn.Fold(
                output_size=([self.input_fdim, self.input_tdim]),
                kernel_size=(self.fshape, self.tshape),
                stride=(self.fstride, self.tstride)
            )
            return (
                fold(real_pred.transpose(1, 2)).squeeze(1).transpose(1, 2).detach(),
                fold(pred_vis.transpose(1, 2)).squeeze(1).transpose(1, 2).detach(),
                fold(masked.transpose(1, 2)).squeeze(1).transpose(1, 2).detach()
            )

    def core(self, x, mask_patch, cluster, show_mask=False):
        """CoRe: Context-driven spectrogram reconstruction.
        Masked patches replaced with mask_embed, full sequence through Transformer.
        H_M -> core (linear) -> X_hat_M, MSE loss."""
        input_unfolded = self.unfold(x).transpose(1, 2)  # [B, N, P]
        B = x.shape[0]
        x_embed = self.v.patch_embed(x)  # [B, N, D]

        mask_index = torch.empty((B, mask_patch), device=x.device, requires_grad=False).long()
        for i in range(B):
            mask_index[i] = (self.gen_maskid_patch(self.num_patches, mask_patch, input_patches=x_embed)
                             if cluster
                             else self.gen_maskid_frame(self.num_patches, mask_patch).to(x.device))

        # Replace masked patch embeddings with mask_embed token (same as PoST)
        x_masked = x_embed.clone()
        for i in range(B):
            x_masked[i, mask_index[i], :] = self.mask_embed

        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            tokens = torch.cat((cls_tokens, dist_token, x_masked), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            tokens = torch.cat((cls_tokens, x_masked), dim=1)
        tokens = tokens + self.v.pos_embed
        tokens = self.v.pos_drop(tokens)
        for blk in self.v.blocks:
            tokens = blk(tokens)
        tokens = self.v.norm(tokens)

        # H_M: hidden states at masked positions -> reconstruct X_M
        patch_tokens = tokens[:, self.cls_token_num:, :]  # [B, N, D]

        pred = torch.empty((B, mask_patch, self.fshape * self.tshape), device=x.device).float()
        target = torch.empty((B, mask_patch, self.fshape * self.tshape), device=x.device).float()
        for i in range(B):
            h_m = patch_tokens[i, mask_index[i], :]              # [M, D]
            pred[i] = self.core_recon_head(h_m)         # [M, P]
            target[i] = input_unfolded[i, mask_index[i], :]

        # L_CoRe = (1/M) * ||X_hat_M - X_M||^2_F
        recon_loss = torch.mean((pred - target) ** 2)

        if not show_mask:
            return recon_loss
        else:
            pred_vis = input_unfolded.clone()
            masked = input_unfolded.clone()
            real_pred = input_unfolded.clone()
            for i in range(B):
                real_pred[i, mask_index[i], :] = pred[i]
                masked[i, mask_index[i], :] = 99.0
                errors = torch.mean((pred[i] - target[i]) ** 2, dim=1)
                max_error = torch.max(errors).item() + 1e-8
                scores = 99.0 * (1.0 - torch.clamp(errors / max_error, 0.0, 1.0))
                pred_vis[i, mask_index[i], :] = scores.unsqueeze(1).expand(-1, self.fshape * self.tshape)

            fold = torch.nn.Fold(
                output_size=([self.input_fdim, self.input_tdim]),
                kernel_size=(self.fshape, self.tshape),
                stride=(self.fstride, self.tstride)
            )
            return (
                fold(real_pred.transpose(1, 2)).squeeze(1).transpose(1, 2).detach(),
                fold(pred_vis.transpose(1, 2)).squeeze(1).transpose(1, 2).detach(),
                fold(masked.transpose(1, 2)).squeeze(1).transpose(1, 2).detach()
            )

    def compass_joint_training(self, x, mask_patch, cluster, show_mask=False):
        """ComPASS joint pre-training: L_ssl = lambda_p * L_PoST + lambda_r * L_CoRe."""
        post_acc, post_loss = self.post(x, mask_patch, cluster, show_mask=False)
        core_loss = self.core(x, mask_patch, cluster, show_mask=False)
        total_loss = post_loss + core_loss

        if show_mask:
            return self.post(x, mask_patch, cluster, show_mask=True)
        else:
            return post_acc, total_loss

    def forward(self, x, task, cluster=True, mask_patch=400, **kwargs):
        # Input: (batch_size, time_frame_num, frequency_bins)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        if task == 'ft_avgtok':
            return self.finetuningavgtok(x)
        elif task == 'ft_cls':
            return self.finetuningcls(x)
        elif task == 'pretrain_post':
            return self.post(x, mask_patch=mask_patch, cluster=cluster)
        elif task == 'pretrain_core':
            return self.core(x, mask_patch=mask_patch, cluster=cluster)
        elif task == 'pretrain_compass':
            return self.compass_joint_training(x, mask_patch=mask_patch, cluster=cluster)
        elif task == 'visualize_post':
            return self.post(x, mask_patch=mask_patch, cluster=cluster, show_mask=True)
        elif task == 'visualize_core':
            return self.core(x, mask_patch=mask_patch, cluster=cluster, show_mask=True)
        elif task == 'visualize_compass':
            return self.compass_joint_training(x, mask_patch=mask_patch, cluster=cluster, show_mask=True)
        else:
            raise Exception('Task unrecognized.')
