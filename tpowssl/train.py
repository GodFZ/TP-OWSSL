from typing import Type, Dict, Tuple, Optional
from collections import defaultdict
import os
import math
import argparse

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR
from clip.clip import _transform
from timm.utils import accuracy

import tpowssl.lib as lib
import tpowssl.vlprompt.tools as vlp_tools
import tpowssl.datasets.tools as dts_tools
from tpowssl.datasets import return_train_val_datasets, return_ood_loaders, return_domains_loaders
from tpowssl.vlprompt import TPOWSSL
from tpowssl.vlprompt.tools import GlobalLocalLoss
from tpowssl.datasets.tools.create_balanced_subset import create_open_world_few_shots_dataset
from transformers import CLIPProcessor, CLIPModel
import json
import open_clip
from clipn_build import get_clipn_classifier
from open_clip import create_model_and_transforms
import torch.nn.functional as F

from simgcd.data.get_datasets import get_datasets, get_class_splits

NoneType = Type[None]

def cluster_acc(y_pred, y_true):

    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size

def traditional_accuracy(output, target):
    num_correct = np.sum(output == target)
    res = num_correct / len(target)
    return res


def CLIPN_pred(args, images, class_names, clip_pred, clipn_classifier):

    probs_no = clipn_classifier(images)
    pred = torch.argmax(probs_no, dim=1)

    PATH_TO_PROMPTS = f'../gpt3_prompts/CuPL_prompts_{args.dataset_name}.json'
    with open(PATH_TO_PROMPTS) as f:
        gpt3_prompts = json.load(f)
    class_origin_order = []
    for item in gpt3_prompts.items():
        class_origin_order.append(item[0])

    new_to_index_map = {class_names[i]: i for i in range(len(class_names))}
    converted_pred = torch.tensor([new_to_index_map[class_origin_order[old_index]] for old_index in pred]).cuda(non_blocking=True)
    result = torch.where(converted_pred == clip_pred, 0, 1)

    return result


def CLIPN_pred_v2(args, images, class_names, clip_pred, clip_maxprob, clipn_classifier):

    probs_no = clipn_classifier(images)
    pred = torch.argmax(probs_no, dim=1)

    PATH_TO_PROMPTS = f'../gpt3_prompts/CuPL_prompts_{args.dataset_name}.json'
    with open(PATH_TO_PROMPTS) as f:
        gpt3_prompts = json.load(f)
    class_origin_order = []
    for item in gpt3_prompts.items():
        class_origin_order.append(item[0])

    new_to_index_map = {class_origin_order[i]: i for i in range(len(class_origin_order))}
    converted_clip_pred = torch.tensor([new_to_index_map[class_names[old_index]] for old_index in clip_pred])
    correspond_clipn_probs = probs_no[torch.arange(pred.shape[0]), converted_clip_pred]
    result = torch.where(correspond_clipn_probs > clip_maxprob, 0, 1)

    return result

def CLIP_pred(images, class_names, args, clipmodel):

    clipknowledge = torch.load("").float().t().cuda()
    train_classes_clipknowledge = clipknowledge[args.train_classes]
    unlabeled_classes_clipknowledge = clipknowledge[args.unlabeled_classes]
    clipknowledge = torch.cat((train_classes_clipknowledge, unlabeled_classes_clipknowledge), dim=0)

    with torch.no_grad():
        image_features = clipmodel.get_image_features(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        logits_base = image_features @ clipknowledge.t()
        pred_base = torch.argmax(logits_base, dim=1)

        logits_base_to_prob = F.softmax(logits_base, dim=1)
        maxprob, _ = torch.max(logits_base_to_prob, dim=1)

    return pred_base, maxprob

def train_one_epoch_owssl(
    model: TPOWSSL,
    train_loader: DataLoader,
    loss_fn: GlobalLocalLoss,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
    epoch: int,
    fp16_scaler: GradScaler,
    args: argparse.Namespace,
    clipmodel: None,
    clipn_classifier: None,
    class_names : None
) -> lib.DictAverage:
    meter = lib.DictAverage()
    progress = lib.ProgressMeter(len(train_loader), meter, prefix=f"Epoch: [{epoch}]")


    if not args.learn_global_prompt and not args.learn_local_prompts:
        with torch.no_grad(), autocast(enabled=args.use_fp16):
            text_features, local_text_features = model.encode_text(class_names)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            local_text_features /= local_text_features.norm(dim=-1, keepdim=True)
    else: 
        text_features = local_text_features = None

    model.train()
    optimizer.zero_grad()
    track_loader = lib.track(train_loader, f"Epoch {epoch} / {args.max_epoch}")


    for i, batch in enumerate(track_loader):

        all_images, all_class_labels, uq_idxs, all_mask_lab = batch
        all_mask_lab = all_mask_lab[:, 0]
        all_class_labels, all_mask_lab = all_class_labels.cuda(non_blocking=True), all_mask_lab.cuda(non_blocking=True).bool()
        all_images = all_images.cuda(non_blocking=True)
        images = all_images[all_mask_lab == 1]
        targets = all_class_labels[all_mask_lab == 1]
        u_images = all_images[all_mask_lab == 0]
        tu_targets = all_class_labels[all_mask_lab == 0]


        u_targets, maxprob = CLIP_pred(u_images, class_names, args, clipmodel)

        mask = CLIPN_pred(args, u_images, class_names, u_targets, clipn_classifier)
        mask2 = CLIPN_pred_v2(args, u_images, class_names, u_targets, maxprob, clipn_classifier)
        mask_all = torch.where((mask == 1) & (mask2 == 1), 1, 0)

        images = torch.cat((images, u_images[mask_all == 1]), dim=0)
        targets = torch.cat((targets, u_targets[mask_all == 1]), dim=0)

        with autocast(enabled=args.use_fp16):

            global_logits, local_logits = model(images, class_names, text_features, local_text_features)

            loss = loss_fn(global_logits, local_logits, targets, model.logit_scale.exp())

        fp16_scaler.scale(loss).backward()

        fp16_scaler.step(optimizer)
        fp16_scaler.update()
        optimizer.zero_grad()

        gl_probs, global_probs, local_probs = model.create_prediction_scores(global_logits, local_logits)

        topk = accuracy(gl_probs, targets, topk=(1,))
        global_topk = accuracy(global_probs, targets, topk=(1,))

        meter.update(
            {
                "loss": loss.detach().item(),
                "top1": topk[0],
                "top1_global": global_topk[0],
            },
            images.size(0),
        )

        if local_probs is not None:
            local_topk = accuracy(local_probs, targets, topk=(1,))
            meter.update(
                {
                    "top1_local": local_topk[0],
                },
                images.size(0),
            )

    progress.display_summary()

    lr_scheduler.step()
    return meter


@torch.no_grad()
def evaluate(
    class_names: None,
    model: TPOWSSL,
    val_loader: DataLoader,
    args: argparse.Namespace,
    return_scores: bool = False,
) -> Tuple[lib.DictAverage, np.ndarray]:
    meter = lib.DictAverage()

    with autocast(enabled=args.use_fp16):
        text_features, local_text_features = model.encode_text(class_names)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        local_text_features /= local_text_features.norm(dim=-1, keepdim=True)

    mode = model.training
    model.eval()
    test_scores = np.zeros(len(val_loader.dataset))
    dataset_name = val_loader.dataset.__class__.__name__[:-7]

    all_preds = np.array([])
    all_targets = np.array([])


    for i, batch in enumerate(val_loader):
        images, targets, _ = batch
        images, targets = images.cuda(non_blocking=True), targets.cuda(non_blocking=True)

        all_targets = np.append(all_targets, targets.cpu().numpy())

        with autocast(enabled=args.use_fp16):
            global_logits, local_logits = model(images, text_features=text_features, local_text_features=local_text_features)

            if return_scores:
                test_scores[batch["index"].numpy()] = model.compute_scores(global_logits, local_logits)

        gl_probs, global_probs, local_probs = model.create_prediction_scores(global_logits, local_logits)
        global_topk = accuracy(global_probs, targets, topk=(1,))

        preds = gl_probs.max(1)[1]
        all_preds = np.append(all_preds, preds.cpu().numpy())

        if local_probs is not None:
            local_topk = accuracy(local_probs, targets, topk=(1,))

            topk = accuracy(gl_probs, targets, topk=(1,))

            logs = {
                "top1": topk[0],
                "top1_global": global_topk[0],
                "top1_local": local_topk[0],
            }
        else:
            logs = {
                "top1": global_topk[0],
                "top1_global": global_topk[0],
            }

        meter.update(logs, images.size(0))


    all_preds = all_preds.astype(int)
    all_targets = all_targets.astype(int)

    seen_mask = all_targets < args.dataset_class_num // 2
    unseen_mask = ~seen_mask
    overall_acc = cluster_acc(all_preds, all_targets)
    seen_acc = traditional_accuracy(all_preds[seen_mask], all_targets[seen_mask])
    unseen_acc = cluster_acc(all_preds[unseen_mask], all_targets[unseen_mask])

    print('Test overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}'.format(overall_acc, seen_acc, unseen_acc))

    model.train(mode)
    return meter, test_scores


if __name__ == "__main__":
    clip_model_names = [
        "clip_vit_b32",
        "clip_vit_b16",
        "clip_resnet50",
        "clip_resnet101",
    ]

    parser = argparse.ArgumentParser("Learning prompts for CLIP with local and global features")
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--data_dir", default="", type=str)
    parser.add_argument("--save_dir", default="./results/", type=str)
    parser.add_argument("--checkpoint_path", default=None, type=str)
    parser.add_argument("--dataset_name", default="imagenet", type=str)
    parser.add_argument("--eval_only", default=False, type=lib.boolean_flags)
    parser.add_argument("--eval_ood", default=False, type=lib.boolean_flags)
    parser.add_argument("--eval_domains", default=False, type=lib.boolean_flags)

    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--use_local_features", default=False, type=lib.boolean_flags)
    parser.add_argument("--use_global_loss", default=False, type=lib.boolean_flags)
    parser.add_argument("--use_local_loss", default=True, type=lib.boolean_flags)
    parser.add_argument("--topk", default=[5, 10, 15, 20], type=int, nargs="+")
    parser.add_argument("--learn_local_proj", default=True, type=lib.boolean_flags)
    parser.add_argument("--learn_global_prompt", default=True, type=lib.boolean_flags)
    parser.add_argument("--learn_local_prompts", default=True, type=lib.boolean_flags)
    parser.add_argument("--n_global_prompts", default=1, type=int)
    parser.add_argument("--n_local_prompts", default=1, type=int)
    parser.add_argument("--global_dropout_p", default=0.75, type=lib.float_range(0.0, 1.0))

    parser.add_argument("--prompts_batch_size", default=math.inf, type=int)

    parser.add_argument("--parallel_text_encoder", default=False, type=lib.boolean_flags)
    parser.add_argument("--parallel_vision_encoder", default=False, type=lib.boolean_flags)

    parser.add_argument("--clip_name", required=False, choices=clip_model_names, type=str)

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--inference_batch_size", default=256, type=int)
    parser.add_argument("--max_epoch", default=50, type=int)
    parser.add_argument("--optimizer", default="sgd", type=str)
    parser.add_argument("--lr_init", default=0.002, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--warmup_epoch", default=0, type=int)
    parser.add_argument("--cons_lr", default=1e-5, type=float)

    parser.add_argument("--use_fp16", default=True, type=lib.boolean_flags)
    parser.add_argument("--persistent_workers", default=False, type=lib.boolean_flags)
    parser.add_argument("--checkpointing_segments", default=4, type=int, help="Number of segments used for gradient checkpointing for the text encoder.")

    parser.add_argument("--eval_freq", default=5, type=int)
    parser.add_argument("--save_freq", default=5, type=int)
    parser.add_argument("--print_freq", default=20, type=int)

    args = parser.parse_args()

    lib.setup_logger()
    lib.random_seed(args.seed)

    if args.exp_name is not None:
        lib.LOGGER.info(f"Running experiment {args.exp_name}")
        args.save_dir = os.path.join(args.save_dir, args.exp_name)

    args.eval_domains = args.eval_domains and (args.dataset_name == "imagenet")
    args.eval_ood = args.eval_ood and (args.dataset_name == "imagenet")

    # seting-up transforms
    train_transform = dts_tools.get_train_transform()
    val_transform = _transform(224)

    # Setting-up Imagenet dataset train

    _, _, template = return_train_val_datasets(args.dataset_name, args.data_dir, train_transform, val_transform)

    template = "A photo of a {}" if (args.learn_global_prompt or args.learn_local_prompts) else template


    args.use_ssb_splits = True
    args.prop_train_labels = 0.5
    dataset_name_dict ={
        "stanford_cars" : "scars",
        "cub" : "cub",
        "flowers102" : "flower",
        "oxfordpets" : "pets"
    }

    args.dataset_name = dataset_name_dict[args.dataset_name]
    print(args.dataset_name)
    args = get_class_splits(args)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    simgcd_train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         val_transform,
                                                                                         args)

    tpowssl_dataset_name_dict ={
        "scars" : "stanford_cars",
        "cub" : "cub",
        "flower" : "flowers102",
        "pets" : "oxfordpets"
    }
    args.dataset_name = tpowssl_dataset_name_dict[args.dataset_name]

    # Setting-up dataloaders
    train_loader = dts_tools.get_train_loader(
        simgcd_train_dataset,
        batch_size=args.batch_size,
        num_workers=10,
        persistent_workers=args.persistent_workers,
    )
    val_loader = dts_tools.get_eval_loader(unlabelled_train_examples_test, batch_size=args.inference_batch_size)


    if args.eval_ood:
        ood_loaders = return_ood_loaders(args.data_dir, val_transform)

    if args.eval_domains:
        domains_loaders = return_domains_loaders(args.data_dir, val_transform)

    # Setting-up model
    PATH_TO_PROMPTS = f'../gpt3_prompts/CuPL_prompts_{args.dataset_name}.json'
    with open(PATH_TO_PROMPTS) as f:
        gpt3_prompts = json.load(f)
    textnames = []
    for item in gpt3_prompts.items():
        textnames.append(item[0])

    train_classes_textnames = [textnames[i] for i in args.train_classes]
    unlabeled_classes_textnames = [textnames[i] for i in args.unlabeled_classes]
    textnames = train_classes_textnames + unlabeled_classes_textnames


    class_names = textnames
    if args.dataset_name == "cub":
        model = TPOWSSL(
            clip_name=args.clip_name,  # 'clip_vit_b16'
            use_local_features=args.use_local_features,  # True
            checkpointing_segments=args.checkpointing_segments,  # 0
            template=template,  # A photo of a {}
            learn_local_proj=args.learn_local_proj,  # True
            learn_local_prompts=args.learn_local_prompts,  # True
            learn_global_prompt=args.learn_global_prompt,  # True
            class_names=class_names,
            n_global_prompts=args.n_global_prompts,  # 4
            n_local_prompts=args.n_local_prompts,  # 4
            prompts_batch_size=args.prompts_batch_size,  # inf, 
            ood_method=args.ood_method,  # GL-MCM
            ood_temp_scale=args.ood_temp_scale,  # 1.0
            topk=args.topk,  # [5, 10, 15, 20]
            parallel_text_encoder=args.parallel_text_encoder,  # True
            parallel_vision_encoder=args.parallel_vision_encoder,  # True
        )
    else:
        model = TPOWSSL(
            clip_name=args.clip_name,  # 'clip_vit_b16'
            use_local_features=args.use_local_features,  # True
            checkpointing_segments=args.checkpointing_segments,  # 0
            template=template,  # A photo of a {}
            learn_local_proj=args.learn_local_proj,  # True
            learn_local_prompts=args.learn_local_prompts,  # True
            learn_global_prompt=args.learn_global_prompt,  # True
            class_names=class_names,
            n_global_prompts=args.n_global_prompts,  # 4
            n_local_prompts=args.n_local_prompts,  # 4
            prompts_batch_size=args.prompts_batch_size,  # inf, 
            ood_method=args.ood_method,  # GL-MCM
            ood_temp_scale=args.ood_temp_scale,  # 1.0
            topk=args.topk,  # [5, 10, 15, 20]
            parallel_text_encoder=args.parallel_text_encoder,  # True
            parallel_vision_encoder=args.parallel_vision_encoder,  # True
        )

    model.initialize_prompt()

    # eventually load pre-trained prompts
    lib.load_checkpoint(model, args.checkpoint_path)

    model.freeze_clip()
    model = model.cuda()

    # setting-up loss
    loss_fn = GlobalLocalLoss(
        use_global_loss=args.use_global_loss,  # True
        use_local_loss=args.use_local_loss,    # True
        topk=args.topk,                        # [5, 10, 15, 20]
        global_dropout_p=args.global_dropout_p,# 0.75
    )

    # Setting-up optimizer
    optimizer = vlp_tools.get_optimizer(args.optimizer, model, args.lr_init, args.weight_decay, args.momentum)

    # Setting-up scheduler
    lr_scheduler = CosineAnnealingLR(optimizer, args.max_epoch)
    if args.warmup_epoch > 0:
        lr_scheduler = vlp_tools.ConstantWarmupScheduler(optimizer, lr_scheduler, args.warmup_epoch, args.cons_lr)

    # Setting-up GradScaler for amp
    fp16_scaler = GradScaler(enabled=args.use_fp16)


    #load teacher model and valiation model
    clipmodel = CLIPModel.from_pretrained(" ").cuda()

    clipmodel.eval()

    pre_train = '../clipn_caches/CLIPN_ATD_Repeat2_epoch_10.pt'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clipn_model, process_train, process_test = open_clip.create_model_and_transforms("ViT-B-16",
                                                                                     pretrained=pre_train,
                                                                                     device=device,
                                                                                     freeze=False)
    clipn_model.eval()
    clipn_classifier = get_clipn_classifier(args, clipn_model)


    clipn_classifier.fc_yes.requires_grad = False
    clipn_classifier.fc_no.requires_grad = False
    clipn_classifier.eval()

    # Training loop
    for epoch in range(args.max_epoch):
        import time
        start_time = time.time()

        print("epoch: ", epoch)
        if not args.eval_only:
            assert args.use_local_loss or args.use_global_loss or args.learn_local_prompts or args.learn_global_prompt, "At least one of use_local_loss or use_global_loss or learn_local_prompts or learn_global_prompt must be True"

            train_meter = train_one_epoch_owssl(
                model=model,
                train_loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=epoch,
                fp16_scaler=fp16_scaler,
                args=args,
                clipmodel=clipmodel,
                clipn_classifier = clipn_classifier,
                class_names = class_names
            )

            lib.save_checkpoint(args.save_dir, epoch, model, optimizer, lr_scheduler, fp16_scaler, train_meter, args)

            lib.LOGGER.info("Evaluation")
            val_meter, test_scores = evaluate(class_names, model, val_loader, args, return_scores=args.eval_ood and (args.eval_only or (epoch + 1 == args.max_epoch)))
            lib.LOGGER.info("Evaluation metrics: " + " ".join([" *"] + val_meter.summary()))
