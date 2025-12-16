#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import cv2
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def compute_edge_and_texture(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge map (Canny)
    edge = cv2.Canny(gray, 50, 150)

    # Texture map = local variance
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    texture = (gray.astype(np.float32) - blur.astype(np.float32))**2
    texture = cv2.GaussianBlur(texture, (9, 9), 0)

    # Normalize to 0~1
    edge = edge.astype(np.float32) / 255.0
    texture = texture / (texture.max() + 1e-6)

    return edge, texture

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack))) # 创建一个列表，里面依次是所有 viewpoint（训练相机）的索引编号，例如 [0,1,2,...,N-1]。
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # --- 只在第一次使用该 camera 时计算 edge/texture，避免重复计算 ---
        if not hasattr(viewpoint_cam, "edge_map") or not hasattr(viewpoint_cam, "texture_map"):
            # original_image: [C,H,W], float32, 0~1, RGB
            img_t = viewpoint_cam.original_image.permute(1, 2, 0).cpu().numpy()  # [H,W,C], RGB
            img_t = np.clip(img_t * 255.0, 0, 255).astype(np.uint8)

            # OpenCV 用的是 BGR，这里转换一下
            img_bgr = cv2.cvtColor(img_t, cv2.COLOR_RGB2BGR)

            edge_map, texture_map = compute_edge_and_texture(img_bgr)

            viewpoint_cam.edge_map = edge_map      # numpy [H,W]
            viewpoint_cam.texture_map = texture_map

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # === Attach edge/texture information from rendering ===
        gaussians.edge_vals = render_pkg["edge_vals"]
        gaussians.tex_vals = render_pkg["tex_vals"]

        # add renderded image saving
        SAVE_INTERVAL = 1000
        if iteration % SAVE_INTERVAL == 0:
            image = render_pkg["render"]
            save_path = f"{dataset.model_path}/vis/iter_{iteration:06d}.png"
            os.makedirs(f"{dataset.model_path}/vis", exist_ok=True)

            import torchvision
            torchvision.utils.save_image(image, save_path)
            print(f"[VIS]Saved rendered image at iteration {iteration} to {save_path}")

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask
        
        # add ply saving
        PLY_INTERVAL = 2000

        if iteration % PLY_INTERVAL == 0:
            ply_path = f"{dataset.model_path}/ply/gaussians_iter_{iteration:06d}.ply"
            os.makedirs(f"{dataset.model_path}/ply", exist_ok=True)

            gaussians.save_ply(ply_path)
            print(f"[PLY]Saved Gaussian model at iteration {iteration} to {ply_path}")
            
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image) # SSIM = Structural Similarity Index 测量两幅图在 亮度、对比度、结构 方面的相似程度。

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value) # DSSIM = 1 - SSIM   L1：保证颜色准确 SSIM：保证结构一致

        # add additional loss of opacity
        opacity = gaussians._opacity
        loss_opacity = 0.01 * torch.mean(opacity ** 2)
        loss = loss + loss_opacity

        # Depth regularization 深度正则化！！！！！！！
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"] 
            """
            invDepth = render_pkg["depth"]
            ✔ 预测的 inverse depth（逆深度）
            来自 3DGS 渲染器 CUDA forward pass。
            这是 模型渲染出来的深度图
            shape：[H, W]
            单位：inverse depth（1 / depth）
            越大 → 离相机越近
            存放在 render() 返回的 dict 里
            也就是预测值。
            """
            mono_invdepth = viewpoint_cam.invdepthmap.cuda() # 真实深度（GT）的 inverse depth
            depth_mask = viewpoint_cam.depth_mask.cuda() # ✔ 有效深度区域的掩码 mask。单目深度模型会有很多无效区域（比如太暗、反光、透明物体），COLMAP depth 也只在 SfM 能看到的区域有效。所以 Depth Loss 不能全图计算，要只对有效像素计算。

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean() # 纯深度 L1。
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure # 深度损失 = 纯深度 L1 × 权重 schedule   depth_l1_weight(iteration)通常是一个随 iteration 增加的 schedule，比如：前 500 iteration：0。之后慢慢增大到 0.05 或 0.1。这保证：前期主要学习颜色、粗结构后期加入深度拉齐增强几何结构稳定性
            loss += Ll1depth # 把深度 supervision 行为叠加到总损失里。
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)

                    # ===== Edge-aware sharpening + background clamp =====
                    if hasattr(gaussians, "edge_vals") and hasattr(gaussians, "tex_vals"):
                        # Convert numpy → torch
                        edge_vals = torch.from_numpy(gaussians.edge_vals).float().cuda()
                        tex_vals  = torch.from_numpy(gaussians.tex_vals).float().cuda()

                        # 当前 gaussians 数量
                        N = gaussians.get_xyz.shape[0]
                        K = edge_vals.shape[0]
                        if K == 0:
                            pass
                        else:
                            M = min(N, K)   # 只对前 M 个做修改

                            # 截取前 M 个高斯的 edge/tex 信息
                            edge_sub = edge_vals[:M]
                            tex_sub  = tex_vals[:M]

                            # 取出当前所有 scale，然后对前 M 个做操作
                            scales = gaussians.get_scaling          # [N,3]
                            scales_head = scales[:M].clone()        # [M,3]

                            # --- Background clamp（纯色背景区域）---
                            low_tex_mask = tex_sub < 0.03
                            max_bg_scale = 0.05
                            if low_tex_mask.any():
                                scales_head[low_tex_mask] = torch.clamp(
                                    scales_head[low_tex_mask],
                                    max=max_bg_scale
                                )

                            # --- Edge sharpening（边界变“薄片”）---
                            edge_mask = edge_sub > 0.10
                            sharpen_factor = 0.5
                            if edge_mask.any():
                                scales_edge = scales_head[edge_mask]
                                min_axis = torch.argmin(scales_edge, dim=1)
                                for i, axis in enumerate(min_axis):
                                    scales_edge[i, axis] *= sharpen_factor
                                scales_head[edge_mask] = scales_edge

                            # 把修改后的前 M 个 scale 写回去，后面的保持不变
                            scales[:M] = scales_head

                            # 写回到 _scaling 参数（log-space）
                            new_scaling_param = gaussians.scaling_inverse_activation(scales)
                            optimizable = gaussians.replace_tensor_to_optimizer(new_scaling_param, "scaling")
                            gaussians._scaling = optimizable["scaling"]

                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
