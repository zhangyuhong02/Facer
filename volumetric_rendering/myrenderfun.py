import torch
import numpy as np
import os, imageio
import matplotlib.pyplot as plt


import math
import torch
import torch.nn as nn

from volumetric_rendering.ray_marcher import MipRayMarcher2
from volumetric_rendering import math_utils

from volumetric_rendering.render_helper import *

# no need for poses
def get_camera_rays(H, W, K):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    rays_d = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    return rays_d

def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d
def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


def sample_from_featuremap(feature_map, sample_coordinates, mode='bilinear', padding_mode="zeros"):
    assert padding_mode == 'zeros'
    bs, c, h, w = feature_map.shape
    projected_coordinates = sample_coordinates[0,:,:2].unsqueeze(0).unsqueeze(0).to(feature_map.device)
    output_feature = torch.nn.functional.grid_sample(feature_map, projected_coordinates, 
                                                     mode=mode,
                                                     padding_mode = padding_mode,
                                                     align_corners=False).permute(0, 3, 2, 1).reshape(bs, 1, 196608, -1).to(feature_map.device)
    return output_feature

class ImportanceRenderer(torch.nn.Module):
    def __init__(self, K=None):
        super().__init__()
        self.ray_marcher = MipRayMarcher2()
        self.rendering_options = {}
        # 暂时将 rendering_options dict放到这里 但是因为forward函数中包含 rendering options，所以不要放这里
        self.rendering_options = {
        'image_resolution': 512,
        'disparity_space_sampling': False,
        'clamp_mode': 'softplus',
        'superresolution_module': None,
        'c_gen_conditioning_zero': None, # if true, fill generator pose conditioning label with dummy zero vector
        'gpc_reg_prob': None,
        'c_scale': 0.5, # mutliplier for generator pose conditioning label
        'superresolution_noise_mode': None, # [random or none], whether to inject pixel noise into super-resolution layers
        'density_reg': 0.25, # strength of density regularization Density regularization strength
        'density_reg_p_dist': 0.004, # distance at which to sample perturbed points for density regularization
        'reg_type': "l1", # for experimenting with variations on density regularization
        'decoder_lr_mul': 1, # learning rate multiplier for decoder
        'sr_antialias': True,
    }
        self.rendering_options.update({
            "ray_start":'auto',
            "ray_end":'auto',
            "box_warp":1,
            'depth_resolution': 48,
            'depth_resolution_importance': 48,
            'box_warp': 1,   # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
            'avg_camera_radius': 2.7,
            'avg_camera_pivot': [0, 0, -0.06],
        })
        self.render_resolusion = 64
        self.K = K
        
        self.focal = 225
        if K is None:
            self.K = np.array([
                [self.focal, 0, 0.5*self.render_resolusion],
                [0, self.focal, 0.5*self.render_resolusion],
                [0, 0, 1]
            ])

        self.ndc = False
        self.N_samples = 48
        self.lindisp = False
        self.perturb = True
        self.pytest = False
        self.N_importance = 48

    # plane shape [16, 3, 32, 256, 256]
    def forward(self, feature_map, decoder, ray_origins, ray_directions, rendering_options):
        with torch.no_grad():
            cam_rays_d = get_camera_rays(self.render_resolusion, self.render_resolusion, self.K)
            cam_rays_d = cam_rays_d.reshape(-1, 3)
            cam_rays_o = torch.tensor(0).expand(cam_rays_d.shape)

            #depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, 
            #                                       self.rendering_options['depth_resolution'], 
            #                                       self.rendering_options['disparity_space_sampling'])
            near = 0.
            far = 1.


            if self.ndc:
                cam_rays_o, cam_rays_d = ndc_rays(H, W, K[0][0], 1., cam_rays_o, cam_rays_d)
            N_rays = cam_rays_d.shape[0]
            near = torch.tensor(near).expand(N_rays, 1)
            far = torch.tensor(far).expand(N_rays, 1)

            t_vals = torch.linspace(0., 1., steps=self.N_samples)
            if not self.lindisp:
                z_vals = near * (1.-t_vals) + far * (t_vals)
            else:
                z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
            z_vals = z_vals.expand([N_rays, self.N_samples])
            if self.perturb > 0.:
                # get intervals between samples
                mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
                upper = torch.cat([mids, z_vals[...,-1:]], -1)
                lower = torch.cat([z_vals[...,:1], mids], -1)
                # stratified samples in those intervals
                t_rand = torch.rand(z_vals.shape)

                # Pytest, overwrite u with numpy's fixed random numbers
                if self.pytest:
                    np.random.seed(0)
                    t_rand = np.random.rand(*list(z_vals.shape))
                    t_rand = torch.Tensor(t_rand)

                z_vals = lower + (upper - lower) * t_rand
            # 得到采样点
            # pts 对应eg3d render中的sample_coordinates 
            # cam_rays_d对应eg3d render中的sample_directions 
            pts = cam_rays_o[...,None,:] + cam_rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
            num_rays, samples_per_ray, _ = pts.shape
            batch_size = 1
            #cam_rays_d= cam_rays_d.unsqueeze(-2).unsqueeze(-2).expand(-1, -1, self.N_samples, -1).reshape(1, -1, 3) 
            sample_directions= cam_rays_d.unsqueeze(-2).unsqueeze(-2).expand(-1, -1, self.N_samples, -1).reshape(1, -1, 3) 
            coordinates = pts.reshape(1,-1,3)
            min_x = coordinates[0,:,0].min()
            max_x = coordinates[0,:,0].max()
            min_y = coordinates[0,:,1].min()
            max_y = coordinates[0,:,1].max()
            # 将坐标归一化到[-1,1]来使用torch.meshgrid函数
            coordinates[0,:,0] = 2 * (coordinates[0,:,0] - min_x) / (max_x - min_x) - 1
            coordinates[0,:,1] = 2 * (coordinates[0,:,1] - min_y) / (max_y - min_y) - 1
            
            depths_coarse = z_vals.unsqueeze(2).unsqueeze(0)

        out = self.run_model(feature_map, decoder, coordinates, sample_directions, self.rendering_options)

        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)



        # Fine Pass
        #N_importance = self.rendering_options['depth_resolution_importance']

        if self.N_importance > 0:
            _, _, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, self.rendering_options)
            depths_fine = self.sample_importance(depths_coarse, weights, self.N_importance)
            sample_directions= cam_rays_d.unsqueeze(-2).unsqueeze(-2).expand(-1, -1, self.N_importance, -1).reshape(1, -1, 3) 
            #print("z_vals.shape", z_vals.shape)
            #print("depths_fine.shape",depths_fine.shape)
            depths_fine = depths_fine.squeeze(-1).squeeze(0).to(cam_rays_d.device)
            pts = cam_rays_o[...,None,:] + cam_rays_d[...,None,:] * depths_fine[...,:,None]
            # pts in coarse
            #pts = cam_rays_o[...,None,:] + cam_rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
            num_rays, samples_per_ray, _ = pts.shape
            #cam_rays_d= cam_rays_d.unsqueeze(-2).unsqueeze(-2).expand(-1, -1, self.N_importance, -1).reshape(1, -1, 3) 
            coordinates = pts.reshape(1, -1, 3)
            out = self.run_model(feature_map, decoder, coordinates, sample_directions, self.rendering_options)

            depths_fine = depths_fine.unsqueeze(2).unsqueeze(0)
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            colors_fine = colors_fine.reshape(batch_size, num_rays, self.N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, self.N_importance, 1)

            all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
                                                                  depths_fine, colors_fine, densities_fine)
            rgb_final, depth_final, weights = self.ray_marcher(all_colors, all_densities, all_depths, self.rendering_options)
        else:
            rgb_final, depth_final, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, self.rendering_options)


        return rgb_final, depth_final, weights.sum(2)
    
    def run_model(self, feature_map, decoder, sample_coordinates, sample_directions, options):
        #print("=====================")
        #print(sample_coordinates.shape) #torch.Size([1, 196608, 3])
        #print(sample_directions.shape) #torch.Size([1, 196608, 3])
        sampled_features = sample_from_featuremap(feature_map, sample_coordinates, padding_mode='zeros')
        out = decoder(sampled_features, sample_directions)
        
        if options.get('density_noise', 0) > 0:
            out['sigma'] += torch.randn_like(out['sigma']) * options['density_noise']
        return out
    

    def sample_importance(self, z_vals, weights, N_importance):
        """
        Return depths of importance sampled points along rays. See NeRF importance sampling for more.
        """
        with torch.no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape

            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1) # -1 to account for loss of 1 sample in MipRayMarcher

            # smooth weights
            weights = torch.nn.functional.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
            weights = torch.nn.functional.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1],
                                             N_importance).detach().reshape(batch_size, num_rays, N_importance, 1)
        return importance_z_vals    
    




    def sample_stratified(self, ray_origins, ray_start, ray_end, depth_resolution, disparity_space_sampling=False):
        """
        Return depths of approximately uniformly spaced samples along rays. 
        返回沿射线进行均匀采样的深度
        """
        N, M, _ = ray_origins.shape
        if disparity_space_sampling:
            depths_coarse = torch.linspace(0,
                                    1,
                                    depth_resolution,
                                    device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
            depth_delta = 1/(depth_resolution - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1./(1./ray_start * (1. - depths_coarse) + 1./ray_end * depths_coarse)
        else:
            if type(ray_start) == torch.Tensor:
                depths_coarse = math_utils.linspace(ray_start, ray_end, depth_resolution).permute(1,2,0,3)
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta[..., None]
            else:
                depths_coarse = torch.linspace(ray_start, ray_end, depth_resolution, device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
                depth_delta = (ray_end - ray_start)/(depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta

        return depths_coarse
    
    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """
        Sample @N_importance samples from @bins with distribution defined by @weights.
        Inputs:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero
        Outputs:
            samples: the sampled samples
        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
        cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1).to(weights.device)  # (N_rays, N_samples_+1)
                                                                   # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        u = u.to(weights.device)
        bins = bins.to(weights.device)

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(inds-1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
        cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[...,1]-cdf_g[...,0]
        denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                             # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
        return samples
    
    def sort_samples(self, all_depths, all_colors, all_densities):
        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))
        return all_depths, all_colors, all_densities

    def unify_samples(self, depths1, colors1, densities1, depths2, colors2, densities2):
        all_depths = torch.cat([depths1, depths2], dim = -2)
        all_colors = torch.cat([colors1, colors2], dim = -2)
        all_densities = torch.cat([densities1, densities2], dim = -2)

        _, indices = torch.sort(all_depths, dim=-2)
        indices = indices.to(all_colors.device)
        all_depths = all_depths.to(all_colors.device)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))

        return all_depths, all_colors, all_densities