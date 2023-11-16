# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
The ray sampler is a module that takes in camera matrices and resolution and batches of rays.
Expects cam2world matrices that use the OpenCV camera coordinate system conventions.
"""

import torch
def save_tensor(tensor, save_path='/data2/zyh/SportsSloMo-v2/SportsSloMo_EBME/flow_preocess/flow.npy'):
    import numpy as np
    np.save(save_path, tensor.detach().cpu().numpy())
    print("saved")      
class RaySampler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ray_origins_h, self.ray_directions, self.depths, self.image_coords, self.rendering_options = None, None, None, None, None


    def forward(self, cam2world_matrix, intrinsics, resolution):
        """
        Create batches of rays and return origins and directions.

        cam2world_matrix: (N, 4, 4)
        intrinsics: (N, 3, 3)
        resolution: int

        ray_origins: (N, M, 3)
        ray_dirs: (N, M, 2)
        """
        # ray sampler添加的对于坐标系求逆得到的世界坐标系
        import numpy as np
        device = cam2world_matrix.device
        N, M = cam2world_matrix.shape[0], resolution**2
        #w2c = cam2world_matrix.detach().cpu().numpy()
        #cam2world_matrix = torch.from_numpy(np.linalg.inv(w2c)).to(device)
        #cam2world_matrix = torch.linalg.inv(cam2world_matrix)

        cam_locs_world = cam2world_matrix[:, :3, 3] # 相机在世界坐标系下的位置
        cam_locs_world = torch.zeros((1,3)).to(cam2world_matrix.device)
        #print(cam_locs_world.shape)
        #print(cam_locs_world)
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]
        sk = intrinsics[:, 0, 1]

        # change to transforms_train.json
        # 这里是一个归一化的uv map
        uv = torch.stack(torch.meshgrid(torch.arange(resolution, dtype=torch.float32, device=cam2world_matrix.device), 
                                        torch.arange(resolution, dtype=torch.float32, device=cam2world_matrix.device), 
                                        indexing='ij')) * (1./resolution) + (0.5/resolution) #后面这两个系数是用来对根据resolusion通过arange生成的meshgrid进行归一化的
        uv = uv.flip(0).reshape(2, -1).transpose(1, 0) # torch.Size([4096, 2])
        uv = uv.unsqueeze(0).repeat(cam2world_matrix.shape[0], 1, 1) # torch.Size([1, 4096, 2])

        x_cam = uv[:, :, 0].view(N, -1)
        y_cam = uv[:, :, 1].view(N, -1)
        z_cam = torch.ones((N, M), device=cam2world_matrix.device)
        # x,y从uv的像素坐标系转换到相机坐标系
        x_lift = (x_cam - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y_cam/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z_cam
        y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z_cam

        cam_rel_points = torch.stack((x_lift, y_lift, z_cam, torch.ones_like(z_cam)), dim=-1)
        #save_tensor(cam2world_matrix, save_path="/home/zyh/psp/pixel2style2pixel-master/models/motion_estimator/vis/w2c.npy")
        #save_tensor(cam_rel_points, save_path="/home/zyh/psp/pixel2style2pixel-master/models/motion_estimator/vis/c2w_camera_points.npy")
        #save_tensor(cam_locs_world, save_path="/home/zyh/psp/pixel2style2pixel-master/models/motion_estimator/vis/cam_locs_world.npy")
        
        world_rel_points = torch.bmm(cam2world_matrix, cam_rel_points.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3]
        #save_tensor(world_rel_points, save_path="/home/zyh/psp/pixel2style2pixel-master/models/motion_estimator/vis/c2w_world_points.npy")
        ray_dirs = world_rel_points - cam_locs_world[:, None, :]
        ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=2)
        #save_tensor(world_rel_points, save_path="/home/zyh/psp/pixel2style2pixel-master/models/motion_estimator/vis/ray_dirs.npy")
        ray_origins = cam_locs_world.unsqueeze(1).repeat(1, ray_dirs.shape[1], 1)




        ### 转换到相机坐标系

        return ray_origins, ray_dirs