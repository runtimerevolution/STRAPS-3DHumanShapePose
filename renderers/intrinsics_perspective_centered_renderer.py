import math

import numpy as np
import pyrender
import trimesh
from pyrender import RenderFlags

import config
from renderers.weak_perspective_pyrender_renderer import Renderer, WeakPerspectiveCamera


class CenteredRenderer(Renderer):

    def __init__(self, resolution=(256, 256), margin_factor=0.8, scale_multiplier=5.0):
        super(CenteredRenderer, self).__init__(resolution)

        self.margin_factor = margin_factor
        self.scale_multiplier = scale_multiplier

    def render(self, verts, cam, img=None, angle=None, axis=None, mesh_filename=None, color=config.MODEL_SKIN_COLOR,
               return_mask=False):

        from utils.segmentation import smpl_body_segmentation, body_to_vertex_colors
        s = smpl_body_segmentation()
        vertex_colors = body_to_vertex_colors(s, verts.shape[0])

        width, height = self.resolution

        # Calculate bounding box of the model
        min_coords = np.min(verts, axis=0)
        max_coords = np.max(verts, axis=0)
        model_size = max_coords - min_coords
        model_center = (min_coords + max_coords) / 2

        # Calculate scale to fit the model within the target dimensions
        scale_x = (width * self.margin_factor) / model_size[0]
        scale_y = (height * self.margin_factor) / model_size[1]
        scale = min(scale_x, scale_y) * self.scale_multiplier

        # Center the model in the target resolution
        center_x = width / 2
        center_y = height / 2

        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces, vertex_colors=vertex_colors)

        # This ensures the model is always centered regardless of its original position
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = -model_center
        mesh.apply_transform(translation_matrix)

        # Apply custom rotation if specified (after centering)
        if angle and axis:
            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            mesh.apply_transform(R)

        if mesh_filename is not None:
            mesh.export(mesh_filename)

        # Create camera with proper intrinsics
        camera = pyrender.IntrinsicsCamera(
            fx=scale,
            fy=scale,
            cx=center_x,
            cy=center_y,
        )

        mesh = pyrender.Mesh.from_trimesh(mesh)
        mesh_node = self.scene.add(mesh, 'mesh')

        # Calculate proper camera distance to avoid cropping
        diagonal_size = np.linalg.norm(model_size)
        camera_distance = diagonal_size * 2.0

        # Create camera pose matrix
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = [0, 0, camera_distance]

        cam_node = self.scene.add(camera, pose=camera_pose)

        rgb, rend_depth = self.renderer.render(self.scene, flags=RenderFlags.RGBA)
        valid_mask = (rend_depth > 0)
        if return_mask:
            return valid_mask
        else:
            if img is None:
                img = np.full_like((self.resolution[0], self.resolution[1], 3),
                                   config.REPOSED_IMAGE_BACKGROUND_RGB_COLOR)
            valid_mask = valid_mask[:, :, None]
            output_img = rgb[:, :, :-1] * valid_mask + (1 - valid_mask) * img
            image = output_img.astype(np.uint8)

            self.scene.remove_node(mesh_node)
            self.scene.remove_node(cam_node)

            return image
