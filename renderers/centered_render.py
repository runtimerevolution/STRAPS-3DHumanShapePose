


class CenteredRenderer(Renderer):

    def __init__(self, *args, **kwargs):
        super(CenteredRenderer, self).__init__(*args, **kwargs)

    def render(self, verts, cam, img=None, angle=None, axis=None, mesh_filename=None, color=[0.8, 0.3, 0.3],
               return_mask=False):

        # Calculate bounding box of the model
        min_coords = np.min(verts, axis=0)
        max_coords = np.max(verts, axis=0)
        model_size = max_coords - min_coords
        model_center = (min_coords + max_coords) / 2

        # Calculate scale to fit the model within the target dimensions
        scale_x = (target_width * margin_factor) / model_size[0]
        scale_y = (target_height * margin_factor) / model_size[1]
        scale = min(scale_x, scale_y) * scale_multiplier

        # Pure geometric centering - no bias
        y_offset = -model_center[1]  # Negative because camera Y+ moves camera up
        x_offset = -model_center[0]  # Center in X axis as well

        # Center the model in the target resolution
        center_x = target_width / 2
        center_y = target_height / 2

        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces)

        Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
        mesh.apply_transform(Rx)

        if mesh_filename is not None:
            mesh.export(mesh_filename)

        if angle and axis:
            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            mesh.apply_transform(R)

        if cam.shape[-1] == 4:
            sx, sy, tx, ty = cam
        elif cam.shape[-1] == 3:
            s, tx, ty = cam
            sx = sy = s

        camera = WeakPerspectiveCamera(
            scale=[sx, sy],
            translation=[x_offset, y_offset, 5.0],
            zfar=scale
        )

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        mesh_node = self.scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        cam_node = self.scene.add(camera, pose=camera_pose)

        rgb, rend_depth = self.renderer.render(self.scene, flags=pyrender.RenderFlags.RGBA)
        valid_mask = (rend_depth > 0)
        if return_mask:
            return valid_mask
        else:
            if img is None:
                img = np.zeros(self.resolution[0], self.resolution[1], 3)
            # white_bg = np.ones_like(color)
            valid_mask = valid_mask[:, :, None]
            output_img = rgb[:, :, :-1] * valid_mask + (1 - valid_mask) * img
            image = output_img.astype(np.uint8)

            self.scene.remove_node(mesh_node)
            self.scene.remove_node(cam_node)

            return image
