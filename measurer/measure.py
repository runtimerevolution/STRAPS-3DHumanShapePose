import os
from typing import Dict, List, Literal

import numpy as np
import smplx
import torch
import trimesh

from measurer.constans import (
    MEASUREMENT_TYPES,
    SMPL_JOINT2IND,
    SMPL_LANDMARK_INDICES,
    SMPL_NUM_JOINTS,
    MeasurementType,
    SMPLMeasurementDefinitions,
)
from measurer.utils import (
    convex_hull_from_3D_points,
    filter_body_part_slices,
    get_joint_regressor,
    load_face_segmentation,
)


class Measurer:
    """
    Measure a parametric body model defined either.
    Parent class for Measure{SMPLX}.

    All the measurements are expressed in cm.
    """

    def __init__(self):
        self._verts = None
        self._faces = None
        self._joints = None
        self._gender = None

        self._measurements = {}
        self._height_normalized_measurements = {}
        self._labeled_measurements = {}
        self._height_normalized_labeled_measurements = {}
        self._labels2names = {}

        measurement_definitions = SMPLMeasurementDefinitions()
        self._all_possible_measurements = measurement_definitions.possible_measurements
        self._landmarks = SMPL_LANDMARK_INDICES
        self._measurement_types = MEASUREMENT_TYPES
        self._length_definitions = measurement_definitions.LENGTHS
        self._circumf_definitions = measurement_definitions.CIRCUMFERENCES
        self._circumf_2_bodypart = measurement_definitions.CIRCUMFERENCE_TO_BODYPARTS

        self._joint2ind = SMPL_JOINT2IND
        self._num_joints = SMPL_NUM_JOINTS

        self._num_points = 6890

        self._face_segmentation = None

    @property
    def all_possible_measurements(self):
        return self._all_possible_measurements

    @property
    def get_height_normalized_measurements(self):
        return {k: v.item() for k, v in self._height_normalized_measurements.items()}

    @staticmethod
    def _get_dist(verts: np.ndarray) -> float:
        """
        The Euclidean distance between vertices.
        The distance is found as the sum of each pair i
        of 3D vertices (i,0,:) and (i,1,:)
        :param verts: np.ndarray (N,2,3) - vertices used
                        to find distances

        Returns:
        :param dist: float, sumed distances between vertices
        """

        verts_distances = np.linalg.norm(verts[:, 1] - verts[:, 0], axis=1)
        distance = np.sum(verts_distances)
        distance_cm = distance * 100  # convert to cm
        return distance_cm

    def from_verts(self, verts: torch.tensor):
        pass

    def from_body_model(self, gender: str, shape: torch.tensor):
        pass

    def measure(self, measurement_names: List[str]):
        """
        Measure the given measurement names from measurement_names list
        :param measurement_names - list of strings of defined measurements
                                    to measure from MeasurementDefinitions class
        """

        for m_name in measurement_names:
            if m_name not in self.all_possible_measurements:
                print(f"Measurement {m_name} not defined.")
                pass

            if m_name in self._measurements:
                pass

            if self._measurement_types[m_name] == MeasurementType().LENGTH:

                value = self.measure_length(m_name)
                self._measurements[m_name] = value

            elif self._measurement_types[m_name] == MeasurementType().CIRCUMFERENCE:

                value = self.measure_circumference(m_name)
                self._measurements[m_name] = value

            else:
                print(f"Measurement {m_name} not defined")

    def measure_length(self, measurement_name: str):
        """
        Measure distance between 2 landmarks
        :param measurement_name: str - defined in MeasurementDefinitions

        Returns
        :float of measurement in cm
        """

        measurement_landmarks_inds = self._length_definitions[measurement_name]

        landmark_points = []
        for i in range(2):
            if isinstance(measurement_landmarks_inds[i], tuple):
                # if touple of indices for landmark, take their average
                lm = (
                    self._verts[measurement_landmarks_inds[i][0]]
                    + self._verts[measurement_landmarks_inds[i][1]]
                ) / 2
            else:
                lm = self._verts[measurement_landmarks_inds[i]]

            landmark_points.append(lm)

        landmark_points = np.vstack(landmark_points)[None, ...]

        return self._get_dist(landmark_points)

    def measure_circumference(self,
                              measurement_name: str,
                              ):
        '''
        Measure circumferences. Circumferences are defined with
        landmarks and joints - the measurement is found by cutting the
        SMPL model with the  plane defined by a point (landmark point) and
        normal (vector connecting the two joints).
        :param measurement_name: str - measurement name

        Return
        float of measurement value in cm
        '''

        measurement_definition = self._circumf_definitions[measurement_name]
        circumf_landmarks = measurement_definition["LANDMARKS"]
        circumf_landmark_indices = [self._landmarks[l_name] for l_name in circumf_landmarks]
        circumf_n1, circumf_n2 = self._circumf_definitions[measurement_name]["JOINTS"]
        circumf_n1, circumf_n2 = self._joint2ind[circumf_n1], self._joint2ind[circumf_n2]

        plane_origin = np.mean(self._verts[circumf_landmark_indices, :], axis=0)
        plane_normal = self._joints[circumf_n1, :] - self._joints[circumf_n2, :]

        mesh = trimesh.Trimesh(vertices=self._verts, faces=self._faces)

        # new version
        slice_segments, sliced_faces = trimesh.intersections.mesh_plane(mesh,
                                                                        plane_normal=plane_normal,
                                                                        plane_origin=plane_origin,
                                                                        return_faces=True)  # (N, 2, 3), (N,)

        slice_segments = filter_body_part_slices(slice_segments,
                                                 sliced_faces,
                                                 measurement_name,
                                                 self._circumf_2_bodypart,
                                                 self._face_segmentation)

        if not slice_segments.size:
            return 0.0

        slice_segments_hull = convex_hull_from_3D_points(slice_segments)

        return self._get_dist(slice_segments_hull)

    def height_normalize_measurements(self, new_height: float):
        """
        Scale all measurements so that the height measurement gets
        the value of new_height:
        new_measurement = (old_measurement / old_height) * new_height
        NOTE the measurements and body model remain unchanged, a new
        dictionary height_normalized_measurements is created.

        Input:
        :param new_height: float, the newly defined height.

        Return:
        self.height_normalized_measurements: dict of
                {measurement:value} pairs with
                height measurement = new_height, and other measurements
                scaled accordingly
        """
        if self._measurements != {}:
            old_height = self._measurements["height"]
            for m_name, m_value in self._measurements.items():
                norm_value = (m_value / old_height) * new_height
                self._height_normalized_measurements[m_name] = norm_value

            if self._labeled_measurements != {}:
                for m_name, m_value in self._labeled_measurements.items():
                    norm_value = (m_value / old_height) * new_height
                    self._height_normalized_labeled_measurements[m_name] = norm_value

    def label_measurements(self, set_measurement_labels: Dict[str, str]):
        """
        Create labeled_measurements dictionary with "label: x cm" structure
        for each given measurement.
        NOTE: This overwrites any prior labeling!

        :param set_measurement_labels: dict of labels and measurement names
                                        (example. {"A": "head_circumference"})
        """

        if self._labeled_measurements != {}:
            print("Overwriting old labels")

        self._labeled_measurements = {}
        self._labels2names = {}

        for set_label, set_name in set_measurement_labels.items():

            if set_name not in self.all_possible_measurements:
                print(f"Measurement {set_name} not defined.")
                pass

            if set_name not in self._measurements.keys():
                self.measure([set_name])

            self._labeled_measurements[set_label] = self._measurements[set_name]
            self._labels2names[set_label] = set_name


class MeasureSMPL(Measurer):
    """
    Measure the SMPL model defined either by the shape parameters or
    by its 6890 vertices.

    All the measurements are expressed in cm.
    """

    def __init__(self):
        super().__init__()

        self._model_type = "smpl"
        self._body_model_root = "additional"
        self._body_model_path = os.path.join(self._body_model_root,
                                            self._model_type)

        self._faces = smplx.SMPL(self._body_model_path, ext="pkl").faces
        face_segmentation_path = os.path.join(self._body_model_path,
                                              f"{self._model_type}_body_parts_2_faces.json")
        self._face_segmentation = load_face_segmentation(face_segmentation_path)

        self._landmarks = SMPL_LANDMARK_INDICES
        self._measurement_types = MEASUREMENT_TYPES
        self._length_definitions = SMPLMeasurementDefinitions().LENGTHS
        self._circumf_definitions = SMPLMeasurementDefinitions().CIRCUMFERENCES
        self._circumf_2_bodypart = SMPLMeasurementDefinitions().CIRCUMFERENCE_TO_BODYPARTS
        self._all_possible_measurements = SMPLMeasurementDefinitions().possible_measurements

        self.joint2ind = SMPL_JOINT2IND
        self.num_joints = SMPL_NUM_JOINTS

        self.num_points = 6890

    def from_verts(self, verts: torch.tensor):
        """
        Construct body model from only vertices.
        :param verts: torch.tensor (6890,3) of SMPL vertices
        """

        verts = verts.squeeze()
        error_msg = f"verts need to be of dimension ({self.num_points},3)"
        assert verts.shape == torch.Size([self.num_points, 3]), error_msg

        joint_regressor = get_joint_regressor(self._model_type,
                                              self._body_model_root,
                                              gender="NEUTRAL",
                                              num_thetas=self.num_joints)
        joints = torch.matmul(joint_regressor, verts)
        self._joints = joints.numpy()
        self._verts = verts.numpy()


class MeasureLoadedSMPL(MeasureSMPL):
    """
    Measure the SMPLX model defined either by the shape parameters or
    by its 10475 vertices.

    All the measurements are expressed in cm.
    """

    def __init__(
        self,
        gender: Literal["MALE", "FEMALE", "NEUTRAL"],
        vertices: torch.tensor,
        joints: torch.tensor,
        faces: torch.tensor = None
    ):
        super().__init__()
        self._verts = vertices
        self._joints = joints
        self._faces = faces
        self._gender = gender

    def from_verts(self, verts: torch.tensor):
        self._verts = verts

    def from_body_model(self, gender: str, shape: torch.tensor):
        raise NotImplementedError

    @property
    def get_inferred_measurements(self):
        cross_chest_4_down_from_hsp = (
            self._height_normalized_measurements["chestCircumference"] * 0.47
        )
        front_length_from_hsp = self._height_normalized_measurements[
            "shoulderToCrotchHeight"
        ] + (self._height_normalized_measurements["chestCircumference"] * 0.03)
        sleeve_length_from_center_back = (
            self._height_normalized_measurements["shoulderBreadth"] / 2
        ) + self._height_normalized_measurements["armLeftLength"]
        high_hip_3_down_from_natural_waist = (
            self._height_normalized_measurements["hipCircumference"]
            - self._height_normalized_measurements["waistCircumference"]
        ) * 0.4 + self._height_normalized_measurements["waistCircumference"]
        high_hip_from_top_edge = (
            self._height_normalized_measurements["shoulderToCrotchHeight"] * 0.08
        )
        low_hip_from_top_edge = (
            self._height_normalized_measurements["shoulderToCrotchHeight"] * 0.18
        )
        front_rise = (
            self._height_normalized_measurements["shoulderToCrotchHeight"] * 0.22
        )
        back_rise = (
            self._height_normalized_measurements["shoulderToCrotchHeight"] * 0.30
        )
        outseam = front_rise + self._height_normalized_measurements["insideLegHeight"]
        knee_circunference = (
            self._height_normalized_measurements["thighLeftCircumference"] * 0.75
        )

        return {
            "crossChest4DownFromHSP": cross_chest_4_down_from_hsp.item(),
            "frontLengthFromHSP": front_length_from_hsp.item(),
            "sleeveLengthFromCenterBack": sleeve_length_from_center_back.item(),
            "highHip3DownFromNaturalWaist": high_hip_3_down_from_natural_waist.item(),
            "highHipFromTopEdge": high_hip_from_top_edge.item(),
            "lowHipFromTopEdge": low_hip_from_top_edge.item(),
            "frontRise": front_rise.item(),
            "backRise": back_rise.item(),
            "outseam": outseam.item(),
            "kneeCircumference": knee_circunference.item(),
        }
