import json
import sys

import numpy as np
import smplx
from scipy.spatial import ConvexHull


def load_face_segmentation(path: str):
    """
    Load face segmentation which defines for each body model part
    the faces that belong to it.
    :param path: str - path to json file with defined face segmentation
    """

    try:
        with open(path, "r") as f:
            face_segmentation = json.load(f)
    except FileNotFoundError:
        sys.exit(f"No such file - {path}")

    return face_segmentation


def convex_hull_from_3D_points(slice_segments: np.ndarray):
    """
    Cretes convex hull from 3D points
    :param slice_segments: np.ndarray, dim N x 2 x 3 representing N 3D segments

    Returns:
    :param slice_segments_hull: np.ndarray, dim N x 2 x 3 representing N 3D segments
                                that form the convex hull
    """

    # stack all points in N x 3 array
    merged_segment_points = np.concatenate(slice_segments)
    unique_segment_points = np.unique(merged_segment_points,
                                      axis=0)

    # points lie in plane -- find which ax of x,y,z is redundant
    redundant_plane_coord = np.argmin(np.max(unique_segment_points, axis=0) -
                                      np.min(unique_segment_points, axis=0))
    non_redundant_coords = [x for x in range(3) if x != redundant_plane_coord]

    # create convex hull
    hull = ConvexHull(unique_segment_points[:, non_redundant_coords])
    segment_point_hull_inds = hull.simplices.reshape(-1)

    slice_segments_hull = unique_segment_points[segment_point_hull_inds]
    slice_segments_hull = slice_segments_hull.reshape(-1, 2, 3)

    return slice_segments_hull


def filter_body_part_slices(
    slice_segments: np.ndarray,
    sliced_faces: np.ndarray,
    measurement_name: str,
    circumf_2_bodypart: dict,
    face_segmentation: dict,
):
    """
    Remove segments that are not in the appropriate body part
    for the given measurement.
    :param slice_segments: np.ndarray - (N,2,3) for N segments
                                        represented as two 3D points
    :param sliced_faces: np.ndarray - (N,) representing the indices of the
                                        faces
    :param measurement_name: str - name of the measurement
    :param circumf_2_bodypart: dict - dict mapping measurement to body part
    :param face_segmentation: dict - dict mapping body part to all faces belonging
                                    to it

    Return:
    :param slice_segments: np.ndarray (K,2,3) where K < N, for K segments
                            represented as two 3D points that are in the
                            appropriate body part
    """

    if measurement_name in circumf_2_bodypart.keys():

        body_parts = circumf_2_bodypart[measurement_name]

        if isinstance(body_parts, list):
            body_part_faces = [
                face_index
                for body_part in body_parts
                for face_index in face_segmentation[body_part]
            ]
        else:
            body_part_faces = face_segmentation[body_parts]

        N_sliced_faces = sliced_faces.shape[0]

        keep_segments = []
        for i in range(N_sliced_faces):
            if sliced_faces[i] in body_part_faces:
                keep_segments.append(i)

        return slice_segments[keep_segments]

    else:
        return slice_segments


def get_joint_regressor(
    body_model_type, body_model_root, gender="NEUTRAL", num_thetas=24
):
    """
    Extract joint regressor from SMPL body model
    :param body_model_type: str of body model type (smpl or smplx, etc.)
    :param body_model_root: str of location of folders where smpl/smplx
                            inside which .pkl models
    :gender: the gender of the model

    Return:
    :param model.J_regressor: torch.tensor (23,N) used to
                              multiply with body model to get
                              joint locations
    """

    model = smplx.create(
        model_path=body_model_root,
        model_type=body_model_type,
        gender=gender,
        use_face_contour=False,
        num_betas=10,
        body_pose=torch.zeros((1, num_thetas - 1 * 3)),
        ext="pkl",
    )
    return model.J_regressor


def evaluate_mae(gt_measurements, estim_measurements):
    """
    Compare two sets of measurements - given as dicts - by finding
    the mean absolute error (MAE) of each measurement.
    :param gt_measurement: dict of {measurement:value} pairs
    :param estim_measurements: dict of {measurement:value} pairs

    Returns
    :param errors: dict of {measurement:value} pairs of measurements
                    that are both in gt_measurement and estim_measurements
                    where value corresponds to the mean absoulte error (MAE)
                    in cm
    """

    MAE = {}

    for m_name, m_value in gt_measurements.items():
        if m_name in estim_measurements.keys():
            error = abs(m_value - estim_measurements[m_name])
            MAE[m_name] = error

    if MAE == {}:
        print("Measurement dicts do not have any matching measurements!")
        print("Returning empty dict!")

    return MAE
