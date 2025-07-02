import os
import argparse

import numpy as np
import torch

import config
from models.regressor import SingleInputRegressor
from predict.predict_3D import predict_3D

from measurer.measure import MeasureLoadedSMPL
from measurer.utils import evaluate_mae


def get_measurers(height, joints, vertices):
    measurer = MeasureLoadedSMPL("NEUTRAL", vertices, joints, faces=np.load(config.SMPL_FACES_PATH))
    measurer.measure(measurer.all_possible_measurements)
    measurer.height_normalize_measurements(height)

    smpl_measurements = measurer.get_height_normalized_measurements
    # smpl_measurements.update(measurer.get_inferred_measurements)

    return smpl_measurements


def main(input_path, checkpoint_path, device, silhouettes_from):
    regressor = SingleInputRegressor(resnet_in_channels=18,
                                     resnet_layers=18,
                                     ief_iters=3)

    print("Regressor loaded. Weights from:", checkpoint_path)
    regressor.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    regressor.load_state_dict(checkpoint['best_model_state_dict'])

    image_joints_vertices = predict_3D(input_path, regressor, device, silhouettes_from=silhouettes_from,
                                      save_proxy_vis=True, render_vis=True)

    # TODO: test
    my_measures = {
        "height": 186.0,
        "shoulderToCrotchHeight": 70.2,
        "armLeftLength": 58,
        "armRightLength": 58,
        "insideLegHeight": 78,
        "shoulderBreadth": 47,
        "headCircumference": 58.5,
        "neckCircumference": 40,
        "chestCircumference": 99,
        "waistCircumference": 90.5,
        "hipCircumference": 96,
        "wristRightCircumference": 18,
        "bicepRightCircumference": 29.5,
        "forearmRightCircumference": 29,
        "thighLeftCircumference": 59,
        "calfLeftCircumference": 41,
        "ankleLeftCircumference": 24.5,
    }

    for image, (joints, vertices) in image_joints_vertices.items():
        image_measurements = get_measurers(186, joints, vertices)
        print(f"Measurements for {image}: {image_measurements}")
        print("MAE:", evaluate_mae(my_measures, image_measurements))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Path to input image/folder of images.')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--silh_from', choices=['densepose', 'pointrend'])
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Regarding body mesh visualisation using pyrender:
    # If you are running this script on a remote machine via ssh, you might need to use EGL
    # to create an OpenGL context. If EGL is installed on the remote machine, uncommenting the
    # following line should work.
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    # If this still doesn't work, just disable rendering visualisation by setting render_vis
    # argument in predict_3D to False.

    main(args.input, args.checkpoint, device, args.silh_from)
