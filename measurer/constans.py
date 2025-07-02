# Landmarks
SMPL_LANDMARK_INDICES = {"HEAD_TOP": 412,
                    "HEAD_LEFT_TEMPLE": 166,
                    "NECK_ADAM_APPLE": 3050,
                    "LEFT_HEEL": 3458,
                    "RIGHT_HEEL": 6858,
                    "LEFT_NIPPLE": 3042,
                    "RIGHT_NIPPLE": 6489,

                    "SHOULDER_TOP": 3068,
                    "INSEAM_POINT": 3149,
                    "BELLY_BUTTON": 3501,
                    "BACK_BELLY_BUTTON": 3022,
                    "CROTCH": 1210,
                    "PUBIC_BONE": 3145,
                    "RIGHT_WRIST": 5559,
                    "LEFT_WRIST": 2241,
                    "RIGHT_BICEP": 4855,
                    "RIGHT_FOREARM": 5197,
                    "LEFT_SHOULDER": 3011,
                    "RIGHT_SHOULDER": 6470,
                    "LOW_LEFT_HIP": 3134,
                    "LEFT_THIGH": 947,
                    "LEFT_CALF": 1103,
                    "LEFT_ANKLE": 3325,
                    "LEFT_ELBOW": 1643,

                    "BUTTHOLE": 3119,

                    # introduce CAESAR landmarks because
                    # i need to measure arms in parts
                    "Cervicale": 829,
                    'Rt. Acromion': 5342,
                    'Rt. Humeral Lateral Epicn': 5090,
                    'Rt. Ulnar Styloid': 5520,
                    }

SMPL_LANDMARK_INDICES["HEELS"] = (SMPL_LANDMARK_INDICES["LEFT_HEEL"],
                                  SMPL_LANDMARK_INDICES["RIGHT_HEEL"])

# Joints
SMPL_NUM_JOINTS = 24
SMPL_IND2JOINT = {
    0: 'pelvis',
     1: 'left_hip',
     2: 'right_hip',
     3: 'spine1',
     4: 'left_knee',
     5: 'right_knee',
     6: 'spine2',
     7: 'left_ankle',
     8: 'right_ankle',
     9: 'spine3',
    10: 'left_foot',
    11: 'right_foot',
    12: 'neck',
    13: 'left_collar',
    14: 'right_collar',
    15: 'head',
    16: 'left_shoulder',
    17: 'right_shoulder',
    18: 'left_elbow',
    19: 'right_elbow',
    20: 'left_wrist',
    21: 'right_wrist',
    22: 'left_hand',
    23: 'right_hand'
}
SMPL_JOINT2IND = {name:ind for ind,name in SMPL_IND2JOINT.items()}

# Measurements
STANDARD_LABELS = {
        'A': 'head circumference',
        'B': 'neck circumference',
        'C': 'shoulder to crotch height',
        'D': 'chest circumference',
        'E': 'waist circumference',
        'F': 'hip circumference',
        'G': 'wrist right circumference',
        'H': 'bicep right circumference',
        'I': 'forearm right circumference',
        'J': 'arm right length',
        'K': 'inside leg height',
        'L': 'thigh left circumference',
        'M': 'calf left circumference',
        'N': 'ankle left circumference',
        'O': 'shoulder breadth',
        'P': 'height'
    }


class MeasurementType:
    CIRCUMFERENCE = "circumference"
    LENGTH = "length"


MEASUREMENT_TYPES = {
    "height": MeasurementType.LENGTH,
    "headCircumference": MeasurementType.CIRCUMFERENCE,
    "neckCircumference": MeasurementType.CIRCUMFERENCE,
    "shoulderToCrotchHeight": MeasurementType.LENGTH,
    "chestCircumference": MeasurementType.CIRCUMFERENCE,
    "waistCircumference": MeasurementType.CIRCUMFERENCE,
    "hipCircumference": MeasurementType.CIRCUMFERENCE,
    "wristRightCircumference": MeasurementType.CIRCUMFERENCE,
    "bicepRightCircumference": MeasurementType.CIRCUMFERENCE,
    "forearmRightCircumference": MeasurementType.CIRCUMFERENCE,
    "armLeftLength": MeasurementType.LENGTH,
    "armRightLength": MeasurementType.LENGTH,
    "insideLegHeight": MeasurementType.LENGTH,
    "thighLeftCircumference": MeasurementType.CIRCUMFERENCE,
    "calfLeftCircumference": MeasurementType.CIRCUMFERENCE,
    "ankleLeftCircumference": MeasurementType.CIRCUMFERENCE,
    "shoulderBreadth": MeasurementType.LENGTH,
    "armLengthShoulderToElbow": MeasurementType.LENGTH,
    "armLengthSpineToWrist": MeasurementType.LENGTH,
    "crotchHeight": MeasurementType.LENGTH,
    "hipCircumferenceMaxHeight": MeasurementType.LENGTH,
}


class SMPLMeasurementDefinitions():
    '''
    Definition of SMPL measurements.

    To add a new measurement:
    1. add it to the measurement_types dict and set the type:
       LENGTH or CIRCUMFERENCE
    2. depending on the type, define the measurement in LENGTHS or
       CIRCUMFERENCES dict
       - LENGTHS are defined using 2 landmarks - the measurement is
                found with distance between landmarks
       - CIRCUMFERENCES are defined with landmarks and joints - the
                measurement is found by cutting the SMPL model with the
                plane defined by a point (landmark point) and normal (
                vector connecting the two joints)
    3. If the body part is a CIRCUMFERENCE, a possible issue that arises is
       that the plane cutting results in multiple body part slices. To alleviate
       that, define the body part where the measurement should be located in
       CIRCUMFERENCE_TO_BODYPARTS dict. This way, only slice in that body part is
       used for finding the measurement. The body parts are defined by the SMPL
       face segmentation.
    '''

    LENGTHS = {"height":
                   (SMPL_LANDMARK_INDICES["HEAD_TOP"],
                    SMPL_LANDMARK_INDICES["HEELS"]
                    ),
               "shoulderToCrotchHeight":
                   (SMPL_LANDMARK_INDICES["SHOULDER_TOP"],
                    SMPL_LANDMARK_INDICES["INSEAM_POINT"]
                    ),
               "armLeftLength":
                   (SMPL_LANDMARK_INDICES["LEFT_SHOULDER"],
                    SMPL_LANDMARK_INDICES["LEFT_WRIST"]
                    ),
               "armRightLength":
                   (SMPL_LANDMARK_INDICES["RIGHT_SHOULDER"],
                    SMPL_LANDMARK_INDICES["RIGHT_WRIST"]
                    ),
               "insideLegHeight":
                   (SMPL_LANDMARK_INDICES["LOW_LEFT_HIP"],
                    SMPL_LANDMARK_INDICES["LEFT_ANKLE"]
                    ),
               "shoulderBreadth":
                   (SMPL_LANDMARK_INDICES["LEFT_SHOULDER"],
                    SMPL_LANDMARK_INDICES["RIGHT_SHOULDER"]
                    ),
               "armLengthShoulderToElbow":
                   (
                       #  SMPL_LANDMARK_INDICES["LEFT_SHOULDER"],
                       #  SMPL_LANDMARK_INDICES["LEFT_ELBOW"]
                       SMPL_LANDMARK_INDICES["Rt. Acromion"],
                       SMPL_LANDMARK_INDICES["Rt. Humeral Lateral Epicn"]
                   ),
               "crotchHeight":
                   (SMPL_LANDMARK_INDICES["CROTCH"],
                    SMPL_LANDMARK_INDICES["HEELS"]
                    ),
               "hipCircumferenceMaxHeight":
                   (SMPL_LANDMARK_INDICES["PUBIC_BONE"],
                    SMPL_LANDMARK_INDICES["HEELS"]
                    ),
               # FIXME: implement geodesic distance for this measurement
               "armLengthSpineToWrist":
                   (
                       #  SMPL_LANDMARK_INDICES["SHOULDER_TOP"],
                       #  SMPL_LANDMARK_INDICES["LEFT_WRIST"]
                       SMPL_LANDMARK_INDICES["Cervicale"],
                       SMPL_LANDMARK_INDICES["Rt. Acromion"],
                       SMPL_LANDMARK_INDICES["Rt. Humeral Lateral Epicn"],
                       SMPL_LANDMARK_INDICES["Rt. Ulnar Styloid"]
                   ),
               }

    # defined with landmarks and joints
    # landmarks are defined with indices of the smpl model points
    # normals are defined with joint names of the smpl model
    CIRCUMFERENCES = {
        "headCircumference": {"LANDMARKS": ["HEAD_LEFT_TEMPLE"],
                               "JOINTS": ["pelvis", "spine3"]},

        "neckCircumference": {"LANDMARKS": ["NECK_ADAM_APPLE"],
                               "JOINTS": ["spine2", "head"]},

        "chestCircumference": {"LANDMARKS": ["LEFT_NIPPLE", "RIGHT_NIPPLE"],
                                "JOINTS": ["pelvis", "spine3"]},

        "waistCircumference": {"LANDMARKS": ["BELLY_BUTTON", "BACK_BELLY_BUTTON"],
                                "JOINTS": ["pelvis", "spine3"]},

        "hipCircumference": {"LANDMARKS": ["PUBIC_BONE"],
                              "JOINTS": ["pelvis", "spine3"]},

        "wristRightCircumference": {"LANDMARKS": ["RIGHT_WRIST"],
                                      "JOINTS": ["right_wrist", "right_hand"]},

        "bicepRightCircumference": {"LANDMARKS": ["RIGHT_BICEP"],
                                      "JOINTS": ["right_shoulder", "right_elbow"]},

        "forearmRightCircumference": {"LANDMARKS": ["RIGHT_FOREARM"],
                                        "JOINTS": ["right_elbow", "right_wrist"]},

        "thighLeftCircumference": {"LANDMARKS": ["LEFT_THIGH"],
                                     "JOINTS": ["pelvis", "spine3"]},

        "calfLeftCircumference": {"LANDMARKS": ["LEFT_CALF"],
                                    "JOINTS": ["pelvis", "spine3"]},

        "ankleLeftCircumference": {"LANDMARKS": ["LEFT_ANKLE"],
                                     "JOINTS": ["pelvis", "spine3"]},

    }

    possible_measurements = list(LENGTHS.keys()) + list(CIRCUMFERENCES.keys())

    CIRCUMFERENCE_TO_BODYPARTS = {
        "headCircumference": "head",
        "neckCircumference": "neck",
        "chestCircumference": ["spine1", "spine2"],
        "waistCircumference": ["hips", "spine"],
        "hipCircumference": "hips",
        "wristRightCircumference": ["rightHand", "rightForeArm"],
        "bicepRightCircumference": "rightArm",
        "forearmRightCircumference": "rightForeArm",
        "thighLeftCircumference": "leftUpLeg",
        "calfLeftCircumference": "leftLeg",
        "ankleLeftCircumference": "leftLeg",
    }
