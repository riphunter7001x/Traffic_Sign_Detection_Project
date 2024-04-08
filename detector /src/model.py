import os
import logging
from ultralytics import YOLO

# Configure logging
LOGS_FOLDER = r".\logs"
LOG_FILE = os.path.join(LOGS_FOLDER, "model_loader.log")
os.makedirs(LOGS_FOLDER, exist_ok=True)

logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_model():
    """
    Load the YOLO object detection model.

    This function loads the YOLO object detection model from the specified path and returns the model instance.
    The model is used for real-time object detection tasks.

    Returns:
        YOLO: The YOLO object detection model instance.
    """
    try:
        logging.info("Loading model...")
        model_path = r'.\src\model\best_model.pt'
        # model_path = r'src\model\bestmy8s.pt'
        
        model = YOLO(model_path)
        logging.info("model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error occurred while loading model: {str(e)}")
        raise

# Declare class names
classnames = [
    "ALL_MOTOR_VEHICLE_PROHIBITED",
    "AXLE_LOAD_LIMIT",
    "BARRIER_AHEAD",
    "BULLOCK_AND_HANDCART_PROHIBITED",
    "BULLOCK_PROHIBITED",
    "CATTLE",
    "COMPULSARY_AHEAD",
    "COMPULSARY_AHEAD_OR_TURN_LEFT",
    "COMPULSARY_AHEAD_OR_TURN_RIGHT",
    "COMPULSARY_CYCLE_TRACK",
    "COMPULSARY_KEEP_LEFT",
    "COMPULSARY_KEEP_RIGHT",
    "COMPULSARY_MINIMUM_SPEED",
    "COMPULSARY_SOUND_HORN",
    "COMPULSARY_TURN_LEFT",
    "COMPULSARY_TURN_LEFT_AHEAD",
    "COMPULSARY_TURN_RIGHT",
    "COMPULSARY_TURN_RIGHT_AHEAD",
    "CROSS_ROAD",
    "CYCLE_CROSSING",
    "CYCLE_PROHIBITED",
    "DANGEROUS_DIP",
    "DIRECTION",
    "FALLING_ROCKS",
    "FERRY",
    "GAP_IN_MEDIAN",
    "GIVE_WAY",
    "GUARDED_LEVEL_CROSSING",
    "HANDCART_PROHIBITED",
    "HEIGHT_LIMIT",
    "HORN_PROHIBITED",
    "HUMP_OR_ROUGH_ROAD",
    "LEFT_HAIR_PIN_BEND",
    "LEFT_HAND_CURVE",
    "LEFT_REVERSE_BEND",
    "LEFT_TURN_PROHIBITED",
    "LENGTH_LIMIT",
    "LOAD_LIMIT",
    "LOOSE_GRAVEL",
    "MEN_AT_WORK",
    "NARROW_BRIDGE",
    "NARROW_ROAD_AHEAD",
    "NO_ENTRY",
    "NO_PARKING",
    "NO_STOPPING_OR_STANDING",
    "OVERTAKING_PROHIBITED",
    "PASS_EITHER_SIDE",
    "PEDESTRIAN_CROSSING",
    "PEDESTRIAN_PROHIBITED",
    "PRIORITY_FOR_ONCOMING_VEHICLES",
    "QUAY_SIDE_OR_RIVER_BANK",
    "RESTRICTION_ENDS",
    "RIGHT_HAIR_PIN_BEND",
    "RIGHT_HAND_CURVE",
    "RIGHT_REVERSE_BEND",
    "RIGHT_TURN_PROHIBITED",
    "ROAD_WIDENS_AHEAD",
    "ROUNDABOUT",
    "SCHOOL_AHEAD",
    "SIDE_ROAD_LEFT",
    "SIDE_ROAD_RIGHT",
    "SLIPPERY_ROAD",
    "SPEED_LIMIT_15",
    "SPEED_LIMIT_20",
    "SPEED_LIMIT_30",
    "SPEED_LIMIT_40",
    "SPEED_LIMIT_5",
    "SPEED_LIMIT_50",
    "SPEED_LIMIT_60",
    "SPEED_LIMIT_70",
    "SPEED_LIMIT_80",
    "STAGGERED_INTERSECTION",
    "STEEP_ASCENT",
    "STEEP_DESCENT",
    "STOP",
    "STRAIGHT_PROHIBITED",
    "TONGA_PROHIBITED",
    "TRAFFIC_SIGNAL",
    "TRUCK_PROHIBITED",
    "TURN_RIGHT",
    "T_INTERSECTION",
    "UNGUARDED_LEVEL_CROSSING",
    "U_TURN_PROHIBITED",
    "WIDTH_LIMIT",
    "Y_INTERSECTION"
]

