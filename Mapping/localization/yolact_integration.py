import polars as pl
import numpy as np
from camera_transform import camera_transform

# Cone types
CONE_TYPE_MAP = {
    0: "blue",
    1: "yellow", 
    2: "small_orange",
    3: "big_orange",
    255: "none"
}

def _to_numpy(x):

    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def normalize_yolact_dict(yolact_output: dict) -> dict:
    out = dict(yolact_output)
    
    if "scores" in out and "confidence_scores" not in out: 
        out["confidence_scores"] = out["scores"]
    if "boxes" in out and "bounding_boxes" not in out:
        out["bounding_boxes"] = out["boxes"]
    
    out["classes"] = _to_numpy(out.get("classes"))
    out["confidence_scores"] = _to_numpy(out.get("confidence_scores"))
    out["bounding_boxes"] = _to_numpy(out.get("bounding_boxes"))
    out["masks"] = _to_numpy(out.get("masks"))
    
    return out


def parse_yolact_detections(yolact_output: dict, 
                           use_masks: bool = True) -> pl.DataFrame:
   
    y = normalize_yolact_dict(yolact_output)
    
    classes = y.get("classes")
    scores = y.get("confidence_scores")
    boxes = y.get("bounding_boxes") 
    masks = y.get("masks") 
    

    if classes is None or scores is None or boxes is None:
        raise ValueError("need classes + (scores/confidence_scores) + (boxes/bounding_boxes)")
    
    if classes.size == 0: 
        return pl.DataFrame( 
            schema = {
                "detection_id": pl.Int32,
                "cone_type": pl.Int32,
                "confidence": pl.Float32,
                "pixel_u": pl.Int32,
                "pixel_v": pl.Int32,
                "mask_index": pl.Int32
            }
        )
    
    if not (len(classes) == len(scores) == len(boxes)):
        raise ValueError(f"Length mismatch: classes={len(classes)}, scores={len(scores)}, boxes={len(boxes)}")
    
    detections = []
    has_masks = masks is not None and len(masks) > 0
    
    
    for i in range(len(classes)): 
        cone_type = int(classes[i]) 
        conf = float(scores[i])
        
        x1, y1, x2, y2 = boxes[i]
        
        center_u = int((float(x1) + float(x2)) / 2.0) # horizontal center
        center_v = int((float(y1) + float(y2)) / 2.0) # vertical center 
        
        if use_masks and has_masks and i < len(masks):
            mask = masks[i].astype(bool)
            ys, xs = np.nonzero(mask)
            if len(xs) > 0: 
                center_u = int(np.round(xs.mean()))
                center_v = int(np.round(ys.mean()))
        
        detections.append({
            "detection_id": i, 
            "cone_type": cone_type, 
            "confidence": conf, 
            "pixel_u": center_u, 
            "pixel_v": center_v, 
            "mask_index": i if has_masks else None
        })
    
    return pl.DataFrame(detections)


def filter_low_confidence_detections(detections_df: pl.DataFrame, 
                                     min_confidence: float = 0.7) -> pl.DataFrame:
    
    return detections_df.filter(pl.col("confidence") >= min_confidence)

def attach_depth(detections_df: pl.DataFrame, depth_map: np.ndarray, masks: np.ndarray = None) -> pl.DataFrame:

    depths = []
    
    for row in detections_df.iter_rows(named=True): 
        u = row["pixel_u"]
        v = row["pixel_v"]

        if masks is not None and row["mask_index"] is not None:
            mask = masks[row["mask_index"]].astype(bool)
            mask_depths = depth_map[mask]  
            depth = float(np.median(mask_depths)) if len(mask_depths) > 0 else depth_map[v, u]
        else:
            depth = float(depth_map[v, u])
        
        depths.append(depth)
    
    return detections_df.with_columns(
        pl.Series("depth_m", depths)
    )

def process_frame(yolact_output: dict, 
                  depth_map: np.ndarray,
                  min_confidence: float = 0.7,
                  use_masks: bool = True) -> pl.DataFrame:
    
    detections = parse_yolact_detections(yolact_output, use_masks=use_masks)
    
    detections = filter_low_confidence_detections(detections, min_confidence)
    
    if len(detections) == 0:
        return pl.DataFrame(
            schema={
                "detection_id": pl.Int32,
                "cone_type": pl.Int32,
                "confidence": pl.Float32,
                "pixel_u": pl.Int32,
                "pixel_v": pl.Int32,
                "depth_m": pl.Float32,
                "x_vehicle": pl.Float32,
                "y_vehicle": pl.Float32,
                "z_vehicle": pl.Float32,
            }
        )
    
    masks = yolact_output.get("masks") if use_masks else None
    detections = attach_depth(detections, depth_map, masks)
    
    detections = camera_transform.transform_detections(detections)
    
    return detections

def get_cone_measurements_for_kalman(detections_df: pl.DataFrame) -> list:
    if len(detections_df) == 0:
        return []
    
    locations = detections_df.select(["x_vehicle", "y_vehicle"]).to_numpy().astype(np.float32)
    
    class_probs = detections_df.select([
        (pl.col("cone_type") == 0).cast(pl.Float32).alias("blue_conf"),
        (pl.col("cone_type") == 1).cast(pl.Float32).alias("yellow_conf"), 
        (pl.col("cone_type") == 2).cast(pl.Float32).alias("small_orange_conf"),
        (pl.col("cone_type") == 3).cast(pl.Float32).alias("big_orange_conf"),
    ]).to_numpy()
    
    measurements = []
    for (x,y), probs in zip(locations, class_probs):
        measurements.append( ((float(x), float(y)), tuple(float(p) for p in probs)) )
    
    return measurements
# ((x_vehicle, y_vehicle), (blue_conf, yellow_conf, small_orange_conf, big_orange_conf))