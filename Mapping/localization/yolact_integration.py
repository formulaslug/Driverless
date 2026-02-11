"""
YOLACT Detection Integration Module

Input from YOLACT Team:
    - classes: Array of cone types [0, 1, 2, 3]
        - 0 = blue, 1 = yellow, 2 = small_orange, 3 = big_orange
    - scores (or confidence_scores): Detection confidence [0.0-1.0]
    - boxes (or bounding_boxes): [x1, y1, x2, y2] coordinates
    - masks (optional): Pixel masks [N, H, W]

Output to Path Planning:
    - DataFrame with 2D cone locations in vehicle frame
"""

import polars as pl
import numpy as np
from camera_transform import camera_transform

# Cone type mapping for clarity
CONE_TYPE_MAP = {
    0: "blue",
    1: "yellow", 
    2: "small_orange",
    3: "big_orange",
    255: "none"
}


def _to_numpy(x):
    """
    Convert torch tensor or list to numpy array. We need numpy array to work with Polar DataFrame
    .detach() removes gradient tracking
    .cpu() moves to CPU if on GPU
    .numpy() converts to NumPy array.
    """
    if x is None:
        return None
    if hasattr(x, "detach"):  # PyTorch tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)


def normalize_yolact_dict(yolact_output: dict) -> dict:
    """
    Normalize YOLACT output to handle different naming conventions.
    Accepts both standard YOLACT names (scores, boxes) and custom names.
    """
    out = dict(yolact_output)
    
    # Handle naming aliases
    if "scores" in out and "confidence_scores" not in out:
        out["confidence_scores"] = out["scores"]
    if "boxes" in out and "bounding_boxes" not in out:
        out["bounding_boxes"] = out["boxes"]
    
    # Convert to numpy
    out["classes"] = _to_numpy(out.get("classes"))
    out["confidence_scores"] = _to_numpy(out.get("confidence_scores"))
    out["bounding_boxes"] = _to_numpy(out.get("bounding_boxes"))
    out["masks"] = _to_numpy(out.get("masks"))  # optional, need the resolution from YOLACT
    
    return out


def parse_yolact_detections(yolact_output: dict, 
                           use_masks: bool = True) -> pl.DataFrame:
    """
    Convert YOLACT output into structured cone detection data.
    
    Args:
        yolact_output: Dictionary containing:
            {
                'classes': [0, 1, 2, 3, ...],
                'scores' (or 'confidence_scores'): [0.95, 0.87, ...],
                'boxes' (or 'bounding_boxes'): [[x1,y1,x2,y2], ...],
                'masks' (optional): [N, H, W] binary masks
            }
        use_masks: If True and masks available, use mask centroid instead of bbox center
            
    Returns:
        DataFrame with columns:
            - detection_id: Frame-local detection ID
            - cone_type: Type of cone (0-3)
            - confidence: Detection confidence score
            - pixel_u: Horizontal pixel (column)
            - pixel_v: Vertical pixel (row)
            - mask_index: Index for mask lookup (if masks provided)
    """
    y = normalize_yolact_dict(yolact_output)
    
    classes = y.get("classes") # [0, 1, 2, 0, ...] cone types
    scores = y.get("confidence_scores") # [0.95, 0.87, ...] confidences
    boxes = y.get("bounding_boxes") # [[x1,y1,x2,y2], ...] bounding boxes
    masks = y.get("masks") # [N, H, W] pixel masks (optional)
    
    # Validation
    if classes is None or scores is None or boxes is None:
        raise ValueError("Need classes + (scores/confidence_scores) + (boxes/bounding_boxes)")
    
    """
    Handle empty detections (YOLACT returns empty tensors)
    When YOLACT sees no cones, it returns empty tensors. Without this check, the code would crash
    """
    if classes.size == 0: # No cones detected in this frame
        return pl.DataFrame( 
            schema={
                "detection_id": pl.Int32,
                "cone_type": pl.Int32,
                "confidence": pl.Float32,
                "pixel_u": pl.Int32,
                "pixel_v": pl.Int32,
                "mask_index": pl.Int32
            }
        )
    
    # Length check
    if not (len(classes) == len(scores) == len(boxes)):
        raise ValueError(f"Length mismatch: classes={len(classes)}, scores={len(scores)}, boxes={len(boxes)}")
    
    detections = []
    has_masks = masks is not None and len(masks) > 0
    
    #Extract its type (0 = blue, 1 = yellow, etc), confidence score, and bounding box coordinates.
    
    for i in range(len(classes)):
        cone_type = int(classes[i])
        conf = float(scores[i])
        
        x1, y1, x2, y2 = boxes[i]
        
        # Default: bbox center
        center_u = int((float(x1) + float(x2)) / 2.0) # horizontal center
        center_v = int((float(y1) + float(y2)) / 2.0) # vertical center 
        
        # Better: mask centroid (if available and requested)
        if use_masks and has_masks and i < len(masks):
            mask = masks[i].astype(bool)
            ys, xs = np.nonzero(mask)
            if len(xs) > 0:
                center_u = int(np.round(xs.mean()))
                center_v = int(np.round(ys.mean()))
        
        # Store detection info in a list of dicts, then convert to DataFrame at the end
        detections.append({
            "detection_id": i, # 0, 1, 2, ... (resets each frame)
            "cone_type": cone_type, # 0-3
            "confidence": conf, # 0.0-1.
            "pixel_u": center_u, # horizontal pixel (column)
            "pixel_v": center_v, # vertical pixel (row)
            "mask_index": i if has_masks else None
        })
    
    return pl.DataFrame(detections)


def filter_low_confidence_detections(detections_df: pl.DataFrame, 
                                     min_confidence: float = 0.7) -> pl.DataFrame:
    """
    Remove detections with low confidence scores.
    
    Args:
        detections_df: Output from parse_yolact_detections()
        min_confidence: Minimum confidence threshold (0.0 to 1.0)
        
    Returns:
        Filtered DataFrame
    """
    return detections_df.filter(pl.col("confidence") >= min_confidence)


def attach_depth(detections_df: pl.DataFrame, 
                depth_map: np.ndarray,
                masks: np.ndarray = None) -> pl.DataFrame:
    """
    Attach depth information to detections.
    
    Args:
        detections_df: Filtered detections from YOLACT
        depth_map: 2D array of floats [H, W] with depth in meters
        masks: Optional masks [N, H, W] for median depth sampling
        
    Returns:
        DataFrame with added 'depth_m' column
    """
    depths = []
    
    for row in detections_df.iter_rows(named=True):
        u = row["pixel_u"]
        v = row["pixel_v"]

        # depth = float(depth_map[v, u])
        
        # If masks available, use median depth inside mask
        if masks is not None and row["mask_index"] is not None:
            mask = masks[row["mask_index"]].astype(bool)
            mask_depths = depth_map[mask] # Get all depth values inside mask 
            depth = float(np.median(mask_depths)) if len(mask_depths) > 0 else depth_map[v, u] # Use median (robust ot outliers)

        else:
            # Single pixel depth
            # Option 2 (less robust)
            depth = float(depth_map[v, u])
        
        depths.append(depth)
    
    return detections_df.with_columns(
        pl.Series("depth_m", depths)
    )


def process_frame(yolact_output: dict, 
                 depth_map: np.ndarray,
                 min_confidence: float = 0.7,
                 use_masks: bool = True) -> pl.DataFrame:
    """
    Complete pipeline: YOLACT detections → vehicle-frame cone locations.
    
    Args:
        yolact_output: Raw YOLACT output dict
        depth_map: Depth estimation output [H, W]
        min_confidence: Confidence threshold for filtering
        use_masks: Whether to use masks for centroid/depth
        
    Returns:
        DataFrame with columns:
            - detection_id, cone_type, confidence
            - pixel_u, pixel_v
            - depth_m
            - x_vehicle, y_vehicle, z_vehicle (cone location in vehicle frame)
    """
    # Step 1: Parse YOLACT detections
    # DataFrame with detection_id, cone_type, confidence, pixel_u, pixel_v
    detections = parse_yolact_detections(yolact_output, use_masks=use_masks)
    
    # Step 2: Filter low confidence
    # Removes weak detections
    detections = filter_low_confidence_detections(detections, min_confidence)
    
    # Handle empty detections
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
                "z_vehicle": pl.Float32
            }
        )
    
    # Step 3: Attach depth
    # Adds depth_m column
    masks = yolact_output.get("masks") if use_masks else None
    detections = attach_depth(detections, depth_map, masks)
    
    # Step 4: Transform to vehicle frame
    # Adds x_vehicle, y_vehicle, z_vehicle columns
    detections = camera_transform.transform_detections(detections)
    
    return detections


def get_2d_cone_locations(detections_df: pl.DataFrame) -> np.ndarray:
    """
    Extract 2D cone locations for path planning.
    
    Args:
        detections_df: Output from process_frame()
        
    Returns:
        2D array of shape [N, 2] with (x_vehicle, y_vehicle) coordinates
    """
    if len(detections_df) == 0:
        return np.array([]).reshape(0, 2)
    
    return detections_df.select(["x_vehicle", "y_vehicle"]).to_numpy()


# Example usage
if __name__ == "__main__":
    # Simulate YOLACT output
    dummy_yolact = {
        "classes": np.array([0, 1, 2]),
        "scores": np.array([0.95, 0.87, 0.92]),
        "boxes": np.array([
            [100, 200, 150, 250],
            [300, 400, 350, 450],
            [500, 600, 550, 650]
        ]),
        "masks": None  # Optional
    }
    
    # Simulate depth map (dummy data)
    dummy_depth = np.random.uniform(2.0, 10.0, size=(2592, 4608))
    
    # Process frame
    result = process_frame(dummy_yolact, dummy_depth, min_confidence=0.7)
    
    print("Detected cones:")
    print(result)
    
    print("\n2D locations for path planning:")
    locations = get_2d_cone_locations(result)
    print(locations)