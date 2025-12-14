import numpy as np
import torch
import cv2 # For morphological operations, connected components
from scipy.spatial import KDTree
from skimage.morphology import skeletonize
from typing import Tuple, List, Any, Dict, Optional, Union # Added for type hinting and Union
from .graph_utils import nms_points
import time


# returns (x, y)
def get_points_and_scores_from_mask(mask, threshold):
    rcs = np.column_stack(np.where(mask > threshold)) # n rows, 2 columns
    xys = rcs[:, ::-1]
    scores = mask[mask > threshold]
    return xys, scores

def remove_small_regions(mask: np.ndarray, area_threshold: float, mask_threshold: float) -> np.ndarray:
    """
    remove small regions from mask, these are usually considered as noise
    
    Args:
        mask: input mask, type is ndarray, value is 0-1 float32,
             shape can be (B, H, W) or (H, W)
        area_threshold: area threshold, regions with area less than this value will be removed
        mask_threshold: threshold, regions with value greater than this value will be retained
        
    Returns:
        processed mask, same shape and dtype as input
    """
    # process input dimension
    original_shape = mask.shape
    original_dtype = mask.dtype
    
    if len(original_shape) == 3:  # batch processing case (B, H, W)
        batch_size, height, width = original_shape
        result = np.zeros_like(mask)
        
        for b in range(batch_size):
            # process each batch
            binary_mask = (mask[b] > mask_threshold).astype(np.uint8)
            filtered_mask = filter_small_regions(binary_mask, area_threshold)
            # keep original non-binary values
            result[b] = mask[b] * filtered_mask
    
    else:  # single image (H, W)
        binary_mask = (mask > mask_threshold).astype(np.uint8)
        filtered_mask = filter_small_regions(binary_mask, area_threshold)
        # keep original non-binary values
        result = mask * filtered_mask

    return result.astype(original_dtype)

def filter_small_regions(binary_mask: np.ndarray, area_threshold: float) -> np.ndarray:
    """
    helper function, remove small regions from binary mask
    use OpenCV's connectedComponentsWithStats for efficient processing
    
    Args:
        binary_mask: binary mask (0 or 1), type is ndarray
        area_threshold: area threshold
        
    Returns:
        processed binary mask
    """
    # connected components analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # vectorized method: create mapping from labels to retained status
    areas = stats[:, cv2.CC_STAT_AREA]
    keep_labels = areas >= area_threshold
    keep_labels[0] = False  # background is never retained
    
    filtered_mask = keep_labels[labels].astype(np.uint8)
    
    return filtered_mask

def extract_graph_points(keypoint_mask, road_mask, config):

    cost_time = {
        'total': 0,
        'remove_small_regions': 0,
        'get_kp_points_and_scores_from_mask': 0,
        'get_road_points_and_scores_from_mask': 0,
        'nms_concatenate_kp_and_road_points': 0
    }
    print(f"keypoint_mask.shape: {keypoint_mask.shape}, road_mask.shape: {road_mask.shape}")

    start_time = time.time()
    keypoint_mask = remove_small_regions(keypoint_mask, config.ITSC_AREA_THRESHOLD, config.ITSC_THRESHOLD)
    road_mask = remove_small_regions(road_mask, config.ROAD_AREA_THRESHOLD, config.ROAD_THRESHOLD)
    end_time = time.time()
    cost_time['remove_small_regions'] = end_time - start_time

    start_time = time.time()
    kp_candidates, kp_scores = get_points_and_scores_from_mask(keypoint_mask, config.ITSC_THRESHOLD)
    kp_scores = kp_scores + 0.1 # assign higher scores to preserve some keypoints
    end_time = time.time()
    print(f"kp_candidates: {kp_candidates.shape}, kp_scores: {kp_scores.shape}")
    cost_time['get_kp_points_and_scores_from_mask'] = end_time - start_time
    
    start_time = time.time()
    road_candidates, road_scores = get_points_and_scores_from_mask(road_mask, config.ROAD_THRESHOLD)
    end_time = time.time()
    print(f"road_candidates: {road_candidates.shape}, road_scores: {road_scores.shape}")
    cost_time['get_road_points_and_scores_from_mask'] = end_time - start_time
    
    start_time = time.time()
    all_candidates = np.concatenate([kp_candidates, road_candidates], axis=0)
    all_scores = np.concatenate([kp_scores, road_scores], axis=0)
    all_points = nms_points(all_candidates, all_scores, config.ROAD_NMS_RADIUS)
    end_time = time.time()
    print(f"after nms, all_points: {all_points.shape}")
    cost_time['nms_concatenate_kp_and_road_points'] = end_time - start_time
    cost_time['total'] = sum(cost_time.values())
    print(f"Cost time in function extract_graph_points: {cost_time}")
    return all_points, keypoint_mask, road_mask

def prepare_toponet_inputs(
    nodes_xy: np.ndarray, 
    config: dict,
    max_graph_points_padded: int 
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepares inputs (graph_points, pairs, valid_mask) for the TopoNet model.
    Args:
        nodes_xy: (N, 2) in (x, y) format, original image coordinates.
        config: Dict with TOPONET_SAMPLE_NUM, NEIGHBOR_RADIUS, MAX_NEIGHBOR_QUERIES, PATCH_SIZE.
        max_graph_points_padded: Padded size for graph_points tensor.
    Returns:
        Tuple of (graph_points_tensor, pairs_tensor, valid_mask_tensor).
    """
    num_nodes = nodes_xy.shape[0]
    # Ensure config values are accessed correctly (e.g., config['TOPO_SAMPLE_NUM'] or config.TOPO_SAMPLE_NUM)
    # Assuming config is an addict.Dict or similar, so attribute access might work.
    # For safety, using .get() with defaults or direct dict access if structure is known.
    topo_sample_num = config.get("TOPO_SAMPLE_NUM", 64) # Default if not in config
    max_neighbor_queries = config.get("MAX_NEIGHBOR_QUERIES", 16)
    neighbor_radius = config.get("NEIGHBOR_RADIUS", 100.0)
    patch_size = config.get("PATCH_SIZE", 1024)

    if num_nodes == 0:
        dummy_points = torch.zeros((1, max_graph_points_padded, 2), dtype=torch.float32)
        dummy_pairs = torch.zeros((1, topo_sample_num, max_neighbor_queries, 2), dtype=torch.long)
        dummy_valid = torch.zeros((1, topo_sample_num, max_neighbor_queries), dtype=torch.bool)
        return dummy_points, dummy_pairs, dummy_valid

    feature_map_size = patch_size // 16 # Assuming ViT patch size of 16
    feature_map_scale = patch_size / feature_map_size 
    graph_points_feat_scale = nodes_xy / feature_map_scale
    graph_points_feat_scale = np.clip(graph_points_feat_scale, 0, (feature_map_size - 1e-4))

    padded_graph_points = np.zeros((max_graph_points_padded, 2), dtype=np.float32)
    num_actual_nodes = min(num_nodes, max_graph_points_padded)
    if num_actual_nodes > 0: # Add this check
        padded_graph_points[:num_actual_nodes] = graph_points_feat_scale[:num_actual_nodes]
    graph_points_for_toponet_tensor = torch.from_numpy(padded_graph_points).unsqueeze(0).float()

    if num_actual_nodes == 0: # Handle case where no nodes fit after potential clipping/scaling
        sample_indices_in_graph = np.array([], dtype=int)
    else:
        sample_indices_in_graph = np.random.choice(
            np.arange(num_actual_nodes), 
            size=topo_sample_num, 
            replace=num_actual_nodes < topo_sample_num
        )

    all_pairs_list = []
    all_valid_list = []

    if num_actual_nodes > 1 and sample_indices_in_graph.size > 0:
        # Ensure query_points are only from actual nodes that were put into padded_graph_points
        valid_sample_indices = sample_indices_in_graph[sample_indices_in_graph < num_actual_nodes]
        if valid_sample_indices.size > 0: 
            query_points = padded_graph_points[valid_sample_indices]
            kdtree = KDTree(padded_graph_points[:num_actual_nodes])
            k_neighbors = min(max_neighbor_queries + 1, num_actual_nodes)
            distances, neighbor_indices_in_graph = kdtree.query(
                query_points, 
                k=k_neighbors, 
                distance_upper_bound=neighbor_radius / feature_map_scale
            )
        else: # No valid samples to query from
            distances = np.full((topo_sample_num, min(max_neighbor_queries + 1, num_actual_nodes if num_actual_nodes > 0 else 1) ), np.inf, dtype=np.float32)
            neighbor_indices_in_graph = np.full((topo_sample_num, min(max_neighbor_queries + 1, num_actual_nodes if num_actual_nodes > 0 else 1) ), num_actual_nodes, dtype=np.int32)

    else: # num_actual_nodes <=1 or no samples
        default_k = min(max_neighbor_queries + 1, num_actual_nodes if num_actual_nodes > 0 else 1)
        distances = np.full((topo_sample_num, default_k), np.inf, dtype=np.float32)
        neighbor_indices_in_graph = np.full((topo_sample_num, default_k), 0, dtype=np.int32)

    for i in range(topo_sample_num):
        if num_actual_nodes == 0 or i >= len(sample_indices_in_graph): # if i is out of bounds for samples taken
            current_pairs_np = np.full((max_neighbor_queries, 2), 0, dtype=np.int32) # Fill with dummy index 0
            current_valid_np = np.zeros(max_neighbor_queries, dtype=bool)
        else:
            source_node_idx_in_padded = sample_indices_in_graph[i]
            if source_node_idx_in_padded >= num_actual_nodes: # Sampled a padded index (should not happen if logic above is correct)
                current_pairs_np = np.full((max_neighbor_queries, 2), source_node_idx_in_padded, dtype=np.int32)
                current_valid_np = np.zeros(max_neighbor_queries, dtype=bool)
            else:
                current_neighbor_indices_for_sample = neighbor_indices_in_graph[i, :] 
                current_distances_for_sample = distances[i, :]
                sample_pairs_for_source = []
                sample_valid_for_source = []
                count = 0
                for k_idx, neighbor_target_idx_in_padded in enumerate(current_neighbor_indices_for_sample):
                    if count >= max_neighbor_queries:
                        break
                    condition1 = (neighbor_target_idx_in_padded == source_node_idx_in_padded)
                    condition2 = (current_distances_for_sample[k_idx] == np.inf)
                    condition3 = (neighbor_target_idx_in_padded >= num_actual_nodes) # Target is a padded node
                    if condition1 or condition2 or condition3:
                        continue
                    sample_pairs_for_source.append([source_node_idx_in_padded, neighbor_target_idx_in_padded])
                    sample_valid_for_source.append(True)
                    count += 1
                current_pairs_np = np.array(sample_pairs_for_source, dtype=np.int32) if sample_pairs_for_source else np.empty((0,2), dtype=np.int32)
                current_valid_np = np.array(sample_valid_for_source, dtype=bool) if sample_valid_for_source else np.empty((0,), dtype=bool)
                padding_needed = max_neighbor_queries - len(current_pairs_np)
                if padding_needed > 0:
                    pad_pairs_np = np.full((padding_needed, 2), source_node_idx_in_padded, dtype=np.int32) # Pad with source index
                    pad_valid_np = np.zeros(padding_needed, dtype=bool)
                    current_pairs_np = np.vstack([current_pairs_np, pad_pairs_np]) if len(current_pairs_np) > 0 else pad_pairs_np
                    current_valid_np = np.concatenate([current_valid_np, pad_valid_np]) if len(current_valid_np) > 0 else pad_valid_np
        all_pairs_list.append(current_pairs_np)
        all_valid_list.append(current_valid_np)

    pairs_tensor = torch.from_numpy(np.array(all_pairs_list)).unsqueeze(0).long() 
    valid_mask_tensor = torch.from_numpy(np.array(all_valid_list)).unsqueeze(0).bool()
    
    return graph_points_for_toponet_tensor, pairs_tensor, valid_mask_tensor

def filter_edges_from_scores(
    nodes_xy: np.ndarray, 
    pairs_indices: torch.Tensor, # [1, NumSamples, MaxNeighbors, 2] from TopoNet input
    edge_scores: torch.Tensor,   # [1, NumSamples, MaxNeighbors, 1] from TopoNet output
    valid_mask: torch.Tensor,    # [1, NumSamples, MaxNeighbors] from TopoNet input
    score_threshold: float,
    num_actual_nodes: int # Number of non-padded nodes in the graph_points tensor used for TopoNet
) -> np.ndarray:
    """
    Filters edges based on their predicted scores.
    Args:
        nodes_xy: Original node coordinates (N_orig, 2) in (x,y) original image scale.
        pairs_indices: Indices from TopoNet, referencing the (potentially padded) graph_points tensor.
        edge_scores: Predicted scores for each pair from TopoNet.
        valid_mask: Boolean mask indicating valid pairs used by TopoNet.
        score_threshold: Threshold to keep an edge.
        num_actual_nodes: The number of actual, unpadded nodes that `pairs_indices` refer to.

    Returns:
        np.ndarray: Array of final edge indices (M, 2) referencing original `nodes_xy`.
    """
    if nodes_xy.shape[0] == 0 or num_actual_nodes == 0:
        return np.empty((0, 2), dtype=np.int32)

    # Convert tensors to numpy
    pairs_indices_np = pairs_indices.squeeze(0).cpu().numpy()
    edge_scores_np = edge_scores.squeeze(0).cpu().numpy()
    valid_mask_np = valid_mask.squeeze(0).cpu().numpy()

    final_edges = set()
    num_samples, max_neighbors, _ = pairs_indices_np.shape
    for i in range(num_samples):
        for j in range(max_neighbors):
            if valid_mask_np[i, j] and edge_scores_np[i, j, 0] > score_threshold:
                u_idx_padded, v_idx_padded = pairs_indices_np[i, j, 0], pairs_indices_np[i, j, 1]
                # Ensure indices are within the range of *actual* (unpadded) nodes
                # and also within the bounds of the original nodes_xy that will be returned.
                if (u_idx_padded < num_actual_nodes and v_idx_padded < num_actual_nodes and
                    u_idx_padded < nodes_xy.shape[0] and v_idx_padded < nodes_xy.shape[0] and 
                    u_idx_padded != v_idx_padded):
                    final_edges.add(tuple(sorted((u_idx_padded, v_idx_padded)))) 
    
    return np.array(list(final_edges), dtype=np.int32)

def convert_to_geojson(nodes_xy: np.ndarray, edges: np.ndarray, image_id: Optional[str] = None) -> Dict:
    """
    Converts nodes and edges to optimized GeoJSON format with separated nodes and edges.
    Filters out isolated nodes (degree=0).
    
    Args:
        nodes_xy: Array of node coordinates (N, 2) in (x, y) format.
        edges: Array of edge indices (M, 2).
        image_id: Optional identifier for the image.

    Returns:
        dict: Optimized GeoJSON FeatureCollection with nodes and edges structure.
    """
    # Remove duplicate edges (undirected)
    unique_edges = set()
    for edge in edges:
        if (0 <= edge[0] < len(nodes_xy) and 0 <= edge[1] < len(nodes_xy) and edge[0] != edge[1]):
            unique_edges.add(tuple(sorted([int(edge[0]), int(edge[1])])))

    # === Step 1: find all points appearing in edges ===
    used_nodes = set()
    for u, v in unique_edges:
        used_nodes.add(u)
        used_nodes.add(v)

    # === Step 2: create mapping from old index to new index ===
    index_map = {old: new for new, old in enumerate(sorted(used_nodes))}

    # === Step 3: build new node list ===
    nodes_list = []
    for old_idx, new_idx in index_map.items():
        x, y = nodes_xy[old_idx]
        nodes_list.append({
            "id": new_idx,
            "x": float(x),
            "y": float(y)
        })

    # === Step 4: remap edge indices ===
    edges_list = []
    for i, (u, v) in enumerate(sorted(unique_edges)):
        edges_list.append({
            "id": i,
            "source": index_map[u],
            "target": index_map[v]
        })

    # === Step 5: metadata ===
    feature_collection_properties = {
        "format_version": "2.0",
        "total_nodes": len(nodes_list),
        "total_edges": len(edges_list),
        "coordinate_system": "image"
    }
    if image_id:
        feature_collection_properties["image_id"] = image_id

    return {
        "type": "FeatureCollection",
        "properties": feature_collection_properties,
        "nodes": nodes_list,
        "edges": edges_list,
        "features": []
    }

from pydantic import BaseModel, Field # Add Field here

# Pydantic models for GeoJSON output
class Geometry(BaseModel):
    type: str = "LineString"
    coordinates: List[List[float]] # List of [lon, lat] pairs

class GeoJSONFeature(BaseModel):
    type: str = "Feature"
    properties: Dict[str, Any] = Field(default_factory=dict)
    geometry: Geometry

class GeoJSONFeatureCollection(BaseModel):
    type: str = "FeatureCollection"
    features: List[GeoJSONFeature] 