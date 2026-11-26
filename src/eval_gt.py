"""Ground-truth 3D evaluation module (Sim(3) 7-DoF alignment & Tanks and Temples metrics).

Aligns predicted 3D meshes or point clouds (.ply, .glb, .obj) to ground truth
using 7-DoF Sim(3) (rotation, translation, and uniform scale) via FPFH-RANSAC
and multi-stage ICP, computing precision, recall, F-score, and median errors.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def load_point_cloud(path: Union[str, Path], max_points: int = 400_000) -> np.ndarray:
    """Load mesh or point cloud from file and return (N, 3) numpy array."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    ext = p.suffix.lower()
    pts: Optional[np.ndarray] = None

    # Handle .glb and .gltf with trimesh first for optimal scene/mesh unpacking
    if ext in (".glb", ".gltf"):
        try:
            import trimesh

            loaded = trimesh.load(str(p), force="mesh")
            if isinstance(loaded, trimesh.Scene):
                geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
                if geoms:
                    loaded = trimesh.util.concatenate(geoms)
            if isinstance(loaded, trimesh.Trimesh):
                n_samp = min(max_points, max(10_000, len(loaded.vertices) * 2))
                pts, _ = trimesh.sample.sample_surface(loaded, n_samp)
        except Exception:
            pts = None

    if pts is None:
        # Try loading as Open3D triangle mesh first (samples faces uniformly on .ply/.obj meshes)
        try:
            mesh = o3d.io.read_triangle_mesh(str(p))
            if len(mesh.triangles) > 0:
                n_sample = min(max_points, max(20_000, len(mesh.triangles)))
                pcd_sampled = mesh.sample_points_uniformly(number_of_points=n_sample)
                pts = np.asarray(pcd_sampled.points, dtype=np.float64)
        except Exception:
            pts = None

    if pts is None:
        # Try loading as Open3D point cloud
        try:
            pcd = o3d.io.read_point_cloud(str(p))
            if len(pcd.points) > 0:
                pts = np.asarray(pcd.points, dtype=np.float64)
        except Exception:
            pts = None

    if pts is None or len(pts) == 0:
        raise ValueError(f"Failed to load 3D geometry from {p}")

    if len(pts) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), max_points, replace=False)
        pts = pts[idx]

    return np.ascontiguousarray(pts, dtype=np.float64)


def _compute_heatmap_colors(errors_m: np.ndarray) -> np.ndarray:
    """Map errors in meters to green (<5cm), yellow (~15cm), red (>30cm) RGB colors."""
    n = len(errors_m)
    colors = np.zeros((n, 3), dtype=np.float64)

    # Green: [46, 204, 113] / 255.0 -> [0.18, 0.80, 0.44]
    # Yellow: [241, 196, 15] / 255.0 -> [0.95, 0.77, 0.06]
    # Red: [231, 76, 60] / 255.0 -> [0.91, 0.30, 0.24]
    c_green = np.array([0.18, 0.80, 0.44])
    c_yellow = np.array([0.95, 0.77, 0.06])
    c_red = np.array([0.91, 0.30, 0.24])

    m_low = errors_m <= 0.05
    m_mid = (errors_m > 0.05) & (errors_m <= 0.15)
    m_high = (errors_m > 0.15) & (errors_m <= 0.30)
    m_top = errors_m > 0.30

    colors[m_low] = c_green

    if np.any(m_mid):
        t = ((errors_m[m_mid] - 0.05) / 0.10)[:, None]
        colors[m_mid] = (1.0 - t) * c_green + t * c_yellow

    if np.any(m_high):
        t = ((errors_m[m_high] - 0.15) / 0.15)[:, None]
        colors[m_high] = (1.0 - t) * c_yellow + t * c_red

    colors[m_top] = c_red
    return colors


def export_heatmap_ply(out_path: Union[str, Path], pts: np.ndarray, errors_m: np.ndarray) -> None:
    """Save point cloud with colorized error heatmap to .ply file."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    colors = _compute_heatmap_colors(errors_m)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(p), pcd)


def _extract_above_ground(pcd: o3d.geometry.PointCloud, dist_thresh: float) -> o3d.geometry.PointCloud:
    """Extract above-ground object geometry by stripping dominant planar ground slab."""
    if len(pcd.points) < 50:
        return pcd
    try:
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=dist_thresh, ransac_n=3, num_iterations=1000
        )
        if len(inliers) >= int(0.25 * len(pcd.points)) and (len(pcd.points) - len(inliers)) >= 150:
            return pcd.select_by_index(inliers, invert=True)
    except Exception:
        pass
    return pcd


def evaluate_against_gt(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
    thresholds: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.30),
    roi_margin: float = 0.5,
) -> Dict[str, Union[float, int, dict, list]]:
    """Align pred_pts to gt_pts via 7-DoF Sim(3) and compute accuracy metrics."""
    pred_pts = np.asarray(pred_pts, dtype=np.float64)
    gt_pts = np.asarray(gt_pts, dtype=np.float64)

    if len(pred_pts) < 10 or len(gt_pts) < 10:
        raise ValueError("Point clouds must contain at least 10 points")

    # 1. Initial span ratio calculation & bounding box
    pred_span = float(np.linalg.norm(np.percentile(pred_pts, 95, axis=0) - np.percentile(pred_pts, 5, axis=0)))
    gt_span = float(np.linalg.norm(np.percentile(gt_pts, 95, axis=0) - np.percentile(gt_pts, 5, axis=0)))
    span_ratio = gt_span / max(pred_span, 1e-6)

    gt_min = np.min(gt_pts, axis=0)
    gt_max = np.max(gt_pts, axis=0)

    # 2. Downsampled representation and above-ground object isolation
    pcd_gt_full = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_pts))
    voxel_size = max(0.02, gt_span * 0.025)

    pcd_gt_down = pcd_gt_full.voxel_down_sample(voxel_size)
    if len(pcd_gt_down.points) < 50:
        pcd_gt_down = pcd_gt_full

    pcd_gt_obj = _extract_above_ground(pcd_gt_down, dist_thresh=max(0.08, gt_span * 0.015))
    if len(pcd_gt_obj.points) < 50:
        pcd_gt_obj = pcd_gt_down
    pcd_gt_obj.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 4.0, max_nn=35))
    gt_obj_pts = np.asarray(pcd_gt_obj.points)
    tree_gt_obj = cKDTree(gt_obj_pts)
    gt_obj_min = np.min(gt_obj_pts, axis=0)
    gt_obj_max = np.max(gt_obj_pts, axis=0)

    fpfh_gt = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_gt_obj, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.5, max_nn=100)
    )

    p5_gt = np.percentile(gt_obj_pts, 95, axis=0) - np.percentile(gt_obj_pts, 5, axis=0)
    gt_obj_span = float(np.linalg.norm(p5_gt))

    pcd_pred_raw = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred_pts))
    pcd_pred_coarse = pcd_pred_raw.voxel_down_sample(max(0.05, pred_span * 0.02))
    pcd_pred_obj_coarse = _extract_above_ground(pcd_pred_coarse, dist_thresh=max(0.12, pred_span * 0.015))
    pred_obj_pts_coarse = np.asarray(pcd_pred_obj_coarse.points)
    if len(pred_obj_pts_coarse) >= 50:
        p5_pred = np.percentile(pred_obj_pts_coarse, 95, axis=0) - np.percentile(pred_obj_pts_coarse, 5, axis=0)
        pred_obj_span = float(np.linalg.norm(p5_pred))
    else:
        pred_obj_span = pred_span

    obj_span_ratio = gt_obj_span / max(pred_obj_span, 1e-6)
    print(f"Diagnostics: gt_obj_span={gt_obj_span:.2f}m, pred_obj_span={pred_obj_span:.2f}m, obj_span_ratio={obj_span_ratio:.4f}, span_ratio={span_ratio:.4f}")

    raw_scales = [
        0.80 * obj_span_ratio, 1.0 * obj_span_ratio, 1.25 * obj_span_ratio,
        0.75 * span_ratio, 1.0 * span_ratio, 1.33 * span_ratio,
        0.80, 1.0, 1.20,
    ]
    scale_candidates: List[float] = []
    for sc in raw_scales:
        if sc < 0.01 or sc > 100.0:
            continue
        if not any(abs(sc - existing) / existing < 0.10 for existing in scale_candidates):
            scale_candidates.append(float(sc))

    best_ransac = None
    best_score = -1.0
    best_init_scale = obj_span_ratio
    est_meth_rigid = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False)
    distance_threshold = voxel_size * 2.5
    checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.85),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
    ]

    for s_cand in scale_candidates:
        pcd_pred_s = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred_pts * s_cand))
        pcd_pred_down = pcd_pred_s.voxel_down_sample(voxel_size)
        if len(pcd_pred_down.points) < 50:
            pcd_pred_down = pcd_pred_s
        pcd_pred_obj = _extract_above_ground(pcd_pred_down, dist_thresh=max(0.08, gt_span * 0.015))
        if len(pcd_pred_obj.points) < 50:
            pcd_pred_obj = pcd_pred_down
        pcd_pred_obj.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 4.0, max_nn=35))

        fpfh_pred = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_pred_obj, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.5, max_nn=100)
        )

        try:
            res_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                pcd_pred_obj,
                pcd_gt_obj,
                fpfh_pred,
                fpfh_gt,
                mutual_filter=True,
                max_correspondence_distance=distance_threshold,
                estimation_method=est_meth_rigid,
                ransac_n=3,
                checkers=checkers,
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 0.999),
            )
        except Exception:
            continue

        # Score candidate by bidirectional F1 between transformed pcd_pred_obj and pcd_gt_obj
        pts_obj_cand = np.asarray(pcd_pred_obj.points)
        T_cand = res_ransac.transformation
        pts_obj_trans = (T_cand[:3, :3] @ pts_obj_cand.T).T + T_cand[:3, 3]
        margin_eval = max(1.0, voxel_size * 4.0)
        in_gt_box = np.all((pts_obj_trans >= gt_obj_min - margin_eval) & (pts_obj_trans <= gt_obj_max + margin_eval), axis=-1)
        pts_obj_trans_roi = pts_obj_trans[in_gt_box] if np.sum(in_gt_box) >= 10 else pts_obj_trans

        d_p2g_cand, _ = tree_gt_obj.query(pts_obj_trans_roi, workers=-1)
        tree_cand_down = cKDTree(pts_obj_trans_roi)
        d_g2p_cand, _ = tree_cand_down.query(gt_obj_pts, workers=-1)
        cand_dist_thresh = voxel_size * 2.5
        p_cand = float(np.mean(d_p2g_cand < cand_dist_thresh))
        r_cand = float(np.mean(d_g2p_cand < cand_dist_thresh))
        f1_cand = float(2.0 * p_cand * r_cand / (p_cand + r_cand + 1e-8))

        if f1_cand > best_score:
            best_score = f1_cand
            best_ransac = res_ransac
            best_init_scale = s_cand

    if best_ransac is None or best_score < 0.05:
        # Fallback to centroid alignment
        c_pred = np.mean(pred_pts, axis=0) * obj_span_ratio
        c_gt = np.mean(gt_pts, axis=0)
        T_init = np.eye(4, dtype=np.float64)
        T_init[:3, 3] = c_gt - c_pred
        best_init_scale = obj_span_ratio
    else:
        T_init = best_ransac.transformation

    # 3. Fast 3-stage ROI-cropped ICP refinement (<10s)
    icp_vox = max(0.02, gt_span * 0.004)
    pcd_gt_icp = pcd_gt_full.voxel_down_sample(icp_vox)
    if len(pcd_gt_icp.points) < 50:
        pcd_gt_icp = pcd_gt_full

    pts_base_scaled = pred_pts * best_init_scale
    T_cur = T_init

    stages = [
        (voxel_size * 2.5, False, 50),
        (voxel_size * 1.2, False, 50),
        (voxel_size * 0.5, True, 40),
    ]

    last_fitness = 1.0
    last_rmse = 0.0

    for stage_dist, allow_scale, max_iter in stages:
        pts_trans_cur = (T_cur[:3, :3] @ pts_base_scaled.T).T + T_cur[:3, 3]
        stage_margin = max(1.2, stage_dist * 3.0)
        in_stage_box = np.all((pts_trans_cur >= gt_min - stage_margin) & (pts_trans_cur <= gt_max + stage_margin), axis=-1)
        if np.sum(in_stage_box) >= 300:
            pts_stage = pts_base_scaled[in_stage_box]
        else:
            pts_stage = pts_base_scaled
        pcd_stage = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_stage))
        pcd_stage_down = pcd_stage.voxel_down_sample(icp_vox)
        if len(pcd_stage_down.points) < 50:
            pcd_stage_down = pcd_stage

        meth = (
            o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
            if allow_scale
            else est_meth_rigid
        )

        try:
            icp_res = o3d.pipelines.registration.registration_icp(
                pcd_stage_down,
                pcd_gt_icp,
                stage_dist,
                T_cur,
                meth,
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
            )
            s_step = float(np.cbrt(np.abs(np.linalg.det(icp_res.transformation[:3, :3]))))
            if allow_scale and (s_step < 0.78 or s_step > 1.28):
                # Fall back to rigid ICP for this stage
                icp_res = o3d.pipelines.registration.registration_icp(
                    pcd_stage_down,
                    pcd_gt_icp,
                    stage_dist,
                    T_cur,
                    est_meth_rigid,
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
                )
            T_cur = icp_res.transformation
            last_fitness = float(icp_res.fitness)
            last_rmse = float(icp_res.inlier_rmse)
        except Exception:
            pass

    # Fold best_init_scale into T_full so transformation maps raw pred_pts -> gt_pts directly
    T_full = T_cur.copy()
    T_full[:3, :3] = T_cur[:3, :3] * best_init_scale
    total_scale = float(np.cbrt(np.abs(np.linalg.det(T_full[:3, :3]))))

    # Apply alignment to all predicted points
    pred_aligned = (T_full[:3, :3] @ pred_pts.T).T + T_full[:3, 3]

    # 4. ROI cropping: points within GT's bounding box + roi_margin
    roi_min = gt_min - roi_margin
    roi_max = gt_max + roi_margin

    in_roi = np.all((pred_aligned >= roi_min) & (pred_aligned <= roi_max), axis=-1)
    if np.sum(in_roi) >= 10:
        pred_roi = pred_aligned[in_roi]
    else:
        pred_roi = pred_aligned

    # 5. Distance and error evaluation via cKDTree
    gt_tree = cKDTree(gt_pts)
    pred_roi_tree = cKDTree(pred_roi)
    pred_full_tree = cKDTree(pred_aligned)

    d_p2g_roi, _ = gt_tree.query(pred_roi, workers=-1)
    d_g2p_roi, _ = pred_roi_tree.query(gt_pts, workers=-1)

    d_p2g_full, _ = gt_tree.query(pred_aligned, workers=-1)
    d_g2p_full, _ = pred_full_tree.query(gt_pts, workers=-1)

    med_p2g_roi = float(np.median(d_p2g_roi) * 100.0)
    med_g2p_roi = float(np.median(d_g2p_roi) * 100.0)
    chamfer_roi = float((np.mean(d_p2g_roi) + np.mean(d_g2p_roi)) * 0.5 * 100.0)

    med_p2g_full = float(np.median(d_p2g_full) * 100.0)
    med_g2p_full = float(np.median(d_g2p_full) * 100.0)
    chamfer_full = float((np.mean(d_p2g_full) + np.mean(d_g2p_full)) * 0.5 * 100.0)

    eval_result: Dict[str, Union[float, int, dict, list]] = {
        "scale": total_scale,
        "fitness": last_fitness,
        "inlier_rmse": last_rmse,
        "transformation": T_full.tolist(),
        "total_pred_points": int(len(pred_pts)),
        "total_gt_points": int(len(gt_pts)),
        "roi_points": int(len(pred_roi)),
        "median_p2g_cm": med_p2g_roi,
        "median_g2p_cm": med_g2p_roi,
        "mean_chamfer_cm": chamfer_roi,
        "roi": {
            "median_p2g_cm": med_p2g_roi,
            "median_g2p_cm": med_g2p_roi,
            "mean_chamfer_cm": chamfer_roi,
            "thresholds": {},
        },
        "full": {
            "median_p2g_cm": med_p2g_full,
            "median_g2p_cm": med_g2p_full,
            "mean_chamfer_cm": chamfer_full,
            "thresholds": {},
        },
        "aligned_pred_pts": pred_aligned,
        "roi_aligned_pred_pts": pred_roi,
        "roi_p2g_errors_m": d_p2g_roi,
        "full_p2g_errors_m": d_p2g_full,
    }

    for th in thresholds:
        th_key = f"{th:.2f}"
        p_roi = float(np.mean(d_p2g_roi < th))
        r_roi = float(np.mean(d_g2p_roi < th))
        f_roi = float(2 * p_roi * r_roi / (p_roi + r_roi + 1e-8))

        p_full = float(np.mean(d_p2g_full < th))
        r_full = float(np.mean(d_g2p_full < th))
        f_full = float(2 * p_full * r_full / (p_full + r_full + 1e-8))

        eval_result[f"precision_{th_key}"] = p_roi
        eval_result[f"recall_{th_key}"] = r_roi
        eval_result[f"f_score_{th_key}"] = f_roi

        eval_result["roi"]["thresholds"][th] = {  # type: ignore
            "precision": p_roi,
            "recall": r_roi,
            "f_score": f_roi,
        }
        eval_result["full"]["thresholds"][th] = {  # type: ignore
            "precision": p_full,
            "recall": r_full,
            "f_score": f_full,
        }

    return eval_result


def print_summary_table(results: Dict[str, Union[float, int, dict, list]], thresholds: Tuple[float, ...]) -> None:
    """Print clean summary table of Tanks & Temples metrics."""
    print("\n" + "=" * 70)
    print(" GROUND-TRUTH 3D RECONSTRUCTION EVALUATION (Tanks & Temples Protocol)")
    print("=" * 70)
    print(f" Recovered Scale:     {results['scale']:.5f}")
    print(f" Registration Fitness:{results['fitness']:.4f} (RMSE: {results['inlier_rmse']:.4f})")
    print(f" Evaluated Points:    ROI: {results['roi_points']:,} / Total: {results['total_pred_points']:,}")
    print("-" * 70)
    print(" ROI Metrics (Cropped to GT AABB + margin):")
    print(f"   Median P2G Error:  {results['roi']['median_p2g_cm']:.2f} cm")  # type: ignore
    print(f"   Median G2P Error:  {results['roi']['median_g2p_cm']:.2f} cm")  # type: ignore
    print(f"   Mean Chamfer Error:{results['roi']['mean_chamfer_cm']:.2f} cm")  # type: ignore
    print("-" * 70)
    print(f" {'Threshold':<12} | {'Precision':<12} | {'Recall':<12} | {'F-Score':<12}")
    print("-" * 70)
    for th in thresholds:
        th_key = f"{th:.2f}"
        p = results.get(f"precision_{th_key}", 0.0)
        r = results.get(f"recall_{th_key}", 0.0)
        f = results.get(f"f_score_{th_key}", 0.0)
        print(f" {th*100:4.1f} cm ({th:.2f}m) | {p*100:9.2f}%   | {r*100:9.2f}%   | {f*100:9.2f}%")
    print("-" * 70)
    print(f" Full-Scene Median Error: {results['full']['median_p2g_cm']:.2f} cm")  # type: ignore
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3D reconstruction against ground truth via Sim(3).")
    parser.add_argument("pred", type=str, help="Path to predicted mesh or point cloud (.ply, .glb, .obj)")
    parser.add_argument("gt", type=str, help="Path to ground truth mesh or point cloud (.ply, .glb, .obj)")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20, 0.30],
        help="Evaluation distance thresholds in meters (default: 0.05 0.10 0.20 0.30)",
    )
    parser.add_argument("--roi-margin", type=float, default=0.5, help="ROI AABB margin in meters (default: 0.5)")
    parser.add_argument("--out-heatmap", type=str, default=None, help="Optional output path for error heatmap .ply")
    args = parser.parse_args()

    t0 = time.perf_counter()
    print(f"Loading prediction from {args.pred}...")
    pred_pts = load_point_cloud(args.pred)
    print(f"Loaded {len(pred_pts):,} prediction points.")

    print(f"Loading ground truth from {args.gt}...")
    gt_pts = load_point_cloud(args.gt)
    print(f"Loaded {len(gt_pts):,} ground truth points.")

    print("Running Sim(3) 7-DoF alignment and computing metrics...")
    thresholds = tuple(float(t) for t in args.thresholds)
    res = evaluate_against_gt(pred_pts, gt_pts, thresholds=thresholds, roi_margin=args.roi_margin)

    print_summary_table(res, thresholds)

    if args.out_heatmap:
        print(f"Writing error heatmap to {args.out_heatmap}...")
        export_heatmap_ply(args.out_heatmap, res["roi_aligned_pred_pts"], res["roi_p2g_errors_m"])  # type: ignore
        print("Heatmap written.")

    print(f"Evaluation finished in {time.perf_counter() - t0:.2f}s.")


if __name__ == "__main__":
    main()
