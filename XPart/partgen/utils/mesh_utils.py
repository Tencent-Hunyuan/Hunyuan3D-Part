import numpy as np
import trimesh
import torch
import torch.nn.functional as F
from skimage import measure
from typing import Callable, Tuple, List, Union
from torch import nn
from tqdm import tqdm
from einops import repeat
import traceback
import pymeshlab
import tempfile


def random_sample_pointcloud(mesh: trimesh.Trimesh, num=30000, seed=42):
    # points, face_idx = mesh.sample(num, return_index=True)
    points, face_idx = trimesh.sample.sample_surface(mesh, num, seed=seed)
    normals = mesh.face_normals[face_idx]
    rng = np.random.default_rng(seed=seed)
    index = rng.choice(num, num, replace=False)
    return points[index], normals[index]


def sharp_sample_pointcloud(mesh: trimesh.Trimesh, num=16384):
    V = mesh.vertices
    N = mesh.face_normals
    VN = mesh.vertex_normals
    F = mesh.faces
    VN2 = np.ones(V.shape[0])
    for i in range(3):
        dot = np.stack((VN2[F[:, i]], np.sum(VN[F[:, i]] * N, axis=-1)), axis=-1)
        VN2[F[:, i]] = np.min(dot, axis=-1)

    sharp_mask = VN2 < 0.985
    # collect edge
    edge_a = np.concatenate((F[:, 0], F[:, 1], F[:, 2]))
    edge_b = np.concatenate((F[:, 1], F[:, 2], F[:, 0]))
    sharp_edge = sharp_mask[edge_a] * sharp_mask[edge_b]
    edge_a = edge_a[sharp_edge > 0]
    edge_b = edge_b[sharp_edge > 0]

    sharp_verts_a = V[edge_a]
    sharp_verts_b = V[edge_b]
    sharp_verts_an = VN[edge_a]
    sharp_verts_bn = VN[edge_b]

    weights = np.linalg.norm(sharp_verts_b - sharp_verts_a, axis=-1)
    weights /= np.sum(weights)

    random_number = np.random.rand(num)
    w = np.random.rand(num, 1)
    index = np.searchsorted(weights.cumsum(), random_number)
    samples = w * sharp_verts_a[index] + (1 - w) * sharp_verts_b[index]
    normals = w * sharp_verts_an[index] + (1 - w) * sharp_verts_bn[index]
    return samples, normals


def SampleMesh(V, F, origin_num, seed=42):
    """Sample a mesh to get random points and normals.
    Args:
        V (np.ndarray): Vertices of the mesh.
        F (np.ndarray): Faces of the mesh.
        origin_num (int): Number of original faces to sample from.
    Returns:
        surface_data (dict): Dictionary containing sampled points and normals.
    The dictionary contains:
            - "random_surface": Sampled points and normals from the mesh.
            - "random_surface_fill": Boolean array indicating whether the points are from the fill region.
            - "sharp_surface": Sampled points and normals from the sharp edges of the mesh.
    """

    mesh = trimesh.Trimesh(vertices=V, faces=F[:origin_num])
    mesh_fill = trimesh.Trimesh(vertices=V, faces=F[origin_num:])

    area = mesh.area
    area_fill = mesh_fill.area
    sample_num = 499712 // 2
    num_fill = int(sample_num * (area_fill / (area + area_fill)))
    num = sample_num - num_fill
    # if not mesh.is_watertight:
    #     raise ValueError
    random_surface, random_normal = random_sample_pointcloud(mesh, num=num, seed=seed)
    if num_fill == 0:
        random_surface_fill, random_normal_fill = np.zeros((0, 3)), np.zeros((0, 3))
    else:
        random_surface_fill, random_normal_fill = random_sample_pointcloud(
            mesh_fill, num=num_fill, seed=seed
        )
    random_sharp_surface, sharp_normal = sharp_sample_pointcloud(mesh, num=sample_num)

    # save_surface
    surface = np.concatenate((random_surface, random_normal), axis=1).astype(np.float16)
    surface_fill = np.concatenate(
        (random_surface_fill, random_normal_fill), axis=1
    ).astype(np.float16)
    sharp_surface = np.concatenate((random_sharp_surface, sharp_normal), axis=1).astype(
        np.float16
    )

    a, b = np.ones(num), np.zeros(num_fill)

    surface_data = {
        "random_surface": np.concatenate((surface, surface_fill), axis=0),
        "random_surface_fill": np.concatenate((a, b)).astype(bool),
        "sharp_surface": sharp_surface,
    }

    return surface_data


def load_surface_points(
    rng,
    random_surface,
    sharpedge_surface,
    pc_size,
    pc_sharpedge_size,
    return_sharpedge_label=True,
    return_normal=True,
):
    """
    sample surface points based on pc_size and pc_sharpedge_size
    Args:
        rng: Random number generator
        random_surface: Array of random surface points
        sharpedge_surface: Array of sharp edge surface points
    Returns:
        surface: Array of surface points and normals
        geo_points: Array of geo points
    """

    surface_normal = []
    if pc_size > 0:
        ind = rng.choice(random_surface.shape[0], pc_size, replace=False)
        random_surface = random_surface[ind]
        if return_sharpedge_label:
            sharpedge_label = np.zeros((pc_size, 1))
            random_surface = np.concatenate((random_surface, sharpedge_label), axis=1)
        surface_normal.append(random_surface)

    if pc_sharpedge_size > 0:
        ind_sharpedge = rng.choice(
            sharpedge_surface.shape[0], pc_sharpedge_size, replace=False
        )
        sharpedge_surface = sharpedge_surface[ind_sharpedge]
        if return_sharpedge_label:
            sharpedge_label = np.ones((pc_sharpedge_size, 1))
            sharpedge_surface = np.concatenate(
                (sharpedge_surface, sharpedge_label), axis=1
            )
        surface_normal.append(sharpedge_surface)

    surface_normal = np.concatenate(surface_normal, axis=0)
    surface_normal = torch.FloatTensor(surface_normal)
    surface = surface_normal[:, 0:3]
    normal = surface_normal[:, 3:6]
    assert surface.shape[0] == pc_size + pc_sharpedge_size

    geo_points = 0.0
    normal = torch.nn.functional.normalize(normal, p=2, dim=1)
    if return_normal:
        surface = torch.cat([surface, normal], dim=-1)
    if return_sharpedge_label:
        surface = torch.cat([surface, surface_normal[:, -1:]], dim=-1)
    return surface, geo_points


def sample_bbox_points_from_trimesh(mesh, aabb, num_points, seed=42):
    _faces = mesh.faces
    _vertices = mesh.vertices
    _faces = np.reshape(_faces, (-1))
    num_parts = aabb.shape[0]
    _points = _points = torch.from_numpy(_vertices[_faces])
    _part_mask = torch.all(
        (_points[None, :, :3] >= aabb[:, :1]) & (_points[None, :, :3] <= aabb[:, 1:]),
        dim=-1,
    )
    _part_mask = torch.any(torch.reshape(_part_mask, (num_parts, -1, 3)), dim=-1)
    faces_idx_in_bbox = [torch.nonzero(x).squeeze(-1).numpy() for x in _part_mask]
    # in case some parts are empty(inside surface)
    valid_parts_mask = torch.tensor(
        [len(x) > 0 for x in faces_idx_in_bbox], dtype=torch.bool, device=_points.device
    )
    aabb = aabb[valid_parts_mask]
    # print(len(faces_idx_in_bbox), len(aabb))
    faces_idx_in_bbox = [x for x in faces_idx_in_bbox if len(x) > 0]
    num_valid_parts = len(faces_idx_in_bbox)
    # process valid parts
    mesh_in_bbox = mesh.submesh(faces_idx_in_bbox, append=False)
    points, normals = [], []
    for part in mesh_in_bbox:
        # part_points, face_idx = part.sample(num_points, return_index=True)
        part_points, face_idx = trimesh.sample.sample_surface(
            part, num_points, seed=seed
        )
        part_normals = part.face_normals[face_idx]
        points.append(torch.from_numpy(part_points))
        normals.append(torch.from_numpy(part_normals))
    out = torch.concat(
        [torch.stack(points, dim=0), torch.stack(normals, dim=0)], dim=-1
    )
    out = torch.concat(
        [
            out,
            torch.zeros(
                [num_valid_parts, num_points, 1], dtype=out.dtype, device=out.device
            ),
        ],
        dim=-1,
    )  # add sharp edge label
    return out, valid_parts_mask


def sample_surface_inbbox(
    rng,
    object_surface_raw,
    aabb,
    pc_size_bbox,
    return_normal=True,
    return_sharpedge_label=True,
):
    """
    Sample surface points within the bounding box defined by aabb.
    Args:
        object_surface_raw: Raw surface points from the object
        aabb: [K,2,3] Axis-aligned bounding box defined by min and max corners
        pc_size_bbox: Number of points to sample within the bounding box
    Returns:
        part_surface_inbbox: Sampled surface points within the bounding box
    """
    num_parts = aabb.shape[0]
    object_all_surface = torch.from_numpy(
        np.concatenate(
            [
                object_surface_raw["random_surface"],
                object_surface_raw["sharp_surface"],
            ],
            axis=0,
        )
    )  # [N,6]
    sharpedge_labels = torch.concat(
        [
            torch.zeros(object_surface_raw["random_surface"].shape[0], 1),
            torch.ones(object_surface_raw["sharp_surface"].shape[0], 1),
        ],
        dim=0,
    )
    sampled_masks = torch.all(
        (object_all_surface[None, :, :3] >= aabb[:, :1])
        & (object_all_surface[None, :, :3] <= aabb[:, 1:]),
        dim=-1,
    )
    surfaces = []
    valid_index = []
    for idx, sampled_mask in enumerate(sampled_masks):
        part_surface_inbbox = object_all_surface[sampled_mask]
        sharpedge_label = sharpedge_labels[sampled_mask]
        # TODO: drop inside parts
        if part_surface_inbbox.shape[0] == 0:
            continue
        try:
            ind = rng.choice(part_surface_inbbox.shape[0], pc_size_bbox, replace=False)
        except ValueError:
            ind = np.arange(part_surface_inbbox.shape[0])
            ind = np.concatenate([
                ind,
                rng.choice(
                    part_surface_inbbox.shape[0],
                    pc_size_bbox - part_surface_inbbox.shape[0],
                    replace=True,
                ),
            ])
        part_surface_inbbox = part_surface_inbbox[ind]
        sharpedge_label = sharpedge_label[ind]
        # point feat
        surface = part_surface_inbbox[:, 0:3]
        normal = part_surface_inbbox[:, 3:6]
        # TODO: check normal
        # normal = torch.nn.functional.normalize(normal, p=2, dim=1)
        if return_normal:
            surface = torch.cat([surface, normal], dim=-1)
        if return_sharpedge_label:
            surface = torch.cat(
                [surface, sharpedge_label],
                dim=-1,
            )
        surfaces.append(surface)
        valid_index.append(idx)
    surface = torch.stack(surfaces, dim=0)
    return surface, torch.tensor(valid_index)


def explode_mesh(mesh, explosion_scale=0.4):

    if isinstance(mesh, trimesh.Scene):
        scene = mesh
    elif isinstance(mesh, trimesh.Trimesh):
        print("Warning: Single mesh provided, can't create exploded view")
        scene = trimesh.Scene(mesh)
        return scene
    else:
        print(f"Warning: Unexpected mesh type: {type(mesh)}")
        scene = mesh

    if len(scene.geometry) <= 1:
        print("Only one geometry found - nothing to explode")
        return scene

    print(f"[EXPLODE_MESH] Starting mesh explosion with scale {explosion_scale}")
    print(f"[EXPLODE_MESH] Processing {len(scene.geometry)} parts")

    exploded_scene = trimesh.Scene()

    part_centers = []
    geometry_names = []

    for geometry_name, geometry in scene.geometry.items():
        if hasattr(geometry, "vertices"):
            transform = scene.graph[geometry_name][0]
            vertices_global = trimesh.transformations.transform_points(
                geometry.vertices, transform
            )
            center = np.mean(vertices_global, axis=0)
            part_centers.append(center)
            geometry_names.append(geometry_name)
            print(f"[EXPLODE_MESH] Part {geometry_name}: center = {center}")

    if not part_centers:
        print("No valid geometries with vertices found")
        return scene

    part_centers = np.array(part_centers)
    global_center = np.mean(part_centers, axis=0)

    print(f"[EXPLODE_MESH] Global center: {global_center}")

    for i, (geometry_name, geometry) in enumerate(scene.geometry.items()):
        if hasattr(geometry, "vertices"):
            if i < len(part_centers):
                part_center = part_centers[i]
                direction = part_center - global_center

                direction_norm = np.linalg.norm(direction)
                if direction_norm > 1e-6:
                    direction = direction / direction_norm
                else:
                    direction = np.random.randn(3)
                    direction = direction / np.linalg.norm(direction)

                offset = direction * explosion_scale
            else:
                offset = np.zeros(3)

            original_transform = scene.graph[geometry_name][0].copy()

            new_transform = original_transform.copy()
            new_transform[:3, 3] = new_transform[:3, 3] + offset

            exploded_scene.add_geometry(
                geometry, transform=new_transform, geom_name=geometry_name
            )

            print(
                f"[EXPLODE_MESH] Part {geometry_name}: moved by"
                f" {np.linalg.norm(offset):.4f}"
            )

    print("[EXPLODE_MESH] Mesh explosion complete")
    return exploded_scene


def generate_dense_grid_points(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    octree_depth: int = 7,
    indexing: str = "ij",
    octree_resolution: int = None,
):
    length = bbox_max - bbox_min
    num_cells = octree_resolution
    if octree_resolution is None:
        length = bbox_max - bbox_min
        num_cells = np.exp2(octree_depth)

    x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
    [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    xyz = xyz.reshape(-1, 3)
    grid_size = [int(num_cells) + 1, int(num_cells) + 1, int(num_cells) + 1]

    return xyz, grid_size, length


def extract_near_surface_volume_fn(input_tensor: torch.Tensor, alpha: float):
    """
    修复维度问题的PyTorch实现
    Args:
        input_tensor: shape [D, D, D], torch.float16
        alpha: 标量偏移值
    Returns:
        mask: shape [D, D, D], torch.int32 表面掩码
    """
    device = input_tensor.device
    D = input_tensor.shape[0]
    signed_val = 0.0

    # 添加偏移并处理无效值
    val = input_tensor + alpha
    valid_mask = val > -9000  # 假设-9000是无效值

    # 改进的邻居获取函数（保持维度一致）
    def get_neighbor(t, shift, axis):
        """根据指定轴进行位移并保持维度一致"""
        if shift == 0:
            return t.clone()

        # 确定填充轴（输入为[D, D, D]对应z,y,x轴）
        pad_dims = [0, 0, 0, 0, 0, 0]  # 格式：[x前，x后，y前，y后，z前，z后]

        # 根据轴类型设置填充
        if axis == 0:  # x轴（最后一个维度）
            pad_idx = 0 if shift > 0 else 1
            pad_dims[pad_idx] = abs(shift)
        elif axis == 1:  # y轴（中间维度）
            pad_idx = 2 if shift > 0 else 3
            pad_dims[pad_idx] = abs(shift)
        elif axis == 2:  # z轴（第一个维度）
            pad_idx = 4 if shift > 0 else 5
            pad_dims[pad_idx] = abs(shift)

        # 执行填充（添加batch和channel维度适配F.pad）
        padded = F.pad(
            t.unsqueeze(0).unsqueeze(0), pad_dims[::-1], mode="replicate"
        )  # 反转顺序适配F.pad

        # 构建动态切片索引
        slice_dims = [slice(None)] * 3  # 初始化为全切片
        if axis == 0:  # x轴（dim=2）
            if shift > 0:
                slice_dims[0] = slice(shift, None)
            else:
                slice_dims[0] = slice(None, shift)
        elif axis == 1:  # y轴（dim=1）
            if shift > 0:
                slice_dims[1] = slice(shift, None)
            else:
                slice_dims[1] = slice(None, shift)
        elif axis == 2:  # z轴（dim=0）
            if shift > 0:
                slice_dims[2] = slice(shift, None)
            else:
                slice_dims[2] = slice(None, shift)

        # 应用切片并恢复维度
        padded = padded.squeeze(0).squeeze(0)
        sliced = padded[slice_dims]
        return sliced

    # 获取各方向邻居（确保维度一致）
    left = get_neighbor(val, 1, axis=0)  # x方向
    right = get_neighbor(val, -1, axis=0)
    back = get_neighbor(val, 1, axis=1)  # y方向
    front = get_neighbor(val, -1, axis=1)
    down = get_neighbor(val, 1, axis=2)  # z方向
    up = get_neighbor(val, -1, axis=2)

    # 处理边界无效值（使用where保持维度一致）
    def safe_where(neighbor):
        return torch.where(neighbor > -9000, neighbor, val)

    left = safe_where(left)
    right = safe_where(right)
    back = safe_where(back)
    front = safe_where(front)
    down = safe_where(down)
    up = safe_where(up)

    # 计算符号一致性（转换为float32确保精度）
    sign = torch.sign(val.to(torch.float32))
    neighbors_sign = torch.stack(
        [
            torch.sign(left.to(torch.float32)),
            torch.sign(right.to(torch.float32)),
            torch.sign(back.to(torch.float32)),
            torch.sign(front.to(torch.float32)),
            torch.sign(down.to(torch.float32)),
            torch.sign(up.to(torch.float32)),
        ],
        dim=0,
    )

    # 检查所有符号是否一致
    same_sign = torch.all(neighbors_sign == sign, dim=0)

    # 生成最终掩码
    mask = (~same_sign).to(torch.int32)
    return mask * valid_mask.to(torch.int32)


@torch.no_grad()
def extract_geometry_fast(
    geometric_func: Callable,
    device: torch.device,
    batch_size: int = 1,
    bounds: Union[Tuple[float], List[float], float] = (
        -1.25,
        -1.25,
        -1.25,
        1.25,
        1.25,
        1.25,
    ),
    octree_depth: int = 7,
    num_chunks: int = 10000,
    disable: bool = True,
    mc_level: float = -1 / 512,
    octree_resolution: int = None,
    diffdmc=None,
    rotation_matrix=None,
    mc_mode="mc",
    dtype=torch.float16,
    min_resolution: int = 95,
):
    """

    Args:
        geometric_func:
        device:
        bounds:
        octree_depth:
        batch_size:
        num_chunks:
        disable:

    Returns:

    """

    if isinstance(bounds, float):
        bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
    if octree_resolution is None:
        octree_resolution = 2**octree_depth

    assert (
        octree_resolution >= 256
    ), "octree resolution must be at least 256 for fast inference"

    resolutions = []
    if octree_resolution < min_resolution:
        resolutions.append(octree_resolution)
    while octree_resolution >= min_resolution:
        resolutions.append(octree_resolution)
        octree_resolution = octree_resolution // 2
    resolutions.reverse()
    bbox_min = np.array(bounds[0:3])
    bbox_max = np.array(bounds[3:6])
    bbox_size = bbox_max - bbox_min

    dilate = nn.Conv3d(1, 1, 3, padding=1, bias=False, device=device, dtype=dtype)
    dilate.weight = torch.nn.Parameter(
        torch.ones(dilate.weight.shape, dtype=dtype, device=device)
    )

    xyz_samples, grid_size, length = generate_dense_grid_points(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        octree_resolution=resolutions[0],
        indexing="ij",
    )

    grid_size = np.array(grid_size)
    xyz_samples = torch.FloatTensor(xyz_samples).to(device).half()

    if mc_level == -1:
        print(
            f"Training with soft labels, inference with sigmoid and marching cubes"
            f" level 0."
        )
    elif mc_level == 0:
        print(f"VAE Trained with TSDF, inference with marching cubes level 0.")
    else:
        print(
            "VAE Trained with Occupancy, inference with marching cubes level"
            f" {mc_level}."
        )
    batch_logits = []
    for start in tqdm(
        range(0, xyz_samples.shape[0], num_chunks),
        desc=f"MC Level {mc_level} Implicit Function:",
        disable=disable,
        leave=False,
    ):
        queries = xyz_samples[start : start + num_chunks, :]
        batch_queries = repeat(queries, "p c -> b p c", b=batch_size)
        logits = geometric_func(batch_queries)
        if mc_level == -1:
            mc_level = 0
            print(
                f"Training with soft labels, inference with sigmoid and marching cubes"
                f" level 0."
            )
            logits = torch.sigmoid(logits) * 2 - 1
        batch_logits.append(logits)

    grid_logits = (
        torch.cat(batch_logits, dim=1)
        .view((batch_size, grid_size[0], grid_size[1], grid_size[2]))
        .half()
    )

    for octree_depth_now in resolutions[1:]:
        grid_size = np.array([octree_depth_now + 1] * 3)
        resolution = bbox_size / octree_depth_now
        next_index = torch.zeros(tuple(grid_size), dtype=dtype, device=device)
        if octree_depth_now == resolutions[-1]:
            next_logits = torch.full(
                next_index.shape, float("nan"), dtype=dtype, device=device
            )
        else:
            next_logits = torch.full(
                next_index.shape, -10000.0, dtype=dtype, device=device
            )

        FN = extract_near_surface_volume_fn
        curr_points = FN(grid_logits.squeeze(0), mc_level)
        curr_points += grid_logits.squeeze(0).abs() < min(
            0.95, 0.95 * 128 * 4 / octree_depth_now
        )
        if octree_depth_now > 510:
            expand_num = 0
        else:
            expand_num = 1
        for i in range(expand_num):
            curr_points = dilate(curr_points.unsqueeze(0).to(dtype)).squeeze(0)
        (cidx_x, cidx_y, cidx_z) = torch.where(curr_points > 0)
        next_index[cidx_x * 2, cidx_y * 2, cidx_z * 2] = 1
        for i in range(1):
            next_index = dilate(next_index.unsqueeze(0)).squeeze(0)
        nidx = torch.where(next_index > 0)
        next_points = torch.stack(nidx, dim=1)
        next_points = next_points * torch.tensor(
            resolution, device=device
        ) + torch.tensor(bbox_min, device=device)
        batch_logits = []
        for start in tqdm(
            range(0, next_points.shape[0], num_chunks),
            desc=f"MC Level {octree_depth_now + 1} Implicit Function:",
            disable=disable,
            leave=False,
        ):
            queries = next_points[start : start + num_chunks, :]
            batch_queries = repeat(queries, "p c -> b p c", b=batch_size)
            logits = geometric_func(batch_queries)
            if mc_level == -1:
                mc_level = 0
                print(
                    f"Training with soft labels, inference with sigmoid and marching"
                    f" cubes level 0."
                )
                logits = torch.sigmoid(logits) * 2 - 1
            batch_logits.append(logits)
        grid_logits = torch.cat(batch_logits, dim=1).half()
        next_logits[nidx] = grid_logits[0]
        grid_logits = next_logits.unsqueeze(0)
    # s_mc = time.time()
    mesh_v_f = []
    has_surface = np.zeros((batch_size,), dtype=np.bool_)
    for i in range(batch_size):
        try:
            if mc_mode == "mc":
                if len(resolutions) > 1:
                    mask = (next_index > 0).cpu().numpy()
                    grid_logits = grid_logits.cpu().numpy()
                    vertices, faces, normals, _ = measure.marching_cubes(
                        grid_logits[i], mc_level, method="lewiner", mask=mask
                    )
                else:
                    vertices, faces, normals, _ = measure.marching_cubes(
                        grid_logits[i].cpu().numpy(), mc_level, method="lewiner"
                    )
                vertices = vertices / (grid_size - 1) * bbox_size + bbox_min
                # vertices[:, [0, 1]] = vertices[:, [1, 0]]
            elif mc_mode == "dmc":
                torch.cuda.empty_cache()
                grid_logits = -grid_logits[i]
                grid_logits = grid_logits.to(torch.float32).contiguous()
                verts, faces = diffdmc(
                    grid_logits, deform=None, return_quads=False, normalize=False
                )
                verts = verts * torch.tensor(resolution, device=device) + torch.tensor(
                    bbox_min, device=device
                )
                vertices = verts.detach().cpu().numpy()
                faces = faces.detach().cpu().numpy()[:, ::-1]
            elif mc_mode == "odc":
                # https://github.com/KAIST-Visual-AI-Group/ODC
                from .occupancy_dual_contouring import occupancy_dual_contouring
                import torch.nn.functional as F

                odc = occupancy_dual_contouring("cuda")

                size = grid_logits.shape[-1]
                grid_logits = grid_logits.reshape(1, 1, size, size, size)

                def implicit_function(xyz):
                    xyz = xyz.reshape(1, -1, 1, 1, 3).float()
                    # print(grid_logits.dtype, xyz.dtype)
                    outputs = F.grid_sample(grid_logits.float(), xyz)
                    outputs = -outputs.reshape(-1)
                    return outputs

                num_cells = (
                    octree_resolution
                    if octree_resolution is not None
                    else np.exp2(octree_depth)
                )
                vertices, triangles = odc.extract_mesh(
                    implicit_function,
                    isolevel=mc_level,
                    min_coord=bbox_min,
                    max_coord=bbox_max,
                    num_grid=1024,
                )
                vertices = vertices.detach().cpu().numpy()
                faces = triangles.detach().cpu().numpy()[:, ::-1]
            else:
                raise ValueError(f"Unknown marching cubes mode: {mc_mode}")
            mesh_v_f.append((vertices.astype(np.float32), np.ascontiguousarray(faces)))
            has_surface[i] = True

        except ValueError:
            traceback.print_exc()
            mesh_v_f.append((None, None))
            has_surface[i] = False

        except RuntimeError:
            traceback.print_exc()
            mesh_v_f.append((None, None))
            has_surface[i] = False
    return mesh_v_f, has_surface


def pymeshlab2trimesh(mesh: pymeshlab.MeshSet):
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as temp_file:
        mesh.save_current_mesh(temp_file.name)
        mesh = trimesh.load(temp_file.name)
    # 检查加载的对象类型
    if isinstance(mesh, trimesh.Scene):
        combined_mesh = trimesh.Trimesh()
        # 如果是Scene，遍历所有的geometry并合并
        for geom in mesh.geometry.values():
            combined_mesh = trimesh.util.concatenate([combined_mesh, geom])
        mesh = combined_mesh
    return mesh


def trimesh2pymeshlab(mesh: trimesh.Trimesh):
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as temp_file:
        if isinstance(mesh, trimesh.scene.Scene):
            for idx, obj in enumerate(mesh.geometry.values()):
                if idx == 0:
                    temp_mesh = obj
                else:
                    temp_mesh = temp_mesh + obj
            mesh = temp_mesh
        mesh.export(temp_file.name)
        mesh = pymeshlab.MeshSet()
        mesh.load_new_mesh(temp_file.name)
    return mesh


def remove_overlength_edge(mesh: pymeshlab.MeshSet, max_length: float):
    mesh.apply_filter("compute_selection_by_edge_length", threshold=max_length)
    mesh.apply_filter("compute_selection_transfer_face_to_vertex", inclusive=False)
    mesh.apply_filter("meshing_remove_selected_vertices_and_faces")
    return mesh


def remove_floater(mesh: pymeshlab.MeshSet):
    mesh.apply_filter(
        "compute_selection_by_small_disconnected_components_per_face", nbfaceratio=0.005
    )
    mesh.apply_filter("compute_selection_transfer_face_to_vertex", inclusive=False)
    mesh.apply_filter("meshing_remove_selected_vertices_and_faces")
    return mesh


def fix_mesh(mesh: trimesh.Trimesh):
    ms = trimesh2pymeshlab(mesh)
    ms = remove_overlength_edge(ms, max_length=8 / 512)
    ms = remove_floater(ms)
    mesh = pymeshlab2trimesh(ms)
    return mesh
