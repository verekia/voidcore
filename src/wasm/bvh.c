#include "bvh.h"

#define MEM(type, byte_offset) WASM_PTR(type, byte_offset)

// Inline min/max — avoid LTO lowering __builtin_fminf to libc calls
static inline f32 bvh_min(f32 a, f32 b) { return a < b ? a : b; }
static inline f32 bvh_max(f32 a, f32 b) { return a > b ? a : b; }

// --- BVH persistent arena ---

static u32 bvh_arena_base;
static u32 bvh_arena_offset;
static u32 bvh_arena_size;

void bvh_arena_init(u32 base, u32 size) {
  bvh_arena_base = base;
  bvh_arena_offset = 0;
  bvh_arena_size = size;
}

u32 bvh_arena_alloc(u32 bytes, u32 align) {
  u32 current = bvh_arena_base + bvh_arena_offset;
  u32 aligned = (current + align - 1) & ~(align - 1);
  u32 next = aligned + bytes;
  if (next - bvh_arena_base > bvh_arena_size) return 0;
  bvh_arena_offset = next - bvh_arena_base;
  return aligned;
}

void bvh_arena_reset(void) {
  bvh_arena_offset = 0;
}

// --- BVH node: 32 bytes ---
// Bounds are stored as [minX, minY, minZ, maxX, maxY, maxZ] so that
// the slab test can index by precomputed near/far indices directly.

typedef struct {
  f32 bounds[6]; // [minX, minY, minZ, maxX, maxY, maxZ]
  u32 child_or_offset; // inner: right-child node index, leaf: first tri offset
  u32 meta;            // inner: split axis (0-2), leaf: count | LEAF_FLAG
} BVHNode;

#define LEAF_FLAG    0x80000000u
#define IS_LEAF(n)   ((n)->meta & LEAF_FLAG)
#define LEAF_COUNT(n) ((n)->meta & ~LEAF_FLAG)
#define SPLIT_AXIS(n) ((n)->meta)

#define SAH_BINS 32
#define MAX_LEAF_TRIS 4
#define BVH_MAX_STACK 64

typedef struct {
  f32 min[3];
  f32 max[3];
  u32 count;
} SAHBin;

// --- Ray-triangle intersection (Moller-Trumbore) ---

static f32 ray_triangle(const f32* origin, const f32* dir,
                         const f32* v0, const f32* v1, const f32* v2,
                         f32* out_normal) {
  f32 e1[3], e2[3];
  v3_sub(e1, v1, v0);
  v3_sub(e2, v2, v0);

  f32 h[3];
  v3_cross(h, dir, e2);
  f32 a = v3_dot(e1, h);

  if (a > -1e-8f && a < 1e-8f) return -1.0f;

  f32 f = 1.0f / a;
  f32 s[3];
  v3_sub(s, origin, v0);
  f32 u = f * v3_dot(s, h);
  if (u < 0.0f || u > 1.0f) return -1.0f;

  f32 q[3];
  v3_cross(q, s, e1);
  f32 v = f * v3_dot(dir, q);
  if (v < 0.0f || u + v > 1.0f) return -1.0f;

  f32 t = f * v3_dot(e2, q);
  if (t > 1e-8f) {
    v3_cross(out_normal, e1, e2);
    v3_normalize(out_normal, out_normal);
    return t;
  }
  return -1.0f;
}

// --- Ray-AABB: slab test with sign-based swap + early axis rejection ---
// near_idx/far_idx are precomputed per-ray from inv_dir sign:
//   axis 0: near=0,far=3 (pos dir) or near=3,far=0 (neg dir)
//   axis 1: near=1,far=4 or near=4,far=1
//   axis 2: near=2,far=5 or near=5,far=2

static inline int ray_aabb(const f32* origin, const f32* inv_dir,
                            const f32* bounds,
                            int near_x, int far_x,
                            int near_y, int far_y,
                            int near_z, int far_z,
                            f32 max_t) {
  f32 tmin = (bounds[near_x] - origin[0]) * inv_dir[0];
  f32 tmax = (bounds[far_x]  - origin[0]) * inv_dir[0];

  f32 tymin = (bounds[near_y] - origin[1]) * inv_dir[1];
  f32 tymax = (bounds[far_y]  - origin[1]) * inv_dir[1];

  if (tmin > tymax || tymin > tmax) return 0; // early reject after 2 axes

  if (tymin > tmin) tmin = tymin;
  if (tymax < tmax) tmax = tymax;

  f32 tzmin = (bounds[near_z] - origin[2]) * inv_dir[2];
  f32 tzmax = (bounds[far_z]  - origin[2]) * inv_dir[2];

  if (tmin > tzmax || tzmin > tmax) return 0;

  if (tzmin > tmin) tmin = tzmin;
  if (tzmax < tmax) tmax = tzmax;

  return tmax >= 0.0f && tmin < max_t;
}

// --- SAH-binned BVH build (recursive, depth-first) ---

static void bvh_build_recursive(BVHNode* nodes, u32* next_node, u32* tri_indices,
                                 const f32* centroids, const f32* tri_aabbs,
                                 u32 first, u32 count) {
  u32 ni = (*next_node)++;
  BVHNode* node = &nodes[ni];

  // Compute AABB over all triangles in this node
  f32 node_min[3] = {1e30f, 1e30f, 1e30f};
  f32 node_max[3] = {-1e30f, -1e30f, -1e30f};

  for (u32 i = first; i < first + count; i++) {
    u32 ti = tri_indices[i];
    const f32* taabb = &tri_aabbs[ti * 6];
    for (int a = 0; a < 3; a++) {
      if (taabb[a] < node_min[a]) node_min[a] = taabb[a];
      if (taabb[a + 3] > node_max[a]) node_max[a] = taabb[a + 3];
    }
  }

  node->bounds[0] = node_min[0];
  node->bounds[1] = node_min[1];
  node->bounds[2] = node_min[2];
  node->bounds[3] = node_max[0];
  node->bounds[4] = node_max[1];
  node->bounds[5] = node_max[2];

  // Leaf if few enough triangles
  if (count <= MAX_LEAF_TRIS) {
    node->child_or_offset = first;
    node->meta = count | LEAF_FLAG;
    return;
  }

  // Compute centroid bounds
  f32 cent_min[3] = {1e30f, 1e30f, 1e30f};
  f32 cent_max[3] = {-1e30f, -1e30f, -1e30f};
  for (u32 i = first; i < first + count; i++) {
    u32 ti = tri_indices[i];
    const f32* c = &centroids[ti * 3];
    for (int a = 0; a < 3; a++) {
      if (c[a] < cent_min[a]) cent_min[a] = c[a];
      if (c[a] > cent_max[a]) cent_max[a] = c[a];
    }
  }

  // SAH binning: find best axis and split
  f32 best_cost = (f32)count;
  int best_axis = -1;
  int best_split = -1;

  for (int axis = 0; axis < 3; axis++) {
    f32 extent = cent_max[axis] - cent_min[axis];
    if (extent < 1e-8f) continue;

    SAHBin bins[SAH_BINS];
    for (int b = 0; b < SAH_BINS; b++) {
      bins[b].min[0] = bins[b].min[1] = bins[b].min[2] = 1e30f;
      bins[b].max[0] = bins[b].max[1] = bins[b].max[2] = -1e30f;
      bins[b].count = 0;
    }

    f32 scale = (f32)SAH_BINS / extent;

    for (u32 i = first; i < first + count; i++) {
      u32 ti = tri_indices[i];
      f32 c = centroids[ti * 3 + axis];
      int bin = (int)((c - cent_min[axis]) * scale);
      if (bin >= SAH_BINS) bin = SAH_BINS - 1;

      const f32* taabb = &tri_aabbs[ti * 6];
      for (int a = 0; a < 3; a++) {
        if (taabb[a] < bins[bin].min[a]) bins[bin].min[a] = taabb[a];
        if (taabb[a + 3] > bins[bin].max[a]) bins[bin].max[a] = taabb[a + 3];
      }
      bins[bin].count++;
    }

    // Sweep from left: accumulate left-child area and count
    f32 left_area[SAH_BINS - 1];
    u32 left_count[SAH_BINS - 1];
    f32 run_min[3] = {1e30f, 1e30f, 1e30f};
    f32 run_max[3] = {-1e30f, -1e30f, -1e30f};
    u32 run_count = 0;

    for (int b = 0; b < SAH_BINS - 1; b++) {
      if (bins[b].count > 0) {
        for (int a = 0; a < 3; a++) {
          if (bins[b].min[a] < run_min[a]) run_min[a] = bins[b].min[a];
          if (bins[b].max[a] > run_max[a]) run_max[a] = bins[b].max[a];
        }
      }
      run_count += bins[b].count;
      left_count[b] = run_count;
      if (run_count > 0) {
        f32 dx = run_max[0] - run_min[0];
        f32 dy = run_max[1] - run_min[1];
        f32 dz = run_max[2] - run_min[2];
        left_area[b] = 2.0f * (dx * dy + dy * dz + dz * dx);
      } else {
        left_area[b] = 0.0f;
      }
    }

    // Sweep from right: evaluate SAH cost for each split
    run_min[0] = run_min[1] = run_min[2] = 1e30f;
    run_max[0] = run_max[1] = run_max[2] = -1e30f;
    run_count = 0;

    for (int b = SAH_BINS - 1; b >= 1; b--) {
      if (bins[b].count > 0) {
        for (int a = 0; a < 3; a++) {
          if (bins[b].min[a] < run_min[a]) run_min[a] = bins[b].min[a];
          if (bins[b].max[a] > run_max[a]) run_max[a] = bins[b].max[a];
        }
      }
      run_count += bins[b].count;
      f32 right_area;
      if (run_count > 0) {
        f32 dx = run_max[0] - run_min[0];
        f32 dy = run_max[1] - run_min[1];
        f32 dz = run_max[2] - run_min[2];
        right_area = 2.0f * (dx * dy + dy * dz + dz * dx);
      } else {
        right_area = 0.0f;
      }

      f32 cost = 1.0f +
                 left_area[b - 1] * (f32)left_count[b - 1] +
                 right_area * (f32)run_count;

      if (cost < best_cost) {
        best_cost = cost;
        best_axis = axis;
        best_split = b;
      }
    }
  }

  // No good split found: make leaf regardless of size
  if (best_axis == -1) {
    node->child_or_offset = first;
    node->meta = count | LEAF_FLAG;
    return;
  }

  // Partition triangles around split position
  f32 split_pos = cent_min[best_axis] +
                  (f32)best_split * (cent_max[best_axis] - cent_min[best_axis]) / (f32)SAH_BINS;

  u32 mid = first;
  u32 end = first + count - 1;
  while (mid <= end) {
    if (centroids[tri_indices[mid] * 3 + best_axis] < split_pos) {
      mid++;
    } else {
      u32 tmp = tri_indices[mid];
      tri_indices[mid] = tri_indices[end];
      tri_indices[end] = tmp;
      if (end == first) break;
      end--;
    }
  }

  u32 left_n = mid - first;
  if (left_n == 0 || left_n == count) {
    left_n = count / 2;
  }

  // Inner node: store split axis
  node->meta = (u32)best_axis;

  // Build left subtree (left child is always ni+1 in depth-first order)
  bvh_build_recursive(nodes, next_node, tri_indices, centroids, tri_aabbs,
                       first, left_n);

  // Right child index is wherever next_node is now
  node->child_or_offset = *next_node;

  bvh_build_recursive(nodes, next_node, tri_indices, centroids, tri_aabbs,
                       first + left_n, count - left_n);
}

// --- Shared traversal core ---

static f32 bvh_traverse(const BVHNode* nodes, const u32* tri_indices,
                          const f32* positions, const u32* indices,
                          const f32* origin, const f32* dir, const f32* inv_dir,
                          f32 max_t, f32* out_normal) {
  // Precompute near/far bound indices from ray direction sign
  int near_x = inv_dir[0] >= 0.0f ? 0 : 3;
  int near_y = inv_dir[1] >= 0.0f ? 1 : 4;
  int near_z = inv_dir[2] >= 0.0f ? 2 : 5;
  int far_x = 3 - near_x;
  int far_y = 5 - near_y;
  int far_z = 7 - near_z;

  f32 closest_t = max_t;
  int found = 0;

  u32 stack[BVH_MAX_STACK];
  int stack_top = 0;
  stack[stack_top++] = 0;

  while (stack_top > 0) {
    u32 ni = stack[--stack_top];
    const BVHNode* node = &nodes[ni];

    if (!ray_aabb(origin, inv_dir, node->bounds,
                  near_x, far_x, near_y, far_y, near_z, far_z, closest_t))
      continue;

    if (IS_LEAF(node)) {
      u32 first_tri = node->child_or_offset;
      u32 tc = LEAF_COUNT(node);
      for (u32 i = 0; i < tc; i++) {
        u32 ti = tri_indices[first_tri + i];
        u32 i0 = indices[ti * 3];
        u32 i1 = indices[ti * 3 + 1];
        u32 i2 = indices[ti * 3 + 2];

        f32 normal[3];
        f32 t = ray_triangle(origin, dir,
                              &positions[i0 * 3], &positions[i1 * 3], &positions[i2 * 3],
                              normal);
        if (t > 0.0f && t < closest_t) {
          closest_t = t;
          out_normal[0] = normal[0];
          out_normal[1] = normal[1];
          out_normal[2] = normal[2];
          found = 1;
        }
      }
    } else {
      // Near-child-first ordering based on split axis
      u32 axis = SPLIT_AXIS(node);
      u32 left = ni + 1;
      u32 right = node->child_or_offset;

      u32 near_child, far_child;
      if (dir[axis] >= 0.0f) {
        near_child = left;
        far_child = right;
      } else {
        near_child = right;
        far_child = left;
      }

      if (stack_top < BVH_MAX_STACK) stack[stack_top++] = far_child;
      if (stack_top < BVH_MAX_STACK) stack[stack_top++] = near_child;
    }
  }

  return found ? closest_t : -1.0f;
}

// --- WASM exports ---

__attribute__((export_name("vc_bvh_build")))
u32 vc_bvh_build(u32 vertices_offset, u32 indices_offset, u32 index_count,
                  u32 stride, u32 is_u32_indices) {
  u32 tri_count = index_count / 3;
  if (tri_count == 0) return 0;

  u32 max_idx = 0;

  if (is_u32_indices) {
    const u32* idx32 = MEM(const u32, indices_offset);
    for (u32 i = 0; i < index_count; i++) {
      if (idx32[i] > max_idx) max_idx = idx32[i];
    }
  } else {
    const unsigned short* idx16 = MEM(const unsigned short, indices_offset);
    for (u32 i = 0; i < index_count; i++) {
      if (idx16[i] > max_idx) max_idx = idx16[i];
    }
  }
  u32 vertex_count = max_idx + 1;

  u32 header_offset = bvh_arena_alloc(7 * 4, 4);
  if (header_offset == 0) return 0;
  u32* header = MEM(u32, header_offset);

  u32 positions_offset_alloc = bvh_arena_alloc(vertex_count * 3 * 4, 4);
  if (positions_offset_alloc == 0) return 0;
  f32* positions = MEM(f32, positions_offset_alloc);

  const f32* src_verts = MEM(const f32, vertices_offset);
  for (u32 i = 0; i < vertex_count; i++) {
    positions[i * 3 + 0] = src_verts[i * stride + 0];
    positions[i * 3 + 1] = src_verts[i * stride + 1];
    positions[i * 3 + 2] = src_verts[i * stride + 2];
  }

  u32 indices_alloc = bvh_arena_alloc(index_count * 4, 4);
  if (indices_alloc == 0) return 0;
  u32* indices_u32 = MEM(u32, indices_alloc);

  if (is_u32_indices) {
    const u32* src = MEM(const u32, indices_offset);
    for (u32 i = 0; i < index_count; i++) indices_u32[i] = src[i];
  } else {
    const unsigned short* src = MEM(const unsigned short, indices_offset);
    for (u32 i = 0; i < index_count; i++) indices_u32[i] = src[i];
  }

  u32 max_nodes = tri_count * 2;
  u32 nodes_alloc = bvh_arena_alloc(max_nodes * (u32)sizeof(BVHNode), 16);
  if (nodes_alloc == 0) return 0;
  BVHNode* nodes = MEM(BVHNode, nodes_alloc);

  u32 tri_indices_alloc = bvh_arena_alloc(tri_count * 4, 4);
  if (tri_indices_alloc == 0) return 0;
  u32* tri_indices = MEM(u32, tri_indices_alloc);
  for (u32 i = 0; i < tri_count; i++) tri_indices[i] = i;

  u32 centroids_alloc = bvh_arena_alloc(tri_count * 3 * 4, 4);
  if (centroids_alloc == 0) return 0;
  f32* centroids = MEM(f32, centroids_alloc);

  u32 tri_aabbs_alloc = bvh_arena_alloc(tri_count * 6 * 4, 4);
  if (tri_aabbs_alloc == 0) return 0;
  f32* tri_aabbs = MEM(f32, tri_aabbs_alloc);

  for (u32 t = 0; t < tri_count; t++) {
    u32 i0 = indices_u32[t * 3];
    u32 i1 = indices_u32[t * 3 + 1];
    u32 i2 = indices_u32[t * 3 + 2];

    const f32* v0 = &positions[i0 * 3];
    const f32* v1 = &positions[i1 * 3];
    const f32* v2 = &positions[i2 * 3];

    centroids[t * 3 + 0] = (v0[0] + v1[0] + v2[0]) / 3.0f;
    centroids[t * 3 + 1] = (v0[1] + v1[1] + v2[1]) / 3.0f;
    centroids[t * 3 + 2] = (v0[2] + v1[2] + v2[2]) / 3.0f;

    f32* tbb = &tri_aabbs[t * 6];
    tbb[0] = bvh_min(v0[0], bvh_min(v1[0], v2[0]));
    tbb[1] = bvh_min(v0[1], bvh_min(v1[1], v2[1]));
    tbb[2] = bvh_min(v0[2], bvh_min(v1[2], v2[2]));
    tbb[3] = bvh_max(v0[0], bvh_max(v1[0], v2[0]));
    tbb[4] = bvh_max(v0[1], bvh_max(v1[1], v2[1]));
    tbb[5] = bvh_max(v0[2], bvh_max(v1[2], v2[2]));
  }

  u32 next_node = 0;
  bvh_build_recursive(nodes, &next_node, tri_indices, centroids, tri_aabbs,
                       0, tri_count);

  header[0] = nodes_alloc;
  header[1] = next_node;
  header[2] = tri_indices_alloc;
  header[3] = positions_offset_alloc;
  header[4] = indices_alloc;
  header[5] = index_count;
  header[6] = tri_count;

  return header_offset;
}

__attribute__((export_name("vc_bvh_raycast")))
f32 vc_bvh_raycast(u32 bvh_offset, f32 ox, f32 oy, f32 oz,
                    f32 dx, f32 dy, f32 dz, f32 max_t, u32 result_offset) {
  const u32* header = MEM(const u32, bvh_offset);

  const BVHNode* nodes   = MEM(const BVHNode, header[0]);
  const u32* tri_indices = MEM(const u32, header[2]);
  const f32* positions   = MEM(const f32, header[3]);
  const u32* indices     = MEM(const u32, header[4]);

  f32 origin[3] = {ox, oy, oz};
  f32 dir[3] = {dx, dy, dz};
  f32 inv_dir[3] = {1.0f / dx, 1.0f / dy, 1.0f / dz};
  f32 normal[3] = {0, 0, 0};

  f32 t = bvh_traverse(nodes, tri_indices, positions, indices,
                        origin, dir, inv_dir, max_t, normal);

  f32* result_f32 = MEM(f32, result_offset);
  if (t >= 0.0f) {
    result_f32[0] = normal[0];
    result_f32[1] = normal[1];
    result_f32[2] = normal[2];
  }
  return t;
}

__attribute__((export_name("vc_bvh_alloc_reset")))
void vc_bvh_alloc_reset(void) {
  bvh_arena_reset();
}
