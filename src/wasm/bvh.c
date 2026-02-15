#include "bvh.h"

// Alias for BVH code
#define MEM(type, byte_offset) WASM_PTR(type, byte_offset)

// Inline min/max to avoid LTO lowering __builtin_fminf to libc calls
static inline f32 bvh_min(f32 a, f32 b) { return a < b ? a : b; }
static inline f32 bvh_max(f32 a, f32 b) { return a > b ? a : b; }

// --- BVH persistent arena (separate from per-frame arena) ---

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

// --- BVH node struct (32 bytes, proper types avoid aliasing issues) ---

typedef struct {
  f32 aabb_min[3];
  f32 aabb_max[3];
  u32 child_or_offset; // inner: right-child index, leaf: first triangle offset
  u32 tri_count;       // 0 = inner node, >0 = leaf
} BVHNode;

#define SAH_BINS 12
#define MAX_LEAF_TRIS 4
#define TRAVERSAL_COST 1.0f
#define INTERSECTION_COST 1.0f
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

// --- Ray-AABB intersection (slab method) ---

static int ray_aabb(const f32* origin, const f32* inv_dir,
                    const f32* aabb_min, const f32* aabb_max, f32 max_t) {
  f32 t1, t2, tmin, tmax;

  t1 = (aabb_min[0] - origin[0]) * inv_dir[0];
  t2 = (aabb_max[0] - origin[0]) * inv_dir[0];
  tmin = bvh_min(t1, t2);
  tmax = bvh_max(t1, t2);

  t1 = (aabb_min[1] - origin[1]) * inv_dir[1];
  t2 = (aabb_max[1] - origin[1]) * inv_dir[1];
  tmin = bvh_max(tmin, bvh_min(t1, t2));
  tmax = bvh_min(tmax, bvh_max(t1, t2));

  t1 = (aabb_min[2] - origin[2]) * inv_dir[2];
  t2 = (aabb_max[2] - origin[2]) * inv_dir[2];
  tmin = bvh_max(tmin, bvh_min(t1, t2));
  tmax = bvh_min(tmax, bvh_max(t1, t2));

  return tmax >= bvh_max(tmin, 0.0f) && tmin < max_t;
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

  node->aabb_min[0] = node_min[0];
  node->aabb_min[1] = node_min[1];
  node->aabb_min[2] = node_min[2];
  node->aabb_max[0] = node_max[0];
  node->aabb_max[1] = node_max[1];
  node->aabb_max[2] = node_max[2];

  // Leaf if few enough triangles
  if (count <= MAX_LEAF_TRIS) {
    node->child_or_offset = first;
    node->tri_count = count;
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
  f32 best_cost = (f32)count * INTERSECTION_COST;
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

    // Sweep from left
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

    // Sweep from right
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

      f32 cost = TRAVERSAL_COST +
                 left_area[b - 1] * (f32)left_count[b - 1] * INTERSECTION_COST +
                 right_area * (f32)run_count * INTERSECTION_COST;

      if (cost < best_cost) {
        best_cost = cost;
        best_axis = axis;
        best_split = b;
      }
    }
  }

  // No good split found: make leaf
  if (best_axis == -1) {
    node->child_or_offset = first;
    node->tri_count = count;
    return;
  }

  // Partition triangles around split position
  f32 split_pos = cent_min[best_axis] +
                  (f32)best_split * (cent_max[best_axis] - cent_min[best_axis]) / (f32)SAH_BINS;

  u32 mid = first;
  u32 end = first + count - 1;
  while (mid <= end) {
    u32 ti = tri_indices[mid];
    if (centroids[ti * 3 + best_axis] < split_pos) {
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

  // Inner node marker
  node->tri_count = 0;

  // Build left subtree (left child at next_node = ni+1)
  bvh_build_recursive(nodes, next_node, tri_indices, centroids, tri_aabbs,
                       first, left_n);

  // Right child index is wherever next_node is now
  node->child_or_offset = *next_node;

  // Build right subtree
  bvh_build_recursive(nodes, next_node, tri_indices, centroids, tri_aabbs,
                       first + left_n, count - left_n);
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

  // Allocate BVH header (7 u32 values = 28 bytes)
  u32 header_offset = bvh_arena_alloc(7 * 4, 4);
  if (header_offset == 0) return 0;
  u32* header = MEM(u32, header_offset);

  // Extract positions (3 floats per vertex)
  u32 positions_offset_alloc = bvh_arena_alloc(vertex_count * 3 * 4, 4);
  if (positions_offset_alloc == 0) return 0;
  f32* positions = MEM(f32, positions_offset_alloc);

  const f32* src_verts = MEM(const f32, vertices_offset);
  for (u32 i = 0; i < vertex_count; i++) {
    positions[i * 3 + 0] = src_verts[i * stride + 0];
    positions[i * 3 + 1] = src_verts[i * stride + 1];
    positions[i * 3 + 2] = src_verts[i * stride + 2];
  }

  // Copy indices as u32
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

  // Allocate BVH nodes (max 2 * tri_count)
  u32 max_nodes = tri_count * 2;
  u32 nodes_alloc = bvh_arena_alloc(max_nodes * (u32)sizeof(BVHNode), 16);
  if (nodes_alloc == 0) return 0;
  BVHNode* nodes = MEM(BVHNode, nodes_alloc);

  // Allocate triangle reorder indices
  u32 tri_indices_alloc = bvh_arena_alloc(tri_count * 4, 4);
  if (tri_indices_alloc == 0) return 0;
  u32* tri_indices = MEM(u32, tri_indices_alloc);
  for (u32 i = 0; i < tri_count; i++) tri_indices[i] = i;

  // Allocate scratch: centroids and tri AABBs
  u32 centroids_alloc = bvh_arena_alloc(tri_count * 3 * 4, 4);
  if (centroids_alloc == 0) return 0;
  f32* centroids = MEM(f32, centroids_alloc);

  u32 tri_aabbs_alloc = bvh_arena_alloc(tri_count * 6 * 4, 4);
  if (tri_aabbs_alloc == 0) return 0;
  f32* tri_aabbs = MEM(f32, tri_aabbs_alloc);

  // Precompute per-triangle centroid and AABB
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

  // Build BVH recursively
  u32 next_node = 0;
  bvh_build_recursive(nodes, &next_node, tri_indices, centroids, tri_aabbs,
                       0, tri_count);

  // Write header
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

  u32 nodes_off   = header[0];
  u32 tri_idx_off = header[2];
  u32 pos_off     = header[3];
  u32 idx_off     = header[4];

  const BVHNode* nodes   = MEM(const BVHNode, nodes_off);
  const u32* tri_indices = MEM(const u32, tri_idx_off);
  const f32* positions   = MEM(const f32, pos_off);
  const u32* indices     = MEM(const u32, idx_off);

  f32 origin[3] = {ox, oy, oz};
  f32 dir[3] = {dx, dy, dz};
  f32 inv_dir[3] = {1.0f / dx, 1.0f / dy, 1.0f / dz};

  f32 closest_t = max_t;
  u32 closest_face = 0xFFFFFFFF;
  f32 closest_normal[3] = {0, 0, 0};

  // Stack-based BVH traversal
  u32 stack[BVH_MAX_STACK];
  int stack_top = 0;
  stack[stack_top++] = 0;

  while (stack_top > 0) {
    u32 ni = stack[--stack_top];
    const BVHNode* node = &nodes[ni];

    if (!ray_aabb(origin, inv_dir, node->aabb_min, node->aabb_max, closest_t))
      continue;

    if (node->tri_count > 0) {
      // Leaf: test each triangle
      u32 first_tri = node->child_or_offset;
      u32 tc = node->tri_count;
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
          closest_face = ti;
          closest_normal[0] = normal[0];
          closest_normal[1] = normal[1];
          closest_normal[2] = normal[2];
        }
      }
    } else {
      // Inner node: push right first so left is tested first
      u32 right_child = node->child_or_offset;
      if (stack_top < BVH_MAX_STACK) stack[stack_top++] = right_child;
      if (stack_top < BVH_MAX_STACK) stack[stack_top++] = ni + 1;
    }
  }

  // Write result
  u32* result_u32 = MEM(u32, result_offset);
  f32* result_f32 = MEM(f32, result_offset + 4);

  if (closest_face != 0xFFFFFFFF) {
    result_u32[0] = closest_face;
    result_f32[0] = closest_normal[0];
    result_f32[1] = closest_normal[1];
    result_f32[2] = closest_normal[2];
    return closest_t;
  }

  return -1.0f;
}

__attribute__((export_name("vc_bvh_alloc_reset")))
void vc_bvh_alloc_reset(void) {
  bvh_arena_reset();
}
