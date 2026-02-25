// Geometry Worker – Offloads heavy geometry computations to a background thread.
//
// This web worker handles CPU-intensive geometry operations that would otherwise block
// the main thread and cause frame drops. It runs BVH construction, palette baking,
// mesh merging, smooth normal computation, and geometry merging entirely off the main
// thread, returning typed array results via postMessage with transferable objects.
//
// The worker is self-contained: all needed math functions are inlined (no imports)
// so it works as a standalone module in any bundler environment.
//
// Supported operations:
//   buildBVH             – Builds a BVH (bounding volume hierarchy) for raycasting.
//   bakePalette          – Resolves palette entries into per-vertex color attributes.
//   mergeStaticIntoSkinned – Merges a static geometry into a skinned one at a bone.
//   computeSmoothNormals – Averages normals for gap-free inverted hull outlines.
//   mergeGeometries      – Combines multiple geometries into one.

// ─── Inlined Math Functions ──────────────────────────────────────────────────

function mat4Create(): Float32Array {
  const out = new Float32Array(16)
  out[0] = 1
  out[5] = 1
  out[10] = 1
  out[15] = 1
  return out
}

function mat4Invert(out: Float32Array, a: Float32Array): Float32Array | null {
  const a00 = a[0]!,
    a01 = a[1]!,
    a02 = a[2]!,
    a03 = a[3]!
  const a10 = a[4]!,
    a11 = a[5]!,
    a12 = a[6]!,
    a13 = a[7]!
  const a20 = a[8]!,
    a21 = a[9]!,
    a22 = a[10]!,
    a23 = a[11]!
  const a30 = a[12]!,
    a31 = a[13]!,
    a32 = a[14]!,
    a33 = a[15]!

  const b00 = a00 * a11 - a01 * a10
  const b01 = a00 * a12 - a02 * a10
  const b02 = a00 * a13 - a03 * a10
  const b03 = a01 * a12 - a02 * a11
  const b04 = a01 * a13 - a03 * a11
  const b05 = a02 * a13 - a03 * a12
  const b06 = a20 * a31 - a21 * a30
  const b07 = a20 * a32 - a22 * a30
  const b08 = a20 * a33 - a23 * a30
  const b09 = a21 * a32 - a22 * a31
  const b10 = a21 * a33 - a23 * a31
  const b11 = a22 * a33 - a23 * a32

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06
  if (Math.abs(det) < 1e-8) return null
  det = 1 / det

  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det

  return out
}

function mat4Multiply(out: Float32Array, a: Float32Array, b: Float32Array): Float32Array {
  const a00 = a[0]!,
    a01 = a[1]!,
    a02 = a[2]!,
    a03 = a[3]!
  const a10 = a[4]!,
    a11 = a[5]!,
    a12 = a[6]!,
    a13 = a[7]!
  const a20 = a[8]!,
    a21 = a[9]!,
    a22 = a[10]!,
    a23 = a[11]!
  const a30 = a[12]!,
    a31 = a[13]!,
    a32 = a[14]!,
    a33 = a[15]!

  let b0 = b[0]!,
    b1 = b[1]!,
    b2 = b[2]!,
    b3 = b[3]!
  out[0] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30
  out[1] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31
  out[2] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32
  out[3] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33

  b0 = b[4]!
  b1 = b[5]!
  b2 = b[6]!
  b3 = b[7]!
  out[4] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30
  out[5] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31
  out[6] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32
  out[7] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33

  b0 = b[8]!
  b1 = b[9]!
  b2 = b[10]!
  b3 = b[11]!
  out[8] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30
  out[9] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31
  out[10] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32
  out[11] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33

  b0 = b[12]!
  b1 = b[13]!
  b2 = b[14]!
  b3 = b[15]!
  out[12] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30
  out[13] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31
  out[14] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32
  out[15] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33

  return out
}

function vec3TransformMat4(out: Float32Array, a: Float32Array, m: Float32Array): Float32Array {
  const x = a[0]!,
    y = a[1]!,
    z = a[2]!
  const w = m[3]! * x + m[7]! * y + m[11]! * z + m[15]!
  const invW = w !== 0 ? 1 / w : 1
  out[0] = (m[0]! * x + m[4]! * y + m[8]! * z + m[12]!) * invW
  out[1] = (m[1]! * x + m[5]! * y + m[9]! * z + m[13]!) * invW
  out[2] = (m[2]! * x + m[6]! * y + m[10]! * z + m[14]!) * invW
  return out
}

function aabbFromPoints(out: Float32Array, positions: Float32Array, count: number): Float32Array {
  out[0] = Infinity
  out[1] = Infinity
  out[2] = Infinity
  out[3] = -Infinity
  out[4] = -Infinity
  out[5] = -Infinity
  for (let i = 0; i < count; i++) {
    const x = positions[i * 3]!
    const y = positions[i * 3 + 1]!
    const z = positions[i * 3 + 2]!
    if (x < out[0]!) out[0] = x
    if (y < out[1]!) out[1] = y
    if (z < out[2]!) out[2] = z
    if (x > out[3]!) out[3] = x
    if (y > out[4]!) out[4] = y
    if (z > out[5]!) out[5] = z
  }
  return out
}

// ─── BVH Construction ────────────────────────────────────────────────────────

const BIN_COUNT = 12
const MAX_LEAF_TRIANGLES = 4
const TRAVERSAL_COST = 1.0
const INTERSECTION_COST = 1.5

function surfaceArea(minX: number, minY: number, minZ: number, maxX: number, maxY: number, maxZ: number): number {
  const ex = maxX - minX,
    ey = maxY - minY,
    ez = maxZ - minZ
  return 2 * (ex * ey + ey * ez + ez * ex)
}

interface BVHResult {
  bvhBuffer: ArrayBuffer
  triOrder: Uint32Array
  nodeCount: number
}

function buildBVH(positions: Float32Array, indices: Uint16Array | Uint32Array): BVHResult {
  const triCount = (indices.length / 3) | 0

  const centroids = new Float32Array(triCount * 3)
  const triAABBs = new Float32Array(triCount * 6)

  for (let t = 0; t < triCount; t++) {
    const i0 = indices[t * 3]! * 3
    const i1 = indices[t * 3 + 1]! * 3
    const i2 = indices[t * 3 + 2]! * 3
    const x0 = positions[i0]!,
      y0 = positions[i0 + 1]!,
      z0 = positions[i0 + 2]!
    const x1 = positions[i1]!,
      y1 = positions[i1 + 1]!,
      z1 = positions[i1 + 2]!
    const x2 = positions[i2]!,
      y2 = positions[i2 + 1]!,
      z2 = positions[i2 + 2]!

    centroids[t * 3] = (x0 + x1 + x2) / 3
    centroids[t * 3 + 1] = (y0 + y1 + y2) / 3
    centroids[t * 3 + 2] = (z0 + z1 + z2) / 3

    triAABBs[t * 6] = Math.min(x0, x1, x2)
    triAABBs[t * 6 + 1] = Math.min(y0, y1, y2)
    triAABBs[t * 6 + 2] = Math.min(z0, z1, z2)
    triAABBs[t * 6 + 3] = Math.max(x0, x1, x2)
    triAABBs[t * 6 + 4] = Math.max(y0, y1, y2)
    triAABBs[t * 6 + 5] = Math.max(z0, z1, z2)
  }

  const triOrder = new Uint32Array(triCount)
  for (let i = 0; i < triCount; i++) triOrder[i] = i

  const maxNodes = Math.max(1, triCount * 2)
  const nodeBuffer = new ArrayBuffer(maxNodes * 32)
  const floatNodes = new Float32Array(nodeBuffer)
  const intNodes = new Int32Array(nodeBuffer)
  let nodeCount = 0

  const binMin = new Float32Array(BIN_COUNT * 3)
  const binMax = new Float32Array(BIN_COUNT * 3)
  const binCnt = new Int32Array(BIN_COUNT)
  const prefixMin = new Float32Array(BIN_COUNT * 3)
  const prefixMax = new Float32Array(BIN_COUNT * 3)
  const prefixCnt = new Int32Array(BIN_COUNT)
  const suffixMin = new Float32Array(BIN_COUNT * 3)
  const suffixMax = new Float32Array(BIN_COUNT * 3)
  const suffixCnt = new Int32Array(BIN_COUNT)

  const buildNode = (start: number, end: number): number => {
    const nodeIdx = nodeCount++
    const count = end - start
    const o = nodeIdx * 8

    let minX = Infinity,
      minY = Infinity,
      minZ = Infinity
    let maxX = -Infinity,
      maxY = -Infinity,
      maxZ = -Infinity
    for (let i = start; i < end; i++) {
      const t = triOrder[i]! * 6
      if (triAABBs[t]! < minX) minX = triAABBs[t]!
      if (triAABBs[t + 1]! < minY) minY = triAABBs[t + 1]!
      if (triAABBs[t + 2]! < minZ) minZ = triAABBs[t + 2]!
      if (triAABBs[t + 3]! > maxX) maxX = triAABBs[t + 3]!
      if (triAABBs[t + 4]! > maxY) maxY = triAABBs[t + 4]!
      if (triAABBs[t + 5]! > maxZ) maxZ = triAABBs[t + 5]!
    }

    floatNodes[o] = minX
    floatNodes[o + 1] = minY
    floatNodes[o + 2] = minZ
    floatNodes[o + 4] = maxX
    floatNodes[o + 5] = maxY
    floatNodes[o + 6] = maxZ

    if (count <= MAX_LEAF_TRIANGLES) {
      intNodes[o + 3] = -count
      intNodes[o + 7] = start
      return nodeIdx
    }

    let cMinX = Infinity,
      cMinY = Infinity,
      cMinZ = Infinity
    let cMaxX = -Infinity,
      cMaxY = -Infinity,
      cMaxZ = -Infinity
    for (let i = start; i < end; i++) {
      const t = triOrder[i]! * 3
      const cx = centroids[t]!,
        cy = centroids[t + 1]!,
        cz = centroids[t + 2]!
      if (cx < cMinX) cMinX = cx
      if (cy < cMinY) cMinY = cy
      if (cz < cMinZ) cMinZ = cz
      if (cx > cMaxX) cMaxX = cx
      if (cy > cMaxY) cMaxY = cy
      if (cz > cMaxZ) cMaxZ = cz
    }

    const parentSA = surfaceArea(minX, minY, minZ, maxX, maxY, maxZ)
    const leafCost = count * INTERSECTION_COST

    let bestCost = Infinity
    let bestAxis = -1
    let bestBin = -1

    for (let axis = 0; axis < 3; axis++) {
      const cMin = axis === 0 ? cMinX : axis === 1 ? cMinY : cMinZ
      const cMax = axis === 0 ? cMaxX : axis === 1 ? cMaxY : cMaxZ
      if (cMax - cMin < 1e-6) continue

      const binInvSize = BIN_COUNT / (cMax - cMin)

      for (let b = 0; b < BIN_COUNT; b++) {
        binMin[b * 3] = Infinity
        binMin[b * 3 + 1] = Infinity
        binMin[b * 3 + 2] = Infinity
        binMax[b * 3] = -Infinity
        binMax[b * 3 + 1] = -Infinity
        binMax[b * 3 + 2] = -Infinity
        binCnt[b] = 0
      }

      for (let i = start; i < end; i++) {
        const t = triOrder[i]!
        const c = centroids[t * 3 + axis]!
        let b = ((c - cMin) * binInvSize) | 0
        if (b >= BIN_COUNT) b = BIN_COUNT - 1
        binCnt[b] = (binCnt[b] ?? 0) + 1
        const ta = t * 6
        const bm = b * 3
        if (triAABBs[ta]! < binMin[bm]!) binMin[bm] = triAABBs[ta]!
        if (triAABBs[ta + 1]! < binMin[bm + 1]!) binMin[bm + 1] = triAABBs[ta + 1]!
        if (triAABBs[ta + 2]! < binMin[bm + 2]!) binMin[bm + 2] = triAABBs[ta + 2]!
        if (triAABBs[ta + 3]! > binMax[bm]!) binMax[bm] = triAABBs[ta + 3]!
        if (triAABBs[ta + 4]! > binMax[bm + 1]!) binMax[bm + 1] = triAABBs[ta + 4]!
        if (triAABBs[ta + 5]! > binMax[bm + 2]!) binMax[bm + 2] = triAABBs[ta + 5]!
      }

      prefixCnt[0] = binCnt[0]!
      prefixMin[0] = binMin[0]!
      prefixMin[1] = binMin[1]!
      prefixMin[2] = binMin[2]!
      prefixMax[0] = binMax[0]!
      prefixMax[1] = binMax[1]!
      prefixMax[2] = binMax[2]!
      for (let b = 1; b < BIN_COUNT; b++) {
        prefixCnt[b] = prefixCnt[b - 1]! + binCnt[b]!
        const p = b * 3,
          pp = (b - 1) * 3,
          bm = b * 3
        prefixMin[p] = Math.min(prefixMin[pp]!, binMin[bm]!)
        prefixMin[p + 1] = Math.min(prefixMin[pp + 1]!, binMin[bm + 1]!)
        prefixMin[p + 2] = Math.min(prefixMin[pp + 2]!, binMin[bm + 2]!)
        prefixMax[p] = Math.max(prefixMax[pp]!, binMax[bm]!)
        prefixMax[p + 1] = Math.max(prefixMax[pp + 1]!, binMax[bm + 1]!)
        prefixMax[p + 2] = Math.max(prefixMax[pp + 2]!, binMax[bm + 2]!)
      }

      const last = BIN_COUNT - 1
      suffixCnt[last] = binCnt[last]!
      suffixMin[last * 3] = binMin[last * 3]!
      suffixMin[last * 3 + 1] = binMin[last * 3 + 1]!
      suffixMin[last * 3 + 2] = binMin[last * 3 + 2]!
      suffixMax[last * 3] = binMax[last * 3]!
      suffixMax[last * 3 + 1] = binMax[last * 3 + 1]!
      suffixMax[last * 3 + 2] = binMax[last * 3 + 2]!
      for (let b = last - 1; b >= 0; b--) {
        suffixCnt[b] = suffixCnt[b + 1]! + binCnt[b]!
        const s = b * 3,
          sn = (b + 1) * 3,
          bm = b * 3
        suffixMin[s] = Math.min(suffixMin[sn]!, binMin[bm]!)
        suffixMin[s + 1] = Math.min(suffixMin[sn + 1]!, binMin[bm + 1]!)
        suffixMin[s + 2] = Math.min(suffixMin[sn + 2]!, binMin[bm + 2]!)
        suffixMax[s] = Math.max(suffixMax[sn]!, binMax[bm]!)
        suffixMax[s + 1] = Math.max(suffixMax[sn + 1]!, binMax[bm + 1]!)
        suffixMax[s + 2] = Math.max(suffixMax[sn + 2]!, binMax[bm + 2]!)
      }

      for (let b = 0; b < BIN_COUNT - 1; b++) {
        const leftCnt = prefixCnt[b]!
        const rightCnt = suffixCnt[b + 1]!
        if (leftCnt === 0 || rightCnt === 0) continue

        const p = b * 3,
          sn = (b + 1) * 3
        const leftSA = surfaceArea(
          prefixMin[p]!,
          prefixMin[p + 1]!,
          prefixMin[p + 2]!,
          prefixMax[p]!,
          prefixMax[p + 1]!,
          prefixMax[p + 2]!,
        )
        const rightSA = surfaceArea(
          suffixMin[sn]!,
          suffixMin[sn + 1]!,
          suffixMin[sn + 2]!,
          suffixMax[sn]!,
          suffixMax[sn + 1]!,
          suffixMax[sn + 2]!,
        )

        const cost =
          TRAVERSAL_COST +
          (leftSA / parentSA) * leftCnt * INTERSECTION_COST +
          (rightSA / parentSA) * rightCnt * INTERSECTION_COST

        if (cost < bestCost) {
          bestCost = cost
          bestAxis = axis
          bestBin = b
        }
      }
    }

    if (bestAxis === -1 || bestCost >= leafCost) {
      intNodes[o + 3] = -count
      intNodes[o + 7] = start
      return nodeIdx
    }

    const cMin = bestAxis === 0 ? cMinX : bestAxis === 1 ? cMinY : cMinZ
    const cMax = bestAxis === 0 ? cMaxX : bestAxis === 1 ? cMaxY : cMaxZ
    const binInvSize = BIN_COUNT / (cMax - cMin)

    let mid = start
    for (let i = start; i < end; i++) {
      const t = triOrder[i]!
      const c = centroids[t * 3 + bestAxis]!
      let b = ((c - cMin) * binInvSize) | 0
      if (b >= BIN_COUNT) b = BIN_COUNT - 1
      if (b <= bestBin) {
        const tmp = triOrder[mid]!
        triOrder[mid] = triOrder[i]!
        triOrder[i] = tmp
        mid++
      }
    }

    if (mid === start || mid === end) mid = (start + end) >> 1

    const leftChild = buildNode(start, mid)
    const rightChild = buildNode(mid, end)

    intNodes[o + 3] = rightChild
    intNodes[o + 7] = leftChild

    return nodeIdx
  }

  buildNode(0, triCount)

  // Trim to actual used nodes
  const usedBuffer = new ArrayBuffer(nodeCount * 32)
  new Float32Array(usedBuffer).set(floatNodes.subarray(0, nodeCount * 8))
  return { bvhBuffer: usedBuffer, triOrder, nodeCount }
}

// ─── Palette Baking ──────────────────────────────────────────────────────────

interface SerializedPaletteEntry {
  color: [number, number, number]
  emissive?: [number, number, number]
  emissiveIntensity?: number
  tiledAoLayerIndex: number
  tiledAoIntensity: number
  tiledNormalLayerIndex: number
  tiledNormalIntensity: number
  tiledNormalScale: number
  color2?: [number, number, number]
  noiseScale?: number
}

interface BakePaletteResult {
  colors: Float32Array
  emissiveColors: Float32Array
  tiledNormalData: Float32Array | null
  noiseColorData: Float32Array | null
}

function bakePaletteCompute(
  materialIndices: Uint8Array | null,
  vertexCount: number,
  palette: SerializedPaletteEntry[],
  hasTiledNormals: boolean,
  hasNoiseColor: boolean,
): BakePaletteResult {
  const colors = new Float32Array(vertexCount * 4)
  const emissiveColors = new Float32Array(vertexCount * 4)
  const tiledNormalData = hasTiledNormals ? new Float32Array(vertexCount * 4) : null
  const noiseColorData = hasNoiseColor ? new Float32Array(vertexCount * 4) : null

  for (let i = 0; i < vertexCount; i++) {
    const idx = materialIndices ? materialIndices[i]! : 0
    const entry: SerializedPaletteEntry = palette[idx] ?? {
      color: [1, 1, 1],
      tiledAoLayerIndex: 0,
      tiledAoIntensity: 0,
      tiledNormalLayerIndex: 0,
      tiledNormalIntensity: 0,
      tiledNormalScale: 0,
    }
    const o = i * 4
    colors[o] = entry.color[0]
    colors[o + 1] = entry.color[1]
    colors[o + 2] = entry.color[2]
    colors[o + 3] = entry.tiledAoLayerIndex / 255

    const intensity = entry.emissiveIntensity ?? 0
    emissiveColors[o] = (entry.emissive?.[0] ?? 0) * intensity
    emissiveColors[o + 1] = (entry.emissive?.[1] ?? 0) * intensity
    emissiveColors[o + 2] = (entry.emissive?.[2] ?? 0) * intensity
    emissiveColors[o + 3] = entry.tiledAoLayerIndex > 0 ? entry.tiledAoIntensity : 0

    if (tiledNormalData) {
      tiledNormalData[o] = entry.tiledNormalLayerIndex / 255
      tiledNormalData[o + 1] = entry.tiledNormalLayerIndex > 0 ? entry.tiledNormalIntensity : 0
      tiledNormalData[o + 2] = entry.tiledNormalLayerIndex > 0 ? entry.tiledNormalScale : 0
      tiledNormalData[o + 3] = 0
    }

    if (noiseColorData) {
      noiseColorData[o] = entry.color2?.[0] ?? 0
      noiseColorData[o + 1] = entry.color2?.[1] ?? 0
      noiseColorData[o + 2] = entry.color2?.[2] ?? 0
      noiseColorData[o + 3] = entry.color2 ? (entry.noiseScale ?? 1.0) : 0
    }
  }

  return { colors, emissiveColors, tiledNormalData, noiseColorData }
}

// ─── Merge Static Into Skinned ───────────────────────────────────────────────

interface GeometryArrays {
  positions: Float32Array
  normals: Float32Array
  indices: Uint16Array | Uint32Array
  uvs: Float32Array | null
  colors: Float32Array | null
  emissiveColors: Float32Array | null
  materialIndices: Uint8Array | null
  joints: Uint8Array | Uint16Array | null
  weights: Float32Array | null
  tiledNormalData: Float32Array | null
  noiseColorData: Float32Array | null
  vertexCount: number
  indexCount: number
}

interface MergeSkinnedResult {
  positions: Float32Array
  normals: Float32Array
  indices: Uint16Array | Uint32Array
  uvs: Float32Array | null
  colors: Float32Array | null
  emissiveColors: Float32Array | null
  joints: Uint8Array | Uint16Array
  weights: Float32Array
  tiledNormalData: Float32Array | null
  noiseColorData: Float32Array | null
  aabb: Float32Array
}

function mergeStaticIntoSkinnedCompute(
  skinned: GeometryArrays,
  staticGeo: GeometryArrays,
  boneIndex: number,
  inverseBindMatrix: Float32Array,
  localTransform: Float32Array | null,
): MergeSkinnedResult {
  const bindPoseMatrix = mat4Create()
  if (!mat4Invert(bindPoseMatrix, inverseBindMatrix)) {
    // Return skinned as-is if inversion fails
    return {
      positions: skinned.positions,
      normals: skinned.normals,
      indices: skinned.indices,
      uvs: skinned.uvs,
      colors: skinned.colors,
      emissiveColors: skinned.emissiveColors,
      joints: skinned.joints || new Uint8Array(skinned.vertexCount * 4),
      weights: skinned.weights || new Float32Array(skinned.vertexCount * 4),
      tiledNormalData: skinned.tiledNormalData,
      noiseColorData: skinned.noiseColorData,
      aabb: aabbFromPoints(new Float32Array(6), skinned.positions, skinned.vertexCount),
    }
  }

  const finalTransform = mat4Create()
  if (localTransform) {
    mat4Multiply(finalTransform, bindPoseMatrix, localTransform)
  } else {
    finalTransform.set(bindPoseMatrix)
  }

  const skinnedVerts = skinned.vertexCount
  const staticVerts = staticGeo.vertexCount
  const totalVerts = skinnedVerts + staticVerts
  const totalIndices = skinned.indexCount + staticGeo.indexCount

  // Transform static positions into bind-pose space
  const positions = new Float32Array(totalVerts * 3)
  positions.set(skinned.positions)
  const tmpVec = new Float32Array(3)
  for (let i = 0; i < staticVerts; i++) {
    tmpVec[0] = staticGeo.positions[i * 3]!
    tmpVec[1] = staticGeo.positions[i * 3 + 1]!
    tmpVec[2] = staticGeo.positions[i * 3 + 2]!
    vec3TransformMat4(tmpVec, tmpVec, finalTransform)
    positions[(skinnedVerts + i) * 3] = tmpVec[0]!
    positions[(skinnedVerts + i) * 3 + 1] = tmpVec[1]!
    positions[(skinnedVerts + i) * 3 + 2] = tmpVec[2]!
  }

  // Transform static normals
  const normals = new Float32Array(totalVerts * 3)
  normals.set(skinned.normals)
  const m = finalTransform
  for (let i = 0; i < staticVerts; i++) {
    const nx = staticGeo.normals[i * 3]!
    const ny = staticGeo.normals[i * 3 + 1]!
    const nz = staticGeo.normals[i * 3 + 2]!
    let rx = m[0]! * nx + m[4]! * ny + m[8]! * nz
    let ry = m[1]! * nx + m[5]! * ny + m[9]! * nz
    let rz = m[2]! * nx + m[6]! * ny + m[10]! * nz
    const len = Math.sqrt(rx * rx + ry * ry + rz * rz)
    if (len > 1e-6) {
      const inv = 1 / len
      rx *= inv
      ry *= inv
      rz *= inv
    }
    normals[(skinnedVerts + i) * 3] = rx
    normals[(skinnedVerts + i) * 3 + 1] = ry
    normals[(skinnedVerts + i) * 3 + 2] = rz
  }

  // Merge indices
  const indices = totalVerts > 65535 ? new Uint32Array(totalIndices) : new Uint16Array(totalIndices)
  indices.set(skinned.indices)
  for (let i = 0; i < staticGeo.indexCount; i++) {
    indices[skinned.indexCount + i] = staticGeo.indices[i]! + skinnedVerts
  }

  // Merge joints
  const useUint16 = skinned.joints instanceof Uint16Array || boneIndex > 255
  const joints = useUint16 ? new Uint16Array(totalVerts * 4) : new Uint8Array(totalVerts * 4)
  if (skinned.joints) joints.set(skinned.joints)
  for (let i = 0; i < staticVerts; i++) {
    joints[(skinnedVerts + i) * 4] = boneIndex
  }

  // Merge weights
  const weights = new Float32Array(totalVerts * 4)
  if (skinned.weights) weights.set(skinned.weights)
  for (let i = 0; i < staticVerts; i++) {
    weights[(skinnedVerts + i) * 4] = 1
  }

  // Merge vertex colors
  let colors: Float32Array | null = null
  if (skinned.colors || staticGeo.colors) {
    colors = new Float32Array(totalVerts * 4)
    if (skinned.colors) {
      colors.set(skinned.colors)
    } else {
      for (let i = 0; i < skinnedVerts; i++) {
        const o = i * 4
        colors[o] = 1
        colors[o + 1] = 1
        colors[o + 2] = 1
        colors[o + 3] = 1
      }
    }
    if (staticGeo.colors) {
      colors.set(staticGeo.colors, skinnedVerts * 4)
    } else {
      for (let i = 0; i < staticVerts; i++) {
        const o = (skinnedVerts + i) * 4
        colors[o] = 1
        colors[o + 1] = 1
        colors[o + 2] = 1
        colors[o + 3] = 1
      }
    }
  }

  // Merge emissive colors
  let emissiveColors: Float32Array | null = null
  if (skinned.emissiveColors || staticGeo.emissiveColors) {
    emissiveColors = new Float32Array(totalVerts * 4)
    if (skinned.emissiveColors) {
      emissiveColors.set(skinned.emissiveColors)
    } else {
      for (let i = 0; i < skinnedVerts; i++) {
        emissiveColors[i * 4 + 3] = 1
      }
    }
    if (staticGeo.emissiveColors) {
      emissiveColors.set(staticGeo.emissiveColors, skinnedVerts * 4)
    } else {
      for (let i = 0; i < staticVerts; i++) {
        emissiveColors[(skinnedVerts + i) * 4 + 3] = 1
      }
    }
  }

  // Merge UVs
  let uvs: Float32Array | null = null
  if (skinned.uvs || staticGeo.uvs) {
    uvs = new Float32Array(totalVerts * 2)
    if (skinned.uvs) uvs.set(skinned.uvs)
    if (staticGeo.uvs) uvs.set(staticGeo.uvs, skinnedVerts * 2)
  }

  // Merge tiled normal data
  let tiledNormalData: Float32Array | null = null
  if (skinned.tiledNormalData || staticGeo.tiledNormalData) {
    tiledNormalData = new Float32Array(totalVerts * 4)
    if (skinned.tiledNormalData) tiledNormalData.set(skinned.tiledNormalData)
    if (staticGeo.tiledNormalData) tiledNormalData.set(staticGeo.tiledNormalData, skinnedVerts * 4)
  }

  // Merge noise color data
  let noiseColorData: Float32Array | null = null
  if (skinned.noiseColorData || staticGeo.noiseColorData) {
    noiseColorData = new Float32Array(totalVerts * 4)
    if (skinned.noiseColorData) noiseColorData.set(skinned.noiseColorData)
    if (staticGeo.noiseColorData) noiseColorData.set(staticGeo.noiseColorData, skinnedVerts * 4)
  }

  const aabb = aabbFromPoints(new Float32Array(6), positions, totalVerts)

  return {
    positions,
    normals,
    indices,
    uvs,
    colors,
    emissiveColors,
    joints,
    weights,
    tiledNormalData,
    noiseColorData,
    aabb,
  }
}

// ─── Smooth Normals ──────────────────────────────────────────────────────────

function computeSmoothNormalsCompute(
  positions: Float32Array,
  normals: Float32Array,
  vertexCount: number,
): Float32Array {
  const smooth = new Float32Array(vertexCount * 3)
  const accum = new Map<string, { x: number; y: number; z: number }>()
  const PRECISION = 1e4

  for (let i = 0; i < vertexCount; i++) {
    const px = Math.round(positions[i * 3]! * PRECISION)
    const py = Math.round(positions[i * 3 + 1]! * PRECISION)
    const pz = Math.round(positions[i * 3 + 2]! * PRECISION)
    const key = `${px},${py},${pz}`

    const nx = normals[i * 3]!
    const ny = normals[i * 3 + 1]!
    const nz = normals[i * 3 + 2]!

    const entry = accum.get(key)
    if (entry) {
      entry.x += nx
      entry.y += ny
      entry.z += nz
    } else {
      accum.set(key, { x: nx, y: ny, z: nz })
    }
  }

  for (let i = 0; i < vertexCount; i++) {
    const px = Math.round(positions[i * 3]! * PRECISION)
    const py = Math.round(positions[i * 3 + 1]! * PRECISION)
    const pz = Math.round(positions[i * 3 + 2]! * PRECISION)
    const key = `${px},${py},${pz}`

    const entry = accum.get(key)!
    let len = Math.sqrt(entry.x * entry.x + entry.y * entry.y + entry.z * entry.z)
    if (len < 1e-8) len = 1
    smooth[i * 3] = entry.x / len
    smooth[i * 3 + 1] = entry.y / len
    smooth[i * 3 + 2] = entry.z / len
  }

  return smooth
}

// ─── Merge Geometries ────────────────────────────────────────────────────────

interface MergeGeometriesResult {
  positions: Float32Array
  normals: Float32Array
  indices: Uint16Array | Uint32Array
  uvs: Float32Array | null
  colors: Float32Array | null
  emissiveColors: Float32Array | null
  materialIndices: Uint8Array | null
  tiledNormalData: Float32Array | null
  noiseColorData: Float32Array | null
  aabb: Float32Array
}

function mergeGeometriesCompute(geometries: GeometryArrays[]): MergeGeometriesResult {
  let totalVertices = 0
  let totalIndices = 0
  let hasUVs = false
  let hasColors = false
  let hasMaterialIndices = false
  let hasTiledNormalData = false
  let hasNoiseColorData = false
  for (const geo of geometries) {
    totalVertices += geo.vertexCount
    totalIndices += geo.indexCount
    if (geo.uvs) hasUVs = true
    if (geo.colors) hasColors = true
    if (geo.materialIndices) hasMaterialIndices = true
    if (geo.tiledNormalData) hasTiledNormalData = true
    if (geo.noiseColorData) hasNoiseColorData = true
  }

  const positions = new Float32Array(totalVertices * 3)
  const normals = new Float32Array(totalVertices * 3)
  const indices = totalVertices > 65535 ? new Uint32Array(totalIndices) : new Uint16Array(totalIndices)
  const uvs = hasUVs ? new Float32Array(totalVertices * 2) : null
  const colors = hasColors ? new Float32Array(totalVertices * 4) : null
  const emissiveColors = hasColors ? new Float32Array(totalVertices * 4) : null
  const materialIndices = hasMaterialIndices ? new Uint8Array(totalVertices) : null
  const tiledNormalData = hasTiledNormalData ? new Float32Array(totalVertices * 4) : null
  const noiseColorData = hasNoiseColorData ? new Float32Array(totalVertices * 4) : null

  let vOff = 0
  let iOff = 0
  for (const geo of geometries) {
    positions.set(geo.positions, vOff * 3)
    normals.set(geo.normals, vOff * 3)
    if (uvs && geo.uvs) uvs.set(geo.uvs, vOff * 2)
    if (colors) {
      if (geo.colors) {
        colors.set(geo.colors, vOff * 4)
      } else {
        for (let j = 0; j < geo.vertexCount; j++) {
          const o = (vOff + j) * 4
          colors[o] = 1
          colors[o + 1] = 1
          colors[o + 2] = 1
          colors[o + 3] = 1
        }
      }
    }
    if (emissiveColors) {
      if (geo.emissiveColors) {
        emissiveColors.set(geo.emissiveColors, vOff * 4)
      } else {
        for (let j = 0; j < geo.vertexCount; j++) {
          emissiveColors[(vOff + j) * 4 + 3] = 1
        }
      }
    }
    if (tiledNormalData && geo.tiledNormalData) {
      tiledNormalData.set(geo.tiledNormalData, vOff * 4)
    }
    if (noiseColorData && geo.noiseColorData) {
      noiseColorData.set(geo.noiseColorData, vOff * 4)
    }
    if (materialIndices && geo.materialIndices) {
      materialIndices.set(geo.materialIndices, vOff)
    }
    for (let j = 0; j < geo.indexCount; j++) {
      indices[iOff + j] = geo.indices[j]! + vOff
    }
    vOff += geo.vertexCount
    iOff += geo.indexCount
  }

  const aabb = aabbFromPoints(new Float32Array(6), positions, totalVertices)
  return {
    positions,
    normals,
    indices,
    uvs,
    colors,
    emissiveColors,
    materialIndices,
    tiledNormalData,
    noiseColorData,
    aabb,
  }
}

// ─── Message Handler ─────────────────────────────────────────────────────────

function collectTransferables(obj: Record<string, unknown>): ArrayBuffer[] {
  const buffers: ArrayBuffer[] = []
  for (const val of Object.values(obj)) {
    if (val instanceof ArrayBuffer) {
      buffers.push(val)
    } else if (ArrayBuffer.isView(val)) {
      buffers.push(val.buffer as ArrayBuffer)
    }
  }
  return buffers
}

self.onmessage = (e: MessageEvent) => {
  const msg = e.data
  const { id, type } = msg

  switch (type) {
    case 'buildBVH': {
      const result = buildBVH(msg.positions, msg.indices)
      ;(self as unknown as Worker).postMessage({ id, type, ...result }, [
        result.bvhBuffer,
        result.triOrder.buffer,
      ] as any)
      break
    }

    case 'bakePalette': {
      const result = bakePaletteCompute(
        msg.materialIndices,
        msg.vertexCount,
        msg.palette,
        msg.hasTiledNormals,
        msg.hasNoiseColor,
      )
      const transfers = collectTransferables(result as unknown as Record<string, unknown>)
      ;(self as unknown as Worker).postMessage({ id, type, ...result }, transfers as any)
      break
    }

    case 'mergeStaticIntoSkinned': {
      const result = mergeStaticIntoSkinnedCompute(
        msg.skinned,
        msg.staticGeo,
        msg.boneIndex,
        msg.inverseBindMatrix,
        msg.localTransform,
      )
      const transfers = collectTransferables(result as unknown as Record<string, unknown>)
      ;(self as unknown as Worker).postMessage({ id, type, ...result }, transfers as any)
      break
    }

    case 'computeSmoothNormals': {
      const result = computeSmoothNormalsCompute(msg.positions, msg.normals, msg.vertexCount)
      ;(self as unknown as Worker).postMessage({ id, type, smoothNormals: result }, [result.buffer] as any)
      break
    }

    case 'mergeGeometries': {
      const result = mergeGeometriesCompute(msg.geometries)
      const transfers = collectTransferables(result as unknown as Record<string, unknown>)
      ;(self as unknown as Worker).postMessage({ id, type, ...result }, transfers as any)
      break
    }
  }
}
