// Clone – Deep-copies a scene graph subtree with optional skeleton remapping.
//
// When instancing skinned characters, each instance needs its own bone hierarchy and
// skeleton so they can animate independently. cloneScene() walks the source node tree,
// creates a structural copy (Meshes share geometry/material references, non-mesh nodes
// become Groups), and remaps any provided skeletons so cloned meshes point to cloned bones.
//
// cloneScene(source, skeletons?, options?) – Returns the cloned root, remapped skeletons,
//   and a node map for attaching objects to specific cloned nodes (e.g. a weapon to a hand bone).
//   The meshFilter option controls which source Meshes are cloned as renderable Meshes vs
//   inert Groups (useful when a model contains helper/LOD meshes that shouldn't all render).

import { Skeleton } from '../animation/skeleton'
import { Group } from './group'
import { Mesh } from './mesh'
import { Node } from './node'

export interface CloneOptions {
  meshFilter?: (mesh: Mesh) => boolean
}

export interface CloneResult {
  root: Node
  skeletons: Skeleton[]
  nodeMap: Map<Node, Node>
}

export const cloneScene = (source: Node, skeletons: Skeleton[] = [], options: CloneOptions = {}): CloneResult => {
  const { meshFilter } = options
  const nodeMap = new Map<Node, Node>()

  const cloneNode = (src: Node): Node => {
    let clone: Node
    if (src instanceof Mesh && (!meshFilter || meshFilter(src))) {
      const meshClone = new Mesh(src.geometry, src.material)
      meshClone.outline = src.outline
      meshClone.maxDistance = src.maxDistance
      meshClone.tintColor = [src.tintColor[0], src.tintColor[1], src.tintColor[2]]
      meshClone.tintIntensity = src.tintIntensity
      clone = meshClone
    } else {
      clone = new Group()
    }
    clone.name = src.name
    clone.visible = src.visible
    clone.frustumCulled = src.frustumCulled
    clone.castShadow = src.castShadow
    clone.receiveShadow = src.receiveShadow
    clone.setPosition(src.position[0]!, src.position[1]!, src.position[2]!)
    clone.setRotation(src.rotation[0]!, src.rotation[1]!, src.rotation[2]!, src.rotation[3]!)
    clone.setScale(src.scale[0]!, src.scale[1]!, src.scale[2]!)
    nodeMap.set(src, clone)
    for (const child of src.children) {
      clone.add(cloneNode(child))
    }
    return clone
  }

  const root = cloneNode(source)

  const clonedSkeletons: Skeleton[] = []
  for (const srcSkeleton of skeletons) {
    const clonedBones = srcSkeleton.bones.map(b => nodeMap.get(b)!)
    const clonedSkeleton = new Skeleton(clonedBones, srcSkeleton.boneInverseBindMatrices)
    clonedSkeletons.push(clonedSkeleton)
  }

  // Assign cloned skeletons to cloned meshes
  if (clonedSkeletons.length > 0) {
    nodeMap.forEach((clone, src) => {
      if (src instanceof Mesh && src.skeleton) {
        const skeletonIndex = skeletons.indexOf(src.skeleton)
        if (skeletonIndex !== -1 && clone instanceof Mesh) {
          clone.skeleton = clonedSkeletons[skeletonIndex]
        }
      }
    })
  }

  return { root, skeletons: clonedSkeletons, nodeMap }
}
