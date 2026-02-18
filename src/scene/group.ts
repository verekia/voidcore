// Group – An empty container node used to organize the scene hierarchy.
//
// Groups don't render anything themselves. They exist to group child nodes together so you
// can move, rotate, or scale them as a unit. For example, a "car" group might contain
// separate mesh children for the body, wheels, and windows.
//
// createGroup() – Factory that creates an empty group node.

import { Node } from './node.ts'

export class Group extends Node {}

export const createGroup = (): Group => new Group()
