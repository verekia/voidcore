// Group – An empty container node used to organize the scene hierarchy.
//
// Groups don't render anything themselves. They exist to group child nodes together so you
// can move, rotate, or scale them as a unit. For example, a "car" group might contain
// separate mesh children for the body, wheels, and windows.
//
// new Group() – Creates an empty group node.

import { Node } from './node'

export class Group extends Node {}
