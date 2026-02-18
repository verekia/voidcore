import { Node } from './node.ts'

export class Group extends Node {}

export const createGroup = (): Group => new Group()
