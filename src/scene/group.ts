import { Node } from './node.ts'

export class Group extends Node {
  constructor() {
    super()
    this.type = 'group'
  }
}

export const createGroup = (): Group => new Group()
