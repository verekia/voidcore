// Float Precision – Configurable floating-point array precision for the engine.
//
// VoidCore's math types (Vec3, Mat4, Quat, AABB) are backed by typed arrays. By default
// these are Float32Array (32-bit, ~7 decimal digits of precision). This module allows
// switching to Float16Array (16-bit, ~3.3 decimal digits, ±65504 range) for reduced
// memory usage and potentially better cache performance.
//
// Float16Array is a newer API (TC39 stage 3+). The 'auto' mode detects support at runtime
// and falls back to Float32Array if unavailable.
//
// setFloatPrecision('auto' | 'float16' | 'float32') – Configures which typed array
//   constructor is used by the math module's create functions.
// getFloatPrecision() – Returns the currently active precision ('float16' | 'float32').
// createFloatArray(n) – Creates a new float array of length n with the active precision.
// createFloatArrayFrom(values) – Creates a new float array from existing values.
// isFloat16Supported() – Returns true if Float16Array is available in this runtime.

export type FloatPrecision = 'float16' | 'float32'

export type FloatArray = Float32Array | Float16Array

type FloatArrayCtor = {
  new (length: number): FloatArray
  new (array: ArrayLike<number>): FloatArray
}

let _precision: FloatPrecision = 'float32'
let _Ctor: FloatArrayCtor = Float32Array

export const isFloat16Supported = (): boolean => typeof Float16Array !== 'undefined'

export const setFloatPrecision = (precision: FloatPrecision | 'auto'): FloatPrecision => {
  if (precision === 'auto') {
    _precision = isFloat16Supported() ? 'float16' : 'float32'
  } else {
    _precision = precision
  }
  _Ctor = _precision === 'float16' ? Float16Array : Float32Array
  return _precision
}

export const getFloatPrecision = (): FloatPrecision => _precision

export const createFloatArray = (length: number): FloatArray => new _Ctor(length)

export const createFloatArrayFrom = (values: ArrayLike<number>): FloatArray => new _Ctor(values)
