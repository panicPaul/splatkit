#ifndef SLANG_CUDA_PRELUDE_H
#define SLANG_CUDA_PRELUDE_H

#define SLANG_PRELUDE_EXPORT

#ifdef __CUDACC_RTC__
#define SLANG_CUDA_RTC 1
#else
#define SLANG_CUDA_RTC 0
#endif

#if SLANG_CUDA_RTC

#else

#include <cstdint>
#include <stdio.h>

#endif

// Define SLANG_CUDA_ENABLE_HALF to use the cuda_fp16 include to add half support.
// For this to work NVRTC needs to have the path to the CUDA SDK.
//
// As it stands the includes paths defined for Slang are passed down to NVRTC. Similarly defines
// defined for the Slang compile are passed down.

#ifdef SLANG_CUDA_ENABLE_HALF
// We don't want half2 operators, because it will implement comparison operators that return a
// bool(!). We want to generate those functions. Doing so means that we will have to define all
// the other half2 operators.
#define __CUDA_NO_HALF2_OPERATORS__
#include <cuda_fp16.h>
#endif

#ifdef SLANG_CUDA_ENABLE_FP8
#include <cuda_fp8.h>
#endif

#ifdef SLANG_CUDA_ENABLE_BF16
#include <cuda_bf16.h>
#endif

#ifdef SLANG_CUDA_ENABLE_OPTIX
#include <optix.h>
#endif

// Define slang offsetof implementation
#ifndef SLANG_OFFSET_OF
#define SLANG_OFFSET_OF(type, member) (size_t)((char*)&(((type*)0)->member) - (char*)0)
#endif

// Must be large enough to cause overflow and therefore infinity
#ifndef SLANG_INFINITY
#define SLANG_INFINITY ((float)(1e+300 * 1e+300))
#endif

// For now we'll disable any asserts in this prelude
#define SLANG_PRELUDE_ASSERT(x)

#ifndef SLANG_CUDA_WARP_SIZE
#define SLANG_CUDA_WARP_SIZE 32
#endif

#define SLANG_CUDA_WARP_MASK \
    (SLANG_CUDA_WARP_SIZE - 1) // Used for masking threadIdx.x to the warp lane index
#define SLANG_CUDA_WARP_BITMASK (~int(0))

//
#define SLANG_FORCE_INLINE inline

#define SLANG_CUDA_CALL __device__

#define SLANG_FORCE_INLINE inline
#define SLANG_INLINE inline


// Since we are using unsigned arithmatic care is need in this comparison.
// It is *assumed* that sizeInBytes >= elemSize. Which means (sizeInBytes >= elemSize) >= 0
// Which means only a single test is needed

// Asserts for bounds checking.
// It is assumed index/count are unsigned types.
#define SLANG_BOUND_ASSERT(index, count) SLANG_PRELUDE_ASSERT(index < count);
#define SLANG_BOUND_ASSERT_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_PRELUDE_ASSERT(index <= (sizeInBytes - elemSize) && (index & 3) == 0);

// Macros to zero index if an access is out of range
#define SLANG_BOUND_ZERO_INDEX(index, count) index = (index < count) ? index : 0;
#define SLANG_BOUND_ZERO_INDEX_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    index = (index <= (sizeInBytes - elemSize)) ? index : 0;

// The 'FIX' macro define how the index is fixed. The default is to do nothing. If
// SLANG_ENABLE_BOUND_ZERO_INDEX the fix macro will zero the index, if out of range
#ifdef SLANG_ENABLE_BOUND_ZERO_INDEX
#define SLANG_BOUND_FIX(index, count) SLANG_BOUND_ZERO_INDEX(index, count)
#define SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_BOUND_ZERO_INDEX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#define SLANG_BOUND_FIX_FIXED_ARRAY(index, count) \
    SLANG_BOUND_ZERO_INDEX(index, count) SLANG_BOUND_ZERO_INDEX(index, count)
#else
#define SLANG_BOUND_FIX(index, count)
#define SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#define SLANG_BOUND_FIX_FIXED_ARRAY(index, count)
#endif

#ifndef SLANG_BOUND_CHECK
#define SLANG_BOUND_CHECK(index, count) \
    SLANG_BOUND_ASSERT(index, count) SLANG_BOUND_FIX(index, count)
#endif

#ifndef SLANG_BOUND_CHECK_BYTE_ADDRESS
#define SLANG_BOUND_CHECK_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_BOUND_ASSERT_BYTE_ADDRESS(index, elemSize, sizeInBytes)    \
    SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#endif

#ifndef SLANG_BOUND_CHECK_FIXED_ARRAY
#define SLANG_BOUND_CHECK_FIXED_ARRAY(index, count) \
    SLANG_BOUND_ASSERT(index, count) SLANG_BOUND_FIX_FIXED_ARRAY(index, count)
#endif

// This macro handles how out-of-range surface coordinates are handled;
// I can equal
// cudaBoundaryModeClamp, in which case out-of-range coordinates are clamped to the valid range
// cudaBoundaryModeZero, in which case out-of-range reads return zero and out-of-range writes are
// ignored cudaBoundaryModeTrap, in which case out-of-range accesses cause the kernel execution to
// fail.

#ifndef SLANG_CUDA_BOUNDARY_MODE
#define SLANG_CUDA_BOUNDARY_MODE cudaBoundaryModeZero

// Can be one of SLANG_CUDA_PTX_BOUNDARY_MODE. Only applies *PTX* emitted CUDA operations
// which currently is just RWTextureRW format writes
//
// .trap         causes an execution trap on out-of-bounds addresses
// .clamp        stores data at the nearest surface location (sized appropriately)
// .zero         drops stores to out-of-bounds addresses

#define SLANG_PTX_BOUNDARY_MODE "zero"
#endif

struct TypeInfo
{
    size_t typeSize;
};

template<typename T, size_t SIZE>
struct FixedArray
{
    SLANG_CUDA_CALL const T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK_FIXED_ARRAY(index, SIZE);
        return m_data[index];
    }
    SLANG_CUDA_CALL T& operator[](size_t index)
    {
        SLANG_BOUND_CHECK_FIXED_ARRAY(index, SIZE);
        return m_data[index];
    }

    T m_data[SIZE];
};

// An array that has no specified size, becomes a 'Array'. This stores the size so it can
// potentially do bounds checking.
template<typename T>
struct Array
{
    SLANG_CUDA_CALL const T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    SLANG_CUDA_CALL T& operator[](size_t index)
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }

    T* data;
    size_t count;
};

// Typically defined in cuda.h, but we can't ship/rely on that, so just define here
typedef unsigned long long CUtexObject;
typedef unsigned long long CUsurfObject;

// On CUDA sampler state is actually bound up with the texture object. We have a SamplerState type,
// backed as a pointer, to simplify code generation, with the downside that such a binding will take
// up uniform space, even though it will have no effect.
// TODO(JS): Consider ways to strip use of variables of this type so have no binding,
struct SamplerStateUnused;
typedef SamplerStateUnused* SamplerState;


// TODO(JS): Not clear yet if this can be handled on CUDA, by just ignoring.
// For now, just map to the index type.
typedef size_t NonUniformResourceIndex;

// Code generator will generate the specific type
template<typename T, int ROWS, int COLS>
struct Matrix;

// Boolean vector types should follow CUDA's builtin vector alignment rules
// Align boolX the same as charX according to CUDA spec:
// char1/uchar1: 1-byte aligned, char2/uchar2: 2-byte aligned
// char3/uchar3: 1-byte aligned, char4/uchar4: 4-byte aligned
struct __align__(1) bool1
{
    bool x;

    SLANG_FORCE_INLINE SLANG_CUDA_CALL bool& operator[](int idx)
    {
        return (&x)[idx];
    }
    SLANG_FORCE_INLINE SLANG_CUDA_CALL const bool& operator[](int idx) const
    {
        return (&x)[idx];
    }
};

struct __align__(2) bool2
{
    bool x, y;

    SLANG_FORCE_INLINE SLANG_CUDA_CALL bool& operator[](int idx)
    {
        return (&x)[idx];
    }
    SLANG_FORCE_INLINE SLANG_CUDA_CALL const bool& operator[](int idx) const
    {
        return (&x)[idx];
    }
};

struct __align__(1) bool3
{
    bool x, y, z;

    SLANG_FORCE_INLINE SLANG_CUDA_CALL bool& operator[](int idx)
    {
        return (&x)[idx];
    }
    SLANG_FORCE_INLINE SLANG_CUDA_CALL const bool& operator[](int idx) const
    {
        return (&x)[idx];
    }
};

struct __align__(4) bool4
{
    bool x, y, z, w;

    SLANG_FORCE_INLINE SLANG_CUDA_CALL bool& operator[](int idx)
    {
        return (&x)[idx];
    }
    SLANG_FORCE_INLINE SLANG_CUDA_CALL const bool& operator[](int idx) const
    {
        return (&x)[idx];
    }
};

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool __ldg(const bool* ptr)
{
    return (bool)(__ldg((const char*)ptr));
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool2 __ldg(const bool2* ptr)
{
    auto val = __ldg((const char2*)ptr);
    return {val.x != 0, val.y != 0};
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool4 __ldg(const bool4* ptr)
{
    auto val = __ldg((const char4*)ptr);
    return {val.x != 0, val.y != 0, val.z != 0, val.w != 0};
}

#if SLANG_CUDA_RTC

typedef signed char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long long int64_t;
typedef ptrdiff_t intptr_t;

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef size_t uintptr_t;

typedef long long longlong;
typedef unsigned long long ulonglong;

#else

// When not using NVRTC, match the platform's int64_t definition for signed type
// On Linux: int64_t is 'long', on Windows: int64_t is 'long long'
typedef int64_t longlong;
// ulonglong must remain 'unsigned long long' to match CUDA's atomic operations
typedef unsigned long long ulonglong;

#endif

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;

#if SLANG_CUDA_ENABLE_HALF
typedef __half half;
#endif

union Union32
{
    uint32_t u;
    int32_t i;
    float f;
};

union Union64
{
    uint64_t u;
    int64_t i;
    double d;
};

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL float make_float(T val)
{
    return (float)val;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float _slang_fmod(float x, float y)
{
    return ::fmodf(x, y);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double _slang_fmod(double x, double y)
{
    return ::fmod(x, y);
}

#if SLANG_CUDA_ENABLE_HALF

// Add the other vector half types
struct __half1
{
    __half x;
};
struct __align__(4) __half3
{
    __half x, y, z;
};
struct __align__(4) __half4
{
    __half x, y, z, w;
};
#endif

#if SLANG_CUDA_ENABLE_BF16

// Add the other vector bfloat16 types
struct __nv_bfloat161
{
    __nv_bfloat16 x;
};
struct __nv_bfloat163
{
    __nv_bfloat16 x, y, z;
};
struct __nv_bfloat164
{
    __nv_bfloat16 x, y, z, w;
};
#endif

#if SLANG_CUDA_ENABLE_FP8

// Add the other vector fp8 types
struct __nv_fp8_e4m31
{
    __nv_fp8_e4m3 x;
};
struct __nv_fp8_e4m32
{
    __nv_fp8_e4m3 x, y;
};
struct __nv_fp8_e4m33
{
    __nv_fp8_e4m3 x, y, z;
};
struct __nv_fp8_e4m34
{
    __nv_fp8_e4m3 x, y, z, w;
};
struct __nv_fp8_e5m21
{
    __nv_fp8_e5m2 x;
};
struct __nv_fp8_e5m22
{
    __nv_fp8_e5m2 x, y;
};
struct __nv_fp8_e5m23
{
    __nv_fp8_e5m2 x, y, z;
};
struct __nv_fp8_e5m24
{
    __nv_fp8_e5m2 x, y, z, w;
};
#endif

#define SLANG_VECTOR_GET_ELEMENT(T)                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##1 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }                                                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##2 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }                                                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##3 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }                                                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##4 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }
SLANG_VECTOR_GET_ELEMENT(int)
SLANG_VECTOR_GET_ELEMENT(bool)
SLANG_VECTOR_GET_ELEMENT(uint)
SLANG_VECTOR_GET_ELEMENT(short)
SLANG_VECTOR_GET_ELEMENT(ushort)
SLANG_VECTOR_GET_ELEMENT(char)
SLANG_VECTOR_GET_ELEMENT(uchar)
SLANG_VECTOR_GET_ELEMENT(longlong)
SLANG_VECTOR_GET_ELEMENT(ulonglong)
SLANG_VECTOR_GET_ELEMENT(float)
SLANG_VECTOR_GET_ELEMENT(double)

#define SLANG_VECTOR_GET_ELEMENT_PTR(T)                                                            \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(const T##1 * x, int index) \
    {                                                                                              \
        return ((T*)(x)) + index;                                                                  \
    }                                                                                              \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(const T##2 * x, int index) \
    {                                                                                              \
        return ((T*)(x)) + index;                                                                  \
    }                                                                                              \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(const T##3 * x, int index) \
    {                                                                                              \
        return ((T*)(x)) + index;                                                                  \
    }                                                                                              \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(const T##4 * x, int index) \
    {                                                                                              \
        return ((T*)(x)) + index;                                                                  \
    }
SLANG_VECTOR_GET_ELEMENT_PTR(int)
SLANG_VECTOR_GET_ELEMENT_PTR(bool)
SLANG_VECTOR_GET_ELEMENT_PTR(uint)
SLANG_VECTOR_GET_ELEMENT_PTR(short)
SLANG_VECTOR_GET_ELEMENT_PTR(ushort)
SLANG_VECTOR_GET_ELEMENT_PTR(char)
SLANG_VECTOR_GET_ELEMENT_PTR(uchar)
SLANG_VECTOR_GET_ELEMENT_PTR(longlong)
SLANG_VECTOR_GET_ELEMENT_PTR(ulonglong)
SLANG_VECTOR_GET_ELEMENT_PTR(float)
SLANG_VECTOR_GET_ELEMENT_PTR(double)

#if SLANG_CUDA_ENABLE_HALF
SLANG_VECTOR_GET_ELEMENT(__half)
SLANG_VECTOR_GET_ELEMENT_PTR(__half)
#endif

#if SLANG_CUDA_ENABLE_BF16
SLANG_VECTOR_GET_ELEMENT(__nv_bfloat16)
SLANG_VECTOR_GET_ELEMENT_PTR(__nv_bfloat16)

SLANG_FORCE_INLINE SLANG_CUDA_CALL __nv_bfloat16
_slang_vector_dot(__nv_bfloat162 v0, __nv_bfloat162 v1)
{
    __nv_bfloat16 result = __nv_bfloat16(0.0f);
    for (int i = 0; i < 2; i++)
    {
        result += _slang_vector_get_element(v0, i) * _slang_vector_get_element(v1, i);
    }
    return result;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __nv_bfloat16
_slang_vector_dot(__nv_bfloat163 v0, __nv_bfloat163 v1)
{
    __nv_bfloat16 result = __nv_bfloat16(0.0f);
    for (int i = 0; i < 3; i++)
    {
        result += _slang_vector_get_element(v0, i) * _slang_vector_get_element(v1, i);
    }
    return result;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __nv_bfloat16
_slang_vector_dot(__nv_bfloat164 v0, __nv_bfloat164 v1)
{
    __nv_bfloat16 result = __nv_bfloat16(0.0f);
    for (int i = 0; i < 4; i++)
    {
        result += _slang_vector_get_element(v0, i) * _slang_vector_get_element(v1, i);
    }
    return result;
}
#endif

#if SLANG_CUDA_ENABLE_FP8
SLANG_VECTOR_GET_ELEMENT(__nv_fp8_e4m3)
SLANG_VECTOR_GET_ELEMENT_PTR(__nv_fp8_e4m3)
SLANG_VECTOR_GET_ELEMENT(__nv_fp8_e5m2)
SLANG_VECTOR_GET_ELEMENT_PTR(__nv_fp8_e5m2)
#endif

#define SLANG_CUDA_VECTOR_BINARY_OP(T, n, op)                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n operator op(T##n thisVal, T##n other)             \
    {                                                                                         \
        T##n result;                                                                          \
        for (int i = 0; i < n; i++)                                                           \
            *_slang_vector_get_element_ptr(&result, i) =                                      \
                _slang_vector_get_element(thisVal, i) op _slang_vector_get_element(other, i); \
        return result;                                                                        \
    }
#define SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, op)                                           \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL bool##n operator op(T##n thisVal, T##n other)            \
    {                                                                                           \
        bool##n result;                                                                         \
        for (int i = 0; i < n; i++)                                                             \
            *_slang_vector_get_element_ptr(&result, i) =                                        \
                (_slang_vector_get_element(thisVal, i) op _slang_vector_get_element(other, i)); \
        return result;                                                                          \
    }
#define SLANG_CUDA_VECTOR_UNARY_OP(T, n, op)                                                       \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n operator op(T##n thisVal)                              \
    {                                                                                              \
        T##n result;                                                                               \
        for (int i = 0; i < n; i++)                                                                \
            *_slang_vector_get_element_ptr(&result, i) = op _slang_vector_get_element(thisVal, i); \
        return result;                                                                             \
    }

#define SLANG_CUDA_VECTOR_INT_OP(T, n)            \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, +)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, -)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, *)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, /)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, %)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, ^)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, &)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, |)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, &&)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, ||)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, >>)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, <<)         \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, ==) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, !=) \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, !)           \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, -)           \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, ~)

#define SLANG_CUDA_VECTOR_INT_OPS(T) \
    SLANG_CUDA_VECTOR_INT_OP(T, 2)   \
    SLANG_CUDA_VECTOR_INT_OP(T, 3)   \
    SLANG_CUDA_VECTOR_INT_OP(T, 4)

SLANG_CUDA_VECTOR_INT_OPS(int)
SLANG_CUDA_VECTOR_INT_OPS(bool)
SLANG_CUDA_VECTOR_INT_OPS(uint)
SLANG_CUDA_VECTOR_INT_OPS(ushort)
SLANG_CUDA_VECTOR_INT_OPS(short)
SLANG_CUDA_VECTOR_INT_OPS(char)
SLANG_CUDA_VECTOR_INT_OPS(uchar)
SLANG_CUDA_VECTOR_INT_OPS(longlong)
SLANG_CUDA_VECTOR_INT_OPS(ulonglong)

#define SLANG_CUDA_VECTOR_FLOAT_OP(T, n)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, +)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, -)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, *)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, /)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, &&)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, ||)         \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, ==) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, !=) \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, -)
#define SLANG_CUDA_VECTOR_FLOAT_OPS(T) \
    SLANG_CUDA_VECTOR_FLOAT_OP(T, 2)   \
    SLANG_CUDA_VECTOR_FLOAT_OP(T, 3)   \
    SLANG_CUDA_VECTOR_FLOAT_OP(T, 4)

SLANG_CUDA_VECTOR_FLOAT_OPS(float)
SLANG_CUDA_VECTOR_FLOAT_OPS(double)
#if SLANG_CUDA_ENABLE_HALF
SLANG_CUDA_VECTOR_FLOAT_OPS(__half)
#endif
#define SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, n)                                             \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n operator%(const T##n& left, const T##n& right) \
    {                                                                                      \
        T##n result;                                                                       \
        for (int i = 0; i < n; i++)                                                        \
            *_slang_vector_get_element_ptr(&result, i) = _slang_fmod(                      \
                _slang_vector_get_element(left, i),                                        \
                _slang_vector_get_element(right, i));                                      \
        return result;                                                                     \
    }
#define SLANG_CUDA_FLOAT_VECTOR_MOD(T)     \
    SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, 2) \
    SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, 3) \
    SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, 4)

SLANG_CUDA_FLOAT_VECTOR_MOD(float)
SLANG_CUDA_FLOAT_VECTOR_MOD(double)

#if SLANG_CUDA_RTC || SLANG_CUDA_ENABLE_HALF
#define SLANG_MAKE_VECTOR(T)                                                \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##2 make_##T##2(T x, T y)           \
    {                                                                       \
        return T##2 {x, y};                                                 \
    }                                                                       \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##3 make_##T##3(T x, T y, T z)      \
    {                                                                       \
        return T##3 {x, y, z};                                              \
    }                                                                       \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##4 make_##T##4(T x, T y, T z, T w) \
    {                                                                       \
        return T##4 {x, y, z, w};                                           \
    }
#endif

#if SLANG_CUDA_RTC
SLANG_MAKE_VECTOR(int)
SLANG_MAKE_VECTOR(uint)
SLANG_MAKE_VECTOR(short)
SLANG_MAKE_VECTOR(ushort)
SLANG_MAKE_VECTOR(char)
SLANG_MAKE_VECTOR(uchar)
SLANG_MAKE_VECTOR(float)
SLANG_MAKE_VECTOR(double)
SLANG_MAKE_VECTOR(longlong)
SLANG_MAKE_VECTOR(ulonglong)
#endif

#if SLANG_CUDA_ENABLE_HALF
SLANG_MAKE_VECTOR(__half)
#endif

#if SLANG_CUDA_ENABLE_BF16
SLANG_MAKE_VECTOR(__nv_bfloat16)
#endif

#if SLANG_CUDA_ENABLE_FP8
SLANG_MAKE_VECTOR(__nv_fp8_e4m3)
SLANG_MAKE_VECTOR(__nv_fp8_e5m2)
#endif

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool1 make_bool1(bool x)
{
    return bool1{x};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool2 make_bool2(bool x, bool y)
{
    return bool2{x, y};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool3 make_bool3(bool x, bool y, bool z)
{
    return bool3{x, y, z};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool4 make_bool4(bool x, bool y, bool z, bool w)
{
    return bool4{x, y, z, w};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool2 make_bool2(bool x)
{
    return bool2{x, x};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool3 make_bool3(bool x)
{
    return bool3{x, x, x};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool4 make_bool4(bool x)
{
    return bool4{x, x, x, x};
}

#if SLANG_CUDA_RTC
#define SLANG_MAKE_VECTOR_FROM_SCALAR(T)                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##1 make_##T##1(T x) \
    {                                                        \
        return T##1 {x};                                     \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##2 make_##T##2(T x) \
    {                                                        \
        return make_##T##2(x, x);                            \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##3 make_##T##3(T x) \
    {                                                        \
        return make_##T##3(x, x, x);                         \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##4 make_##T##4(T x) \
    {                                                        \
        return make_##T##4(x, x, x, x);                      \
    }
#else
#define SLANG_MAKE_VECTOR_FROM_SCALAR(T)                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##2 make_##T##2(T x) \
    {                                                        \
        return make_##T##2(x, x);                            \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##3 make_##T##3(T x) \
    {                                                        \
        return make_##T##3(x, x, x);                         \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##4 make_##T##4(T x) \
    {                                                        \
        return make_##T##4(x, x, x, x);                      \
    }
#endif
SLANG_MAKE_VECTOR_FROM_SCALAR(int)
SLANG_MAKE_VECTOR_FROM_SCALAR(uint)
SLANG_MAKE_VECTOR_FROM_SCALAR(short)
SLANG_MAKE_VECTOR_FROM_SCALAR(ushort)
SLANG_MAKE_VECTOR_FROM_SCALAR(char)
SLANG_MAKE_VECTOR_FROM_SCALAR(uchar)
SLANG_MAKE_VECTOR_FROM_SCALAR(longlong)
SLANG_MAKE_VECTOR_FROM_SCALAR(ulonglong)
SLANG_MAKE_VECTOR_FROM_SCALAR(float)
SLANG_MAKE_VECTOR_FROM_SCALAR(double)
#if SLANG_CUDA_ENABLE_HALF
SLANG_MAKE_VECTOR_FROM_SCALAR(__half)
#if !SLANG_CUDA_RTC
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half1 make___half1(__half x)
{
    return __half1{x};
}
#endif
#endif
#if SLANG_CUDA_ENABLE_BF16
SLANG_MAKE_VECTOR_FROM_SCALAR(__nv_bfloat16)
#if !SLANG_CUDA_RTC
SLANG_FORCE_INLINE SLANG_CUDA_CALL __nv_bfloat16 make___nv_bfloat161(__nv_bfloat16 x)
{
    return __nv_bfloat16{x};
}
#endif
#endif

#if SLANG_CUDA_ENABLE_FP8
SLANG_MAKE_VECTOR_FROM_SCALAR(__nv_fp8_e4m3)
SLANG_MAKE_VECTOR_FROM_SCALAR(__nv_fp8_e5m2)
#if !SLANG_CUDA_RTC
SLANG_FORCE_INLINE SLANG_CUDA_CALL __nv_fp8_e4m3 make___nv_fp8_e4m31(__nv_fp8_e4m3 x)
{
    return __nv_fp8_e4m3{x};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __nv_fp8_e5m2 make___nv_fp8_e5m21(__nv_fp8_e5m2 x)
{
    return __nv_fp8_e5m2{x};
}
#endif
#endif

#define SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(Fn, T, N)                                            \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##N Fn(T##N* address, T##N val)                           \
    {                                                                                             \
        T##N result;                                                                              \
        for (int i = 0; i < N; i++)                                                               \
            *_slang_vector_get_element_ptr(&result, i) =                                          \
                Fn(_slang_vector_get_element_ptr(address, i), _slang_vector_get_element(val, i)); \
        return result;                                                                            \
    }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, float, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, float, 4)
#endif
#if defined(SLANG_CUDA_ENABLE_HALF) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, __half, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, __half, 4)
#endif
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, float, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, int, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, int, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, int, 4)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, uint, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, uint, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, uint, 4)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, ulonglong, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, ulonglong, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, ulonglong, 4)

template<typename T, int n>
struct GetVectorTypeImpl
{
};

#define GET_VECTOR_TYPE_IMPL(T, n)                                     \
    template<>                                                         \
    struct GetVectorTypeImpl<T, n>                                     \
    {                                                                  \
        typedef T##n type;                                             \
        static SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n fromScalar(T v) \
        {                                                              \
            return make_##T##n(v);                                     \
        }                                                              \
    };
#define GET_VECTOR_TYPE_IMPL_N(T) \
    GET_VECTOR_TYPE_IMPL(T, 1)    \
    GET_VECTOR_TYPE_IMPL(T, 2)    \
    GET_VECTOR_TYPE_IMPL(T, 3)    \
    GET_VECTOR_TYPE_IMPL(T, 4)

GET_VECTOR_TYPE_IMPL_N(int)
GET_VECTOR_TYPE_IMPL_N(bool)
GET_VECTOR_TYPE_IMPL_N(uint)
GET_VECTOR_TYPE_IMPL_N(short)
GET_VECTOR_TYPE_IMPL_N(ushort)
GET_VECTOR_TYPE_IMPL_N(char)
GET_VECTOR_TYPE_IMPL_N(uchar)
GET_VECTOR_TYPE_IMPL_N(longlong)
GET_VECTOR_TYPE_IMPL_N(ulonglong)
GET_VECTOR_TYPE_IMPL_N(float)
GET_VECTOR_TYPE_IMPL_N(double)
#if SLANG_CUDA_ENABLE_HALF
GET_VECTOR_TYPE_IMPL_N(__half)
#endif
#if SLANG_CUDA_ENABLE_BF16
GET_VECTOR_TYPE_IMPL_N(__nv_bfloat16)
#endif
#if SLANG_CUDA_ENABLE_FP8
GET_VECTOR_TYPE_IMPL_N(__nv_fp8_e4m3)
GET_VECTOR_TYPE_IMPL_N(__nv_fp8_e5m2)
#endif

template<typename T, int n>
using Vector = typename GetVectorTypeImpl<T, n>::type;

template<typename T, int n, typename OtherT, int m>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Vector<T, n> _slang_vector_reshape(const Vector<OtherT, m> other)
{
    Vector<T, n> result;
    for (int i = 0; i < n; i++)
    {
        OtherT otherElement = T(0);
        if (i < m)
            otherElement = _slang_vector_get_element(other, i);
        *_slang_vector_get_element_ptr(&result, i) = (T)otherElement;
    }
    return result;
}

template<typename T, int ROWS, int COLS>
struct Matrix
{
    Vector<T, COLS> rows[ROWS];
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Vector<T, COLS>& operator[](size_t index)
    {
        return rows[index];
    }

    SLANG_FORCE_INLINE SLANG_CUDA_CALL const Vector<T, COLS>& operator[](size_t index) const
    {
        return rows[index];
    }
};


template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(T scalar)
{
    Matrix<T, ROWS, COLS> result;
    for (int i = 0; i < ROWS; i++)
        result.rows[i] = GetVectorTypeImpl<T, COLS>::fromScalar(scalar);
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(const Vector<T, COLS>& row0)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Vector<T, COLS>& row0,
    const Vector<T, COLS>& row1)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    result.rows[1] = row1;
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Vector<T, COLS>& row0,
    const Vector<T, COLS>& row1,
    const Vector<T, COLS>& row2)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    result.rows[1] = row1;
    result.rows[2] = row2;
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Vector<T, COLS>& row0,
    const Vector<T, COLS>& row1,
    const Vector<T, COLS>& row2,
    const Vector<T, COLS>& row3)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    result.rows[1] = row1;
    result.rows[2] = row2;
    result.rows[3] = row3;
    return result;
}

template<typename T, int ROWS, int COLS, typename U, int otherRow, int otherCol>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Matrix<U, otherRow, otherCol>& other)
{
    Matrix<T, ROWS, COLS> result;
    int minRow = ROWS;
    int minCol = COLS;
    if (minRow > otherRow)
        minRow = otherRow;
    if (minCol > otherCol)
        minCol = otherCol;
    for (int i = 0; i < minRow; i++)
        for (int j = 0; j < minCol; j++)
            *_slang_vector_get_element_ptr(result.rows + i, j) =
                (T)_slang_vector_get_element(other.rows[i], j);
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(T v0, T v1, T v2, T v3)
{
    Matrix<T, ROWS, COLS> rs;
    rs.rows[0].x = v0;
    rs.rows[0].y = v1;
    rs.rows[1].x = v2;
    rs.rows[1].y = v3;
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5)
{
    Matrix<T, ROWS, COLS> rs;
    if (COLS == 3)
    {
        *_slang_vector_get_element_ptr(&rs.rows[0], 0) = v0;
        *_slang_vector_get_element_ptr(&rs.rows[0], 1) = v1;
        *_slang_vector_get_element_ptr(&rs.rows[0], 2) = v2;
        *_slang_vector_get_element_ptr(&rs.rows[1], 0) = v3;
        *_slang_vector_get_element_ptr(&rs.rows[1], 1) = v4;
        *_slang_vector_get_element_ptr(&rs.rows[1], 2) = v5;
    }
    else
    {
        rs.rows[0].x = v0;
        rs.rows[0].y = v1;
        rs.rows[1].x = v2;
        rs.rows[1].y = v3;
        rs.rows[2].x = v4;
        rs.rows[2].y = v5;
    }
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7)
{
    Matrix<T, ROWS, COLS> rs;
    if (COLS == 4)
    {
        *_slang_vector_get_element_ptr(&rs.rows[0], 0) = v0;
        *_slang_vector_get_element_ptr(&rs.rows[0], 1) = v1;
        *_slang_vector_get_element_ptr(&rs.rows[0], 2) = v2;
        *_slang_vector_get_element_ptr(&rs.rows[0], 3) = v3;
        *_slang_vector_get_element_ptr(&rs.rows[1], 0) = v4;
        *_slang_vector_get_element_ptr(&rs.rows[1], 1) = v5;
        *_slang_vector_get_element_ptr(&rs.rows[1], 2) = v6;
        *_slang_vector_get_element_ptr(&rs.rows[1], 3) = v7;
    }
    else
    {
        rs.rows[0].x = v0;
        rs.rows[0].y = v1;
        rs.rows[1].x = v2;
        rs.rows[1].y = v3;
        rs.rows[2].x = v4;
        rs.rows[2].y = v5;
        rs.rows[3].x = v6;
        rs.rows[3].y = v7;
    }
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7,
    T v8)
{
    Matrix<T, ROWS, COLS> rs;
    rs.rows[0].x = v0;
    rs.rows[0].y = v1;
    rs.rows[0].z = v2;
    rs.rows[1].x = v3;
    rs.rows[1].y = v4;
    rs.rows[1].z = v5;
    rs.rows[2].x = v6;
    rs.rows[2].y = v7;
    rs.rows[2].z = v8;
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7,
    T v8,
    T v9,
    T v10,
    T v11)
{
    Matrix<T, ROWS, COLS> rs;
    if (COLS == 4)
    {
        *_slang_vector_get_element_ptr(&rs.rows[0], 0) = v0;
        *_slang_vector_get_element_ptr(&rs.rows[0], 1) = v1;
        *_slang_vector_get_element_ptr(&rs.rows[0], 2) = v2;
        *_slang_vector_get_element_ptr(&rs.rows[0], 3) = v3;
        *_slang_vector_get_element_ptr(&rs.rows[1], 0) = v4;
        *_slang_vector_get_element_ptr(&rs.rows[1], 1) = v5;
        *_slang_vector_get_element_ptr(&rs.rows[1], 2) = v6;
        *_slang_vector_get_element_ptr(&rs.rows[1], 3) = v7;
        *_slang_vector_get_element_ptr(&rs.rows[2], 0) = v8;
        *_slang_vector_get_element_ptr(&rs.rows[2], 1) = v9;
        *_slang_vector_get_element_ptr(&rs.rows[2], 2) = v10;
        *_slang_vector_get_element_ptr(&rs.rows[2], 3) = v11;
    }
    else
    {
        rs.rows[0].x = v0;
        rs.rows[0].y = v1;
        rs.rows[0].z = v2;
        rs.rows[1].x = v3;
        rs.rows[1].y = v4;
        rs.rows[1].z = v5;
        rs.rows[2].x = v6;
        rs.rows[2].y = v7;
        rs.rows[2].z = v8;
        rs.rows[3].x = v9;
        rs.rows[3].y = v10;
        rs.rows[3].z = v11;
    }
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7,
    T v8,
    T v9,
    T v10,
    T v11,
    T v12,
    T v13,
    T v14,
    T v15)
{
    Matrix<T, ROWS, COLS> rs;
    rs.rows[0].x = v0;
    rs.rows[0].y = v1;
    rs.rows[0].z = v2;
    rs.rows[0].w = v3;
    rs.rows[1].x = v4;
    rs.rows[1].y = v5;
    rs.rows[1].z = v6;
    rs.rows[1].w = v7;
    rs.rows[2].x = v8;
    rs.rows[2].y = v9;
    rs.rows[2].z = v10;
    rs.rows[2].w = v11;
    rs.rows[3].x = v12;
    rs.rows[3].y = v13;
    rs.rows[3].z = v14;
    rs.rows[3].w = v15;
    return rs;
}

#define SLANG_MATRIX_BINARY_OP(T, op)                                   \
    template<int R, int C>                                              \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator op(     \
        const Matrix<T, R, C>& thisVal,                                 \
        const Matrix<T, R, C>& other)                                   \
    {                                                                   \
        Matrix<T, R, C> result;                                         \
        for (int i = 0; i < R; i++)                                     \
            for (int j = 0; j < C; j++)                                 \
                *_slang_vector_get_element_ptr(result.rows + i, j) =    \
                    _slang_vector_get_element(thisVal.rows[i], j)       \
                        op _slang_vector_get_element(other.rows[i], j); \
        return result;                                                  \
    }

#define SLANG_MATRIX_UNARY_OP(T, op)                                                               \
    template<int R, int C>                                                                         \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator op(const Matrix<T, R, C>& thisVal) \
    {                                                                                              \
        Matrix<T, R, C> result;                                                                    \
        for (int i = 0; i < R; i++)                                                                \
            for (int j = 0; j < C; j++)                                                            \
                *_slang_vector_get_element_ptr(result.rows + i, j) =                               \
                    op _slang_vector_get_element(thisVal.rows[i], j);                              \
        return result;                                                                             \
    }
#define SLANG_INT_MATRIX_OPS(T)   \
    SLANG_MATRIX_BINARY_OP(T, +)  \
    SLANG_MATRIX_BINARY_OP(T, -)  \
    SLANG_MATRIX_BINARY_OP(T, *)  \
    SLANG_MATRIX_BINARY_OP(T, /)  \
    SLANG_MATRIX_BINARY_OP(T, &)  \
    SLANG_MATRIX_BINARY_OP(T, |)  \
    SLANG_MATRIX_BINARY_OP(T, &&) \
    SLANG_MATRIX_BINARY_OP(T, ||) \
    SLANG_MATRIX_BINARY_OP(T, ^)  \
    SLANG_MATRIX_BINARY_OP(T, %)  \
    SLANG_MATRIX_UNARY_OP(T, !)   \
    SLANG_MATRIX_UNARY_OP(T, ~)
#define SLANG_FLOAT_MATRIX_OPS(T) \
    SLANG_MATRIX_BINARY_OP(T, +)  \
    SLANG_MATRIX_BINARY_OP(T, -)  \
    SLANG_MATRIX_BINARY_OP(T, *)  \
    SLANG_MATRIX_BINARY_OP(T, /)  \
    SLANG_MATRIX_UNARY_OP(T, -)
SLANG_INT_MATRIX_OPS(int)
SLANG_INT_MATRIX_OPS(uint)
SLANG_INT_MATRIX_OPS(short)
SLANG_INT_MATRIX_OPS(ushort)
SLANG_INT_MATRIX_OPS(char)
SLANG_INT_MATRIX_OPS(uchar)
SLANG_INT_MATRIX_OPS(longlong)
SLANG_INT_MATRIX_OPS(ulonglong)
SLANG_FLOAT_MATRIX_OPS(float)
SLANG_FLOAT_MATRIX_OPS(double)
#if SLANG_CUDA_ENABLE_HALF
SLANG_FLOAT_MATRIX_OPS(__half)
#endif
#define SLANG_MATRIX_INT_NEG_OP(T)                                                        \
    template<int R, int C>                                                                \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator-(Matrix<T, R, C> thisVal) \
    {                                                                                     \
        Matrix<T, R, C> result;                                                           \
        for (int i = 0; i < R; i++)                                                       \
            for (int j = 0; j < C; j++)                                                   \
                *_slang_vector_get_element_ptr(result.rows + i, j) =                      \
                    0 - _slang_vector_get_element(thisVal.rows[i], j);                    \
        return result;                                                                    \
    }
SLANG_MATRIX_INT_NEG_OP(int)
SLANG_MATRIX_INT_NEG_OP(uint)
SLANG_MATRIX_INT_NEG_OP(short)
SLANG_MATRIX_INT_NEG_OP(ushort)
SLANG_MATRIX_INT_NEG_OP(char)
SLANG_MATRIX_INT_NEG_OP(uchar)
SLANG_MATRIX_INT_NEG_OP(longlong)
SLANG_MATRIX_INT_NEG_OP(ulonglong)

#define SLANG_FLOAT_MATRIX_MOD(T)                                                 \
    template<int R, int C>                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator%(                 \
        Matrix<T, R, C> left,                                                     \
        Matrix<T, R, C> right)                                                    \
    {                                                                             \
        Matrix<T, R, C> result;                                                   \
        for (int i = 0; i < R; i++)                                               \
            for (int j = 0; j < C; j++)                                           \
                *_slang_vector_get_element_ptr(result.rows + i, j) = _slang_fmod( \
                    _slang_vector_get_element(left.rows[i], j),                   \
                    _slang_vector_get_element(right.rows[i], j));                 \
        return result;                                                            \
    }

SLANG_FLOAT_MATRIX_MOD(float)
SLANG_FLOAT_MATRIX_MOD(double)
#if SLANG_CUDA_ENABLE_HALF
template<int R, int C>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<__half, R, C> operator%(
    Matrix<__half, R, C> left,
    Matrix<__half, R, C> right)
{
    Matrix<__half, R, C> result;
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++)
            *_slang_vector_get_element_ptr(result.rows + i, j) = __float2half(_slang_fmod(
                __half2float(_slang_vector_get_element(left.rows[i], j)),
                __half2float(_slang_vector_get_element(right.rows[i], j))));
    return result;
}
#endif
#undef SLANG_FLOAT_MATRIX_MOD
#undef SLANG_MATRIX_BINARY_OP
#undef SLANG_MATRIX_UNARY_OP
#undef SLANG_INT_MATRIX_OPS
#undef SLANG_FLOAT_MATRIX_OPS
#undef SLANG_MATRIX_INT_NEG_OP
#undef SLANG_FLOAT_MATRIX_MOD

#define SLANG_SELECT_IMPL(T, N)                                                                  \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Vector<T, N> _slang_select(                               \
        bool##N condition,                                                                       \
        Vector<T, N> v0,                                                                         \
        Vector<T, N> v1)                                                                         \
    {                                                                                            \
        Vector<T, N> result;                                                                     \
        for (int i = 0; i < N; i++)                                                              \
        {                                                                                        \
            *_slang_vector_get_element_ptr(&result, i) = _slang_vector_get_element(condition, i) \
                                                             ? _slang_vector_get_element(v0, i)  \
                                                             : _slang_vector_get_element(v1, i); \
        }                                                                                        \
        return result;                                                                           \
    }
#define SLANG_SELECT_T(T)   \
    SLANG_SELECT_IMPL(T, 2) \
    SLANG_SELECT_IMPL(T, 3) \
    SLANG_SELECT_IMPL(T, 4)

SLANG_SELECT_T(int)
SLANG_SELECT_T(bool)
SLANG_SELECT_T(uint)
SLANG_SELECT_T(short)
SLANG_SELECT_T(ushort)
SLANG_SELECT_T(char)
SLANG_SELECT_T(uchar)
SLANG_SELECT_T(float)
SLANG_SELECT_T(double)

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_select(bool condition, T v0, T v1)
{
    return condition ? v0 : v1;
}

//
// Half support
//

#if SLANG_CUDA_ENABLE_HALF
SLANG_SELECT_T(__half)

// Convenience functions ushort -> half

SLANG_FORCE_INLINE SLANG_CUDA_CALL __half2 __ushort_as_half(const ushort2& i)
{
    return __halves2half2(__ushort_as_half(i.x), __ushort_as_half(i.y));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half3 __ushort_as_half(const ushort3& i)
{
    return __half3{__ushort_as_half(i.x), __ushort_as_half(i.y), __ushort_as_half(i.z)};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half4 __ushort_as_half(const ushort4& i)
{
    return __half4{
        __ushort_as_half(i.x),
        __ushort_as_half(i.y),
        __ushort_as_half(i.z),
        __ushort_as_half(i.w)};
}

// Convenience functions half -> ushort

SLANG_FORCE_INLINE SLANG_CUDA_CALL ushort2 __half_as_ushort(const __half2& i)
{
    return make_ushort2(__half_as_ushort(i.x), __half_as_ushort(i.y));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL ushort3 __half_as_ushort(const __half3& i)
{
    return make_ushort3(__half_as_ushort(i.x), __half_as_ushort(i.y), __half_as_ushort(i.z));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL ushort4 __half_as_ushort(const __half4& i)
{
    return make_ushort4(
        __half_as_ushort(i.x),
        __half_as_ushort(i.y),
        __half_as_ushort(i.z),
        __half_as_ushort(i.w));
}

// This is a little bit of a hack. Fortunately CUDA has the definitions of the templated types in
// include/surface_indirect_functions.h
// Here we find the template definition requires a specialization of __nv_isurf_trait to allow
// a specialization of the surface write functions.
// This *isn't* a problem on the read functions as they don't have a return type that uses this
// mechanism

template<>
struct __nv_isurf_trait<__half>
{
    typedef void type;
};
template<>
struct __nv_isurf_trait<__half2>
{
    typedef void type;
};
template<>
struct __nv_isurf_trait<__half4>
{
    typedef void type;
};

#define SLANG_DROP_PARENS(...) __VA_ARGS__

#define SLANG_SURFACE_READ(FUNC_NAME, TYPE_ARGS, ARGS)                                             \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL __half FUNC_NAME<__half>(                                   \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        return __ushort_as_half(FUNC_NAME<ushort>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode)); \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL __half2 FUNC_NAME<__half2>(                                 \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        return __ushort_as_half(                                                                   \
            FUNC_NAME<ushort2>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode));                    \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL __half4 FUNC_NAME<__half4>(                                 \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        return __ushort_as_half(                                                                   \
            FUNC_NAME<ushort4>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode));                    \
    }

SLANG_SURFACE_READ(surf1Dread, (int x), (x))
SLANG_SURFACE_READ(surf2Dread, (int x, int y), (x, y))
SLANG_SURFACE_READ(surf3Dread, (int x, int y, int z), (x, y, z))
SLANG_SURFACE_READ(surf1DLayeredread, (int x, int layer), (x, layer))
SLANG_SURFACE_READ(surf2DLayeredread, (int x, int y, int layer), (x, y, layer))
SLANG_SURFACE_READ(surfCubemapread, (int x, int y, int face), (x, y, face))
SLANG_SURFACE_READ(surfCubemapLayeredread, (int x, int y, int layerFace), (x, y, layerFace))

#define SLANG_SURFACE_WRITE(FUNC_NAME, TYPE_ARGS, ARGS)                                            \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void FUNC_NAME<__half>(                                     \
        __half data,                                                                               \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        FUNC_NAME<ushort>(__half_as_ushort(data), surfObj, SLANG_DROP_PARENS ARGS, boundaryMode);  \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void FUNC_NAME<__half2>(                                    \
        __half2 data,                                                                              \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        FUNC_NAME<ushort2>(__half_as_ushort(data), surfObj, SLANG_DROP_PARENS ARGS, boundaryMode); \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void FUNC_NAME<__half4>(                                    \
        __half4 data,                                                                              \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        FUNC_NAME<ushort4>(__half_as_ushort(data), surfObj, SLANG_DROP_PARENS ARGS, boundaryMode); \
    }

SLANG_SURFACE_WRITE(surf1Dwrite, (int x), (x))
SLANG_SURFACE_WRITE(surf2Dwrite, (int x, int y), (x, y))
SLANG_SURFACE_WRITE(surf3Dwrite, (int x, int y, int z), (x, y, z))
SLANG_SURFACE_WRITE(surf1DLayeredwrite, (int x, int layer), (x, layer))
SLANG_SURFACE_WRITE(surf2DLayeredwrite, (int x, int y, int layer), (x, y, layer))
SLANG_SURFACE_WRITE(surfCubemapwrite, (int x, int y, int face), (x, y, face))
SLANG_SURFACE_WRITE(surfCubemapLayeredwrite, (int x, int y, int layerFace), (x, y, layerFace))

// ! Hack to test out reading !!!
// Only works converting *from* half

// template <typename T>
// SLANG_FORCE_INLINE SLANG_CUDA_CALL T surf2Dread_convert(cudaSurfaceObject_t surfObj, int x, int
// y, cudaSurfaceBoundaryMode boundaryMode);

#define SLANG_SURFACE_READ_HALF_CONVERT(FUNC_NAME, TYPE_ARGS, ARGS)                              \
                                                                                                 \
    template<typename T>                                                                         \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T FUNC_NAME##_convert(                                    \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode);                                                   \
                                                                                                 \
    template<>                                                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL float FUNC_NAME##_convert<float>(                         \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode)                                                    \
    {                                                                                            \
        return __ushort_as_half(                                                                 \
            FUNC_NAME<uint16_t>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode));                 \
    }                                                                                            \
                                                                                                 \
    template<>                                                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL float2 FUNC_NAME##_convert<float2>(                       \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode)                                                    \
    {                                                                                            \
        const __half2 v =                                                                        \
            __ushort_as_half(FUNC_NAME<ushort2>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode)); \
        return float2{v.x, v.y};                                                                 \
    }                                                                                            \
                                                                                                 \
    template<>                                                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL float4 FUNC_NAME##_convert<float4>(                       \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode)                                                    \
    {                                                                                            \
        const __half4 v =                                                                        \
            __ushort_as_half(FUNC_NAME<ushort4>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode)); \
        return float4{v.x, v.y, v.z, v.w};                                                       \
    }

SLANG_SURFACE_READ_HALF_CONVERT(surf1Dread, (int x), (x))
SLANG_SURFACE_READ_HALF_CONVERT(surf2Dread, (int x, int y), (x, y))
SLANG_SURFACE_READ_HALF_CONVERT(surf3Dread, (int x, int y, int z), (x, y, z))

#endif

// Support for doing format conversion when writing to a surface/RWTexture

// NOTE! For normal surface access x values are *byte* addressed.
// For the _convert versions they are *not*. They don't need to be because sust.p does not require
// it.

// https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-sust


// surf1Dwrite_convert

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert(
    T v,
    cudaSurfaceObject_t surfObj,
    int x,
    cudaSurfaceBoundaryMode boundaryMode);

#define SLANG_SURF1DWRITE_CONVERT_IMPL(T, c)                                                     \
    template<>                                                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert<T>(                              \
        T v,                                                                                     \
        cudaSurfaceObject_t surfObj,                                                             \
        int x,                                                                                   \
        cudaSurfaceBoundaryMode boundaryMode)                                                    \
    {                                                                                            \
        asm volatile(                                                                            \
            "sust.p.1d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1}], {%2};" ::"l"(surfObj),        \
            "r"(x),                                                                              \
            c(v));                                                                               \
    }                                                                                            \
    template<>                                                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert<T##2>(                           \
        T##2 v,                                                                                  \
        cudaSurfaceObject_t surfObj,                                                             \
        int x,                                                                                   \
        cudaSurfaceBoundaryMode boundaryMode)                                                    \
    {                                                                                            \
        const T vx = v.x, vy = v.y;                                                              \
        asm volatile(                                                                            \
            "sust.p.1d.v2.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1}], {%2, %3};" ::"l"(surfObj), \
            "r"(x),                                                                              \
            c(vx),                                                                               \
            c(vy));                                                                              \
    }                                                                                            \
    template<>                                                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert<T##4>(                           \
        T##4 v,                                                                                  \
        cudaSurfaceObject_t surfObj,                                                             \
        int x,                                                                                   \
        cudaSurfaceBoundaryMode boundaryMode)                                                    \
    {                                                                                            \
        const T vx = v.x, vy = v.y, vz = v.z, vw = v.w;                                          \
        asm volatile(                                                                            \
            "sust.p.1d.v4.b32." SLANG_PTX_BOUNDARY_MODE                                          \
            " [%0, {%1}], {%2, %3, %4, %5};" ::"l"(surfObj),                                     \
            "r"(x),                                                                              \
            c(vx),                                                                               \
            c(vy),                                                                               \
            c(vz),                                                                               \
            c(vw));                                                                              \
    }

SLANG_SURF1DWRITE_CONVERT_IMPL(float, "f")
SLANG_SURF1DWRITE_CONVERT_IMPL(uint, "r")
SLANG_SURF1DWRITE_CONVERT_IMPL(int, "r")

// surf1DLayeredwrite_convert (not supported)

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1DLayeredwrite_convert(
    T v,
    cudaSurfaceObject_t surfObj,
    int x,
    int layer,
    cudaSurfaceBoundaryMode boundaryMode)
{
    // TODO: static_assert(false) can fail on some compilers, even if template is not instantiated.
    // We should check for this in hlsl.meta.slang instead.
    // static_assert(false, "CUDA doesn't support formatted surface writes on 1D array surfaces");
}

// surf2Dwrite_convert

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert(
    T v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    cudaSurfaceBoundaryMode boundaryMode);

#define SLANG_SURF2DWRITE_CONVERT_IMPL(T, c)                                                  \
    template<>                                                                                \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert<T>(                           \
        T v,                                                                                  \
        cudaSurfaceObject_t surfObj,                                                          \
        int x,                                                                                \
        int y,                                                                                \
        cudaSurfaceBoundaryMode boundaryMode)                                                 \
    {                                                                                         \
        asm volatile(                                                                         \
            "sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1, %2}], {%3};" ::"l"(surfObj), \
            "r"(x),                                                                           \
            "r"(y),                                                                           \
            c(v));                                                                            \
    }                                                                                         \
    template<>                                                                                \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert<T##2>(                        \
        T##2 v,                                                                               \
        cudaSurfaceObject_t surfObj,                                                          \
        int x,                                                                                \
        int y,                                                                                \
        cudaSurfaceBoundaryMode boundaryMode)                                                 \
    {                                                                                         \
        const T vx = v.x, vy = v.y;                                                           \
        asm volatile(                                                                         \
            "sust.p.2d.v2.b32." SLANG_PTX_BOUNDARY_MODE                                       \
            " [%0, {%1, %2}], {%3, %4};" ::"l"(surfObj),                                      \
            "r"(x),                                                                           \
            "r"(y),                                                                           \
            c(vx),                                                                            \
            c(vy));                                                                           \
    }                                                                                         \
    template<>                                                                                \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert<T##4>(                        \
        T##4 v,                                                                               \
        cudaSurfaceObject_t surfObj,                                                          \
        int x,                                                                                \
        int y,                                                                                \
        cudaSurfaceBoundaryMode boundaryMode)                                                 \
    {                                                                                         \
        const T vx = v.x, vy = v.y, vz = v.z, vw = v.w;                                       \
        asm volatile(                                                                         \
            "sust.p.2d.v4.b32." SLANG_PTX_BOUNDARY_MODE                                       \
            " [%0, {%1, %2}], {%3, %4, %5, %6};" ::"l"(surfObj),                              \
            "r"(x),                                                                           \
            "r"(y),                                                                           \
            c(vx),                                                                            \
            c(vy),                                                                            \
            c(vz),                                                                            \
            c(vw));                                                                           \
    }

SLANG_SURF2DWRITE_CONVERT_IMPL(float, "f")
SLANG_SURF2DWRITE_CONVERT_IMPL(uint, "r")
SLANG_SURF2DWRITE_CONVERT_IMPL(int, "r")

// surf2DLayeredwrite_convert (not supported)

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2DLayeredwrite_convert(
    T v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    int layer,
    cudaSurfaceBoundaryMode boundaryMode)
{
    // TODO: static_assert(false) can fail on some compilers, even if template is not instantiated.
    // We should check for this in hlsl.meta.slang instead.
    // static_assert(false, "CUDA doesn't support formatted surface writes on 2D array surfaces");
}

// surf3Dwrite_convert

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert(
    T v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    int z,
    cudaSurfaceBoundaryMode boundaryMode);

#define SLANG_SURF3DWRITE_CONVERT_IMPL(T, c)                             \
    template<>                                                           \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert<T>(      \
        T v,                                                             \
        cudaSurfaceObject_t surfObj,                                     \
        int x,                                                           \
        int y,                                                           \
        int z,                                                           \
        cudaSurfaceBoundaryMode boundaryMode)                            \
    {                                                                    \
        asm volatile(                                                    \
            "sust.p.3d.b32." SLANG_PTX_BOUNDARY_MODE                     \
            " [%0, {%1, %2, %3, %4}], {%5};" ::"l"(surfObj),             \
            "r"(x),                                                      \
            "r"(y),                                                      \
            "r"(z),                                                      \
            "r"(0),                                                      \
            c(v));                                                       \
    }                                                                    \
    template<>                                                           \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert<T##2>(   \
        T##2 v,                                                          \
        cudaSurfaceObject_t surfObj,                                     \
        int x,                                                           \
        int y,                                                           \
        int z,                                                           \
        cudaSurfaceBoundaryMode boundaryMode)                            \
    {                                                                    \
        const T vx = v.x, vy = v.y;                                      \
        asm volatile(                                                    \
            "sust.p.3d.v2.b32." SLANG_PTX_BOUNDARY_MODE                  \
            " [%0, {%1, %2, %3, %4}], {%5, %6};" ::"l"(surfObj),         \
            "r"(x),                                                      \
            "r"(y),                                                      \
            "r"(z),                                                      \
            "r"(0),                                                      \
            c(vx),                                                       \
            c(vy));                                                      \
    }                                                                    \
    template<>                                                           \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert<T##4>(   \
        T##4 v,                                                          \
        cudaSurfaceObject_t surfObj,                                     \
        int x,                                                           \
        int y,                                                           \
        int z,                                                           \
        cudaSurfaceBoundaryMode boundaryMode)                            \
    {                                                                    \
        const T vx = v.x, vy = v.y, vz = v.z, vw = v.w;                  \
        asm volatile(                                                    \
            "sust.p.3d.v4.b32." SLANG_PTX_BOUNDARY_MODE                  \
            " [%0, {%1, %2, %3, %4}], {%5, %6, %7, %8};" ::"l"(surfObj), \
            "r"(x),                                                      \
            "r"(y),                                                      \
            "r"(z),                                                      \
            "r"(0),                                                      \
            c(vx),                                                       \
            c(vy),                                                       \
            c(vz),                                                       \
            c(vw));                                                      \
    }

SLANG_SURF3DWRITE_CONVERT_IMPL(float, "f")
SLANG_SURF3DWRITE_CONVERT_IMPL(uint, "r")
SLANG_SURF3DWRITE_CONVERT_IMPL(int, "r")

// ----------------------------- F16 -----------------------------------------
#if SLANG_CUDA_ENABLE_HALF
// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_ceil(__half f)
{
    return ::hceil(f);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_floor(__half f)
{
    return ::hfloor(f);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_round(__half f)
{
    return ::hrint(f);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_sin(__half f)
{
    return ::hsin(f);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_cos(__half f)
{
    return ::hcos(f);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL void F16_sincos(__half f, __half* s, __half* c)
{
    *s = ::hsin(f);
    *c = ::hcos(f);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_tan(__half f)
{
    return __float2half(::tanf(__half2float(f)));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_asin(__half f)
{
    return __float2half(::asinf(__half2float(f)));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_acos(__half f)
{
    return __float2half(::acosf(__half2float(f)));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_atan(__half f)
{
    return __float2half(::atanf(__half2float(f)));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_sinh(__half f)
{
    return __float2half(::sinhf(__half2float(f)));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_cosh(__half f)
{
    return __float2half(::coshf(__half2float(f)));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_tanh(__half f)
{
    return __float2half(::tanhf(__half2float(f)));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_asinh(__half f)
{
    return __float2half(::asinhf(__half2float(f)));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_acosh(__half f)
{
    return __float2half(::acoshf(__half2float(f)));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_atanh(__half f)
{
    return __float2half(::atanhf(__half2float(f)));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_log2(__half f)
{
    return ::hlog2(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_log(__half f)
{
    return ::hlog(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_log10(__half f)
{
    return ::hlog10(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_exp2(__half f)
{
    return ::hexp2(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_exp(__half f)
{
    return ::hexp(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_abs(__half f)
{
    return __habs(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_trunc(__half f)
{
    return ::htrunc(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_sqrt(__half f)
{
    return ::hsqrt(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_rsqrt(__half f)
{
    return ::hrsqrt(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int F16_sign(__half f)
{
    return (f == __half(0.0f)) ? 0 : ((f < __half(0.0f)) ? -1 : 1);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_frac(__half f)
{
    return f - F16_floor(f);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F16_isnan(__half f)
{
    return __hisnan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F16_isfinite(__half f)
{
    return !__hisinf(f) && !__hisnan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F16_isinf(__half f)
{
    return __hisinf(f);
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_min(__half a, __half b)
{
    return __hmin(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_max(__half a, __half b)
{
    return __hmax(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_pow(__half a, __half b)
{
    return __float2half(::powf(__half2float(a), __half2float(b)));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_fmod(__half a, __half b)
{
    return __float2half(::fmodf(__half2float(a), __half2float(b)));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_remainder(__half a, __half b)
{
    return __float2half(::remainderf(__half2float(a), __half2float(b)));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_atan2(__half a, __half b)
{
    return __float2half(::atan2(__half2float(a), __half2float(b)));
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_frexp(__half x, int* e)
{
    return __float2half(frexpf(__half2float(x), e));
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_modf(__half x, __half* ip)
{
    float ipf;
    float res = ::modff(__half2float(x), &ipf);
    *ip = __float2half(ipf);
    return __float2half(res);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint16_t F16_asuint(__half h)
{
    return __half_as_ushort(h);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int16_t F16_asint(__half h)
{
    return __half_as_short(h);
}

// Ternary
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half F16_fma(__half a, __half b, __half c)
{
    return __hfma(a, b, c);
}

#endif

// ----------------------------- F32 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_ceil(float f)
{
    return ::ceilf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_floor(float f)
{
    return ::floorf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_round(float f)
{
    return ::roundf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sin(float f)
{
    return ::sinf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_cos(float f)
{
    return ::cosf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL void F32_sincos(float f, float* s, float* c)
{
    ::sincosf(f, s, c);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_tan(float f)
{
    return ::tanf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_asin(float f)
{
    return ::asinf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_acos(float f)
{
    return ::acosf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_atan(float f)
{
    return ::atanf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sinh(float f)
{
    return ::sinhf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_cosh(float f)
{
    return ::coshf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_tanh(float f)
{
    return ::tanhf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_asinh(float f)
{
    return ::asinhf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_acosh(float f)
{
    return ::acoshf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_atanh(float f)
{
    return ::atanhf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_log2(float f)
{
    return ::log2f(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_log(float f)
{
    return ::logf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_log10(float f)
{
    return ::log10f(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_exp2(float f)
{
    return ::exp2f(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_exp(float f)
{
    return ::expf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_abs(float f)
{
    return ::fabsf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_trunc(float f)
{
    return ::truncf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sqrt(float f)
{
    return ::sqrtf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_rsqrt(float f)
{
    return ::rsqrtf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int F32_sign(float f)
{
    return (f == 0.0f) ? 0 : ((f < 0.0f) ? -1 : 1);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_frac(float f)
{
    return f - F32_floor(f);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F32_isnan(float f)
{
    return isnan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F32_isfinite(float f)
{
    return isfinite(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F32_isinf(float f)
{
    return isinf(f);
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_min(float a, float b)
{
    return ::fminf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_max(float a, float b)
{
    return ::fmaxf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_pow(float a, float b)
{
    return ::powf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_fmod(float a, float b)
{
    return ::fmodf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_remainder(float a, float b)
{
    return ::remainderf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_atan2(float a, float b)
{
    return float(::atan2(a, b));
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_frexp(float x, int* e)
{
    return frexpf(x, e);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_modf(float x, float* ip)
{
    return ::modff(x, ip);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t F32_asuint(float f)
{
    Union32 u;
    u.f = f;
    return u.u;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t F32_asint(float f)
{
    Union32 u;
    u.f = f;
    return u.i;
}

// Ternary
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_fma(float a, float b, float c)
{
    return ::fmaf(a, b, c);
}


// ----------------------------- F64 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_ceil(double f)
{
    return ::ceil(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_floor(double f)
{
    return ::floor(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_round(double f)
{
    return ::round(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sin(double f)
{
    return ::sin(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_cos(double f)
{
    return ::cos(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL void F64_sincos(double f, double* s, double* c)
{
    ::sincos(f, s, c);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_tan(double f)
{
    return ::tan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_asin(double f)
{
    return ::asin(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_acos(double f)
{
    return ::acos(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_atan(double f)
{
    return ::atan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sinh(double f)
{
    return ::sinh(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_cosh(double f)
{
    return ::cosh(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_tanh(double f)
{
    return ::tanh(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_log2(double f)
{
    return ::log2(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_log(double f)
{
    return ::log(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_log10(float f)
{
    return ::log10(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_exp2(double f)
{
    return ::exp2(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_exp(double f)
{
    return ::exp(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_abs(double f)
{
    return ::fabs(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_trunc(double f)
{
    return ::trunc(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sqrt(double f)
{
    return ::sqrt(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_rsqrt(double f)
{
    return ::rsqrt(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int F64_sign(double f)
{
    return (f == 0.0) ? 0 : ((f < 0.0) ? -1 : 1);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_frac(double f)
{
    return f - F64_floor(f);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F64_isnan(double f)
{
    return isnan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F64_isfinite(double f)
{
    return isfinite(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F64_isinf(double f)
{
    return isinf(f);
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_min(double a, double b)
{
    return ::fmin(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_max(double a, double b)
{
    return ::fmax(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_pow(double a, double b)
{
    return ::pow(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_fmod(double a, double b)
{
    return ::fmod(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_remainder(double a, double b)
{
    return ::remainder(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_atan2(double a, double b)
{
    return ::atan2(a, b);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_frexp(double x, int* e)
{
    return ::frexp(x, e);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_modf(double x, double* ip)
{
    return ::modf(x, ip);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL void F64_asuint(double d, uint32_t* low, uint32_t* hi)
{
    Union64 u;
    u.d = d;
    *low = uint32_t(u.u);
    *hi = uint32_t(u.u >> 32);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL void F64_asint(double d, int32_t* low, int32_t* hi)
{
    Union64 u;
    u.d = d;
    *low = int32_t(u.u);
    *hi = int32_t(u.u >> 32);
}

// Ternary
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_fma(double a, double b, double c)
{
    return ::fma(a, b, c);
}

// ----------------------------- U8 -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U8_countbits(uint8_t v)
{
    // No native 8bit popc yet, just cast and use 32bit variant
    return __popc(uint32_t(v));
}

// ----------------------------- I8 -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t I8_countbits(int8_t v)
{
    return U8_countbits(uint8_t(v));
}

// ----------------------------- U16 -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U16_countbits(uint16_t v)
{
    // No native 16bit popc yet, just cast and use 32bit variant
    return __popc(uint32_t(v));
}

// ----------------------------- I16 -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t I16_countbits(int16_t v)
{
    return U16_countbits(uint16_t(v));
}

// ----------------------------- U32 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_abs(uint32_t f)
{
    return f;
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_min(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_max(uint32_t a, uint32_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float U32_asfloat(uint32_t x)
{
    Union32 u;
    u.u = x;
    return u.f;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_asint(int32_t x)
{
    return uint32_t(x);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL double U32_asdouble(uint32_t low, uint32_t hi)
{
    Union64 u;
    u.u = (uint64_t(hi) << 32) | low;
    return u.d;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_countbits(uint32_t v)
{
    return __popc(v);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_firstbitlow(uint32_t v)
{
    // __ffs returns 1-based bit position or 0 if no bits set
    // firstbitlow should return 0-based bit position or ~0u if no bits set
    return v == 0 ? ~0u : (__ffs(v) - 1);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_firstbithigh(uint32_t v)
{
    // maps to hlsl firstbithigh
    if ((int32_t)v < 0)
        v = ~v;
    if (v == 0)
        return ~0u;
    return 31 - __clz(v);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_reversebits(uint32_t v)
{
    return __brev(v);
}

// ----------------------------- I32 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t I32_abs(int32_t f)
{
    return (f < 0) ? -f : f;
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t I32_min(int32_t a, int32_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t I32_max(int32_t a, int32_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float I32_asfloat(int32_t x)
{
    Union32 u;
    u.i = x;
    return u.f;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t I32_asuint(int32_t x)
{
    return uint32_t(x);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double I32_asdouble(int32_t low, int32_t hi)
{
    Union64 u;
    u.u = (uint64_t(hi) << 32) | uint32_t(low);
    return u.d;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t I32_countbits(int32_t v)
{
    return U32_countbits(uint32_t(v));
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t I32_firstbitlow(int32_t v)
{
    return U32_firstbitlow(uint32_t(v));
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t I32_firstbithigh(int32_t v)
{
    return U32_firstbithigh(uint32_t(v));
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t I32_reversebits(int32_t v)
{
    return int32_t(U32_reversebits(uint32_t(v)));
}

// ----------------------------- U64 -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t U64_abs(uint64_t f)
{
    return f;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t U64_min(uint64_t a, uint64_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t U64_max(uint64_t a, uint64_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U64_countbits(uint64_t v)
{
    return __popcll(v);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U64_firstbitlow(uint64_t v)
{
    // __ffs returns 1-based bit position or 0 if no bits set
    // firstbitlow should return 0-based bit position or ~0u if no bits set
    return v == 0 ? ~uint32_t(0) : (__ffsll(v) - 1u);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U64_firstbithigh(uint64_t v)
{
    if (v == 0)
        return ~uint32_t(0);
    return 63 - __clzll(v);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint64_t U64_reversebits(uint64_t v)
{
    return __brevll(v);
}

// ----------------------------- I64 -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t I64_abs(int64_t f)
{
    return (f < 0) ? -f : f;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t I64_min(int64_t a, int64_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t I64_max(int64_t a, int64_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t I64_countbits(int64_t v)
{
    return U64_countbits(uint64_t(v));
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t I64_firstbitlow(int64_t v)
{
    return U64_firstbitlow(uint64_t(v));
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t I64_firstbithigh(int64_t v)
{
    if (v < 0)
        v = ~v;
    return U64_firstbithigh(uint64_t(v));
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t I64_reversebits(int64_t v)
{
    return int64_t(U64_reversebits(uint64_t(v)));
}

// ----------------------------- IPTR -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL intptr_t IPTR_abs(intptr_t f)
{
    return (f < 0) ? -f : f;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL intptr_t IPTR_min(intptr_t a, intptr_t b)
{
    return a < b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL intptr_t IPTR_max(intptr_t a, intptr_t b)
{
    return a > b ? a : b;
}

// ----------------------------- UPTR -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL uintptr_t UPTR_abs(uintptr_t f)
{
    return f;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uintptr_t UPTR_min(uintptr_t a, uintptr_t b)
{
    return a < b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uintptr_t UPTR_max(uintptr_t a, uintptr_t b)
{
    return a > b ? a : b;
}

// ----------------------------- ResourceType -----------------------------------------


// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-structuredbuffer-getdimensions
// Missing  Load(_In_  int  Location, _Out_ uint Status);

template<typename T>
struct StructuredBuffer
{
    SLANG_CUDA_CALL T& operator[](size_t index) const
    {
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
        SLANG_BOUND_CHECK(index, count);
#endif
        return data[index];
    }

    SLANG_CUDA_CALL T& Load(size_t index) const
    {
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
        SLANG_BOUND_CHECK(index, count);
#endif
        return data[index];
    }

#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
    SLANG_CUDA_CALL void GetDimensions(uint32_t* outNumStructs, uint32_t* outStride) const
    {
        *outNumStructs = uint32_t(count);
        *outStride = uint32_t(sizeof(T));
    }
#endif

    T* data;
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
    size_t count;
#endif
};

template<typename T>
struct RWStructuredBuffer : StructuredBuffer<T>
{
    SLANG_CUDA_CALL T& operator[](size_t index) const
    {
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
        SLANG_BOUND_CHECK(index, this->count);
#endif
        return this->data[index];
    }
};

// Missing  Load(_In_  int  Location, _Out_ uint Status);
struct ByteAddressBuffer
{
    SLANG_CUDA_CALL void GetDimensions(uint32_t* outDim) const { *outDim = uint32_t(sizeInBytes); }
    SLANG_CUDA_CALL uint32_t Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        return data[index >> 2];
    }
    SLANG_CUDA_CALL uint2 Load2(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint2{data[dataIdx], data[dataIdx + 1]};
    }
    SLANG_CUDA_CALL uint3 Load3(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint3{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2]};
    }
    SLANG_CUDA_CALL uint4 Load4(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint4{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2], data[dataIdx + 3]};
    }
    template<typename T>
    SLANG_CUDA_CALL T Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        T data;
        memcpy(&data, ((const char*)this->data) + index, sizeof(T));
        return data;
    }
    template<typename T>
    SLANG_CUDA_CALL StructuredBuffer<T> asStructuredBuffer() const
    {
        StructuredBuffer<T> rs;
        rs.data = (T*)data;
        rs.count = sizeInBytes / sizeof(T);
        return rs;
    }
    const uint32_t* data;
    size_t sizeInBytes; //< Must be multiple of 4
};

// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-rwbyteaddressbuffer
// Atomic operations support

// Signed 64-bit atomic wrappers
// CUDA only supports unsigned long long atomics, so we cast signed to unsigned
// Use longlong type with explicit unsigned long long casts for platform portability
__device__ __forceinline__ longlong atomicExch(longlong* address, longlong val)
{
    return (longlong)atomicExch((unsigned long long*)address, (unsigned long long)val);
}

__device__ __forceinline__ longlong atomicCAS(longlong* address, longlong compare, longlong val)
{
    return (longlong)atomicCAS(
        (unsigned long long*)address,
        (unsigned long long)compare,
        (unsigned long long)val);
}

__device__ __forceinline__ longlong atomicAdd(longlong* address, longlong val)
{
    return (longlong)atomicAdd((unsigned long long*)address, (unsigned long long)val);
}

// Float bitwise atomic compare-and-swap
// Uses integer atomics to preserve exact float bit patterns
__device__ __forceinline__ float atomicCAS(float* address, float compare, float val)
{
    int* addr_as_int = (int*)address;
    int old = atomicCAS(addr_as_int, __float_as_int(compare), __float_as_int(val));
    return __int_as_float(old);
}

// =====================================================================
// Atomic Reduction Operations (PTX `red` instruction)
// These are in-place atomic operations that don't return the old value.
// They are faster than the corresponding atomic operations that return values
// because they use the PTX `red` instruction with relaxed memory ordering.
//
// Supported operations based on PTX ISA:
// - add: .s32, .u32, .u64, .s64, .f16, .f16x2, .bf16, .bf16x2, .f32, .f64
// - min/max: .s32, .u32, .s64, .u64, .f32, .f64, .f16, .f16x2
// - and/or/xor: .b32, .b64
// - inc/dec: .u32
// =====================================================================

// Atomic reduction ADD operations
__device__ __forceinline__ void __slang_atomic_reduce_add(int32_t* addr, int32_t val, int order)
{
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;" : : "l"(addr), "r"(val) : "memory");
}

__device__ __forceinline__ void __slang_atomic_reduce_add(uint32_t* addr, uint32_t val, int order)
{
    asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" : : "l"(addr), "r"(val) : "memory");
}

__device__ __forceinline__ void __slang_atomic_reduce_add(int64_t* addr, int64_t val, int order)
{
    asm volatile("red.relaxed.gpu.global.add.s64 [%0], %1;" : : "l"(addr), "l"(val) : "memory");
}

__device__ __forceinline__ void __slang_atomic_reduce_add(uint64_t* addr, uint64_t val, int order)
{
    asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" : : "l"(addr), "l"(val) : "memory");
}

__device__ __forceinline__ void __slang_atomic_reduce_add(float* addr, float val, int order)
{
    asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;" : : "l"(addr), "f"(val) : "memory");
}

__device__ __forceinline__ void __slang_atomic_reduce_add(double* addr, double val, int order)
{
    asm volatile("red.relaxed.gpu.global.add.f64 [%0], %1;" : : "l"(addr), "d"(val) : "memory");
}

#if SLANG_CUDA_ENABLE_HALF
__device__ __forceinline__ void __slang_atomic_reduce_add(__half* addr, __half val, int order)
{
    unsigned short val_as_ushort = *reinterpret_cast<unsigned short*>(&val);
    asm volatile("red.relaxed.gpu.global.add.noftz.f16 [%0], %1;"
                 :
                 : "l"(addr), "h"(val_as_ushort)
                 : "memory");
}

__device__ __forceinline__ void __slang_atomic_reduce_add(__half2* addr, __half2 val, int order)
{
    unsigned int val_as_uint = *reinterpret_cast<unsigned int*>(&val);
    asm volatile("red.relaxed.gpu.global.add.noftz.f16x2 [%0], %1;"
                 :
                 : "l"(addr), "r"(val_as_uint)
                 : "memory");
}
#endif

#if SLANG_CUDA_ENABLE_BF16
__device__ __forceinline__ void __slang_atomic_reduce_add(
    __nv_bfloat16* addr,
    __nv_bfloat16 val,
    int order)
{
    unsigned short val_as_ushort = *reinterpret_cast<unsigned short*>(&val);
    asm volatile("red.relaxed.gpu.global.add.noftz.bf16 [%0], %1;"
                 :
                 : "l"(addr), "h"(val_as_ushort)
                 : "memory");
}

__device__ __forceinline__ void __slang_atomic_reduce_add(
    __nv_bfloat162* addr,
    __nv_bfloat162 val,
    int order)
{
    unsigned int val_as_uint = *reinterpret_cast<unsigned int*>(&val);
    asm volatile("red.relaxed.gpu.global.add.noftz.bf16x2 [%0], %1;"
                 :
                 : "l"(addr), "r"(val_as_uint)
                 : "memory");
}
#endif

// Atomic reduction MIN operations
__device__ __forceinline__ void __slang_atomic_reduce_min(int32_t* addr, int32_t val, int order)
{
    asm volatile("red.relaxed.gpu.global.min.s32 [%0], %1;" : : "l"(addr), "r"(val) : "memory");
}

__device__ __forceinline__ void __slang_atomic_reduce_min(uint32_t* addr, uint32_t val, int order)
{
    asm volatile("red.relaxed.gpu.global.min.u32 [%0], %1;" : : "l"(addr), "r"(val) : "memory");
}

__device__ __forceinline__ void __slang_atomic_reduce_min(int64_t* addr, int64_t val, int order)
{
    asm volatile("red.relaxed.gpu.global.min.s64 [%0], %1;" : : "l"(addr), "l"(val) : "memory");
}

__device__ __forceinline__ void __slang_atomic_reduce_min(uint64_t* addr, uint64_t val, int order)
{
    asm volatile("red.relaxed.gpu.global.min.u64 [%0], %1;" : : "l"(addr), "l"(val) : "memory");
}

// NOTE: PTX `red` instruction does NOT support min/max for floating-point types.
// Only integer types (.u32, .u64, .s32, .s64) are supported for min/max.
// For floating-point min/max atomics, use the regular `atom` instruction via
// __atomic_min/__atomic_max.

// Atomic reduction MAX operations
__device__ __forceinline__ void __slang_atomic_reduce_max(int32_t* addr, int32_t val, int order)
{
    asm volatile("red.relaxed.gpu.global.max.s32 [%0], %1;" : : "l"(addr), "r"(val) : "memory");
}

__device__ __forceinline__ void __slang_atomic_reduce_max(uint32_t* addr, uint32_t val, int order)
{
    asm volatile("red.relaxed.gpu.global.max.u32 [%0], %1;" : : "l"(addr), "r"(val) : "memory");
}

__device__ __forceinline__ void __slang_atomic_reduce_max(int64_t* addr, int64_t val, int order)
{
    asm volatile("red.relaxed.gpu.global.max.s64 [%0], %1;" : : "l"(addr), "l"(val) : "memory");
}

__device__ __forceinline__ void __slang_atomic_reduce_max(uint64_t* addr, uint64_t val, int order)
{
    asm volatile("red.relaxed.gpu.global.max.u64 [%0], %1;" : : "l"(addr), "l"(val) : "memory");
}

// NOTE: PTX `red` instruction does NOT support min/max for floating-point types.
// Only integer types (.u32, .u64, .s32, .s64) are supported for min/max.
// For floating-point min/max atomics, use the regular `atom` instruction via
// __atomic_min/__atomic_max.

// Atomic reduction AND operations (bitwise, integers only)
__device__ __forceinline__ void __slang_atomic_reduce_and(int32_t* addr, int32_t val, int order)
{
    asm volatile("red.relaxed.gpu.global.and.b32 [%0], %1;" : : "l"(addr), "r"(val) : "memory");
}

__device__ __forceinline__ void __slang_atomic_reduce_and(uint32_t* addr, uint32_t val, int order)
{
    asm volatile("red.relaxed.gpu.global.and.b32 [%0], %1;" : : "l"(addr), "r"(val) : "memory");
}

__device__ __forceinline__ void __slang_atomic_reduce_and(int64_t* addr, int64_t val, int order)
{
    asm volatile("red.relaxed.gpu.global.and.b64 [%0], %1;" : : "l"(addr), "l"(val) : "memory");
}

__device__ __forceinline__ void __slang_atomic_reduce_and(uint64_t* addr, uint64_t val, int order)
{
    asm volatile("red.relaxed.gpu.global.and.b64 [%0], %1;" : : "l"(addr), "l"(val) : "memory");
}

// Atomic reduction OR operations (bitwise, integers only)
__device__ __forceinline__ void __slang_atomic_reduce_or(int32_t* addr, int32_t val, int order)
{
    asm volatile("red.relaxed.gpu.global.or.b32 [%0], %1;" : : "l"(addr), "r"(val) : "memory");
}

__device__ __forceinline__ void __slang_atomic_reduce_or(uint32_t* addr, uint32_t val, int order)
{
    asm volatile("red.relaxed.gpu.global.or.b32 [%0], %1;" : : "l"(addr), "r"(val) : "memory");
}

__device__ __forceinline__ void __slang_atomic_reduce_or(int64_t* addr, int64_t val, int order)
{
    asm volatile("red.relaxed.gpu.global.or.b64 [%0], %1;" : : "l"(addr), "l"(val) : "memory");
}

__device__ __forceinline__ void __slang_atomic_reduce_or(uint64_t* addr, uint64_t val, int order)
{
    asm volatile("red.relaxed.gpu.global.or.b64 [%0], %1;" : : "l"(addr), "l"(val) : "memory");
}

// Atomic reduction XOR operations (bitwise, integers only)
__device__ __forceinline__ void __slang_atomic_reduce_xor(int32_t* addr, int32_t val, int order)
{
    asm volatile("red.relaxed.gpu.global.xor.b32 [%0], %1;" : : "l"(addr), "r"(val) : "memory");
}

__device__ __forceinline__ void __slang_atomic_reduce_xor(uint32_t* addr, uint32_t val, int order)
{
    asm volatile("red.relaxed.gpu.global.xor.b32 [%0], %1;" : : "l"(addr), "r"(val) : "memory");
}

__device__ __forceinline__ void __slang_atomic_reduce_xor(int64_t* addr, int64_t val, int order)
{
    asm volatile("red.relaxed.gpu.global.xor.b64 [%0], %1;" : : "l"(addr), "l"(val) : "memory");
}

__device__ __forceinline__ void __slang_atomic_reduce_xor(uint64_t* addr, uint64_t val, int order)
{
    asm volatile("red.relaxed.gpu.global.xor.b64 [%0], %1;" : : "l"(addr), "l"(val) : "memory");
}

// Atomic reduction INC/DEC operations (unsigned 32-bit only in PTX)
// Note: PTX inc/dec have specific semantics:
//   inc: d = (old >= b) ? 0 : old + 1
//   dec: d = ((old == 0) || (old > b)) ? b : old - 1
// For simple increment by 1, we use add instead
__device__ __forceinline__ void __slang_atomic_reduce_inc(uint32_t* addr, int order)
{
    asm volatile("red.relaxed.gpu.global.add.u32 [%0], 1;" : : "l"(addr) : "memory");
}

__device__ __forceinline__ void __slang_atomic_reduce_inc(int32_t* addr, int order)
{
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], 1;" : : "l"(addr) : "memory");
}

__device__ __forceinline__ void __slang_atomic_reduce_dec(uint32_t* addr, int order)
{
    asm volatile("red.relaxed.gpu.global.add.u32 [%0], -1;" : : "l"(addr) : "memory");
}

__device__ __forceinline__ void __slang_atomic_reduce_dec(int32_t* addr, int order)
{
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], -1;" : : "l"(addr) : "memory");
}

// =====================================================================
// End of Atomic Reduction Operations
// =====================================================================

// Missing support for Load with status
struct RWByteAddressBuffer
{
    SLANG_CUDA_CALL void GetDimensions(uint32_t* outDim) const { *outDim = uint32_t(sizeInBytes); }

    SLANG_CUDA_CALL uint32_t Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        return data[index >> 2];
    }
    SLANG_CUDA_CALL uint2 Load2(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint2{data[dataIdx], data[dataIdx + 1]};
    }
    SLANG_CUDA_CALL uint3 Load3(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint3{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2]};
    }
    SLANG_CUDA_CALL uint4 Load4(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint4{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2], data[dataIdx + 3]};
    }
    template<typename T>
    SLANG_CUDA_CALL T Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        T data;
        memcpy(&data, ((const char*)this->data) + index, sizeof(T));
        return data;
    }

    SLANG_CUDA_CALL void Store(size_t index, uint32_t v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        data[index >> 2] = v;
    }
    SLANG_CUDA_CALL void Store2(size_t index, uint2 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
    }
    SLANG_CUDA_CALL void Store3(size_t index, uint3 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
        data[dataIdx + 2] = v.z;
    }
    SLANG_CUDA_CALL void Store4(size_t index, uint4 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
        data[dataIdx + 2] = v.z;
        data[dataIdx + 3] = v.w;
    }
    template<typename T>
    SLANG_CUDA_CALL void Store(size_t index, T const& value) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        memcpy((char*)data + index, &value, sizeof(T));
    }

    /// Can be used in the core module to gain access
    template<typename T>
    SLANG_CUDA_CALL T* _getPtrAt(size_t index)
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        return (T*)(((char*)data) + index);
    }
    template<typename T>
    SLANG_CUDA_CALL RWStructuredBuffer<T> asStructuredBuffer() const
    {
        RWStructuredBuffer<T> rs;
        rs.data = (T*)data;
        rs.count = sizeInBytes / sizeof(T);
        return rs;
    }
    uint32_t* data;
    size_t sizeInBytes; //< Must be multiple of 4
};


// ---------------------- Wave --------------------------------------

// TODO(JS): It appears that cuda does not have a simple way to get a lane index.
//
// Another approach could be...
// laneId = ((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x) &
// SLANG_CUDA_WARP_MASK If that is really true another way to do this, would be for code generator
// to add this function with the [numthreads] baked in.
//
// For now I'll just assume you have a launch that makes the following correct if the kernel uses
// WaveGetLaneIndex()
#ifndef SLANG_USE_ASM_LANE_ID
__forceinline__ __device__ uint32_t _getLaneId()
{
    // If the launch is (or I guess some multiple of the warp size)
    // we try this mechanism, which is apparently faster.
    return threadIdx.x & SLANG_CUDA_WARP_MASK;
}
#else
__forceinline__ __device__ uint32_t _getLaneId()
{
    // https://stackoverflow.com/questions/44337309/whats-the-most-efficient-way-to-calculate-the-warp-id-lane-id-in-a-1-d-grid#
    // This mechanism is not the fastest way to do it, and that is why the other mechanism
    // is the default. But the other mechanism relies on a launch that makes the assumption
    // true.
    unsigned ret;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}
#endif

typedef int WarpMask;

// It appears that the __activemask() cannot always be used because
// threads need to be converged.
//
// For CUDA the article claims mask has to be used carefully
// https://devblogs.nvidia.com/using-cuda-warp-level-primitives/
// With the Warp intrinsics there is no mask, and it's just the 'active lanes'.
// __activemask() though does not require there is convergence, so that doesn't work.
//
// '__ballot_sync' produces a convergance.
//
// From the CUDA docs:
// ```For __all_sync, __any_sync, and __ballot_sync, a mask must be passed that specifies the
// threads participating in the call. A bit, representing the thread's lane ID, must be set for each
// participating thread to ensure they are properly converged before the intrinsic is executed by
// the hardware. All active threads named in mask must execute the same intrinsic with the same
// mask, or the result is undefined.```
//
// Currently there isn't a mechanism to correctly get the mask without it being passed through.
// Doing so will most likely require some changes to slang code generation to track masks, for now
// then we use _getActiveMask.

// Return mask of all the lanes less than the current lane
__forceinline__ __device__ WarpMask _getLaneLtMask()
{
    return (int(1) << _getLaneId()) - 1;
}

// TODO(JS):
// THIS IS NOT CORRECT! That determining the appropriate active mask requires appropriate
// mask tracking.
__forceinline__ __device__ WarpMask _getActiveMask()
{
    return __ballot_sync(__activemask(), true);
}

// Return a mask suitable for the 'MultiPrefix' style functions
__forceinline__ __device__ WarpMask _getMultiPrefixMask(int mask)
{
    return mask;
}

// Note! Note will return true if mask is 0, but thats okay, because there must be one
// lane active to execute anything
__inline__ __device__ bool _waveIsSingleLane(WarpMask mask)
{
    return (mask & (mask - 1)) == 0;
}

// Returns the power of 2 size of run of set bits. Returns 0 if not a suitable run.
// Examples:
// 0b00000000'00000000'00000000'11111111 -> 8
// 0b11111111'11111111'11111111'11111111 -> 32
// 0b00000000'00000000'00000000'00011111 -> 0 (since 5 is not a power of 2)
// 0b00000000'00000000'00000000'11110000 -> 0 (since the run of bits does not start at the LSB)
// 0b00000000'00000000'00000000'00100111 -> 0 (since it is not a single contiguous run)
__inline__ __device__ int _waveCalcPow2Offset(WarpMask mask)
{
    // This should be the most common case, so fast path it
    if (mask == SLANG_CUDA_WARP_BITMASK)
    {
        return SLANG_CUDA_WARP_SIZE;
    }
    // Is it a contiguous run of bits?
    if ((mask & (mask + 1)) == 0)
    {
        // const int offsetSize = __ffs(mask + 1) - 1;
        const int offset = 32 - __clz(mask);
        // Is it a power of 2 size
        if ((offset & (offset - 1)) == 0)
        {
            return offset;
        }
    }
    return 0;
}

__inline__ __device__ bool _waveIsFirstLane()
{
    const WarpMask mask = __activemask();
    // We special case bit 0, as that most warps are expected to be fully active.

    // mask & -mask, isolates the lowest set bit.
    // return (mask & 1 ) || ((mask & -mask) == (1 << _getLaneId()));

    // This mechanism is most similar to what was in an nVidia post, so assume it is prefered.
    return (mask & 1) || ((__ffs(mask) - 1) == _getLaneId());
}

template<typename T>
struct WaveOpOr
{
    __inline__ __device__ static T getInitial(T a) { return 0; }
    __inline__ __device__ static T doOp(T a, T b) { return a | b; }
};

template<typename T>
struct WaveOpAnd
{
    __inline__ __device__ static T getInitial(T a) { return ~T(0); }
    __inline__ __device__ static T doOp(T a, T b) { return a & b; }
};

template<typename T>
struct WaveOpXor
{
    __inline__ __device__ static T getInitial(T a) { return 0; }
    __inline__ __device__ static T doOp(T a, T b) { return a ^ b; }
    __inline__ __device__ static T doInverse(T a, T b) { return a ^ b; }
};

template<typename T>
struct WaveOpAdd
{
    __inline__ __device__ static T getInitial(T a) { return 0; }
    __inline__ __device__ static T doOp(T a, T b) { return a + b; }
    __inline__ __device__ static T doInverse(T a, T b) { return a - b; }
};

template<typename T>
struct WaveOpMul
{
    __inline__ __device__ static T getInitial(T a) { return T(1); }
    __inline__ __device__ static T doOp(T a, T b) { return a * b; }
    // Using this inverse for int is probably undesirable - because in general it requires T to have
    // more precision There is also a performance aspect to it, where divides are generally
    // significantly slower
    __inline__ __device__ static T doInverse(T a, T b) { return a / b; }
};

template<typename T>
struct WaveOpMax
{
    __inline__ __device__ static T getInitial(T a, bool exclusive = false);
    __inline__ __device__ static T doOp(T a, T b) { return a > b ? a : b; }
};

template<typename T>
struct WaveOpMin
{
    __inline__ __device__ static T getInitial(T a, bool exclusive = false);
    __inline__ __device__ static T doOp(T a, T b) { return a < b ? a : b; }
};

// Compact specializations using macro for getInitial
#define SLANG_WAVE_MIN_SPEC(T, EXCL_VAL)                                  \
    template<>                                                            \
    __inline__ __device__ T WaveOpMin<T>::getInitial(T a, bool exclusive) \
    {                                                                     \
        return exclusive ? (EXCL_VAL) : a;                                \
    }

#define SLANG_WAVE_MAX_SPEC(T, EXCL_VAL)                                  \
    template<>                                                            \
    __inline__ __device__ T WaveOpMax<T>::getInitial(T a, bool exclusive) \
    {                                                                     \
        return exclusive ? (EXCL_VAL) : a;                                \
    }

// Min specializations (exclusive identity = max value)
SLANG_WAVE_MIN_SPEC(float, SLANG_INFINITY)
SLANG_WAVE_MIN_SPEC(double, SLANG_INFINITY)
SLANG_WAVE_MIN_SPEC(int, 0x7FFFFFFF)
SLANG_WAVE_MIN_SPEC(uint, 0xFFFFFFFF)
SLANG_WAVE_MIN_SPEC(char, (char)0x7F)
SLANG_WAVE_MIN_SPEC(int8_t, (int8_t)0x7F)
SLANG_WAVE_MIN_SPEC(uint8_t, (uint8_t)0xFF)
SLANG_WAVE_MIN_SPEC(int16_t, (int16_t)0x7FFF)
SLANG_WAVE_MIN_SPEC(uint16_t, (uint16_t)0xFFFF)
SLANG_WAVE_MIN_SPEC(int64_t, 0x7FFFFFFFFFFFFFFFLL)
SLANG_WAVE_MIN_SPEC(uint64_t, 0xFFFFFFFFFFFFFFFFULL)
#if SLANG_CUDA_ENABLE_HALF
SLANG_WAVE_MIN_SPEC(__half, __ushort_as_half(0x7BFF))
#endif

// Max specializations (exclusive identity = min value)
SLANG_WAVE_MAX_SPEC(float, -SLANG_INFINITY)
SLANG_WAVE_MAX_SPEC(double, -SLANG_INFINITY)
SLANG_WAVE_MAX_SPEC(int, (int)0x80000000)
SLANG_WAVE_MAX_SPEC(uint, 0)
SLANG_WAVE_MAX_SPEC(char, (char)0x80)
SLANG_WAVE_MAX_SPEC(int8_t, (int8_t)0x80)
SLANG_WAVE_MAX_SPEC(uint8_t, 0)
SLANG_WAVE_MAX_SPEC(int16_t, (int16_t)0x8000)
SLANG_WAVE_MAX_SPEC(uint16_t, 0)
SLANG_WAVE_MAX_SPEC(int64_t, (int64_t)0x8000000000000000LL)
SLANG_WAVE_MAX_SPEC(uint64_t, 0)
#if SLANG_CUDA_ENABLE_HALF
SLANG_WAVE_MAX_SPEC(__half, __ushort_as_half(0xFBFF))
#endif

#undef SLANG_WAVE_MIN_SPEC
#undef SLANG_WAVE_MAX_SPEC

template<typename T>
struct ElementTypeTrait;

// Scalar
template<>
struct ElementTypeTrait<int>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<uint>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<float>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<double>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<uint64_t>
{
    typedef uint64_t Type;
};
template<>
struct ElementTypeTrait<int64_t>
{
    typedef int64_t Type;
};
template<>
struct ElementTypeTrait<char>
{
    typedef char Type;
};
template<>
struct ElementTypeTrait<uchar>
{
    typedef uchar Type;
};
template<>
struct ElementTypeTrait<short>
{
    typedef short Type;
};
template<>
struct ElementTypeTrait<ushort>
{
    typedef ushort Type;
};
#if SLANG_CUDA_ENABLE_HALF
template<>
struct ElementTypeTrait<__half>
{
    typedef __half Type;
};
#endif

// Vector
template<>
struct ElementTypeTrait<int1>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<int2>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<int3>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<int4>
{
    typedef int Type;
};

template<>
struct ElementTypeTrait<uint1>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<uint2>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<uint3>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<uint4>
{
    typedef uint Type;
};

template<>
struct ElementTypeTrait<float1>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<float2>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<float3>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<float4>
{
    typedef float Type;
};

template<>
struct ElementTypeTrait<double1>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<double2>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<double3>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<double4>
{
    typedef double Type;
};

// Additional vector types
template<>
struct ElementTypeTrait<char2>
{
    typedef char Type;
};
template<>
struct ElementTypeTrait<char3>
{
    typedef char Type;
};
template<>
struct ElementTypeTrait<char4>
{
    typedef char Type;
};
template<>
struct ElementTypeTrait<uchar2>
{
    typedef uchar Type;
};
template<>
struct ElementTypeTrait<uchar3>
{
    typedef uchar Type;
};
template<>
struct ElementTypeTrait<uchar4>
{
    typedef uchar Type;
};
template<>
struct ElementTypeTrait<short2>
{
    typedef short Type;
};
template<>
struct ElementTypeTrait<short3>
{
    typedef short Type;
};
template<>
struct ElementTypeTrait<short4>
{
    typedef short Type;
};
template<>
struct ElementTypeTrait<ushort2>
{
    typedef ushort Type;
};
template<>
struct ElementTypeTrait<ushort3>
{
    typedef ushort Type;
};
template<>
struct ElementTypeTrait<ushort4>
{
    typedef ushort Type;
};
template<>
struct ElementTypeTrait<longlong2>
{
    typedef int64_t Type;
};
template<>
struct ElementTypeTrait<longlong3>
{
    typedef int64_t Type;
};
template<>
struct ElementTypeTrait<longlong4>
{
    typedef int64_t Type;
};
template<>
struct ElementTypeTrait<ulonglong2>
{
    typedef uint64_t Type;
};
template<>
struct ElementTypeTrait<ulonglong3>
{
    typedef uint64_t Type;
};
template<>
struct ElementTypeTrait<ulonglong4>
{
    typedef uint64_t Type;
};
#if SLANG_CUDA_ENABLE_HALF
template<>
struct ElementTypeTrait<__half2>
{
    typedef __half Type;
};
template<>
struct ElementTypeTrait<__half3>
{
    typedef __half Type;
};
template<>
struct ElementTypeTrait<__half4>
{
    typedef __half Type;
};
#endif

// Matrix
template<typename T, int ROWS, int COLS>
struct ElementTypeTrait<Matrix<T, ROWS, COLS>>
{
    typedef T Type;
};

// Scalar
template<typename INTF, typename T>
__device__ T _waveReduceScalar(WarpMask mask, T val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);
    if (offsetSize > 0)
    {
        // Fast path O(log2(activeLanes))
        for (int offset = offsetSize >> 1; offset > 0; offset >>= 1)
        {
            val = INTF::doOp(val, __shfl_xor_sync(mask, val, offset));
        }
    }
    else if (!_waveIsSingleLane(mask))
    {
        T result = INTF::getInitial(val);
        int remaining = mask;
        while (remaining)
        {
            const int laneBit = remaining & -remaining;
            // Get the sourceLane
            const int srcLane = __ffs(laneBit) - 1;
            // Broadcast (can also broadcast to self)
            result = INTF::doOp(result, __shfl_sync(mask, val, srcLane));
            remaining &= ~laneBit;
        }
        return result;
    }
    return val;
}


// Multiple values
template<typename INTF, typename T, size_t COUNT>
__device__ void _waveReduceMultiple(WarpMask mask, T* val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);
    if (offsetSize > 0)
    {
        // Fast path O(log2(activeLanes))
        for (int offset = offsetSize >> 1; offset > 0; offset >>= 1)
        {
            for (size_t i = 0; i < COUNT; ++i)
            {
                val[i] = INTF::doOp(val[i], __shfl_xor_sync(mask, val[i], offset));
            }
        }
    }
    else if (!_waveIsSingleLane(mask))
    {
        // Copy the original
        T originalVal[COUNT];
        for (size_t i = 0; i < COUNT; ++i)
        {
            const T v = val[i];
            originalVal[i] = v;
            val[i] = INTF::getInitial(v);
        }

        int remaining = mask;
        while (remaining)
        {
            const int laneBit = remaining & -remaining;
            // Get the sourceLane
            const int srcLane = __ffs(laneBit) - 1;
            // Broadcast (can also broadcast to self)
            for (size_t i = 0; i < COUNT; ++i)
            {
                val[i] = INTF::doOp(val[i], __shfl_sync(mask, originalVal[i], srcLane));
            }
            remaining &= ~laneBit;
        }
    }
}

template<typename INTF, typename T>
__device__ void _waveReduceMultiple(WarpMask mask, T* val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<INTF, ElemType, sizeof(T) / sizeof(ElemType)>(mask, (ElemType*)val);
}

template<typename T>
__inline__ __device__ T _waveOr(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpOr<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveAnd(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpAnd<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveXor(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpXor<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveProduct(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpMul<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveSum(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpAdd<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveMin(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpMin<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveMax(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpMax<T>, T>(mask, val);
}

// Fast-path specializations when CUDA warp reduce operators are available
#if __CUDA_ARCH__ >= 800 // 8.x or higher
template<>
__inline__ __device__ unsigned _waveOr<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_or_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveAnd<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_and_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveXor<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_xor_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveSum<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_add_sync(mask, val);
}

template<>
__inline__ __device__ int _waveSum<int>(WarpMask mask, int val)
{
    return __reduce_add_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveMin<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_min_sync(mask, val);
}

template<>
__inline__ __device__ int _waveMin<int>(WarpMask mask, int val)
{
    return __reduce_min_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveMax<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_max_sync(mask, val);
}

template<>
__inline__ __device__ int _waveMax<int>(WarpMask mask, int val)
{
    return __reduce_max_sync(mask, val);
}
#endif


// Multiple

template<typename T>
__inline__ __device__ T _waveOrMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpOr<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveAndMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpAnd<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveXorMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpXor<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveProductMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpMul<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveSumMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpAdd<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveMinMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpMin<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveMaxMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpMax<ElemType>>(mask, &val);
    return val;
}


template<typename T>
__inline__ __device__ bool _waveAllEqual(WarpMask mask, T val)
{
    int pred;
    __match_all_sync(mask, val, &pred);
    return pred != 0;
}

template<typename T>
__inline__ __device__ bool _waveAllEqualMultiple(WarpMask mask, T inVal)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    int pred;
    const ElemType* src = (const ElemType*)&inVal;
    for (size_t i = 0; i < count; ++i)
    {
        __match_all_sync(mask, src[i], &pred);
        if (pred == 0)
        {
            return false;
        }
    }
    return true;
}

template<typename T>
__inline__ __device__ T _waveReadFirst(WarpMask mask, T val)
{
    const int lowestLaneId = __ffs(mask) - 1;
    return __shfl_sync(mask, val, lowestLaneId);
}

template<typename T>
__inline__ __device__ T _waveReadFirstMultiple(WarpMask mask, T inVal)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    T outVal;
    const ElemType* src = (const ElemType*)&inVal;
    ElemType* dst = (ElemType*)&outVal;
    const int lowestLaneId = __ffs(mask) - 1;
    for (size_t i = 0; i < count; ++i)
    {
        dst[i] = __shfl_sync(mask, src[i], lowestLaneId);
    }
    return outVal;
}

template<typename T>
__inline__ __device__ T _waveShuffleMultiple(WarpMask mask, T inVal, int lane)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    T outVal;
    const ElemType* src = (const ElemType*)&inVal;
    ElemType* dst = (ElemType*)&outVal;
    for (size_t i = 0; i < count; ++i)
    {
        dst[i] = __shfl_sync(mask, src[i], lane);
    }
    return outVal;
}

// Scalar

// Invertable means that when we get to the end of the reduce, we can remove val (to make
// exclusive), using the inverse of the op.
template<typename INTF, typename T>
__device__ T _wavePrefixInvertableScalar(WarpMask mask, T val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();
    T result;
    if (offsetSize > 0)
    {
        // Sum is calculated inclusive of this lanes value
        result = val;
        for (int i = 1; i < offsetSize; i += i)
        {
            const T readVal = __shfl_up_sync(mask, result, i, offsetSize);
            if (laneId >= i)
            {
                result = INTF::doOp(result, readVal);
            }
        }
        // Remove val from the result, by applyin inverse
        result = INTF::doInverse(result, val);
    }
    else
    {
        result = INTF::getInitial(val);
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;
                // Broadcast (can also broadcast to self)
                const T readValue = __shfl_sync(mask, val, srcLane);
                // Only accumulate if srcLane is less than this lane
                if (srcLane < laneId)
                {
                    result = INTF::doOp(result, readValue);
                }
                remaining &= ~laneBit;
            }
        }
    }
    return result;
}


// This implementation separately tracks the value to be propogated, and the value
// that is the final result
template<typename INTF, typename T>
__device__ T _wavePrefixScalar(WarpMask mask, T val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();
    T result = INTF::getInitial(val);
    if (offsetSize > 0)
    {
        // For transmitted value we will do it inclusively with this lanes value
        // For the result we do not include the lanes value. This means an extra multiply for each
        // iteration but means we don't need to have a divide at the end and also removes overflow
        // issues in that scenario.
        for (int i = 1; i < offsetSize; i += i)
        {
            const T readVal = __shfl_up_sync(mask, val, i, offsetSize);
            if (laneId >= i)
            {
                result = INTF::doOp(result, readVal);
                val = INTF::doOp(val, readVal);
            }
        }
    }
    else
    {
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;
                // Broadcast (can also broadcast to self)
                const T readValue = __shfl_sync(mask, val, srcLane);
                // Only accumulate if srcLane is less than this lane
                if (srcLane < laneId)
                {
                    result = INTF::doOp(result, readValue);
                }
                remaining &= ~laneBit;
            }
        }
    }
    return result;
}


template<typename INTF, typename T, size_t COUNT>
__device__ T _waveOpCopy(T* dst, const T* src)
{
    for (size_t j = 0; j < COUNT; ++j)
    {
        dst[j] = src[j];
    }
}


template<typename INTF, typename T, size_t COUNT>
__device__ T _waveOpDoInverse(T* inOut, const T* val)
{
    for (size_t j = 0; j < COUNT; ++j)
    {
        inOut[j] = INTF::doInverse(inOut[j], val[j]);
    }
}

template<typename INTF, typename T, size_t COUNT>
__device__ T _waveOpSetInitial(T* out, const T* val)
{
    for (size_t j = 0; j < COUNT; ++j)
    {
        out[j] = INTF::getInitial(val[j]);
    }
}

template<typename INTF, typename T, size_t COUNT>
__device__ T _wavePrefixInvertableMultiple(WarpMask mask, T* val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();
    T originalVal[COUNT];
    _waveOpCopy<INTF, T, COUNT>(originalVal, val);

    if (offsetSize > 0)
    {
        // Sum is calculated inclusive of this lanes value
        for (int i = 1; i < offsetSize; i += i)
        {
            // TODO(JS): Note that here I don't split the laneId outside so it's only tested once.
            // This may be better but it would also mean that there would be shfl between lanes
            // that are on different (albeit identical) instructions. So this seems more likely to
            // work as expected with everything in lock step.
            for (size_t j = 0; j < COUNT; ++j)
            {
                const T readVal = __shfl_up_sync(mask, val[j], i, offsetSize);
                if (laneId >= i)
                {
                    val[j] = INTF::doOp(val[j], readVal);
                }
            }
        }
        // Remove originalVal from the result, by applyin inverse
        _waveOpDoInverse<INTF, T, COUNT>(val, originalVal);
    }
    else
    {
        _waveOpSetInitial<INTF, T, COUNT>(val, val);
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;

                for (size_t j = 0; j < COUNT; ++j)
                {
                    // Broadcast (can also broadcast to self)
                    const T readValue = __shfl_sync(mask, originalVal[j], srcLane);
                    // Only accumulate if srcLane is less than this lane
                    if (srcLane < laneId)
                    {
                        val[j] = INTF::doOp(val[j], readValue);
                    }
                    remaining &= ~laneBit;
                }
            }
        }
    }
}

template<typename INTF, typename T, size_t COUNT>
__device__ T _wavePrefixMultiple(WarpMask mask, T* val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();

    T work[COUNT];
    _waveOpCopy<INTF, T, COUNT>(work, val);
    _waveOpSetInitial<INTF, T, COUNT>(val, val);

    if (offsetSize > 0)
    {
        // For transmitted value we will do it inclusively with this lanes value
        // For the result we do not include the lanes value. This means an extra op for each
        // iteration but means we don't need to have a divide at the end and also removes overflow
        // issues in that scenario.
        for (int i = 1; i < offsetSize; i += i)
        {
            for (size_t j = 0; j < COUNT; ++j)
            {
                const T readVal = __shfl_up_sync(mask, work[j], i, offsetSize);
                if (laneId >= i)
                {
                    work[j] = INTF::doOp(work[j], readVal);
                    val[j] = INTF::doOp(val[j], readVal);
                }
            }
        }
    }
    else
    {
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;

                for (size_t j = 0; j < COUNT; ++j)
                {
                    // Broadcast (can also broadcast to self)
                    const T readValue = __shfl_sync(mask, work[j], srcLane);
                    // Only accumulate if srcLane is less than this lane
                    if (srcLane < laneId)
                    {
                        val[j] = INTF::doOp(val[j], readValue);
                    }
                }
                remaining &= ~laneBit;
            }
        }
    }
}

template<typename T>
__inline__ __device__ T _wavePrefixProduct(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpMul<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixSum(WarpMask mask, T val)
{
    return _wavePrefixInvertableScalar<WaveOpAdd<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixXor(WarpMask mask, T val)
{
    return _wavePrefixInvertableScalar<WaveOpXor<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixOr(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpOr<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixAnd(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpAnd<T>, T>(mask, val);
}


template<typename T>
__inline__ __device__ T _wavePrefixProductMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixInvertableMultiple<WaveOpMul<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixSumMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixInvertableMultiple<WaveOpAdd<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixXorMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixInvertableMultiple<WaveOpXor<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixOrMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixMultiple<WaveOpOr<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixAndMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixMultiple<WaveOpAnd<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixMin(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpMin<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixMax(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpMax<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixMinMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixMultiple<WaveOpMin<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixMaxMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixMultiple<WaveOpMax<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

// Wrapper structures for exclusive operations that use the overloaded getInitial method
template<typename T>
struct WaveOpExclusiveMin
{
    __inline__ __device__ static T getInitial(T a) { return WaveOpMin<T>::getInitial(a, true); }
    __inline__ __device__ static T doOp(T a, T b) { return WaveOpMin<T>::doOp(a, b); }
};

template<typename T>
struct WaveOpExclusiveMax
{
    __inline__ __device__ static T getInitial(T a) { return WaveOpMax<T>::getInitial(a, true); }
    __inline__ __device__ static T doOp(T a, T b) { return WaveOpMax<T>::doOp(a, b); }
};

// Inclusive prefix min/max functions (for WaveMultiPrefixInclusive*)
template<typename T>
__inline__ __device__ T _wavePrefixInclusiveMin(WarpMask mask, T val)
{
    return _wavePrefixMin(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixInclusiveMax(WarpMask mask, T val)
{
    return _wavePrefixMax(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixInclusiveMinMultiple(WarpMask mask, T val)
{
    return _wavePrefixMinMultiple(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixInclusiveMaxMultiple(WarpMask mask, T val)
{
    return _wavePrefixMaxMultiple(mask, val);
}

// Explicit exclusive prefix min/max functions (for WaveMultiPrefixExclusive*)
template<typename T>
__inline__ __device__ T _wavePrefixExclusiveMin(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpExclusiveMin<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixExclusiveMax(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpExclusiveMax<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixExclusiveMinMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixMultiple<WaveOpExclusiveMin<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixExclusiveMaxMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixMultiple<WaveOpExclusiveMax<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ uint4 _waveMatchScalar(WarpMask mask, T val)
{
    int pred;
    return make_uint4(__match_all_sync(mask, val, &pred), 0, 0, 0);
}

template<typename T>
__inline__ __device__ uint4 _waveMatchMultiple(WarpMask mask, const T& inVal)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    int pred;
    const ElemType* src = (const ElemType*)&inVal;
    uint matchBits = 0xffffffff;
    for (size_t i = 0; i < count && matchBits; ++i)
    {
        matchBits = matchBits & __match_all_sync(mask, src[i], &pred);
    }
    return make_uint4(matchBits, 0, 0, 0);
}

__inline__ __device__ uint getAt(dim3 a, int b)
{
    SLANG_PRELUDE_ASSERT(b >= 0 && b < 3);
    return (&a.x)[b];
}
__inline__ __device__ uint3 operator*(uint3 a, dim3 b)
{
    uint3 r;
    r.x = a.x * b.x;
    r.y = a.y * b.y;
    r.z = a.z * b.z;
    return r;
}

template<typename TResult, typename TInput>
__inline__ __device__ TResult slang_bit_cast(TInput val)
{
    return *(TResult*)(&val);
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */


/* Type that defines the uniform entry point params. The actual content of this type is dependent on
the entry point parameters, and can be found via reflection or defined such that it matches the
shader appropriately.
*/
struct UniformEntryPointParams;
struct UniformState;

// ---------------------- OptiX Ray Payload --------------------------------------
#ifdef SLANG_CUDA_ENABLE_OPTIX

struct RayDesc
{
    float3 Origin;
    float TMin;
    float3 Direction;
    float TMax;
};

static __forceinline__ __device__ void* unpackOptiXRayPayloadPointer(uint32_t i0, uint32_t i1)
{
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

static __forceinline__ __device__ void packOptiXRayPayloadPointer(
    void* ptr,
    uint32_t& i0,
    uint32_t& i1)
{
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ void* getOptiXRayPayloadPtr()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return unpackOptiXRayPayloadPointer(u0, u1);
}

// Maximum number of 32-bit registers for OptiX payload (32 registers = 128 bytes)
static constexpr size_t kMaxOptiXPayloadRegisters = 32;

// Helper to pack/unpack payload to/from registers for small payloads (<= 128 bytes)
template<typename T, size_t N = (sizeof(T) + 3) / 4>
struct PayloadRegisters
{
    uint32_t regs[N > 0 ? N : 1];

    __forceinline__ __device__ void pack(const T& payload) { memcpy(regs, &payload, sizeof(T)); }

    __forceinline__ __device__ void unpack(T& payload) { memcpy(&payload, regs, sizeof(T)); }
};

// Internal helper to call optixTrace with the right number of register arguments
template<typename T, size_t N = (sizeof(T) + 3) / 4>
__forceinline__ __device__ void optixTraceWithRegs(
    OptixTraversableHandle AccelerationStructure,
    float3 Origin,
    float3 Direction,
    float TMin,
    float TMax,
    float Time,
    uint32_t InstanceInclusionMask,
    uint32_t RayFlags,
    uint32_t RayContributionToHitGroupIndex,
    uint32_t MultiplierForGeometryContributionToHitGroupIndex,
    uint32_t MissShaderIndex,
    PayloadRegisters<T, N>& pr)
{
    // Call optixTrace with the appropriate number of payload registers
    if constexpr (N == 0)
    {
        optixTrace(
            AccelerationStructure,
            Origin,
            Direction,
            TMin,
            TMax,
            Time,
            InstanceInclusionMask,
            RayFlags,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex);
    }
    else if constexpr (N == 1)
    {
        optixTrace(
            AccelerationStructure,
            Origin,
            Direction,
            TMin,
            TMax,
            Time,
            InstanceInclusionMask,
            RayFlags,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex,
            pr.regs[0]);
    }
    else if constexpr (N == 2)
    {
        optixTrace(
            AccelerationStructure,
            Origin,
            Direction,
            TMin,
            TMax,
            Time,
            InstanceInclusionMask,
            RayFlags,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex,
            pr.regs[0],
            pr.regs[1]);
    }
    else if constexpr (N == 3)
    {
        optixTrace(
            AccelerationStructure,
            Origin,
            Direction,
            TMin,
            TMax,
            Time,
            InstanceInclusionMask,
            RayFlags,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex,
            pr.regs[0],
            pr.regs[1],
            pr.regs[2]);
    }
    else if constexpr (N == 4)
    {
        optixTrace(
            AccelerationStructure,
            Origin,
            Direction,
            TMin,
            TMax,
            Time,
            InstanceInclusionMask,
            RayFlags,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex,
            pr.regs[0],
            pr.regs[1],
            pr.regs[2],
            pr.regs[3]);
    }
    else if constexpr (N == 5)
    {
        optixTrace(
            AccelerationStructure,
            Origin,
            Direction,
            TMin,
            TMax,
            Time,
            InstanceInclusionMask,
            RayFlags,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex,
            pr.regs[0],
            pr.regs[1],
            pr.regs[2],
            pr.regs[3],
            pr.regs[4]);
    }
    else if constexpr (N == 6)
    {
        optixTrace(
            AccelerationStructure,
            Origin,
            Direction,
            TMin,
            TMax,
            Time,
            InstanceInclusionMask,
            RayFlags,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex,
            pr.regs[0],
            pr.regs[1],
            pr.regs[2],
            pr.regs[3],
            pr.regs[4],
            pr.regs[5]);
    }
    else if constexpr (N == 7)
    {
        optixTrace(
            AccelerationStructure,
            Origin,
            Direction,
            TMin,
            TMax,
            Time,
            InstanceInclusionMask,
            RayFlags,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex,
            pr.regs[0],
            pr.regs[1],
            pr.regs[2],
            pr.regs[3],
            pr.regs[4],
            pr.regs[5],
            pr.regs[6]);
    }
    else if constexpr (N == 8)
    {
        optixTrace(
            AccelerationStructure,
            Origin,
            Direction,
            TMin,
            TMax,
            Time,
            InstanceInclusionMask,
            RayFlags,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex,
            pr.regs[0],
            pr.regs[1],
            pr.regs[2],
            pr.regs[3],
            pr.regs[4],
            pr.regs[5],
            pr.regs[6],
            pr.regs[7]);
    }
    else if constexpr (N <= 16)
    {
        optixTrace(
            AccelerationStructure,
            Origin,
            Direction,
            TMin,
            TMax,
            Time,
            InstanceInclusionMask,
            RayFlags,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex,
            pr.regs[0],
            pr.regs[1],
            pr.regs[2],
            pr.regs[3],
            pr.regs[4],
            pr.regs[5],
            pr.regs[6],
            pr.regs[7],
            pr.regs[8],
            pr.regs[9],
            pr.regs[10],
            pr.regs[11],
            pr.regs[12],
            pr.regs[13],
            pr.regs[14],
            pr.regs[15]);
    }
    else if constexpr (N <= kMaxOptiXPayloadRegisters)
    {
        optixTrace(
            AccelerationStructure,
            Origin,
            Direction,
            TMin,
            TMax,
            Time,
            InstanceInclusionMask,
            RayFlags,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex,
            pr.regs[0],
            pr.regs[1],
            pr.regs[2],
            pr.regs[3],
            pr.regs[4],
            pr.regs[5],
            pr.regs[6],
            pr.regs[7],
            pr.regs[8],
            pr.regs[9],
            pr.regs[10],
            pr.regs[11],
            pr.regs[12],
            pr.regs[13],
            pr.regs[14],
            pr.regs[15],
            pr.regs[16],
            pr.regs[17],
            pr.regs[18],
            pr.regs[19],
            pr.regs[20],
            pr.regs[21],
            pr.regs[22],
            pr.regs[23],
            pr.regs[24],
            pr.regs[25],
            pr.regs[26],
            pr.regs[27],
            pr.regs[28],
            pr.regs[29],
            pr.regs[30],
            pr.regs[31]);
    }
}

template<typename T>
__forceinline__ __device__ void optixTrace(
    OptixTraversableHandle AccelerationStructure,
    uint32_t RayFlags,
    uint32_t InstanceInclusionMask,
    uint32_t RayContributionToHitGroupIndex,
    uint32_t MultiplierForGeometryContributionToHitGroupIndex,
    uint32_t MissShaderIndex,
    RayDesc Ray,
    T* Payload)
{
    constexpr size_t numRegs = (sizeof(T) + 3) / 4;

    if constexpr (numRegs <= kMaxOptiXPayloadRegisters)
    {
        // Register-based approach for small payloads
        PayloadRegisters<T> pr;
        pr.pack(*Payload);

        optixTraceWithRegs<T>(
            AccelerationStructure,
            Ray.Origin,
            Ray.Direction,
            Ray.TMin,
            Ray.TMax,
            0.f, /* Time for motion blur */
            InstanceInclusionMask,
            RayFlags,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex,
            pr);

        // Read back updated payload registers
        // Native optixTrace updates regs in place
        pr.unpack(*Payload);
    }
    else
    {
        // Pointer-based fallback for large payloads
        uint32_t r0, r1;
        packOptiXRayPayloadPointer((void*)Payload, r0, r1);
        optixTrace(
            AccelerationStructure,
            Ray.Origin,
            Ray.Direction,
            Ray.TMin,
            Ray.TMax,
            0.f,
            InstanceInclusionMask,
            RayFlags,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex,
            r0,
            r1);
    }
}

// Non-template overload for empty payload case.
// When Slang's type legalization eliminates an empty payload struct,
// the generated code calls optixTrace without a payload argument.
__forceinline__ __device__ void optixTrace(
    OptixTraversableHandle AccelerationStructure,
    uint32_t RayFlags,
    uint32_t InstanceInclusionMask,
    uint32_t RayContributionToHitGroupIndex,
    uint32_t MultiplierForGeometryContributionToHitGroupIndex,
    uint32_t MissShaderIndex,
    RayDesc Ray)
{
    optixTrace(
        AccelerationStructure,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        Ray.TMax,
        0.f,
        InstanceInclusionMask,
        RayFlags,
        RayContributionToHitGroupIndex,
        MultiplierForGeometryContributionToHitGroupIndex,
        MissShaderIndex);
}

#if (OPTIX_VERSION >= 90000)
__forceinline__ __device__ float4 optixGetSpherePositionAndRadius()
{
    float4 data[1];
    optixGetSphereData(data);
    return data[0];
}
#endif

#if (OPTIX_VERSION >= 90000)
__forceinline__ __device__ float4
optixHitObjectGetSpherePositionAndRadius(OptixTraversableHandle* Obj)
{
    float4 data[1];
    optixHitObjectGetSphereData(data);
    return data[0];
}
#endif

#if (OPTIX_VERSION >= 90000)
__forceinline__ __device__ Matrix<float, 2, 4> optixGetLssPositionsAndRadii()
{
    float4 data[2];
    optixGetLinearCurveVertexData(data);
    return makeMatrix<float, 2, 4>(data[0], data[1]);
}
#endif

#if (OPTIX_VERSION >= 90000)
__forceinline__ __device__ Matrix<float, 2, 4> optixHitObjectGetLssPositionsAndRadii(
    OptixTraversableHandle* Obj)
{
    float4 data[2];
    optixHitObjectGetLinearCurveVertexData(data);
    return makeMatrix<float, 2, 4>(data[0], data[1]);
}
#endif

#if (OPTIX_VERSION >= 90000)
__forceinline__ __device__ bool optixIsSphereHit()
{
    return optixGetPrimitiveType() == OPTIX_PRIMITIVE_TYPE_SPHERE;
}
#endif

#if (OPTIX_VERSION >= 90000)
__forceinline__ __device__ bool optixHitObjectIsSphereHit(OptixTraversableHandle* Obj)
{
    return optixGetPrimitiveType(optixHitObjectGetHitKind()) == OPTIX_PRIMITIVE_TYPE_SPHERE;
}
#endif

#if (OPTIX_VERSION >= 90000)
__forceinline__ __device__ bool optixIsLSSHit()
{
    return optixGetPrimitiveType() == OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
}
#endif

#if (OPTIX_VERSION >= 90000)
__forceinline__ __device__ bool optixHitObjectIsLSSHit(OptixTraversableHandle* Obj)
{
    return optixGetPrimitiveType(optixHitObjectGetHitKind()) == OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
}
#endif

// Internal helper to call optixTraverse with the right number of register arguments
template<typename T, size_t N = (sizeof(T) + 3) / 4>
__forceinline__ __device__ void optixTraverseWithRegs(
    OptixTraversableHandle AccelerationStructure,
    float3 Origin,
    float3 Direction,
    float TMin,
    float TMax,
    float Time,
    uint32_t InstanceInclusionMask,
    uint32_t RayFlags,
    uint32_t RayContributionToHitGroupIndex,
    uint32_t MultiplierForGeometryContributionToHitGroupIndex,
    uint32_t MissShaderIndex,
    PayloadRegisters<T, N>& pr)
{
    // Call optixTraverse with the appropriate number of payload registers
    if constexpr (N == 0)
    {
        optixTraverse(
            AccelerationStructure,
            Origin,
            Direction,
            TMin,
            TMax,
            Time,
            InstanceInclusionMask,
            RayFlags,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex);
    }
    else if constexpr (N == 1)
    {
        optixTraverse(
            AccelerationStructure,
            Origin,
            Direction,
            TMin,
            TMax,
            Time,
            InstanceInclusionMask,
            RayFlags,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex,
            pr.regs[0]);
    }
    else if constexpr (N == 2)
    {
        optixTraverse(
            AccelerationStructure,
            Origin,
            Direction,
            TMin,
            TMax,
            Time,
            InstanceInclusionMask,
            RayFlags,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex,
            pr.regs[0],
            pr.regs[1]);
    }
    else if constexpr (N == 3)
    {
        optixTraverse(
            AccelerationStructure,
            Origin,
            Direction,
            TMin,
            TMax,
            Time,
            InstanceInclusionMask,
            RayFlags,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex,
            pr.regs[0],
            pr.regs[1],
            pr.regs[2]);
    }
    else if constexpr (N == 4)
    {
        optixTraverse(
            AccelerationStructure,
            Origin,
            Direction,
            TMin,
            TMax,
            Time,
            InstanceInclusionMask,
            RayFlags,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex,
            pr.regs[0],
            pr.regs[1],
            pr.regs[2],
            pr.regs[3]);
    }
    else if constexpr (N == 5)
    {
        optixTraverse(
            AccelerationStructure,
            Origin,
            Direction,
            TMin,
            TMax,
            Time,
            InstanceInclusionMask,
            RayFlags,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex,
            pr.regs[0],
            pr.regs[1],
            pr.regs[2],
            pr.regs[3],
            pr.regs[4]);
    }
    else if constexpr (N == 6)
    {
        optixTraverse(
            AccelerationStructure,
            Origin,
            Direction,
            TMin,
            TMax,
            Time,
            InstanceInclusionMask,
            RayFlags,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex,
            pr.regs[0],
            pr.regs[1],
            pr.regs[2],
            pr.regs[3],
            pr.regs[4],
            pr.regs[5]);
    }
    else if constexpr (N == 7)
    {
        optixTraverse(
            AccelerationStructure,
            Origin,
            Direction,
            TMin,
            TMax,
            Time,
            InstanceInclusionMask,
            RayFlags,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex,
            pr.regs[0],
            pr.regs[1],
            pr.regs[2],
            pr.regs[3],
            pr.regs[4],
            pr.regs[5],
            pr.regs[6]);
    }
    else if constexpr (N == 8)
    {
        optixTraverse(
            AccelerationStructure,
            Origin,
            Direction,
            TMin,
            TMax,
            Time,
            InstanceInclusionMask,
            RayFlags,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex,
            pr.regs[0],
            pr.regs[1],
            pr.regs[2],
            pr.regs[3],
            pr.regs[4],
            pr.regs[5],
            pr.regs[6],
            pr.regs[7]);
    }
    else if constexpr (N <= 16)
    {
        optixTraverse(
            AccelerationStructure,
            Origin,
            Direction,
            TMin,
            TMax,
            Time,
            InstanceInclusionMask,
            RayFlags,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex,
            pr.regs[0],
            pr.regs[1],
            pr.regs[2],
            pr.regs[3],
            pr.regs[4],
            pr.regs[5],
            pr.regs[6],
            pr.regs[7],
            pr.regs[8],
            pr.regs[9],
            pr.regs[10],
            pr.regs[11],
            pr.regs[12],
            pr.regs[13],
            pr.regs[14],
            pr.regs[15]);
    }
    else if constexpr (N <= kMaxOptiXPayloadRegisters)
    {
        optixTraverse(
            AccelerationStructure,
            Origin,
            Direction,
            TMin,
            TMax,
            Time,
            InstanceInclusionMask,
            RayFlags,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex,
            pr.regs[0],
            pr.regs[1],
            pr.regs[2],
            pr.regs[3],
            pr.regs[4],
            pr.regs[5],
            pr.regs[6],
            pr.regs[7],
            pr.regs[8],
            pr.regs[9],
            pr.regs[10],
            pr.regs[11],
            pr.regs[12],
            pr.regs[13],
            pr.regs[14],
            pr.regs[15],
            pr.regs[16],
            pr.regs[17],
            pr.regs[18],
            pr.regs[19],
            pr.regs[20],
            pr.regs[21],
            pr.regs[22],
            pr.regs[23],
            pr.regs[24],
            pr.regs[25],
            pr.regs[26],
            pr.regs[27],
            pr.regs[28],
            pr.regs[29],
            pr.regs[30],
            pr.regs[31]);
    }
}

template<typename T>
__forceinline__ __device__ void optixTraverse(
    OptixTraversableHandle AccelerationStructure,
    uint32_t RayFlags,
    uint32_t InstanceInclusionMask,
    uint32_t RayContributionToHitGroupIndex,
    uint32_t MultiplierForGeometryContributionToHitGroupIndex,
    uint32_t MissShaderIndex,
    RayDesc Ray,
    T* Payload,
    OptixTraversableHandle* hitObj)
{
    constexpr size_t numRegs = (sizeof(T) + 3) / 4;

    if constexpr (numRegs <= kMaxOptiXPayloadRegisters)
    {
        // Register-based approach for small payloads
        PayloadRegisters<T> pr;
        pr.pack(*Payload);

        optixTraverseWithRegs<T>(
            AccelerationStructure,
            Ray.Origin,
            Ray.Direction,
            Ray.TMin,
            Ray.TMax,
            0.f, /* Time for motion blur */
            InstanceInclusionMask,
            RayFlags,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex,
            pr);

        // Read back updated payload registers
        // Native optixTrace updates regs in place
        pr.unpack(*Payload);
    }
    else
    {
        // Pointer-based fallback for large payloads
        uint32_t r0, r1;
        packOptiXRayPayloadPointer((void*)Payload, r0, r1);
        optixTraverse(
            AccelerationStructure,
            Ray.Origin,
            Ray.Direction,
            Ray.TMin,
            Ray.TMax,
            0.f,
            InstanceInclusionMask,
            RayFlags,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex,
            r0,
            r1);
    }
}

template<typename T>
__forceinline__ __device__ void optixTraverse(
    OptixTraversableHandle AccelerationStructure,
    uint32_t RayFlags,
    uint32_t InstanceInclusionMask,
    uint32_t RayContributionToHitGroupIndex,
    uint32_t MultiplierForGeometryContributionToHitGroupIndex,
    uint32_t MissShaderIndex,
    RayDesc Ray,
    float RayTime,
    T* Payload,
    OptixTraversableHandle* hitObj)
{
    constexpr size_t numRegs = (sizeof(T) + 3) / 4;

    if constexpr (numRegs <= kMaxOptiXPayloadRegisters)
    {
        // Register-based approach for small payloads
        PayloadRegisters<T> pr;
        pr.pack(*Payload);

        optixTraverseWithRegs<T>(
            AccelerationStructure,
            Ray.Origin,
            Ray.Direction,
            Ray.TMin,
            Ray.TMax,
            RayTime,
            InstanceInclusionMask,
            RayFlags,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex,
            pr);

        // Read back updated payload registers
        // Native optixTrace updates regs in place
        pr.unpack(*Payload);
    }
    else
    {
        // Pointer-based fallback for large payloads
        uint32_t r0, r1;
        packOptiXRayPayloadPointer((void*)Payload, r0, r1);
        optixTraverse(
            AccelerationStructure,
            Ray.Origin,
            Ray.Direction,
            Ray.TMin,
            Ray.TMax,
            RayTime,
            InstanceInclusionMask,
            RayFlags,
            RayContributionToHitGroupIndex,
            MultiplierForGeometryContributionToHitGroupIndex,
            MissShaderIndex,
            r0,
            r1);
    }
}

// Non-template overload for empty payload case.
// When Slang's type legalization eliminates an empty payload struct,
// the generated code calls optixTraverse without a payload argument.
__forceinline__ __device__ void optixTraverse(
    OptixTraversableHandle AccelerationStructure,
    uint32_t RayFlags,
    uint32_t InstanceInclusionMask,
    uint32_t RayContributionToHitGroupIndex,
    uint32_t MultiplierForGeometryContributionToHitGroupIndex,
    uint32_t MissShaderIndex,
    RayDesc Ray,
    OptixTraversableHandle* hitObj)
{
    optixTraverse(
        AccelerationStructure,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        Ray.TMax,
        0.f,
        InstanceInclusionMask,
        RayFlags,
        RayContributionToHitGroupIndex,
        MultiplierForGeometryContributionToHitGroupIndex,
        MissShaderIndex);
}

#if (OPTIX_VERSION >= 80100)
static __forceinline__ __device__ bool slangOptixHitObjectIsHit(OptixTraversableHandle* hitObj)
{
    return optixHitObjectIsHit();
}
#endif

#if (OPTIX_VERSION >= 80100)
static __forceinline__ __device__ bool slangOptixHitObjectIsMiss(OptixTraversableHandle* hitObj)
{
    return optixHitObjectIsMiss();
}
#endif

#if (OPTIX_VERSION >= 80100)
static __forceinline__ __device__ bool slangOptixHitObjectIsNop(OptixTraversableHandle* hitObj)
{
    return optixHitObjectIsNop();
}
#endif

#if (OPTIX_VERSION >= 90000)
static __forceinline__ __device__ uint
slangOptixHitObjectGetClusterId(OptixTraversableHandle* hitObj)
{
    return optixHitObjectGetClusterId();
}
#endif

#if (OPTIX_VERSION >= 80100)
static __forceinline__ __device__ void optixMakeMissHitObject(
    uint MissShaderIndex,
    RayDesc Ray,
    OptixTraversableHandle* missObj)
{
    optixMakeMissHitObject(
        MissShaderIndex,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        Ray.TMax,
        0.f /* rayTime */
#if (OPTIX_VERSION >= 90000)
        ,
        OPTIX_RAY_FLAG_NONE /* rayFlags*/
#endif
    );
}
#endif

#if (OPTIX_VERSION >= 80100)
static __forceinline__ __device__ void optixMakeMissHitObject(
    uint MissShaderIndex,
    RayDesc Ray,
    float CurrentTime,
    OptixTraversableHandle* missObj)
{
    optixMakeMissHitObject(
        MissShaderIndex,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        Ray.TMax,
        CurrentTime
#if (OPTIX_VERSION >= 90000)
        ,
        OPTIX_RAY_FLAG_NONE /* rayFlags*/
#endif
    );
}
#endif

#if (OPTIX_VERSION >= 90000)
template<typename T>
static __forceinline__ __device__ void optixMakeHitObject(
    OptixTraversableHandle AccelerationStructure,
    uint InstanceIndex,
    uint GeometryIndex,
    uint PrimitiveIndex,
    uint HitKind,
    uint RayContributionToHitGroupIndex,
    uint MultiplierForGeometryContributionToHitGroupIndex,
    RayDesc Ray,
    T attr,
    OptixTraversableHandle* handle)
{
    OptixTraverseData data{};
    optixHitObjectGetTraverseData(&data);
    optixMakeHitObject(
        AccelerationStructure,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        0.f,
        OPTIX_RAY_FLAG_NONE, /* rayFlags*/
        data,
        nullptr, /*OptixTraversableHandle* transforms*/
        0 /*numTransforms */);
}
#elif (OPTIX_VERSION >= 80100)
template<typename T>
static __forceinline__ __device__ void optixMakeHitObject(
    OptixTraversableHandle AccelerationStructure,
    uint InstanceIndex,
    uint GeometryIndex,
    uint PrimitiveIndex,
    uint HitKind,
    uint RayContributionToHitGroupIndex,
    uint MultiplierForGeometryContributionToHitGroupIndex,
    RayDesc Ray,
    T attr,
    OptixTraversableHandle* handle)
{
    // OptiX 8.1 version: call native optixMakeHitObject directly
    optixMakeHitObject(
        AccelerationStructure,                            // handle
        Ray.Origin,                                       // rayOrigin
        Ray.Direction,                                    // rayDirection
        Ray.TMin,                                         // tmin
        Ray.TMax,                                         // tmax
        0.f,                                              // rayTime
        RayContributionToHitGroupIndex,                   // sbtOffset
        MultiplierForGeometryContributionToHitGroupIndex, // sbtStride
        InstanceIndex,                                    // instIdx
        nullptr,                                          // transforms
        0,                                                // numTransforms
        GeometryIndex,                                    // sbtGASIdx
        PrimitiveIndex,                                   // primIdx
        HitKind                                           // hitKind
        /* no attributes passed - empty variadic pack */
    );
}
#endif

#if (OPTIX_VERSION >= 90000)
template<typename T>
static __forceinline__ __device__ void optixMakeHitObject(
    uint HitGroupRecordIndex,
    OptixTraversableHandle AccelerationStructure,
    uint InstanceIndex,
    uint GeometryIndex,
    uint PrimitiveIndex,
    uint HitKind,
    RayDesc Ray,
    T attr,
    OptixTraversableHandle* handle)
{
    OptixTraverseData data{};
    optixHitObjectGetTraverseData(&data);
    optixMakeHitObject(
        AccelerationStructure,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        0.f,
        OPTIX_RAY_FLAG_NONE, /* rayFlags*/
        data,
        nullptr, /*OptixTraversableHandle* transforms*/
        0 /*numTransforms */);
}
#elif (OPTIX_VERSION >= 80100)
template<typename T>
static __forceinline__ __device__ void optixMakeHitObject(
    uint HitGroupRecordIndex,
    OptixTraversableHandle AccelerationStructure,
    uint InstanceIndex,
    uint GeometryIndex,
    uint PrimitiveIndex,
    uint HitKind,
    RayDesc Ray,
    T attr,
    OptixTraversableHandle* handle)
{
    // OptiX 8.1 version: call optixMakeHitObjectWithRecord directly
    optixMakeHitObjectWithRecord(
        AccelerationStructure, // handle
        Ray.Origin,            // rayOrigin
        Ray.Direction,         // rayDirection
        Ray.TMin,              // tmin
        Ray.TMax,              // tmax
        0.f,                   // rayTime
        HitGroupRecordIndex,   // sbtRecordIndex
        InstanceIndex,         // instIdx
        nullptr,               // transforms
        0,                     // numTransforms
        GeometryIndex,         // sbtGASIdx
        PrimitiveIndex,        // primIdx
        HitKind                // hitKind
        /* no attributes passed - empty variadic pack */
    );
}
#endif

#if (OPTIX_VERSION >= 90000)
template<typename T>
static __forceinline__ __device__ void optixMakeHitObject(
    OptixTraversableHandle AccelerationStructure,
    uint InstanceIndex,
    uint GeometryIndex,
    uint PrimitiveIndex,
    uint HitKind,
    uint RayContributionToHitGroupIndex,
    uint MultiplierForGeometryContributionToHitGroupIndex,
    RayDesc Ray,
    float CurrentTime,
    T attr,
    OptixTraversableHandle* handle)
{
    OptixTraverseData data{};
    optixHitObjectGetTraverseData(&data);
    optixMakeHitObject(
        AccelerationStructure,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        CurrentTime,
        OPTIX_RAY_FLAG_NONE, /* rayFlags*/
        data,
        nullptr, /*OptixTraversableHandle* transforms*/
        0 /*numTransforms */);
}
#elif (OPTIX_VERSION >= 80100)
template<typename T>
static __forceinline__ __device__ void optixMakeHitObject(
    OptixTraversableHandle AccelerationStructure,
    uint InstanceIndex,
    uint GeometryIndex,
    uint PrimitiveIndex,
    uint HitKind,
    uint RayContributionToHitGroupIndex,
    uint MultiplierForGeometryContributionToHitGroupIndex,
    RayDesc Ray,
    float CurrentTime,
    T attr,
    OptixTraversableHandle* handle)
{
    // OptiX 8.1 version: call native optixMakeHitObject directly
    optixMakeHitObject(
        AccelerationStructure,                            // handle
        Ray.Origin,                                       // rayOrigin
        Ray.Direction,                                    // rayDirection
        Ray.TMin,                                         // tmin
        Ray.TMax,                                         // tmax
        CurrentTime,                                      // rayTime
        RayContributionToHitGroupIndex,                   // sbtOffset
        MultiplierForGeometryContributionToHitGroupIndex, // sbtStride
        InstanceIndex,                                    // instIdx
        nullptr,                                          // transforms
        0,                                                // numTransforms
        GeometryIndex,                                    // sbtGASIdx
        PrimitiveIndex,                                   // primIdx
        HitKind                                           // hitKind
        /* no attributes passed - empty variadic pack */
    );
}
#endif

#if (OPTIX_VERSION >= 90000)
template<typename T>
static __forceinline__ __device__ void optixMakeHitObject(
    uint HitGroupRecordIndex,
    OptixTraversableHandle AccelerationStructure,
    uint InstanceIndex,
    uint GeometryIndex,
    uint PrimitiveIndex,
    uint HitKind,
    RayDesc Ray,
    float CurrentTime,
    T attr,
    OptixTraversableHandle* handle)
{
    OptixTraverseData data{};
    optixHitObjectGetTraverseData(&data);
    optixMakeHitObject(
        AccelerationStructure,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        CurrentTime,
        OPTIX_RAY_FLAG_NONE, /* rayFlags*/
        data,
        nullptr, /*OptixTraversableHandle* transforms*/
        0 /*numTransforms */);
}
#elif (OPTIX_VERSION >= 80100)
template<typename T>
static __forceinline__ __device__ void optixMakeHitObject(
    uint HitGroupRecordIndex,
    OptixTraversableHandle AccelerationStructure,
    uint InstanceIndex,
    uint GeometryIndex,
    uint PrimitiveIndex,
    uint HitKind,
    RayDesc Ray,
    float CurrentTime,
    T attr,
    OptixTraversableHandle* handle)
{
    // OptiX 8.1 version: call optixMakeHitObjectWithRecord directly
    optixMakeHitObjectWithRecord(
        AccelerationStructure, // handle
        Ray.Origin,            // rayOrigin
        Ray.Direction,         // rayDirection
        Ray.TMin,              // tmin
        Ray.TMax,              // tmax
        CurrentTime,           // rayTime
        HitGroupRecordIndex,   // sbtRecordIndex
        InstanceIndex,         // instIdx
        nullptr,               // transforms
        0,                     // numTransforms
        GeometryIndex,         // sbtGASIdx
        PrimitiveIndex,        // primIdx
        HitKind                // hitKind
        /* no attributes passed - empty variadic pack */
    );
}
#endif

#if (OPTIX_VERSION >= 80100)
static __forceinline__ __device__ void slangOptixMakeNopHitObject(OptixTraversableHandle* Obj)
{
    optixMakeNopHitObject();
}
#endif

#if (OPTIX_VERSION >= 80100)
// Internal helper to call optixInvoke with the right number of register arguments
template<typename T, size_t N = (sizeof(T) + 3) / 4>
__forceinline__ __device__ void optixInvokeWithRegs(PayloadRegisters<T, N>& pr)
{
    if constexpr (N == 0)
    {
        optixInvoke();
    }
    else if constexpr (N == 1)
    {
        optixInvoke(pr.regs[0]);
    }
    else if constexpr (N == 2)
    {
        optixInvoke(pr.regs[0], pr.regs[1]);
    }
    else if constexpr (N == 3)
    {
        optixInvoke(pr.regs[0], pr.regs[1], pr.regs[2]);
    }
    else if constexpr (N == 4)
    {
        optixInvoke(pr.regs[0], pr.regs[1], pr.regs[2], pr.regs[3]);
    }
    else if constexpr (N == 5)
    {
        optixInvoke(pr.regs[0], pr.regs[1], pr.regs[2], pr.regs[3], pr.regs[4]);
    }
    else if constexpr (N == 6)
    {
        optixInvoke(pr.regs[0], pr.regs[1], pr.regs[2], pr.regs[3], pr.regs[4], pr.regs[5]);
    }
    else if constexpr (N == 7)
    {
        optixInvoke(
            pr.regs[0],
            pr.regs[1],
            pr.regs[2],
            pr.regs[3],
            pr.regs[4],
            pr.regs[5],
            pr.regs[6]);
    }
    else if constexpr (N == 8)
    {
        optixInvoke(
            pr.regs[0],
            pr.regs[1],
            pr.regs[2],
            pr.regs[3],
            pr.regs[4],
            pr.regs[5],
            pr.regs[6],
            pr.regs[7]);
    }
    else if constexpr (N <= 16)
    {
        optixInvoke(
            pr.regs[0],
            pr.regs[1],
            pr.regs[2],
            pr.regs[3],
            pr.regs[4],
            pr.regs[5],
            pr.regs[6],
            pr.regs[7],
            pr.regs[8],
            pr.regs[9],
            pr.regs[10],
            pr.regs[11],
            pr.regs[12],
            pr.regs[13],
            pr.regs[14],
            pr.regs[15]);
    }
    else if constexpr (N <= kMaxOptiXPayloadRegisters)
    {
        optixInvoke(
            pr.regs[0],
            pr.regs[1],
            pr.regs[2],
            pr.regs[3],
            pr.regs[4],
            pr.regs[5],
            pr.regs[6],
            pr.regs[7],
            pr.regs[8],
            pr.regs[9],
            pr.regs[10],
            pr.regs[11],
            pr.regs[12],
            pr.regs[13],
            pr.regs[14],
            pr.regs[15],
            pr.regs[16],
            pr.regs[17],
            pr.regs[18],
            pr.regs[19],
            pr.regs[20],
            pr.regs[21],
            pr.regs[22],
            pr.regs[23],
            pr.regs[24],
            pr.regs[25],
            pr.regs[26],
            pr.regs[27],
            pr.regs[28],
            pr.regs[29],
            pr.regs[30],
            pr.regs[31]);
    }
}

template<typename T>
static __forceinline__ __device__ void optixInvoke(
    OptixTraversableHandle AccelerationStructure,
    OptixTraversableHandle* HitOrMiss,
    T* Payload)
{
    constexpr size_t numRegs = (sizeof(T) + 3) / 4;

    if constexpr (numRegs <= kMaxOptiXPayloadRegisters)
    {
        // Register-based approach for small payloads
        PayloadRegisters<T> pr;
        pr.pack(*Payload);
        optixInvokeWithRegs<T>(pr);
        // Read back updated payload registers
        pr.unpack(*Payload);
    }
    else
    {
        // Pointer-based fallback for large payloads
        uint32_t r0, r1;
        packOptiXRayPayloadPointer((void*)Payload, r0, r1);
        optixInvoke(r0, r1);
    }
}

// Overload for empty payloads (when payload is eliminated by type legalization)
static __forceinline__ __device__ void optixInvoke(
    OptixTraversableHandle AccelerationStructure,
    OptixTraversableHandle* HitOrMiss)
{
    // Call OptiX invoke with no payload for empty payload case
    optixInvoke();
}
#endif

#if (OPTIX_VERSION >= 80100)
static __forceinline__ __device__ RayDesc optixHitObjectGetRayDesc(OptixTraversableHandle* obj)
{
    RayDesc ray = {
        optixHitObjectGetWorldRayOrigin(),
        optixHitObjectGetRayTmin(),
        optixHitObjectGetWorldRayDirection(),
        optixHitObjectGetRayTmax()};
    return ray;
}
#endif

#if (OPTIX_VERSION >= 80100)
static __forceinline__ __device__ uint
slangOptixHitObjectGetInstanceIndex(OptixTraversableHandle* Obj)
{
    return optixHitObjectGetInstanceIndex();
}
#endif

#if (OPTIX_VERSION >= 80100)
static __forceinline__ __device__ uint slangOptixHitObjectGetInstanceId(OptixTraversableHandle* Obj)
{
    return optixHitObjectGetInstanceId();
}
#endif

#if (OPTIX_VERSION >= 80000)
static __forceinline__ __device__ float slangOptixHitObjectGetRayTime(OptixTraversableHandle* Obj)
{
    return optixHitObjectGetRayTime();
}
#endif

#if (OPTIX_VERSION >= 80100)
static __forceinline__ __device__ float slangOptixHitObjectGetRayTmax(OptixTraversableHandle* Obj)
{
    return optixHitObjectGetRayTmax();
}
#endif

#if (OPTIX_VERSION >= 80100)
static __forceinline__ __device__ uint
slangOptixHitObjectGetSbtGASIndex(OptixTraversableHandle* Obj)
{
    return optixHitObjectGetSbtGASIndex();
}
#endif

#if (OPTIX_VERSION >= 80100)
static __forceinline__ __device__ uint
slangOptixHitObjectGetPrimitiveIndex(OptixTraversableHandle* Obj)
{
    return optixHitObjectGetPrimitiveIndex();
}
#endif

#if (OPTIX_VERSION >= 80100)
template<typename T>
static __forceinline__ __device__ T optixHitObjectGetAttribute(OptixTraversableHandle* Obj)
{
    constexpr size_t numInts = (sizeof(T) + sizeof(uint32_t) - 1) /
                               sizeof(uint32_t); // Number of 32-bit values, rounded up
    static_assert(numInts <= 8, "Attribute type is too large");

    // Create an array to hold the attribute values
    uint32_t values[numInts == 0 ? 1 : numInts] = {0}; // Ensure we have at least one element

    // Read the appropriate number of attribute registers
    if constexpr (numInts > 0)
        values[0] = optixHitObjectGetAttribute_0();
    if constexpr (numInts > 1)
        values[1] = optixHitObjectGetAttribute_1();
    if constexpr (numInts > 2)
        values[2] = optixHitObjectGetAttribute_2();
    if constexpr (numInts > 3)
        values[3] = optixHitObjectGetAttribute_3();
    if constexpr (numInts > 4)
        values[4] = optixHitObjectGetAttribute_4();
    if constexpr (numInts > 5)
        values[5] = optixHitObjectGetAttribute_5();
    if constexpr (numInts > 6)
        values[6] = optixHitObjectGetAttribute_6();
    if constexpr (numInts > 7)
        values[7] = optixHitObjectGetAttribute_7();

    // Reinterpret the array as the desired type
    T result;
    memcpy(&result, values, sizeof(T));
    return result;
}
#endif

#if (OPTIX_VERSION >= 80100)
static __forceinline__ __device__ uint
slangOptixHitObjectGetSbtRecordIndex(OptixTraversableHandle* Obj)
{
    return optixHitObjectGetSbtRecordIndex();
}
#endif

#if (OPTIX_VERSION >= 90000)
static __forceinline__ __device__ void slangOptixHitObjectSetSbtRecordIndex(
    OptixTraversableHandle* Obj,
    uint sbtRecordIndex)
{
    optixHitObjectSetSbtRecordIndex(sbtRecordIndex);
}
#endif

// HitObject transform matrix wrappers for SER (Shader Execution Reordering)
// These wrappers convert OptiX's float[12] matrix format to Slang's Matrix type
// Available in RG, CH, MS, CC, DC stages per OptiX documentation
// Note: optixHitObjectGetWorldToObjectTransformMatrix/optixHitObjectGetObjectToWorldTransformMatrix
// were added in OptiX 9.0 (not available in 8.0 or 8.1)
#if (OPTIX_VERSION >= 90000)
static __forceinline__ __device__ Matrix<float, 4, 3> slangOptixHitObjectGetWorldToObject(
    OptixTraversableHandle* hitObj)
{
    float m[12];
    optixHitObjectGetWorldToObjectTransformMatrix(m);
    // OptiX stores matrix as 3 rows of float4, we need to transpose to 4 rows of float3
    return makeMatrix<float, 4, 3>(
        make_float3(m[0], m[4], m[8]),
        make_float3(m[1], m[5], m[9]),
        make_float3(m[2], m[6], m[10]),
        make_float3(m[3], m[7], m[11]));
}
#endif

#if (OPTIX_VERSION >= 90000)
static __forceinline__ __device__ Matrix<float, 4, 3> slangOptixHitObjectGetObjectToWorld(
    OptixTraversableHandle* hitObj)
{
    float m[12];
    optixHitObjectGetObjectToWorldTransformMatrix(m);
    // OptiX stores matrix as 3 rows of float4, we need to transpose to 4 rows of float3
    return makeMatrix<float, 4, 3>(
        make_float3(m[0], m[4], m[8]),
        make_float3(m[1], m[5], m[9]),
        make_float3(m[2], m[6], m[10]),
        make_float3(m[3], m[7], m[11]));
}
#endif

// OptiX multi-level traversal wrappers
// These wrappers convert OptiX's float[12] matrix pointer returns to Slang's Matrix type
__device__ __forceinline__ Matrix<float, 3, 4> _slang_optixGetInstanceTransformFromHandle(
    ulonglong handle)
{
    const float4* m = optixGetInstanceTransformFromHandle(handle);
    // OptiX stores matrix as 3 rows of float4 in the array
    return makeMatrix<float, 3, 4>(m[0], m[1], m[2]);
}

__device__ __forceinline__ Matrix<float, 3, 4> _slang_optixGetInstanceInverseTransformFromHandle(
    ulonglong handle)
{
    const float4* m = optixGetInstanceInverseTransformFromHandle(handle);
    // OptiX stores matrix as 3 rows of float4 in the array
    return makeMatrix<float, 3, 4>(m[0], m[1], m[2]);
}

// OptiX transformation matrix wrappers
// These wrappers convert OptiX's float[12] matrix format to Slang's Matrix type
__device__ __forceinline__ Matrix<float, 3, 4> slangOptixGetObjectToWorldTransformMatrix()
{
    float m[12];
    optixGetObjectToWorldTransformMatrix(m);
    // OptiX stores matrix as 3 rows of float4 in the array
    return makeMatrix<float, 3, 4>(
        make_float4(m[0], m[1], m[2], m[3]),
        make_float4(m[4], m[5], m[6], m[7]),
        make_float4(m[8], m[9], m[10], m[11]));
}

__device__ __forceinline__ Matrix<float, 3, 4> slangOptixGetWorldToObjectTransformMatrix()
{
    float m[12];
    optixGetWorldToObjectTransformMatrix(m);
    // OptiX stores matrix as 3 rows of float4 in the array
    return makeMatrix<float, 3, 4>(
        make_float4(m[0], m[1], m[2], m[3]),
        make_float4(m[4], m[5], m[6], m[7]),
        make_float4(m[8], m[9], m[10], m[11]));
}

__device__ __forceinline__ Matrix<float, 4, 3> slangOptixGetObjectToWorldTransformMatrix4x3()
{
    float m[12];
    optixGetObjectToWorldTransformMatrix(m);
    // OptiX stores matrix as 3 rows of float4, we need to transpose to 4 rows of float3
    return makeMatrix<float, 4, 3>(
        make_float3(m[0], m[4], m[8]),
        make_float3(m[1], m[5], m[9]),
        make_float3(m[2], m[6], m[10]),
        make_float3(m[3], m[7], m[11]));
}

__device__ __forceinline__ Matrix<float, 4, 3> slangOptixGetWorldToObjectTransformMatrix4x3()
{
    float m[12];
    optixGetWorldToObjectTransformMatrix(m);
    // OptiX stores matrix as 3 rows of float4, we need to transpose to 4 rows of float3
    return makeMatrix<float, 4, 3>(
        make_float3(m[0], m[4], m[8]),
        make_float3(m[1], m[5], m[9]),
        make_float3(m[2], m[6], m[10]),
        make_float3(m[3], m[7], m[11]));
}

#else
// Define OptixTraversableHandle even if OptiX is not enabled.
// This allows RaytracingAccelerationStructure to be properly reflected in non-OptiX code.
typedef unsigned long long OptixTraversableHandle;
#endif
static const int kSlangTorchTensorMaxDim = 5;

// TensorView
// NOTE: If you change this struct's layout, also update the hard-coded size/alignment
// in _createTypeLayout() in slang-type-layout.cpp.
struct TensorView
{
    uint8_t* data;
    uint32_t strides[kSlangTorchTensorMaxDim];
    uint32_t sizes[kSlangTorchTensorMaxDim];
    uint32_t dimensionCount;

    template<typename T>
    __device__ T* data_ptr()
    {
        return reinterpret_cast<T*>(data);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint32_t index)
    {
        uint64_t offset = strides[0] * index;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint2 index)
    {
        uint64_t offset = strides[0] * index.x + strides[1] * index.y;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint3 index)
    {
        uint64_t offset = strides[0] * index.x + strides[1] * index.y + strides[2] * index.z;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint4 index)
    {
        uint64_t offset = strides[0] * index.x + strides[1] * index.y + strides[2] * index.z +
                          strides[3] * index.w;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T, unsigned int N>
    __device__ T* data_ptr_at(uint index[N])
    {
        uint64_t offset = 0;
        for (unsigned int i = 0; i < N; ++i)
        {
            offset += strides[i] * index[i];
        }
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T& load(uint32_t x)
    {
        return *reinterpret_cast<T*>(data + strides[0] * x);
    }
    template<typename T>
    __device__ T& load(uint32_t x, uint32_t y)
    {
        return *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y);
    }
    template<typename T>
    __device__ T& load(uint2 index)
    {
        return *reinterpret_cast<T*>(data + strides[0] * index.x + strides[1] * index.y);
    }
    template<typename T>
    __device__ T& load(uint32_t x, uint32_t y, uint32_t z)
    {
        return *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y + strides[2] * z);
    }
    template<typename T>
    __device__ T& load(uint3 index)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z);
    }
    template<typename T>
    __device__ T& load(uint32_t x, uint32_t y, uint32_t z, uint32_t w)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * x + strides[1] * y + strides[2] * z + strides[3] * w);
    }
    template<typename T>
    __device__ T& load(uint4 index)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z +
            strides[3] * index.w);
    }
    template<typename T>
    __device__ T& load(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t i4)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * i0 + strides[1] * i1 + strides[2] * i2 + strides[3] * i3 +
            strides[4] * i4);
    }

    // Generic version of load
    template<typename T, unsigned int N>
    __device__ T& load(uint index[N])
    {
        uint64_t offset = 0;
        for (unsigned int i = 0; i < N; ++i)
        {
            offset += strides[i] * index[i];
        }
        return *reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ void store(uint32_t x, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * x) = val;
    }
    template<typename T>
    __device__ void store(uint32_t x, uint32_t y, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y) = val;
    }
    template<typename T>
    __device__ void store(uint2 index, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * index.x + strides[1] * index.y) = val;
    }
    template<typename T>
    __device__ void store(uint32_t x, uint32_t y, uint32_t z, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y + strides[2] * z) = val;
    }
    template<typename T>
    __device__ void store(uint3 index, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z) = val;
    }
    template<typename T>
    __device__ void store(uint32_t x, uint32_t y, uint32_t z, uint32_t w, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * x + strides[1] * y + strides[2] * z + strides[3] * w) = val;
    }
    template<typename T>
    __device__ void store(uint4 index, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z +
            strides[3] * index.w) = val;
    }
    template<typename T>
    __device__ void store(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t i4, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * i0 + strides[1] * i1 + strides[2] * i2 + strides[3] * i3 +
            strides[4] * i4) = val;
    }

    // Generic version
    template<typename T, unsigned int N>
    __device__ void store(uint index[N], T val)
    {
        uint64_t offset = 0;
        for (unsigned int i = 0; i < N; ++i)
        {
            offset += strides[i] * index[i];
        }
        *reinterpret_cast<T*>(data + offset) = val;
    }
};

// Implementations for texture fetch/load functions using tex PTX intrinsics
// These are used for read-only texture access with integer coordinates.

// 1D is not supported via PTX. Keeping the implementation below in case it ever gets supported.
template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL T tex1Dfetch_int(CUtexObject texObj, int x, int mip)
{
    // TODO: static_assert(false) can fail on some compilers, even if template is not instantiated.
    // We should check for this in hlsl.meta.slang instead.
    // static_assert(false, "CUDA does not support fetching from 1D textures");
}

#if 0
#define SLANG_TEX1DFETCH_INT_IMPL(T, dtype, c)                                                 \
    template<>                                                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T tex1Dfetch_int(CUtexObject texObj, int x, int mip)    \
    {                                                                                          \
        T result;                                                                              \
        [[maybe_unused]] T stub;                                                               \
        asm("tex.level.1d.v4." dtype ".s32 {%0, %1, %2, %3}, [%4, {%5}], %6;"                  \
            : c(result), c(stub), c(stub), c(stub)                                             \
            : "l"(texObj), "r"(x), "r"(mip));                                                  \
        return result;                                                                         \
    }                                                                                          \
    template<>                                                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##2 tex1Dfetch_int(CUtexObject texObj, int x, int mip) \
    {                                                                                          \
        T result_x, result_y;                                                                  \
        [[maybe_unused]] T stub;                                                               \
        asm("tex.level.1d.v4." dtype ".s32 {%0, %1, %2, %3}, [%4, {%5}], %6;"                  \
            : c(result_x), c(result_y), c(stub), c(stub)                                       \
            : "l"(texObj), "r"(x), "r"(mip));                                                  \
        return make_##T##2(result_x, result_y);                                                \
    }                                                                                          \
    template<>                                                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##4 tex1Dfetch_int(CUtexObject texObj, int x, int mip) \
    {                                                                                          \
        T result_x, result_y, result_z, result_w;                                              \
        asm("tex.level.1d.v4." dtype ".s32 {%0, %1, %2, %3}, [%4, {%5}], %6;"                  \
            : c(result_x), c(result_y), c(result_z), c(result_w)                               \
            : "l"(texObj), "r"(x), "r"(mip));                                                  \
        return make_##T##4(result_x, result_y, result_z, result_w);                            \
    }

SLANG_TEX1DFETCH_INT_IMPL(float, "f32", "=f")
SLANG_TEX1DFETCH_INT_IMPL(uint, "u32", "=r")
SLANG_TEX1DFETCH_INT_IMPL(int, "s32", "=r")
#endif

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL T tex2Dfetch_int(CUtexObject texObj, int x, int y, int mip);

#define SLANG_TEX2DFETCH_INT_IMPL(T, dtype, c)                                                     \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T tex2Dfetch_int(CUtexObject texObj, int x, int y, int mip) \
    {                                                                                              \
        T result;                                                                                  \
        [[maybe_unused]] T stub;                                                                   \
        asm("tex.level.2d.v4." dtype ".s32 {%0, %1, %2, %3}, [%4, {%5, %6}], %7;"                  \
            : c(result), c(stub), c(stub), c(stub)                                                 \
            : "l"(texObj), "r"(x), "r"(y), "r"(mip));                                              \
        return result;                                                                             \
    }                                                                                              \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL                                                             \
        T##2 tex2Dfetch_int(CUtexObject texObj, int x, int y, int mip)                             \
    {                                                                                              \
        T result_x, result_y;                                                                      \
        [[maybe_unused]] T stub;                                                                   \
        asm("tex.level.2d.v4." dtype ".s32 {%0, %1, %2, %3}, [%4, {%5, %6}], %7;"                  \
            : c(result_x), c(result_y), c(stub), c(stub)                                           \
            : "l"(texObj), "r"(x), "r"(y), "r"(mip));                                              \
        return make_##T##2(result_x, result_y);                                                    \
    }                                                                                              \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL                                                             \
        T##4 tex2Dfetch_int(CUtexObject texObj, int x, int y, int mip)                             \
    {                                                                                              \
        T result_x, result_y, result_z, result_w;                                                  \
        asm("tex.level.2d.v4." dtype ".s32 {%0, %1, %2, %3}, [%4, {%5, %6}], %7;"                  \
            : c(result_x), c(result_y), c(result_z), c(result_w)                                   \
            : "l"(texObj), "r"(x), "r"(y), "r"(mip));                                              \
        return make_##T##4(result_x, result_y, result_z, result_w);                                \
    }

SLANG_TEX2DFETCH_INT_IMPL(float, "f32", "=f")
SLANG_TEX2DFETCH_INT_IMPL(uint, "u32", "=r")
SLANG_TEX2DFETCH_INT_IMPL(int, "s32", "=r")


template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL T
tex3Dfetch_int(CUtexObject texObj, int x, int y, int z, int mip);

#define SLANG_TEX3DFETCH_INT_IMPL(T, dtype, c)                                            \
    template<>                                                                            \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T                                                  \
    tex3Dfetch_int(CUtexObject texObj, int x, int y, int z, int mip)                      \
    {                                                                                     \
        T result;                                                                         \
        [[maybe_unused]] T stub;                                                          \
        asm("tex.level.3d.v4." dtype ".s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}], %9;" \
            : c(result), c(stub), c(stub), c(stub)                                        \
            : "l"(texObj), "r"(x), "r"(y), "r"(z), "r"(z) /* ignored */, "r"(mip));       \
        return result;                                                                    \
    }                                                                                     \
    template<>                                                                            \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL                                                    \
        T##2 tex3Dfetch_int(CUtexObject texObj, int x, int y, int z, int mip)             \
    {                                                                                     \
        T result_x, result_y;                                                             \
        [[maybe_unused]] T stub;                                                          \
        asm("tex.level.3d.v4." dtype ".s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}], %9;" \
            : c(result_x), c(result_y), c(stub), c(stub)                                  \
            : "l"(texObj), "r"(x), "r"(y), "r"(z), "r"(z) /* ignored */, "r"(mip));       \
        return make_##T##2(result_x, result_y);                                           \
    }                                                                                     \
    template<>                                                                            \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL                                                    \
        T##4 tex3Dfetch_int(CUtexObject texObj, int x, int y, int z, int mip)             \
    {                                                                                     \
        T result_x, result_y, result_z, result_w;                                         \
        asm("tex.level.3d.v4." dtype ".s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}], %9;" \
            : c(result_x), c(result_y), c(result_z), c(result_w)                          \
            : "l"(texObj), "r"(x), "r"(y), "r"(z), "r"(z) /* ignored */, "r"(mip));       \
        return make_##T##4(result_x, result_y, result_z, result_w);                       \
    }

SLANG_TEX3DFETCH_INT_IMPL(float, "f32", "=f")
SLANG_TEX3DFETCH_INT_IMPL(uint, "u32", "=r")
SLANG_TEX3DFETCH_INT_IMPL(int, "s32", "=r")

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL T
tex1DArrayfetch_int(CUtexObject texObj, int x, int layer, int mip);

#define SLANG_TEX1DARRAYFETCH_INT_IMPL(T, dtype, c)                                \
    template<>                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T                                           \
    tex1DArrayfetch_int(CUtexObject texObj, int x, int layer, int mip)             \
    {                                                                              \
        T result;                                                                  \
        [[maybe_unused]] T stub;                                                   \
        asm("tex.level.a1d.v4." dtype ".s32 {%0, %1, %2, %3}, [%4, {%5, %6}], %7;" \
            : c(result), c(stub), c(stub), c(stub)                                 \
            : "l"(texObj), "r"(layer), "r"(x), "r"(mip));                          \
        return result;                                                             \
    }                                                                              \
    template<>                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL                                             \
        T##2 tex1DArrayfetch_int(CUtexObject texObj, int x, int layer, int mip)    \
    {                                                                              \
        T result_x, result_y;                                                      \
        [[maybe_unused]] T stub;                                                   \
        asm("tex.level.a1d.v4." dtype ".s32 {%0, %1, %2, %3}, [%4, {%5, %6}], %7;" \
            : c(result_x), c(result_y), c(stub), c(stub)                           \
            : "l"(texObj), "r"(layer), "r"(x), "r"(mip));                          \
        return make_##T##2(result_x, result_y);                                    \
    }                                                                              \
    template<>                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL                                             \
        T##4 tex1DArrayfetch_int(CUtexObject texObj, int x, int layer, int mip)    \
    {                                                                              \
        T result_x, result_y, result_z, result_w;                                  \
        asm("tex.level.a1d.v4." dtype ".s32 {%0, %1, %2, %3}, [%4, {%5, %6}], %7;" \
            : c(result_x), c(result_y), c(result_z), c(result_w)                   \
            : "l"(texObj), "r"(layer), "r"(x), "r"(mip));                          \
        return make_##T##4(result_x, result_y, result_z, result_w);                \
    }

SLANG_TEX1DARRAYFETCH_INT_IMPL(float, "f32", "=f")
SLANG_TEX1DARRAYFETCH_INT_IMPL(uint, "u32", "=r")
SLANG_TEX1DARRAYFETCH_INT_IMPL(int, "s32", "=r")

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL T
tex2DArrayfetch_int(CUtexObject texObj, int x, int y, int layer, int mip);

#define SLANG_TEX2DARRAYFETCH_INT_IMPL(T, dtype, c)                                         \
    template<>                                                                              \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T                                                    \
    tex2DArrayfetch_int(CUtexObject texObj, int x, int y, int layer, int mip)               \
    {                                                                                       \
        T result;                                                                           \
        [[maybe_unused]] T stub;                                                            \
        asm("tex.level.a2d.v4." dtype ".s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}], %9;"  \
            : c(result), c(stub), c(stub), c(stub)                                          \
            : "l"(texObj), "r"(layer), "r"(x), "r"(y), "r"(layer) /* ignored */, "r"(mip)); \
        return result;                                                                      \
    }                                                                                       \
    template<>                                                                              \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL                                                      \
        T##2 tex2DArrayfetch_int(CUtexObject texObj, int x, int y, int layer, int mip)      \
    {                                                                                       \
        T result_x, result_y;                                                               \
        [[maybe_unused]] T stub;                                                            \
        asm("tex.level.a2d.v4." dtype ".s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}], %9;"  \
            : c(result_x), c(result_y), c(stub), c(stub)                                    \
            : "l"(texObj), "r"(layer), "r"(x), "r"(y), "r"(layer) /* ignored */, "r"(mip)); \
        return make_##T##2(result_x, result_y);                                             \
    }                                                                                       \
    template<>                                                                              \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL                                                      \
        T##4 tex2DArrayfetch_int(CUtexObject texObj, int x, int y, int layer, int mip)      \
    {                                                                                       \
        T result_x, result_y, result_z, result_w;                                           \
        asm("tex.level.a2d.v4." dtype ".s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}], %9;"  \
            : c(result_x), c(result_y), c(result_z), c(result_w)                            \
            : "l"(texObj), "r"(layer), "r"(x), "r"(y), "r"(layer) /* ignored */, "r"(mip)); \
        return make_##T##4(result_x, result_y, result_z, result_w);                         \
    }

SLANG_TEX2DARRAYFETCH_INT_IMPL(float, "f32", "=f")
SLANG_TEX2DARRAYFETCH_INT_IMPL(uint, "u32", "=r")
SLANG_TEX2DARRAYFETCH_INT_IMPL(int, "s32", "=r")

// Wave rotate helper functions - templated approach
#define SLANG_WARP_FULL_MASK 0xFFFFFFFF

// Macro-based wave rotate implementation following codebase patterns
#define SLANG_WAVE_ROTATE_IMPL(T)                                                     \
    __device__ __forceinline__ T##2 _slang_waveRotate(T##2 value, unsigned int delta) \
    {                                                                                 \
        return make_##T##2(                                                           \
            (T)__shfl_sync(                                                           \
                SLANG_WARP_FULL_MASK,                                                 \
                value.x,                                                              \
                (_getLaneId() + delta) % SLANG_CUDA_WARP_SIZE),                       \
            (T)__shfl_sync(                                                           \
                SLANG_WARP_FULL_MASK,                                                 \
                value.y,                                                              \
                (_getLaneId() + delta) % SLANG_CUDA_WARP_SIZE));                      \
    }                                                                                 \
    __device__ __forceinline__ T##3 _slang_waveRotate(T##3 value, unsigned int delta) \
    {                                                                                 \
        return make_##T##3(                                                           \
            (T)__shfl_sync(                                                           \
                SLANG_WARP_FULL_MASK,                                                 \
                value.x,                                                              \
                (_getLaneId() + delta) % SLANG_CUDA_WARP_SIZE),                       \
            (T)__shfl_sync(                                                           \
                SLANG_WARP_FULL_MASK,                                                 \
                value.y,                                                              \
                (_getLaneId() + delta) % SLANG_CUDA_WARP_SIZE),                       \
            (T)__shfl_sync(                                                           \
                SLANG_WARP_FULL_MASK,                                                 \
                value.z,                                                              \
                (_getLaneId() + delta) % SLANG_CUDA_WARP_SIZE));                      \
    }                                                                                 \
    __device__ __forceinline__ T##4 _slang_waveRotate(T##4 value, unsigned int delta) \
    {                                                                                 \
        return make_##T##4(                                                           \
            (T)__shfl_sync(                                                           \
                SLANG_WARP_FULL_MASK,                                                 \
                value.x,                                                              \
                (_getLaneId() + delta) % SLANG_CUDA_WARP_SIZE),                       \
            (T)__shfl_sync(                                                           \
                SLANG_WARP_FULL_MASK,                                                 \
                value.y,                                                              \
                (_getLaneId() + delta) % SLANG_CUDA_WARP_SIZE),                       \
            (T)__shfl_sync(                                                           \
                SLANG_WARP_FULL_MASK,                                                 \
                value.z,                                                              \
                (_getLaneId() + delta) % SLANG_CUDA_WARP_SIZE),                       \
            (T)__shfl_sync(                                                           \
                SLANG_WARP_FULL_MASK,                                                 \
                value.w,                                                              \
                (_getLaneId() + delta) % SLANG_CUDA_WARP_SIZE));                      \
    }

// Generate wave rotate functions for all standard vector types
SLANG_WAVE_ROTATE_IMPL(uint)
SLANG_WAVE_ROTATE_IMPL(int)
SLANG_WAVE_ROTATE_IMPL(float)
SLANG_WAVE_ROTATE_IMPL(short)
SLANG_WAVE_ROTATE_IMPL(ushort)
SLANG_WAVE_ROTATE_IMPL(char)
SLANG_WAVE_ROTATE_IMPL(uchar)
SLANG_WAVE_ROTATE_IMPL(longlong)
SLANG_WAVE_ROTATE_IMPL(ulonglong)

#ifdef SLANG_CUDA_ENABLE_HALF
SLANG_WAVE_ROTATE_IMPL(__half)
#endif

// Special handling for boolean vectors (requires int conversion)
__device__ __forceinline__ bool2 _slang_waveRotate(bool2 value, unsigned int delta)
{
    int2 intValue = make_int2((int)value.x, (int)value.y);
    int2 result = _slang_waveRotate(intValue, delta);
    return make_bool2((bool)result.x, (bool)result.y);
}

__device__ __forceinline__ bool3 _slang_waveRotate(bool3 value, unsigned int delta)
{
    int3 intValue = make_int3((int)value.x, (int)value.y, (int)value.z);
    int3 result = _slang_waveRotate(intValue, delta);
    return make_bool3((bool)result.x, (bool)result.y, (bool)result.z);
}

__device__ __forceinline__ bool4 _slang_waveRotate(bool4 value, unsigned int delta)
{
    int4 intValue = make_int4((int)value.x, (int)value.y, (int)value.z, (int)value.w);
    int4 result = _slang_waveRotate(intValue, delta);
    return make_bool4((bool)result.x, (bool)result.y, (bool)result.z, (bool)result.w);
}

#undef SLANG_WAVE_ROTATE_IMPL

// Quad control operations for CUDA
__device__ __forceinline__ bool _slang_quadAny(bool expr)
{
    // Get values from all 4 lanes in the quad
    bool v0 = __shfl_sync(0xFFFFFFFF, expr, (_getLaneId() & 0xFFFFFFFC) | 0);
    bool v1 = __shfl_sync(0xFFFFFFFF, expr, (_getLaneId() & 0xFFFFFFFC) | 1);
    bool v2 = __shfl_sync(0xFFFFFFFF, expr, (_getLaneId() & 0xFFFFFFFC) | 2);
    bool v3 = __shfl_sync(0xFFFFFFFF, expr, (_getLaneId() & 0xFFFFFFFC) | 3);
    return v0 || v1 || v2 || v3;
}

__device__ __forceinline__ bool _slang_quadAll(bool expr)
{
    // Get values from all 4 lanes in the quad
    bool v0 = __shfl_sync(0xFFFFFFFF, expr, (_getLaneId() & 0xFFFFFFFC) | 0);
    bool v1 = __shfl_sync(0xFFFFFFFF, expr, (_getLaneId() & 0xFFFFFFFC) | 1);
    bool v2 = __shfl_sync(0xFFFFFFFF, expr, (_getLaneId() & 0xFFFFFFFC) | 2);
    bool v3 = __shfl_sync(0xFFFFFFFF, expr, (_getLaneId() & 0xFFFFFFFC) | 3);
    return v0 && v1 && v2 && v3;
}

// Clustered wave rotate operations for CUDA
// Clustered rotate rotates values within clusters of specified size
#define SLANG_WAVE_CLUSTERED_ROTATE_IMPL(T)                                                       \
    __device__ __forceinline__ T                                                                  \
    _slang_waveClusteredRotate(T value, unsigned int delta, unsigned int clusterSize)             \
    {                                                                                             \
        unsigned int laneId = _getLaneId();                                                       \
        unsigned int clusterStart = (laneId / clusterSize) * clusterSize;                         \
        unsigned int targetLane = clusterStart + ((laneId - clusterStart + delta) % clusterSize); \
        return __shfl_sync(SLANG_WARP_FULL_MASK, value, targetLane);                              \
    }                                                                                             \
    __device__ __forceinline__                                                                    \
        T##2 _slang_waveClusteredRotate(T##2 value, unsigned int delta, unsigned int clusterSize) \
    {                                                                                             \
        unsigned int laneId = _getLaneId();                                                       \
        unsigned int clusterStart = (laneId / clusterSize) * clusterSize;                         \
        unsigned int targetLane = clusterStart + ((laneId - clusterStart + delta) % clusterSize); \
        return make_##T##2(                                                                       \
            (T)__shfl_sync(SLANG_WARP_FULL_MASK, value.x, targetLane),                            \
            (T)__shfl_sync(SLANG_WARP_FULL_MASK, value.y, targetLane));                           \
    }                                                                                             \
    __device__ __forceinline__                                                                    \
        T##3 _slang_waveClusteredRotate(T##3 value, unsigned int delta, unsigned int clusterSize) \
    {                                                                                             \
        unsigned int laneId = _getLaneId();                                                       \
        unsigned int clusterStart = (laneId / clusterSize) * clusterSize;                         \
        unsigned int targetLane = clusterStart + ((laneId - clusterStart + delta) % clusterSize); \
        return make_##T##3(                                                                       \
            (T)__shfl_sync(SLANG_WARP_FULL_MASK, value.x, targetLane),                            \
            (T)__shfl_sync(SLANG_WARP_FULL_MASK, value.y, targetLane),                            \
            (T)__shfl_sync(SLANG_WARP_FULL_MASK, value.z, targetLane));                           \
    }                                                                                             \
    __device__ __forceinline__                                                                    \
        T##4 _slang_waveClusteredRotate(T##4 value, unsigned int delta, unsigned int clusterSize) \
    {                                                                                             \
        unsigned int laneId = _getLaneId();                                                       \
        unsigned int clusterStart = (laneId / clusterSize) * clusterSize;                         \
        unsigned int targetLane = clusterStart + ((laneId - clusterStart + delta) % clusterSize); \
        return make_##T##4(                                                                       \
            (T)__shfl_sync(SLANG_WARP_FULL_MASK, value.x, targetLane),                            \
            (T)__shfl_sync(SLANG_WARP_FULL_MASK, value.y, targetLane),                            \
            (T)__shfl_sync(SLANG_WARP_FULL_MASK, value.z, targetLane),                            \
            (T)__shfl_sync(SLANG_WARP_FULL_MASK, value.w, targetLane));                           \
    }

// Generate clustered wave rotate functions for all standard types
SLANG_WAVE_CLUSTERED_ROTATE_IMPL(uint)
SLANG_WAVE_CLUSTERED_ROTATE_IMPL(int)
SLANG_WAVE_CLUSTERED_ROTATE_IMPL(float)
SLANG_WAVE_CLUSTERED_ROTATE_IMPL(short)
SLANG_WAVE_CLUSTERED_ROTATE_IMPL(ushort)
SLANG_WAVE_CLUSTERED_ROTATE_IMPL(char)
SLANG_WAVE_CLUSTERED_ROTATE_IMPL(uchar)
SLANG_WAVE_CLUSTERED_ROTATE_IMPL(longlong)
SLANG_WAVE_CLUSTERED_ROTATE_IMPL(ulonglong)

#ifdef SLANG_CUDA_ENABLE_HALF
SLANG_WAVE_CLUSTERED_ROTATE_IMPL(__half)
#endif

// Special handling for boolean clustered rotate
__device__ __forceinline__ bool _slang_waveClusteredRotate(
    bool value,
    unsigned int delta,
    unsigned int clusterSize)
{
    int intValue = (int)value;
    int result = _slang_waveClusteredRotate(intValue, delta, clusterSize);
    return (bool)result;
}

__device__ __forceinline__ bool2
_slang_waveClusteredRotate(bool2 value, unsigned int delta, unsigned int clusterSize)
{
    int2 intValue = make_int2((int)value.x, (int)value.y);
    int2 result = _slang_waveClusteredRotate(intValue, delta, clusterSize);
    return make_bool2((bool)result.x, (bool)result.y);
}

__device__ __forceinline__ bool3
_slang_waveClusteredRotate(bool3 value, unsigned int delta, unsigned int clusterSize)
{
    int3 intValue = make_int3((int)value.x, (int)value.y, (int)value.z);
    int3 result = _slang_waveClusteredRotate(intValue, delta, clusterSize);
    return make_bool3((bool)result.x, (bool)result.y, (bool)result.z);
}

__device__ __forceinline__ bool4
_slang_waveClusteredRotate(bool4 value, unsigned int delta, unsigned int clusterSize)
{
    int4 intValue = make_int4((int)value.x, (int)value.y, (int)value.z, (int)value.w);
    int4 result = _slang_waveClusteredRotate(intValue, delta, clusterSize);
    return make_bool4((bool)result.x, (bool)result.y, (bool)result.z, (bool)result.w);
}

#undef SLANG_WAVE_CLUSTERED_ROTATE_IMPL

// ---------------------- OptiX Cooperative Vector Wrappers --------------------------------------
#ifdef SLANG_CUDA_ENABLE_OPTIX

#if (OPTIX_VERSION >= 90000)

// Constexpr function to map Slang component type enum to OptiX cooperative vector element type
__host__ __device__ constexpr OptixCoopVecElemType slangToOptixComponentType(unsigned slangEnum)
{
    switch (slangEnum)
    {
    case 0:
        return OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E4M3; // FloatE4M3
    case 1:
        return OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E5M2; // FloatE5M2
    case 2:
        return OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16; // Float16
    case 3:
        return OPTIX_COOP_VEC_ELEM_TYPE_FLOAT32; // Float32
    case 5:
        return OPTIX_COOP_VEC_ELEM_TYPE_INT8; // SignedInt8
    case 7:
        return OPTIX_COOP_VEC_ELEM_TYPE_INT32; // SignedInt32
    case 10:
        return OPTIX_COOP_VEC_ELEM_TYPE_UINT8; // UnsignedInt8
    case 12:
        return OPTIX_COOP_VEC_ELEM_TYPE_UINT32; // UnsignedInt32
    default:
        return OPTIX_COOP_VEC_ELEM_TYPE_FLOAT32; // Default
    }
}

// Constexpr function to map Slang matrix layout enum to OptiX cooperative vector matrix layout
__host__ __device__ constexpr OptixCoopVecMatrixLayout slangToOptixMatrixLayout(unsigned slangEnum)
{
    switch (slangEnum)
    {
    case 0:
        return OPTIX_COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR; // RowMajor
    case 1:
        return OPTIX_COOP_VEC_MATRIX_LAYOUT_COLUMN_MAJOR; // ColumnMajor
    case 2:
        return OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL; // InferencingOptimal
    case 3:
        return OPTIX_COOP_VEC_MATRIX_LAYOUT_TRAINING_OPTIMAL; // TrainingOptimal
    default:
        return OPTIX_COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR; // Default
    }
}

// Wrapper structs to maintain compatibility with existing template-based interface
template<unsigned SlangEnum>
struct SlangToOptixComponentType
{
    static constexpr OptixCoopVecElemType value = slangToOptixComponentType(SlangEnum);
};

template<unsigned SlangEnum>
struct SlangToOptixMatrixLayout
{
    static constexpr OptixCoopVecMatrixLayout value = slangToOptixMatrixLayout(SlangEnum);
};

// Template trait to extract vector size from OptixCoopVec<T, N>
// Conditional compilation for NVRTC compatibility
template<typename T>
struct OptixCoopVecTraits;

// Template specialization for OptiX's OptixCoopVec - only enabled when cooperative vectors are
// available NVRTC explicitly disables cooperative vectors by setting
// OPTIX_INCLUDE_COOPERATIVE_VECTOR to 0
#if defined(OPTIX_VERSION) && OPTIX_VERSION > 90000
template<typename T, unsigned int N>
struct OptixCoopVecTraits<OptixCoopVec<T, N>>
{
    static constexpr unsigned int size = N;
};
#endif

template<
    typename VecTOut,
    typename VecTIn,
    unsigned inputInterpretation,
    unsigned matrixInterpretation,
    unsigned matrixLayout>
__forceinline__ __device__ VecTOut slangOptixCoopVecMatMul(
    const VecTIn& inputVector,
    CUdeviceptr matrix,
    unsigned matrixOffset,
    bool transpose,
    unsigned matrixStride)
{
    constexpr unsigned N = OptixCoopVecTraits<VecTOut>::size; // Output vector size
    constexpr unsigned K = OptixCoopVecTraits<VecTIn>::size;  // Input vector size

    return optixCoopVecMatMul<
        VecTOut,
        VecTIn,
        SlangToOptixComponentType<inputInterpretation>::value,
        SlangToOptixMatrixLayout<matrixLayout>::value,
        false,
        N,
        K,
        SlangToOptixComponentType<matrixInterpretation>::value>(
        inputVector,
        matrix,
        matrixOffset,
        matrixStride);
}

// OptiX cooperative vector matrix multiplication wrapper (WITH bias - 6 runtime params)
template<
    typename VecTOut,
    typename VecTIn,
    unsigned inputInterpretation,
    unsigned matrixInterpretation,
    unsigned matrixLayout,
    unsigned biasInterpretation>
__forceinline__ __device__ VecTOut slangOptixCoopVecMatMul(
    const VecTIn& inputVector,
    CUdeviceptr matrix,
    unsigned matrixOffset,
    CUdeviceptr bias,
    unsigned biasOffset,
    unsigned matrixStride)
{
    constexpr unsigned N = OptixCoopVecTraits<VecTOut>::size; // Output vector size
    constexpr unsigned K = OptixCoopVecTraits<VecTIn>::size;  // Input vector size

    // Call OptiX SDK with bias (6 runtime parameters)
    return optixCoopVecMatMul<
        VecTOut,
        VecTIn,
        SlangToOptixComponentType<inputInterpretation>::value,
        SlangToOptixMatrixLayout<matrixLayout>::value,
        false,
        N,
        K,
        SlangToOptixComponentType<matrixInterpretation>::value,
        SlangToOptixComponentType<biasInterpretation>::value>(
        inputVector,
        matrix,
        matrixOffset,
        bias,
        biasOffset,
        matrixStride);
}

// OptiX cooperative vector matrix multiplication wrapper (WITHOUT bias, 4 runtime params -
// StructuredBuffer variant)
template<
    typename VecTOut,
    typename VecTIn,
    unsigned inputInterpretation,
    unsigned matrixInterpretation,
    unsigned matrixLayout>
__forceinline__ __device__ VecTOut slangOptixCoopVecMatMul(
    const VecTIn& inputVector,
    CUdeviceptr matrix,
    unsigned matrixOffset,
    unsigned matrixStride)
{
    constexpr unsigned N = OptixCoopVecTraits<VecTOut>::size; // Output vector size
    constexpr unsigned K = OptixCoopVecTraits<VecTIn>::size;  // Input vector size

    // Call OptiX SDK without bias and without transpose (4 runtime parameters)
    return optixCoopVecMatMul<
        VecTOut,
        VecTIn,
        SlangToOptixComponentType<inputInterpretation>::value,
        SlangToOptixMatrixLayout<matrixLayout>::value,
        false,
        N,
        K,
        SlangToOptixComponentType<matrixInterpretation>::value>(
        inputVector,
        matrix,
        matrixOffset,
        matrixStride);
}

#endif // (OPTIX_VERSION >= 90000)

#endif // SLANG_CUDA_ENABLE_OPTIX


// This implementation can only be enabled on CUDA Toolkit 12.5+
#if ((__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5)) || \
    (CUDA_VERSION >= 12050)
// The reason we have to implement our own wmma operation on CUDA is the interface
// design of cooperative_matrix on Vulkan is quite different from CUDA WMMA API, where
// SPIRV spec doesn't require the matrix layout during declaration of the cooperative_matrix,
// instead it is only required during load/store operations. However, in CUDA WMMA API, the layout
// has to be specified during the declaration of the fragment itself. Slang's interface desgin
// is more similar to SPIRV's cooperative_matrix. So to bridge this gap, we have to implement our
// wmma operation by using PTX wmma instructions directly, because PTX wmma instructions is quite
// similar to SPIRV's cooperative_matrix spec.
namespace Slang_CUDA_WMMA
{

// Enums for template specialization
enum MatrixUse : int
{
    MatrixA = 0,
    MatrixB = 1,
    MatrixC = 2,
    MatrixD = 3,
};

enum Layout : int
{
    RowMajor = 0,
    ColMajor = 1
};

enum ShapeCombination : int
{
    m16n16k16 = 0,
    m8n32k16 = 1,
    m32n8k16 = 2
};

// ====================================================================================
// PTX Name Helpers
// ====================================================================================

// Shape names
template<int M, int N, int K>
struct PtxShapeName;
template<>
struct PtxShapeName<16, 16, 16>
{
    static constexpr const char name[] = "m16n16k16";
};
template<>
struct PtxShapeName<8, 32, 16>
{
    static constexpr const char name[] = "m8n32k16";
};
template<>
struct PtxShapeName<32, 8, 16>
{
    static constexpr const char name[] = "m32n8k16";
};

// Matrix role names
template<MatrixUse use>
struct PtxMatrixRoleName;
template<>
struct PtxMatrixRoleName<MatrixUse::MatrixA>
{
    static constexpr const char name[] = "a";
};
template<>
struct PtxMatrixRoleName<MatrixUse::MatrixB>
{
    static constexpr const char name[] = "b";
};
template<>
struct PtxMatrixRoleName<MatrixUse::MatrixC>
{
    static constexpr const char name[] = "c";
};
template<>
struct PtxMatrixRoleName<MatrixUse::MatrixD>
{
    static constexpr const char name[] = "d";
};

// Layout names
template<Layout layout>
struct PtxLayoutName;
template<>
struct PtxLayoutName<Layout::RowMajor>
{
    static constexpr const char name[] = "row";
};
template<>
struct PtxLayoutName<Layout::ColMajor>
{
    static constexpr const char name[] = "col";
};

// Type names
template<typename T>
struct PtxTypeName;

#if SLANG_CUDA_ENABLE_HALF
template<>
struct PtxTypeName<half>
{
    static constexpr const char name[] = "f16";
};
#endif // #if SLANG_CUDA_ENABLE_HALF

#if SLANG_CUDA_ENABLE_FP8
template<>
struct PtxTypeName<__nv_fp8_e4m3>
{
    static constexpr const char name[] = "f8e4m3";
};
template<>
struct PtxTypeName<__nv_fp8_e5m2>
{
    static constexpr const char name[] = "f8e5m2";
};
#endif // #if SLANG_CUDA_ENABLE_FP8

#if SLANG_CUDA_ENABLE_BF16
template<>
struct PtxTypeName<__nv_bfloat16>
{
    static constexpr const char name[] = "bf16";
};
#endif

template<>
struct PtxTypeName<float>
{
    static constexpr const char name[] = "f32";
};
template<>
struct PtxTypeName<char>
{
    static constexpr const char name[] = "s8";
};
template<>
struct PtxTypeName<unsigned char>
{
    static constexpr const char name[] = "u8";
};
template<>
struct PtxTypeName<int32_t>
{
    static constexpr const char name[] = "s32";
};

// ====================================================================================
// Register Counts for different matrices
// ====================================================================================
template<typename ElemT, int M, int N, int K, MatrixUse use>
struct RegisterCount;

#if SLANG_CUDA_ENABLE_HALF
// Half (f16) - 8 regs for A/B, 4 regs for C/D
template<int M, int N, int K>
struct RegisterCount<half, M, N, K, MatrixUse::MatrixA>
{
    static constexpr int value = 8;
};
template<int M, int N, int K>
struct RegisterCount<half, M, N, K, MatrixUse::MatrixB>
{
    static constexpr int value = 8;
};
template<int M, int N, int K>
struct RegisterCount<half, M, N, K, MatrixUse::MatrixC>
{
    static constexpr int value = 4;
};
template<int M, int N, int K>
struct RegisterCount<half, M, N, K, MatrixUse::MatrixD>
{
    static constexpr int value = 4;
};
#endif // #if SLANG_CUDA_ENABLE_HALF

#if SLANG_CUDA_ENABLE_BF16
// bfloat16 - 8 regs for A/B, 4 regs for C/D
template<int M, int N, int K>
struct RegisterCount<__nv_bfloat16, M, N, K, MatrixUse::MatrixA>
{
    static constexpr int value = 8;
};
template<int M, int N, int K>
struct RegisterCount<__nv_bfloat16, M, N, K, MatrixUse::MatrixB>
{
    static constexpr int value = 8;
};
template<int M, int N, int K>
struct RegisterCount<__nv_bfloat16, M, N, K, MatrixUse::MatrixC>
{
    static constexpr int value = 4;
};
template<int M, int N, int K>
struct RegisterCount<__nv_bfloat16, M, N, K, MatrixUse::MatrixD>
{
    static constexpr int value = 4;
};
#endif // #if SLANG_CUDA_ENABLE_BF16

// Float (f32) - 8 regs for C/D only
template<int M, int N, int K>
struct RegisterCount<float, M, N, K, MatrixUse::MatrixC>
{
    static constexpr int value = 8;
};
template<int M, int N, int K>
struct RegisterCount<float, M, N, K, MatrixUse::MatrixD>
{
    static constexpr int value = 8;
};

// Int32 (s32) - 8 regs for C/D (accumulator for int8 operations)
template<int M, int N, int K>
struct RegisterCount<int32_t, M, N, K, MatrixUse::MatrixC>
{
    static constexpr int value = 8;
};
template<int M, int N, int K>
struct RegisterCount<int32_t, M, N, K, MatrixUse::MatrixD>
{
    static constexpr int value = 8;
};

// Uint8 (u8) - varies by shape
template<>
struct RegisterCount<unsigned char, 16, 16, 16, MatrixUse::MatrixA>
{
    static constexpr int value = 2;
};
template<>
struct RegisterCount<unsigned char, 16, 16, 16, MatrixUse::MatrixB>
{
    static constexpr int value = 2;
};
template<>
struct RegisterCount<unsigned char, 8, 32, 16, MatrixUse::MatrixA>
{
    static constexpr int value = 1;
};
template<>
struct RegisterCount<unsigned char, 8, 32, 16, MatrixUse::MatrixB>
{
    static constexpr int value = 4;
};
template<>
struct RegisterCount<unsigned char, 32, 8, 16, MatrixUse::MatrixA>
{
    static constexpr int value = 4;
};
template<>
struct RegisterCount<unsigned char, 32, 8, 16, MatrixUse::MatrixB>
{
    static constexpr int value = 1;
};

// Int8 (s8) - same as u8
template<>
struct RegisterCount<char, 16, 16, 16, MatrixUse::MatrixA>
{
    static constexpr int value = 2;
};
template<>
struct RegisterCount<char, 16, 16, 16, MatrixUse::MatrixB>
{
    static constexpr int value = 2;
};
template<>
struct RegisterCount<char, 8, 32, 16, MatrixUse::MatrixA>
{
    static constexpr int value = 1;
};
template<>
struct RegisterCount<char, 8, 32, 16, MatrixUse::MatrixB>
{
    static constexpr int value = 4;
};
template<>
struct RegisterCount<char, 32, 8, 16, MatrixUse::MatrixA>
{
    static constexpr int value = 4;
};
template<>
struct RegisterCount<char, 32, 8, 16, MatrixUse::MatrixB>
{
    static constexpr int value = 1;
};

#if SLANG_CUDA_ENABLE_FP8
// fp8 - same as u8
template<>
struct RegisterCount<__nv_fp8_e4m3, 16, 16, 16, MatrixUse::MatrixA>
{
    static constexpr int value = 2;
};
template<>
struct RegisterCount<__nv_fp8_e4m3, 16, 16, 16, MatrixUse::MatrixB>
{
    static constexpr int value = 2;
};
template<>
struct RegisterCount<__nv_fp8_e4m3, 8, 32, 16, MatrixUse::MatrixA>
{
    static constexpr int value = 1;
};
template<>
struct RegisterCount<__nv_fp8_e4m3, 8, 32, 16, MatrixUse::MatrixB>
{
    static constexpr int value = 4;
};
template<>
struct RegisterCount<__nv_fp8_e4m3, 32, 8, 16, MatrixUse::MatrixA>
{
    static constexpr int value = 4;
};
template<>
struct RegisterCount<__nv_fp8_e4m3, 32, 8, 16, MatrixUse::MatrixB>
{
    static constexpr int value = 1;
};

template<>
struct RegisterCount<__nv_fp8_e5m2, 16, 16, 16, MatrixUse::MatrixA>
{
    static constexpr int value = 2;
};
template<>
struct RegisterCount<__nv_fp8_e5m2, 16, 16, 16, MatrixUse::MatrixB>
{
    static constexpr int value = 2;
};
template<>
struct RegisterCount<__nv_fp8_e5m2, 8, 32, 16, MatrixUse::MatrixA>
{
    static constexpr int value = 1;
};
template<>
struct RegisterCount<__nv_fp8_e5m2, 8, 32, 16, MatrixUse::MatrixB>
{
    static constexpr int value = 4;
};
template<>
struct RegisterCount<__nv_fp8_e5m2, 32, 8, 16, MatrixUse::MatrixA>
{
    static constexpr int value = 4;
};
template<>
struct RegisterCount<__nv_fp8_e5m2, 32, 8, 16, MatrixUse::MatrixB>
{
    static constexpr int value = 1;
};
#endif


// ====================================================================================
// Saturation at the output for integer MMA
// ====================================================================================
template<bool saturatingAccumulation>
struct IsSaturated;

template<>
struct IsSaturated<true>
{
    static constexpr const char name[] = ".satfinite";
};

template<>
struct IsSaturated<false>
{
    static constexpr const char name[] = "";
};

// ====================================================================================
// WMMA Load - Inline PTX
// ====================================================================================

template<typename ElemT, int M, int N, int K, MatrixUse use, Layout layout>
__device__ inline void wmmaLoad(uint32_t* regs, const void* ptr, int stride)
{
    constexpr int nregs = RegisterCount<ElemT, M, N, K, use>::value;

    switch (nregs)
    {
    case 1:
        asm volatile("wmma.load.%1.sync.aligned.%2.%3.%4 {%0}, [%5], %6;\n"
                     : "=r"(regs[0])
                     : "C"(PtxMatrixRoleName<use>::name),
                       "C"(PtxLayoutName<layout>::name),
                       "C"(PtxShapeName<M, N, K>::name),
                       "C"(PtxTypeName<ElemT>::name),
                       "l"(ptr),
                       "r"(stride));
        break;

    case 2:
        asm volatile("wmma.load.%2.sync.aligned.%3.%4.%5 {%0, %1}, [%6], %7;\n"
                     : "=r"(regs[0]), "=r"(regs[1])
                     : "C"(PtxMatrixRoleName<use>::name),
                       "C"(PtxLayoutName<layout>::name),
                       "C"(PtxShapeName<M, N, K>::name),
                       "C"(PtxTypeName<ElemT>::name),
                       "l"(ptr),
                       "r"(stride));
        break;

    case 4:
        asm volatile("wmma.load.%4.sync.aligned.%5.%6.%7 {%0, %1, %2, %3}, [%8], %9;\n"
                     : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
                     : "C"(PtxMatrixRoleName<use>::name),
                       "C"(PtxLayoutName<layout>::name),
                       "C"(PtxShapeName<M, N, K>::name),
                       "C"(PtxTypeName<ElemT>::name),
                       "l"(ptr),
                       "r"(stride));
        break;

    case 8:
        asm volatile("wmma.load.%8.sync.aligned.%9.%10.%11 "
                     "{%0, %1, %2, %3, %4, %5, %6, %7}, [%12], %13;\n"
                     : "=r"(regs[0]),
                       "=r"(regs[1]),
                       "=r"(regs[2]),
                       "=r"(regs[3]),
                       "=r"(regs[4]),
                       "=r"(regs[5]),
                       "=r"(regs[6]),
                       "=r"(regs[7])
                     : "C"(PtxMatrixRoleName<use>::name),
                       "C"(PtxLayoutName<layout>::name),
                       "C"(PtxShapeName<M, N, K>::name),
                       "C"(PtxTypeName<ElemT>::name),
                       "l"(ptr),
                       "r"(stride));
        break;
    }
}

// ====================================================================================
// WMMA Store - Inline PTX
// ====================================================================================

template<typename ElemT, int M, int N, int K, Layout layout>
__device__ inline void wmmaStore(void* ptr, const uint32_t* regs, int stride)
{
    constexpr int nregs = RegisterCount<ElemT, M, N, K, MatrixUse::MatrixD>::value;

    switch (nregs)
    {
    case 4:
        asm volatile("wmma.store.d.sync.aligned.%0.%1.%2 [%3], {%4, %5, %6, %7}, %8;\n"
                     :
                     : "C"(PtxLayoutName<layout>::name),
                       "C"(PtxShapeName<M, N, K>::name),
                       "C"(PtxTypeName<ElemT>::name),
                       "l"(ptr),
                       "r"(regs[0]),
                       "r"(regs[1]),
                       "r"(regs[2]),
                       "r"(regs[3]),
                       "r"(stride));
        break;

    case 8:
        asm volatile("wmma.store.d.sync.aligned.%0.%1.%2 "
                     "[%3], {%4, %5, %6, %7, %8, %9, %10, %11}, %12;\n"
                     :
                     : "C"(PtxLayoutName<layout>::name),
                       "C"(PtxShapeName<M, N, K>::name),
                       "C"(PtxTypeName<ElemT>::name),
                       "l"(ptr),
                       "r"(regs[0]),
                       "r"(regs[1]),
                       "r"(regs[2]),
                       "r"(regs[3]),
                       "r"(regs[4]),
                       "r"(regs[5]),
                       "r"(regs[6]),
                       "r"(regs[7]),
                       "r"(stride));
        break;
    }
}

// Helper to get M, N, K from ShapeCombination
template<ShapeCombination shape>
struct ShapeToMNK;
template<>
struct ShapeToMNK<ShapeCombination::m16n16k16>
{
    static constexpr int M = 16, N = 16, K = 16;
};
template<>
struct ShapeToMNK<ShapeCombination::m8n32k16>
{
    static constexpr int M = 8, N = 32, K = 16;
};
template<>
struct ShapeToMNK<ShapeCombination::m32n8k16>
{
    static constexpr int M = 32, N = 8, K = 16;
};

template<typename T>
inline unsigned __device__ Pack32Helper(T value);

#if SLANG_CUDA_ENABLE_HALF
template<>
inline unsigned __device__ Pack32Helper<half>(half value)
{
    return __half_as_ushort(value) | (__half_as_ushort(value) << 16);
};
#endif

template<>
inline unsigned __device__ Pack32Helper<float>(float value)
{
    return __float_as_uint(value);
};

template<>
inline unsigned __device__ Pack32Helper<int>(int value)
{
    return (unsigned)value;
};
template<>
inline unsigned __device__ Pack32Helper<char>(char value)
{
    return value << 24 | value << 16 | value << 8 | value;
};
template<>
inline unsigned __device__ Pack32Helper<unsigned char>(unsigned char value)
{
    return value << 24 | value << 16 | value << 8 | value;
};


// The dimensions of the fragment are specified by M, N, K which are totally determined during
// compile time, so slang already did the pre-filter on the shape & type combination.
template<typename T, int M, int N, int K, MatrixUse R>
struct WmmaFragment
{
    __device__ WmmaFragment() {}
    __device__ WmmaFragment(T scalarValue) { fill(scalarValue); }

    typedef WmmaFragment<T, M, N, K, R> This;
    template<Layout layout>
    void __device__ Store(RWStructuredBuffer<T> buffer, uint element, uint stride)
    {
        Store<layout>(buffer.data, element, stride);
    }

    template<Layout layout>
    static This __device__ Load(StructuredBuffer<T> buffer, uint element, uint stride)
    {
        return Load<layout>(buffer.data, element, stride);
    }

    // There is no fill intrinsic in PTX wmma, so it's just 'move' value
    // to the fragment registers.
    void __device__ fill(T value)
    {
        unsigned packed = Pack32Helper(value);
        constexpr int nregs = RegisterCount<T, M, N, K, R>::value;
        for (int i = 0; i < nregs; i++)
        {
            regs[i] = packed;
        }
    }

    __device__ This operator*(T b)
    {
        This result;

        // This loop will be unrolled by the compiler becuase nregs is constexpr
        for (int i = 0; i < GetLength(); i++)
        {
            result.set(i, get(i) * b);
        }
        return result;
    }

    __device__ This operator*(const This& b)
    {
        This result;

        // This loop will be unrolled by the compiler becuase nregs is constexpr
        for (int i = 0; i < GetLength(); i++)
        {
            result.set(i, get(i) * b.get(i));
        }
        return result;
    }

    __device__ This operator/(const This& other)
    {
        This result;

        for (int i = 0; i < GetLength(); i++)
        {
            result.set(i, get(i) / other.get(i));
        }
        return result;
    }

    __device__ This operator-(const This& other)
    {
        This result;

        for (int i = 0; i < GetLength(); i++)
        {
            result.set(i, get(i) - other.get(i));
        }
        return result;
    }

    __device__ This operator-()
    {
        This result;

        for (int i = 0; i < GetLength(); i++)
        {
            result.set(i, -get(i));
        }
        return result;
    }

    __device__ This operator+(const This& other)
    {
        This result;

        for (int i = 0; i < GetLength(); i++)
        {
            result.set(i, get(i) + other.get(i));
        }
        return result;
    }

    __device__ This operator%(const This& other)
    {
        This result;

        for (int i = 0; i < GetLength(); i++)
        {
            result.set(i, get(i) % other.get(i));
        }
        return result;
    }

    template<typename U>
    __device__ void copyFrom(const WmmaFragment<U, M, N, K, R>& other)
    {
        // If the data type is different, we need to copy element by element.
        // Since the shape of two matrices are the same, they have the same
        // number of elements.
        for (int i = 0; i < GetLength(); i++)
        {
            set(i, static_cast<T>(other.get(i)));
        }
    }

    // Get element by index (handles bit-level access for packed types)
    // For example: u8/s8 matrices have 4 elements per register (32-bit)
    //   - index 0: bits [0:7]   of regs[0]
    //   - index 1: bits [8:15]  of regs[0]
    //   - index 2: bits [16:23] of regs[0]
    //   - index 3: bits [24:31] of regs[0]
    __device__ T get(int index) const
    {
        if constexpr (sizeof(T) == 4)
        {
            // T is 32-bit (float or int32): 1 element per register
            T v;
            memcpy(&v, &regs[index], 4);
            return v;
        }
        else if constexpr (sizeof(T) == 2)
        {
            // T is 16-bit (half): 2 elements per register
            // Elements per register: [0:15] and [16:31]
            int regIndex = index / 2;
            int elementOffset = index % 2;
            int bitOffset = elementOffset * 16;
            uint32_t extracted = (regs[regIndex] >> bitOffset) & 0xFFFF;
            uint16_t value16 = static_cast<uint16_t>(extracted);
            T v;
            memcpy(&v, &value16, 2);
            return v;
        }
        else if constexpr (sizeof(T) == 1)
        {
            // T is 8-bit (int8_t, uint8_t): 4 elements per register
            // Elements per register: [0:7], [8:15], [16:23], [24:31]
            int regIndex = index / 4;
            int elementOffset = index % 4;
            int bitOffset = elementOffset * 8;
            uint32_t extracted = (regs[regIndex] >> bitOffset) & 0xFF;
            uint8_t value8 = static_cast<uint8_t>(extracted);
            return *reinterpret_cast<const T*>(&value8);
        }
    }

    // Set element by index (handles bit-level access for packed types)
    __device__ void set(int index, T value)
    {
        if constexpr (sizeof(T) == 4)
        {
            // T is 32-bit (float or int32): 1 element per register
            memcpy(&regs[index], &value, 4);
        }
        else if constexpr (sizeof(T) == 2)
        {
            // T is 16-bit (half): 2 elements per register
            int regIndex = index / 2;
            int elementOffset = index % 2;
            int bitOffset = elementOffset * 16;
            uint32_t mask = 0xFFFF;
            uint16_t value16;
            memcpy(&value16, &value, 2);

            // Clear the bits at the target position
            regs[regIndex] &= ~(mask << bitOffset);

            // Set the new value
            regs[regIndex] |= (static_cast<uint32_t>(value16) << bitOffset);
        }
        else if constexpr (sizeof(T) == 1)
        {
            // T is 8-bit (int8_t, uint8_t): 4 elements per register
            int regIndex = index / 4;
            int elementOffset = index % 4;
            int bitOffset = elementOffset * 8;
            uint32_t mask = 0xFF;
            uint8_t value8 = *reinterpret_cast<const uint8_t*>(&value);

            // Clear the bits at the target position
            regs[regIndex] &= ~(mask << bitOffset);

            // Set the new value
            regs[regIndex] |= (static_cast<uint32_t>(value8) << bitOffset);
        }
    }

    template<Layout layout>
    void __device__ Store(T* buffer, uint element, uint stride)
    {
        // Force compile-time check, so we know the template parameter comibination is valid.
        (void)RegisterCount<T, M, N, K, R>::value;
        wmmaStore<T, M, N, K, layout>(buffer + element, regs, stride);
    }

    template<Layout layout, typename U>
    void __device__ Store(U* buffer, uint stride)
    {
        // Force compile-time check, so we know the template parameter comibination is valid.
        (void)RegisterCount<T, M, N, K, R>::value;
        wmmaStore<T, M, N, K, layout>(buffer, regs, stride * sizeof(U) / sizeof(T));
    }

    template<Layout layout>
    static This __device__ Load(T* buffer, uint element, uint stride)
    {
        WmmaFragment<T, M, N, K, R> fragment;

        // Force compile-time check, so we know the template parameter comibination is valid.
        (void)RegisterCount<T, M, N, K, R>::value;
        wmmaLoad<T, M, N, K, R, layout>(fragment.regs, buffer + element, stride);
        fragment.m_layout = layout;
        return fragment;
    }

    template<Layout layout, typename U>
    static This __device__ Load(U* buffer, uint stride)
    {
        WmmaFragment<T, M, N, K, R> fragment;

        // Force compile-time check, so we know the template parameter comibination is valid.
        (void)RegisterCount<T, M, N, K, R>::value;
        wmmaLoad<T, M, N, K, R, layout>(fragment.regs, buffer, stride * sizeof(U) / sizeof(T));
        fragment.m_layout = layout;
        return fragment;
    }

    static constexpr __device__ uint32_t GetLength() { return This::elements_per_thread; }

    // For referencing those template parameters outside the struct
    using ElementType = T;
    static constexpr int m_M = M;
    static constexpr int m_N = N;
    static constexpr int m_K = K;
    Layout m_layout = Layout::RowMajor;

    // Register Count requirement
    static constexpr int RegsCount = RegisterCount<T, M, N, K, R>::value;
    unsigned regs[RegsCount] = {};

    static constexpr uint32_t elements_per_thread = RegsCount * (4 / sizeof(T));
};

// ====================================================================================
// FP16 MMA Helper - For half x half inputs
// Specialized on CType and DType (accumulator types)
//
// PTX Syntax: wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype d, a, b, c;
//   where:
//     dtype = type of d (output accumulator): {.f16, .f32}
//     ctype = type of c (input accumulator):  {.f16, .f32}
//
// Note: Types of a and b are implicitly f16 (not specified in PTX instruction).
//       Shape (M, N, K) is passed as template parameters, so one template handles all shapes.
//       We only need to specialize on CType and DType.
// ====================================================================================

template<typename CType, typename DType, int M, int N, int K, Layout LayoutA, Layout LayoutB>
struct Fp16MMAHelper;

#if SLANG_CUDA_ENABLE_HALF
// Specialization: c=half, d=half (f16.f16)
template<int M, int N, int K, Layout LayoutA, Layout LayoutB>
struct Fp16MMAHelper<half, half, M, N, K, LayoutA, LayoutB>
{
    __device__ static void eval(
        WmmaFragment<half, M, N, K, MatrixC>& d,
        const WmmaFragment<half, M, N, K, MatrixUse::MatrixA>& a,
        const WmmaFragment<half, M, N, K, MatrixUse::MatrixB>& b,
        const WmmaFragment<half, M, N, K, MatrixUse::MatrixC>& c)
    {
        asm volatile("wmma.mma.sync.aligned.%4.%5.%6.%7.%8 "
                     "{%0, %1, %2, %3}, "
                     "{%9, %10, %11, %12, %13, %14, %15, %16}, "
                     "{%17, %18, %19, %20, %21, %22, %23, %24}, "
                     "{%25, %26, %27, %28};\n"
                     : "=r"(d.regs[0]), "=r"(d.regs[1]), "=r"(d.regs[2]), "=r"(d.regs[3])
                     : "C"(PtxLayoutName<LayoutA>::name),
                       "C"(PtxLayoutName<LayoutB>::name),
                       "C"(PtxShapeName<M, N, K>::name),
                       "C"(PtxTypeName<half>::name),
                       "C"(PtxTypeName<half>::name),
                       "r"(a.regs[0]),
                       "r"(a.regs[1]),
                       "r"(a.regs[2]),
                       "r"(a.regs[3]),
                       "r"(a.regs[4]),
                       "r"(a.regs[5]),
                       "r"(a.regs[6]),
                       "r"(a.regs[7]),
                       "r"(b.regs[0]),
                       "r"(b.regs[1]),
                       "r"(b.regs[2]),
                       "r"(b.regs[3]),
                       "r"(b.regs[4]),
                       "r"(b.regs[5]),
                       "r"(b.regs[6]),
                       "r"(b.regs[7]),
                       "r"(c.regs[0]),
                       "r"(c.regs[1]),
                       "r"(c.regs[2]),
                       "r"(c.regs[3]));
    }
};

// Specialization: c=float, d=half (f16.f32)
template<int M, int N, int K, Layout LayoutA, Layout LayoutB>
struct Fp16MMAHelper<float, half, M, N, K, LayoutA, LayoutB>
{
    __device__ static void eval(
        WmmaFragment<half, M, N, K, MatrixUse::MatrixC>& d,
        const WmmaFragment<half, M, N, K, MatrixUse::MatrixA>& a,
        const WmmaFragment<half, M, N, K, MatrixUse::MatrixB>& b,
        const WmmaFragment<float, M, N, K, MatrixUse::MatrixC>& c)
    {
        asm volatile("wmma.mma.sync.aligned.%4.%5.%6.%7.%8 "
                     "{%0, %1, %2, %3}, "
                     "{%9, %10, %11, %12, %13, %14, %15, %16}, "
                     "{%17, %18, %19, %20, %21, %22, %23, %24}, "
                     "{%25, %26, %27, %28, %29, %30, %31, %32};\n"
                     : "=r"(d.regs[0]), "=r"(d.regs[1]), "=r"(d.regs[2]), "=r"(d.regs[3])
                     : "C"(PtxLayoutName<LayoutA>::name),
                       "C"(PtxLayoutName<LayoutB>::name),
                       "C"(PtxShapeName<M, N, K>::name),
                       "C"(PtxTypeName<half>::name),
                       "C"(PtxTypeName<float>::name),
                       "r"(a.regs[0]),
                       "r"(a.regs[1]),
                       "r"(a.regs[2]),
                       "r"(a.regs[3]),
                       "r"(a.regs[4]),
                       "r"(a.regs[5]),
                       "r"(a.regs[6]),
                       "r"(a.regs[7]),
                       "r"(b.regs[0]),
                       "r"(b.regs[1]),
                       "r"(b.regs[2]),
                       "r"(b.regs[3]),
                       "r"(b.regs[4]),
                       "r"(b.regs[5]),
                       "r"(b.regs[6]),
                       "r"(b.regs[7]),
                       "r"(c.regs[0]),
                       "r"(c.regs[1]),
                       "r"(c.regs[2]),
                       "r"(c.regs[3]),
                       "r"(c.regs[4]),
                       "r"(c.regs[5]),
                       "r"(c.regs[6]),
                       "r"(c.regs[7]));
    }
};

// Specialization: c=half, d=float (f32.f16)
template<int M, int N, int K, Layout LayoutA, Layout LayoutB>
struct Fp16MMAHelper<half, float, M, N, K, LayoutA, LayoutB>
{
    __device__ static void eval(
        WmmaFragment<float, M, N, K, MatrixUse::MatrixC>& d,
        const WmmaFragment<half, M, N, K, MatrixUse::MatrixA>& a,
        const WmmaFragment<half, M, N, K, MatrixUse::MatrixB>& b,
        const WmmaFragment<half, M, N, K, MatrixUse::MatrixC>& c)
    {
        asm volatile("wmma.mma.sync.aligned.%8.%9.%10.%11.%12 "
                     "{%0, %1, %2, %3, %4, %5, %6, %7}, "
                     "{%13, %14, %15, %16, %17, %18, %19, %20}, "
                     "{%21, %22, %23, %24, %25, %26, %27, %28}, "
                     "{%29, %30, %31, %32};\n"
                     : "=r"(d.regs[0]),
                       "=r"(d.regs[1]),
                       "=r"(d.regs[2]),
                       "=r"(d.regs[3]),
                       "=r"(d.regs[4]),
                       "=r"(d.regs[5]),
                       "=r"(d.regs[6]),
                       "=r"(d.regs[7])
                     : "C"(PtxLayoutName<LayoutA>::name),
                       "C"(PtxLayoutName<LayoutB>::name),
                       "C"(PtxShapeName<M, N, K>::name),
                       "C"(PtxTypeName<float>::name),
                       "C"(PtxTypeName<half>::name),
                       "r"(a.regs[0]),
                       "r"(a.regs[1]),
                       "r"(a.regs[2]),
                       "r"(a.regs[3]),
                       "r"(a.regs[4]),
                       "r"(a.regs[5]),
                       "r"(a.regs[6]),
                       "r"(a.regs[7]),
                       "r"(b.regs[0]),
                       "r"(b.regs[1]),
                       "r"(b.regs[2]),
                       "r"(b.regs[3]),
                       "r"(b.regs[4]),
                       "r"(b.regs[5]),
                       "r"(b.regs[6]),
                       "r"(b.regs[7]),
                       "r"(c.regs[0]),
                       "r"(c.regs[1]),
                       "r"(c.regs[2]),
                       "r"(c.regs[3]));
    }
};

// Specialization: c=float, d=float (f32.f32)
template<int M, int N, int K, Layout LayoutA, Layout LayoutB>
struct Fp16MMAHelper<float, float, M, N, K, LayoutA, LayoutB>
{
    __device__ static void eval(
        WmmaFragment<float, M, N, K, MatrixUse::MatrixC>& d,
        const WmmaFragment<half, M, N, K, MatrixUse::MatrixA>& a,
        const WmmaFragment<half, M, N, K, MatrixUse::MatrixB>& b,
        const WmmaFragment<float, M, N, K, MatrixUse::MatrixC>& c)
    {
        asm volatile("wmma.mma.sync.aligned.%8.%9.%10.%11.%12 "
                     "{%0, %1, %2, %3, %4, %5, %6, %7}, "
                     "{%13, %14, %15, %16, %17, %18, %19, %20}, "
                     "{%21, %22, %23, %24, %25, %26, %27, %28}, "
                     "{%29, %30, %31, %32, %33, %34, %35, %36};\n"
                     : "=r"(d.regs[0]),
                       "=r"(d.regs[1]),
                       "=r"(d.regs[2]),
                       "=r"(d.regs[3]),
                       "=r"(d.regs[4]),
                       "=r"(d.regs[5]),
                       "=r"(d.regs[6]),
                       "=r"(d.regs[7])
                     : "C"(PtxLayoutName<LayoutA>::name),
                       "C"(PtxLayoutName<LayoutB>::name),
                       "C"(PtxShapeName<M, N, K>::name),
                       "C"(PtxTypeName<float>::name),
                       "C"(PtxTypeName<float>::name),
                       "r"(a.regs[0]),
                       "r"(a.regs[1]),
                       "r"(a.regs[2]),
                       "r"(a.regs[3]),
                       "r"(a.regs[4]),
                       "r"(a.regs[5]),
                       "r"(a.regs[6]),
                       "r"(a.regs[7]),
                       "r"(b.regs[0]),
                       "r"(b.regs[1]),
                       "r"(b.regs[2]),
                       "r"(b.regs[3]),
                       "r"(b.regs[4]),
                       "r"(b.regs[5]),
                       "r"(b.regs[6]),
                       "r"(b.regs[7]),
                       "r"(c.regs[0]),
                       "r"(c.regs[1]),
                       "r"(c.regs[2]),
                       "r"(c.regs[3]),
                       "r"(c.regs[4]),
                       "r"(c.regs[5]),
                       "r"(c.regs[6]),
                       "r"(c.regs[7]));
    }
};
#endif // #if SLANG_CUDA_ENABLE_HALF

// ====================================================================================
// Integer MMA Helper - For int8/uint8 inputs
// Specialized on shape (register counts depend on shape)
//
// PTX Syntax: wmma.mma.sync.aligned.alayout.blayout.shape.s32.atype.btype.s32{.satfinite} d, a, b,
// c;
//   where:
//     atype = type of a (input matrix A): {.s8, .u8}
//     btype = type of b (input matrix B): {.s8, .u8}
//     C and D are always s32 (int32)
//
// Note: Unlike FP16, integer operations explicitly specify atype and btype in the instruction.
//       We must specialize on shape because register counts vary:
//         m16n16k16: a=2 regs, b=2 regs
//         m8n32k16:  a=1 reg,  b=4 regs
//         m32n8k16:  a=4 regs, b=1 reg
//       C and D always use 8 registers (int32).
// ====================================================================================

template<
    typename AType,
    typename BType,
    ShapeCombination shape,
    Layout LayoutA,
    Layout LayoutB,
    bool saturatingAccumulation>
struct IntegerMMAHelper;

// Specialization: m16n16k16 (a=2 regs, b=2 regs)
template<
    typename AType,
    typename BType,
    Layout LayoutA,
    Layout LayoutB,
    bool saturatingAccumulation>
struct IntegerMMAHelper<
    AType,
    BType,
    ShapeCombination::m16n16k16,
    LayoutA,
    LayoutB,
    saturatingAccumulation>
{
    __device__ static void eval(
        WmmaFragment<int, 16, 16, 16, MatrixUse::MatrixC>& d,
        const WmmaFragment<AType, 16, 16, 16, MatrixUse::MatrixA>& a,
        const WmmaFragment<BType, 16, 16, 16, MatrixUse::MatrixB>& b,
        const WmmaFragment<int, 16, 16, 16, MatrixUse::MatrixC>& c)
    {
        asm volatile("wmma.mma.sync.aligned.%8.%9.%10.s32.%11.%12.s32%13 "
                     "{%0, %1, %2, %3, %4, %5, %6, %7}, "
                     "{%14, %15}, "
                     "{%16, %17}, "
                     "{%18, %19, %20, %21, %22, %23, %24, %25};\n"
                     : "=r"(d.regs[0]),
                       "=r"(d.regs[1]),
                       "=r"(d.regs[2]),
                       "=r"(d.regs[3]),
                       "=r"(d.regs[4]),
                       "=r"(d.regs[5]),
                       "=r"(d.regs[6]),
                       "=r"(d.regs[7])
                     : "C"(PtxLayoutName<LayoutA>::name),
                       "C"(PtxLayoutName<LayoutB>::name),
                       "C"(PtxShapeName<16, 16, 16>::name),
                       "C"(PtxTypeName<AType>::name),
                       "C"(PtxTypeName<BType>::name),
                       "C"(IsSaturated<saturatingAccumulation>::name),
                       "r"(a.regs[0]),
                       "r"(a.regs[1]),
                       "r"(b.regs[0]),
                       "r"(b.regs[1]),
                       "r"(c.regs[0]),
                       "r"(c.regs[1]),
                       "r"(c.regs[2]),
                       "r"(c.regs[3]),
                       "r"(c.regs[4]),
                       "r"(c.regs[5]),
                       "r"(c.regs[6]),
                       "r"(c.regs[7]));
    }
};

// Specialization: m8n32k16 (a=1 reg, b=4 regs)
template<
    typename AType,
    typename BType,
    Layout LayoutA,
    Layout LayoutB,
    bool saturatingAccumulation>
struct IntegerMMAHelper<
    AType,
    BType,
    ShapeCombination::m8n32k16,
    LayoutA,
    LayoutB,
    saturatingAccumulation>
{
    __device__ static void eval(
        WmmaFragment<int, 8, 32, 16, MatrixUse::MatrixC>& d,
        const WmmaFragment<AType, 8, 32, 16, MatrixUse::MatrixA>& a,
        const WmmaFragment<BType, 8, 32, 16, MatrixUse::MatrixB>& b,
        const WmmaFragment<int, 8, 32, 16, MatrixUse::MatrixC>& c)
    {
        asm volatile("wmma.mma.sync.aligned.%8.%9.%10.s32.%11.%12.s32%13 "
                     "{%0, %1, %2, %3, %4, %5, %6, %7}, "
                     "{%14}, "
                     "{%15, %16, %17, %18}, "
                     "{%19, %20, %21, %22, %23, %24, %25, %26};\n"
                     : "=r"(d.regs[0]),
                       "=r"(d.regs[1]),
                       "=r"(d.regs[2]),
                       "=r"(d.regs[3]),
                       "=r"(d.regs[4]),
                       "=r"(d.regs[5]),
                       "=r"(d.regs[6]),
                       "=r"(d.regs[7])
                     : "C"(PtxLayoutName<LayoutA>::name),
                       "C"(PtxLayoutName<LayoutB>::name),
                       "C"(PtxShapeName<8, 32, 16>::name),
                       "C"(PtxTypeName<AType>::name),
                       "C"(PtxTypeName<BType>::name),
                       "C"(IsSaturated<saturatingAccumulation>::name),
                       "r"(a.regs[0]),
                       "r"(b.regs[0]),
                       "r"(b.regs[1]),
                       "r"(b.regs[2]),
                       "r"(b.regs[3]),
                       "r"(c.regs[0]),
                       "r"(c.regs[1]),
                       "r"(c.regs[2]),
                       "r"(c.regs[3]),
                       "r"(c.regs[4]),
                       "r"(c.regs[5]),
                       "r"(c.regs[6]),
                       "r"(c.regs[7]));
    }
};

// Specialization: m32n8k16 (a=4 regs, b=1 reg)
template<
    typename AType,
    typename BType,
    Layout LayoutA,
    Layout LayoutB,
    bool saturatingAccumulation>
struct IntegerMMAHelper<
    AType,
    BType,
    ShapeCombination::m32n8k16,
    LayoutA,
    LayoutB,
    saturatingAccumulation>
{
    __device__ static void eval(
        WmmaFragment<int, 32, 8, 16, MatrixUse::MatrixC>& d,
        const WmmaFragment<AType, 32, 8, 16, MatrixUse::MatrixA>& a,
        const WmmaFragment<BType, 32, 8, 16, MatrixUse::MatrixB>& b,
        const WmmaFragment<int, 32, 8, 16, MatrixUse::MatrixC>& c)
    {
        asm volatile("wmma.mma.sync.aligned.%8.%9.%10.s32.%11.%12.s32%13 "
                     "{%0, %1, %2, %3, %4, %5, %6, %7}, "
                     "{%14, %15, %16, %17}, "
                     "{%18}, "
                     "{%19, %20, %21, %22, %23, %24, %25, %26};\n"
                     : "=r"(d.regs[0]),
                       "=r"(d.regs[1]),
                       "=r"(d.regs[2]),
                       "=r"(d.regs[3]),
                       "=r"(d.regs[4]),
                       "=r"(d.regs[5]),
                       "=r"(d.regs[6]),
                       "=r"(d.regs[7])
                     : "C"(PtxLayoutName<LayoutA>::name),
                       "C"(PtxLayoutName<LayoutB>::name),
                       "C"(PtxShapeName<32, 8, 16>::name),
                       "C"(PtxTypeName<AType>::name),
                       "C"(PtxTypeName<BType>::name),
                       "C"(IsSaturated<saturatingAccumulation>::name),
                       "r"(a.regs[0]),
                       "r"(a.regs[1]),
                       "r"(a.regs[2]),
                       "r"(a.regs[3]),
                       "r"(b.regs[0]),
                       "r"(c.regs[0]),
                       "r"(c.regs[1]),
                       "r"(c.regs[2]),
                       "r"(c.regs[3]),
                       "r"(c.regs[4]),
                       "r"(c.regs[5]),
                       "r"(c.regs[6]),
                       "r"(c.regs[7]));
    }
};


// ====================================================================================
// MMA Helper - Primary Template (dispatcher)
// ====================================================================================

template<
    typename AType,
    typename BType,
    typename CType,
    typename DType,
    ShapeCombination shape,
    Layout LayoutA,
    Layout LayoutB,
    bool saturatingAccumulation>
struct MMAHelper
{
    static constexpr int M = ShapeToMNK<shape>::M;
    static constexpr int N = ShapeToMNK<shape>::N;
    static constexpr int K = ShapeToMNK<shape>::K;

    __device__ static void eval(
        WmmaFragment<DType, M, N, K, MatrixUse::MatrixC>& d,
        const WmmaFragment<AType, M, N, K, MatrixUse::MatrixA>& a,
        const WmmaFragment<BType, M, N, K, MatrixUse::MatrixB>& b,
        const WmmaFragment<CType, M, N, K, MatrixUse::MatrixC>& c,
        bool saturate = false)
    {
        // Dispatch to appropriate helper based on input types
        if constexpr (sizeof(AType) == 2 && sizeof(BType) == 2)
        {
            // FP16 inputs: dispatch to Fp16MMAHelper
            Fp16MMAHelper<CType, DType, M, N, K, LayoutA, LayoutB>::eval(d, a, b, c);
        }
        else
        {
            // Integer inputs (int8/uint8): dispatch to IntegerMMAHelper
            IntegerMMAHelper<AType, BType, shape, LayoutA, LayoutB, saturatingAccumulation>::eval(
                d,
                a,
                b,
                c);
        }
    }
};

//
template<
    typename AType,
    typename BType,
    typename CType,
    typename DType,
    int M,
    int N,
    int K,
    bool saturatingAccumulation>
WmmaFragment<DType, M, N, K, MatrixC> __device__ coopMatMulAdd(
    WmmaFragment<AType, M, N, K, MatrixUse::MatrixA> matA,
    WmmaFragment<BType, M, N, K, MatrixUse::MatrixB> matB,
    WmmaFragment<CType, M, N, K, MatrixUse::MatrixC> matC)
{
    constexpr ShapeCombination shape = (M == 16 && N == 16 && K == 16) ? ShapeCombination::m16n16k16
                                       : (M == 8 && N == 32 && K == 16)
                                           ? ShapeCombination::m8n32k16
                                           : ShapeCombination::m32n8k16;

    WmmaFragment<DType, M, N, K, MatrixC> matD;
    uint32_t encodedLayout = (matA.m_layout == Layout::RowMajor ? 1 : 0) << 1 |
                             (matB.m_layout == Layout::RowMajor ? 1 : 0);

    switch (encodedLayout)
    {
    // 00011
    case 0x3:
        MMAHelper<
            AType,
            BType,
            CType,
            DType,
            shape,
            Layout::RowMajor,
            Layout::RowMajor,
            saturatingAccumulation>::eval(matD, matA, matB, matC);
        break;
    // 00010
    case 0x2:
        MMAHelper<
            AType,
            BType,
            CType,
            DType,
            shape,
            Layout::RowMajor,
            Layout::ColMajor,
            saturatingAccumulation>::eval(matD, matA, matB, matC);
        break;
    // 0001
    case 0x01:
        MMAHelper<
            AType,
            BType,
            CType,
            DType,
            shape,
            Layout::ColMajor,
            Layout::RowMajor,
            saturatingAccumulation>::eval(matD, matA, matB, matC);
        break;
    // 0000
    case 0x00:
        MMAHelper<
            AType,
            BType,
            CType,
            DType,
            shape,
            Layout::ColMajor,
            Layout::ColMajor,
            saturatingAccumulation>::eval(matD, matA, matB, matC);
        break;
    }

    return matD;
}

} // namespace Slang_CUDA_WMMA
#endif // #if (((__CUDACC_VER_MAJOR__ >=12)&&(__CUDACC_VER_MINOR__>=5)) || (CUDA_VERSION >= 12050))

#endif

struct shRadiativeParticle_Parameters_0
{
    FixedArray<float3 , 16>  sphCoefficients_0;
};

__device__ shRadiativeParticle_Parameters_0 shRadiativeParticle_Parameters_x24_syn_dzero_0()
{
    shRadiativeParticle_Parameters_0 result_0;
    float3  _S1 = make_float3 (0.0f);
    (&result_0)->sphCoefficients_0[int(0)] = _S1;
    (&result_0)->sphCoefficients_0[int(1)] = _S1;
    (&result_0)->sphCoefficients_0[int(2)] = _S1;
    (&result_0)->sphCoefficients_0[int(3)] = _S1;
    (&result_0)->sphCoefficients_0[int(4)] = _S1;
    (&result_0)->sphCoefficients_0[int(5)] = _S1;
    (&result_0)->sphCoefficients_0[int(6)] = _S1;
    (&result_0)->sphCoefficients_0[int(7)] = _S1;
    (&result_0)->sphCoefficients_0[int(8)] = _S1;
    (&result_0)->sphCoefficients_0[int(9)] = _S1;
    (&result_0)->sphCoefficients_0[int(10)] = _S1;
    (&result_0)->sphCoefficients_0[int(11)] = _S1;
    (&result_0)->sphCoefficients_0[int(12)] = _S1;
    (&result_0)->sphCoefficients_0[int(13)] = _S1;
    (&result_0)->sphCoefficients_0[int(14)] = _S1;
    (&result_0)->sphCoefficients_0[int(15)] = _S1;
    return result_0;
}

__device__ shRadiativeParticle_Parameters_0 shRadiativeParticle_Parameters_x24_syn_dadd_0(shRadiativeParticle_Parameters_0 * SLANG_anonymous_0_0, shRadiativeParticle_Parameters_0 * SLANG_anonymous_1_0)
{
    shRadiativeParticle_Parameters_0 result_1;
    (&result_1)->sphCoefficients_0[int(0)] = SLANG_anonymous_0_0->sphCoefficients_0[int(0)] + SLANG_anonymous_1_0->sphCoefficients_0[int(0)];
    (&result_1)->sphCoefficients_0[int(1)] = SLANG_anonymous_0_0->sphCoefficients_0[int(1)] + SLANG_anonymous_1_0->sphCoefficients_0[int(1)];
    (&result_1)->sphCoefficients_0[int(2)] = SLANG_anonymous_0_0->sphCoefficients_0[int(2)] + SLANG_anonymous_1_0->sphCoefficients_0[int(2)];
    (&result_1)->sphCoefficients_0[int(3)] = SLANG_anonymous_0_0->sphCoefficients_0[int(3)] + SLANG_anonymous_1_0->sphCoefficients_0[int(3)];
    (&result_1)->sphCoefficients_0[int(4)] = SLANG_anonymous_0_0->sphCoefficients_0[int(4)] + SLANG_anonymous_1_0->sphCoefficients_0[int(4)];
    (&result_1)->sphCoefficients_0[int(5)] = SLANG_anonymous_0_0->sphCoefficients_0[int(5)] + SLANG_anonymous_1_0->sphCoefficients_0[int(5)];
    (&result_1)->sphCoefficients_0[int(6)] = SLANG_anonymous_0_0->sphCoefficients_0[int(6)] + SLANG_anonymous_1_0->sphCoefficients_0[int(6)];
    (&result_1)->sphCoefficients_0[int(7)] = SLANG_anonymous_0_0->sphCoefficients_0[int(7)] + SLANG_anonymous_1_0->sphCoefficients_0[int(7)];
    (&result_1)->sphCoefficients_0[int(8)] = SLANG_anonymous_0_0->sphCoefficients_0[int(8)] + SLANG_anonymous_1_0->sphCoefficients_0[int(8)];
    (&result_1)->sphCoefficients_0[int(9)] = SLANG_anonymous_0_0->sphCoefficients_0[int(9)] + SLANG_anonymous_1_0->sphCoefficients_0[int(9)];
    (&result_1)->sphCoefficients_0[int(10)] = SLANG_anonymous_0_0->sphCoefficients_0[int(10)] + SLANG_anonymous_1_0->sphCoefficients_0[int(10)];
    (&result_1)->sphCoefficients_0[int(11)] = SLANG_anonymous_0_0->sphCoefficients_0[int(11)] + SLANG_anonymous_1_0->sphCoefficients_0[int(11)];
    (&result_1)->sphCoefficients_0[int(12)] = SLANG_anonymous_0_0->sphCoefficients_0[int(12)] + SLANG_anonymous_1_0->sphCoefficients_0[int(12)];
    (&result_1)->sphCoefficients_0[int(13)] = SLANG_anonymous_0_0->sphCoefficients_0[int(13)] + SLANG_anonymous_1_0->sphCoefficients_0[int(13)];
    (&result_1)->sphCoefficients_0[int(14)] = SLANG_anonymous_0_0->sphCoefficients_0[int(14)] + SLANG_anonymous_1_0->sphCoefficients_0[int(14)];
    (&result_1)->sphCoefficients_0[int(15)] = SLANG_anonymous_0_0->sphCoefficients_0[int(15)] + SLANG_anonymous_1_0->sphCoefficients_0[int(15)];
    return result_1;
}

struct gaussianParticle_Parameters_0
{
    float3  position_0;
    float3  scale_0;
    Matrix<float, 3, 3>  rotationT_0;
    float density_0;
};

__device__ gaussianParticle_Parameters_0 gaussianParticle_Parameters_x24_syn_dzero_0()
{
    gaussianParticle_Parameters_0 result_2;
    float3  _S2 = make_float3 (0.0f);
    (&result_2)->position_0 = _S2;
    (&result_2)->scale_0 = _S2;
    (&result_2)->rotationT_0 = makeMatrix<float, 3, 3> (0.0f);
    (&result_2)->density_0 = 0.0f;
    return result_2;
}

struct gaussianParticle_RawParameters_0
{
    float3  position_1;
    float density_1;
    float4  quaternion_0;
    float3  scale_1;
    float padding_0;
};

__device__ gaussianParticle_RawParameters_0 gaussianParticle_RawParameters_x24_syn_dzero_0()
{
    gaussianParticle_RawParameters_0 result_3;
    float3  _S3 = make_float3 (0.0f);
    (&result_3)->position_1 = _S3;
    (&result_3)->density_1 = 0.0f;
    (&result_3)->quaternion_0 = make_float4 (0.0f);
    (&result_3)->scale_1 = _S3;
    (&result_3)->padding_0 = 0.0f;
    return result_3;
}

__device__ Matrix<float, 3, 3>  transforms_rotationMatrixTranspose_0(float4  quaternion_1)
{
    float _S4 = quaternion_1.y;
    float xx_0 = _S4 * _S4;
    float _S5 = quaternion_1.z;
    float yy_0 = _S5 * _S5;
    float _S6 = quaternion_1.w;
    float zz_0 = _S6 * _S6;
    float xy_0 = _S4 * _S5;
    float xz_0 = _S4 * _S6;
    float yz_0 = _S5 * _S6;
    float _S7 = quaternion_1.x;
    float rx_0 = _S7 * _S4;
    float ry_0 = _S7 * _S5;
    float rz_0 = _S7 * _S6;
    return makeMatrix<float, 3, 3> (make_float3 (1.0f - 2.0f * (yy_0 + zz_0), 2.0f * (xy_0 + rz_0), 2.0f * (xz_0 - ry_0)), make_float3 (2.0f * (xy_0 - rz_0), 1.0f - 2.0f * (xx_0 + zz_0), 2.0f * (yz_0 + rx_0)), make_float3 (2.0f * (xz_0 + ry_0), 2.0f * (yz_0 - rx_0), 1.0f - 2.0f * (xx_0 + yy_0)));
}

__device__ gaussianParticle_Parameters_0 gaussianParticle_Parameters_x24init_0(float3  position_2, float3  scale_2, Matrix<float, 3, 3>  rotationT_1, float density_2)
{
    gaussianParticle_Parameters_0 _S8;
    (&_S8)->position_0 = position_2;
    (&_S8)->scale_0 = scale_2;
    (&_S8)->rotationT_0 = rotationT_1;
    (&_S8)->density_0 = density_2;
    return _S8;
}

struct gaussianParticle_RawParametersBuffer_0
{
    gaussianParticle_RawParameters_0 * _dataPtr_0;
    gaussianParticle_RawParameters_0 * _gradPtr_0;
    bool exclusiveGradient_0;
};

struct gaussianParticle_CommonParameters_0
{
    gaussianParticle_RawParametersBuffer_0 parametersBuffer_0;
};

__device__ gaussianParticle_Parameters_0 particleDensityParameters(uint particleIdx_0, gaussianParticle_CommonParameters_0 commonParameters_0)
{
    gaussianParticle_RawParameters_0 * _S9 = commonParameters_0.parametersBuffer_0._dataPtr_0 + particleIdx_0;
    return gaussianParticle_Parameters_x24init_0((*_S9).position_1, (*_S9).scale_1, transforms_rotationMatrixTranspose_0((*_S9).quaternion_0), (*_S9).density_1);
}

struct DiffPair_matrixx3Cfloatx2C3x2C3x3E_0
{
    Matrix<float, 3, 3>  primal_0;
    Matrix<float, 3, 3>  differential_0;
};

struct DiffPair_vectorx3Cfloatx2C3x3E_0
{
    float3  primal_0;
    float3  differential_0;
};

__device__ void _d_mul_0(DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 * left_0, DiffPair_vectorx3Cfloatx2C3x3E_0 * right_0, float3  dOut_0)
{
    float _S10 = (*left_0).primal_0.rows[int(0)].x * dOut_0.x;
    Matrix<float, 3, 3>  left_d_result_0;
    *&(((&left_d_result_0)->rows + (int(0)))->x) = (*right_0).primal_0.x * dOut_0.x;
    float sum_0 = _S10 + (*left_0).primal_0.rows[int(1)].x * dOut_0.y;
    *&(((&left_d_result_0)->rows + (int(1)))->x) = (*right_0).primal_0.x * dOut_0.y;
    float sum_1 = sum_0 + (*left_0).primal_0.rows[int(2)].x * dOut_0.z;
    *&(((&left_d_result_0)->rows + (int(2)))->x) = (*right_0).primal_0.x * dOut_0.z;
    float3  right_d_result_0;
    *&((&right_d_result_0)->x) = sum_1;
    float _S11 = (*left_0).primal_0.rows[int(0)].y * dOut_0.x;
    *&(((&left_d_result_0)->rows + (int(0)))->y) = (*right_0).primal_0.y * dOut_0.x;
    float sum_2 = _S11 + (*left_0).primal_0.rows[int(1)].y * dOut_0.y;
    *&(((&left_d_result_0)->rows + (int(1)))->y) = (*right_0).primal_0.y * dOut_0.y;
    float sum_3 = sum_2 + (*left_0).primal_0.rows[int(2)].y * dOut_0.z;
    *&(((&left_d_result_0)->rows + (int(2)))->y) = (*right_0).primal_0.y * dOut_0.z;
    *&((&right_d_result_0)->y) = sum_3;
    float _S12 = (*left_0).primal_0.rows[int(0)].z * dOut_0.x;
    *&(((&left_d_result_0)->rows + (int(0)))->z) = (*right_0).primal_0.z * dOut_0.x;
    float sum_4 = _S12 + (*left_0).primal_0.rows[int(1)].z * dOut_0.y;
    *&(((&left_d_result_0)->rows + (int(1)))->z) = (*right_0).primal_0.z * dOut_0.y;
    float sum_5 = sum_4 + (*left_0).primal_0.rows[int(2)].z * dOut_0.z;
    *&(((&left_d_result_0)->rows + (int(2)))->z) = (*right_0).primal_0.z * dOut_0.z;
    *&((&right_d_result_0)->z) = sum_5;
    left_0->primal_0 = (*left_0).primal_0;
    left_0->differential_0 = left_d_result_0;
    right_0->primal_0 = (*right_0).primal_0;
    right_0->differential_0 = right_d_result_0;
    return;
}

__device__ float3  mul_0(Matrix<float, 3, 3>  left_1, float3  right_1)
{
    float3  result_4;
    int i_0 = int(0);
    for(;;)
    {
        if(i_0 < int(3))
        {
        }
        else
        {
            break;
        }
        int j_0 = int(0);
        float sum_6 = 0.0f;
        for(;;)
        {
            if(j_0 < int(3))
            {
            }
            else
            {
                break;
            }
            float sum_7 = sum_6 + _slang_vector_get_element(left_1.rows[i_0], j_0) * _slang_vector_get_element(right_1, j_0);
            j_0 = j_0 + int(1);
            sum_6 = sum_7;
        }
        *_slang_vector_get_element_ptr(&result_4, i_0) = sum_6;
        i_0 = i_0 + int(1);
    }
    return result_4;
}

struct DiffPair_float_0
{
    float primal_0;
    float differential_0;
};

__device__ void _d_max_0(DiffPair_float_0 * dpx_0, DiffPair_float_0 * dpy_0, float dOut_1)
{
    DiffPair_float_0 _S13 = *dpx_0;
    float _S14;
    if(((*dpx_0).primal_0) > ((*dpy_0).primal_0))
    {
        _S14 = dOut_1;
    }
    else
    {
        if(((*dpx_0).primal_0) < ((*dpy_0).primal_0))
        {
            _S14 = 0.0f;
        }
        else
        {
            _S14 = 0.5f * dOut_1;
        }
    }
    dpx_0->primal_0 = _S13.primal_0;
    dpx_0->differential_0 = _S14;
    DiffPair_float_0 _S15 = *dpy_0;
    if(((*dpy_0).primal_0) > (_S13.primal_0))
    {
        _S14 = dOut_1;
    }
    else
    {
        if(((*dpy_0).primal_0) < ((*dpx_0).primal_0))
        {
            _S14 = 0.0f;
        }
        else
        {
            _S14 = 0.5f * dOut_1;
        }
    }
    dpy_0->primal_0 = _S15.primal_0;
    dpy_0->differential_0 = _S14;
    return;
}

__device__ void _d_sqrt_0(DiffPair_float_0 * dpx_1, float dOut_2)
{
    float _S16 = 0.5f / (F32_sqrt(((F32_max((1.00000001168609742e-07f), ((*dpx_1).primal_0)))))) * dOut_2;
    dpx_1->primal_0 = (*dpx_1).primal_0;
    dpx_1->differential_0 = _S16;
    return;
}

__device__ void _d_dot_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpx_2, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpy_1, float dOut_3)
{
    float3  x_d_result_0;
    *&((&x_d_result_0)->x) = (*dpy_1).primal_0.x * dOut_3;
    float3  y_d_result_0;
    *&((&y_d_result_0)->x) = (*dpx_2).primal_0.x * dOut_3;
    *&((&x_d_result_0)->y) = (*dpy_1).primal_0.y * dOut_3;
    *&((&y_d_result_0)->y) = (*dpx_2).primal_0.y * dOut_3;
    *&((&x_d_result_0)->z) = (*dpy_1).primal_0.z * dOut_3;
    *&((&y_d_result_0)->z) = (*dpx_2).primal_0.z * dOut_3;
    dpx_2->primal_0 = (*dpx_2).primal_0;
    dpx_2->differential_0 = x_d_result_0;
    dpy_1->primal_0 = (*dpy_1).primal_0;
    dpy_1->differential_0 = y_d_result_0;
    return;
}

__device__ float dot_0(float3  x_0, float3  y_0)
{
    int i_1 = int(0);
    float result_5 = 0.0f;
    for(;;)
    {
        if(i_1 < int(3))
        {
        }
        else
        {
            break;
        }
        float result_6 = result_5 + _slang_vector_get_element(x_0, i_1) * _slang_vector_get_element(y_0, i_1);
        i_1 = i_1 + int(1);
        result_5 = result_6;
    }
    return result_5;
}

__device__ float length_0(float3  x_1)
{
    return (F32_sqrt((dot_0(x_1, x_1))));
}

__device__ float3  normalize_0(float3  x_2)
{
    return x_2 / make_float3 (length_0(x_2));
}

__device__ void _d_cross_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * a_0, DiffPair_vectorx3Cfloatx2C3x3E_0 * b_0, float3  dOut_4)
{
    float _S17 = dOut_4.y;
    float _S18 = dOut_4.z;
    float _S19 = dOut_4.x;
    float _S20 = (*a_0).primal_0.z * _S17 + - (*a_0).primal_0.y * _S18;
    float _S21 = - (*a_0).primal_0.z * _S19 + (*a_0).primal_0.x * _S18;
    float _S22 = (*a_0).primal_0.y * _S19 + - (*a_0).primal_0.x * _S17;
    float3  _S23 = make_float3 (- (*b_0).primal_0.z * _S17 + (*b_0).primal_0.y * _S18, (*b_0).primal_0.z * _S19 + - (*b_0).primal_0.x * _S18, - (*b_0).primal_0.y * _S19 + (*b_0).primal_0.x * _S17);
    a_0->primal_0 = (*a_0).primal_0;
    a_0->differential_0 = _S23;
    float3  _S24 = make_float3 (_S20, _S21, _S22);
    b_0->primal_0 = (*b_0).primal_0;
    b_0->differential_0 = _S24;
    return;
}

__device__ float3  cross_0(float3  left_2, float3  right_2)
{
    float _S25 = left_2.y;
    float _S26 = right_2.z;
    float _S27 = left_2.z;
    float _S28 = right_2.y;
    float _S29 = right_2.x;
    float _S30 = left_2.x;
    return make_float3 (_S25 * _S26 - _S27 * _S28, _S27 * _S29 - _S30 * _S26, _S30 * _S28 - _S25 * _S29);
}

__device__ void _d_exp_0(DiffPair_float_0 * dpx_3, float dOut_5)
{
    float _S31 = (F32_exp(((*dpx_3).primal_0))) * dOut_5;
    dpx_3->primal_0 = (*dpx_3).primal_0;
    dpx_3->differential_0 = _S31;
    return;
}

__device__ void _d_min_0(DiffPair_float_0 * dpx_4, DiffPair_float_0 * dpy_2, float dOut_6)
{
    DiffPair_float_0 _S32 = *dpx_4;
    float _S33;
    if(((*dpx_4).primal_0) < ((*dpy_2).primal_0))
    {
        _S33 = dOut_6;
    }
    else
    {
        if(((*dpx_4).primal_0) > ((*dpy_2).primal_0))
        {
            _S33 = 0.0f;
        }
        else
        {
            _S33 = 0.5f * dOut_6;
        }
    }
    dpx_4->primal_0 = _S32.primal_0;
    dpx_4->differential_0 = _S33;
    DiffPair_float_0 _S34 = *dpy_2;
    if(((*dpy_2).primal_0) < (_S32.primal_0))
    {
        _S33 = dOut_6;
    }
    else
    {
        if(((*dpy_2).primal_0) > ((*dpx_4).primal_0))
        {
            _S33 = 0.0f;
        }
        else
        {
            _S33 = 0.5f * dOut_6;
        }
    }
    dpy_2->primal_0 = _S34.primal_0;
    dpy_2->differential_0 = _S33;
    return;
}

__device__ void _d_mul_1(DiffPair_vectorx3Cfloatx2C3x3E_0 * left_3, DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 * right_3, float3  dOut_7)
{
    float _S35 = (*right_3).primal_0.rows[int(0)].x * dOut_7.x;
    Matrix<float, 3, 3>  right_d_result_1;
    *&(((&right_d_result_1)->rows + (int(0)))->x) = (*left_3).primal_0.x * dOut_7.x;
    float sum_8 = _S35 + (*right_3).primal_0.rows[int(0)].y * dOut_7.y;
    *&(((&right_d_result_1)->rows + (int(0)))->y) = (*left_3).primal_0.x * dOut_7.y;
    float sum_9 = sum_8 + (*right_3).primal_0.rows[int(0)].z * dOut_7.z;
    *&(((&right_d_result_1)->rows + (int(0)))->z) = (*left_3).primal_0.x * dOut_7.z;
    float3  left_d_result_1;
    *&((&left_d_result_1)->x) = sum_9;
    float _S36 = (*right_3).primal_0.rows[int(1)].x * dOut_7.x;
    *&(((&right_d_result_1)->rows + (int(1)))->x) = (*left_3).primal_0.y * dOut_7.x;
    float sum_10 = _S36 + (*right_3).primal_0.rows[int(1)].y * dOut_7.y;
    *&(((&right_d_result_1)->rows + (int(1)))->y) = (*left_3).primal_0.y * dOut_7.y;
    float sum_11 = sum_10 + (*right_3).primal_0.rows[int(1)].z * dOut_7.z;
    *&(((&right_d_result_1)->rows + (int(1)))->z) = (*left_3).primal_0.y * dOut_7.z;
    *&((&left_d_result_1)->y) = sum_11;
    float _S37 = (*right_3).primal_0.rows[int(2)].x * dOut_7.x;
    *&(((&right_d_result_1)->rows + (int(2)))->x) = (*left_3).primal_0.z * dOut_7.x;
    float sum_12 = _S37 + (*right_3).primal_0.rows[int(2)].y * dOut_7.y;
    *&(((&right_d_result_1)->rows + (int(2)))->y) = (*left_3).primal_0.z * dOut_7.y;
    float sum_13 = sum_12 + (*right_3).primal_0.rows[int(2)].z * dOut_7.z;
    *&(((&right_d_result_1)->rows + (int(2)))->z) = (*left_3).primal_0.z * dOut_7.z;
    *&((&left_d_result_1)->z) = sum_13;
    left_3->primal_0 = (*left_3).primal_0;
    left_3->differential_0 = left_d_result_1;
    right_3->primal_0 = (*right_3).primal_0;
    right_3->differential_0 = right_d_result_1;
    return;
}

__device__ float3  mul_1(float3  left_4, Matrix<float, 3, 3>  right_4)
{
    float3  result_7;
    int j_1 = int(0);
    for(;;)
    {
        if(j_1 < int(3))
        {
        }
        else
        {
            break;
        }
        int i_2 = int(0);
        float sum_14 = 0.0f;
        for(;;)
        {
            if(i_2 < int(3))
            {
            }
            else
            {
                break;
            }
            float sum_15 = sum_14 + _slang_vector_get_element(left_4, i_2) * _slang_vector_get_element(right_4.rows[i_2], j_1);
            i_2 = i_2 + int(1);
            sum_14 = sum_15;
        }
        *_slang_vector_get_element_ptr(&result_7, j_1) = sum_14;
        j_1 = j_1 + int(1);
    }
    return result_7;
}

__device__ bool particleDensityHit(float3  rayOrigin_0, float3  rayDirection_0, gaussianParticle_Parameters_0 parameters_0, float * alpha_0, float * depth_0, bool enableNormal_0, float3  * normal_0)
{
    float3  giscl_0 = make_float3 (1.0f) / parameters_0.scale_0;
    float3  canonicalRayOrigin_0 = giscl_0 * mul_0(parameters_0.rotationT_0, rayOrigin_0 - parameters_0.position_0);
    float3  canonicalRayDirection_0 = normalize_0(giscl_0 * mul_0(parameters_0.rotationT_0, rayDirection_0));
    float3  gcrod_0 = cross_0(canonicalRayDirection_0, canonicalRayOrigin_0);
    float _S38 = dot_0(gcrod_0, gcrod_0);
    float _S39 = (F32_exp((-0.0555555559694767f * _S38 * _S38)));
    *alpha_0 = (F32_min((0.99000000953674316f), (_S39 * parameters_0.density_0)));
    bool acceptHit_0;
    if(_S39 > 0.01130000036209822f)
    {
        acceptHit_0 = (*alpha_0) > 0.00392156885936856f;
    }
    else
    {
        acceptHit_0 = false;
    }
    if(acceptHit_0)
    {
        float3  grds_0 = parameters_0.scale_0 * canonicalRayDirection_0 * make_float3 (dot_0(canonicalRayDirection_0, make_float3 (-1.0f) * canonicalRayOrigin_0));
        *depth_0 = (F32_sqrt((dot_0(grds_0, grds_0))));
        if(enableNormal_0)
        {
            float3  surfelNm_0 = make_float3 (0.0f, 0.0f, 1.0f);
            float3  surfelNm_1;
            if((dot_0(surfelNm_0, canonicalRayDirection_0)) > 0.0f)
            {
                surfelNm_1 = surfelNm_0 * make_float3 (-1.0f);
            }
            else
            {
                surfelNm_1 = surfelNm_0;
            }
            float3  _S40 = normalize_0(mul_1(surfelNm_1 * parameters_0.scale_0, parameters_0.rotationT_0));
            *normal_0 = _S40;
        }
    }
    return acceptHit_0;
}

__device__ void _d_lerp_0(DiffPair_float_0 * dpx_5, DiffPair_float_0 * dpy_3, DiffPair_float_0 * dps_0, float dOut_8)
{
    float _S41 = (1.0f - (*dps_0).primal_0) * dOut_8;
    dpx_5->primal_0 = (*dpx_5).primal_0;
    dpx_5->differential_0 = _S41;
    DiffPair_float_0 _S42 = *dpy_3;
    float _S43 = (*dps_0).primal_0 * dOut_8;
    dpy_3->primal_0 = (*dpy_3).primal_0;
    dpy_3->differential_0 = _S43;
    float _S44 = (_S42.primal_0 - (*dpx_5).primal_0) * dOut_8;
    dps_0->primal_0 = _S42.primal_0;
    dps_0->differential_0 = _S44;
    return;
}

__device__ void _d_lerp_vector_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpx_6, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpy_4, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpz_0, float3  dOut_9)
{
    DiffPair_float_0 left_dp_0;
    (&left_dp_0)->primal_0 = (*dpx_6).primal_0.x;
    (&left_dp_0)->differential_0 = 0.0f;
    DiffPair_float_0 middle_dp_0;
    (&middle_dp_0)->primal_0 = (*dpy_4).primal_0.x;
    (&middle_dp_0)->differential_0 = 0.0f;
    DiffPair_float_0 right_dp_0;
    (&right_dp_0)->primal_0 = (*dpz_0).primal_0.x;
    (&right_dp_0)->differential_0 = 0.0f;
    _d_lerp_0(&left_dp_0, &middle_dp_0, &right_dp_0, dOut_9.x);
    float3  left_d_result_2;
    *&((&left_d_result_2)->x) = left_dp_0.differential_0;
    float3  middle_d_result_0;
    *&((&middle_d_result_0)->x) = middle_dp_0.differential_0;
    float3  right_d_result_2;
    *&((&right_d_result_2)->x) = right_dp_0.differential_0;
    DiffPair_float_0 left_dp_1;
    (&left_dp_1)->primal_0 = (*dpx_6).primal_0.y;
    (&left_dp_1)->differential_0 = 0.0f;
    DiffPair_float_0 middle_dp_1;
    (&middle_dp_1)->primal_0 = (*dpy_4).primal_0.y;
    (&middle_dp_1)->differential_0 = 0.0f;
    DiffPair_float_0 right_dp_1;
    (&right_dp_1)->primal_0 = (*dpz_0).primal_0.y;
    (&right_dp_1)->differential_0 = 0.0f;
    _d_lerp_0(&left_dp_1, &middle_dp_1, &right_dp_1, dOut_9.y);
    *&((&left_d_result_2)->y) = left_dp_1.differential_0;
    *&((&middle_d_result_0)->y) = middle_dp_1.differential_0;
    *&((&right_d_result_2)->y) = right_dp_1.differential_0;
    DiffPair_float_0 left_dp_2;
    (&left_dp_2)->primal_0 = (*dpx_6).primal_0.z;
    (&left_dp_2)->differential_0 = 0.0f;
    DiffPair_float_0 middle_dp_2;
    (&middle_dp_2)->primal_0 = (*dpy_4).primal_0.z;
    (&middle_dp_2)->differential_0 = 0.0f;
    DiffPair_float_0 right_dp_2;
    (&right_dp_2)->primal_0 = (*dpz_0).primal_0.z;
    (&right_dp_2)->differential_0 = 0.0f;
    _d_lerp_0(&left_dp_2, &middle_dp_2, &right_dp_2, dOut_9.z);
    *&((&left_d_result_2)->z) = left_dp_2.differential_0;
    *&((&middle_d_result_0)->z) = middle_dp_2.differential_0;
    *&((&right_d_result_2)->z) = right_dp_2.differential_0;
    dpx_6->primal_0 = (*dpx_6).primal_0;
    dpx_6->differential_0 = left_d_result_2;
    dpy_4->primal_0 = (*dpy_4).primal_0;
    dpy_4->differential_0 = middle_d_result_0;
    dpz_0->primal_0 = (*dpz_0).primal_0;
    dpz_0->differential_0 = right_d_result_2;
    return;
}

__device__ float particleDensityIntegrateHit(float alpha_1, float * transmittance_0, float depth_1, float * integratedDepth_0, bool enableNormal_1, float3  normal_1, float3  * integratedNormal_0)
{
    float _S45 = alpha_1 * *transmittance_0;
    *integratedDepth_0 = *integratedDepth_0 + depth_1 * _S45;
    if(enableNormal_1)
    {
        *integratedNormal_0 = *integratedNormal_0 + normal_1 * make_float3 (_S45);
    }
    *transmittance_0 = *transmittance_0 * (1.0f - alpha_1);
    return _S45;
}

__device__ float particleDensityProcessHitFwdFromBuffer(float3  rayOrigin_1, float3  rayDirection_1, uint particleIdx_1, gaussianParticle_CommonParameters_0 commonParameters_1, float * transmittance_1, float * integratedDepth_1, bool enableNormal_2, float3  * integratedNormal_1)
{
    float depth_2;
    for(;;)
    {
        gaussianParticle_RawParameters_0 * _S46 = commonParameters_1.parametersBuffer_0._dataPtr_0 + particleIdx_1;
        gaussianParticle_Parameters_0 _S47 = gaussianParticle_Parameters_x24init_0((*_S46).position_1, (*_S46).scale_1, transforms_rotationMatrixTranspose_0((*_S46).quaternion_0), (*_S46).density_1);
        float3  giscl_1 = make_float3 (1.0f) / _S47.scale_0;
        float3  canonicalRayOrigin_1 = giscl_1 * mul_0(_S47.rotationT_0, rayOrigin_1 - _S47.position_0);
        float3  canonicalRayDirection_1 = normalize_0(giscl_1 * mul_0(_S47.rotationT_0, rayDirection_1));
        float3  gcrod_1 = cross_0(canonicalRayDirection_1, canonicalRayOrigin_1);
        float _S48 = dot_0(gcrod_1, gcrod_1);
        float _S49 = (F32_exp((-0.0555555559694767f * _S48 * _S48)));
        float alpha_2 = (F32_min((0.99000000953674316f), (_S49 * _S47.density_0)));
        bool acceptHit_1;
        if(_S49 > 0.01130000036209822f)
        {
            acceptHit_1 = alpha_2 > 0.00392156885936856f;
        }
        else
        {
            acceptHit_1 = false;
        }
        float3  normal_2;
        if(acceptHit_1)
        {
            float3  grds_1 = _S47.scale_0 * canonicalRayDirection_1 * make_float3 (dot_0(canonicalRayDirection_1, make_float3 (-1.0f) * canonicalRayOrigin_1));
            float _S50 = (F32_sqrt((dot_0(grds_1, grds_1))));
            if(enableNormal_2)
            {
                float3  surfelNm_2 = make_float3 (0.0f, 0.0f, 1.0f);
                if((dot_0(surfelNm_2, canonicalRayDirection_1)) > 0.0f)
                {
                    normal_2 = surfelNm_2 * make_float3 (-1.0f);
                }
                else
                {
                    normal_2 = surfelNm_2;
                }
                float3  _S51 = normalize_0(mul_1(normal_2 * _S47.scale_0, _S47.rotationT_0));
                normal_2 = _S51;
            }
            depth_2 = _S50;
        }
        if(acceptHit_1)
        {
            float _S52 = alpha_2 * *transmittance_1;
            *integratedDepth_1 = *integratedDepth_1 + depth_2 * _S52;
            if(enableNormal_2)
            {
                *integratedNormal_1 = *integratedNormal_1 + normal_2 * make_float3 (_S52);
            }
            *transmittance_1 = *transmittance_1 * (1.0f - alpha_2);
            depth_2 = _S52;
            break;
        }
        depth_2 = 0.0f;
        break;
    }
    return depth_2;
}

struct s_bwd_prop_gaussianParticle_processHitFromBuffer_Intermediates_0
{
    float _S53;
    float _S54;
    float3  _S55;
    gaussianParticle_RawParameters_0 _S56;
};

__device__ Matrix<float, 3, 3>  s_primal_ctx_transforms_rotationMatrixTranspose_0(float4  dpquaternion_0)
{
    float _S57 = dpquaternion_0.y;
    float xx_1 = _S57 * _S57;
    float _S58 = dpquaternion_0.z;
    float yy_1 = _S58 * _S58;
    float _S59 = dpquaternion_0.w;
    float zz_1 = _S59 * _S59;
    float xy_1 = _S57 * _S58;
    float xz_1 = _S57 * _S59;
    float yz_1 = _S58 * _S59;
    float _S60 = dpquaternion_0.x;
    float rx_1 = _S60 * _S57;
    float ry_1 = _S60 * _S58;
    float rz_1 = _S60 * _S59;
    return makeMatrix<float, 3, 3> (make_float3 (1.0f - 2.0f * (yy_1 + zz_1), 2.0f * (xy_1 + rz_1), 2.0f * (xz_1 - ry_1)), make_float3 (2.0f * (xy_1 - rz_1), 1.0f - 2.0f * (xx_1 + zz_1), 2.0f * (yz_1 + rx_1)), make_float3 (2.0f * (xz_1 + ry_1), 2.0f * (yz_1 - rx_1), 1.0f - 2.0f * (xx_1 + yy_1)));
}

__device__ gaussianParticle_Parameters_0 s_primal_ctx_gaussianParticle_Parameters_x24init_0(float3  dpposition_0, float3  dpscale_0, Matrix<float, 3, 3>  dprotationT_0, float dpdensity_0)
{
    gaussianParticle_Parameters_0 _S61 = { dpposition_0, dpscale_0, dprotationT_0, dpdensity_0 };
    return _S61;
}

__device__ float3  s_primal_ctx_mul_0(Matrix<float, 3, 3>  _S62, float3  _S63)
{
    return mul_0(_S62, _S63);
}

__device__ float3  s_primal_ctx_cross_0(float3  _S64, float3  _S65)
{
    return cross_0(_S64, _S65);
}

__device__ float s_primal_ctx_dot_0(float3  _S66, float3  _S67)
{
    return dot_0(_S66, _S67);
}

__device__ float s_primal_ctx_exp_0(float _S68)
{
    return (F32_exp((_S68)));
}

__device__ float s_primal_ctx_sqrt_0(float _S69)
{
    return (F32_sqrt((_S69)));
}

__device__ float3  s_primal_ctx_mul_1(float3  _S70, Matrix<float, 3, 3>  _S71)
{
    return mul_1(_S70, _S71);
}

__device__ void s_bwd_prop_lerp_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S72, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S73, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S74, float3  _S75)
{
    _d_lerp_vector_0(_S72, _S73, _S74, _S75);
    return;
}

__device__ void s_bwd_prop_lerp_1(DiffPair_float_0 * _S76, DiffPair_float_0 * _S77, DiffPair_float_0 * _S78, float _S79)
{
    _d_lerp_0(_S76, _S77, _S78, _S79);
    return;
}

__device__ void s_bwd_prop_sqrt_0(DiffPair_float_0 * _S80, float _S81)
{
    _d_sqrt_0(_S80, _S81);
    return;
}

__device__ void s_bwd_prop_length_impl_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpx_7, float _s_dOut_0)
{
    float _S82 = (*dpx_7).primal_0.x;
    float _S83 = (*dpx_7).primal_0.y;
    float _S84 = (*dpx_7).primal_0.z;
    DiffPair_float_0 _S85;
    (&_S85)->primal_0 = _S82 * _S82 + _S83 * _S83 + _S84 * _S84;
    (&_S85)->differential_0 = 0.0f;
    s_bwd_prop_sqrt_0(&_S85, _s_dOut_0);
    float _S86 = (*dpx_7).primal_0.z * _S85.differential_0;
    float _S87 = _S86 + _S86;
    float _S88 = (*dpx_7).primal_0.y * _S85.differential_0;
    float _S89 = _S88 + _S88;
    float _S90 = (*dpx_7).primal_0.x * _S85.differential_0;
    float _S91 = _S90 + _S90;
    float3  _S92 = make_float3 (0.0f);
    *&((&_S92)->z) = _S87;
    *&((&_S92)->y) = _S89;
    *&((&_S92)->x) = _S91;
    dpx_7->primal_0 = (*dpx_7).primal_0;
    dpx_7->differential_0 = _S92;
    return;
}

__device__ void s_bwd_length_impl_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S93, float _S94)
{
    s_bwd_prop_length_impl_0(_S93, _S94);
    return;
}

__device__ void s_bwd_prop_normalize_impl_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpx_8, float3  _s_dOut_1)
{
    float _S95 = length_0((*dpx_8).primal_0);
    float3  _S96 = (*dpx_8).primal_0 * _s_dOut_1;
    float3  _S97 = make_float3 (1.0f / _S95) * _s_dOut_1;
    float _S98 = - ((_S96.x + _S96.y + _S96.z) / (_S95 * _S95));
    float3  _S99 = make_float3 (0.0f);
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S100;
    (&_S100)->primal_0 = (*dpx_8).primal_0;
    (&_S100)->differential_0 = _S99;
    s_bwd_length_impl_0(&_S100, _S98);
    float3  _S101 = _S97 + _S100.differential_0;
    dpx_8->primal_0 = (*dpx_8).primal_0;
    dpx_8->differential_0 = _S101;
    return;
}

__device__ void s_bwd_normalize_impl_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S102, float3  _S103)
{
    s_bwd_prop_normalize_impl_0(_S102, _S103);
    return;
}

__device__ void s_bwd_prop_mul_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S104, DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 * _S105, float3  _S106)
{
    _d_mul_1(_S104, _S105, _S106);
    return;
}

__device__ void s_bwd_prop_dot_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S107, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S108, float _S109)
{
    _d_dot_0(_S107, _S108, _S109);
    return;
}

__device__ void s_bwd_prop_exp_0(DiffPair_float_0 * _S110, float _S111)
{
    _d_exp_0(_S110, _S111);
    return;
}

__device__ void s_bwd_prop_cross_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S112, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S113, float3  _S114)
{
    _d_cross_0(_S112, _S113, _S114);
    return;
}

__device__ void s_bwd_prop_mul_1(DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 * _S115, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S116, float3  _S117)
{
    _d_mul_0(_S115, _S116, _S117);
    return;
}

__device__ void s_bwd_prop_gaussianParticle_Parameters_x24init_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpposition_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpscale_1, DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 * dprotationT_1, DiffPair_float_0 * dpdensity_1, gaussianParticle_Parameters_0 * _s_dOut_2)
{
    float3  _S118 = _s_dOut_2->position_0;
    float3  _S119 = _s_dOut_2->scale_0;
    Matrix<float, 3, 3>  _S120 = _s_dOut_2->rotationT_0;
    float _S121 = _s_dOut_2->density_0;
    dpdensity_1->primal_0 = (*dpdensity_1).primal_0;
    dpdensity_1->differential_0 = _S121;
    dprotationT_1->primal_0 = (*dprotationT_1).primal_0;
    dprotationT_1->differential_0 = _S120;
    dpscale_1->primal_0 = (*dpscale_1).primal_0;
    dpscale_1->differential_0 = _S119;
    dpposition_1->primal_0 = (*dpposition_1).primal_0;
    dpposition_1->differential_0 = _S118;
    return;
}

struct DiffPair_vectorx3Cfloatx2C4x3E_0
{
    float4  primal_0;
    float4  differential_0;
};

__device__ void s_bwd_prop_transforms_rotationMatrixTranspose_0(DiffPair_vectorx3Cfloatx2C4x3E_0 * dpquaternion_1, Matrix<float, 3, 3>  _s_dOut_3)
{
    float _S122 = (*dpquaternion_1).primal_0.y;
    float _S123 = (*dpquaternion_1).primal_0.z;
    float _S124 = (*dpquaternion_1).primal_0.w;
    float _S125 = (*dpquaternion_1).primal_0.x;
    float _S126 = 2.0f * - _s_dOut_3.rows[int(2)].z;
    float _S127 = 2.0f * _s_dOut_3.rows[int(2)].y;
    float _S128 = 2.0f * _s_dOut_3.rows[int(2)].x;
    float _S129 = 2.0f * _s_dOut_3.rows[int(1)].z;
    float _S130 = 2.0f * - _s_dOut_3.rows[int(1)].y;
    float _S131 = 2.0f * _s_dOut_3.rows[int(1)].x;
    float _S132 = 2.0f * _s_dOut_3.rows[int(0)].z;
    float _S133 = 2.0f * _s_dOut_3.rows[int(0)].y;
    float _S134 = 2.0f * - _s_dOut_3.rows[int(0)].x;
    float _S135 = - _S131 + _S133;
    float _S136 = _S128 + - _S132;
    float _S137 = - _S127 + _S129;
    float _S138 = _S127 + _S129;
    float _S139 = _S128 + _S132;
    float _S140 = _S131 + _S133;
    float _S141 = _S124 * (_S130 + _S134);
    float _S142 = _S123 * (_S126 + _S134);
    float _S143 = _S122 * (_S126 + _S130);
    float4  _S144 = make_float4 (_S124 * _S135 + _S123 * _S136 + _S122 * _S137, _S125 * _S137 + _S124 * _S139 + _S123 * _S140 + _S143 + _S143, _S125 * _S136 + _S124 * _S138 + _S122 * _S140 + _S142 + _S142, _S125 * _S135 + _S123 * _S138 + _S122 * _S139 + _S141 + _S141);
    dpquaternion_1->primal_0 = (*dpquaternion_1).primal_0;
    dpquaternion_1->differential_0 = _S144;
    return;
}

__device__ void particleDensityProcessHitBwdToBuffer(float3  rayOrigin_2, float3  rayDirection_2, uint particleIdx_2, gaussianParticle_CommonParameters_0 commonParameters_2, float alpha_3, float alphaGrad_0, float * transmittance_2, float * transmittanceGrad_0, float depth_3, float * integratedDepth_2, float * integratedDepthGrad_0, bool enableNormal_3, float3  normal_3, float3  * integratedNormal_2, float3  * integratedNormalGrad_0)
{
    if(alpha_3 > 0.0f)
    {
        float weight_0 = 1.0f / (1.0f - alpha_3);
        float _S145 = *transmittance_2 * weight_0;
        *transmittance_2 = _S145;
        float _S146 = *transmittanceGrad_0;
        float _S147 = (*integratedDepth_2 - depth_3 * alpha_3) * weight_0;
        *integratedDepth_2 = _S147;
        float _S148 = *integratedDepthGrad_0;
        DiffPair_vectorx3Cfloatx2C3x3E_0 integratedNormalDiff_0;
        if(enableNormal_3)
        {
            float3  _S149 = (*integratedNormal_2 - normal_3 * make_float3 (alpha_3)) * make_float3 (weight_0);
            *integratedNormal_2 = _S149;
            (&integratedNormalDiff_0)->primal_0 = _S149;
            (&integratedNormalDiff_0)->differential_0 = *integratedNormalGrad_0;
        }
        else
        {
            int3  _S150 = make_int3 (int(0));
            float3  _S151 = make_float3 ((float)_S150.x, (float)_S150.y, (float)_S150.z);
            (&integratedNormalDiff_0)->primal_0 = _S151;
            (&integratedNormalDiff_0)->differential_0 = _S151;
        }
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S152 = integratedNormalDiff_0;
        s_bwd_prop_gaussianParticle_processHitFromBuffer_Intermediates_0 _S153;
        float3  _S154 = make_float3 (0.0f);
        float4  _S155 = make_float4 (0.0f);
        gaussianParticle_RawParameters_0 _S156 = { _S154, 0.0f, _S155, _S154, 0.0f };
        (&_S153)->_S53 = 0.0f;
        (&_S153)->_S54 = 0.0f;
        (&_S153)->_S55 = _S154;
        (&_S153)->_S56 = _S156;
        (&(&_S153)->_S56)->position_1 = _S154;
        (&(&_S153)->_S56)->density_1 = 0.0f;
        (&(&_S153)->_S56)->quaternion_0 = _S155;
        (&(&_S153)->_S56)->scale_1 = _S154;
        (&(&_S153)->_S56)->padding_0 = 0.0f;
        (&_S153)->_S53 = _S145;
        (&_S153)->_S54 = _S147;
        (&_S153)->_S55 = _S152.primal_0;
        (&_S153)->_S56 = *(commonParameters_2.parametersBuffer_0._dataPtr_0 + particleIdx_2);
        s_bwd_prop_gaussianParticle_processHitFromBuffer_Intermediates_0 _S157 = _S153;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S158 = integratedNormalDiff_0;
        Matrix<float, 3, 3>  _S159 = s_primal_ctx_transforms_rotationMatrixTranspose_0(_S153._S56.quaternion_0);
        gaussianParticle_Parameters_0 _S160 = s_primal_ctx_gaussianParticle_Parameters_x24init_0(_S153._S56.position_1, _S153._S56.scale_1, _S159, _S153._S56.density_1);
        float3  giscl_2 = make_float3 (1.0f) / _S160.scale_0;
        float3  _S161 = _S160.scale_0 * _S160.scale_0;
        float3  gposc_0 = rayOrigin_2 - _S160.position_0;
        float3  _S162 = s_primal_ctx_mul_0(_S160.rotationT_0, gposc_0);
        float3  canonicalRayOrigin_2 = giscl_2 * _S162;
        float3  _S163 = s_primal_ctx_mul_0(_S160.rotationT_0, rayDirection_2);
        float3  grdu_0 = giscl_2 * _S163;
        float3  _S164 = normalize_0(grdu_0);
        float3  _S165 = s_primal_ctx_cross_0(_S164, canonicalRayOrigin_2);
        float _S166 = s_primal_ctx_dot_0(_S165, _S165);
        float _S167 = -0.0555555559694767f * _S166;
        float _S168 = _S167 * _S166;
        float _S169 = s_primal_ctx_exp_0(_S168);
        float _S170 = _S169 * _S160.density_0;
        float _S171 = (F32_min((0.99000000953674316f), (_S170)));
        bool acceptHit_2;
        if(_S169 > 0.01130000036209822f)
        {
            acceptHit_2 = _S171 > 0.00392156885936856f;
        }
        else
        {
            acceptHit_2 = false;
        }
        float3  normal_4;
        float3  _S172;
        float3  _S173;
        float3  _S174;
        float3  _S175;
        float3  _S176;
        float3  _S177;
        float3  _S178;
        float3  _S179;
        float depth_4;
        float _S180;
        if(acceptHit_2)
        {
            float3  _S181 = _S160.scale_0 * _S164;
            float3  _S182 = make_float3 (-1.0f) * canonicalRayOrigin_2;
            float _S183 = s_primal_ctx_dot_0(_S164, _S182);
            float3  _S184 = make_float3 (_S183);
            float3  grds_2 = _S181 * make_float3 (_S183);
            float _S185 = s_primal_ctx_dot_0(grds_2, grds_2);
            float _S186 = s_primal_ctx_sqrt_0(_S185);
            if(enableNormal_3)
            {
                float3  surfelNm_3 = make_float3 (0.0f, 0.0f, 1.0f);
                if((s_primal_ctx_dot_0(surfelNm_3, _S164)) > 0.0f)
                {
                    normal_4 = surfelNm_3 * make_float3 (-1.0f);
                }
                else
                {
                    normal_4 = surfelNm_3;
                }
                float3  _S187 = normal_4 * _S160.scale_0;
                float3  _S188 = s_primal_ctx_mul_1(_S187, _S160.rotationT_0);
                float3  _S189 = normalize_0(_S188);
                float3  _S190 = normal_4;
                normal_4 = _S189;
                _S172 = _S188;
                _S173 = _S187;
                _S174 = _S190;
                _S175 = surfelNm_3;
            }
            else
            {
                _S172 = _S154;
                _S173 = _S154;
                _S174 = _S154;
                _S175 = _S154;
            }
            depth_4 = _S186;
            _S180 = _S185;
            _S176 = grds_2;
            _S177 = _S181;
            _S178 = _S184;
            _S179 = _S182;
        }
        else
        {
            _S172 = _S154;
            _S173 = _S154;
            _S174 = _S154;
            _S175 = _S154;
            _S180 = 0.0f;
            _S176 = _S154;
            _S177 = _S154;
            _S178 = _S154;
            _S179 = _S154;
        }
        bool _runFlag_0;
        float3  dpintegratedNormal_0;
        float _S191;
        if(acceptHit_2)
        {
            if(enableNormal_3)
            {
                dpintegratedNormal_0 = make_float3 (_S171);
            }
            else
            {
                dpintegratedNormal_0 = _S154;
            }
            float _S192 = 1.0f - _S171;
            _runFlag_0 = false;
            _S191 = _S192;
        }
        else
        {
            _runFlag_0 = true;
            _S191 = 0.0f;
            dpintegratedNormal_0 = _S154;
        }
        Matrix<float, 3, 3>  _S193 = makeMatrix<float, 3, 3> (0.0f);
        float dpintegratedDepth_0;
        if(_runFlag_0)
        {
            dpintegratedDepth_0 = 0.0f;
        }
        else
        {
            dpintegratedDepth_0 = alphaGrad_0;
        }
        float dptransmittance_0;
        if(acceptHit_2)
        {
            float _S194 = _S191 * _S146;
            float _S195 = - (_S157._S53 * _S146) + dpintegratedDepth_0;
            if(enableNormal_3)
            {
                DiffPair_vectorx3Cfloatx2C3x3E_0 _S196;
                (&_S196)->primal_0 = _S157._S55;
                (&_S196)->differential_0 = _S154;
                DiffPair_vectorx3Cfloatx2C3x3E_0 _S197;
                (&_S197)->primal_0 = normal_4;
                (&_S197)->differential_0 = _S154;
                DiffPair_vectorx3Cfloatx2C3x3E_0 _S198;
                (&_S198)->primal_0 = dpintegratedNormal_0;
                (&_S198)->differential_0 = _S154;
                s_bwd_prop_lerp_0(&_S196, &_S197, &_S198, _S158.differential_0);
                _S191 = _S198.differential_0.x + _S198.differential_0.y + _S198.differential_0.z + _S195;
                normal_4 = _S197.differential_0;
                dpintegratedNormal_0 = _S196.differential_0;
            }
            else
            {
                _S191 = _S195;
                normal_4 = _S154;
                dpintegratedNormal_0 = _S158.differential_0;
            }
            DiffPair_float_0 _S199;
            (&_S199)->primal_0 = _S157._S54;
            (&_S199)->differential_0 = 0.0f;
            DiffPair_float_0 _S200;
            (&_S200)->primal_0 = depth_4;
            (&_S200)->differential_0 = 0.0f;
            DiffPair_float_0 _S201;
            (&_S201)->primal_0 = _S171;
            (&_S201)->differential_0 = 0.0f;
            s_bwd_prop_lerp_1(&_S199, &_S200, &_S201, _S148);
            float _S202 = _S201.differential_0 + _S191;
            depth_4 = _S200.differential_0;
            _S191 = _S202;
            dpintegratedDepth_0 = _S199.differential_0;
            dptransmittance_0 = _S194;
        }
        else
        {
            depth_4 = 0.0f;
            normal_4 = _S154;
            _S191 = dpintegratedDepth_0;
            dpintegratedNormal_0 = _S158.differential_0;
            dpintegratedDepth_0 = _S148;
            dptransmittance_0 = _S146;
        }
        Matrix<float, 3, 3>  _S203;
        if(acceptHit_2)
        {
            if(enableNormal_3)
            {
                DiffPair_vectorx3Cfloatx2C3x3E_0 _S204;
                (&_S204)->primal_0 = _S172;
                (&_S204)->differential_0 = _S154;
                s_bwd_normalize_impl_0(&_S204, normal_4);
                DiffPair_vectorx3Cfloatx2C3x3E_0 _S205;
                (&_S205)->primal_0 = _S173;
                (&_S205)->differential_0 = _S154;
                DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S206;
                (&_S206)->primal_0 = _S160.rotationT_0;
                (&_S206)->differential_0 = _S193;
                s_bwd_prop_mul_0(&_S205, &_S206, _S204.differential_0);
                float3  _S207 = _S174 * _S205.differential_0;
                DiffPair_vectorx3Cfloatx2C3x3E_0 _S208;
                (&_S208)->primal_0 = _S175;
                (&_S208)->differential_0 = _S154;
                DiffPair_vectorx3Cfloatx2C3x3E_0 _S209;
                (&_S209)->primal_0 = _S164;
                (&_S209)->differential_0 = _S154;
                s_bwd_prop_dot_0(&_S208, &_S209, 0.0f);
                normal_4 = _S209.differential_0;
                _S172 = _S207;
                _S203 = _S206.differential_0;
            }
            else
            {
                normal_4 = _S154;
                _S172 = _S154;
                _S203 = _S193;
            }
            DiffPair_float_0 _S210;
            (&_S210)->primal_0 = _S180;
            (&_S210)->differential_0 = 0.0f;
            s_bwd_prop_sqrt_0(&_S210, depth_4);
            DiffPair_vectorx3Cfloatx2C3x3E_0 _S211;
            (&_S211)->primal_0 = _S176;
            (&_S211)->differential_0 = _S154;
            DiffPair_vectorx3Cfloatx2C3x3E_0 _S212;
            (&_S212)->primal_0 = _S176;
            (&_S212)->differential_0 = _S154;
            s_bwd_prop_dot_0(&_S211, &_S212, _S210.differential_0);
            float3  _S213 = _S212.differential_0 + _S211.differential_0;
            float3  _S214 = _S177 * _S213;
            float3  _S215 = _S178 * _S213;
            float _S216 = _S214.x + _S214.y + _S214.z;
            DiffPair_vectorx3Cfloatx2C3x3E_0 _S217;
            (&_S217)->primal_0 = _S164;
            (&_S217)->differential_0 = _S154;
            DiffPair_vectorx3Cfloatx2C3x3E_0 _S218;
            (&_S218)->primal_0 = _S179;
            (&_S218)->differential_0 = _S154;
            s_bwd_prop_dot_0(&_S217, &_S218, _S216);
            float3  _S219 = make_float3 (-1.0f) * _S218.differential_0;
            float3  _S220 = _S164 * _S215 + _S172;
            normal_4 = _S217.differential_0 + _S160.scale_0 * _S215 + normal_4;
            _S172 = _S219;
            _S173 = _S220;
        }
        else
        {
            normal_4 = _S154;
            _S172 = _S154;
            _S203 = _S193;
            _S173 = _S154;
        }
        DiffPair_float_0 _S221;
        (&_S221)->primal_0 = 0.99000000953674316f;
        (&_S221)->differential_0 = 0.0f;
        DiffPair_float_0 _S222;
        (&_S222)->primal_0 = _S170;
        (&_S222)->differential_0 = 0.0f;
        _d_min_0(&_S221, &_S222, _S191);
        float _S223 = _S169 * _S222.differential_0;
        float _S224 = _S160.density_0 * _S222.differential_0;
        DiffPair_float_0 _S225;
        (&_S225)->primal_0 = _S168;
        (&_S225)->differential_0 = 0.0f;
        s_bwd_prop_exp_0(&_S225, _S224);
        float _S226 = _S167 * _S225.differential_0 + -0.0555555559694767f * (_S166 * _S225.differential_0);
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S227;
        (&_S227)->primal_0 = _S165;
        (&_S227)->differential_0 = _S154;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S228;
        (&_S228)->primal_0 = _S165;
        (&_S228)->differential_0 = _S154;
        s_bwd_prop_dot_0(&_S227, &_S228, _S226);
        float3  _S229 = _S228.differential_0 + _S227.differential_0;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S230;
        (&_S230)->primal_0 = _S164;
        (&_S230)->differential_0 = _S154;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S231;
        (&_S231)->primal_0 = canonicalRayOrigin_2;
        (&_S231)->differential_0 = _S154;
        s_bwd_prop_cross_0(&_S230, &_S231, _S229);
        float3  _S232 = _S230.differential_0 + normal_4;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S233;
        (&_S233)->primal_0 = grdu_0;
        (&_S233)->differential_0 = _S154;
        s_bwd_normalize_impl_0(&_S233, _S232);
        float3  _S234 = giscl_2 * _S233.differential_0;
        float3  _S235 = _S163 * _S233.differential_0;
        DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S236;
        (&_S236)->primal_0 = _S160.rotationT_0;
        (&_S236)->differential_0 = _S193;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S237;
        (&_S237)->primal_0 = rayDirection_2;
        (&_S237)->differential_0 = _S154;
        s_bwd_prop_mul_1(&_S236, &_S237, _S234);
        float3  _S238 = _S231.differential_0 + _S172;
        float3  _S239 = giscl_2 * _S238;
        float3  _S240 = _S162 * _S238;
        DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S241;
        (&_S241)->primal_0 = _S160.rotationT_0;
        (&_S241)->differential_0 = _S193;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S242;
        (&_S242)->primal_0 = gposc_0;
        (&_S242)->differential_0 = _S154;
        s_bwd_prop_mul_1(&_S241, &_S242, _S239);
        Matrix<float, 3, 3>  _S243 = _S236.differential_0 + _S241.differential_0 + _S203;
        float3  _S244 = - _S242.differential_0;
        float3  _S245 = - ((_S235 + _S240) / _S161) + _S173;
        gaussianParticle_Parameters_0 _S246 = gaussianParticle_Parameters_x24_syn_dzero_0();
        (&_S246)->density_0 = _S223;
        (&_S246)->rotationT_0 = _S243;
        (&_S246)->position_0 = _S244;
        (&_S246)->scale_0 = _S245;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S247;
        (&_S247)->primal_0 = _S157._S56.position_1;
        (&_S247)->differential_0 = _S154;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S248;
        (&_S248)->primal_0 = _S157._S56.scale_1;
        (&_S248)->differential_0 = _S154;
        DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S249;
        (&_S249)->primal_0 = _S159;
        (&_S249)->differential_0 = _S193;
        DiffPair_float_0 _S250;
        (&_S250)->primal_0 = _S157._S56.density_1;
        (&_S250)->differential_0 = 0.0f;
        gaussianParticle_Parameters_0 _S251 = _S246;
        s_bwd_prop_gaussianParticle_Parameters_x24init_0(&_S247, &_S248, &_S249, &_S250, &_S251);
        DiffPair_vectorx3Cfloatx2C4x3E_0 _S252;
        (&_S252)->primal_0 = _S157._S56.quaternion_0;
        (&_S252)->differential_0 = _S155;
        s_bwd_prop_transforms_rotationMatrixTranspose_0(&_S252, _S249.differential_0);
        gaussianParticle_RawParameters_0 _S253 = gaussianParticle_RawParameters_x24_syn_dzero_0();
        (&_S253)->density_1 = _S250.differential_0;
        (&_S253)->quaternion_0 = _S252.differential_0;
        (&_S253)->scale_1 = _S248.differential_0;
        (&_S253)->position_1 = _S247.differential_0;
        gaussianParticle_RawParameters_0 _S254 = _S253;
        if(commonParameters_2.parametersBuffer_0.exclusiveGradient_0)
        {
            gaussianParticle_RawParameters_0 * _S255 = commonParameters_2.parametersBuffer_0._gradPtr_0 + particleIdx_2;
            _S255->density_1 = _S255->density_1 + _S254.density_1;
            _S255->position_1 = _S255->position_1 + _S254.position_1;
            _S255->quaternion_0 = _S255->quaternion_0 + _S254.quaternion_0;
            _S255->scale_1 = _S255->scale_1 + _S254.scale_1;
        }
        else
        {
            gaussianParticle_RawParameters_0 * _S256 = commonParameters_2.parametersBuffer_0._gradPtr_0 + particleIdx_2;
            float _S257 = atomicAdd(&_S256->density_1, _S254.density_1);
            float _S258 = atomicAdd(&((&_S256->position_1)->x), _S254.position_1.x);
            float _S259 = atomicAdd(&((&_S256->position_1)->y), _S254.position_1.y);
            float _S260 = atomicAdd(&((&_S256->position_1)->z), _S254.position_1.z);
            float _S261 = atomicAdd(&((&_S256->quaternion_0)->x), _S254.quaternion_0.x);
            float _S262 = atomicAdd(&((&_S256->quaternion_0)->y), _S254.quaternion_0.y);
            float _S263 = atomicAdd(&((&_S256->quaternion_0)->z), _S254.quaternion_0.z);
            float _S264 = atomicAdd(&((&_S256->quaternion_0)->w), _S254.quaternion_0.w);
            float _S265 = atomicAdd(&((&_S256->scale_1)->x), _S254.scale_1.x);
            float _S266 = atomicAdd(&((&_S256->scale_1)->y), _S254.scale_1.y);
            float _S267 = atomicAdd(&((&_S256->scale_1)->z), _S254.scale_1.z);
        }
        *transmittanceGrad_0 = dptransmittance_0;
        *integratedDepthGrad_0 = dpintegratedDepth_0;
        if(enableNormal_3)
        {
            *integratedNormalGrad_0 = dpintegratedNormal_0;
        }
    }
    return;
}

__device__ bool particleDensityHitCustom(float3  rayOrigin_3, float3  rayDirection_3, int particleIdx_3, gaussianParticle_CommonParameters_0 commonParameters_3, float minHitDistance_0, float maxHitDistance_0, float maxParticleSquaredDistance_0, float * hitDistance_0)
{
    gaussianParticle_RawParameters_0 * _S268 = commonParameters_3.parametersBuffer_0._dataPtr_0 + uint(particleIdx_3);
    gaussianParticle_Parameters_0 _S269 = gaussianParticle_Parameters_x24init_0((*_S268).position_1, (*_S268).scale_1, transforms_rotationMatrixTranspose_0((*_S268).quaternion_0), (*_S268).density_1);
    float3  giscl_3 = make_float3 (1.0f) / _S269.scale_0;
    float3  canonicalRayOrigin_3 = giscl_3 * mul_0(_S269.rotationT_0, rayOrigin_3 - _S269.position_0);
    float3  canonicalRayDirection_2 = normalize_0(giscl_3 * mul_0(_S269.rotationT_0, rayDirection_3));
    float3  grds_3 = _S269.scale_0 * canonicalRayDirection_2 * make_float3 (dot_0(canonicalRayDirection_2, make_float3 (-1.0f) * canonicalRayOrigin_3));
    float _S270 = (F32_sqrt((dot_0(grds_3, grds_3))));
    *hitDistance_0 = _S270;
    bool _S271;
    if(_S270 > minHitDistance_0)
    {
        _S271 = (*hitDistance_0) < maxHitDistance_0;
    }
    else
    {
        _S271 = false;
    }
    if(_S271)
    {
        float3  gcrod_2 = cross_0(canonicalRayDirection_2, canonicalRayOrigin_3);
        _S271 = (dot_0(gcrod_2, gcrod_2)) < maxParticleSquaredDistance_0;
    }
    else
    {
        _S271 = false;
    }
    return _S271;
}

__device__ float rcp_0(float x_3)
{
    return 1.0f / x_3;
}

__device__ bool particleDensityHitInstance(float3  canonicalRayOrigin_4, float3  canonicalUnormalizedRayDirection_0, float minHitDistance_1, float maxHitDistance_1, float maxParticleSquaredDistance_1, float * hitDistance_1)
{
    float _S272 = - dot_0(canonicalRayOrigin_4, canonicalUnormalizedRayDirection_0) * rcp_0(dot_0(canonicalUnormalizedRayDirection_0, canonicalUnormalizedRayDirection_0));
    *hitDistance_1 = _S272;
    bool _S273;
    if(_S272 > minHitDistance_1)
    {
        _S273 = (*hitDistance_1) < maxHitDistance_1;
    }
    else
    {
        _S273 = false;
    }
    if(_S273)
    {
        float3  gcrod_3 = cross_0(normalize_0(canonicalUnormalizedRayDirection_0), canonicalRayOrigin_4);
        _S273 = (dot_0(gcrod_3, gcrod_3)) < maxParticleSquaredDistance_1;
    }
    else
    {
        _S273 = false;
    }
    return _S273;
}

__device__ float3  particleDensityIncidentDirection(gaussianParticle_Parameters_0 parameters_1, float3  sourcePosition_0)
{
    return normalize_0(parameters_1.position_0 - sourcePosition_0);
}

struct s_bwd_prop_gaussianParticle_incidentDirectionFromBuffer_Intermediates_0
{
    gaussianParticle_RawParameters_0 _S274;
};

__device__ void particleDensityIncidentDirectionBwdToBuffer(uint particleIdx_4, gaussianParticle_CommonParameters_0 commonParameters_4, float3  sourcePosition_1, float3  incidentDirectionGrad_0)
{
    float3  _S275 = make_float3 (0.0f);
    float4  _S276 = make_float4 (0.0f);
    gaussianParticle_RawParameters_0 _S277 = { _S275, 0.0f, _S276, _S275, 0.0f };
    s_bwd_prop_gaussianParticle_incidentDirectionFromBuffer_Intermediates_0 _S278;
    (&_S278)->_S274 = _S277;
    (&(&_S278)->_S274)->position_1 = _S275;
    (&(&_S278)->_S274)->density_1 = 0.0f;
    (&(&_S278)->_S274)->quaternion_0 = _S276;
    (&(&_S278)->_S274)->scale_1 = _S275;
    (&(&_S278)->_S274)->padding_0 = 0.0f;
    (&_S278)->_S274 = *(commonParameters_4.parametersBuffer_0._dataPtr_0 + particleIdx_4);
    Matrix<float, 3, 3>  _S279 = s_primal_ctx_transforms_rotationMatrixTranspose_0(_S278._S274.quaternion_0);
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S280;
    (&_S280)->primal_0 = s_primal_ctx_gaussianParticle_Parameters_x24init_0(_S278._S274.position_1, _S278._S274.scale_1, _S279, _S278._S274.density_1).position_0 - sourcePosition_1;
    (&_S280)->differential_0 = _S275;
    s_bwd_normalize_impl_0(&_S280, incidentDirectionGrad_0);
    gaussianParticle_Parameters_0 _S281 = gaussianParticle_Parameters_x24_syn_dzero_0();
    (&_S281)->position_0 = _S280.differential_0;
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S282;
    (&_S282)->primal_0 = _S278._S274.position_1;
    (&_S282)->differential_0 = _S275;
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S283;
    (&_S283)->primal_0 = _S278._S274.scale_1;
    (&_S283)->differential_0 = _S275;
    Matrix<float, 3, 3>  _S284 = makeMatrix<float, 3, 3> (0.0f);
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S285;
    (&_S285)->primal_0 = _S279;
    (&_S285)->differential_0 = _S284;
    DiffPair_float_0 _S286;
    (&_S286)->primal_0 = _S278._S274.density_1;
    (&_S286)->differential_0 = 0.0f;
    gaussianParticle_Parameters_0 _S287 = _S281;
    s_bwd_prop_gaussianParticle_Parameters_x24init_0(&_S282, &_S283, &_S285, &_S286, &_S287);
    DiffPair_vectorx3Cfloatx2C4x3E_0 _S288;
    (&_S288)->primal_0 = _S278._S274.quaternion_0;
    (&_S288)->differential_0 = _S276;
    s_bwd_prop_transforms_rotationMatrixTranspose_0(&_S288, _S285.differential_0);
    gaussianParticle_RawParameters_0 _S289 = gaussianParticle_RawParameters_x24_syn_dzero_0();
    (&_S289)->density_1 = _S286.differential_0;
    (&_S289)->quaternion_0 = _S288.differential_0;
    (&_S289)->scale_1 = _S283.differential_0;
    (&_S289)->position_1 = _S282.differential_0;
    gaussianParticle_RawParameters_0 _S290 = _S289;
    if(commonParameters_4.parametersBuffer_0.exclusiveGradient_0)
    {
        gaussianParticle_RawParameters_0 * _S291 = commonParameters_4.parametersBuffer_0._gradPtr_0 + particleIdx_4;
        _S291->density_1 = _S291->density_1 + _S290.density_1;
        _S291->position_1 = _S291->position_1 + _S290.position_1;
        _S291->quaternion_0 = _S291->quaternion_0 + _S290.quaternion_0;
        _S291->scale_1 = _S291->scale_1 + _S290.scale_1;
    }
    else
    {
        gaussianParticle_RawParameters_0 * _S292 = commonParameters_4.parametersBuffer_0._gradPtr_0 + particleIdx_4;
        float _S293 = atomicAdd(&_S292->density_1, _S290.density_1);
        float _S294 = atomicAdd(&((&_S292->position_1)->x), _S290.position_1.x);
        float _S295 = atomicAdd(&((&_S292->position_1)->y), _S290.position_1.y);
        float _S296 = atomicAdd(&((&_S292->position_1)->z), _S290.position_1.z);
        float _S297 = atomicAdd(&((&_S292->quaternion_0)->x), _S290.quaternion_0.x);
        float _S298 = atomicAdd(&((&_S292->quaternion_0)->y), _S290.quaternion_0.y);
        float _S299 = atomicAdd(&((&_S292->quaternion_0)->z), _S290.quaternion_0.z);
        float _S300 = atomicAdd(&((&_S292->quaternion_0)->w), _S290.quaternion_0.w);
        float _S301 = atomicAdd(&((&_S292->scale_1)->x), _S290.scale_1.x);
        float _S302 = atomicAdd(&((&_S292->scale_1)->y), _S290.scale_1.y);
        float _S303 = atomicAdd(&((&_S292->scale_1)->z), _S290.scale_1.z);
    }
    return;
}

__device__ void _d_max_vector_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpx_9, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpy_5, float3  dOut_10)
{
    DiffPair_float_0 left_dp_3;
    (&left_dp_3)->primal_0 = (*dpx_9).primal_0.x;
    (&left_dp_3)->differential_0 = 0.0f;
    DiffPair_float_0 right_dp_3;
    (&right_dp_3)->primal_0 = (*dpy_5).primal_0.x;
    (&right_dp_3)->differential_0 = 0.0f;
    _d_max_0(&left_dp_3, &right_dp_3, dOut_10.x);
    float3  left_d_result_3;
    *&((&left_d_result_3)->x) = left_dp_3.differential_0;
    float3  right_d_result_3;
    *&((&right_d_result_3)->x) = right_dp_3.differential_0;
    DiffPair_float_0 left_dp_4;
    (&left_dp_4)->primal_0 = (*dpx_9).primal_0.y;
    (&left_dp_4)->differential_0 = 0.0f;
    DiffPair_float_0 right_dp_4;
    (&right_dp_4)->primal_0 = (*dpy_5).primal_0.y;
    (&right_dp_4)->differential_0 = 0.0f;
    _d_max_0(&left_dp_4, &right_dp_4, dOut_10.y);
    *&((&left_d_result_3)->y) = left_dp_4.differential_0;
    *&((&right_d_result_3)->y) = right_dp_4.differential_0;
    DiffPair_float_0 left_dp_5;
    (&left_dp_5)->primal_0 = (*dpx_9).primal_0.z;
    (&left_dp_5)->differential_0 = 0.0f;
    DiffPair_float_0 right_dp_5;
    (&right_dp_5)->primal_0 = (*dpy_5).primal_0.z;
    (&right_dp_5)->differential_0 = 0.0f;
    _d_max_0(&left_dp_5, &right_dp_5, dOut_10.z);
    *&((&left_d_result_3)->z) = left_dp_5.differential_0;
    *&((&right_d_result_3)->z) = right_dp_5.differential_0;
    dpx_9->primal_0 = (*dpx_9).primal_0;
    dpx_9->differential_0 = left_d_result_3;
    dpy_5->primal_0 = (*dpy_5).primal_0;
    dpy_5->differential_0 = right_d_result_3;
    return;
}

__device__ float3  max_0(float3  x_4, float3  y_1)
{
    float3  result_8;
    int i_3 = int(0);
    for(;;)
    {
        if(i_3 < int(3))
        {
        }
        else
        {
            break;
        }
        *_slang_vector_get_element_ptr(&result_8, i_3) = (F32_max((_slang_vector_get_element(x_4, i_3)), (_slang_vector_get_element(y_1, i_3))));
        i_3 = i_3 + int(1);
    }
    return result_8;
}

__device__ float3  sphericalHarmonics_decode_0(int degree_0, FixedArray<float3 , 16>  * coefficients_0, float3  direction_0)
{
    float3  features_0 = make_float3 (0.282094806432724f) * (*coefficients_0)[int(0)];
    float3  features_1;
    if(degree_0 > int(0))
    {
        float x_5 = direction_0.x;
        float y_2 = direction_0.y;
        float z_0 = direction_0.z;
        float3  features_2 = features_0 - make_float3 (0.48860251903533936f * y_2) * (*coefficients_0)[int(1)] + make_float3 (0.48860251903533936f * z_0) * (*coefficients_0)[int(2)] - make_float3 (0.48860251903533936f * x_5) * (*coefficients_0)[int(3)];
        if(degree_0 > int(1))
        {
            float xx_2 = x_5 * x_5;
            float yy_2 = y_2 * y_2;
            float zz_2 = z_0 * z_0;
            float xy_2 = x_5 * y_2;
            float _S304 = 2.0f * zz_2;
            float _S305 = xx_2 - yy_2;
            float3  features_3 = features_2 + make_float3 (1.09254848957061768f * xy_2) * (*coefficients_0)[int(4)] + make_float3 (-1.09254848957061768f * (y_2 * z_0)) * (*coefficients_0)[int(5)] + make_float3 (0.31539157032966614f * (_S304 - xx_2 - yy_2)) * (*coefficients_0)[int(6)] + make_float3 (-1.09254848957061768f * (x_5 * z_0)) * (*coefficients_0)[int(7)] + make_float3 (0.54627424478530884f * _S305) * (*coefficients_0)[int(8)];
            if(degree_0 > int(2))
            {
                float _S306 = 3.0f * xx_2;
                float _S307 = 4.0f * zz_2 - xx_2 - yy_2;
                float _S308 = 3.0f * yy_2;
                features_1 = features_3 + make_float3 (-0.59004360437393188f * y_2 * (_S306 - yy_2)) * (*coefficients_0)[int(9)] + make_float3 (2.89061141014099121f * xy_2 * z_0) * (*coefficients_0)[int(10)] + make_float3 (-0.4570457935333252f * y_2 * _S307) * (*coefficients_0)[int(11)] + make_float3 (0.37317633628845215f * z_0 * (_S304 - _S306 - _S308)) * (*coefficients_0)[int(12)] + make_float3 (-0.4570457935333252f * x_5 * _S307) * (*coefficients_0)[int(13)] + make_float3 (1.44530570507049561f * z_0 * _S305) * (*coefficients_0)[int(14)] + make_float3 (-0.59004360437393188f * x_5 * (xx_2 - _S308)) * (*coefficients_0)[int(15)];
            }
            else
            {
                features_1 = features_3;
            }
        }
        else
        {
            features_1 = features_2;
        }
    }
    else
    {
        features_1 = features_0;
    }
    int3  _S309 = make_int3 (int(0));
    float3  _S310 = make_float3 ((float)_S309.x, (float)_S309.y, (float)_S309.z);
    return max_0(features_1 + make_float3 (0.5f), _S310);
}

struct shRadiativeParticle_ParametersBuffer_0
{
    float3  * _dataPtr_1;
    float3  * _gradPtr_1;
    bool exclusiveGradient_1;
};

struct shRadiativeParticle_CommonParameters_0
{
    shRadiativeParticle_ParametersBuffer_0 parametersBuffer_1;
    int sphDegree_0;
};

__device__ float3  particleFeaturesFromBuffer(uint particleIdx_5, shRadiativeParticle_CommonParameters_0 commonParameters_5, float3  incidentDirection_0)
{
    shRadiativeParticle_Parameters_0 _S311;
    for(;;)
    {
        shRadiativeParticle_Parameters_0 parameters_2;
        uint _S312 = particleIdx_5 * 16U;
        int i_4 = int(0);
        #pragma unroll
        for(;;)
        {
            if(i_4 < int(16))
            {
            }
            else
            {
                break;
            }
            (&parameters_2)->sphCoefficients_0[i_4] = *(commonParameters_5.parametersBuffer_1._dataPtr_1 + (_S312 + uint(i_4)));
            i_4 = i_4 + int(1);
        }
        _S311 = parameters_2;
        break;
    }
    FixedArray<float3 , 16>  _S313 = _S311.sphCoefficients_0;
    float3  _S314 = sphericalHarmonics_decode_0(commonParameters_5.sphDegree_0, &_S313, incidentDirection_0);
    return _S314;
}

__device__ void particleFeaturesIntegrateFwd(float weight_1, float3  features_4, float3  * integratedFeatures_0)
{
    if(weight_1 > 0.0f)
    {
        *integratedFeatures_0 = *integratedFeatures_0 + features_4 * make_float3 (weight_1);
    }
    return;
}

__device__ void particleFeaturesIntegrateFwdFromBuffer(float3  incidentDirection_1, float weight_2, uint particleIdx_6, shRadiativeParticle_CommonParameters_0 commonParameters_6, float3  * integratedFeatures_1)
{
    shRadiativeParticle_Parameters_0 _S315;
    uint _S316 = uint(commonParameters_6.sphDegree_0);
    for(;;)
    {
        for(;;)
        {
            shRadiativeParticle_Parameters_0 parameters_3;
            uint _S317 = particleIdx_6 * 16U;
            int i_5 = int(0);
            #pragma unroll
            for(;;)
            {
                if(i_5 < int(16))
                {
                }
                else
                {
                    break;
                }
                (&parameters_3)->sphCoefficients_0[i_5] = *(commonParameters_6.parametersBuffer_1._dataPtr_1 + (_S317 + uint(i_5)));
                i_5 = i_5 + int(1);
            }
            _S315 = parameters_3;
            break;
        }
        bool _S318 = weight_2 > 0.0f;
        if(_S318)
        {
            int _S319 = int(_S316);
            FixedArray<float3 , 16>  _S320 = _S315.sphCoefficients_0;
            float3  _S321 = sphericalHarmonics_decode_0(_S319, &_S320, incidentDirection_1);
            if(_S318)
            {
                *integratedFeatures_1 = *integratedFeatures_1 + _S321 * make_float3 (weight_2);
            }
        }
        break;
    }
    return;
}

struct s_bwd_prop_shRadiativeParticle_integrateRadiance_Intermediates_0
{
    float3  _S322;
};

__device__ void particleFeaturesIntegrateBwd(float alpha_4, float * alphaGrad_1, float3  features_5, float3  * featuresGrad_0, float3  * integratedFeatures_2, float3  * integratedFeaturesGrad_0)
{
    bool _S323 = alpha_4 > 0.0f;
    if(_S323)
    {
        float3  _S324 = (*integratedFeatures_2 - features_5 * make_float3 (alpha_4)) * make_float3 (1.0f / (1.0f - alpha_4));
        *integratedFeatures_2 = _S324;
        float3  _S325 = *integratedFeaturesGrad_0;
        float3  _S326 = make_float3 (0.0f);
        s_bwd_prop_shRadiativeParticle_integrateRadiance_Intermediates_0 _S327;
        (&_S327)->_S322 = _S326;
        (&_S327)->_S322 = _S324;
        s_bwd_prop_shRadiativeParticle_integrateRadiance_Intermediates_0 _S328 = _S327;
        float3  dpintegratedRadiance_0;
        if(_S323)
        {
            dpintegratedRadiance_0 = make_float3 (alpha_4);
        }
        else
        {
            dpintegratedRadiance_0 = _S326;
        }
        float3  _S329;
        float _S330;
        if(_S323)
        {
            DiffPair_vectorx3Cfloatx2C3x3E_0 _S331;
            (&_S331)->primal_0 = _S328._S322;
            (&_S331)->differential_0 = _S326;
            DiffPair_vectorx3Cfloatx2C3x3E_0 _S332;
            (&_S332)->primal_0 = features_5;
            (&_S332)->differential_0 = _S326;
            DiffPair_vectorx3Cfloatx2C3x3E_0 _S333;
            (&_S333)->primal_0 = dpintegratedRadiance_0;
            (&_S333)->differential_0 = _S326;
            s_bwd_prop_lerp_0(&_S331, &_S332, &_S333, _S325);
            float _S334 = _S333.differential_0.x + _S333.differential_0.y + _S333.differential_0.z;
            dpintegratedRadiance_0 = _S331.differential_0;
            _S329 = _S332.differential_0;
            _S330 = _S334;
        }
        else
        {
            dpintegratedRadiance_0 = _S325;
            _S329 = _S326;
            _S330 = 0.0f;
        }
        *alphaGrad_1 = _S330;
        *featuresGrad_0 = _S329;
        *integratedFeaturesGrad_0 = dpintegratedRadiance_0;
    }
    return;
}

struct s_bwd_prop_shRadiativeParticle_integrateRadianceFromBuffer_Intermediates_0
{
    float3  _S335;
    shRadiativeParticle_Parameters_0 _S336;
};

__device__ float3  s_primal_ctx_max_0(float3  _S337, float3  _S338)
{
    return max_0(_S337, _S338);
}

__device__ float3  s_primal_ctx_sphericalHarmonics_decode_0(int degree_1, FixedArray<float3 , 16>  * dpcoefficients_0, float3  dpdirection_0)
{
    float3  features_6 = make_float3 (0.282094806432724f) * (*dpcoefficients_0)[int(0)];
    float3  features_7;
    if(degree_1 > int(0))
    {
        float x_6 = dpdirection_0.x;
        float y_3 = dpdirection_0.y;
        float z_1 = dpdirection_0.z;
        float3  features_8 = features_6 - make_float3 (0.48860251903533936f * y_3) * (*dpcoefficients_0)[int(1)] + make_float3 (0.48860251903533936f * z_1) * (*dpcoefficients_0)[int(2)] - make_float3 (0.48860251903533936f * x_6) * (*dpcoefficients_0)[int(3)];
        if(degree_1 > int(1))
        {
            float xx_3 = x_6 * x_6;
            float yy_3 = y_3 * y_3;
            float zz_3 = z_1 * z_1;
            float xy_3 = x_6 * y_3;
            float _S339 = 2.0f * zz_3;
            float _S340 = xx_3 - yy_3;
            float3  features_9 = features_8 + make_float3 (1.09254848957061768f * xy_3) * (*dpcoefficients_0)[int(4)] + make_float3 (-1.09254848957061768f * (y_3 * z_1)) * (*dpcoefficients_0)[int(5)] + make_float3 (0.31539157032966614f * (_S339 - xx_3 - yy_3)) * (*dpcoefficients_0)[int(6)] + make_float3 (-1.09254848957061768f * (x_6 * z_1)) * (*dpcoefficients_0)[int(7)] + make_float3 (0.54627424478530884f * _S340) * (*dpcoefficients_0)[int(8)];
            if(degree_1 > int(2))
            {
                float _S341 = 3.0f * xx_3;
                float _S342 = 4.0f * zz_3 - xx_3 - yy_3;
                float _S343 = 3.0f * yy_3;
                features_7 = features_9 + make_float3 (-0.59004360437393188f * y_3 * (_S341 - yy_3)) * (*dpcoefficients_0)[int(9)] + make_float3 (2.89061141014099121f * xy_3 * z_1) * (*dpcoefficients_0)[int(10)] + make_float3 (-0.4570457935333252f * y_3 * _S342) * (*dpcoefficients_0)[int(11)] + make_float3 (0.37317633628845215f * z_1 * (_S339 - _S341 - _S343)) * (*dpcoefficients_0)[int(12)] + make_float3 (-0.4570457935333252f * x_6 * _S342) * (*dpcoefficients_0)[int(13)] + make_float3 (1.44530570507049561f * z_1 * _S340) * (*dpcoefficients_0)[int(14)] + make_float3 (-0.59004360437393188f * x_6 * (xx_3 - _S343)) * (*dpcoefficients_0)[int(15)];
            }
            else
            {
                features_7 = features_9;
            }
        }
        else
        {
            features_7 = features_8;
        }
    }
    else
    {
        features_7 = features_6;
    }
    int3  _S344 = make_int3 (int(0));
    float3  _S345 = make_float3 ((float)_S344.x, (float)_S344.y, (float)_S344.z);
    return s_primal_ctx_max_0(features_7 + make_float3 (0.5f), _S345);
}

struct DiffPair_arrayx3Cvectorx3Cfloatx2C3x3Ex2C16x3E_0
{
    FixedArray<float3 , 16>  primal_0;
    FixedArray<float3 , 16>  differential_0;
};

__device__ void s_bwd_prop_max_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S346, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S347, float3  _S348)
{
    _d_max_vector_0(_S346, _S347, _S348);
    return;
}

__device__ void s_bwd_prop_sphericalHarmonics_decode_0(int degree_2, DiffPair_arrayx3Cvectorx3Cfloatx2C3x3Ex2C16x3E_0 * dpcoefficients_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpdirection_1, float3  _s_dOut_4)
{
    FixedArray<float3 , 16>  _S349 = dpcoefficients_1->primal_0;
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S350 = *dpdirection_1;
    float3  _S351 = make_float3 (0.0f);
    float3  features_10 = make_float3 (0.282094806432724f) * dpcoefficients_1->primal_0[int(0)];
    bool _S352 = degree_2 > int(0);
    float3  features_11;
    float3  _S353;
    float3  _S354;
    float3  _S355;
    float3  _S356;
    float3  _S357;
    float3  _S358;
    float3  _S359;
    float3  _S360;
    float3  _S361;
    float3  _S362;
    float3  _S363;
    float3  _S364;
    float3  _S365;
    float3  _S366;
    float3  _S367;
    float3  _S368;
    float3  _S369;
    float3  _S370;
    float3  _S371;
    float3  _S372;
    float3  _S373;
    float3  _S374;
    float3  _S375;
    float3  _S376;
    float3  _S377;
    float3  _S378;
    float3  _S379;
    float3  _S380;
    float3  _S381;
    float3  _S382;
    float _S383;
    float _S384;
    float _S385;
    float _S386;
    float _S387;
    float _S388;
    float _S389;
    float _S390;
    float _S391;
    float _S392;
    float _S393;
    float _S394;
    float _S395;
    float _S396;
    float _S397;
    bool _S398;
    bool _S399;
    if(_S352)
    {
        float x_7 = _S350.primal_0.x;
        float y_4 = _S350.primal_0.y;
        float z_2 = _S350.primal_0.z;
        float _S400 = 0.48860251903533936f * y_4;
        float3  _S401 = make_float3 (_S400);
        float _S402 = 0.48860251903533936f * z_2;
        float3  _S403 = make_float3 (_S402);
        float _S404 = 0.48860251903533936f * x_7;
        float3  _S405 = make_float3 (_S404);
        float3  features_12 = features_10 - make_float3 (_S400) * _S349[int(1)] + make_float3 (_S402) * _S349[int(2)] - make_float3 (_S404) * _S349[int(3)];
        bool _S406 = degree_2 > int(1);
        if(_S406)
        {
            float xx_4 = x_7 * x_7;
            float yy_4 = y_4 * y_4;
            float zz_4 = z_2 * z_2;
            float xy_4 = x_7 * y_4;
            float _S407 = 1.09254848957061768f * xy_4;
            float3  _S408 = make_float3 (_S407);
            float _S409 = -1.09254848957061768f * (y_4 * z_2);
            float3  _S410 = make_float3 (_S409);
            float _S411 = 2.0f * zz_4;
            float _S412 = 0.31539157032966614f * (_S411 - xx_4 - yy_4);
            float3  _S413 = make_float3 (_S412);
            float _S414 = -1.09254848957061768f * (x_7 * z_2);
            float3  _S415 = make_float3 (_S414);
            float _S416 = xx_4 - yy_4;
            float _S417 = 0.54627424478530884f * _S416;
            float3  _S418 = make_float3 (_S417);
            float3  features_13 = features_12 + make_float3 (_S407) * _S349[int(4)] + make_float3 (_S409) * _S349[int(5)] + make_float3 (_S412) * _S349[int(6)] + make_float3 (_S414) * _S349[int(7)] + make_float3 (_S417) * _S349[int(8)];
            bool _S419 = degree_2 > int(2);
            if(_S419)
            {
                float _S420 = -0.59004360437393188f * y_4;
                float _S421 = 3.0f * xx_4;
                float _S422 = _S421 - yy_4;
                float _S423 = _S420 * _S422;
                float3  _S424 = make_float3 (_S423);
                float _S425 = 2.89061141014099121f * xy_4;
                float _S426 = _S425 * z_2;
                float3  _S427 = make_float3 (_S426);
                float _S428 = -0.4570457935333252f * y_4;
                float _S429 = 4.0f * zz_4 - xx_4 - yy_4;
                float _S430 = _S428 * _S429;
                float3  _S431 = make_float3 (_S430);
                float _S432 = 0.37317633628845215f * z_2;
                float _S433 = 3.0f * yy_4;
                float _S434 = _S411 - _S421 - _S433;
                float _S435 = _S432 * _S434;
                float3  _S436 = make_float3 (_S435);
                float _S437 = -0.4570457935333252f * x_7;
                float _S438 = _S437 * _S429;
                float3  _S439 = make_float3 (_S438);
                float _S440 = 1.44530570507049561f * z_2;
                float _S441 = _S440 * _S416;
                float3  _S442 = make_float3 (_S441);
                float _S443 = -0.59004360437393188f * x_7;
                float _S444 = xx_4 - _S433;
                float _S445 = _S443 * _S444;
                float3  _S446 = make_float3 (_S445);
                features_11 = features_13 + make_float3 (_S423) * _S349[int(9)] + make_float3 (_S426) * _S349[int(10)] + make_float3 (_S430) * _S349[int(11)] + make_float3 (_S435) * _S349[int(12)] + make_float3 (_S438) * _S349[int(13)] + make_float3 (_S441) * _S349[int(14)] + make_float3 (_S445) * _S349[int(15)];
                _S353 = _S446;
                _S354 = _S349[int(15)];
                _S383 = _S443;
                _S384 = _S444;
                _S355 = _S442;
                _S356 = _S349[int(14)];
                _S385 = _S440;
                _S357 = _S439;
                _S358 = _S349[int(13)];
                _S386 = _S437;
                _S387 = _S429;
                _S359 = _S436;
                _S360 = _S349[int(12)];
                _S388 = _S432;
                _S389 = _S434;
                _S361 = _S431;
                _S362 = _S349[int(11)];
                _S390 = _S428;
                _S363 = _S427;
                _S364 = _S349[int(10)];
                _S391 = _S425;
                _S365 = _S424;
                _S366 = _S349[int(9)];
                _S392 = _S420;
                _S393 = _S422;
            }
            else
            {
                features_11 = features_13;
                _S353 = _S351;
                _S354 = _S351;
                _S383 = 0.0f;
                _S384 = 0.0f;
                _S355 = _S351;
                _S356 = _S351;
                _S385 = 0.0f;
                _S357 = _S351;
                _S358 = _S351;
                _S386 = 0.0f;
                _S387 = 0.0f;
                _S359 = _S351;
                _S360 = _S351;
                _S388 = 0.0f;
                _S389 = 0.0f;
                _S361 = _S351;
                _S362 = _S351;
                _S390 = 0.0f;
                _S363 = _S351;
                _S364 = _S351;
                _S391 = 0.0f;
                _S365 = _S351;
                _S366 = _S351;
                _S392 = 0.0f;
                _S393 = 0.0f;
            }
            float _S447 = _S386;
            float _S448 = _S387;
            float _S449 = _S388;
            float _S450 = _S389;
            float _S451 = _S390;
            float _S452 = _S391;
            float _S453 = _S392;
            float _S454 = _S393;
            _S398 = _S419;
            _S386 = _S416;
            _S387 = _S447;
            _S388 = _S448;
            _S389 = _S449;
            _S390 = _S450;
            _S391 = _S451;
            _S392 = _S452;
            _S393 = _S453;
            _S394 = _S454;
            _S367 = _S418;
            _S368 = _S349[int(8)];
            _S369 = _S415;
            _S370 = _S349[int(7)];
            _S371 = _S413;
            _S372 = _S349[int(6)];
            _S373 = _S410;
            _S374 = _S349[int(5)];
            _S375 = _S408;
            _S376 = _S349[int(4)];
        }
        else
        {
            features_11 = features_12;
            _S398 = false;
            _S353 = _S351;
            _S354 = _S351;
            _S383 = 0.0f;
            _S384 = 0.0f;
            _S355 = _S351;
            _S356 = _S351;
            _S385 = 0.0f;
            _S386 = 0.0f;
            _S357 = _S351;
            _S358 = _S351;
            _S387 = 0.0f;
            _S388 = 0.0f;
            _S359 = _S351;
            _S360 = _S351;
            _S389 = 0.0f;
            _S390 = 0.0f;
            _S361 = _S351;
            _S362 = _S351;
            _S391 = 0.0f;
            _S363 = _S351;
            _S364 = _S351;
            _S392 = 0.0f;
            _S365 = _S351;
            _S366 = _S351;
            _S393 = 0.0f;
            _S394 = 0.0f;
            _S367 = _S351;
            _S368 = _S351;
            _S369 = _S351;
            _S370 = _S351;
            _S371 = _S351;
            _S372 = _S351;
            _S373 = _S351;
            _S374 = _S351;
            _S375 = _S351;
            _S376 = _S351;
        }
        bool _S455 = _S398;
        float _S456 = _S393;
        float _S457 = _S394;
        _S398 = _S406;
        _S399 = _S455;
        _S393 = z_2;
        _S394 = _S456;
        _S395 = _S457;
        _S396 = x_7;
        _S397 = y_4;
        _S377 = _S405;
        _S378 = _S349[int(3)];
        _S379 = _S403;
        _S380 = _S349[int(2)];
        _S381 = _S401;
        _S382 = _S349[int(1)];
    }
    else
    {
        features_11 = features_10;
        _S398 = false;
        _S399 = false;
        _S353 = _S351;
        _S354 = _S351;
        _S383 = 0.0f;
        _S384 = 0.0f;
        _S355 = _S351;
        _S356 = _S351;
        _S385 = 0.0f;
        _S386 = 0.0f;
        _S357 = _S351;
        _S358 = _S351;
        _S387 = 0.0f;
        _S388 = 0.0f;
        _S359 = _S351;
        _S360 = _S351;
        _S389 = 0.0f;
        _S390 = 0.0f;
        _S361 = _S351;
        _S362 = _S351;
        _S391 = 0.0f;
        _S363 = _S351;
        _S364 = _S351;
        _S392 = 0.0f;
        _S393 = 0.0f;
        _S365 = _S351;
        _S366 = _S351;
        _S394 = 0.0f;
        _S395 = 0.0f;
        _S367 = _S351;
        _S368 = _S351;
        _S369 = _S351;
        _S370 = _S351;
        _S371 = _S351;
        _S372 = _S351;
        _S373 = _S351;
        _S374 = _S351;
        _S375 = _S351;
        _S376 = _S351;
        _S396 = 0.0f;
        _S397 = 0.0f;
        _S377 = _S351;
        _S378 = _S351;
        _S379 = _S351;
        _S380 = _S351;
        _S381 = _S351;
        _S382 = _S351;
    }
    float3  _S458 = features_11 + make_float3 (0.5f);
    int3  _S459 = make_int3 (int(0));
    float3  _S460 = make_float3 ((float)_S459.x, (float)_S459.y, (float)_S459.z);
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S461;
    (&_S461)->primal_0 = _S458;
    (&_S461)->differential_0 = _S351;
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S462;
    (&_S462)->primal_0 = _S460;
    (&_S462)->differential_0 = _S351;
    s_bwd_prop_max_0(&_S461, &_S462, _s_dOut_4);
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S463 = _S461;
    FixedArray<float3 , 16>  _S464;
    if(_S352)
    {
        if(_S398)
        {
            if(_S399)
            {
                float3  _S465 = _S353 * _S463.differential_0;
                float3  _S466 = _S354 * _S463.differential_0;
                float _S467 = _S466.x + _S466.y + _S466.z;
                float _S468 = _S383 * _S467;
                float3  _S469 = _S355 * _S463.differential_0;
                float3  _S470 = _S356 * _S463.differential_0;
                float _S471 = _S470.x + _S470.y + _S470.z;
                float _S472 = _S385 * _S471;
                float _S473 = 1.44530570507049561f * (_S386 * _S471);
                float3  _S474 = _S357 * _S463.differential_0;
                float3  _S475 = _S358 * _S463.differential_0;
                float _S476 = _S475.x + _S475.y + _S475.z;
                float3  _S477 = _S359 * _S463.differential_0;
                float3  _S478 = _S360 * _S463.differential_0;
                float _S479 = _S478.x + _S478.y + _S478.z;
                float _S480 = _S389 * _S479;
                float _S481 = - _S480;
                float _S482 = 3.0f * (- _S468 + _S481);
                float _S483 = 0.37317633628845215f * (_S390 * _S479);
                float3  _S484 = _S361 * _S463.differential_0;
                float3  _S485 = _S362 * _S463.differential_0;
                float _S486 = _S485.x + _S485.y + _S485.z;
                float _S487 = _S387 * _S476 + _S391 * _S486;
                float _S488 = - _S487;
                float _S489 = 4.0f * _S487;
                float _S490 = -0.4570457935333252f * (_S388 * _S486);
                float3  _S491 = _S363 * _S463.differential_0;
                float3  _S492 = _S364 * _S463.differential_0;
                float _S493 = _S492.x + _S492.y + _S492.z;
                float _S494 = _S392 * _S493;
                float _S495 = 2.89061141014099121f * (_S393 * _S493);
                float3  _S496 = _S365 * _S463.differential_0;
                float3  _S497 = _S366 * _S463.differential_0;
                float _S498 = _S497.x + _S497.y + _S497.z;
                float _S499 = _S394 * _S498;
                float _S500 = - _S499;
                float _S501 = 3.0f * (_S481 + _S499);
                float _S502 = -0.59004360437393188f * (_S395 * _S498);
                float _S503 = -0.59004360437393188f * (_S384 * _S467) + -0.4570457935333252f * (_S388 * _S476);
                FixedArray<float3 , 16>  _S504;
                _S504[int(0)] = _S351;
                _S504[int(1)] = _S351;
                _S504[int(2)] = _S351;
                _S504[int(3)] = _S351;
                _S504[int(4)] = _S351;
                _S504[int(5)] = _S351;
                _S504[int(6)] = _S351;
                _S504[int(7)] = _S351;
                _S504[int(8)] = _S351;
                _S504[int(9)] = _S351;
                _S504[int(10)] = _S351;
                _S504[int(11)] = _S351;
                _S504[int(12)] = _S351;
                _S504[int(13)] = _S351;
                _S504[int(14)] = _S351;
                _S504[int(15)] = _S351;
                _S504[int(15)] = _S465;
                _S504[int(14)] = _S469;
                _S504[int(13)] = _S474;
                _S504[int(12)] = _S477;
                _S504[int(11)] = _S484;
                _S504[int(10)] = _S491;
                _S504[int(9)] = _S496;
                float _S505 = _S482 + _S488 + _S500;
                float _S506 = _S468 + _S488 + _S501;
                float _S507 = _S473 + _S483 + _S494;
                float _S508 = _S490 + _S502;
                _S383 = _S472;
                _S384 = _S480;
                _S385 = _S495;
                _S386 = _S489;
                _S387 = _S505;
                _S388 = _S506;
                _S464[int(0)] = _S504[int(0)];
                _S464[int(1)] = _S504[int(1)];
                _S464[int(2)] = _S504[int(2)];
                _S464[int(3)] = _S504[int(3)];
                _S464[int(4)] = _S504[int(4)];
                _S464[int(5)] = _S504[int(5)];
                _S464[int(6)] = _S504[int(6)];
                _S464[int(7)] = _S504[int(7)];
                _S464[int(8)] = _S504[int(8)];
                _S464[int(9)] = _S496;
                _S464[int(10)] = _S491;
                _S464[int(11)] = _S484;
                _S464[int(12)] = _S477;
                _S464[int(13)] = _S474;
                _S464[int(14)] = _S469;
                _S464[int(15)] = _S465;
                _S389 = _S503;
                _S390 = _S508;
                _S391 = _S507;
            }
            else
            {
                _S383 = 0.0f;
                _S384 = 0.0f;
                _S385 = 0.0f;
                _S386 = 0.0f;
                _S387 = 0.0f;
                _S388 = 0.0f;
                _S464[int(0)] = _S351;
                _S464[int(1)] = _S351;
                _S464[int(2)] = _S351;
                _S464[int(3)] = _S351;
                _S464[int(4)] = _S351;
                _S464[int(5)] = _S351;
                _S464[int(6)] = _S351;
                _S464[int(7)] = _S351;
                _S464[int(8)] = _S351;
                _S464[int(9)] = _S351;
                _S464[int(10)] = _S351;
                _S464[int(11)] = _S351;
                _S464[int(12)] = _S351;
                _S464[int(13)] = _S351;
                _S464[int(14)] = _S351;
                _S464[int(15)] = _S351;
                _S389 = 0.0f;
                _S390 = 0.0f;
                _S391 = 0.0f;
            }
            float3  _S509 = _S367 * _S463.differential_0;
            float3  _S510 = _S368 * _S463.differential_0;
            float _S511 = 0.54627424478530884f * (_S510.x + _S510.y + _S510.z) + _S383;
            float3  _S512 = _S369 * _S463.differential_0;
            float3  _S513 = _S370 * _S463.differential_0;
            float s_diff_xz_T_0 = -1.09254848957061768f * (_S513.x + _S513.y + _S513.z);
            float3  _S514 = _S371 * _S463.differential_0;
            float3  _S515 = _S372 * _S463.differential_0;
            float _S516 = 0.31539157032966614f * (_S515.x + _S515.y + _S515.z);
            float _S517 = - _S516;
            float3  _S518 = _S373 * _S463.differential_0;
            float3  _S519 = _S374 * _S463.differential_0;
            float s_diff_yz_T_0 = -1.09254848957061768f * (_S519.x + _S519.y + _S519.z);
            float3  _S520 = _S375 * _S463.differential_0;
            float3  _S521 = _S376 * _S463.differential_0;
            float _S522 = _S396 * s_diff_xz_T_0;
            float _S523 = _S393 * s_diff_xz_T_0;
            float _S524 = _S397 * s_diff_yz_T_0;
            float _S525 = _S393 * s_diff_yz_T_0;
            float _S526 = 1.09254848957061768f * (_S521.x + _S521.y + _S521.z) + _S385;
            float _S527 = _S396 * _S526;
            float _S528 = _S397 * _S526;
            float _S529 = _S393 * (2.0f * (_S516 + _S384) + _S386);
            float _S530 = _S397 * (- _S511 + _S517 + _S387);
            float _S531 = _S396 * (_S511 + _S517 + _S388);
            FixedArray<float3 , 16>  _S532;
            _S532[int(0)] = _S351;
            _S532[int(1)] = _S351;
            _S532[int(2)] = _S351;
            _S532[int(3)] = _S351;
            _S532[int(4)] = _S351;
            _S532[int(5)] = _S351;
            _S532[int(6)] = _S351;
            _S532[int(7)] = _S351;
            _S532[int(8)] = _S351;
            _S532[int(9)] = _S351;
            _S532[int(10)] = _S351;
            _S532[int(11)] = _S351;
            _S532[int(12)] = _S351;
            _S532[int(13)] = _S351;
            _S532[int(14)] = _S351;
            _S532[int(15)] = _S351;
            _S532[int(8)] = _S509;
            _S532[int(7)] = _S512;
            _S532[int(6)] = _S514;
            _S532[int(5)] = _S518;
            _S532[int(4)] = _S520;
            float3  _S533 = _S464[int(0)] + _S532[int(0)];
            float3  _S534 = _S464[int(1)] + _S532[int(1)];
            float3  _S535 = _S464[int(2)] + _S532[int(2)];
            float3  _S536 = _S464[int(3)] + _S532[int(3)];
            float3  _S537 = _S464[int(4)] + _S520;
            float3  _S538 = _S464[int(5)] + _S518;
            float3  _S539 = _S464[int(6)] + _S514;
            float3  _S540 = _S464[int(7)] + _S512;
            float3  _S541 = _S464[int(8)] + _S509;
            float3  _S542 = _S464[int(9)] + _S532[int(9)];
            float3  _S543 = _S464[int(10)] + _S532[int(10)];
            float3  _S544 = _S464[int(11)] + _S532[int(11)];
            float3  _S545 = _S464[int(12)] + _S532[int(12)];
            float3  _S546 = _S464[int(13)] + _S532[int(13)];
            float3  _S547 = _S464[int(14)] + _S532[int(14)];
            float3  _S548 = _S464[int(15)] + _S532[int(15)];
            float _S549 = _S523 + _S528 + _S531 + _S531 + _S389;
            float _S550 = _S525 + _S527 + _S530 + _S530 + _S390;
            _S383 = _S522 + _S524 + _S529 + _S529 + _S391;
            _S384 = _S550;
            _S385 = _S549;
            _S464[int(0)] = _S533;
            _S464[int(1)] = _S534;
            _S464[int(2)] = _S535;
            _S464[int(3)] = _S536;
            _S464[int(4)] = _S537;
            _S464[int(5)] = _S538;
            _S464[int(6)] = _S539;
            _S464[int(7)] = _S540;
            _S464[int(8)] = _S541;
            _S464[int(9)] = _S542;
            _S464[int(10)] = _S543;
            _S464[int(11)] = _S544;
            _S464[int(12)] = _S545;
            _S464[int(13)] = _S546;
            _S464[int(14)] = _S547;
            _S464[int(15)] = _S548;
        }
        else
        {
            _S383 = 0.0f;
            _S384 = 0.0f;
            _S385 = 0.0f;
            _S464[int(0)] = _S351;
            _S464[int(1)] = _S351;
            _S464[int(2)] = _S351;
            _S464[int(3)] = _S351;
            _S464[int(4)] = _S351;
            _S464[int(5)] = _S351;
            _S464[int(6)] = _S351;
            _S464[int(7)] = _S351;
            _S464[int(8)] = _S351;
            _S464[int(9)] = _S351;
            _S464[int(10)] = _S351;
            _S464[int(11)] = _S351;
            _S464[int(12)] = _S351;
            _S464[int(13)] = _S351;
            _S464[int(14)] = _S351;
            _S464[int(15)] = _S351;
        }
        float3  _S551 = - _S463.differential_0;
        float3  _S552 = _S377 * _S551;
        float3  _S553 = _S378 * _S551;
        float3  _S554 = _S379 * _S463.differential_0;
        float3  _S555 = _S380 * _S463.differential_0;
        float3  _S556 = _S381 * _S551;
        float3  _S557 = _S382 * _S551;
        float _S558 = 0.48860251903533936f * (_S555.x + _S555.y + _S555.z) + _S383;
        float _S559 = 0.48860251903533936f * (_S557.x + _S557.y + _S557.z) + _S384;
        float _S560 = 0.48860251903533936f * (_S553.x + _S553.y + _S553.z) + _S385;
        FixedArray<float3 , 16>  _S561;
        _S561[int(0)] = _S351;
        _S561[int(1)] = _S351;
        _S561[int(2)] = _S351;
        _S561[int(3)] = _S351;
        _S561[int(4)] = _S351;
        _S561[int(5)] = _S351;
        _S561[int(6)] = _S351;
        _S561[int(7)] = _S351;
        _S561[int(8)] = _S351;
        _S561[int(9)] = _S351;
        _S561[int(10)] = _S351;
        _S561[int(11)] = _S351;
        _S561[int(12)] = _S351;
        _S561[int(13)] = _S351;
        _S561[int(14)] = _S351;
        _S561[int(15)] = _S351;
        _S561[int(3)] = _S552;
        _S561[int(2)] = _S554;
        _S561[int(1)] = _S556;
        float3  _S562 = _S464[int(0)] + _S561[int(0)];
        float3  _S563 = _S464[int(1)] + _S556;
        float3  _S564 = _S464[int(2)] + _S554;
        float3  _S565 = _S464[int(3)] + _S552;
        float3  _S566 = _S464[int(4)] + _S561[int(4)];
        float3  _S567 = _S464[int(5)] + _S561[int(5)];
        float3  _S568 = _S464[int(6)] + _S561[int(6)];
        float3  _S569 = _S464[int(7)] + _S561[int(7)];
        float3  _S570 = _S464[int(8)] + _S561[int(8)];
        float3  _S571 = _S464[int(9)] + _S561[int(9)];
        float3  _S572 = _S464[int(10)] + _S561[int(10)];
        float3  _S573 = _S464[int(11)] + _S561[int(11)];
        float3  _S574 = _S464[int(12)] + _S561[int(12)];
        float3  _S575 = _S464[int(13)] + _S561[int(13)];
        float3  _S576 = _S464[int(14)] + _S561[int(14)];
        float3  _S577 = _S464[int(15)] + _S561[int(15)];
        features_11 = make_float3 (_S560, _S559, _S558);
        _S464[int(0)] = _S562;
        _S464[int(1)] = _S563;
        _S464[int(2)] = _S564;
        _S464[int(3)] = _S565;
        _S464[int(4)] = _S566;
        _S464[int(5)] = _S567;
        _S464[int(6)] = _S568;
        _S464[int(7)] = _S569;
        _S464[int(8)] = _S570;
        _S464[int(9)] = _S571;
        _S464[int(10)] = _S572;
        _S464[int(11)] = _S573;
        _S464[int(12)] = _S574;
        _S464[int(13)] = _S575;
        _S464[int(14)] = _S576;
        _S464[int(15)] = _S577;
    }
    else
    {
        features_11 = _S351;
        _S464[int(0)] = _S351;
        _S464[int(1)] = _S351;
        _S464[int(2)] = _S351;
        _S464[int(3)] = _S351;
        _S464[int(4)] = _S351;
        _S464[int(5)] = _S351;
        _S464[int(6)] = _S351;
        _S464[int(7)] = _S351;
        _S464[int(8)] = _S351;
        _S464[int(9)] = _S351;
        _S464[int(10)] = _S351;
        _S464[int(11)] = _S351;
        _S464[int(12)] = _S351;
        _S464[int(13)] = _S351;
        _S464[int(14)] = _S351;
        _S464[int(15)] = _S351;
    }
    float3  _S578 = make_float3 (0.282094806432724f) * _S463.differential_0;
    dpdirection_1->primal_0 = (*dpdirection_1).primal_0;
    dpdirection_1->differential_0 = features_11;
    FixedArray<float3 , 16>  _S579;
    _S579[int(0)] = _S351;
    _S579[int(1)] = _S351;
    _S579[int(2)] = _S351;
    _S579[int(3)] = _S351;
    _S579[int(4)] = _S351;
    _S579[int(5)] = _S351;
    _S579[int(6)] = _S351;
    _S579[int(7)] = _S351;
    _S579[int(8)] = _S351;
    _S579[int(9)] = _S351;
    _S579[int(10)] = _S351;
    _S579[int(11)] = _S351;
    _S579[int(12)] = _S351;
    _S579[int(13)] = _S351;
    _S579[int(14)] = _S351;
    _S579[int(15)] = _S351;
    _S579[int(0)] = _S578;
    FixedArray<float3 , 16>  _S580 = {
        _S464[int(0)] + _S578, _S464[int(1)] + _S579[int(1)], _S464[int(2)] + _S579[int(2)], _S464[int(3)] + _S579[int(3)], _S464[int(4)] + _S579[int(4)], _S464[int(5)] + _S579[int(5)], _S464[int(6)] + _S579[int(6)], _S464[int(7)] + _S579[int(7)], _S464[int(8)] + _S579[int(8)], _S464[int(9)] + _S579[int(9)], _S464[int(10)] + _S579[int(10)], _S464[int(11)] + _S579[int(11)], _S464[int(12)] + _S579[int(12)], _S464[int(13)] + _S579[int(13)], _S464[int(14)] + _S579[int(14)], _S464[int(15)] + _S579[int(15)]
    };
    dpcoefficients_1->primal_0 = dpcoefficients_1->primal_0;
    dpcoefficients_1->differential_0 = _S580;
    return;
}

__device__ void particleFeaturesIntegrateBwdToBuffer(float3  incidentDirection_2, float alpha_5, float * alphaGrad_2, uint particleIdx_7, shRadiativeParticle_CommonParameters_0 commonParameters_7, float3  features_14, float3  * integratedFeatures_3, float3  * integratedFeaturesGrad_1)
{
    float3  _S581;
    FixedArray<float3 , 16>  _S582;
    uint _S583;
    shRadiativeParticle_Parameters_0 _S584;
    bool _S585 = alpha_5 > 0.0f;
    if(_S585)
    {
        int i_6;
        float3  _S586 = (*integratedFeatures_3 - features_14 * make_float3 (alpha_5)) * make_float3 (1.0f / (1.0f - alpha_5));
        *integratedFeatures_3 = _S586;
        float3  _S587 = *integratedFeaturesGrad_1;
        uint _S588 = uint(commonParameters_7.sphDegree_0);
        s_bwd_prop_shRadiativeParticle_integrateRadianceFromBuffer_Intermediates_0 _S589;
        for(;;)
        {
            float3  _S590 = make_float3 (0.0f);
            _S581 = _S590;
            FixedArray<float3 , 16>  _S591 = {
                _S590, _S590, _S590, _S590, _S590, _S590, _S590, _S590, _S590, _S590, _S590, _S590, _S590, _S590, _S590, _S590
            };
            _S582[int(0)] = _S590;
            _S582[int(1)] = _S590;
            _S582[int(2)] = _S590;
            _S582[int(3)] = _S590;
            _S582[int(4)] = _S590;
            _S582[int(5)] = _S590;
            _S582[int(6)] = _S590;
            _S582[int(7)] = _S590;
            _S582[int(8)] = _S590;
            _S582[int(9)] = _S590;
            _S582[int(10)] = _S590;
            _S582[int(11)] = _S590;
            _S582[int(12)] = _S590;
            _S582[int(13)] = _S590;
            _S582[int(14)] = _S590;
            _S582[int(15)] = _S590;
            shRadiativeParticle_Parameters_0 _S592 = { _S591 };
            (&_S589)->_S335 = _S590;
            (&_S589)->_S336 = _S592;
            (&(&_S589)->_S336)->sphCoefficients_0 = _S591;
            (&_S589)->_S335 = _S586;
            for(;;)
            {
                shRadiativeParticle_Parameters_0 parameters_4;
                uint _S593 = particleIdx_7 * 16U;
                _S583 = _S593;
                i_6 = int(0);
                #pragma unroll
                for(;;)
                {
                    if(i_6 < int(16))
                    {
                    }
                    else
                    {
                        break;
                    }
                    (&parameters_4)->sphCoefficients_0[i_6] = *(commonParameters_7.parametersBuffer_1._dataPtr_1 + (_S593 + uint(i_6)));
                    i_6 = i_6 + int(1);
                }
                _S584 = parameters_4;
                break;
            }
            (&_S589)->_S336 = _S584;
            break;
        }
        float _S594;
        float3  dpintegratedRadiance_1;
        s_bwd_prop_shRadiativeParticle_integrateRadianceFromBuffer_Intermediates_0 _S595 = _S589;
        for(;;)
        {
            float3  _S596;
            FixedArray<float3 , 16>  _S597;
            if(_S585)
            {
                int _S598 = int(_S588);
                FixedArray<float3 , 16>  _S599 = _S595._S336.sphCoefficients_0;
                float3  _S600 = s_primal_ctx_sphericalHarmonics_decode_0(_S598, &_S599, incidentDirection_2);
                if(_S585)
                {
                    _S596 = make_float3 (alpha_5);
                }
                else
                {
                    _S596 = _S581;
                }
                float3  _S601 = _S596;
                _S596 = _S600;
                dpintegratedRadiance_1 = _S601;
                _S597 = _S595._S336.sphCoefficients_0;
                i_6 = _S598;
            }
            else
            {
                _S596 = _S581;
                dpintegratedRadiance_1 = _S581;
                _S597 = _S582;
                i_6 = int(0);
            }
            shRadiativeParticle_Parameters_0 _S602 = shRadiativeParticle_Parameters_x24_syn_dzero_0();
            shRadiativeParticle_Parameters_0 _S603;
            if(_S585)
            {
                if(_S585)
                {
                    DiffPair_vectorx3Cfloatx2C3x3E_0 _S604;
                    (&_S604)->primal_0 = _S595._S335;
                    (&_S604)->differential_0 = _S581;
                    DiffPair_vectorx3Cfloatx2C3x3E_0 _S605;
                    (&_S605)->primal_0 = _S596;
                    (&_S605)->differential_0 = _S581;
                    DiffPair_vectorx3Cfloatx2C3x3E_0 _S606;
                    (&_S606)->primal_0 = dpintegratedRadiance_1;
                    (&_S606)->differential_0 = _S581;
                    s_bwd_prop_lerp_0(&_S604, &_S605, &_S606, _S587);
                    float _S607 = _S606.differential_0.x + _S606.differential_0.y + _S606.differential_0.z;
                    _S596 = _S605.differential_0;
                    dpintegratedRadiance_1 = _S604.differential_0;
                    _S594 = _S607;
                }
                else
                {
                    _S596 = _S581;
                    dpintegratedRadiance_1 = _S587;
                    _S594 = 0.0f;
                }
                FixedArray<float3 , 16>  _S608 = { _S581, _S581, _S581, _S581, _S581, _S581, _S581, _S581, _S581, _S581, _S581, _S581, _S581, _S581, _S581, _S581 };
                DiffPair_arrayx3Cvectorx3Cfloatx2C3x3Ex2C16x3E_0 _S609;
                (&_S609)->primal_0 = _S597;
                (&_S609)->differential_0 = _S608;
                DiffPair_vectorx3Cfloatx2C3x3E_0 _S610;
                (&_S610)->primal_0 = incidentDirection_2;
                (&_S610)->differential_0 = _S581;
                s_bwd_prop_sphericalHarmonics_decode_0(i_6, &_S609, &_S610, _S596);
                shRadiativeParticle_Parameters_0 _S611 = _S602;
                (&_S611)->sphCoefficients_0 = (&_S609)->differential_0;
                shRadiativeParticle_Parameters_0 _S612 = _S602;
                shRadiativeParticle_Parameters_0 _S613 = _S611;
                shRadiativeParticle_Parameters_0 _S614 = shRadiativeParticle_Parameters_x24_syn_dadd_0(&_S612, &_S613);
                _S603 = _S614;
            }
            else
            {
                _S603 = _S602;
                dpintegratedRadiance_1 = _S587;
                _S594 = 0.0f;
            }
            for(;;)
            {
                i_6 = int(0);
                #pragma unroll
                for(;;)
                {
                    if(i_6 < int(16))
                    {
                    }
                    else
                    {
                        break;
                    }
                    shRadiativeParticle_Parameters_0 _S615 = _S603;
                    int _S616 = i_6;
                    int j_2;
                    if(commonParameters_7.parametersBuffer_1.exclusiveGradient_1)
                    {
                        j_2 = int(0);
                        #pragma unroll
                        for(;;)
                        {
                            if(j_2 < int(3))
                            {
                            }
                            else
                            {
                                break;
                            }
                            *_slang_vector_get_element_ptr(commonParameters_7.parametersBuffer_1._gradPtr_1 + (_S583 + uint(i_6)), j_2) = *_slang_vector_get_element_ptr(commonParameters_7.parametersBuffer_1._gradPtr_1 + (_S583 + uint(i_6)), j_2) + _slang_vector_get_element(_S615.sphCoefficients_0[_S616], j_2);
                            j_2 = j_2 + int(1);
                        }
                    }
                    else
                    {
                        j_2 = int(0);
                        #pragma unroll
                        for(;;)
                        {
                            if(j_2 < int(3))
                            {
                            }
                            else
                            {
                                break;
                            }
                            float _S617 = atomicAdd(_slang_vector_get_element_ptr(commonParameters_7.parametersBuffer_1._gradPtr_1 + (_S583 + uint(i_6)), j_2), _slang_vector_get_element(_S615.sphCoefficients_0[_S616], j_2));
                            j_2 = j_2 + int(1);
                        }
                    }
                    i_6 = i_6 + int(1);
                }
                break;
            }
            break;
        }
        *integratedFeaturesGrad_1 = dpintegratedRadiance_1;
        *alphaGrad_2 = _S594;
    }
    return;
}

struct s_bwd_prop_shRadiativeParticle_radianceFromBuffer_Intermediates_0
{
    shRadiativeParticle_Parameters_0 _S618;
};

__device__ void particleFeaturesBwdToBuffer(uint particleIdx_8, shRadiativeParticle_CommonParameters_0 commonParameters_8, float3  featuresGrad_1, float3  incidentDirection_3)
{
    int i_7;
    uint _S619;
    shRadiativeParticle_Parameters_0 _S620;
    uint _S621 = uint(commonParameters_8.sphDegree_0);
    float3  _S622 = make_float3 (0.0f);
    FixedArray<float3 , 16>  _S623 = {
        _S622, _S622, _S622, _S622, _S622, _S622, _S622, _S622, _S622, _S622, _S622, _S622, _S622, _S622, _S622, _S622
    };
    shRadiativeParticle_Parameters_0 _S624 = { _S623 };
    s_bwd_prop_shRadiativeParticle_radianceFromBuffer_Intermediates_0 _S625;
    (&_S625)->_S618 = _S624;
    (&(&_S625)->_S618)->sphCoefficients_0 = _S623;
    for(;;)
    {
        shRadiativeParticle_Parameters_0 parameters_5;
        uint _S626 = particleIdx_8 * 16U;
        _S619 = _S626;
        i_7 = int(0);
        #pragma unroll
        for(;;)
        {
            if(i_7 < int(16))
            {
            }
            else
            {
                break;
            }
            (&parameters_5)->sphCoefficients_0[i_7] = *(commonParameters_8.parametersBuffer_1._dataPtr_1 + (_S626 + uint(i_7)));
            i_7 = i_7 + int(1);
        }
        _S620 = parameters_5;
        break;
    }
    (&_S625)->_S618 = _S620;
    int _S627 = int(_S621);
    FixedArray<float3 , 16>  _S628 = { _S622, _S622, _S622, _S622, _S622, _S622, _S622, _S622, _S622, _S622, _S622, _S622, _S622, _S622, _S622, _S622 };
    DiffPair_arrayx3Cvectorx3Cfloatx2C3x3Ex2C16x3E_0 _S629;
    (&_S629)->primal_0 = (&(&_S625)->_S618)->sphCoefficients_0;
    (&_S629)->differential_0 = _S628;
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S630;
    (&_S630)->primal_0 = incidentDirection_3;
    (&_S630)->differential_0 = _S622;
    s_bwd_prop_sphericalHarmonics_decode_0(_S627, &_S629, &_S630, featuresGrad_1);
    shRadiativeParticle_Parameters_0 _S631 = shRadiativeParticle_Parameters_x24_syn_dzero_0();
    (&_S631)->sphCoefficients_0 = (&_S629)->differential_0;
    shRadiativeParticle_Parameters_0 _S632 = _S631;
    for(;;)
    {
        i_7 = int(0);
        #pragma unroll
        for(;;)
        {
            if(i_7 < int(16))
            {
            }
            else
            {
                break;
            }
            int _S633 = i_7;
            int j_3;
            if(commonParameters_8.parametersBuffer_1.exclusiveGradient_1)
            {
                j_3 = int(0);
                #pragma unroll
                for(;;)
                {
                    if(j_3 < int(3))
                    {
                    }
                    else
                    {
                        break;
                    }
                    *_slang_vector_get_element_ptr(commonParameters_8.parametersBuffer_1._gradPtr_1 + (_S619 + uint(i_7)), j_3) = *_slang_vector_get_element_ptr(commonParameters_8.parametersBuffer_1._gradPtr_1 + (_S619 + uint(i_7)), j_3) + _slang_vector_get_element(_S632.sphCoefficients_0[_S633], j_3);
                    j_3 = j_3 + int(1);
                }
            }
            else
            {
                j_3 = int(0);
                #pragma unroll
                for(;;)
                {
                    if(j_3 < int(3))
                    {
                    }
                    else
                    {
                        break;
                    }
                    float _S634 = atomicAdd(_slang_vector_get_element_ptr(commonParameters_8.parametersBuffer_1._gradPtr_1 + (_S619 + uint(i_7)), j_3), _slang_vector_get_element(_S632.sphCoefficients_0[_S633], j_3));
                    j_3 = j_3 + int(1);
                }
            }
            i_7 = i_7 + int(1);
        }
        break;
    }
    return;
}

