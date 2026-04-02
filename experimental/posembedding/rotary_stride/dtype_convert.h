#include "stdio.h"

#define TYPE_SWITCH(dtype, DType, ...)                           \
    switch (dtype)                                                       \
    {                                                                   \
        case 1:                                                         \
            {                                                           \
                typedef half DType;                                     \
                {__VA_ARGS__}                                           \
            }                                                           \
            break;                                                      \
        case 27:                                                        \
            {                                                           \
                typedef bfloat16_t DType;                               \
                {__VA_ARGS__}                                           \
            }                                                           \
            break;                                                      \
        default:                                                        \
            {                                                           \
                typedef bfloat16_t DType;                               \
                {__VA_ARGS__}                                           \
            }                                                           \
            break;                                                      \
    }