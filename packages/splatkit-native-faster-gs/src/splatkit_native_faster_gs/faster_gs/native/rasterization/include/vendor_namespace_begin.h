#pragma once

// Redirect the vendored FasterGS implementation into a private binary
// namespace so it can coexist with external FasterGS extensions in one
// process.

#ifndef faster_gs
#define faster_gs splatkit_faster_gs_core_vendor
#endif
