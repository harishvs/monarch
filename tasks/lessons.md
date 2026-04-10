# Lessons Learned

## 2026-04-09: Tool permission explanations
**Pattern**: User denied tool calls because they didn't understand what the command does.
**Rule**: Every time a tool needs user permission, include a plain-language explanation of what it does and why. Don't assume the user reads bash/cargo syntax.

## 2026-04-09: Don't take shortcuts on tests
**Pattern**: Removed GPU tests instead of finding the right allocation API (CudaAllocator was already available). Wrote a skip function with a bug (always returned true). Marked tasks complete without running tests.
**Rule**: 
- Search the codebase for existing utilities before concluding something is missing.
- Always run tests and verify output before marking anything done.
- If a test needs to skip, the skip condition must be verifiably correct.

## 2026-04-09: Investigate root causes before reporting "pre-existing"
**Pattern**: Reported `#[hyperactor::export]` error as "pre-existing" and stopped. It was introduced by the WIP commit (not main) and fixable with `cfg_attr`.
**Rule**: Before labeling something "pre-existing", check if it exists on main. If it was introduced by the working branch, fix it.

## 2026-04-09: Static inline functions and bindgen
**Pattern**: libfabric functions are `static inline` in C headers. Bindgen can't capture them. The WIP code used them as if they were available.
**Rule**: When wrapping a C library with bindgen, check which functions are `static inline`. Those need plain-C wrappers in a `.c` file compiled via `cc` crate.

## 2026-04-09: HMEM domain changes all MR registration
**Pattern**: When EFA negotiates FI_HMEM, the `mr_mode` includes `FI_MR_HMEM`, meaning ALL memory registrations (even CPU) must go through `fi_mr_regattr` with the appropriate `iface` (FI_HMEM_SYSTEM for CPU, FI_HMEM_CUDA for GPU).
**Rule**: Don't mix `fi_mr_reg` and `fi_mr_regattr` on the same HMEM domain. Use regattr for everything.
