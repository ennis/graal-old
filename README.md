# GRAAL 

This repository contains the source code of graal, a modern C++17 wrapper over OpenGL that makes it easy to optimize resource allocation in complex rendering pipelines.

### Disclaimer
This is a work in progress.

### How it looks
The API takes inspiration from [SYCL 2020][1], with additional types and features for graphics.
TODO

### How it works
By declaring how a resource is accessed within a task, graal is able to discover optimization opportunities, such as aliasing the memory of two images if their contents are not _alive_ at the same time.
TODO

### How to build
TODO

### What it means
Graphics Resource Aliasing Abstraction Layer, but don't look too much into it.

### Credits
- This software uses a modified version of https://github.com/grisumbras/enum-flags. See ext/flags/LICENSE.

[1]: https://www.khronos.org/sycl/
