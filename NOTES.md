

# Accessors
They describe how a resource (image or buffer) is accessed within a task. 
They contain the intended usage of the resource and its _access mode_ (read-only/read-write/write-only)

For images, the possible usages are:
- sampled image
- storage image
- transfer source/destination
- color attachment
- depth attachment
- input attachment

For buffers, the possible usages are:
- vertex buffer
- uniform buffer
- storage buffer
- transform feedback buffer
- index buffer
- transfer source
- transfer destination
- uniform texel buffer 
- storage texel buffer
- indirect buffer (commands)

## Accessor syntax(es)

Questions:
- what to optimize?
	- verbosity
	- complexity of implementation
- type-erased accessors?
- how are accessors used afterwards?


### Option A: single template
The same `accessor` template is used for both images and buffers. 
A large set of constructor overloads and deduction guides are provided for ease of use and type safety.

```c++
void task(handler& h) {
accessor color_target { image, color_attachment, ..., h };
accessor rw_storage { image, storage_image, read_write, h };
accessor ubo { buffer, uniform_buffer, h  };
accessor ssbo { buffer, storage_buffer, read_write, h };
accessor vbo { buffer, vertex_buffer, h };
}
```

`accessor` primary template:
```
template <
	typename DataT, 
	image_type Type,
	usage Usage,
	access_mode AccessMode>
class accessor; 
```

A problem with this is that some of the template params only make sense for one particular type of resource.
For instance, `DataT` only makes sense for buffers, while `Type` only makes sense for images. 
In addition, `Usage` has values that are only valid for images, and others for buffers.
For some usages, not all values of `AccessMode` are valid (for instance, when `Usage == sampled_image` then `AccessMode` must be `read`).


### Option B: many types
One accessor template per usage:

```c++
void task(handler& h) {
color_attachment_accessor color_target { image, ..., h };
storage_image_accessor storage {image, read_write, h};
uniform_buffer_accessor ubo { buffer, h  };
storage_buffer_accessor ssbo { buffer, read_write, h };
vertex_buffer_accessor vbo { buffer, h };
}
```

Solves the issue of unused template parameters.

### Option C: buffer and image accessors

```
image_accessor color_target { image, color_attachment, ..., h };
image_accessor rw_storage { image, storage_image, read_write, h };
buffer_accessor ubo { buffer, uniform_buffer, h  };
buffer_accessor ssbo { buffer, storage_buffer, read_write, h };
buffer_accessor vbo { buffer, vertex_buffer, h };
```

More verbose than A, but solves the issue of unused template parameters.

### Option D: methods on resources:
```
auto color_target = image->access_as_color_attachment(image, ..., h );
auto rw_storage = image->access_as_storage_image(image, read_write, h);
auto ubo = buffer->access_as_uniform_buffer(buffer, h);
auto ssbo = buffer->access_as_storage_buffer(buffer, read_write, h);
auto vbo = buffer->access_as_vertex_buffer_buffer(buffer, h);
```

## Conclusion
Option A vs option B: the only difference is that option A has a single "accessor" type in the declaration, 
making it like a "keyword". 
However, it can be more difficult to read, and more difficult to write down the full type. Using it in a function signature is harder (but it's possible to add alias templates).

Q: do we want functions that are fully generic over the type of accessors?
I.e. a (template) function that takes an accessor and then, regarless of resource type and usage, does something with it.
The "something" cannot be:
	- storing it in a list (unless it uses type erasure somehow)

Option A is one primary template with many specializations.
Option B is many different templates.

Q: what do we want to do with accessors?
A: only use them in draw/compute/transfer commands in during submit-time;

Q: can we access the same resource in two different ways within one task? 
A: possibly; what about accessing an image as a `sampled_image` and as a `storage_image` at the same time => YES

accessors without external memory dependencies: "scratch buffers", only used within the task.


## Using accessors

For binding vertex buffers:
```c++

template <typename VertexType>
void bind_vertex_buffer(
	size_t slot, 
	const accessor<VertexType, image_type::image_1d, usage::vertex_buffer, access_mode::read_only>& access) 
{
	...
}
```

## Comparison with SYCL
In SYCL, accessors serve a double role:
- signalling dependencies
- accessing resource data in the device function (provides operator[])

With Vulkan, the resources are not accessed directly: they are bound to the pipeline instead, and referred to in command buffers.
But there's no need for data access operators in c++ (data in accessed in shaders).

## Are accessors necessary?

Definitions:
- DAG build-time: when calling schedule, and executing the closures with handler => building the DAG
- Submit-time: during calling `queue::submit_pending` => creating concrete resources; creating command buffers and filling them.

What about signalling usages when using the resource in a draw command?
```c++
void task(handler& h) 
{
	// problem: image is not a concrete resource during DAG-build, so 
	// we need to create a callback function 
	// (which is what we would do anyway)
	h.clear_image(image, color);

	// problem: image is accessed another time, need to check that the access is compatible
	h.clear_image(image, color);

	// assume this:

	h.draw(image, ...);		// repeated 1000s of times

	// all draws are emitted during DAG-build; can't parallelize
	// => by putting all commands into the DAG-build, it removes submit-time callbacks (for better or worse).
	// => it should be possible to do custom work during submit-time callbacks (e.g. parallel command buffer generation)
	// => split DAG-time (accessors => dependency declaration) and submit-time (command buffer generation)
}
```


# Pipeline creation

In contrast with the "queue" abstraction, I don't think that there's anything to gain by abstracting stuff here.
There are a *lot* of things to abstract here:
- render states [RS]
- render passes [RP]
	- attachments
	- attachment dependencies
- descriptor set layouts [DESC]
- shader modules [SH]
- vertex input layout [VTX]


Among those, I suppose render passes could be "deduced from usage"; however, pipeline creation would need to be deferred to 
submit-time.
Q: deduce render passes from usage?
A: tricky to implement, many questions to solve (should subpasses correspond to different tasks or not?), pipeline creation deferred to submit-time, need caching (and hashing)

Some of those could be deduced from the shader interface; for instance:
- descriptor sets layouts could be inferred from the shader interface
- vertex input layout: not really, but it can be inferred from the passed data and what the shader expects
- render passes: just use a single pass
	- render pass attachments: infer from shader and queue usages

=> let's be clear: the API will be for **experimentation**; not necessarily fit for high-performance rendering (as in games)

## Descriptor set layouts
In order to change the interface of a shader:
1. edit the uniforms as defined in the shader (GLSL)
2. edit the descriptor set layout (C++)
Ideally, we should factor out common parts of interfaces into a common descriptor set layout.

Note that the two steps convey almost the same information.

Proposal:
- Generate descriptor set layouts automatically by parsing GLSL source code
- Detect identical descriptor set layouts across shaders and re-use them
	- in addition, use "interface include files" that define a descriptor set layout:
```c
// defines 
#pragma graal_shader_interface (set=0)

layout(set=0,binding=0) uniform FrameUniforms {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;
```
	- these files can be parsed independently of other shaders, and can be used to create a descriptor set layout
	- eventually, they could also be parsed at build-time to generate C++ headers that facilitate the creation of matching descriptor sets
	- shaders include those files to define a part of their interface
