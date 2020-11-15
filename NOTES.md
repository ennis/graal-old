

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

### Proposal:
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

In the application, when loading a pipeline, the user can specify zero or more automatically-generated descriptor set layouts to conform to.
This can even be encoded in the pipeline type.
	
### Rebuttal
The GLSL uniform declaration does not contain enough information to specify a descriptor set layout. For instance:
- there's no way to specify that a uniform buffer binding is dynamic **
- there's no way to specify immutable samplers
- no way to specify which shader stages are going to use the uniforms
We need custom "annotations" for that, which we need to parse manually.

Additionally, in order to get proper reflection from an interface file, it needs to be compiled to SPIR-V first, which means that it needs to be a valid shader.
A solution is to append `void main() {}` to the source in order to make it a valid shader, but that's a bit hacky.

### Proposal 2: interface description files
Interface description files contain JSON data that contains information about a shader interface, from the point-of-view of both the shader
and the application. Those files will be written by hand, and then used to create both a GLSL include and a C++ include.

```json
{
	"structs": [
		{
			"name": "FrameUniforms",
			"members": [ 
				{"name": "model", "type": "mat4"},
				{"name": "view", "type": "mat4"},
				{"name": "projection", "type": "mat4"}
			],
		}
	],
	"bindings": {
		"0": {
			"descriptor_type": "uniform_buffer_dynamic",
			"type": "FrameUniforms",
			"count": 1,
			"stage_flags": "all_graphics"
		}
	}
}
```


### Proposal 3: set conventions
In order to determine whether a uniform buffer should be dynamic, look at the set number. By convention:
- set #0 is for data that changes per-frame => uniform buffer should be state
- set #1 is for data that changes per-object => uniform buffer should be dynamic
- push constants will be for data that changes per-draw 

Q: should render pass information be put in the shader?
A: No. It does not belong in a shader interface.


## Case study: an extensible compositing application 
Extensible with (fragment) shaders.
Nodes define a number of input images, output images, parameters, and allowed formats.

The node provides:
- a fragment shader SPIR-V binary
- something that identifies the well-known uniform interfaces (descriptor sets) that the shader supports
- for each input, a list of supported input formats
- for each output, given input formats, a list of supported output formats


Ideal situation:
- in the application, "invoke" a shader by filling a struct corresponding to the descriptor set layout; 
	it automatically creates the descriptor set;
	- the struct type (==interface type) is a parameter of the pipeline so it can be statically verified
	- there can be more than one interface type
- the "static interface" is in the template parameters, the "dynamic interface" can be queried 
	- it returns information about descriptor set layouts, and layout of buffers (types, members)

- "user" in control:
	- shader source
	- number of inputs
	- number of outputs
- "application" in control:
	- 

- hot-reloading: just re-build the pipeline
	- the static interface cannot change

- need to create pipeline variants for 
	- different render target formats
	- different vertex buffer inputs

Idea: a "pipeline" class that is like a query for a concrete pipeline
- template parameters: descriptor set interface types, vertex bindings
- for maximum flexibility: temporarily erase some template params
	- e.g. may want a function that operates only on "pipelines with specific vertex bindings"
		or "pipelines with that specific set interface"
	- 
- pipeline_base base class without any template parameters
	- query

- e.g. full-screen quad shader
```cpp
pipeline_base quad_pipeline();

void test() 
{
	auto pp = quad_pipeline();
	pp.set_fragment_shader(spv);
	pp.set_rasterization_state(...);
	auto tpp = pp.enforce<global_params>();	// copy, not reference
	// tpp : pipeline<unknown, global_params>

	auto variant_1 = pp; // copy semantics, not a reference
	pp.set_rasterization_state(...);
	auto variant_2 = pp;

	// 
	variant_1.resolve();	// creates a concrete VkPipeline
	variant_2.resolve();	// creates a concrete VkPipeline

	// 
	auto vpp = pp_base.with_vertex_input<ty>();	// derived pipeline with specific vertex bindings
	auto vpp2 = pp_base.with_vertex_input<ty2>();	// same but with different vertex bindings
	// 

	// "pipeline_base" is a query object, like QFont.
	// pipeline_base caches its last resolved vkpipeline, which is invalidated when calling set_xxx
	// 
}
```

### Shader interface types
A type, associated to a descriptor set layout, used to build a descriptor set.
Contains references to buffers.
The type has a static "set" index associated to it.
Defines dynamic offsets (indices).
Each instance can be turned into a descriptor set: `descriptor_set<T,dynamic_indices>`
The size_t parameters are for the dynamic indices.
Bind to the pipeline with cmd.bind_descriptor_set(set, index_1, index_2, ...)

```cpp

struct frame_parameters {
	buffer_view<T> ubo;
	image_view<...> 
};

```


# Exception safety

Consider this:
```cpp
int main() {
	queue q{...};
	image img{...};

	try {
		q.submit([](handler& h) {
			accessor img{..., h};
			do_something_that_throws();		// may throw
		});

	} catch (std::exception e) {
		// the accessor will have modified the current batch and the last_write_sequence_number of the image
		// leaving the queue in an invalid state
		// => the handler should only modify the current task
		// => force the handler callback to be noexcept?
	}
}
```

This would be safe:
```cpp
int main() {
	queue q{...};
	image img{...};

	q.submit([](handler& h) {
		try {
			accessor img{..., h};
			do_something_that_throws();		// may throw
		} catch (...) {
			// recover somehow, or terminate
		}
	});
}
```

I think it's reasonable to force the handler callback to be noexcept. 
Being able to throw exception across the handler boundary would be useful only for this:
```cpp
int main() {
	queue q{...};
	image img{...};

	try {
		q.submit([](handler& h) { ... });
		q.submit([](handler& h) { ... });
		q.submit([](handler& h) { ... });
		//...
		q.submit([](handler& h) { ... });
	} catch(std::exception e) {
		// ...
	}
}
```
but even that would leave the queue in an indeterminate state (which tasks were submitted?).

`queue::submit` could fail as well (why?). So same problem here.
If `queue::submit` fails, then there's a guarantee that the handler callback was not called and no resources were touched.
=> queue submit cannot fail once the resources have been modified: no rollback => add_task is noexcept

Question: explicit batch?
Could be useful to recover from an exception during submission (just drop the batch). However it needs some refactoring work. 
There's also the problem that currently it's unclear when a batch starts and ends. This is somewhat implicit.

Issues with exception safety: we can't mutate resources for tracking, because the tracking info may become invalid if batch submission fails because of an exception.
- tmp_index
- last_write_sequence_number

Replace with a function in batch:
```
batch::get_resource_tmp_index(resource&) -> size_t;
queue::last_write_sequence_number(resource&) -> uint64_t;
```

Need to store a `map<resource*,size_t>`. Must be efficient.
TODO Assign a generational index to resources. 
There needs to be a big map resource->size_t in the queue to track sequence numbers across batches.
=> don't do that. instead, once a batch is submitted, "commit" the last_write_sequence_number to all resources (there's nothing cancellable above batches anyway)

Questions:
- can an individual submission fail? YES
	- and leave the queue in an indeterminate state? NO
- can a batch fail? 
	- if yes, then the queue should be in a determinate state if a batch fails mid-building
	- otherwise, don't care

This boils down to: where should we put the "point-of-no-return" for exceptions?
	- task submit?
	- batch submit?

One could argue that the point of no return should be the submission to the backend API (i.e. vulkan). In this case,
that's _batch submit_.

## Explicit batches
```cpp
int main() {
	queue q{...};

	batch b{q};
	image img{...};

	try {
		b.submit([](handler& h) { ... });
		b.submit([](handler& h) { ... });
		b.submit([](handler& h) { ... });
		//...
		b.submit([](handler& h) { ... });

	} catch(std::exception e) {
		// queue is unaffected here
		// return or rethrow.
		return -1;
	}

	q.submit(std::move(b));	// could also submit in the destructor of the batch, but that's sketchy.
}
```
Issues:
- one more concept exposed to the user(batches)
- b.submit() could return awaitable events. However waiting on them will deadlock q.submit() has been called. That's one more way to screw up. 

## Implicit batches
```cpp
int main() {
	queue q{...};

	image img{...};

	try {
		q.submit([](handler& h) { ... });
		q.submit([](handler& h) { ... });
		q.submit([](handler& h) { ... });
		//...
		q.submit([](handler& h) { ... });

	} catch(std::exception e) {
		// state of the queue may be affected here
		// option 1: reset the current batch (but it's unclear which commands will be cancelled)
		//q.reset();
		// option 2: do nothing; what's submitted can't be taken back. **
	}

	q.finish_batch();	// this must be called
}
```
Issues:
- could forget to finish the batch
- finish_batch() is sometimes implicit
	- waiting on a submit-returned event can force a finish_batch
	- destruction of the queue forces a finish_batch and waitDeviceIdle
- submits can't be taken back
- can't create multiple batches in parallel
	- is that useful?
	- do games do that?
	- command buffer creation would be the thing that takes the most time anyway
		- might be able to pipeline it across batches 
		- create command buffers asynchronously, but allow the next batch to start


#### Conclusion:
Let's go with implicit batches for now. 

#### When to generate the command buffer?

Determined during batch analysis:
- memory allocations
According to the vulkan spec (https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#resources-association)

	Non-sparse resources must be bound completely and contiguously to a single VkDeviceMemory object 
	before the resource is passed as a parameter to any of the following operations:
    	* creating image or buffer views
    	* updating descriptor sets
    	* recording commands in a command buffer

So command buffer generation must happen after batch analysis