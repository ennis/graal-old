
if (resource->batch == batch_index_
    && (mode == access_mode::read_only || mode == access_mode::read_write))
{
    // we know that the producer is a task in this batch
    // add the producer to the current list of predecessors

    // if the batch is finished, then 
}
else {
    // we are read-accessing a resource that was written in a previous batch.
    // Ideally, we would like to wait on a semaphore for the resource, but
    // there's no guarantee that one was created and signalled in the previous batch 
    // (we can't predict whether the next frame is going to use it)

    //

    // this is an inter-batch dependency: we sync on previous batch finished, 
    // this is suboptimal, but we are not sure to have a semaphore for the 
    // resource in the previous batch, and we can't do anything about it because 
    // it's already submitted. the only

    // "we can't predict whether the next frame is going to use it" => actually we can,
    // because when submitting we know if a resource is still "visible to the user", i.e.
    // whether there are potential references to the resource across batches.
    // If there are, then allocate a semaphore for the resource so that following batches may sync on it.

    // for write access modes, however, don't bother waiting for the previous frame to complete.
    // instead, just orphan the image
}

// there should be a way to "recycle" the raw vk image if we are writing to 

// let's say we want to write to the resource, but we can't prove that
// the previous batches have finished accessing it
// - we want to "orphan" the currently bound image object, and replace it with another
// - the batch has a resource_ptr, so it holds a ref
// - we *cannot* modify the resource_ptr, because it's shared

// -> put it at an higher level: i.e. replace the image_impl or buffer_impl, and do it 
// *in the accessor*


class render_pass_ctx {
public:

    // set arguments
    // set textures
    // set uniforms
    // set pipelines

    void set_uniform_1i(program_handle& prog, int value);
    void set_uniform_3fv(program_handle& prog, std::span<const float, 3> v);
    void set_uniform_4fv(program_handle& prog, std::span<const float> v);
    void set_uniform_matrix_4fv(program_handle& prog, std::span<const float> v, bool transpose);

    void bind_texture(size_t slot, texture_handle& tex);

    // what do we take here?
    // - buffer_handle: loses stride information
    // - accessor<vertex_buffer>: 
    void bind_vertex_buffer(size_t slot, buffer_handle&)

        void draw(size_t vertex_count, size_t instance_count, size_t first_vertex, size_t first_instance);
};

for (auto&& geom : geometries) {
    // set common material uniforms
    geom.bind();

    // what could go wrong here (with the stateful design)?
    // - bind() does not care about the previously bound vertex buffers, so some of them may overlap
    //      eventually all vertex buffer slots will be filled (until the bound buffers are deleted)
    // 
    // the thing is that most APIs, even modern ones, are stateful: you bind stuff to the command context
    // and it affects all subsequent commands
    // - a "stateless" API on top of it needs a state cache to eliminate redundant state changes, which 
    //   represents more code to maintain
    // - stateless draw calls also need a bunch of parameters for each submission, which is more verbose
    //      - or store stuff into "parameter groups", but that's more code on the library side
    //
    //
    // 3 options:
    // - full stateless: specify shaders, uniforms, render states, vertex input, and target on every draw-call
    // - stateful: separate bind functions, remains bound until unbound (either manually or by deletion of the underlying buffer)
    // - hybrid (closure-based): 
    //      - queue commands using a series of nested closures that describe one bind point
    //      - see luminance (rust) for the principle ("gates")
    //        https://docs.rs/luminance/0.42.1/luminance/pipeline/index.html
    // Hybrid solution:
    // - fixed hierarchy of closures, sort by (approximate) context-switching cost
    // See https://computergraphics.stackexchange.com/questions/37/what-is-the-cost-of-changing-state
    // - issue: the "state hierarchy" is fixed, changing it means changing all call sites
    // - can't "split" some states (e.g. all textures specified together, all uniforms, etc.)
    // 
    // Modern APIs:
    // - descriptor sets (group of ubo & tex bindings)
    // - framebuffers
    // - vertex buffers
    // - pipelines (program+render states)
    //
    // OpenGL:
    // - Render target (framebuffers)
    // - Program
    // - Textures
    // - VAO
    // - UBO
    // - VBO
    // 
    // Proposed state hierarchy:
    // 1. framebuffer (& render pass)
    // 2. pipeline (& dynamic render states)
    // 3. descriptor sets (2~3 levels)
    // 4. vertex buffers
    // 5. draw calls

    // Corresponding types:
    // begin() 
    // - 1. -> pipeline_ctx
    // - 2. -> arguments_ctx
    // - 3. -> geometry_ctx
    // - 4. -> draw_ctx

    // Issues:
    // 1. render target
    //      - framebuffer
    // 2. pipeline
    //      - depth test
    //      - stencil
    //      - blending
    //      - program
    // 3. descriptor sets (arguments)
    //      - uniform buffers
    //      - immediate uniforms
    //      - textures, images
    // 4. geometry
    //      - vertex buffers
    // 5. draw (draw items)
    //      - (OpenGL) topology
    //      - (OpenGL) data layout
    //      - first vertex
    //      - number of vertices
    //      - instance count
    //  
    // Problem:
    // - the "draw_item" abstraction (lv. 5) also contains the primitive topology , which is lv.2 (pipeline) in Vulkan.
    //      -> meh. there's a recent vulkan extension for dynamic primitive topologies, and DX12 has IASetPrimitiveTopology
    //      -> assume topology is dynamic; 
    //      -> aside: vulkan is very rigid, but does it even reflect the way GPU works? (no way to know, GPU archs are poorly documented)
    // - metal has "topology classes" (point,line,triangle) and "primitive types" (point,line,linestrip,triangle,trianglestrip)
    //      -> primitive types are set dynamically
    // - same for D3D12
    // - webgpu, being the shitty lowest common denominator that it is, has the pipeline topology hard-coded in the pipeline.
    //      -> https://github.com/gpuweb/gpuweb/issues/26
    // 
    // Options:
    // - A: for vulkan, create pipeline objects on the fly OR store variants (created on-demand) for other pipeline topologies
    // - B: require the topology in the geometry to match the one set in the pipeline
    //
    // Same problem for vertex inputs
    // - OpenGL: separate, in VAO
    // - Vulkan: in pipeline
    // - D3D12: in pipeline
    //
    // -> How about creating the geometries *after* the pipelines
    //      -> this way, geometry objects can be "tailored" for a pipeline
    //      -> this make the geometry objects too specific 
    // -> the geometry must conform to the "input layout"
    //      -> share an object that describes an input layout
    //      -> OpenGL: pass a VAO
    //
    // -> issue: complicated logic in geometries so that the data conforms to a format
    //      -> conversions, etc.
    // -> predefined formats?
    //
    // ** we need to know what kind of data we want to handle
    //  -> Possible answer: it's for experimentation, so we don't know...
    //  -> Goal: avoid permutation hell  
    //
    // - Proposition 1: a few predefined formats for the base geometry data
    //          
    // - Proposition 2:
    //     vertex colors in separate (auxiliary) buffers (always float4)
    //
    // - Proposition: vertex attributes in one buffer are always known statically (encoded in a type)
    // - Proposition: however, the *number of buffers* may be dynamic.
    //  => OK
    // - When loading geometries, make the loaded data conform to the static vertex type
    //
    // template<class... vertex>
    // class vertex_buffer
    // 

    // Take inspiration from Metal (apparently it's somewhat easier to use?)
    // - the only scope is the "render pass"
    //      - input: framebuffer (attachements)
    //      - set pipeline is dynamic
    //      - "argument buffers" => ~ descriptor set



    for (auto&& draw_item : geom.draw_items()) {
        // set material-specific uniforms, or filter by material

        draw_item.draw();
    }

/*live_set prev_in;
live_set prev_out;

bool end;
do {
  end = true;
  for (size_t i = 0; i < num_tasks; ++i) {
    auto &in = in_sets[i];
    prev_in = in;
    auto &out = out_sets[i];
    prev_out = out;
    auto &task = tasks[i];

    in = out;
    out.reset();
    for (auto w : task.writes) {
      in.reset(w);
    }
    for (auto r : task.reads) {
      in.set(r);
      //out.set(r);
    }

    //for (auto s : task.succs) {
        if (i < num_tasks - 1) {
          // assume sequential execution, single successor
        out |= in_sets[i+1];
    }
    //}

    fmt::print("in [{:2d}]: ", i);
    for (size_t j = 0; j < num_temporaries; ++j) {
      if (in[j]) {
        fmt::print("{:02d} ", j);
      } else {
        fmt::print("   ");
      }
    }
    fmt::print("\n");

    fmt::print("out[{:2d}]: ", i);
    for (size_t j = 0; j < num_temporaries; ++j) {
      if (out[j]) {
        fmt::print("{:02d} ", j);
      } else {
        fmt::print("   ");
      }
    }
    fmt::print("\n");

    end &= (prev_in == in) && (prev_out == out);
  }
  // fmt::print("\n");

} while (!end);*/

/*
namespace detail {
    template <size_t N>
    constexpr size_t find_attribute_by_semantic(const std::array<vertex_attribute, N>& attribs, semantic semantic) {
        for (size_t i = 0; i < N; ++i) {
            if (attribs[i].semantic == semantic) { return i; }
        }
        return static_cast<size_t>(-1);
    }
}*/

/*
template <typename VertexType>
constexpr VertexType
build_from_generic_vertex(const glm::vec3 &position, const glm::vec2 &texcoords,
                          const glm::vec3 &normals, const glm::vec3 &tangents,
                          const glm::vec3 &bitangents)
{
   auto attributes = std::span{ vertex_traits<VertexType>::attributes };

  VertexType out;


  if constexpr (pos_index != static_cast<size_t>(-1)) {
      out.get<pos_index>() = ;
  }
}*/

// 3. Adjust live-sets for parallel execution.
//
// Some tasks may run in parallel because they have no data-dependencies
// between them. However, the liveness analysis performed above assumes serial
// execution of all tasks in the order of which they appear in the list.
// Following this analysis, resource allocation may map two tasks on the same
// resource, even if those two tasks could run in parallel. This forces the
// two tasks to run serially, which negatively impacts performance.
//
// In other words, we only want to map the same resource to two different
// tasks if we can prove that they can't possibly run in parallel. To account
// for that, we modify the live-sets of each task by adding the union of the
// live sets of all tasks that could run in parallel.

/*for (size_t i = 0; i < num_tasks; ++i) {
  // add variables that are potentially alive at the same time due to parallel
  // execution
  for (size_t j = 0; j < i; ++j) {
    // i || j if there's no path from j to i AND there's no path from i to j
    if (i != j && !reachability[i][j] && !reachability[j][i]) {
      fmt::print("#{} || #{}\n",i,j);
      in_sets[i] |= out_sets[j];
          // but remove variables that are "known dead" on our branch
          //
      in_sets[j] |= out_sets[i]; // the same is also true =
    }
  }
}*/
// live_set(t) = live_set(t-1) U defs(t) U uses(t)

// remove dead vars

// dead vars are vars with a def in the successors without a use between
// and no use on a parallel branch

// TODO should only consider vars that are in the def/use set
// otherwise we're searching for nothing
/* auto it = std::remove_if(cur_live.begin(), cur_live.end(), [&](auto v) {
   // if var in current def/use set, skip
   if (std::binary_search(tasks[t].writes.begin(), tasks[t].writes.end(),
                          v) ||
       std::binary_search(tasks[t].reads.begin(), tasks[t].reads.end(), v)) {
     return false;
   }

   bool killed = true;
   for (size_t succ = t + 1; succ < num_tasks; ++succ) {
     if (std::binary_search(tasks[succ].writes.begin(),
                            tasks[succ].writes.end(), v)) {
       break; // there's a def, so it's dead
     }
     if (std::binary_search(tasks[succ].reads.begin(),
                            tasks[succ].reads.end(), v)) {
       killed = false; // there's a use without a def before
       break;
     }
   }
   // if we fall off the loop, this means that there were no
   // no defs, and no uses: assume it's dead
   return killed;
 });
 cur_live.erase(it, cur_live.end());

 // live_set(t) = (live_set(t-1) U defs(t) U uses(t)) - killed(t)

 live_sets.push_back(cur_live);
}*/

 {namespace detail {

class task_graph_impl;

// represents a task
// - dependencies
// - status (scheduled | finished | error)
// - error message (optional)
class task;
using task_id = int;

/// @brief A possibly shared image resource (wrapper around an opengl texture).
struct image_resource;

// - parent graph (optional)
// - pointer to image resource (possibly shared)
// - list of revisions
class image_impl;

} // namespace detail

class evaluation_error;
class invalid_operation;

/// Execution graph
class task_graph;

/// Represents a simple, single value of type T
template <typename T> class value;

/// Represents a buffer resource
// size [specified or unspecified]
// residency
template <typename T> class buffer;

/// When the data is accessible.
enum class visibility {
  evaluation_only,     ///< Only accessible during evaluation
  evaluation_and_host, ///< Enable host access
};

using image_1d = image<1>;
using image_2d = image<2>;
using image_3d = image<3>;

enum class access_mode {
  read_only,
  write_only,
  read_write,
};

/// Abstraction for an access to a resource
// residency: device | host
// resource: value | buffer | image
template <typename Resource, access_mode AccessMode> class accessor;

// for graphics:
// - buffers: vertex_buffer | index_buffer | uniform_buffer | storage_buffer |
// host_read | host_write
// - images: sampled_image | storage_image | framebuffer_attachment | upload |
// readback Image can be texture or renderbuffer underneath (that's abstracted
// depending on usage). Create multiple accessors if the resource is accessed in
// multiple ways.
//
// Under the hood, accessors can do "complicated" stuff, like:
// - for upload, allocate or use a staging buffer
// - for readback, use a PBO + fence
// - for sampled images: also set the sampling information
//
//
//  type of access | read/write
//  vertex_buffer  | RO
//  index_buffer   | RO
//  uniform_buffer | RO
//  storage_buffer | RO/RW
//  host_write
//
// constructor forms:
// { buf, vertex_buffer (mode+target) }           => <buffer<>,
// target::vertex_buffer, access_mode::read_only> { buf, index_buffer
// (mode+target) }          => <buffer<>, target::index_buffer,
// access_mode::read_only> { buf, uniform_buffer (mode+target) }
// => <buffer<>, target::uniform_buffer, access_mode::read_only> { buf,
// transform_feedback (mode+target) }           => <buffer<>,
// target::transform_feedback, access_mode::write_only>

// { buf, storage_buffer }            => <buffer<>,
// target::storage_buffer, access_mode::read_only> { buf, storage_buffer,
// read_only }       => <buffer<>, target::storage_buffer,
// access_mode::read_only> { buf, storage_buffer, write_only }      =>
// <buffer<>, target::storage_buffer, access_mode::write_only> { buf,
// storage_buffer, read_write }      => <buffer<>, target::storage_buffer,
// access_mode::read_write>

// { img, sampled_image }             =>
// <image<>,target::sampled_image, access_mode::read_only> { img, storage_image
// }          => <image<>,target::sampled_image,
// access_mode::read_only> { img, storage_image, read_only }    =>
// <image<>,target::sampled_image, access_mode::read_only> { img, storage_image,
// read_write }   => <image<>,target::sampled_image,
// access_mode::read_write> { img, storage_image, write_only }      =>
// <image<>,target::sampled_image, access_mode::write_only>

/// Provides a way to schedule commands.
class scheduler;

} // namespace graal

    // create an execution graph
    // a new execution graph should be created for every evaluation
    task_graph graph;

    // create an image resource
    image_2d img{image_format::r16g16_sfloat, {1280, 720}};
    // create a virtual image resource: it can only be read within a queue operation
    image_2d vimg { image_format::r16g16_sfloat, virtual_image };


    buffer<Vertex> mesh_vtx_buffer{ 6 };

    // schedule a task
    graph.schedule([&](scheduler &sched) {

        // use mesh_vtx as an upload target
        // will automatically enable a `data() -> T*` function to write data
        accessor vbo_upload{mesh_vtx, host_write, sched};

        // use mesh_vtx as a vertex buffer
        accessor vbo_use{mesh_vtx, vertex_buffer, sched};

      // request write access to the image
      accessor img_access{img, write_access, sched};

      // accessor<image<N>, access_mode::render_target
      accessor rt{img, framebuffer_attachment, sched};

      accessor cam_ubo{cam_buf, uniform_buffer, sched};

      accessor img{img, transfer_destination, sched};

      // schedule a simple task
      sched.gl_commands([=] {
        // fetch the OpenGL texture object
        img_access.get_texture_object();

        // do something to the texture via the current OpenGL context
      });
    });

    // schedule another task
    graph.schedule([&](scheduler &sched) {
      // request read access to img
      // this creates a dependency between the previous task that writes to img
      // and this task you can ask for device (GPU) or host access (CPU)
      // depending on how the node is implemented.
      // 

      accessor img_access{img, sampled_image, sched};

      // request access to img as a framebuffer attachment
      // if the resource is virtual AND access is write-only (discard)
      // then 

      // uniform buffer
      accessor {buf, uniform_buffer, sched};
      
      // storage buffer
      accessor {buf, storage_buffer, sched};
      accessor {buf, storage_buffer, read_only, sched};
      accessor {buf, storage_buffer, read_write, sched};
      accessor {buf, storage_buffer, write_only, sched};

      // vertex/index buffer
      accessor {buf, vertex_buffer, sched};
      accessor {buf, index_buffer, sched};

      // transfer
      accessor {buf, transfer_source, sched };
      accessor {buf, transfer_destination, sched };
      accessor {img, transfer_source, sched };       // pixel transfer source
      accessor {img, transfer_destination, sched };  // pixel transfer destination

      // sampled_image
      accessor {img, sampled_image, sched };

      // storage_image
      accessor {img, storage_image, sched };
      accessor {img, storage_image, read_only, sched };
      accessor {img, storage_image, read_write, sched };
      accessor {img, storage_image, write_only, sched };

      // framebuffer attachment
      accessor {img, framebuffer_attachment, clear { 0.0, 0.0, 0.0, 1.0 }, sched };
      accessor {img, framebuffer_attachment, discard, sched };
      accessor {img, framebuffer_attachment, keep, sched };
      accessor {img, framebuffer_attachment, sched }; // default is keep


      // overloads:
      // framebuffer_attachment_tag_t, framebuffer_load_op_clear_t
      // framebuffer_attachment_tag_t, framebuffer_load_op_discard_t
      // framebuffer_attachment_tag_t, framebuffer_load_op_keep_t
      // framebuffer_attachment_tag_t 
      // 


      // schedule a simple task
      sched.compute([=] {
        // fetch the OpenGL texture object
        img_access.get_texture_object();

        // do something with the texture via the current OpenGL context
        // ...
      });
    });
        

    // trying to access an image here will launch the graph and block until
    // everything has finished computing

    // I think that if get_texture_object is called, 
    // and if the texture is fully specified, 
    // then it should *always* create and return 
    // a texture object, even if it is bound to a graph. 
    // It could be a way to "force" materialization of a texture.

    // Another point of view: this can be "surprising", as calling get_texture_object() at
    // the wrong place would prevent optimization
    // Instead, put *in the type* whether it's possible to externally access the texture or not.
    GLuint tex_obj = img.get_texture_object();
  }

  // in this API, the "resources" (images, etc.) are specified externally
  // - image
  // - buffer<T>
  // - value<T>
  //
  // images can be "virtual", which means that they can't be accessed outside of
  // a task "virtual" images are resolved to concrete images by the evaluator

  // problem:
  // - a function that produces an image from an input should return an image
  // object
  // - however, it is up to the caller to decide whether this would be a
  // "virtual" image
  //   or a concrete image accessible outside evaluation
  // This means that the specification of an image is "split" between two parts:
  // - the "node", that determines the width, height, format of the image
  // depending on the input
  // - the client, that defines where the image is going to live

  // image objects can have "unspecified" properties: width, height, format,
  // residency

  // rebuilding the graph for every evaluation?
  // - already done right now

  image img_input;

  // virtual_image v_img;

  string_value str;

  load_image("test.exr", v_img);

// the delayed image specification is purely an ergonomic decision
// this is because we expect to support functions of the form:
//
//    void filter(const image& input, image& output);
//
// where the filter (the callee) decides the size of the image,
// but the *caller* decides other properties of the image, such as its required
// access (evaluation only, or externally visible)

// this causes issues:
// - should the image be fully specified on first access?
//    Yes (and that's different from )


external access : statically 
dimensions      : statically
size            : before access, no default
format          : before access, no default
multisampling   : before access, default is single sample
mipmaps         : before access, default is one mipmap


// 
// caller must compute the number of mipmaps 
// but does not need to specify OutputExt when calling
template<int D, bool InputExt, bool OutputExt>
build_mipmaps(image<D,InputExt> & input, image<D,OutputExt>& output)

// alt.
// caller must specify external access for the output image,
// but does not need to compute the number of mipmaps
build_mipmaps<OutputExt, InputExt>(image<D>& input) -> image<D,OutputExt>()

// what about a filter that supports multiple output formats?
// - either pass it as a parameter
// - or use the provided 



template <int D, typename InputExt, typename OutputExt
build_mipmaps(image<D,InputExt>& input, image<D,OutputExt>& output);

// callee-specified:
// size
// format
// multisample


  image in { ... };

  image_2d out { virtual_image };

  build_mipmaps(in, out);


image: 

format array dimensions mipmaps multisample(n) virtual_image sparse_storage
// -> too many combinations (2**7 = 128 constructors?)
// keep all params that can be used to deduce the image static type, and put the rest in an unordered property list:

array dimensions virtual_image 
  properties:
    - mipmaps
    - multisample
    - sparse_storage 

// issue: multiply-defined properties of the same type?

image in { array, range{512,512,512} };   // image 2D array, 512x512x512
image in { range{512, 512, 512} };        // image 3D, 512x512x512
image in { array, range{512,512}, virtual_image }; // virtual image 1D array, 512x512
image in { array, range{512,512}, mipmaps{0}, virtual_image }; // virtual image 1D array, 512x512
image in { array, range{512,512}, auto_mipmaps, virtual_image }; // virtual image 1D array, 512x512
image in { range{512,512}, auto_mipmaps, multisample{8}, virtual_image }; // virtual image 1D array, 512x512
image_2d in { multisample{8}, virtual_image }; // virtual image 1D array, 512x512


array_image { image_format::r8g8b8a8, range{512,512,512} }

image in { image_format::r8g8b8a8, image_array, range{512,512,512}, virtual_image, 

          image_properties { auto_mipmaps, multisample{8}, sparse }};

          image in { image_format::r8g8b8a8, }

format
format array 
format array dimensions
format array dimensions multisample
format array dimensions multisample virtual_image

       array
format array 
format array dimensions
format array dimensions multisample
format array dimensions multisample virtual_image

             dimensions
       array dimensions 
format array dimensions
format array dimensions multisample
format array dimensions multisample virtual_image

                        multisample
             dimensions multisample
       array dimensions multisample
format array dimensions multisample
format array dimensions multisample virtual_image


                                    virtual_image
                        multisample virtual_image
             dimensions multisample virtual_image
       array dimensions multisample virtual_image
format array dimensions multisample virtual_image

32 combinations