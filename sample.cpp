
struct vertex_3d {
	glm::vec3 position;
	glm::vec3 normals;
	glm::vec3 tangents;
};

// format of seed points within the buffer; only used for specifying the buffer size
struct seed_point {
	float x,y,z;
};

struct curve {
	float coefs[16];
};

// scene uniforms
struct scene_uniforms {
	glm::mat4 view;
	glm::mat4 proj;
	glm::mat4 view_proj_inv;
	// ...
}


struct pipelines {
	graphics_pipeline gen_gbuffers;
	graphics_pipeline gen_solid_noise;
	compute_pipeline collect_points;
	compute_pipeline gen_flow;
	compute_pipeline gen_curves;
	compute_pipeline bucket_curves;
	compute_pipeline eval_curves;
}

constexpr image_format gbuffers_normals_format = image_format::r32g32b32a32;

pipelines load_pipelines() 
{
	// compile shader modules
	const auto gbuffers_vert  = compile_shader("path/to/gbuffers.vert");
	const auto gbuffers_frag = compile_shader("path/to/gbuffers.frag");
	const auto gen_solid_noise_vert = compile_shader("path/to/solid_noise.vert");
	const auto gen_solid_noise_frag = compile_shader("path/to/solid_noise.frag");

	const auto collect_points = compile_shader("path/to/collect_points.comp");
	const auto gen_flow = compile_shader("path/to/gen_flow.comp");
	const auto gen_curves = compile_shader("path/to/gen_curves.comp");
	const auto bucket_curves = compile_shader("path/to/bucket_curves.comp");
	const auto eval_curves = compile_shader("path/to/eval_curves.comp");

	// create gbuffers render pass
	render_pass_builder gbuffers_rp;
	// no input attachments, single subpass, two outputs: mask & 
	gbuffers_rp.subpass(0).color_attachment(0, )

	//attachment { gbuffers_normals_format,  }

}

int main()
{
	// first, create the output window (omitted here)

	// create the device
	device dev{required_instance_extensions};

	// create the queue
	queue queue{dev};

	// create resources that are constant across frames
    buffer<vertex_3d> model_vertices{...};	// TODO load from OBJ

    // create shaders and pipelines
    shader_module gbuf_vertex_shader = 


	// enter frame loop
  	while (!glfwWindowShouldClose(window)) {
    	glfwPollEvents();

    	// get window width and height
    	const size_t width = ...;
    	const size_t height = ...;
    	// tile size in pixels
    	const size_t tile_width = 64; 
    	const size_t tile_height = 64; 
    	// estimate the maximum number of points (say, 2048 per tile)
    	const size_t max_points = width*height/2;

    	// create resources (aliasing is automatic)
    	image gbuf_normals { image_format::r32g32b32, range{width, height} };	// RGBA32F normals
    	buffer<seed_point> output_points {max_points}; 
    	buffer<curve> output_curves {max_points}; 


    	// vertex buffers


    	// 1. draw gbuffers
	}
}