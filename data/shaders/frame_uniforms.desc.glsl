#version 450
#pragma graal_shader_interface

layout(set=0,binding=0,std140) uniform frame_uniforms {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

//void main()
//{}

// generated C++:
// 
// <includes>
//
// namespace <whatever> {
//   struct frame_uniforms {
//   	<mat4-type> model;
//		<mat4-type> view;
//		<mat4-type> proj;
// 	 }
// 
//	 struct frame_uniforms_desc {
//		buffer<frame_uniforms> ubo;	// set=0, binding=0
//   }
//	 
//   extern const VkDescriptorSetLayoutBinding frame_uniforms_layout_bindings[1];
// }
// 