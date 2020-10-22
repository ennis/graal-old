#include <graal/accessor.hpp>
#include <graal/buffer.hpp>
#include <graal/debug_output.hpp>
#include <graal/glad.h>
#include <graal/image.hpp>
#include <graal/program.hpp>
#include <graal/queue.hpp>
#include <graal/shader.hpp>
#include <graal/vertex_array.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <typeindex>

using namespace graal;

#define CIMG(NAME)                                                             \
  image NAME{image_format::r16g16_sfloat, range{1280, 720}};                   \
  NAME.set_name(#NAME);

#define VIMG(NAME)                                                             \
  image NAME{virtual_image, image_format::r16g16_sfloat, range{1280, 720}};    \
  NAME.set_name(#NAME);

#define READ(img) accessor read_##img{img, sampled_image, h};
#define DRAW(img) accessor draw_##img{img, framebuffer_attachment, h};
#define WRITE(img) accessor draw_##img{img, framebuffer_attachment, discard, h};

void test_case_1(graal::queue &q);
void test_case_2(graal::queue &q);

/// @brief Returns the contents of a text file as a string
/// @param path
/// @return
static std::string read_text_file(std::string_view path) {
  std::ifstream file_in{std::string{path}};
  std::string   source;

  if (!file_in) {
    throw std::runtime_error{
        fmt::format("could not open file `{}` for reading")};
  }

  file_in.seekg(0, std::ios::end);
  source.reserve(file_in.tellg());
  file_in.seekg(0, std::ios::beg);

  source.assign(std::istreambuf_iterator<char>{file_in},
                std::istreambuf_iterator<char>{});
  return source;
}

template <typename T> struct vertex_traits;

struct vertex_2d {
  std::array<float, 2> pos;
  std::array<float, 2> tex;
};

template <> struct vertex_traits<vertex_2d> {
  static constexpr vertex_attribute attributes[] = {
      {data_type::float_, 2, offsetof(vertex_2d, pos)},
      {data_type::float_, 2, offsetof(vertex_2d, tex)}};
};

class vertex_array_cache {
public:
  template <typename VertexType> std::shared_ptr<vertex_array_handle> get() {
    static_assert(std::is_standard_layout_v<VertexType>,
                  "VertexType must be standard-layout");
    const type_index vtx_type_index{typeid(VertexType)};
    if (auto it = vertex_arrays_.find(vtx_type_index);
        it != vertex_arrays_.end()) {
      return it->second;
    } else {
      vertex_array_builder vao_builder;
      vao_builder.set_attributes(0, 0, vertex_traits<VertexType>::attributes);
      auto vao =
          std::make_shared<vertex_array_handle>(vao_builder.get_vertex_array());
      vertex_arrays_.insert({vtx_type_index, std::move(vao)});
    }
  }

  static vertex_array_cache &global_cache() noexcept {
    static vertex_array_cache instance;
    return instance;
  }

private:
  std::unordered_map<std::type_index, std::shared_ptr<vertex_array_handle>>
      vertex_arrays_;
};

/// @brief Application state
class app_state {
public:
  app_state() { reload_shaders(); }

  void reload_shaders() {
    auto vertex_src = read_text_file("data/shaders/background.vert");
    auto fragment_src = read_text_file("data/shaders/background.frag");

    try {
      auto vertex_shader = compile_shader(shader_stage::vertex, vertex_src);
      auto fragment_shader =
          compile_shader(shader_stage::fragment, fragment_src);
      bg_program = program_handle{};
      bg_program = create_program(vertex_shader.get(), fragment_shader.get(), 0,
                                  0, 0, std::cerr);
    } catch (const shader_compilation_error &err) {
      fmt::print(stderr, "could not compile shader:\n {}\n", err.log());
    } catch (const program_link_error &err) {
      fmt::print(stderr, "could not link program:\n {}\n", err.log());
    }
  }

  void setup() {
    // build buffer for screen rect
    const float left = -1.0f;
    const float right = -1.0f;
    const float top = 1.0f;
    const float bottom = 1.0f;

    const vertex_2d data[] = {
        {{left, top}, {0.0, 0.0}},    {{right, top}, {1.0, 0.0}},
        {{left, bottom}, {0.0, 1.0}}, {{left, bottom}, {0.0, 1.0}},
        {{right, top}, {1.0, 0.0}},   {{right, bottom}, {1.0, 1.0}}};

    vertices = buffer{std::span{data}};

    // build VAO for 2D vertices
    vertex_array_builder vao_builder{};
    vao_builder.set_attributes(0, 0, vertex_traits<vertex_2d>::attributes);
    vao = vao_builder.get_vertex_array();

    // what is needed?
    // - VAO cache (class vertex_array_cache)
    // - "scenes", containing a hierarchy of objects
    // - *one* function to import objs in the scene
    // - objects are meshes and lights
    // - meshes contain vertex data
    //     - at least positions, but maybe also normals, tangents, bitangents,
    //     texcoords, arbitrary vertex colors...
    //     - shaders must be aware of this distinction
    //     - a single input signature?
    //        - in vec3 position;
    //        - in vec3 normals;
  }

  void render(queue &q) {
    q.schedule("background render", [](handler &h) {

    });
  }

private:
  buffer<vertex_2d>   vertices;
  vertex_array_handle vao;
  program_handle      bg_program;
};

int main() {

  //=======================================================
  // SETUP

  GLFWwindow *window;
  if (!glfwInit())
    return -1;

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
  if (!window) {
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);

  gladLoadGL();
  setup_debug_output();
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  (void)io;
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 450");

  //=======================================================
  queue q;

  // ---- Main loop ----
  bool  reload_shaders = true;
  float clear_color[4] = {0.0, 0.0, 0.0, 1.0};

  while (!glfwWindowShouldClose(window)) {

    glfwPollEvents();
    //    draw_frame();
    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (ImGui::Begin("GRAAL")) {

      if (ImGui::Button("Reload shaders")) {
        reload_shaders = true;
      }

      ImGui::End();
    }

    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
  }

  // Cleanup
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}

//=============================================================================

void draw_frame(graal::queue &q) {}

//=============================================================================
void test_case_1(graal::queue &q) {

  VIMG(A)
  VIMG(B)
  VIMG(C)
  VIMG(D1)
  VIMG(D2)
  VIMG(E)
  VIMG(F)
  VIMG(G)
  VIMG(H)
  VIMG(I)
  VIMG(J)
  VIMG(K)

  // q.schedule("INIT", [&](scheduler &sched) { WRITE(A) WRITE(B) WRITE(C) });

  q.schedule("T0", [&](handler &h) { DRAW(A) });
  q.schedule("T1", [&](handler &h) { DRAW(B) });
  q.schedule("T2", [&](handler &h) { DRAW(C) });

  q.schedule("T3", [&](handler &h) {
    READ(A)
    READ(B)
    WRITE(D1)
    WRITE(D2)
  });

  q.schedule("T4", [&](handler &h) {
    READ(D2)
    READ(C)
    WRITE(E)
  });

  q.schedule("T5", [&](handler &h) {
    READ(D1)
    WRITE(F)
  });

  q.schedule("T6", [&](handler &h) {
    READ(E)
    READ(F)
    WRITE(G)
  });

  q.schedule("T7", [&](handler &h) { READ(G) WRITE(H) });
  q.schedule("T8", [&](handler &h) { READ(H) WRITE(I) });
  q.schedule("T9", [&](handler &h) { READ(I) READ(G) WRITE(J) });
  q.schedule("T10", [&](handler &h) { READ(J) WRITE(K) });

  A.discard();
  B.discard();
  C.discard();
  D1.discard();
  D2.discard();
  E.discard();
  F.discard();
  G.discard();
  H.discard();
  I.discard();
  J.discard();

  q.enqueue_pending_tasks();

  q.schedule("T10", [&](handler &h) {
    accessor write_K{K, framebuffer_attachment, discard, h};

    auto f = [=] {
      // do something with tex
      auto tex = write_K.get_gl_object();
    };
  });
  q.enqueue_pending_tasks();
}

void test_case_2(queue &q) {
  VIMG(S);

  VIMG(C);

  {
    VIMG(A1);
    VIMG(A2);
    VIMG(A3);
    VIMG(B1);
    VIMG(B2);
    VIMG(B3);
    q.schedule([&](handler &h) { WRITE(S) });
    // A branch
    q.schedule([&](handler &h) { READ(S) WRITE(A1) });
    q.schedule([&](handler &h) { READ(A1) WRITE(A2) });
    q.schedule([&](handler &h) { READ(A2) WRITE(A3) });
    // B branch
    q.schedule([&](handler &h) { READ(S) WRITE(B1) });
    q.schedule([&](handler &h) { READ(B1) WRITE(B2) });
    q.schedule([&](handler &h) { READ(B2) WRITE(B3) });
    // C
    q.schedule([&](handler &h) { READ(A3) READ(B3) WRITE(C) });
  }
  q.enqueue_pending_tasks();
}