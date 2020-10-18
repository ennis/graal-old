#include <graal/accessor.hpp>
#include <graal/gl/debug_output.hpp>
#include <graal/gl/glad.h>
#include <graal/image.hpp>
#include <graal/queue.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

#include <fstream>

using namespace graal;

#define CIMG(NAME)                                                             \
  image NAME{image_format::r16g16_sfloat, range{1280, 720}};                   \
  NAME.set_name(#NAME);

#define VIMG(NAME)                                                             \
  image NAME{virtual_image, image_format::r16g16_sfloat, range{1280, 720}};    \
  NAME.set_name(#NAME);

#define READ(img) accessor read_##img{img, sampled_image, sched};
#define DRAW(img) accessor draw_##img{img, framebuffer_attachment, sched};
#define WRITE(img)                                                             \
  accessor draw_##img{img, framebuffer_attachment, discard, sched};

void test_case_1(graal::queue &q);
void test_case_2(graal::queue &q);

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
  gl::setup_debug_output();
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  (void)io;
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 450");

  //=======================================================
  // TEST

  queue q;

  // image i0 { range{1280, 720} };
  // image_cube_map i1 { range{1280, 720} };

  image i0{image_format::r16g16_sfloat, range{1280, 720}};
  image i1{virtual_image, image_format::r16g16_sfloat, range{1280, 720}};

  image_2d       i2{virtual_image};
  image_cube_map i3{range{128, 128}};
  image_2d_array i4{range{128, 128, 256}};
  image_1d_array i5{range{128, 16}, image_properties{mipmaps{4}}};

  multisample_image_2d_array i6{range{128, 16, 4},
                                image_properties{multisample{2}}};

  (void)i1.has_mipmaps();
  (void)i2.has_mipmaps();
  (void)i6.has_samples();
  (void)i6.samples();

  test_case_1(q);
  test_case_2(q);

  //=======================================================
  // TEST

  // Main loop
  bool  show_demo_window = true;
  bool  show_another_window = false;
  float clear_color[4];

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(clear_color[0], clear_color[1], clear_color[2],
                 clear_color[3]);
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

  q.schedule("T0", [&](scheduler &sched) { DRAW(A) });
  q.schedule("T1", [&](scheduler &sched) { DRAW(B) });
  q.schedule("T2", [&](scheduler &sched) { DRAW(C) });

  q.schedule("T3", [&](scheduler &sched) {
    READ(A)
    READ(B)
    WRITE(D1)
    WRITE(D2)
  });

  q.schedule("T4", [&](scheduler &sched) {
    READ(D2)
    READ(C)
    WRITE(E)
  });

  q.schedule("T5", [&](scheduler &sched) {
    READ(D1)
    WRITE(F)
  });

  q.schedule("T6", [&](scheduler &sched) {
    READ(E)
    READ(F)
    WRITE(G)
  });

  q.schedule("T7", [&](scheduler &sched) { READ(G) WRITE(H) });
  q.schedule("T8", [&](scheduler &sched) { READ(H) WRITE(I) });
  q.schedule("T9", [&](scheduler &sched) { READ(I) WRITE(J) });
  q.schedule("T10", [&](scheduler &sched) { READ(J) WRITE(K) });

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

  q.schedule("T10", [&](scheduler &sched) { DRAW(K) });
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
    q.schedule([&](scheduler &sched) { WRITE(S) });
    // A branch
    q.schedule([&](scheduler &sched) { READ(S) WRITE(A1) });
    q.schedule([&](scheduler &sched) { READ(A1) WRITE(A2) });
    q.schedule([&](scheduler &sched) { READ(A2) WRITE(A3) });
    // B branch
    q.schedule([&](scheduler &sched) { READ(S) WRITE(B1) });
    q.schedule([&](scheduler &sched) { READ(B1) WRITE(B2) });
    q.schedule([&](scheduler &sched) { READ(B2) WRITE(B3) });
    // C
    q.schedule([&](scheduler &sched) { READ(A3) READ(B3) WRITE(C) });
  }
  q.enqueue_pending_tasks();
}