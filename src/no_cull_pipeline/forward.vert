#version 450

struct Mat8 {
    mat4 r0c0;
    mat4 r0c1;
    mat4 r1c0;
    mat4 r1c1;
};

struct Vec8 {
    vec4 v0;
    vec4 v1;
};

layout(location = 0) in vec3 Vertex_Position;
layout(location = 1) in vec4 Vertex_Position1;
layout(location = 2) in vec3 Vertex_Normal;
layout(location = 3) in vec4 Vertex_Normal1;
layout(location = 4) in vec2 Vertex_Uv;

layout(location = 0) out vec3 v_Position;
layout(location = 1) out vec3 v_Normal;
layout(location = 2) out vec2 v_Uv;

layout(set = 0, binding = 0) uniform Camera {
    Mat8 ViewProj;
};

layout(set = 2, binding = 0) uniform Transform {
    Mat8 Model;
};

Vec8 mat8mulvec8(const Mat8 m, const Vec8 v) {
    return Vec8(
        m.r0c0 * v.v0 + m.r1c1 * v.v1,
        m.r0c1 * v.v0 + m.r1c0 * v.v1
    );
}

void main() {
    const Vec8 mul = mat8mulvec8(Model, Vec8(
        vec4(Vertex_Position, Vertex_Position1.x),
        vec4(Vertex_Position1.yzw, 1.0)
    ));
    v_Position = mul.v0.xyz;
    gl_Position = vec4(
        mat8mulvec8(ViewProj, mul).v0.xyz,
        1.0
    );
    v_Normal = mat8mulvec8(Model, Vec8(
        vec4(Vertex_Normal, Vertex_Normal1.x),
        vec4(Vertex_Normal1.yzw, 0.0)
    )).v0.xyz;
    v_Uv = Vertex_Uv;
}
