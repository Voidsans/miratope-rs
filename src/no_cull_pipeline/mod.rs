use bevy::prelude::*;
use bevy::core::FixedTimestep;
use bevy::asset::{Assets, HandleUntyped, Handle};
use bevy::reflect::TypeUuid;
use bevy::ecs::Bundle;
use bevy::render::{
    mesh::Mesh,
    render_graph::base::MainPass,
    pipeline::{
        BlendDescriptor, BlendFactor, BlendOperation, ColorStateDescriptor, ColorWrite,
        CompareFunction, CullMode, DepthStencilStateDescriptor, FrontFace, PipelineDescriptor,
        RasterizationStateDescriptor, StencilStateDescriptor, StencilStateFaceDescriptor,
        RenderPipeline
    },
    shader::{Shader, ShaderStage, ShaderStages},
    texture::TextureFormat,
};
type TF7 = nalgebra::Transform<f32, nalgebra::U7, nalgebra::TProjective>;

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct Transform7(pub TF7);

impl Default for Transform7 {
    fn default() -> Self {
        Transform7(TF7::identity())
    }
}

impl AsRef<TF7> for Transform7 {
    fn as_ref(&self) -> &TF7 {
        &self.0
    }
}

impl AsMut<TF7> for Transform7 {
    fn as_mut(&mut self) -> &mut TF7 {
        &mut self.0
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct GlobalTransform7(pub TF7);

impl AsRef<TF7> for GlobalTransform7 {
    fn as_ref(&self) -> &TF7 {
        &self.0
    }
}

impl AsMut<TF7> for GlobalTransform7 {
    fn as_mut(&mut self) -> &mut TF7 {
        &mut self.0
    }
}

impl Default for GlobalTransform7 {
    fn default() -> Self {
        GlobalTransform7(TF7::identity())
    }
}

impl From<Transform7> for GlobalTransform7 {
    fn from(Transform7(tf): Transform7) -> Self {
        GlobalTransform7(tf)
    }
}

pub const NO_CULL_PIPELINE_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(PipelineDescriptor::TYPE_UUID, 0x7CAE7047DEE79C84);

pub fn build_no_cull_pipeline(shaders: &mut Assets<Shader>) -> PipelineDescriptor {
    PipelineDescriptor {
        rasterization_state: Some(RasterizationStateDescriptor {
            front_face: FrontFace::Ccw,
            cull_mode: CullMode::None,
            depth_bias: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
            clamp_depth: false,
        }),
        depth_stencil_state: Some(DepthStencilStateDescriptor {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: CompareFunction::Less,
            stencil: StencilStateDescriptor {
                front: StencilStateFaceDescriptor::IGNORE,
                back: StencilStateFaceDescriptor::IGNORE,
                read_mask: 0,
                write_mask: 0,
            },
        }),
        color_states: vec![ColorStateDescriptor {
            format: TextureFormat::default(),
            color_blend: BlendDescriptor {
                src_factor: BlendFactor::SrcAlpha,
                dst_factor: BlendFactor::OneMinusSrcAlpha,
                operation: BlendOperation::Add,
            },
            alpha_blend: BlendDescriptor {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Add,
            },
            write_mask: ColorWrite::ALL,
        }],
        name: Some("N-D Projection Pipeline".to_string()),
        ..PipelineDescriptor::new(ShaderStages {
            vertex: shaders.add(Shader::from_glsl(
                ShaderStage::Vertex,
                include_str!("forward.vert"),
            )),
            fragment: Some(shaders.add(Shader::from_glsl(
                ShaderStage::Fragment,
                include_str!("forward.frag"),
            ))),
        })
    }
}

#[derive(Bundle)]
pub struct PbrNoBackfaceBundle {
    pub mesh: Handle<Mesh>,
    pub material: Handle<StandardMaterial>,
    pub main_pass: MainPass,
    pub draw: Draw,
    pub visible: Visible,
    pub render_pipelines: RenderPipelines,
    pub transform: Transform7,
    pub global_transform: Transform7,
}

impl Default for PbrNoBackfaceBundle {
    fn default() -> Self {
        Self {
            render_pipelines: RenderPipelines::from_pipelines(vec![RenderPipeline::new(
                NO_CULL_PIPELINE_HANDLE.typed(),
            )]),
            mesh: Default::default(),
            visible: Default::default(),
            material: Default::default(),
            main_pass: Default::default(),
            draw: Default::default(),
            transform: Default::default(),
            global_transform: Default::default(),
        }
    }
}

pub const PBY7D_RENDER_STAGE: &str = "pbr 7d render";

#[derive(Debug, Clone, Copy)]
pub struct Pbr7DPlugin {
    pub fps_cap: f64,
}

impl Default for Pbr7DPlugin {
    fn default() -> Self {
        Pbr7DPlugin {
            fps_cap: 60.0,
        }
    }
}

impl Plugin for Pbr7DPlugin {
    fn build(&self, app: &mut AppBuilder) {
        app
            .add_system_to_stage(stage::POST_UPDATE, update_globals_from_local_tfs.system())
            .add_stage_after(stage::UPDATE, PBY7D_RENDER_STAGE, SystemStage::parallel()
                .with_run_criteria(FixedTimestep::steps_per_second(self.fps_cap))
        );
    }
}

fn update_globals_from_local_tfs(
    mut root_query: Query<(Entity, Option<&Children>, &Transform7, &mut GlobalTransform7), (Without<Parent>, With<GlobalTransform7>)>, 
    mut transform_query: Query<(&Transform7, &mut GlobalTransform7), With<Parent>>, 
    changed_transform_query: Query<Entity, Changed<Transform7>>, 
    children_query: Query<Option<&Children>, (With<Parent>, With<GlobalTransform7>)>
) {
    for (entity, children, transform, mut global_transform) in root_query.iter_mut() {
        let mut changed = false;
        if changed_transform_query.get(entity).is_ok() {
            *global_transform = GlobalTransform7::from(*transform);
            changed = true;
        }

        if let Some(children) = children {
            for child in children.iter() {
                update_globals_from_local_tfs_rec(
                    &global_transform,
                    &changed_transform_query,
                    &mut transform_query,
                    &children_query,
                    *child,
                    changed,
                );
            }
        }
    }
}

fn update_globals_from_local_tfs_rec(
    parent: &GlobalTransform7,
    changed_transform_query: &Query<Entity, Changed<Transform7>>,
    transform_query: &mut Query<(&Transform7, &mut GlobalTransform7), With<Parent>>,
    children_query: &Query<Option<&Children>, (With<Parent>, With<GlobalTransform7>)>,
    entity: Entity,
    mut changed: bool,
) {
    changed |= changed_transform_query.get(entity).is_ok();

    let global_matrix = {
        if let Ok((transform, mut global_transform)) = transform_query.get_mut(entity) {
            if changed {
                global_transform.0 = parent.0 * transform.0;
            }
            *global_transform
        } else {
            return;
        }
    };

    if let Ok(Some(children)) = children_query.get(entity) {
        for child in children.iter() {
            update_globals_from_local_tfs_rec(
                &global_matrix,
                changed_transform_query,
                transform_query,
                children_query,
                *child,
                changed,
            );
        }
    }
}
