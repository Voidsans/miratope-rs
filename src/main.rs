#![allow(dead_code)]

//! A renderer for polytopes, spinned off from [Miratope JS](https://github.com/OfficialURL/miratope).
//! Still in alpha development.
//!
//! ## What can Miratope do now?
//! Not much. We're still in the early stages of porting the original Miratope's functionality.
//!
//! ## What are Miratope's goals?
//! We plan to eventually support all of the original Miratope's features,
//! as well as the following:
//!
//! * Various families of polytopes to build and render
//!   * All [regular polytopes](https://polytope.miraheze.org/wiki/Regular_polytope)
//!   * All 3D and 4D known [uniform polytopes](https://polytope.miraheze.org/wiki/Uniform_polytope)
//!   * Many of the known [CRFs](https://polytope.miraheze.org/wiki/Convex_regular-faced_polytope)
//! * Many operations to apply to these polytopes
//!   * [Duals](https://polytope.miraheze.org/wiki/Dual)
//!   * [Petrials](https://polytope.miraheze.org/wiki/Petrial)
//!   * [Prism products](https://polytope.miraheze.org/wiki/Prism_product)
//!   * [Tegum products](https://polytope.miraheze.org/wiki/Tegum_product)
//! * Loading and saving into various formats
//!   * Support for the [Stella OFF format](https://www.software3d.com/StellaManual.php?prod=stella4D#import)
//!   * Support for the [GeoGebra GGB format](https://wiki.geogebra.org/en/Reference:File_Format)
//! * Localization
//!   * Automatic name generation in various languages for many shapes
//!
//! ## How do I use Miratope?
//! Miratope doesn't have an interface yet, so you'll have to download the source code to do much of anything.
//!
//! ## Where do I get these "OFF files"?
//! The OFF file format is a format for storing certain kinds of geometric shapes.
//! Although not in widespread use, it has become the standard format for those who investigate polyhedra and polytopes.
//! It was initially meant for the [Geomview software](https://people.sc.fsu.edu/~jburkardt/data/off/off.html),
//! and was later adapted for the [Stella software](https://www.software3d.com/StellaManual.php?prod=stella4D#import).
//! Miratope uses a further generalization of the Stella OFF format for any amount of dimensions.
//!
//! Miratope does not yet include a library of OFF files. Nevertheless, many of them can be downloaded from
//! [OfficialURL's personal collection](https://drive.google.com/drive/u/0/folders/1nQZ-QVVBfgYSck4pkZ7he0djF82T9MVy).
//! Eventually, they'll be browsable from Miratope itself.
//!
//! ## Why is the rendering buggy?
//! Proper rendering, even in 3D is a work in progress.

use bevy::prelude::*;
use bevy::reflect::TypeUuid;
use bevy::render::{camera::PerspectiveProjection, pipeline::PipelineDescriptor};
use bevy_egui::{egui, EguiContext, EguiPlugin, EguiSettings};
use no_cull_pipeline::PbrNoBackfaceBundle;
use polytope::geometry::Point;
use polytope::{off, shapes, ElementList, Polytope};

mod input;
mod no_cull_pipeline;
mod polytope;

fn main() {
    App::build()
        .add_resource(ClearColor(Color::rgb(0.0, 0.0, 0.0)))
        .add_resource(Msaa { samples: 4 })
        .add_plugins(DefaultPlugins)
        .add_plugin(EguiPlugin)
        .add_plugin(input::InputPlugin)
        .add_startup_system(setup.system())
        .add_startup_system(load_assets.system())
        .add_system(update_ui_scale_factor.system())
        .add_system(ui_example.system())
        .add_system_to_stage(stage::POST_UPDATE, update_changed_polytopes.system())
        .run();
}

const WIREFRAME_SELECTED_MATERIAL: HandleUntyped =
    HandleUntyped::weak_from_u64(StandardMaterial::TYPE_UUID, 0x82A3A5DD3A34CC21);
const WIREFRAME_UNSELECTED_MATERIAL: HandleUntyped =
    HandleUntyped::weak_from_u64(StandardMaterial::TYPE_UUID, 0x82A3A5DD3A34CC22);
const BEVY_TEXTURE_ID: u64 = 0;

#[derive(Default)]
struct UiState {
    label: String,
    value: f32,
    inverted: bool,
}

fn load_assets(_world: &mut World, resources: &mut Resources) {
    let mut egui_context = resources.get_mut::<EguiContext>().unwrap();
    let asset_server = resources.get::<AssetServer>().unwrap();

    let texture_handle = asset_server.load("icon.png");
    egui_context.set_egui_texture(BEVY_TEXTURE_ID, texture_handle);
}

fn update_ui_scale_factor(mut egui_settings: ResMut<EguiSettings>, windows: Res<Windows>) {
    if let Some(window) = windows.get_primary() {
        egui_settings.scale_factor = 1.0 / window.scale_factor();
    }
}

fn ui_example(mut egui_ctx: ResMut<EguiContext>, mut query: Query<&mut Polytope>) {
    let ctx = &mut egui_ctx.ctx;

    egui::TopPanel::top("top_panel").show(ctx, |ui| {
        // The top panel is often a good place for a menu bar:
        egui::menu::bar(ui, |ui| {
            egui::menu::menu(ui, "File", |ui| {
                if ui.button("Quit").clicked() {
                    std::process::exit(0);
                }
            });
        });

        if ui.button("Dual").clicked() {
            for mut p in query.iter_mut() {
                println!("Dual");
                *p = p.dual();
            }
        }

        if ui.button("Verf").clicked() {
            for mut p in query.iter_mut() {
                println!("Verf");
                *p = p.verf(0);
            }
        }
    });
}

fn setup(
    commands: &mut Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut shaders: ResMut<Assets<Shader>>,
    mut pipelines: ResMut<Assets<PipelineDescriptor>>,
) {
    let poly: Polytope = shapes::hypercube(3);
    println!("{}", off::to_src(&poly, Default::default()));

    pipelines.set_untracked(
        no_cull_pipeline::NO_CULL_PIPELINE_HANDLE,
        no_cull_pipeline::build_no_cull_pipeline(&mut shaders),
    );

    materials.set_untracked(
        WIREFRAME_SELECTED_MATERIAL,
        Color::rgb_u8(126, 192, 236).into(),
    );

    let wf_unselected = materials.set(
        WIREFRAME_UNSELECTED_MATERIAL,
        Color::rgb_u8(56, 68, 236).into(),
    );

    commands
        .spawn(PbrNoBackfaceBundle {
            mesh: meshes.add(poly.get_mesh()),
            visible: Visible {
                is_visible: false,
                ..Default::default()
            },
            material: materials.add(Color::rgb(0.93, 0.5, 0.93).into()),
            ..Default::default()
        })
        .with_children(|cb| {
            cb.spawn(PbrNoBackfaceBundle {
                mesh: meshes.add(poly.get_wireframe()),
                material: wf_unselected,
                ..Default::default()
            });
        })
        .with(poly)
        .spawn(LightBundle {
            transform: Transform::from_translation(Vec3::new(-2.0, 2.5, 2.0)),
            ..Default::default()
        })
        // camera anchor
        .spawn((
            GlobalTransform::default(),
            Transform::from_translation(Vec3::new(0.02, -0.025, -0.05))
            * Transform::from_translation(Vec3::new(-0.02, 0.025, 0.05))
            .looking_at(Vec3::default(), Vec3::unit_y()),
        ))
        .with_children(|cb| {
            // camera
            cb.spawn(Camera3dBundle {
                transform: Transform::from_translation(Vec3::new(0.0, 0.0, 5.0)),
                perspective_projection: PerspectiveProjection {
                    near: 0.0001,
                    ..Default::default()
                },
                ..Default::default()
            });
        });
}

fn update_changed_polytopes(
    mut meshes: ResMut<Assets<Mesh>>,
    polies: Query<(&Polytope, &Handle<Mesh>, &Children), Changed<Polytope>>,
    wfs: Query<(&Handle<Mesh>), Without<Polytope>>,
) {
    for (poly, mesh_handle, children) in polies.iter() {
        let mesh: &mut Mesh = meshes.get_mut(mesh_handle).unwrap();
        *mesh = poly.get_mesh();

        for child in children.iter() {
            if let Ok(wf_handle) = wfs.get_component::<Handle<Mesh>>(*child) {
                let wf: &mut Mesh = meshes.get_mut(wf_handle).unwrap();
                *wf = poly.get_wireframe();
                break;
            }
        }
    }
}
