//! MBTransient: a multiband transient shaper inspired by LHI Audio ST4b.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use nih_plug::prelude::*;
use nih_plug_egui::{create_egui_editor, egui, resizable_window::ResizableWindow, EguiState};

mod dsp;

use dsp::{
    Crossover, CrossoverState, SpectrumAnalyzer, SpectrumData, TransientShaper, MAX_BANDS,
    MAX_CROSSOVERS, SPECTRUM_BINS, SPECTRUM_FFT_SIZE,
};

/// Reference window size used for proportional scaling.
const REF_WIDTH: f32 = 440.0;
const REF_HEIGHT: f32 = 160.0;

/// How many dB the "cut" side of the shaping scheme removes (i.e. the "0%" of
/// the `att0sus100 - att100sus100 - att100sus0` sweep).
const MAX_CUT_DB: f32 = 45.0;

// Muted, not-too-bright, flat palette (no rounded corners anywhere).
const BG_COLOR: egui::Color32 = egui::Color32::from_rgb(201, 205, 212);
const DOT_COLOR: egui::Color32 = egui::Color32::from_rgb(190, 194, 202);
const PANEL_COLOR: egui::Color32 = egui::Color32::from_rgb(220, 224, 231);
const GRID_COLOR: egui::Color32 = egui::Color32::from_rgb(208, 212, 220);
const LINE_COLOR: egui::Color32 = egui::Color32::from_rgb(116, 122, 134);
const LINE_HOVER: egui::Color32 = egui::Color32::from_rgb(64, 68, 78);
const TEXT_COLOR: egui::Color32 = egui::Color32::from_rgb(46, 48, 54);
const TEXT_DIM: egui::Color32 = egui::Color32::from_rgb(118, 123, 132);
const TOOLTIP_BG: egui::Color32 = egui::Color32::from_rgb(58, 60, 66);
const TOOLTIP_TEXT: egui::Color32 = egui::Color32::from_rgb(230, 232, 236);

// Spectrum curve colour ramp (soft -> neutral -> punchy).
const SHAPE_LIGHT: egui::Color32 = egui::Color32::from_rgb(176, 181, 190);
const SHAPE_MID: egui::Color32 = egui::Color32::from_rgb(104, 110, 120);
const SHAPE_DARK: egui::Color32 = egui::Color32::from_rgb(34, 36, 40);

const DEFAULT_XOVERS: [f32; MAX_CROSSOVERS] = [150.0, 1500.0, 8000.0, 12000.0, 16000.0];
const DEFAULT_SPEED: f32 = 50.0;

pub struct MbTransient {
    params: Arc<MbTransientParams>,

    sample_rate: f32,
    crossover: Crossover,
    channel_states: Vec<ChannelState>,
    last_crossover_freqs: [f32; MAX_CROSSOVERS],
    last_num_bands: i32,

    spectrum: Arc<SpectrumData>,
    analyzer: SpectrumAnalyzer,
}

struct ChannelState {
    crossover: CrossoverState,
    shapers: [TransientShaper; MAX_BANDS],
}

impl Default for ChannelState {
    fn default() -> Self {
        Self {
            crossover: CrossoverState::default(),
            shapers: Default::default(),
        }
    }
}

#[derive(Params)]
pub struct MbTransientParams {
    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,

    #[id = "bands"]
    pub num_bands: IntParam,

    #[id = "xover1"]
    pub crossover_1: FloatParam,
    #[id = "xover2"]
    pub crossover_2: FloatParam,
    #[id = "xover3"]
    pub crossover_3: FloatParam,
    #[id = "xover4"]
    pub crossover_4: FloatParam,
    #[id = "xover5"]
    pub crossover_5: FloatParam,

    #[id = "b0"]
    pub band_0: FloatParam,
    #[id = "b1"]
    pub band_1: FloatParam,
    #[id = "b2"]
    pub band_2: FloatParam,
    #[id = "b3"]
    pub band_3: FloatParam,
    #[id = "b4"]
    pub band_4: FloatParam,
    #[id = "b5"]
    pub band_5: FloatParam,

    #[id = "speed"]
    pub speed: FloatParam,
    #[id = "level"]
    pub output: FloatParam,
}

impl MbTransientParams {
    pub fn crossover(&self, i: usize) -> &FloatParam {
        match i {
            0 => &self.crossover_1,
            1 => &self.crossover_2,
            2 => &self.crossover_3,
            3 => &self.crossover_4,
            4 => &self.crossover_5,
            _ => unreachable!(),
        }
    }

    pub fn band(&self, i: usize) -> &FloatParam {
        match i {
            0 => &self.band_0,
            1 => &self.band_1,
            2 => &self.band_2,
            3 => &self.band_3,
            4 => &self.band_4,
            5 => &self.band_5,
            _ => unreachable!(),
        }
    }
}

fn crossover_param(name: &str, default: f32) -> FloatParam {
    FloatParam::new(
        name,
        default,
        FloatRange::Skewed {
            min: 20.0,
            max: 20_000.0,
            factor: FloatRange::skew_factor(-1.0),
        },
    )
    .with_smoother(SmoothingStyle::Logarithmic(100.0))
    .with_value_to_string(formatters::v2s_f32_hz_then_khz(0))
    .with_string_to_value(formatters::s2v_f32_hz_then_khz())
}

fn shape_param(name: &str) -> FloatParam {
    FloatParam::new(name, 0.0, FloatRange::Linear { min: -1.0, max: 1.0 })
        .with_smoother(SmoothingStyle::Linear(5.0))
        .with_step_size(0.01)
}

fn speed_param(name: &str) -> FloatParam {
    FloatParam::new(
        name,
        DEFAULT_SPEED,
        FloatRange::Skewed {
            min: 2.0,
            max: 200.0,
            factor: FloatRange::skew_factor(-1.0),
        },
    )
    .with_smoother(SmoothingStyle::Linear(50.0))
    .with_unit("ms")
    .with_value_to_string(Arc::new(|value| format!("{value:.0}")))
    .with_string_to_value(Arc::new(|string| {
        let string = string.trim().trim_end_matches("ms").trim();
        string.parse::<f32>().ok()
    }))
}

fn level_param(name: &str) -> FloatParam {
    FloatParam::new(
        name,
        -6.0,
        FloatRange::Linear {
            min: -30.0,
            max: 6.0,
        },
    )
    .with_smoother(SmoothingStyle::Linear(10.0))
    .with_step_size(0.5)
    .with_unit("dB")
    .with_value_to_string(Arc::new(|value| format!("{value:+.1}")))
    .with_string_to_value(Arc::new(|string| {
        let string = string.trim().trim_end_matches("dB").trim();
        string.parse::<f32>().ok()
    }))
}

impl Default for MbTransientParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(440, 160),

            num_bands: IntParam::new("Bands", 4, IntRange::Linear { min: 2, max: 6 }),

            crossover_1: crossover_param("X-Over 1", DEFAULT_XOVERS[0]),
            crossover_2: crossover_param("X-Over 2", DEFAULT_XOVERS[1]),
            crossover_3: crossover_param("X-Over 3", DEFAULT_XOVERS[2]),
            crossover_4: crossover_param("X-Over 4", DEFAULT_XOVERS[3]),
            crossover_5: crossover_param("X-Over 5", DEFAULT_XOVERS[4]),

            band_0: shape_param("Band 1 Shape"),
            band_1: shape_param("Band 2 Shape"),
            band_2: shape_param("Band 3 Shape"),
            band_3: shape_param("Band 4 Shape"),
            band_4: shape_param("Band 5 Shape"),
            band_5: shape_param("Band 6 Shape"),

            speed: speed_param("Speed"),
            output: level_param("Level"),
        }
    }
}

impl Default for MbTransient {
    fn default() -> Self {
        Self {
            params: Arc::new(MbTransientParams::default()),

            sample_rate: 44_100.0,
            crossover: Crossover::new(44_100.0, &DEFAULT_XOVERS[..3]),
            channel_states: Vec::new(),
            last_crossover_freqs: DEFAULT_XOVERS,
            last_num_bands: 4,

            spectrum: Arc::new(SpectrumData::new()),
            analyzer: SpectrumAnalyzer::new(44_100.0),
        }
    }
}

impl Plugin for MbTransient {
    const NAME: &'static str = "mbt";
    const VENDOR: &'static str = "mbtransient";
    const URL: &'static str = "";
    const EMAIL: &'static str = "";

    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(1),
            ..AudioIOLayout::const_default()
        },
    ];

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        let params = self.params.clone();
        let spectrum = self.spectrum.clone();
        let egui_state = params.editor_state.clone();

        create_egui_editor(
            self.params.editor_state.clone(),
            GuiState::default(),
            |ctx, _state| {
                configure_style(ctx);
                install_font(ctx);
            },
            move |egui_ctx, setter, gui_state| {
                // Proportional scaling (manual, no egui zoom — avoids the
                // resize feedback loop that caused twitching).
                let screen = egui_ctx.screen_rect();
                let scale = (screen.width() / REF_WIDTH)
                    .min(screen.height() / REF_HEIGHT)
                    .clamp(0.8, 2.0);

                // Ease the displayed values toward the actual values so the
                // bars and splits animate smoothly instead of snapping. A fixed
                // per-frame factor guarantees visible easing at any frame rate.
                let num_bands_now = params.num_bands.value();
                if num_bands_now != gui_state.last_num_bands {
                    // When bands are added/removed, snap the bars and splits to
                    // their new positions (no ease) so they don't cross over.
                    gui_state.last_num_bands = num_bands_now;
                    for band in 0..MAX_BANDS {
                        gui_state.display_p[band] = params.band(band).value();
                    }
                    for i in 0..MAX_CROSSOVERS {
                        gui_state.display_xover[i] = params.crossover(i).value();
                    }
                }

                let ease = 0.2;
                for band in 0..MAX_BANDS {
                    let target = params.band(band).value();
                    gui_state.display_p[band] += (target - gui_state.display_p[band]) * ease;
                }
                for i in 0..MAX_CROSSOVERS {
                    let target = params.crossover(i).value();
                    gui_state.display_xover[i] +=
                        (target - gui_state.display_xover[i]) * ease;
                }

                let mut cursor = egui::CursorIcon::Default;

                ResizableWindow::new("mbtransient")
                    .min_size(egui::Vec2::new(400.0, 150.0))
                    .show(egui_ctx, egui_state.as_ref(), |ui| {
                        draw_dot_texture(ui, scale);

                        ui.add_space(6.0 * scale);
                        let margin = 10.0 * scale;
                        ui.horizontal(|ui| {
                            ui.add_space(margin);
                            ui.label(
                                egui::RichText::new("mbt")
                                    .size(13.0 * scale)
                                    .strong()
                                    .color(TEXT_COLOR),
                            );
                        });
                        ui.add_space(4.0 * scale);

                        // Spectrum with horizontal margins so it never touches
                        // the window edges.
                        let spectrum_h =
                            (ui.available_height() - 30.0 * scale).max(90.0);
                        ui.horizontal(|ui| {
                            ui.add_space(margin);
                            let width = ui.available_width() - margin;
                            if let Some(c) = draw_spectrum(
                                ui,
                                &spectrum,
                                &params,
                                setter,
                                width,
                                spectrum_h,
                                scale,
                                gui_state,
                            ) {
                                cursor = c;
                            }
                            ui.add_space(margin);
                        });

                        ui.add_space(4.0 * scale);

                        ui.horizontal(|ui| {
                            ui.add_space(margin);
                            if let Some(c) = text_param(ui, "speed", &params.speed, setter, scale) {
                                cursor = c;
                            }
                            ui.add_space(18.0 * scale);
                            if let Some(c) = text_param(ui, "level", &params.output, setter, scale) {
                                cursor = c;
                            }
                        });
                    });

                apply_cursor(egui_ctx, cursor);
            },
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        self.sample_rate = buffer_config.sample_rate;

        self.spectrum
            .sample_rate
            .store(buffer_config.sample_rate, Ordering::Relaxed);
        self.analyzer = SpectrumAnalyzer::new(buffer_config.sample_rate);

        let num_bands = self.params.num_bands.value() as usize;
        self.last_num_bands = num_bands as i32;
        self.last_crossover_freqs = DEFAULT_XOVERS;
        self.crossover.update(self.sample_rate, &DEFAULT_XOVERS[..num_bands - 1]);

        self.reset();

        true
    }

    fn reset(&mut self) {
        for state in &mut self.channel_states {
            state.crossover.reset();
            for shaper in &mut state.shapers {
                shaper.reset();
            }
        }

        self.analyzer.reset();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let num_channels = buffer.channels();
        if self.channel_states.len() != num_channels {
            self.channel_states
                .resize_with(num_channels, ChannelState::default);
            self.reset();
        }

        let num_bands = self.params.num_bands.value() as usize;
        if self.last_num_bands != num_bands as i32 {
            self.last_num_bands = num_bands as i32;
            self.reset();
        }
        self.update_shaper_coefficients();

        let sample_rate = self.sample_rate;

        let MbTransient {
            params,
            crossover,
            channel_states,
            last_crossover_freqs,
            spectrum,
            analyzer,
            ..
        } = self;

        let spectrum: &SpectrumData = spectrum;

        for mut channel_samples in buffer.iter_samples() {
            let mut freqs = [0.0f32; MAX_CROSSOVERS];
            for i in 0..(num_bands - 1) {
                freqs[i] = params.crossover(i).smoothed.next();
            }
            if freqs[..num_bands - 1] != last_crossover_freqs[..num_bands - 1] {
                crossover.update(sample_rate, &freqs[..num_bands - 1]);
                last_crossover_freqs[..num_bands - 1]
                    .copy_from_slice(&freqs[..num_bands - 1]);
            }

            let output_gain = util::db_to_gain(params.output.smoothed.next());

            let mut attack_gains = [0.0f32; MAX_BANDS];
            let mut sustain_gains = [0.0f32; MAX_BANDS];
            for band in 0..num_bands {
                let p = params.band(band).smoothed.next();
                let attack_db = if p < 0.0 { p * MAX_CUT_DB } else { 0.0 };
                let sustain_db = if p > 0.0 { -p * MAX_CUT_DB } else { 0.0 };
                attack_gains[band] = util::db_to_gain(attack_db);
                sustain_gains[band] = util::db_to_gain(sustain_db);
            }

            let mut mono_in = 0.0;
            let mut mono_out = 0.0;

            for ch in 0..num_channels {
                let sample = channel_samples.get_mut(ch).unwrap();
                let x = *sample;

                let bands = crossover.process(&mut channel_states[ch].crossover, x, num_bands);

                let mut wet = 0.0;
                for band in 0..num_bands {
                    wet += channel_states[ch].shapers[band].process(
                        bands[band],
                        attack_gains[band],
                        sustain_gains[band],
                    );
                }

                let out_pre = wet;

                let out = out_pre * output_gain;
                *sample = out;

                mono_in += x;
                mono_out += out;
            }

            if num_channels > 0 {
                let inv = 1.0 / num_channels as f32;
                analyzer.process(mono_in * inv, mono_out * inv, spectrum);
            }
        }

        ProcessStatus::Normal
    }
}

impl MbTransient {
    fn update_shaper_coefficients(&mut self) {
        let speed_ms = self.params.speed.value();
        for state in &mut self.channel_states {
            for shaper in &mut state.shapers {
                shaper.update(self.sample_rate, speed_ms);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GUI
// ---------------------------------------------------------------------------

/// GUI-only state: eased display values and sticky hover/cursor state.
struct GuiState {
    display_p: [f32; MAX_BANDS],
    display_xover: [f32; MAX_CROSSOVERS],
    hovered_split: Option<usize>,
    last_num_bands: i32,
}

impl Default for GuiState {
    fn default() -> Self {
        Self {
            display_p: [0.0; MAX_BANDS],
            display_xover: DEFAULT_XOVERS,
            hovered_split: None,
            last_num_bands: 4,
        }
    }
}

fn configure_style(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::light();
    visuals.panel_fill = BG_COLOR;
    visuals.window_fill = BG_COLOR;
    visuals.extreme_bg_color = BG_COLOR;
    visuals.faint_bg_color = PANEL_COLOR;
    visuals.override_text_color = Some(TEXT_COLOR);
    ctx.set_visuals(visuals);

    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(6.0, 4.0);
    ctx.set_style(style);
}

fn install_font(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "ubuntu_mono".to_owned(),
        Arc::new(egui::FontData::from_owned(
            include_bytes!("../assets/fonts/UbuntuMono-Regular.ttf").to_vec(),
        )),
    );
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .insert(0, "ubuntu_mono".to_owned());
    fonts
        .families
        .entry(egui::FontFamily::Monospace)
        .or_default()
        .insert(0, "ubuntu_mono".to_owned());
    ctx.set_fonts(fonts);
}

fn draw_dot_texture(ui: &egui::Ui, scale: f32) {
    let rect = ui.max_rect();
    let painter = ui.painter();
    let spacing = 16.0 * scale;
    let mut y = rect.top();
    while y < rect.bottom() {
        let mut x = rect.left();
        while x < rect.right() {
            painter.rect_filled(
                egui::Rect::from_center_size(
                    egui::pos2(x, y),
                    egui::vec2(1.5 * scale, 1.5 * scale),
                ),
                0.0,
                DOT_COLOR,
            );
            x += spacing;
        }
        y += spacing;
    }
}

fn lerp_color(a: egui::Color32, b: egui::Color32, t: f32) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    let mix = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    egui::Color32::from_rgb(mix(a.r(), b.r()), mix(a.g(), b.g()), mix(a.b(), b.b()))
}

fn shape_color(p: f32) -> egui::Color32 {
    let t = (p.clamp(-1.0, 1.0) + 1.0) * 0.5;
    if t < 0.5 {
        lerp_color(SHAPE_LIGHT, SHAPE_MID, t * 2.0)
    } else {
        lerp_color(SHAPE_MID, SHAPE_DARK, (t - 0.5) * 2.0)
    }
}

fn format_freq(freq: f32) -> String {
    if freq >= 1000.0 {
        format!("{:.0}k", freq / 1000.0)
    } else {
        format!("{freq:.0}")
    }
}

/// Apply a mouse cursor. On macOS, baseview doesn't support changing the cursor,
/// so we drive `NSCursor` directly on every frame; on other platforms we use
/// egui's own cursor output.
fn apply_cursor(ctx: &egui::Context, icon: egui::CursorIcon) {
    #[cfg(target_os = "macos")]
    {
        use objc::runtime::Object;
        use objc::{class, msg_send, sel, sel_impl};

        let cursor: *mut Object = unsafe {
            match icon {
                egui::CursorIcon::ResizeVertical => msg_send![class!(NSCursor), resizeUpDownCursor],
                egui::CursorIcon::ResizeHorizontal => {
                    msg_send![class!(NSCursor), resizeLeftRightCursor]
                }
                egui::CursorIcon::PointingHand => msg_send![class!(NSCursor), pointingHandCursor],
                _ => msg_send![class!(NSCursor), arrowCursor],
            }
        };
        let _: () = unsafe { msg_send![cursor, set] };
        let _ = ctx;
    }

    #[cfg(not(target_os = "macos"))]
    {
        ctx.set_cursor_icon(icon);
    }
}

fn text_param(
    ui: &mut egui::Ui,
    label: &str,
    param: &FloatParam,
    setter: &ParamSetter,
    scale: f32,
) -> Option<egui::CursorIcon> {
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(label)
                .size(11.0 * scale)
                .monospace()
                .color(TEXT_DIM),
        );
        ui.add_space(4.0 * scale);
        value_control(ui, param, setter, scale)
    })
    .inner
}

/// A numeric value rendered as text (no hover highlight). Supports mouse-wheel
/// and vertical drag to change the value, and double-click to reset. Returns
/// the cursor to show while hovered, if any.
fn value_control(
    ui: &mut egui::Ui,
    param: &FloatParam,
    setter: &ParamSetter,
    scale: f32,
) -> Option<egui::CursorIcon> {
    let text = param.to_string();
    let font = egui::FontId::monospace(11.0 * scale);
    let galley = ui.painter().layout_no_wrap(text, font, TEXT_COLOR);
    let size = galley.size() + egui::vec2(10.0 * scale, 4.0 * scale);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
    let painter = ui.painter_at(rect);
    painter.galley(rect.min + egui::vec2(5.0 * scale, 2.0 * scale), galley, TEXT_COLOR);

    if response.hovered() {
        let scroll = ui.input(|i| i.raw_scroll_delta.y);
        if scroll != 0.0 {
            let fine = ui.input(|i| i.modifiers.shift);
            let k = if fine { 0.0002 } else { 0.001 };
            let new_normalized =
                (param.modulated_normalized_value() + scroll * k).clamp(0.0, 1.0);
            setter.set_parameter(param, param.preview_plain(new_normalized));
        }
    }
    if response.drag_started() {
        setter.begin_set_parameter(param);
    }
    if response.dragged() {
        let fine = ui.input(|i| i.modifiers.shift);
        let k = if fine { 0.001 } else { 0.004 };
        let new_normalized =
            (param.modulated_normalized_value() - response.drag_delta().y * k).clamp(0.0, 1.0);
        setter.set_parameter(param, param.preview_plain(new_normalized));
    }
    if response.double_clicked() {
        setter.set_parameter(param, param.default_plain_value());
    }
    if response.drag_stopped() {
        setter.end_set_parameter(param);
    }

    if response.hovered() {
        Some(egui::CursorIcon::ResizeVertical)
    } else {
        None
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_spectrum(
    ui: &mut egui::Ui,
    spectrum: &SpectrumData,
    params: &MbTransientParams,
    setter: &ParamSetter,
    width: f32,
    height: f32,
    scale: f32,
    gui_state: &mut GuiState,
) -> Option<egui::CursorIcon> {
    const MIN_DB: f32 = -70.0;
    const MAX_DB: f32 = 0.0;

    let sample_rate = spectrum.sample_rate.load(Ordering::Relaxed).max(1.0);
    let min_freq = 20.0f32;
    let max_freq = sample_rate * 0.5;
    let num_bands = params.num_bands.value() as usize;

    let (rect, mut response) =
        ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::click_and_drag());
    let painter = ui.painter_at(rect);

    let freq_to_x = |f: f32| {
        rect.left() + (f / min_freq).ln() / (max_freq / min_freq).ln() * rect.width()
    };
    let db_to_y = |db: f32| {
        let t = ((db - MIN_DB) / (MAX_DB - MIN_DB)).clamp(0.0, 1.0);
        rect.top() + (1.0 - t) * rect.height()
    };
    let x_to_freq = |x: f32| {
        let t = ((x - rect.left()) / rect.width()).clamp(0.0, 1.0);
        min_freq * (max_freq / min_freq).powf(t)
    };
    // The nearest split, using the *eased* (displayed) split positions.
    let nearest_line = |x: f32, threshold: f32| -> Option<usize> {
        let mut best = None;
        let mut best_dist = threshold;
        for i in 0..(num_bands - 1) {
            let dist = (freq_to_x(gui_state.display_xover[i]) - x).abs();
            if dist < best_dist {
                best_dist = dist;
                best = Some(i);
            }
        }
        best
    };
    let band_at = |x: f32| -> usize {
        let f = x_to_freq(x);
        for i in 0..(num_bands - 1) {
            if f < params.crossover(i).value() {
                return i;
            }
        }
        num_bands - 1
    };

    // --- interaction ---
    let grab_id = egui::Id::new("mbtransient_xover_grab");
    let shape_grab_id = egui::Id::new("mbtransient_shape_grab");
    let drag_start_p_id = egui::Id::new("mbtransient_shape_drag_start_p");
    let drag_start_y_id = egui::Id::new("mbtransient_shape_drag_start_y");

    // Sticky hover: once a split is highlighted, its hit region stays enlarged
    // so dragging it feels responsive.
    let hovered_split = if let Some(p) = response.hover_pos() {
        if let Some(line) = nearest_line(p.x, 10.0 * scale) {
            Some(line)
        } else if let Some(prev) = gui_state.hovered_split {
            if prev < num_bands - 1
                && (freq_to_x(gui_state.display_xover[prev]) - p.x).abs() < 22.0 * scale
            {
                Some(prev)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };
    gui_state.hovered_split = hovered_split;

    if response.drag_started() {
        if let Some(pos) = response.interact_pointer_pos() {
            if let Some(line) = hovered_split {
                ui.memory_mut(|m| m.data.insert_temp(grab_id, line as i32));
                setter.begin_set_parameter(params.crossover(line));
            } else {
                let band = band_at(pos.x);
                ui.memory_mut(|m| m.data.insert_temp(shape_grab_id, band as i32));
                ui.memory_mut(|m| m.data.insert_temp(drag_start_p_id, params.band(band).value()));
                ui.memory_mut(|m| m.data.insert_temp(drag_start_y_id, pos.y));
                setter.begin_set_parameter(params.band(band));
            }
        }
    }
    if response.dragged() {
        if let Some(pos) = response.interact_pointer_pos() {
            if let Some(line) = ui.memory(|m| m.data.get_temp::<i32>(grab_id)) {
                if line >= 0 && (line as usize) < num_bands - 1 {
                    let freq = x_to_freq(pos.x).clamp(min_freq, max_freq);
                    setter.set_parameter(params.crossover(line as usize), freq);
                    response.mark_changed();
                }
            } else if let Some(band) = ui.memory(|m| m.data.get_temp::<i32>(shape_grab_id)) {
                if band >= 0 && (band as usize) < num_bands {
                    let start_p = ui
                        .memory(|m| m.data.get_temp::<f32>(drag_start_p_id))
                        .unwrap_or(params.band(band as usize).value());
                    let start_y = ui
                        .memory(|m| m.data.get_temp::<f32>(drag_start_y_id))
                        .unwrap_or(pos.y);
                    // Relative drag: full-height drag sweeps the whole range.
                    let p = (start_p + (start_y - pos.y) * (2.0 / rect.height()))
                        .clamp(-1.0, 1.0);
                    setter.set_parameter(params.band(band as usize), p);
                    response.mark_changed();
                }
            }
        }
    }
    if response.hovered() {
        let scroll = ui.input(|i| i.raw_scroll_delta.y);
        if scroll != 0.0 {
            if let Some(pos) = response.hover_pos() {
                if hovered_split.is_none() {
                    let band = band_at(pos.x);
                    let fine = ui.input(|i| i.modifiers.shift);
                    let k = if fine { 0.003 } else { 0.015 };
                    let p = params.band(band).value();
                    let new_p = (p + scroll * k).clamp(-1.0, 1.0);
                    setter.set_parameter(params.band(band), new_p);
                }
            }
        }
    }
    // Right-click deletes the band under the cursor.
    if response.secondary_clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            delete_band(setter, params, x_to_freq(pos.x));
        }
    }
    // Double-click on the bar resets it to neutral; elsewhere it adds a band.
    if response.double_clicked() {
        let click_pos = response
            .hover_pos()
            .or_else(|| response.interact_pointer_pos());
        if let Some(pos) = click_pos {
            let band = band_at(pos.x);
            let bar_y = rect.top() + (0.5 - gui_state.display_p[band] * 0.5) * rect.height();
            if (pos.y - bar_y).abs() < 14.0 * scale {
                setter.set_parameter(params.band(band), 0.0);
            } else if hovered_split.is_none() {
                add_band(setter, params, x_to_freq(pos.x));
            }
        }
    }
    if response.drag_stopped() {
        if let Some(line) = ui.memory(|m| m.data.get_temp::<i32>(grab_id)) {
            if line >= 0 && (line as usize) < num_bands - 1 {
                setter.end_set_parameter(params.crossover(line as usize));
            }
        }
        if let Some(band) = ui.memory(|m| m.data.get_temp::<i32>(shape_grab_id)) {
            if band >= 0 && (band as usize) < num_bands {
                setter.end_set_parameter(params.band(band as usize));
            }
        }
        ui.memory_mut(|m| {
            m.data.remove::<i32>(grab_id);
            m.data.remove::<i32>(shape_grab_id);
        });
    }

    // --- drawing (sharp corners) ---
    painter.rect_filled(rect, 0.0, PANEL_COLOR);
    painter.rect_stroke(
        rect,
        0.0,
        egui::Stroke::new(1.0, GRID_COLOR),
        egui::StrokeKind::Inside,
    );

    // dB grid.
    for db in [-60.0, -40.0, -20.0] {
        let y = db_to_y(db);
        painter.line_segment(
            [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
            egui::Stroke::new(1.0, GRID_COLOR),
        );
        painter.text(
            egui::pos2(rect.left() + 4.0 * scale, y - 7.0 * scale),
            egui::Align2::LEFT_BOTTOM,
            format!("{db:.0}"),
            egui::FontId::monospace(8.0 * scale),
            TEXT_DIM,
        );
    }

    // Frequency grid.
    for freq in [100.0, 1000.0, 10000.0] {
        if freq <= min_freq || freq >= max_freq {
            continue;
        }
        let x = freq_to_x(freq);
        painter.line_segment(
            [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
            egui::Stroke::new(1.0, GRID_COLOR),
        );
        painter.text(
            egui::pos2(x, rect.bottom() - 2.0 * scale),
            egui::Align2::CENTER_BOTTOM,
            format_freq(freq),
            egui::FontId::monospace(8.0 * scale),
            TEXT_DIM,
        );
    }

    // Spectrum curve + gradient fill (per band).
    let num_cols = (rect.width() / 3.0).max(2.0) as usize;
    let mut xs = Vec::with_capacity(num_cols);
    let mut ys = Vec::with_capacity(num_cols);
    for j in 0..num_cols {
        let t = j as f32 / (num_cols - 1) as f32;
        let freq = min_freq * (max_freq / min_freq).powf(t);
        let bin_f = freq * SPECTRUM_FFT_SIZE as f32 / sample_rate;
        let bin0 = (bin_f.floor() as usize).min(SPECTRUM_BINS - 1);
        let bin1 = (bin0 + 1).min(SPECTRUM_BINS - 1);
        let frac = bin_f - bin0 as f32;
        let db = spectrum.output[bin0].load(Ordering::Relaxed) * (1.0 - frac)
            + spectrum.output[bin1].load(Ordering::Relaxed) * frac;
        xs.push(freq_to_x(freq));
        ys.push(db_to_y(db.clamp(MIN_DB, MAX_DB)));
    }

    let mut crossover_x = [0.0f32; MAX_CROSSOVERS];
    for i in 0..(num_bands - 1) {
        crossover_x[i] = freq_to_x(gui_state.display_xover[i]);
    }

    for band in 0..num_bands {
        let x0 = if band == 0 {
            rect.left()
        } else {
            crossover_x[band - 1]
        };
        let x1 = if band == num_bands - 1 {
            rect.right()
        } else {
            crossover_x[band]
        };

        // Use the eased value so the whole band (curve, fill and bar) animates
        // smoothly toward the target instead of snapping.
        let color = shape_color(gui_state.display_p[band]);
        let c_top = color.gamma_multiply(0.35);
        let c_bottom = color.gamma_multiply(0.0);

        let mut curve: Vec<egui::Pos2> = Vec::new();
        let mut mesh = egui::Mesh::default();
        let mut prev: Option<(f32, f32)> = None;

        for j in 0..num_cols {
            let x = xs[j];
            if x < x0 - 0.5 || x > x1 + 0.5 {
                continue;
            }
            let y = ys[j];
            curve.push(egui::pos2(x, y));

            if let Some((px, py)) = prev {
                let i = mesh.vertices.len() as u32;
                mesh.colored_vertex(egui::pos2(px, py), c_top);
                mesh.colored_vertex(egui::pos2(px, rect.bottom()), c_bottom);
                mesh.colored_vertex(egui::pos2(x, y), c_top);
                mesh.colored_vertex(egui::pos2(x, rect.bottom()), c_bottom);
                mesh.add_triangle(i, i + 1, i + 2);
                mesh.add_triangle(i + 2, i + 1, i + 3);
            }
            prev = Some((x, y));
        }

        if mesh.vertices.len() >= 4 {
            painter.add(egui::Shape::mesh(mesh));
        }
        if curve.len() >= 2 {
            painter.line(curve, egui::Stroke::new(1.5, color));
        }

        // Horizontal bar indicating the shaping value for this band (eased).
        // Hovering near the bar highlights it and expands its hit region.
        let p = gui_state.display_p[band];
        let bar_y = rect.top() + (0.5 - p * 0.5) * rect.height();
        let bar_grab_half = 14.0 * scale;
        let bar_hovered = response
            .hover_pos()
            .map(|hp| {
                hp.x >= x0 && hp.x <= x1
                    && (hp.y - bar_y).abs() < bar_grab_half
                    && hovered_split.is_none()
            })
            .unwrap_or(false);

        if bar_hovered {
            // Faint strip showing the enlarged hit region.
            painter.rect_filled(
                egui::Rect::from_min_max(
                    egui::pos2(x0, bar_y - bar_grab_half),
                    egui::pos2(x1, bar_y + bar_grab_half),
                ),
                0.0,
                LINE_HOVER.gamma_multiply(0.12),
            );
        }
        let bar_color = if bar_hovered { LINE_HOVER } else { TEXT_COLOR };
        let bar_w = if bar_hovered { 4.0 } else { 2.0 };
        painter.line_segment(
            [
                egui::pos2(x0 + 3.0 * scale, bar_y),
                egui::pos2(x1 - 3.0 * scale, bar_y),
            ],
            egui::Stroke::new(bar_w, bar_color),
        );
    }

    // Crossover lines + handles (highlight on hover). Drawn at the eased
    // positions so the splits animate smoothly.
    for i in 0..(num_bands - 1) {
        let freq = gui_state.display_xover[i];
        if freq <= min_freq || freq >= max_freq {
            continue;
        }
        let x = freq_to_x(freq);
        let hovered = hovered_split == Some(i);
        let color = if hovered { LINE_HOVER } else { LINE_COLOR };
        let stroke_w = if hovered { 2.0 } else { 1.0 };
        painter.line_segment(
            [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
            egui::Stroke::new(stroke_w, color),
        );
        painter.rect_filled(
            egui::Rect::from_center_size(
                egui::pos2(x, rect.top() + 7.0 * scale),
                egui::vec2((if hovered { 14.0 } else { 12.0 }) * scale, 7.0 * scale),
            ),
            0.0,
            color,
        );
    }

    // Hover tooltip showing the shaping readout for the band under the cursor.
    if response.hovered() {
        if let Some(pos) = response.hover_pos() {
            if hovered_split.is_none() {
                let band = band_at(pos.x);
                let p = params.band(band).value();
                let attack_db = if p < 0.0 { p * MAX_CUT_DB } else { 0.0 };
                let sustain_db = if p > 0.0 { -p * MAX_CUT_DB } else { 0.0 };
                let text = format!("att {:+.0}dB  sus {:+.0}dB", attack_db, sustain_db);

                let galley = painter.layout_no_wrap(
                    text,
                    egui::FontId::monospace(10.0 * scale),
                    TOOLTIP_TEXT,
                );
                let tooltip_rect = egui::Rect::from_min_size(
                    pos + egui::vec2(12.0 * scale, 10.0 * scale),
                    galley.size() + egui::vec2(8.0 * scale, 4.0 * scale),
                );
                painter.rect_filled(tooltip_rect, 0.0, TOOLTIP_BG);
                painter.galley(
                    tooltip_rect.min + egui::vec2(4.0 * scale, 2.0 * scale),
                    galley,
                    TOOLTIP_TEXT,
                );
            }
        }
    }

    // The cursor to show while hovered (applied by the caller).
    if response.hovered() {
        if hovered_split.is_some() {
            Some(egui::CursorIcon::ResizeHorizontal)
        } else {
            Some(egui::CursorIcon::ResizeVertical)
        }
    } else {
        None
    }
}

/// Split the band containing `freq` into two bands by inserting a crossover.
fn add_band(setter: &ParamSetter, params: &MbTransientParams, freq: f32) {
    let n = params.num_bands.value() as usize;
    if n >= MAX_BANDS {
        return;
    }

    let k = band_index(params, freq, n);
    let old_shape_k = params.band(k).value();

    // Shift crossovers at/after `k` one slot to the right.
    for j in (0..(n - 1)).rev() {
        if j >= k {
            setter.set_parameter(params.crossover(j + 1), params.crossover(j).value());
        }
    }
    setter.set_parameter(params.crossover(k), freq);

    // Shift band shapes after `k` right, and copy band `k` into the new band.
    for j in (0..n).rev() {
        if j >= k + 1 {
            setter.set_parameter(params.band(j + 1), params.band(j).value());
        }
    }
    setter.set_parameter(params.band(k + 1), old_shape_k);

    setter.set_parameter(&params.num_bands, (n + 1) as i32);
}

/// Merge the band containing `freq` with its neighbour by removing a crossover.
fn delete_band(setter: &ParamSetter, params: &MbTransientParams, freq: f32) {
    let n = params.num_bands.value() as usize;
    if n <= 2 {
        return;
    }

    let k = band_index(params, freq, n);
    let remove = k.min(n - 2);

    for j in remove..(n - 2) {
        setter.set_parameter(params.crossover(j), params.crossover(j + 1).value());
    }
    for j in (remove + 1)..(n - 1) {
        setter.set_parameter(params.band(j), params.band(j + 1).value());
    }

    setter.set_parameter(&params.num_bands, (n - 1) as i32);
}

fn band_index(params: &MbTransientParams, freq: f32, num_bands: usize) -> usize {
    for i in 0..(num_bands - 1) {
        if freq < params.crossover(i).value() {
            return i;
        }
    }
    num_bands - 1
}

impl ClapPlugin for MbTransient {
    const CLAP_ID: &'static str = "com.mbtransient.mbtransient";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("A multiband transient shaper inspired by LHI Audio ST4b");
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::TransientShaper,
        ClapFeature::Stereo,
        ClapFeature::Mono,
        ClapFeature::Utility,
    ];
}

impl Vst3Plugin for MbTransient {
    const VST3_CLASS_ID: [u8; 16] = *b"MBTransientShape";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Dynamics];
}

nih_export_clap!(MbTransient);
nih_export_vst3!(MbTransient);
