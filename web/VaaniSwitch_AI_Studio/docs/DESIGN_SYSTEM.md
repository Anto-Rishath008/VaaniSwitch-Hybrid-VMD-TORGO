# Design system

Midnight navy application canvas, charcoal instrument panels, mint/cyan primary actions, subdued violet for model comparisons, and amber for limitations. The locally bundled Inter variable typeface avoids external font requests. `frontend/src/theme.css` defines shared colors; `styles.css` contains layout/component styles; `extras.css` includes scientific views and observed accessibility repairs.

Persistent desktop navigation becomes a keyboard-accessible mobile drawer. Layouts adapt at 1250, 1000, 760 and 540 px. Input, signal workspace and model output form three zones on large screens and stack on phones. The clear-speech route removes advanced classifier controls.

Motion uses route fades, staggered card entrances, animated chart widths, schematic signal paths and real microphone amplitude. MotionConfig and CSS respect reduced-motion preferences. Recording canvas draws actual analyser samples. Uploaded/playback waveforms and STFTs derive from decoded samples, not decorative matrices. Architecture animations are labeled schematics.

Every control performs navigation, processing, state changes, playback, download, disclosure or selection. Disabled controls explain missing inputs or scientific availability. Keyboard focus, readable contrast, semantic controls, error alerts and recording status are verified by automated browser checks; this does not replace a human assistive-technology audit.
