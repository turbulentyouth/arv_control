slint::include_modules!();

fn main() -> Result<(), slint::PlatformError> {
    let ui = AppWindow::new()?;

    let weak_ui = ui.as_weak();
    slint::invoke_from_event_loop(move || {
        let ui = weak_ui.unwrap();
        ui.window().set_maximized(true);
    }).unwrap();

    ui.run()
}
