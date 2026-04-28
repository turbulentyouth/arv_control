pub struct InputState {
    pub w: bool,
    pub s: bool,
    pub a: bool,
    pub d: bool,
    pub i: bool,
    pub k: bool,
    pub j: bool,
    pub l: bool,
    pub gripper_open: bool,
    pub gripper_close: bool,
    pub power_percent: u8,
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            w: false,
            s: false,
            a: false,
            d: false,
            i: false,
            k: false,
            j: false,
            l: false,
            gripper_open: false,
            gripper_close: false,
            power_percent: 20,
        }
    }
}
