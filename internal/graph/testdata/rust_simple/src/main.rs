use crate::handler::{Handler, make_handler};
use crate::text::upper;

mod handler;
mod text;

fn helper() -> i32 {
    1
}

fn main() {
    let h = make_handler(String::from("world"));
    let msg = upper("hi");
    let _ = Handler::new(msg);
    helper();
}
