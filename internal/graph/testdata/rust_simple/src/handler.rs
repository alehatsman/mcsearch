pub trait Greeter {
    fn greet(&self) -> String;
}

pub struct Handler {
    pub name: String,
}

impl Handler {
    pub fn new(name: String) -> Self {
        Handler { name }
    }

    pub fn format(&self, x: &str) -> String {
        format!("Hello, {}", x)
    }
}

impl Greeter for Handler {
    fn greet(&self) -> String {
        self.format(&self.name)
    }
}

pub fn make_handler(name: String) -> Handler {
    Handler::new(name)
}
