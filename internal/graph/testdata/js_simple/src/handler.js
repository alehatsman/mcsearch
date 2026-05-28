export class Handler {
  constructor(name) {
    this.name = name;
  }

  greet() {
    return this.format(this.name);
  }

  format(x) {
    return "Hello, " + x;
  }
}

export function makeHandler(name) {
  return new Handler(name);
}
