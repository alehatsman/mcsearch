export class Handler {
  name: string;

  constructor(name: string) {
    this.name = name;
  }

  greet(): string {
    return this.format(this.name);
  }

  format(x: string): string {
    return "Hello, " + x;
  }
}

export function makeHandler(name: string): Handler {
  return new Handler(name);
}
