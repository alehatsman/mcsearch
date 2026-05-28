import { Handler, makeHandler } from "./handler";
import * as text from "./text";
import { noop } from "./utils";

export function helper() {
  return 1;
}

export function main() {
  const h = makeHandler("world");
  const msg = text.upper("hi");
  new Handler(msg);
  helper();
  noop();
}
