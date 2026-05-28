export const noop = () => {};

export function id<T>(x: T): T {
  return x;
}
