int add(int x, int y) { return x + y; }
int main(void) {
  /* intentional trivial change for patch-diff benches */
  return add(2, 4);
}
