/* diamond: one two-way conditional and its join.
 *
 * The two arms store to two *different* volatile globals: with a single
 * sink the compiler is entitled to if-convert (select the value, store
 * once — a cmov, no diamond), and at -O1 it does exactly that. Distinct
 * addresses make the arms differ in effect, so both branches survive.
 *
 * Source CFG (ground truth, recorded in src/evalfx.rs):
 *
 *        0  if (x > 0)
 *       / \
 *      1   2   then / else
 *       \ /
 *        3  return
 *
 * nodes 0..3, edges 0->1 0->2 1->3 2->3
 */

volatile int diamond_pos;
volatile int diamond_neg;

__attribute__((noinline)) int diamond(int x) {
    if (x > 0)
        diamond_pos = x * 3;
    else
        diamond_neg = 5 - x;
    return diamond_pos + diamond_neg;
}

int main(int argc, char **argv) {
    (void)argv;
    return diamond(argc);
}
