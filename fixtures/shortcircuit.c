/* shortcircuit: an `&&`/`||` chain — three conditions deciding one
 * two-armed conditional, each condition its own source basic block.
 *
 * Source CFG (ground truth, recorded in src/evalfx.rs):
 *
 *        0  a > 0
 *       / \
 *      1   |   b > a
 *     / \  |
 *    |   \ |
 *    |    2    c == 5
 *    |   / \
 *      3    4   then / else
 *       \  /
 *        5  return
 *
 * nodes 0..5, edges 0->1 0->2 1->3 1->2 2->3 2->4 3->5 4->5
 */

volatile int sc_sink;

__attribute__((noinline)) int shortcircuit(int a, int b, int c) {
    if ((a > 0 && b > a) || c == 5)
        sc_sink = a + b;
    else
        sc_sink = c;
    return sc_sink;
}

int main(int argc, char **argv) {
    (void)argv;
    return shortcircuit(argc, argc + 1, argc + 2);
}
