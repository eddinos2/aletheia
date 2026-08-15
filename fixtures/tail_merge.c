/* tail_merge: two textually identical tails at different nesting
 * depths — `tm_sink = tm_sink * 3; return tm_sink;` appears once
 * inside the nest and once at top level — which is exactly the shape
 * the compiler's cross-jumping merges into one block with predecessors
 * at incompatible nesting (the SAILR tail-merge case). The volatile
 * stores keep each head distinct.
 *
 * Source CFG (ground truth, recorded in src/evalfx.rs):
 *
 *        0  x > 0
 *       / \
 *      1   4   y > 0 / top-level tail (y - x, * 3, return)
 *     / \
 *    2   3     nested tail (x + y, * 3, return) / (x - y, return)
 *
 * nodes 0..4, edges 0->1 0->4 1->2 1->3
 */

volatile int tm_sink;

__attribute__((noinline)) int tail_merge(int x, int y) {
    if (x > 0) {
        if (y > 0) {
            tm_sink = x + y;
            tm_sink = tm_sink * 3;
            return tm_sink;
        }
        tm_sink = x - y;
        return tm_sink;
    }
    tm_sink = y - x;
    tm_sink = tm_sink * 3;
    return tm_sink;
}

int main(int argc, char **argv) {
    (void)argv;
    return tail_merge(argc, argc - 2);
}
