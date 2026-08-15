/* loop_bc: a counted loop carrying both a continue and a break. The
 * volatile global keeps the body's loads and stores observable so no
 * arm folds away.
 *
 * Source CFG (ground truth, recorded in src/evalfx.rs):
 *
 *        0  entry (acc = 0, i = 0)
 *        |
 *        1  i < n ----------------+
 *        |                        |
 *        2  i == loop_sink --> 5  |   continue: straight to increment
 *        |                     ^  |
 *        3  acc > 100 ------------+   break: to the exit
 *        |                     |  |
 *        4  acc += i ----------+  |
 *                                 |
 *        6  return acc <----------+
 *
 * nodes 0..6, edges 0->1 1->2 1->6 2->5 2->3 3->6 3->4 4->5 5->1
 */

volatile int loop_sink;

__attribute__((noinline)) int loop_bc(int n) {
    int acc = 0;
    for (int i = 0; i < n; i++) {
        if (i == loop_sink)
            continue;
        if (acc > 100)
            break;
        acc += i;
        loop_sink = acc;
    }
    return acc;
}

int main(int argc, char **argv) {
    (void)argv;
    return loop_bc(argc);
}
