/* switch_dense: a dense 6-case switch plus default, every case a
 * different operation on a second unknown argument so the compiler
 * cannot collapse the dispatch into a value lookup table — a jump
 * table must be emitted.
 *
 * Source CFG (ground truth, recorded in src/evalfx.rs):
 *
 *              0  switch (x)
 *      / / / / | \ \
 *     1 2 3 4  5  6 7   six cases + default
 *      \ \ \ \ | / /
 *              8  return
 *
 * nodes 0..8, edges 0->k and k->8 for k in 1..=7.
 * At nine nodes this graph exceeds the exact-GED bound on purpose:
 * the fixture is the harness's documented CFGED refusal.
 *
 * The cases start at 1, deliberately: rebasing the scrutinee makes the
 * compiler fold the adjustment into the index register itself
 * (`decl %edi; cmpl $5, %edi; ja; movslq (%rax,%rdi,4), ...`), so the
 * bounds check and the table index are one register — the exact
 * recognized idiom. Zero-based cases index through a bare `movl`
 * zero-extending copy instead, which the recognizer's honest
 * backward def-walk refuses.
 */

volatile int switch_sink;

__attribute__((noinline)) int switch_dense(int x, int y) {
    switch (x) {
    case 1: switch_sink = y + 10; break;
    case 2: switch_sink = y * 3; break;
    case 3: switch_sink = y - 7; break;
    case 4: switch_sink = y ^ 21; break;
    case 5: switch_sink = y << 2; break;
    case 6: switch_sink = -y; break;
    default: switch_sink = y; break;
    }
    return switch_sink;
}

int main(int argc, char **argv) {
    (void)argv;
    return switch_dense(argc, argc + 3);
}
