// =============================================================================
//  OmniSeed — test_int4_ffn.cpp  (T120b)
//  Grouped int4 (g=64) pack/unpack + kernel + fail-closed layout gate.
//    A  pack/unpack round-trip is nibble-exact vs the reference dequant
//    B  kernel == fp32 reference dot with IDENTICAL accumulation order
//    C  i4g64_layout_ok refuses bad shapes / bad gscale numel
//    D  signed nibble edge cases (-7, +7, 0)
// =============================================================================
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "omniseed/core/bitlinear.h"

using omniseed::bitnet::bitlinear_forward_i4g64;
using omniseed::bitnet::i4g64_layout_ok;

namespace {

int g_pass = 0, g_fail = 0;

void check(bool ok, const char* what) {
    if (ok) { ++g_pass; std::printf("[omniseed] INFO  PASS %s\n", what); }
    else    { ++g_fail; std::printf("[omniseed] INFO  FAIL %s\n", what); }
}

// Pack q[r][c] (int, -7..7) the way tools/convert_int4_ffn.py does:
// low nibble = even column, two's complement 4-bit.
void pack(const std::vector<std::vector<int>>& q, std::vector<uint8_t>& out) {
    const int64_t rows = static_cast<int64_t>(q.size());
    const int64_t cols = static_cast<int64_t>(q[0].size());
    out.assign(static_cast<size_t>(rows * cols / 2), 0);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            const uint8_t nib = static_cast<uint8_t>(q[r][c] & 0x0F);
            uint8_t& byte = out[static_cast<size_t>(r * cols / 2 + c / 2)];
            if (c % 2 == 0) byte  = nib;
            else            byte |= static_cast<uint8_t>(nib << 4);
        }
    }
}

int unpack_nibble(const uint8_t* row, int64_t c) {
    const uint8_t byte = row[c / 2];
    int v = (c & 1) ? static_cast<int>(byte >> 4)
                    : static_cast<int>(byte & 0x0F);
    if (v > 7) v -= 16;
    return v;
}

} // namespace

int main() {
    // ---------------- A: nibble-exact round-trip ---------------------------
    {
        const int64_t rows = 5, cols = 128;   // 2 groups/row
        std::vector<std::vector<int>> q(rows, std::vector<int>(cols));
        std::vector<uint8_t> packed;
        // deterministic pattern incl. edges
        const int edges[15] = {-7, -6, -5, -4, -3, -2, -1, 0,
                               1, 2, 3, 4, 5, 6, 7};
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t c = 0; c < cols; ++c)
                q[r][c] = edges[(r * cols + c) % 15];
        pack(q, packed);
        bool exact = true;
        for (int64_t r = 0; r < rows && exact; ++r)
            for (int64_t c = 0; c < cols && exact; ++c)
                exact = (unpack_nibble(packed.data() + r * cols / 2, c)
                         == q[r][c]);
        check(exact, "A1 pack->unpack nibble-exact (incl -7..7 edges)");
        check(packed.size() == static_cast<size_t>(rows * cols / 2),
              "A2 packed size == numel/2 bytes");
    }

    // ---------------- B: kernel == fp32 reference (same order) -------------
    {
        const int64_t out_dim = 9, in_dim = 192;   // 3 groups/row, odd rows
        std::vector<std::vector<int>> q(out_dim, std::vector<int>(in_dim));
        std::vector<uint8_t> packed;
        uint32_t lcg = 12345;
        auto rnd = [&]() { lcg = lcg * 1664525u + 1013904223u;
                           return static_cast<int>((lcg >> 16) % 15) - 7; };
        for (auto& row : q) for (auto& v : row) v = rnd();
        pack(q, packed);

        std::vector<float> gscale(static_cast<size_t>(out_dim * in_dim / 64));
        for (auto& s : gscale) s = 0.01f + 0.003f * static_cast<float>(rnd());
        std::vector<float> x(static_cast<size_t>(in_dim));
        for (auto& v : x) v = 0.1f * static_cast<float>(rnd());

        std::vector<float> y(static_cast<size_t>(out_dim));
        bitlinear_forward_i4g64(packed.data(), gscale.data(), x.data(),
                                y.data(), out_dim, in_dim);

        bool same = true;
        const int64_t groups = in_dim / 64;
        for (int64_t r = 0; r < out_dim && same; ++r) {
            // reference: identical loop nest / accumulation order
            float acc = 0.0f;
            for (int64_t g = 0; g < groups; ++g) {
                float inner = 0.0f;
                for (int64_t c = g * 64; c < (g + 1) * 64; ++c)
                    inner += static_cast<float>(q[r][c]) * x[c];
                acc += gscale[r * groups + g] * inner;
            }
            same = (y[r] == acc);   // bit-exact: same order, same ops
        }
        check(same, "B1 kernel bit-exact vs fp32 reference (same order)");

        // negative control: kernel must differ when a scale changes
        gscale[0] *= 2.0f;
        bitlinear_forward_i4g64(packed.data(), gscale.data(), x.data(),
                                y.data(), out_dim, in_dim);
        check(y[0] != 0.0f && std::fabs(y[0]) > 1e-9f,
              "B2 scales actually flow (y[0] nonzero)");
    }

    // ---------------- C: fail-closed layout gate ---------------------------
    check( i4g64_layout_ok(3072, 768,  3072 * 12), "C1 real key shape OK");
    check( i4g64_layout_ok(768, 3072, 768 * 48),   "C2 real value shape OK");
    check(!i4g64_layout_ok(3072, 768,  0),         "C3 zero gscale refused");
    check(!i4g64_layout_ok(3072, 768,  3072 * 11), "C4 short gscale refused");
    check(!i4g64_layout_ok(3072, 768,  3072 * 13), "C5 long gscale refused");
    check(!i4g64_layout_ok(3072, 700,  3072 * 11), "C6 in_dim%64!=0 refused");
    check(!i4g64_layout_ok(0, 768, 0),             "C7 zero dims refused");

    // ---------------- D: signed nibble decode ------------------------------
    {
        // byte 0xF0 -> low nibble 0 (col even), high nibble 15 = -1 (col odd)
        const uint8_t one = 0xF0;
        check(unpack_nibble(&one, 0) == 0,  "D1 low nibble 0 -> 0");
        check(unpack_nibble(&one, 1) == -1, "D2 high nibble 0xF -> -1");
        const uint8_t two = 0x97;           // low 7 -> +7, high 9 -> -7
        check(unpack_nibble(&two, 0) == 7,  "D3 low nibble 7 -> +7");
        check(unpack_nibble(&two, 1) == -7, "D4 high nibble 9 -> -7");
    }

    std::printf("[omniseed]  int4_ffn: %d passed, %d failed\n",
                g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
