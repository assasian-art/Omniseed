#!/usr/bin/env python3
"""OmniSeed — tests/test_lora_bilingual.py (T082).

Bilingual LoRA corpus honesty, OFFLINE (tokenizer only, no training):
  1. the bilingual fixture (English + ROMANISED Bengali) loads and roundtrips
     through the shipped world vocab -> trainable;
  2. native-Bengali text containing an UNCOVERED char (U+0986/U+09AD measured)
     fails roundtrip -> `refuse_uncovered` must flag it (the guard that stops
     the trainer from teaching U+FFFD corruption);
  3. every covered Bengali char found in the vocab still roundtrips.
Usage: python tests/test_lora_bilingual.py
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PASSED, FAILED = 0, 0
CURRENT = ""


def check(name, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
    else:
        FAILED += 1
        print("  FAIL %s [%s] %s" % (CURRENT, name, detail))


def test(name):
    global CURRENT
    CURRENT = name
    print("-- %s" % name)


def load_lora_chat():
    """Import tools/lora_chat.py WITHOUT running main (torch import only)."""
    spec = importlib.util.spec_from_file_location(
        "lora_chat_uut", os.path.join(ROOT, "tools", "lora_chat.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    print("== bilingual LoRA corpus honesty ==")
    sys.path.insert(0, os.path.join(ROOT, "models", "hf-orig"))
    from hf_rwkv_tokenizer import RwkvTokenizer
    tok = RwkvTokenizer(vocab_file=os.path.join(
        ROOT, "tools", "data", "rwkv_vocab_v20230424.txt"))
    lc = load_lora_chat()

    test("bilingual fixture loads and is trainable (all roundtrip)")
    fixture = os.path.join(HERE, "fixtures", "lora_chat_bilingual.tsv")
    pairs = lc.load_corpus(fixture)
    check("pairs loaded", len(pairs) >= 8, str(len(pairs)))
    check("guard sees no offence", lc.refuse_uncovered(pairs, tok) == [])

    test("native Bengali with an uncovered char is FLAGGED (fail-closed)")
    # U+0986 (আ) and U+09AD (ভ) measured uncovered in the shipped vocab:
    # they encode to a token that decodes to U+FFFD.
    bad_pairs = [("ami ", "tumi "), ("আমি ", "ভালো ")]
    flagged = lc.refuse_uncovered(bad_pairs, tok)
    check("only the corrupted pair flagged", len(flagged) == 1, str(flagged))
    check("index reported", flagged and flagged[0][0] == 2, str(flagged))
    check("offender is the Bengali line",
          flagged and "আ" in flagged[0][1], str(flagged))

    test("measured vocabulary facts stay true (regression tripwire)")
    check("U+0986 does NOT roundtrip",
          tok.decode(tok.encode("\u0986")) != "\u0986")
    check("U+09AD does NOT roundtrip",
          tok.decode(tok.encode("\u09ad")) != "\u09ad")
    for cp in (0x09AE, 0x09BF, 0x09B2, 0x09BE):  # measured covered
        ch = chr(cp)
        check("U+%04X roundtrips" % cp, tok.decode(tok.encode(ch)) == ch)

    test("romanised Bengali (the honest bilingual path) roundtrips")
    for s in ("ami bhalo achi", "amar nam OmniSeed", "ajke kemon hobe?"):
        check("%r roundtrips" % s, tok.decode(tok.encode(s)) == s)

    print("\nRESULT: %d passed, %d failed" % (PASSED, FAILED))
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
