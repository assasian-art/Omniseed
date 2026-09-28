#!/usr/bin/env python3
# =============================================================================
#  OmniSeed — tools/make_head_data.py
#
#  Emits tools/data/head_language.tsv, the labelled example set the offline
#  head trainer fits against. One row per utterance:
#
#      text <TAB> language.intent <TAB> language.language <TAB> language.sentiment
#
#  WHY THIS IS A FILE AND NOT A TEST. tests/test_language_heads.cpp asserts
#  behaviour of the LEXICAL classifier; those assertions are not a training
#  set (they are chosen to probe edge cases, they are not balanced, and they
#  encode expected outputs of a different component). Reusing them would fit
#  the neural head to the lexical head's quirks. This set is written as DATA,
#  with the label convention stated up front, so the fit can be audited.
#
#  LABEL CONVENTION (stated because it is a choice, not a fact):
#    intent    question | statement | command | greeting | farewell | thanks | unknown
#    language  en | bn | mixed | unknown
#              "unknown" = no letters in any script (digits/punctuation only)
#    sentiment positive | negative | neutral
#              Neutral is the default: a greeting or a farewell carries no
#              valence on its own. "thanks" / "ধন্যবাদ" is positive; "good
#              morning" is positive; a bare "hello" is neutral.
#
#  BENGALI IS WRITTEN IN BENGALI SCRIPT, never romanised. A romanised Bengali
#  string is indistinguishable from English at the byte level, so training on
#  romanisations would teach the head nothing about the language label and
#  would silently contradict the lexical classifier's script rule.
#
#  usage:  python tools/make_head_data.py            # writes the TSV
# =============================================================================
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "head_language.tsv")

HEADER = ["text", "language.intent", "language.language", "language.sentiment"]

# (text, intent, language, sentiment)
ROWS = [
    # ---------------------------------------------------------------- greeting
    ("Hello", "greeting", "en", "neutral"),
    ("Hi", "greeting", "en", "neutral"),
    ("Hey there", "greeting", "en", "neutral"),
    ("Hello there", "greeting", "en", "neutral"),
    ("Good morning", "greeting", "en", "positive"),
    ("Good evening", "greeting", "en", "positive"),
    ("Hey, good to see you", "greeting", "en", "positive"),
    ("Greetings", "greeting", "en", "neutral"),
    ("Hi, are you there", "greeting", "en", "neutral"),
    ("Hello again", "greeting", "en", "neutral"),
    ("নমস্কার", "greeting", "bn", "neutral"),
    ("হ্যালো", "greeting", "bn", "neutral"),
    ("শুভ সকাল", "greeting", "bn", "positive"),
    ("শুভ সন্ধ্যা", "greeting", "bn", "positive"),
    ("আসসালামু আলাইকুম", "greeting", "bn", "neutral"),
    ("কেমন আছেন", "greeting", "bn", "neutral"),
    ("হ্যালো, আপনি কি আছেন", "greeting", "bn", "neutral"),
    ("তুমি কেমন আছো", "greeting", "bn", "neutral"),
    ("Hi আমি ভালো আছি", "greeting", "mixed", "positive"),
    ("Hello ভাই", "greeting", "mixed", "neutral"),
    ("হ্যালো there", "greeting", "mixed", "neutral"),
    ("Good morning বন্ধু", "greeting", "mixed", "positive"),

    # --------------------------------------------------------------- farewell
    ("Goodbye", "farewell", "en", "neutral"),
    ("Bye", "farewell", "en", "neutral"),
    ("See you later", "farewell", "en", "neutral"),
    ("Take care", "farewell", "en", "positive"),
    ("Talk to you tomorrow", "farewell", "en", "neutral"),
    ("I have to go now", "farewell", "en", "neutral"),
    ("Catch you later", "farewell", "en", "neutral"),
    ("Good night", "farewell", "en", "positive"),
    ("বিদায়", "farewell", "bn", "neutral"),
    ("আবার দেখা হবে", "farewell", "bn", "neutral"),
    ("আমি এখন যাচ্ছি", "farewell", "bn", "neutral"),
    ("ভালো থাকবেন", "farewell", "bn", "positive"),
    ("পরে কথা হবে", "farewell", "bn", "neutral"),
    ("শুভ রাত্রি", "farewell", "bn", "positive"),
    ("Bye বন্ধু", "farewell", "mixed", "neutral"),
    ("আবার দেখা হবে, take care", "farewell", "mixed", "positive"),
    ("See you পরে", "farewell", "mixed", "neutral"),

    # ----------------------------------------------------------------- thanks
    ("Thanks", "thanks", "en", "positive"),
    ("Thank you", "thanks", "en", "positive"),
    ("Thanks a lot", "thanks", "en", "positive"),
    ("Thank you so much", "thanks", "en", "positive"),
    ("Thanks, that helps", "thanks", "en", "positive"),
    ("Much appreciated", "thanks", "en", "positive"),
    ("Thank you for the update", "thanks", "en", "positive"),
    ("Thanks, I appreciate it", "thanks", "en", "positive"),
    ("ধন্যবাদ", "thanks", "bn", "positive"),
    ("অনেক ধন্যবাদ", "thanks", "bn", "positive"),
    ("আপনাকে ধন্যবাদ", "thanks", "bn", "positive"),
    ("সাহায্যের জন্য ধন্যবাদ", "thanks", "bn", "positive"),
    ("খুব ভালো লাগলো, ধন্যবাদ", "thanks", "bn", "positive"),
    ("Thanks ভাই", "thanks", "mixed", "positive"),
    ("ধন্যবাদ, that was quick", "thanks", "mixed", "positive"),
    ("Thank you অনেক", "thanks", "mixed", "positive"),

    # --------------------------------------------------------------- question
    ("Where is the config", "question", "en", "neutral"),
    ("What does this function do", "question", "en", "neutral"),
    ("How does this work", "question", "en", "neutral"),
    ("Why did the build fail", "question", "en", "neutral"),
    ("Is the build green", "question", "en", "neutral"),
    ("Can you explain the router", "question", "en", "neutral"),
    ("When is the next release", "question", "en", "neutral"),
    ("Who owns this module", "question", "en", "neutral"),
    ("Which branch is current", "question", "en", "neutral"),
    ("Do we need a migration", "question", "en", "neutral"),
    ("What is the risk here", "question", "en", "neutral"),
    ("Are the tests passing", "question", "en", "neutral"),
    ("How long does it take", "question", "en", "neutral"),
    ("Should I merge this now", "question", "en", "neutral"),
    ("What went wrong yesterday", "question", "en", "negative"),
    ("Is this a good idea", "question", "en", "neutral"),
    ("Why is this so slow", "question", "en", "negative"),
    ("এটা কী", "question", "bn", "neutral"),
    ("কেন ব্যর্থ হলো", "question", "bn", "negative"),
    ("এটা কিভাবে কাজ করে", "question", "bn", "neutral"),
    ("ফাইলটা কোথায়", "question", "bn", "neutral"),
    ("আপনি কি আমাকে সাহায্য করতে পারবেন", "question", "bn", "neutral"),
    ("কত সময় লাগবে", "question", "bn", "neutral"),
    ("আজ কি ছুটি আছে", "question", "bn", "neutral"),
    ("এটা কি ভালো ধারণা", "question", "bn", "neutral"),
    ("কেন এত ধীর", "question", "bn", "negative"),
    ("কোন শাখা চালু আছে", "question", "bn", "neutral"),
    ("What is এটা", "question", "mixed", "neutral"),
    ("কেন the build failed", "question", "mixed", "negative"),
    ("How does এটা কাজ করে", "question", "mixed", "neutral"),
    ("আপনি কি explain করতে পারবেন", "question", "mixed", "neutral"),
    ("Is it ঠিক আছে", "question", "mixed", "neutral"),

    # ---------------------------------------------------------------- command
    ("Explain the regime engine", "command", "en", "neutral"),
    ("Delete the stale branch", "command", "en", "neutral"),
    ("Run the full test suite", "command", "en", "neutral"),
    ("Fix the failing test", "command", "en", "neutral"),
    ("Show me the last commit", "command", "en", "neutral"),
    ("Rebuild in release mode", "command", "en", "neutral"),
    ("Summarise this document", "command", "en", "neutral"),
    ("Check the logs", "command", "en", "neutral"),
    ("Update the runbook", "command", "en", "neutral"),
    ("Stop the daemon", "command", "en", "neutral"),
    ("Please review this patch", "command", "en", "neutral"),
    ("Rewrite this function", "command", "en", "neutral"),
    ("Clean up the build directory", "command", "en", "neutral"),
    ("Add a regression test", "command", "en", "neutral"),
    ("List the open issues", "command", "en", "neutral"),
    ("ফাইলটা ডিলিট করো", "command", "bn", "neutral"),
    ("ব্যাখ্যা করো", "command", "bn", "neutral"),
    ("পরীক্ষা চালাও", "command", "bn", "neutral"),
    ("লগ দেখাও", "command", "bn", "neutral"),
    ("ঠিক করো এটা", "command", "bn", "neutral"),
    ("আমাকে রিপোর্ট দাও", "command", "bn", "neutral"),
    ("ডকুমেন্ট আপডেট করো", "command", "bn", "neutral"),
    ("সব টেস্ট চালাও", "command", "bn", "neutral"),
    ("Please ব্যাখ্যা করো", "command", "mixed", "neutral"),
    ("Delete করো ওটা", "command", "mixed", "neutral"),
    ("Show me লগ", "command", "mixed", "neutral"),
    ("Run করো the full suite", "command", "mixed", "neutral"),

    # -------------------------------------------------------------- statement
    ("This tree has zero dependencies", "statement", "en", "positive"),
    ("I prefer the second option", "statement", "en", "neutral"),
    ("The build is green", "statement", "en", "positive"),
    ("The tests are failing", "statement", "en", "negative"),
    ("I am working on the parser", "statement", "en", "neutral"),
    ("It compiles without warnings", "statement", "en", "positive"),
    ("The daemon crashed overnight", "statement", "en", "negative"),
    ("We shipped the release", "statement", "en", "positive"),
    ("This approach is too slow", "statement", "en", "negative"),
    ("I will handle it tomorrow", "statement", "en", "neutral"),
    ("The config was wrong", "statement", "en", "negative"),
    ("It works on my machine", "statement", "en", "neutral"),
    ("The docs are out of date", "statement", "en", "negative"),
    ("I think this is the right call", "statement", "en", "positive"),
    ("The migration is complete", "statement", "en", "positive"),
    ("There is a bug in the loader", "statement", "en", "negative"),
    ("I have finished the review", "statement", "en", "positive"),
    ("The numbers look good", "statement", "en", "positive"),
    ("আমি ভালো আছি", "statement", "bn", "positive"),
    ("আজ আবহাওয়া ভালো", "statement", "bn", "positive"),
    ("আজ খারাপ দিন", "statement", "bn", "negative"),
    ("এটা কাজ করে না", "statement", "bn", "negative"),
    ("আমি দ্বিতীয় অপশন পছন্দ করি", "statement", "bn", "neutral"),
    ("বিল্ড সফল হয়েছে", "statement", "bn", "positive"),
    ("টেস্ট ব্যর্থ হয়েছে", "statement", "bn", "negative"),
    ("আমি কাল এটা করব", "statement", "bn", "neutral"),
    ("কোডে একটি বাগ আছে", "statement", "bn", "negative"),
    ("সব ঠিক আছে", "statement", "bn", "positive"),
    ("এটা খুব ধীর", "statement", "bn", "negative"),
    ("ডকুমেন্টেশন পুরনো", "statement", "bn", "negative"),
    ("I prefer দ্বিতীয় option", "statement", "mixed", "neutral"),
    ("The build সফল হয়েছে", "statement", "mixed", "positive"),
    ("এটা কাজ করে না, it is broken", "statement", "mixed", "negative"),
    ("আমি কাল handle করব", "statement", "mixed", "neutral"),
    ("সব ঠিক আছে, we are green", "statement", "mixed", "positive"),

    # ---------------------------------------------------------------- unknown
    ("", "unknown", "unknown", "neutral"),
    ("12345", "unknown", "unknown", "neutral"),
    ("...", "unknown", "unknown", "neutral"),
    ("!!!", "unknown", "unknown", "neutral"),
    ("???", "unknown", "unknown", "neutral"),
    ("---", "unknown", "unknown", "neutral"),
    ("42", "unknown", "unknown", "neutral"),
    ("0.85", "unknown", "unknown", "neutral"),
    ("@#$%", "unknown", "unknown", "neutral"),
    ("   ", "unknown", "unknown", "neutral"),
    ("()", "unknown", "unknown", "neutral"),
    ("3.14159", "unknown", "unknown", "neutral"),

    # ================================================== batch 2: rebalancing
    # Batch 1 came out 101/161 neutral on sentiment and light on `unknown`.
    # A head fitted on a 63%-neutral set learns "neutral" as the prior and its
    # confidence is dominated by that prior, which then has to be undone by a
    # large temperature. These rows add valence where the utterance genuinely
    # has it, so the fit sees a less lopsided target.
    # ---------------------------------------------------- positive statements
    ("The refactor made the code much clearer", "statement", "en", "positive"),
    ("I am happy with how this turned out", "statement", "en", "positive"),
    ("This is a solid improvement", "statement", "en", "positive"),
    ("The new loader is twice as fast", "statement", "en", "positive"),
    ("Everything works exactly as intended", "statement", "en", "positive"),
    ("That fix was clean and minimal", "statement", "en", "positive"),
    ("The coverage went up nicely", "statement", "en", "positive"),
    ("I like this design a lot", "statement", "en", "positive"),
    ("The error message is genuinely helpful", "statement", "en", "positive"),
    ("We are in good shape for the release", "statement", "en", "positive"),
    ("This is the best version so far", "statement", "en", "positive"),
    ("অনেক ভালো হয়েছে", "statement", "bn", "positive"),
    ("এই পরিবর্তনটা দারুণ", "statement", "bn", "positive"),
    ("কাজটা সুন্দর হয়েছে", "statement", "bn", "positive"),
    ("আমি খুশি হয়েছি", "statement", "bn", "positive"),
    ("এটা অনেক দ্রুত", "statement", "bn", "positive"),
    ("সব ঠিকঠাক চলছে", "statement", "bn", "positive"),
    ("This is দারুণ", "statement", "mixed", "positive"),
    ("The fix অনেক ভালো", "statement", "mixed", "positive"),
    ("আমি happy with this", "statement", "mixed", "positive"),
    ("দুর্দান্ত, very clean", "statement", "mixed", "positive"),

    # ---------------------------------------------------- negative statements
    ("The build broke again", "statement", "en", "negative"),
    ("I am frustrated with this bug", "statement", "en", "negative"),
    ("This is a terrible idea", "statement", "en", "negative"),
    ("The parser is hopelessly slow", "statement", "en", "negative"),
    ("Nothing works the way it should", "statement", "en", "negative"),
    ("That refactor made things worse", "statement", "en", "negative"),
    ("The error message is useless", "statement", "en", "negative"),
    ("I dislike this approach", "statement", "en", "negative"),
    ("We are in bad shape for the release", "statement", "en", "negative"),
    ("The tests are flaky and annoying", "statement", "en", "negative"),
    ("This regression is painful", "statement", "en", "negative"),
    ("এটা খুব খারাপ", "statement", "bn", "negative"),
    ("কাজটা খারাপ হয়েছে", "statement", "bn", "negative"),
    ("আমি হতাশ হয়েছি", "statement", "bn", "negative"),
    ("কিছুই ঠিকভাবে কাজ করছে না", "statement", "bn", "negative"),
    ("এটা বিরক্তিকর", "statement", "bn", "negative"),
    ("আবার ভেঙে গেছে", "statement", "bn", "negative"),
    ("This is খারাপ", "statement", "mixed", "negative"),
    ("আমি frustrated with this bug", "statement", "mixed", "negative"),
    ("Nothing works ঠিকভাবে", "statement", "mixed", "negative"),
    ("The parser is hopelessly ধীর", "statement", "mixed", "negative"),

    # --------------------------------------- more questions / commands / thanks
    ("What is the expected output", "question", "en", "neutral"),
    ("Could you clarify the second point", "question", "en", "neutral"),
    ("Is there a cheaper option", "question", "en", "neutral"),
    ("Where did this number come from", "question", "en", "neutral"),
    ("Which test covers this path", "question", "en", "neutral"),
    ("এটা কীভাবে ঠিক করব", "question", "bn", "neutral"),
    ("কোনটা ভালো অপশন", "question", "bn", "neutral"),
    ("এই সংখ্যাটা কোথা থেকে এলো", "question", "bn", "neutral"),
    ("Where did এই সংখ্যা come from", "question", "mixed", "neutral"),
    ("Which test covers এটা", "question", "mixed", "neutral"),
    ("Refactor the loader", "command", "en", "neutral"),
    ("Write a test for the edge case", "command", "en", "neutral"),
    ("Commit these changes", "command", "en", "neutral"),
    ("Measure the throughput again", "command", "en", "neutral"),
    ("রিফ্যাক্টর করো", "command", "bn", "neutral"),
    ("টেস্ট লিখো", "command", "bn", "neutral"),
    ("Commit করো এই পরিবর্তন", "command", "mixed", "neutral"),
    ("Thanks for catching that", "thanks", "en", "positive"),
    ("Thank you, that was very helpful", "thanks", "en", "positive"),
    ("অনেক উপকার হলো, ধন্যবাদ", "thanks", "bn", "positive"),
    ("Thanks, আমি বুঝেছি", "thanks", "mixed", "positive"),
    ("শুভেচ্ছা", "greeting", "bn", "neutral"),
    ("Hi, welcome back", "greeting", "en", "positive"),
    ("আবার স্বাগতম", "greeting", "bn", "positive"),
    ("শুভ বিদায়", "farewell", "bn", "neutral"),
    ("Goodbye, ভালো থেকো", "farewell", "mixed", "positive"),
    ("ক", "unknown", "bn", "neutral"),
    ("A", "unknown", "en", "neutral"),
    ("x", "unknown", "en", "neutral"),
    ("0", "unknown", "unknown", "neutral"),
    ("--help", "unknown", "unknown", "neutral"),
    ("[]", "unknown", "unknown", "neutral"),
    ("####", "unknown", "unknown", "neutral"),
    ("%", "unknown", "unknown", "neutral"),
    ("\u00a0", "unknown", "unknown", "neutral"),
]


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    seen = set()
    dupes = []
    for r in ROWS:
        if r[0] in seen:
            dupes.append(r[0])
        seen.add(r[0])
    if dupes:
        print("duplicate texts: %r" % dupes, file=sys.stderr)
        return 2

    # A tab or newline inside a field would silently corrupt the TSV.
    for r in ROWS:
        for cell in r:
            if "\t" in cell or "\n" in cell:
                print("field contains a tab or newline: %r" % (r,), file=sys.stderr)
                return 2

    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(HEADER) + "\n")
        for r in ROWS:
            f.write("\t".join(r) + "\n")

    # Per-set label balance, printed so a degenerate label distribution is
    # visible here rather than discovered as "the head always says neutral".
    from collections import Counter
    for col, name in enumerate(HEADER[1:], start=1):
        c = Counter(r[col] for r in ROWS)
        print("%-20s %s" % (name, dict(sorted(c.items()))))
    print("wrote %d rows -> %s" % (len(ROWS), OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
