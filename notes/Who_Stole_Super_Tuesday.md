# Who_Stole_Super_Tuesday - Error Analysis

**Winner (A)**: iterative  
**Loser (B)**: concat  
**Ground Truth Culprit**: Simon Knowles

---

## Results Summary

| Method | Predicted Culprit | Correct? |
|--------|-------------------|----------|
| **Iterative** | Simon Knowles | Yes |
| **Concat** | Barry | No |

---

## A's Prediction (Iterative - Winner)

- **Predicted**: Simon Knowles
- **Correct**: Yes

**Reasoning chain (from o3)**:
1. Ricky was tied up by a three-hour exam (10am-1pm) and couldn't have been in the lab at noon
2. Xavier has no friends to help with the distraction scheme
3. Simon was seen leaving with five friends in a pickup truck - a ready-made crew to distract poll workers

---

## B's Prediction (Concat - Loser)

- **Predicted**: Barry
- **Correct**: No

**Reasoning chain (from o3)**:
1. The ONLY direct evidence of a hacking plot comes from Barry's mouth
2. Barry knows technical details before anyone else suspects foul play
3. Barry volunteers to "station himself" at the polling place while sending Amy away
4. Barry steered the investigation toward three names, classic "magician's force" trick
5. Concluded Barry orchestrated everything himself

---

## Q1: What error did B (concat) make?

**Error type: Missing critical timing information leading to wrong inference**

The concat summary failed to preserve the key timing clues that enable the deduction:

1. **Missing: Barry heard the plot at noon** - The concat summary says Barry "overheard" the plot but doesn't specify it was "during my lunch break at noon."

2. **Missing: Ricky's exam timing** - Concat says Ricky was "studying for a final exam earlier that day." The iterative summary preserves the exact timing: "the test lasted from 10 a.m. to 1 p.m."

Without both pieces of information, the reader cannot construct the alibi logic: "If Barry was in the lab at noon and Ricky's test was 10am-1pm, then Ricky couldn't have been the one Barry overheard."

Because the concat summary omits this alibi-establishing timing, o3 had no way to eliminate Ricky. Instead, it focused on Barry's suspicious behavior (knowing too much, positioning himself at the polling place) and constructed an elaborate alternative theory.

---

## Q2: Why didn't A (iterative) make this error?

The iterative summary preserved the critical timing details across incremental updates:

**From iterative penultimate summary:**
> "He overheard this plot during his lunch break in the college computer lab"
> "Ricky had a final exam that morning...the test lasted from 10 a.m. to 1 p.m."

The iterative method's step-by-step approach maintained the specific timing information because:
1. Each chunk's details were summarized with the previous context
2. The timing clue appeared in chunk 3 (Ricky's exam hours) and was explicitly connected to the noon timing in the final deduction
3. The incremental nature forced preservation of details that might later prove important

With both timing pieces preserved, o3 could correctly reason: Ricky has an alibi (exam at noon), Xavier has no friends, therefore Simon (with 5 roommates who just left together) must be the culprit.

---

## Q3: Information in A (iterative) missing from B (concat)?

**Critical information present in iterative but absent/degraded in concat:**

| Detail | Iterative | Concat |
|--------|-----------|--------|
| Barry's timing | "during his lunch break" (implies noon) | "overheard" (no timing) |
| Ricky's exam hours | "10 a.m. to 1 p.m." | "earlier that day" |
| Xavier has no friends | "has no friends to help him study" | "social isolation" |
| Simon left with friends | "five friends had just left" | "five housemates left" |

The most damaging omission is the **timing specificity**. The concat summary generalizes "10am-1pm exam" to "earlier that day," which destroys the alibi logic entirely. A reader cannot deduce that Ricky was unavailable at noon if they only know he had "an exam earlier."

---

## Q4: Other remarks

### The "suspicious Barry" red herring

The concat summary's lack of timing information left o3 with incomplete data. Interestingly, o3 then constructed a plausible alternative theory: Barry fabricated the story to frame others while positioning himself as the hacker. This reasoning is internally consistent but wrong.

This illustrates a key failure mode: **when critical disambiguating information is lost, the model will find SOME explanation, often a plausible-sounding wrong one**.

### Pattern: Alibi information is fragile under compression

This mystery relies on a **process of elimination** structure:
- 3 suspects on the sign-in sheet
- Each has a different alibi or characteristic that eliminates them
- Only one (Simon) has both opportunity AND the required accomplices

This type of mystery is particularly vulnerable to concat's over-compression because:
1. Alibis require **specific details** (exact times, exact circumstances)
2. Generalizing these details destroys their usefulness
3. The deduction chain breaks if any link is missing

### Word counts

- Concat penultimate summary: ~310 words
- Iterative penultimate summary: ~750 words

The iterative summary is ~2.4x longer, which allowed preservation of the timing details that concat compressed away.

### Solve rate context

Original human solve rate: 35.3% (912 attempts)

This is a moderately difficult puzzle for humans, requiring careful tracking of multiple suspects and their alibis. The timing-based deduction is the key insight.

---

## Summary

**Concat failed because** it over-compressed the timing information needed to establish Ricky's alibi. Without knowing (1) Barry was in the lab at noon and (2) Ricky's exam was 10am-1pm, the reader cannot eliminate Ricky through alibi logic. O3 then constructed an alternative (wrong) theory implicating Barry.

**Iterative succeeded because** its incremental approach preserved the specific timing details across summary updates. With the full alibi chain intact, o3 could correctly eliminate Ricky and Xavier, leaving Simon as the only viable suspect.

**Error category**: Missing critical reasoning chain (alibi/timing information)

---

*Analysis date: 2026-01-22*
