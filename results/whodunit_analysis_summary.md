# Whodunit Evaluation Results Analysis

## Summary Table

| Input Type | Range Spec | Method | Items | Culprit Accuracy | Both Correct | Avg Length (chars) | Notes |
|------------|------------|--------|-------|------------------|--------------|-------------------|-------|
| **Chunks** | all | - | 67 | **0.761** | 0.642 | 36,408 | Full story, best performance |
| **Chunks** | all-but-last | - | 34 | 0.618 | 0.441 | 30,228 | Pre-reveal chunks |
| **Summaries** | penultimate | concat | 135 | 0.474 | 0.289 | 3,036 | Pre-reveal concat summaries |
| **Summaries** | penultimate | iterative | 62 | 0.500 | 0.370 | 4,113 | Pre-reveal iterative summaries |

## Specific Question Analysis

### 1. Chunk Accuracy (especially "all" range)

**Key Finding: Chunks with "all" range perform exceptionally well**

- **Accuracy: 76.1%** (51/67 correct out of 67 total items)
- **Average text length: 36,408 characters** (~7,280 words)
- **Performance: Best among all conditions tested**

**Common Mistakes (16 incorrect cases):**
- **Different suspect identified: 56% of errors** (9/16 cases)
- **Confused culprit with accomplice: 38% of errors** (6/16 cases)  
- **Included other suspects: 31% of errors** (5/16 cases)
- **Only provided alias: 6% of errors** (1/16 cases)

**Conclusion:** The "all" version does quite well, achieving the highest accuracy. When it makes mistakes, it's primarily by identifying a completely different suspect rather than minor naming issues.

### 2. Pre-reveal Comparison (chunks vs concat summaries)

**Chunks (all-but-last range):**
- **Accuracy: 61.8%** (21/34 correct)
- **Average length: 30,228 characters** (~6,046 words)

**Concat Summaries (penultimate range):**
- **Accuracy: 47.4%** (64/135 correct)
- **Average length: 3,036 characters** (~607 words)

**Short Concat Summaries (<500 words):**
- **Accuracy: 44.1%** (15/34 correct)
- **Average length: 1,262 characters** (~252 words)

**Conclusion:** Pre-reveal chunks significantly outperform concat summaries (61.8% vs 47.4%). Even short concat summaries perform reasonably well at 44.1%, suggesting the summarization captures key detective elements even in very condensed form.

### 3. Effect of Length on Concat Summaries

**Length Analysis:**
- **Short (1,262 chars):** 44.1% accuracy
- **Medium (2,810 chars):** 47.1% accuracy  
- **Medium (3,188 chars):** 55.9% accuracy
- **Medium (4,939 chars):** 42.4% accuracy

**Conclusion:** There's a **non-linear relationship** between length and accuracy. The sweet spot appears to be around 3,200 characters (~640 words), with the longest summaries actually performing worse, possibly due to including irrelevant details or noise.

### 4. Concat vs Iterative Summaries

**Concat Summaries:**
- **Accuracy: 47.4%** (64/135 correct)
- **Average length: 3,036 characters**
- **4 different evaluation runs**

**Iterative Summaries:**
- **Accuracy: 50.0%** (31/62 correct)  
- **Average length: 4,113 characters**
- **2 different evaluation runs**

**Conclusion:** Iterative summaries show a **slight advantage** over concat summaries (50.0% vs 47.4%), while being ~36% longer on average. The difference is modest but suggests iterative summarization may preserve more relevant detective story elements.

## Key Insights

1. **Full chunks are superior:** The "all" range with chunks achieves 76% accuracy, significantly outperforming all summary-based approaches.

2. **Pre-reveal is challenging:** Both chunks and summaries see substantial accuracy drops when the reveal is excluded (chunks: 76% → 62%, summaries: ~47%).

3. **Summaries are surprisingly effective:** Even very short summaries (~250 words) achieve 44% accuracy, demonstrating that key detective elements can be preserved in condensed form.

4. **Length optimization matters:** For concat summaries, there's an optimal length around 600-650 words. Longer isn't always better.

5. **Iterative has slight edge:** Iterative summarization shows modest improvements over concatenation, possibly due to better information integration across chunks.

## Recommendations

- **For highest accuracy:** Use full chunks with "all" range (76% accuracy)
- **For pre-reveal analysis:** Chunks still outperform summaries significantly  
- **For summary-based evaluation:** Iterative summaries with ~600-800 words appear optimal
- **For efficiency:** Even very short summaries maintain reasonable performance for screening purposes