---
alwaysApply: true
---

# Change-Based Debugging Rule

When something breaks, ALWAYS start by asking: "What was the most obvious thing that changed that could cause this?" Investigate that first.

## Core Requirements

1. **Pause and think** - Before diving into debugging, stop and consider what recently changed
2. **Identify the obvious suspect** - What's the most likely cause based on recent changes?
3. **Test the obvious first** - Investigate the simplest explanation before complex ones
4. **Only then go deeper** - If obvious causes are ruled out, proceed to complex debugging

## Debugging Process

1. **What changed recently?** - File moves, config changes, dependency updates, etc.
2. **What's the obvious consequence?** - Path issues after moves, missing deps after updates, etc.  
3. **Test that hypothesis first** - Quick verification of the obvious cause
4. **If that fails, then complex debugging** - Only dive deep after ruling out simple causes

## Examples

### ❌ DON'T (Missing the obvious):
```
Script moved to new directory, now fails
→ Spend 20+ minutes debugging complex data pipeline logic
→ Add extensive debug traces throughout system
→ Finally discover it was just missing "../" in paths
```

### ✅ DO (Think first):
```
Script moved to new directory, now fails
→ PAUSE: "What changed? Script location."
→ "What's the obvious consequence? Relative paths broke."
→ Check paths first → Fix in 30 seconds
```

This rule prevents wasting time on complex solutions to simple problems by systematically considering what changed.