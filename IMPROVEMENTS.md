# Improving Accuracy

A plan for reducing false alarms, written after the system flagged **six things in thirteen minutes** on a desk camera where nothing was happening — someone sitting, standing up, and using a phone.

This file explains what went wrong, what commercial systems do differently, and every option we have. In plain terms.

---

## The problem, in numbers

These are measurements from the actual camera feed, not guesses:

| What we measured | Value | What it should be |
|---|---|---|
| How much of the frame the person fills | **81% tall, 69% wide** | 20-40% tall |
| Box shape (height ÷ width) | **0.65** — wider than tall | ~2.5 — clearly upright |
| Hips visible? | **No** | Yes |
| Knees / ankles visible? | **No** | Yes |
| Body cut off by frame edge? | **Yes** | No |

And what it flagged:

| Time | Flagged as | Confidence | What was really happening |
|---|---|---|---|
| 06:22:29 | spit | 56% | sitting at desk |
| 06:22:57 | spit | 63% | sitting at desk |
| 06:23:48 | vandalism | 56% | working |
| 06:29:57 | vandalism | 65% | hand near face |
| 06:34:57 | fall | 79% | sitting, holding phone |
| 06:35:58 | vandalism | 56% | working |

## Why it happened

**Not bad AI. Bad geometry.** All four detectors assume a CCTV view — a whole person, at a distance, standing upright. Every one of those assumptions is broken by a close-up desk camera, so the maths underneath produces nonsense:

- **Fall** decides "lying down" from box shape. Anything wider than tall looks horizontal. Sitting close to the camera measures 0.65, so you permanently look like you're on the floor. It also needs shoulders-to-hips to judge body angle, and the hips aren't visible — so that angle is invented from noise.
- **Vandalism** measures hand speed relative to body height, so distance doesn't matter. But it assumes the box is a *whole* person. Yours is head-and-shoulders — about a third of your real height. So the yardstick is three times too small and every hand movement reads three times too fast. Reaching for a phone clears the "violent strike" bar easily.
- **Spit** looks for a bowed head with a hand near the mouth. Looking down at a phone is exactly that.

**One root cause, three symptoms.** No amount of threshold tuning fixes it, because the inputs are wrong before any threshold is applied.

---

## What commercial systems do differently

Products in this space advertise long lists — loitering, intrusion, tailgating, abandoned objects, queue length, crowd density, PPE. Worth knowing: **most of that isn't behaviour AI at all.** It's geometry and time on top of a plain object detector:

| Their feature | What it actually is |
|---|---|
| Intrusion | Did a person enter this drawn area? |
| Loitering | Has this person stayed in the area over 60 seconds? |
| Tailgating | Two people crossed this line within 2 seconds |
| Abandoned object | A bag stayed still 30 seconds with nobody near it |
| Crowd density | Count the people in this area |

Reliable, cheap, easy to demo. **Fight, vandalism, spitting and falls — what we built — are the genuinely hard ones**, because no shape or boundary defines them. They're motion patterns over time. Vendors list them beside "object counting" as if equally solved; they aren't.

So our models aren't the weak part. What we're missing is the scaffolding around them:

```
0.  CAMERA PLACEMENT        ← biggest lever, costs nothing
1.  Zones and masks         ← ignore most of the picture
2.  Detect and TRACK        ← follow the same person over time
3.  Judge the behaviour     ← our detectors do this bit
4.  Hold for N seconds      ← ignore split-second blips
5.  Threshold and cooldown  ← don't repeat yourself
6.  Operator feedback       ← mark false alarms, retune
```

We have step 3. Steps 0, 1, 2, 4, 5 and 6 are missing — and four of them map directly onto the false alarms above.

The honest summary: **accuracy in these products is earned per site over weeks of tuning, not shipped in the box.**

---

## Every option we have

Ordered by value for effort. "Code?" means whether it needs development work.

| # | What | Code? | Effort | Impact |
|---|---|---|---|---|
| 0 | **Build a way to measure** | Yes | Small | Enables everything else |
| 1 | **Move the camera** | No | None | Very high |
| 2 | **Usable-pose gate** | Yes | Small | Very high |
| 3 | **Per-camera detector choice** | Yes | Small | High |
| 4 | **Cooldown between alerts** | Yes | Tiny | Medium |
| 5 | **Raise the confidence threshold** | No | None | Low, blunt |
| 6 | **Zones and masks** | Yes | Medium | High |
| 7 | **Overlapping chunks** | Yes | Small | Fixes missed events |
| 8 | **Hold-for-N-seconds confirmation** | Yes | Medium | Very high |
| 9 | **Train on your own footage** | Yes | Large | Highest, long term |
| 10 | **Buy a GPU machine** | No | Cost | Removes sampling |

### 0. Build a way to measure — do this first

Right now there is **no way to tell whether any change helped.** Every decision is guesswork.

The material already exists. `spool/` holds real footage of normal desk activity — that's a **"nothing happening" set**. The 22 clips in `uploaded_clips/` have known answers — that's a **"something happening" set**.

One script runs both and reports two numbers:

- **false alarms per hour** on the quiet footage
- **how many real events were still caught** on the known clips

Every change after this gets judged against those two numbers. Without them we're guessing, and a change that fixes falls might quietly break fights without anyone noticing.

### 1. Move the camera — free, and nothing beats it

Mount it high, looking across the room, so whole people are visible including legs. Box shape flips from 0.65 to around 2.5, hips become visible, and the size yardstick becomes correct.

This fixes all three broken detectors at once, costs nothing, and takes ten minutes. **Do this before writing any code.**

### 2. Usable-pose gate — small job, would have stopped all six

Refuse to judge a person when the maths can't work. Skip them if:

- their hips aren't visible (no way to measure body angle or true size)
- their box is over 60% of the frame height (too close, the yardstick breaks)
- their box touches the edge of the frame (body is cut off)

These aren't invented thresholds — they're conditions the calculations already quietly assume. One file, shared by spit, vandalism and fall.

Worth being honest about the trade: a genuine incident very close to the camera would also be skipped. That's the right call — a judgement made on broken inputs isn't worth having.

### 3. Per-camera detector choice — small job, removes whole categories

A desk camera has no business watching for falls. Let each camera say what it cares about:

```yaml
cameras:
  - name: desk-cam
    detect: [fight]        # skip spit, vandalism and fall entirely
```

Three of the six false alarms disappear by definition. It also makes each chunk faster to check, which helps the speed problem at the same time.

### 4. Cooldown between alerts — tiny job

Don't record the same action on the same camera more than once every few minutes. It doesn't make the system more correct, but it turns "six alerts about nothing" into one — and that is what people actually experience as reliability.

### 5. Raise the confidence threshold — free, but blunt

`min_confidence` in `stream.yaml` is currently `0.5`. The false alarms scored 56-79%, so you'd need about `0.8` to remove most of them — and that would throw away genuine events too. A stopgap while the real fixes land, not a solution.

### 6. Zones and masks — medium job

Draw the area that matters and ignore everything else, before any model runs. For a desk camera you'd mask everything except the doorway. This is how commercial systems throw away most of the picture cheaply.

### 7. Overlapping chunks — small job

Video is cut into 10-second chunks. An incident straddling a boundary gets sliced in half and missed by both halves. Recording a 2-second overlap fixes it.

### 8. Hold-for-N-seconds confirmation — medium job, very effective

Require an event to persist for several seconds before it counts. A phone reach looks like a violent strike for a fraction of a second, not for three.

This is the single most effective trick commercial systems use. It needs the system to follow the same person over time, which we don't do today — we start from scratch every chunk with no memory of the last one. Bigger job, high payoff.

### 9. Train on your own footage — the real long-term answer

Our spit, vandalism and fall detectors are hand-written rules about body position. They generalise poorly because someone wrote down what the behaviour "looks like."

A model trained on footage from *your* cameras, of *your* space, learns what normal looks like there. This is how accuracy actually gets won — and it needs labelled examples, which step 0 starts collecting for free.

### 10. Buy a GPU machine — money instead of compromise

Checking a 10-second chunk takes ~40 seconds on this Mac, so we only sample a fraction of the footage (see `analyze_one_in` in the README). A machine with a decent NVIDIA GPU does it in a second or two, and full coverage becomes realistic.

Worth noting this improves *coverage*, not correctness. A faster machine produces false alarms faster too — so it belongs after the fixes above, not before.

---

## What to do, in order

**This week, no code:**

1. **Move the camera** so whole people are visible. Nothing else comes close.
2. Optionally raise `min_confidence` to `0.7` to quieten things while the rest is built.

**Then, first round of code:**

3. **Step 0** — the measuring script, so we get a baseline *before* changing detector logic.
4. **Step 2** — the usable-pose gate.
5. **Step 3** — per-camera detector choice.
6. **Step 4** — cooldown.

Those four are self-contained, need no new hardware, and between them should take six-false-alarms-in-thirteen-minutes to near zero — with the measuring script proving it rather than us hoping.

**After that**, in rough order of value: hold-for-N-seconds (8), zones (6), overlapping chunks (7), then training on your own footage (9) once enough has been collected.

**The one-line version:** move the camera first, build a way to measure second, and only then change the code.
