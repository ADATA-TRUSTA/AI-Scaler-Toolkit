# Short-drama screenwriting rules (excerpted from ai-short-drama-screenwriter)

Source: https://github.com/POUND0423/AI-drama-pound (MIT License, Copyright (c) 2026 POUND0423; the full licence text is at the end of this file).
This file is the basis for story.py's prompts; the full skill is installed at `../ai-short-drama-screenwriter/`.
When a human needs to write a script, review one, or make a single targeted revision, use that one; only the parts the automated pipeline needs are kept here.

---

# Short-drama creation process

The full process first builds a creative brief, then produces output stage by stage as decisions require; unknown non-critical fields are marked as assumptions, and multiple questions are not asked at once. The minimum creative brief always contains: genre, core premise, protagonist's goal, main opposition, audience, number of episodes, episode length, production constraints, deliverables. Adjust to the genre; do not force romance, melodrama or a fixed episode count.

## 1. Creative brief and constraints

### Need to know

Genre, core premise, protagonist's goal, main opposition, audience, number of episodes, episode length, production constraints and deliverables.

### Key decisions

Confirm which constraints cannot change, and whether missing information would materially change the story's direction.

### Deliverables

A creative brief that marks the known conditions and the necessary assumptions.

### Check

Every specified condition can be traced in the later structure and script.

## 2. Subject, story premise and core concept

### Need to know

The creative brief, the protagonist's desire, the source of conflict and the audience's expectations.

### Key decisions

Condense the premise into a core conflict that can keep generating choices and consequences.

### Deliverables

Subject positioning, story premise, core concept and a one-sentence hook.

### Check

The premise can form filmable events within the specified length and production constraints.

## 3. Whole-series and single-episode structure

### Need to know

Number of episodes, episode length, the main mystery or goal, key turning points and the direction of the ending.

### Key decisions

Arrange the progression of the whole series and each episode's goal, opposition, turn and landing point.

### Deliverables

Whole-series beats, single-episode nodes, and each episode's opening or end-of-episode hook.

### Check

Every episode has irreversible new information, a choice or a cost; it is not padded with repeated events.

## 4. Character design and relationships

### Need to know

The main characters' goals, fears, secrets, resources and relationships with each other.

### Key decisions

Make the characters' needs conflict with each other, and set up relationship pressure that forces characters to change.

### Deliverables

Character bios, character relationships, and the key choices in each character arc.

### Check

Every change in a character is supported by a preceding event, visible action or accumulated cost.

## 5. Scene design and escalating conflict

### Need to know

The episode's goal, character relationships, available locations and production constraints.

### Key decisions

For each scene, set the character's goal, the opposition, the change of strategy and the new state at the end.

### Deliverables

A scene list, with each scene's location, time, conflict and change.

### Check

Every scene changes at least one of: information, relationship, goal, risk or emotion.

## 6. Dialogue and subtext

### Need to know

The character's immediate goal, the power balance of the relationship, and what is known and what is concealed.

### Key decisions

Let characters express unspoken needs through evasion, probing, interruption or action.

### Deliverables

Dialogue that fits each character's voice, plus any necessary very short action or pause cues.

Deliverable shape for a single targeted dialogue revision: list the rewritten specified lines directly; if it helps the performance, one very short action or pause cue may be added next to a line.

### Check

Dialogue advances the conflict or the relationship and does not just explain background; a fine-tuning task changes only the specified passage.

## 7. Conflict, emotion, comedy, suspense, twists and end-of-episode hooks

### Need to know

The audience's expectations, earlier clues, the characters' costs, and the question the next episode must advance.

### Key decisions

Choose an emotional rhythm and a twist method suited to the genre, so that hooks come from the consequences of characters' choices.

### Deliverables

Conflict-escalation nodes, emotional landing points, clue placement, twists and end-of-episode hooks.

### Check

Twists hold up when looking back at what came before; hooks pose the next question or cost rather than merely cutting off an event.

## 8. Writing and revising the first draft

### Need to know

The approved brief, structure, characters, scenes, format and production constraints.

### Key decisions

Decide whether to deliver a script, an episode outline or a specified excerpt, and first fix problems that affect comprehension and pacing.

### Deliverables

A filmable first draft, and a revised version or revision suggestions organised by degree of impact.

### Check

Action is written as visible, audible content happening in the present, the format is consistent, and nothing goes beyond the deliverable scope the user specified.


---

# Short-drama script format

## Episode title

Put "第 N 集〈集名〉" (Episode N 〈episode name〉) at the top; if the user did not ask for it, there is no need to fill in episode names for the whole season yourself.

## Scene number and scene heading

Each scene has a heading made of scene number, interior/exterior, location and time, for example "1. 內景・便利商店・夜" (1. INT. convenience store, night). When the user asks for standard short-drama or standard vertical short-drama format, this is a required field for every scene; it must not be replaced by a vague "人物／場景" (characters / setting) preamble.

## Action

Describe characters, objects, picture and sound with visible, audible actions happening right now, and let the action change the state of the scene.

## Character names

Character names go on their own line and stay consistent; when first needed, very short identity information may be added to help recognition.

## Dialogue

Dialogue goes below the character name; avoid replacing conflict, action or reaction with long explanatory passages.

## Parenthetical cues

Only when necessary, use parentheses to mark a brief manner of speaking or an immediate action; do not use them to explain a character's inner thoughts.

## OS / VO

OS means the voice of a character who is not in frame; VO is used for narration or inner voice. The mark must directly follow the character name, for example "阿晴（OS）".

## On-screen text

Clearly mark messages, interfaces or captions the audience can read with "手機畫面——" (phone screen —) or "畫面文字：" (on-screen text:).

## Flashbacks and transitions

Start with "閃回——" (flashback —) and end with an action that returns to the present; mark necessary transitions with "切黑。" (cut to black.) or brief text.

## Vertical-screen considerations

Only when the user asks for vertical format, make key characters, messages and actions easy to recognise within a single vertical frame; do not add shot numbers, shot sizes, camera moves or video-model parameters.

## Full short example

```text
第 1 集

1. 內景・便利商店・夜

雨水沿著玻璃門往下流。阿晴獨自整理貨架，手機在口袋裡連震兩次。

手機畫面——一個停用三年的號碼傳來訊息：「別找我。」

阿晴僵住。她抬頭看向門外，一把紅傘停在玻璃門前。

阿晴
媽？

門外的人影轉身離開。阿晴衝出門，櫃檯上的電話同時響起。

切黑。
```


---

# Short-drama review and revision checklist

## Priority issues

1. Issue: point out specifically where it fails.
   - Basis: the corresponding story beat, scene or line of dialogue.
   - Impact: explain why it hinders comprehension, emotion or pacing.
   - Revision: provide an actionable direction for the fix.

## Recommended to keep

List only elements that genuinely work and should be kept during revision.

## Subject and audience

- [ ] The subject, core premise and target audience are clear and consistent with each other.
- [ ] The genre promise is recognisable to the audience in the opening and the main events.

## Structure and pacing

- [ ] The whole series and each episode have a clear goal, opposition, turn and landing point.
- [ ] Each episode's length fits the specified duration, with no repetitive or stalled passages.

## Character motivation and continuity

- [ ] Characters' choices are supported by established goals, pressures or new events.
- [ ] Character changes have continuous visible actions and costs.

## Scenes and conflict

- [ ] Every scene has a character goal and opposition, and changes information, relationship, goal, risk or emotion.
- [ ] Conflict escalates as the scenes progress, not merely through characters explaining it.

## Dialogue and subtext

- [ ] Dialogue fits the characters' relationships and immediate purposes.
- [ ] Key information is carried as much as possible by action, reaction, evasion or probing.

## Twists and hooks

- [ ] Twists have earlier clues and, looking back, are consistent with the characters and with cause and effect.
- [ ] End-of-episode hooks bring the next question, risk or irreversible cost.

## Filmability and production constraints

- [ ] Locations, characters, props and events fit the specified budget and production conditions.
- [ ] Actions can be seen or heard and do not rely on hard-to-film inner explanation.

## Format and final revision

- [ ] Episode titles, scene headings, character names, dialogue and on-screen text are consistent.
- [ ] When standard short-drama or standard vertical format is requested, every scene heading contains scene number, interior/exterior, location and time.
- [ ] Revisions prioritise problems that hinder comprehension, emotion or pacing, and keep elements that work.

---

## License (of the excerpted source)

```
MIT License

Copyright (c) 2026 POUND0423

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
