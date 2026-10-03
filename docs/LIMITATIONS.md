# Limitations

Jisr is a student demo. Read this before using it with real people.

## What it can and cannot do

- **Only a short vocabulary.** The app knows the words listed in `config/vocab.json` (see
  `docs/REPORT.md` for the final list and how well each one works). Any other sign is, at best,
  reported as "unclear".
- **Isolated words only, not continuous signing.** The signer must make one sign, lower the hands,
  wait for the candidates, and confirm. Natural sign language, where signs flow into each other and
  the face and body carry grammar, is not recognized.
- **The sentence composer adds words the user did not sign.** For example "دواء" + "صداع" becomes
  "أحتاج إلى دواء للصداع". The rules are simple and were written by us, not by a deaf person.
  The sentence is always shown in large text before it is spoken, so the signer can check it.
- **Signs differ across the Arab world.** The training data (KArSL) follows the unified Arabic sign
  language dictionary, recorded in Saudi Arabia. Signs used in Oman, or by a particular community,
  may be different for the same word.

## Words that work badly for a new signer

On the held-out test signer (`reports/metrics.json`), 18 of the 23 words are recognized in 90% or more of
the clips. The rest:

- **شكراً** is read as **ذكي** in every clip of that signer: in KArSL both signs touch the forehead and
  open outwards, and signer 3 makes them the same way. The app will often show ذكي when the student signs
  شكراً; she can tap the alternative, or end with الحمد لله instead.
- **بنت، يفكر، أم، إعاقة سمعية** are recognized in only about half of signer 3's clips, because he performs
  them differently in one of his two recording sessions. The gallery shows signers 1 and 3 side by side; the
  student should copy signer 1.
- **أهلاً وسهلاً** and **السلام عليكم** are the same hand-to-forehead gesture in KArSL, so the app has one
  greeting class and always says both: «السلام عليكم وأهلاً وسهلاً بكم».

## Where the accuracy numbers come from, and what they do not cover

- The model was trained on **two signers** and tested on a **third signer** who was never used for
  any decision (`reports/metrics.json`). This is an honest test for "a new person", but three people
  is a very small group. Accuracy on new signers can be lower than the reported number, especially
  for signers who are left-handed, sign faster, or sign in a slightly different style.
- The training videos were recorded with a **Kinect camera in a studio**: the signer is seated, well
  lit, in front of a plain green background, filling the picture. The app runs on a **phone camera**
  in real places. This difference ("domain gap") was not measured, because the rules of this project
  forbid recording new videos.
- The training clips start when the hand is already up and stop before it comes down. The app cuts
  off the raising and lowering of the hand with a rule (step 3 in `docs/features.md`). The rule
  was checked on the dataset, where it removes very few frames, but it could not be checked on real
  phone recordings.
- The "other sign" class was built from 60 KArSL signs outside the vocabulary. It helps the app say
  "unclear" for unknown signs, but it was never shown everyday movements such as scratching one's
  head or waving, so some of those may still produce a word.

## Technical limits

- Needs a reasonably recent phone. On a slow phone the camera loop drops below 15 frames per second,
  and fast signs may be missed.
- Needs a front camera, good light, and both shoulders in the picture.
- Speech needs an Arabic voice installed on the device. Many desktop computers have none; the app
  then shows the sentence in very large text instead.
- The first visit downloads about 30 MB (the two MediaPipe models and the ONNX runtime). After that
  the app works offline.

## Not a replacement for a human interpreter

Jisr can help with a few simple words at a pharmacy counter. It cannot replace a qualified sign
language interpreter, and it must not be used for medical, legal or emergency communication.

**Recommendation:** before any real use, test the app with a deaf community member or a sign
language interpreter, go through `docs/manual_test_checklist.md` with them, and write down which
signs they would make differently. Their feedback matters more than any number in this repository.
