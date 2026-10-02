# Manual test checklist

Automatic tests cannot check everything (a real camera, a real voice, a real phone).
Go through this list once on a desktop computer and once on an Android phone, and tick each line.

## Before you start

- Desktop: run `npm run build` then `npm run preview` in `web/`, and open http://localhost:4173 in Chrome.
- Android: the camera only works over HTTPS. Use the deployed address, or connect the phone by USB,
  open `chrome://inspect` on the computer, add port forwarding 4173 -> localhost:4173, then open
  http://localhost:4173 in Chrome on the phone.

## Welcome screen

| # | Check | Desktop | Android |
|---|---|---|---|
| 1 | The page is in Arabic and reads right to left | ☐ | ☐ |
| 2 | The three tips and the privacy sentence are visible without zooming | ☐ | ☐ |
| 3 | "افتح الكاميرا" asks for camera permission | ☐ | ☐ |
| 4 | If permission is refused, a clear Arabic message appears and the button works again | ☐ | ☐ |

## Camera screen

| # | Check | Desktop | Android |
|---|---|---|---|
| 5 | The camera picture appears, mirrored like a mirror | ☐ | ☐ |
| 6 | With shoulders in view the status says "جاهز، ارفع يدك لتبدأ" | ☐ | ☐ |
| 7 | Moving out of view changes the status to "أظهر كتفيك داخل الصورة" | ☐ | ☐ |
| 8 | "إظهار الهيكل" switches the lines on the hands and arms on and off | ☐ | ☐ |
| 9 | Raising a hand changes the status to "جارٍ التقاط الإشارة" | ☐ | ☐ |
| 10 | The picture stays smooth while signing (about 15 frames per second or more) | ☐ | ☐ |

## Recognition and confirmation

| # | Check | Desktop | Android |
|---|---|---|---|
| 11 | After lowering the hand, up to three large word buttons appear with percentages | ☐ | ☐ |
| 12 | Tapping a word adds it to the strip, and the phone vibrates briefly (Android) | ☐ | ☐ |
| 13 | "أعد الإشارة" closes the candidates without adding a word | ☐ | ☐ |
| 14 | A random gesture (for example scratching the head) gives "غير واضح، أعد الإشارة" | ☐ | ☐ |
| 15 | After a tap, the app waits until the hands are down before listening again | ☐ | ☐ |

## Sentence and speech

| # | Check | Desktop | Android |
|---|---|---|---|
| 16 | The × on a chip removes only that word | ☐ | ☐ |
| 17 | "امسح الكل" removes all words | ☐ | ☐ |
| 18 | "دواء" then "صداع" shows "أحتاج إلى دواء للصداع" in large text | ☐ | ☐ |
| 19 | "انطق" speaks the sentence in Arabic | ☐ | ☐ |
| 20 | "أعد النطق" repeats it, and the three speeds sound different | ☐ | ☐ |
| 21 | On a device with no Arabic voice, the notice appears and the large text stays readable | ☐ | ☐ |

## Offline and install

| # | Check | Desktop | Android |
|---|---|---|---|
| 22 | After one visit, turn on airplane mode and reload: the app still opens and recognizes signs | ☐ | ☐ |
| 23 | Chrome offers "Install app" / "Add to Home screen", and the installed app opens full screen | ☐ | ☐ |
| 24 | In the browser's network tools, no request leaves the site while signing and speaking | ☐ | n/a |

## With a signer

| # | Check | Result |
|---|---|---|
| 25 | A person who knows Arabic sign language tries every word in the vocabulary 3 times. Write down how many were in the top 3. | |
| 26 | Ask them whether the signs in the dataset match the signs used in Oman. Write down the words that differ. | |
