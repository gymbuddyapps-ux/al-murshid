// The learning page: one card per word, with short videos of the sign from two signers.
// The list and the videos are made by scripts/make_sign_videos.py (public/signs/).

export async function fillLearnPage(grid) {
  let data;
  try {
    data = await (await fetch('signs/signs.json')).json();
  } catch {
    grid.textContent = 'تعذّر تحميل مقاطع الإشارات.';
    return;
  }

  grid.replaceChildren(
    ...data.signs.map((sign) => {
      const card = document.createElement('article');
      card.className = 'sign-card';

      const title = document.createElement('h3');
      title.className = 'sign-card__title';
      title.textContent = sign.arabic;
      const subtitle = document.createElement('span');
      subtitle.className = 'sign-card__english';
      subtitle.textContent = sign.english;
      title.append(' ', subtitle);

      const row = document.createElement('div');
      row.className = 'sign-card__videos';
      for (const clip of sign.videos) {
        const figure = document.createElement('figure');
        const video = document.createElement('video');
        video.src = `signs/${clip.file}`;
        video.muted = true;
        video.loop = true;
        video.playsInline = true;
        video.preload = 'metadata';
        video.setAttribute('aria-label', `${sign.arabic}، الموقّع ${clip.signer}`);
        // Play only while the card is on screen, to save battery.
        const caption = document.createElement('figcaption');
        caption.textContent = `الموقّع ${clip.signer}`;
        figure.append(video, caption);
        row.append(figure);
      }
      card.append(title, row);
      return card;
    }),
  );

  // Start and stop the videos as they scroll into and out of view.
  const watcher = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        const video = entry.target;
        if (entry.isIntersecting) video.play().catch(() => {});
        else video.pause();
      }
    },
    { threshold: 0.25 },
  );
  grid.querySelectorAll('video').forEach((video) => watcher.observe(video));
}
