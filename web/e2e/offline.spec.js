// Offline test: after one visit, the app must open and load its models with no network.
import { expect, test } from '@playwright/test';
import { openApp } from './browser.js';

test('the app works offline after the first visit', async () => {
  const { browser, page, errors } = await openApp(null, false);

  // Wait until the service worker has saved every file and controls the page.
  // (page.evaluate is used on purpose: waitForFunction runs in a separate world
  // where navigator.serviceWorker does not exist.)
  await page.evaluate(async () => {
    await navigator.serviceWorker.ready;
    while (!navigator.serviceWorker.controller) {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  });
  await page.waitForTimeout(1000);

  // Cut the network and open the app again.
  await page.context().setOffline(true);
  await page.goto('http://localhost:4174/?e2e=1');
  await page.waitForFunction(() => window.__jisr?.ready, null, { timeout: 120000 });
  const status = await page.textContent('#status');
  const manifest = await page.evaluate(() =>
    fetch('/manifest.webmanifest').then((r) => r.ok).catch(() => false),
  );
  await browser.close();

  expect(errors).toEqual([]);
  expect(manifest).toBe(true);
  expect(status).not.toContain('تعذّر');
});
