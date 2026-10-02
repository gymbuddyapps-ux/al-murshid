import { createHash } from 'node:crypto';
import { readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { join, relative } from 'node:path';
import { defineConfig } from 'vite';

// List every file inside a folder (and its sub-folders).
function listFiles(folder) {
  return readdirSync(folder).flatMap((name) => {
    const path = join(folder, name);
    return statSync(path).isDirectory() ? listFiles(path) : [path];
  });
}

// After the build, write the list of all built files into the service worker.
// The service worker saves these files on the phone so the app works offline.
function offlineFileList() {
  return {
    name: 'jisr-offline-file-list',
    apply: 'build',
    closeBundle() {
      const dist = join(import.meta.dirname, 'dist');
      const files = listFiles(dist)
        .map((path) => '/' + relative(dist, path).replaceAll('\\', '/'))
        .filter((url) => url !== '/sw.js');
      // The version changes whenever any file changes, so phones pick up new builds.
      const swPath = join(dist, 'sw.js');
      const swSource = readFileSync(swPath, 'utf-8');
      const hash = createHash('sha256').update(swSource);
      for (const url of files) hash.update(readFileSync(join(dist, url)));
      const version = hash.digest('hex').slice(0, 12);

      const sw = swSource
        .replace('"__FILES__"', JSON.stringify(files))
        .replace('__VERSION__', version);
      writeFileSync(swPath, sw);
      console.log(`service worker: ${files.length} files, version ${version}`);
    },
  };
}

export default defineConfig({
  plugins: [offlineFileList()],
  server: {
    // Allow importing the shared settings in ../config/
    fs: { allow: ['..'] },
  },
  test: {
    include: ['tests/**/*.test.js'],
  },
});
