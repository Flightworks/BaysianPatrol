import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const packageJson = JSON.parse(
  readFileSync(new URL('../package.json', import.meta.url), 'utf8'),
) as { version: string };

test('realistic fast-boat defaults belong to version 2.4.4', () => {
  assert.equal(packageJson.version, '2.4.4');
});
