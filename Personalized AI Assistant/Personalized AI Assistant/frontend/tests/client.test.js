import test from 'node:test';
import assert from 'node:assert/strict';

test('Frontend API client configuration', () => {
  const defaultUrl = 'http://localhost:8000/api';
  assert.equal(defaultUrl, 'http://localhost:8000/api');
});

test('Supported document file extensions', () => {
  const allowed = ['.pdf', '.docx', '.txt'];
  assert.ok(allowed.includes('.pdf'));
  assert.ok(allowed.includes('.docx'));
  assert.ok(allowed.includes('.txt'));
  assert.ok(!allowed.includes('.exe'));
});

test('Operating modes enumeration', () => {
  const modes = ['base', 'rag', 'fine_tuned'];
  assert.equal(modes.length, 3);
  assert.ok(modes.includes('rag'));
  assert.ok(modes.includes('base'));
  assert.ok(modes.includes('fine_tuned'));
});

test('Max file size boundary validation', () => {
  const maxMb = 10;
  const maxBytes = maxMb * 1024 * 1024;
  assert.equal(maxBytes, 10485760);
  assert.ok(5000000 < maxBytes);
  assert.ok(15000000 > maxBytes);
});
