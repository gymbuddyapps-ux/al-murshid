// Sentence composer: turns the confirmed words into a correct Arabic sentence.
//
// The rules live in config/templates.json so they can be changed without touching code.
// Example: ["السلام عليكم", "دواء", "صداع"]  ->  "السلام عليكم، أحتاج إلى دواء للصداع"
//
// How it works: we walk through the words from the first to the last. At each position
// we try the rules in order. The first rule whose pattern matches the next words wins,
// and we jump past the words it used. If no rule matches, the word is kept as it is.

// Does one pattern item (an exact word, or a group like "<symptom>") match this word?
function itemMatches(item, word, groups) {
  if (item.startsWith('<') && item.endsWith('>')) {
    const group = groups[item.slice(1, -1)] || [];
    return group.includes(word);
  }
  return item === word;
}

// Does the whole pattern match the words starting at `position`?
function patternMatches(pattern, words, position, groups) {
  if (position + pattern.length > words.length) return false;
  return pattern.every((item, i) => itemMatches(item, words[position + i], groups));
}

// Replace {0}, {1}, {1.for} ... in a rule's output with the matched words.
function fillOutput(output, matchedWords, forms) {
  return output.replace(/\{(\d+)(?:\.(\w+))?\}/g, (_, index, formName) => {
    const word = matchedWords[Number(index)];
    if (!formName) return word;
    const wordForms = forms[word] || {};
    return wordForms[formName] || word; // no special form known: use the plain word
  });
}

// Main function. `words` is a list of Arabic words, `templates` is config/templates.json.
export function composeSentence(words, templates) {
  const { rules = [], groups = {}, forms = {}, joiner = ' ' } = templates;
  const pieces = [];
  let position = 0;

  while (position < words.length) {
    const rule = rules.find((r) => patternMatches(r.pattern, words, position, groups));
    if (rule) {
      const matched = words.slice(position, position + rule.pattern.length);
      pieces.push(fillOutput(rule.output, matched, forms));
      position += rule.pattern.length;
    } else {
      pieces.push(words[position]); // fallback: keep the word
      position += 1;
    }
  }
  return pieces.join(joiner);
}
