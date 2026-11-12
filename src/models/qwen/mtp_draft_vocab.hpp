#ifndef GUFO_MODELS_QWEN_MTP_DRAFT_VOCAB_HPP_
#define GUFO_MODELS_QWEN_MTP_DRAFT_VOCAB_HPP_

#include <algorithm>
#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

#include "src/models/qwen/tokenizer.hpp"

namespace gufo::speculative {

/// Whether an MTP draft scores a token with these decoded bytes.
///
/// The LM head is the widest read of a draft step: one byte per weight for
/// every vocabulary row. A draft that never proposes a token only loses that
/// proposal (the target model still decides every token), so the head skips
/// the rows of scripts this deployment does not write: CJK, Hangul, Arabic,
/// Hebrew, Indic and South-East Asian. It keeps ASCII (code, control tokens),
/// U+0080..U+053F (Latin with diacritics, IPA, Greek, Cyrillic),
/// U+2000..U+2FFF (punctuation, arrows, mathematics, box drawing) and
/// U+1D000..U+1FFFF (mathematical alphabets, emoji). A token that is a
/// fragment of a UTF-8 sequence is judged by the lead bytes it carries.
[[nodiscard]] inline bool MtpDraftScoresToken(std::string_view bytes) noexcept {
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    const auto byte = static_cast<unsigned char>(bytes[i]);
    if (byte <= 0xD4 || byte == 0xE2) {
      continue;  // ASCII, continuation bytes, U+0080..U+053F, U+2000..U+2FFF
    }
    if (byte == 0xF0) {
      if (i + 1 == bytes.size()) {
        continue;  // a bare lead byte: emoji as likely as anything else
      }
      const auto next = static_cast<unsigned char>(bytes[i + 1]);
      if (next >= 0x9D && next <= 0x9F) {
        continue;  // U+1D000..U+1FFFF
      }
    }
    return false;
  }
  return true;
}

/// Tokens of history that decide which vocabulary a draft scores.
inline constexpr std::size_t kMtpDraftScriptWindow = 64;

/// Whether the text is being written in a script the draft vocabulary lacks:
/// one of the last kMtpDraftScriptWindow tokens of `history` is outside
/// `listed` (indexed by token id). Such a step scores the whole vocabulary,
/// so those texts keep their drafts. The answer depends on the committed
/// tokens alone, which keeps seeded replay independent of cache restores.
[[nodiscard]] inline bool MtpDraftNeedsWholeVocabulary(
    std::span<const tokenization::TokenId> history,
    const std::vector<bool>& listed) noexcept {
  const std::size_t window = std::min(history.size(), kMtpDraftScriptWindow);
  for (const tokenization::TokenId token : history.last(window)) {
    const auto id = static_cast<std::size_t>(token);
    if (id >= listed.size() || !listed[id]) {
      return true;
    }
  }
  return false;
}

}  // namespace gufo::speculative

#endif  // GUFO_MODELS_QWEN_MTP_DRAFT_VOCAB_HPP_
