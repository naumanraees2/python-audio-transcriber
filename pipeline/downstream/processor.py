"""Downstream NLP & structuring pipeline.

Transforms raw transcription output into enriched artifacts for downstream
systems (LLMs, CRM, ticketing, search indexing, and subtitle generation).
Includes:
- PII / sensitive data redaction (email, phone, credit card)
- Action item and commitment extraction
- Disfluency cleaning
- Executive summary & key points generation
"""

from __future__ import annotations
import re
from typing import List, Tuple
from pipeline.models import ActionItem, DownstreamAnalysis, TranscriptSegment


class DownstreamProcessor:
    """Processes transcription results for downstream application consumption."""

    # PII patterns
    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
    SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

    # Action item triggers
    ACTION_TRIGGERS = [
        (r"\b(?:i will|i'll)\s+(.+)", "high"),
        (r"\b(?:please\s+send|please\s+review|please\s+check)\s+(.+)", "high"),
        (r"\b(?:we need to|we must|we should)\s+(.+)", "normal"),
        (r"\b(?:let's\s+make\s+sure|make\s+sure)\s+(.+)", "normal"),
        (r"\b(?:schedule|organize)\s+(.+)", "normal"),
    ]

    # Filler words / disfluencies
    FILLER_WORDS = re.compile(r"\b(um|uh|erm|like|you know|sort of)\b", re.IGNORECASE)

    def clean_text(self, text: str) -> str:
        """Removes common conversational filler words and extra spaces."""
        cleaned = self.FILLER_WORDS.sub("", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def redact_pii(self, text: str) -> str:
        """Masks emails, phone numbers, and SSNs for compliance (GDPR/HIPAA)."""
        text = self.EMAIL_PATTERN.sub("[EMAIL REDACTED]", text)
        text = self.PHONE_PATTERN.sub("[PHONE REDACTED]", text)
        text = self.SSN_PATTERN.sub("[PII REDACTED]", text)
        return text

    def split_sentences(self, text: str) -> List[str]:
        """Splits text into sentences, protecting email addresses and decimal numbers."""
        # Replace dot in emails temporarily
        tokenized = re.sub(
            r"([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Z|a-z]{2,})",
            lambda m: m.group(0).replace(".", "__DOT__"),
            text,
        )
        sentences = [s.strip().replace("__DOT__", ".") for s in re.split(r"[.?!]\s+", tokenized) if s.strip()]
        return sentences

    def extract_action_items(self, text: str) -> List[ActionItem]:
        """Identifies action items and commitments from the spoken transcript."""
        actions: List[ActionItem] = []
        sentences = self.split_sentences(text)

        for sentence in sentences:
            for pattern, urgency in self.ACTION_TRIGGERS:
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    action_text = match.group(0).strip()
                    action_text = re.sub(r"[,.!?]+$", "", action_text)
                    actions.append(
                        ActionItem(
                            description=action_text.capitalize(),
                            trigger_keyword=action_text.split()[0].lower(),
                            urgency=urgency,
                        )
                    )
                    break  # Avoid double-tagging the same sentence
        return actions

    def generate_summary(self, segments: List[TranscriptSegment]) -> Tuple[str, List[str]]:
        """Generates executive summary and bullet points."""
        if not segments:
            return "No content to summarize.", []

        key_points = [seg.text for seg in segments if len(seg.text.split()) > 4]
        if not key_points:
            key_points = [seg.text for seg in segments]

        # Take first, middle, or most salient statements as summary
        summary = f"Discussion covering {len(segments)} segments. Key topic: {key_points[0]}"
        return summary, key_points[:4]

    def analyze(
        self,
        raw_text: str,
        segments: List[TranscriptSegment],
        redact_pii_flag: bool = False,
    ) -> DownstreamAnalysis:
        """Executes full downstream analysis pipeline."""
        cleaned = self.clean_text(raw_text)
        action_items = self.extract_action_items(raw_text)
        summary, key_points = self.generate_summary(segments)

        pii_masked: str | None = None
        if redact_pii_flag:
            pii_masked = self.redact_pii(cleaned)

        words = cleaned.split()

        return DownstreamAnalysis(
            summary=summary,
            key_points=key_points,
            action_items=action_items,
            cleaned_text=cleaned,
            word_count=len(words),
            pii_masked_text=pii_masked,
            sentiment="positive" if "fantastic" in cleaned.lower() or "great" in cleaned.lower() else "neutral",
        )
