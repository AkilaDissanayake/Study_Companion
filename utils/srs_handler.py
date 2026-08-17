# srs_handler.py
"""
Spaced Repetition (SM-2) engine.

Implements the classic SuperMemo-2 algorithm — the same one Anki is built
on — where each card carries its own ease factor that adapts over time,
rather than a fixed day-ladder shared by every card. One update function
(`apply_sm2_review`) is the single source of truth, used both by a manual
flashcard review and by automatic scheduling of a wrong quiz answer, so
there is never a second scheduling code path to keep in sync.
"""
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from utils.db_models import Flashcard
from utils.logger import get_logger

logger = get_logger(__name__, "srs_handler.log")

DEFAULT_EASE_FACTOR = 2.5
MIN_EASE_FACTOR = 1.3

# Classic SM-2 quality scale (0-5), mapped to an Anki-style 4-button rating.
QUALITY_MAP = {"again": 0, "hard": 3, "good": 4, "easy": 5}

# Daily review batch cap — SM-2 already spaces reviews out naturally, this
# just bounds a pathological case (e.g. a huge backlog after being away).
DUE_CARDS_LIMIT = 50


def apply_sm2_review(ease_factor: float, interval_days: float, repetitions: int, quality: int) -> dict:
    """Classic SM-2 update. quality 0 ("Again") resets the card to day 1;
    quality >= 3 advances it (1 day -> 6 days -> interval * ease_factor)."""
    if quality < 3:
        repetitions = 0
        interval_days = 1
    else:
        if repetitions == 0:
            interval_days = 1
        elif repetitions == 1:
            interval_days = 6
        else:
            interval_days = round(interval_days * ease_factor)
        repetitions += 1

    ease_factor = max(
        MIN_EASE_FACTOR,
        ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)),
    )

    return {
        "ease_factor": ease_factor,
        "interval_days": interval_days,
        "repetitions": repetitions,
        "next_review_at": datetime.utcnow() + timedelta(days=interval_days),
    }


def create_flashcards(
    db: Session,
    user_id: str,
    source: str,
    cards: list,
    subject: Optional[str] = None,
    source_ref: Optional[str] = None,
) -> int:
    """Bulk-inserts new Flashcard rows, due immediately (never reviewed
    yet). `cards` is a list of {"front": ..., "back": ...} dicts."""
    now = datetime.utcnow()
    rows = [
        Flashcard(
            user_id=user_id,
            subject=subject,
            front=card["front"],
            back=card["back"],
            source=source,
            source_ref=source_ref,
            ease_factor=DEFAULT_EASE_FACTOR,
            interval_days=0,
            repetitions=0,
            next_review_at=now,
        )
        for card in cards
        if card.get("front") and card.get("back")
    ]
    db.add_all(rows)
    db.commit()
    logger.info(f"Created {len(rows)} flashcards for user {user_id} (source={source}, source_ref={source_ref})")
    return len(rows)


def schedule_wrong_quiz_answer(
    db: Session,
    user_id: str,
    quiz_id: str,
    front: str,
    back: str,
    subject: Optional[str] = None,
) -> Flashcard:
    """Creates one Flashcard for a wrongly-answered quiz question and
    immediately applies a quality=0 ("Again") review to it — the exact same
    update function a manual "Again" press uses — so it lands 1 day out,
    same as any other freshly-failed card."""
    card = Flashcard(
        user_id=user_id,
        subject=subject,
        front=front,
        back=back,
        source="quiz",
        source_ref=quiz_id,
        ease_factor=DEFAULT_EASE_FACTOR,
        interval_days=0,
        repetitions=0,
        next_review_at=datetime.utcnow(),
    )
    db.add(card)

    update = apply_sm2_review(card.ease_factor, card.interval_days, card.repetitions, quality=0)
    card.ease_factor = update["ease_factor"]
    card.interval_days = update["interval_days"]
    card.repetitions = update["repetitions"]
    card.next_review_at = update["next_review_at"]

    db.commit()
    return card


def get_due_cards(db: Session, user_id: str, limit: int = DUE_CARDS_LIMIT) -> list:
    """Cards due for review now, oldest-due-first, capped at `limit`."""
    return (
        db.query(Flashcard)
        .filter(Flashcard.user_id == user_id, Flashcard.next_review_at <= datetime.utcnow())
        .order_by(Flashcard.next_review_at.asc())
        .limit(limit)
        .all()
    )
