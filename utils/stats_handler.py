# stats_handler.py
"""
Derives the motivational/gamification stats (study streak, quiz/chat/file
totals, badge unlocks) shown on the frontend's Overview tab.

Almost everything here is computed fresh from existing tables (TokenUsage's
sibling tables QuizRecord/ChatSession, plus an uploads/ filesystem walk) —
no new schema needed for the numbers themselves. The only persisted state is
`UserStats.last_seen_*`, which tracks which milestones the user has already
been shown so a celebration only fires once (see utils/db_models.py).
"""
import os
from datetime import datetime, timedelta

from sqlalchemy import func
from sqlalchemy.orm import Session

from utils.db_models import QuizRecord, ChatSession, UserStats
from utils.logger import get_logger

logger = get_logger(__name__, "stats.log")

# Streak-day milestones that trigger a celebration.
STREAK_MILESTONES = [3, 7, 14, 30, 60, 100]

# Badge criteria — kept intentionally small and derivable from data that's
# already persisted. (A "perfect score" badge isn't included: quiz grading
# results are computed on the fly in /quizzes/{id}/grade and never written
# back to the database, so there's nothing to derive it from.)
BADGE_DEFINITIONS = {
    "first_quiz": {"label": "First Quiz", "description": "Completed your first quiz."},
    "ten_quizzes": {"label": "Quiz Regular", "description": "Completed 10 quizzes."},
    "five_files": {"label": "Well Stocked", "description": "Uploaded 5 study documents."},
}


def _upload_dir(user_id: str) -> str:
    return os.path.join("uploads", user_id)


def _activity_dates(db: Session, user_id: str) -> set:
    """Every calendar date (UTC) the user did something countable toward a
    study streak: took a quiz, sent a chat message, or uploaded a file."""
    dates = set()

    for (ts,) in db.query(QuizRecord.created_at).filter(QuizRecord.user_id == user_id).all():
        if ts:
            dates.add(ts.date())

    # ChatSession.updated_at moves forward on every message appended to the
    # session (see main.py's /chat handler), so it's a reasonable proxy for
    # "sent a chat message today", not just "created a chat".
    for (ts,) in db.query(ChatSession.updated_at).filter(ChatSession.user_id == user_id).all():
        if ts:
            dates.add(ts.date())

    upload_dir = _upload_dir(user_id)
    if os.path.isdir(upload_dir):
        for root, _, files in os.walk(upload_dir):
            for filename in files:
                if filename.startswith("."):
                    continue
                try:
                    mtime = datetime.utcfromtimestamp(os.path.getmtime(os.path.join(root, filename)))
                    dates.add(mtime.date())
                except OSError:
                    continue

    return dates


def _compute_streak(activity_dates: set):
    """Returns (current_streak, longest_streak) in days.

    Current streak counts backward from today; if there's no activity yet
    today, it counts from yesterday instead, so the streak isn't shown as
    broken before the current day is even over.
    """
    if not activity_dates:
        return 0, 0

    today = datetime.utcnow().date()

    current = 0
    cursor = today if today in activity_dates else today - timedelta(days=1)
    while cursor in activity_dates:
        current += 1
        cursor -= timedelta(days=1)

    longest = 0
    run = 0
    for d in sorted(activity_dates):
        if (d - timedelta(days=1)) in activity_dates:
            run += 1
        else:
            run = 1
        longest = max(longest, run)

    return current, max(longest, current)


def _get_or_create_stats_row(db: Session, user_id: str) -> UserStats:
    row = db.query(UserStats).filter(UserStats.user_id == user_id).first()
    if not row:
        row = UserStats(user_id=user_id)
        db.add(row)
        db.commit()
        db.refresh(row)
    return row


def get_stats_summary(db: Session, user_id: str) -> dict:
    activity_dates = _activity_dates(db, user_id)
    current_streak, longest_streak = _compute_streak(activity_dates)
    # Powers a gentle loss-aversion cue on the frontend's streak chip (amber,
    # not punitive) when today has no activity yet but the streak is still
    # alive (see _compute_streak's yesterday-fallback) — distinct from the
    # streak actually having lapsed to 0.
    studied_today = datetime.utcnow().date() in activity_dates

    quiz_count = db.query(func.count(QuizRecord.id)).filter(QuizRecord.user_id == user_id).scalar() or 0
    chat_count = db.query(func.count(ChatSession.id)).filter(ChatSession.user_id == user_id).scalar() or 0

    file_count = 0
    upload_dir = _upload_dir(user_id)
    if os.path.isdir(upload_dir):
        for _, _, files in os.walk(upload_dir):
            file_count += len([f for f in files if not f.startswith(".")])

    unlocked_badges = []
    if quiz_count >= 1:
        unlocked_badges.append("first_quiz")
    if quiz_count >= 10:
        unlocked_badges.append("ten_quizzes")
    if file_count >= 5:
        unlocked_badges.append("five_files")

    stats_row = _get_or_create_stats_row(db, user_id)
    previously_seen_badges = list(stats_row.last_seen_badge_ids or [])

    newly_unlocked_streaks = [
        m for m in STREAK_MILESTONES if m <= current_streak and m > stats_row.last_seen_streak_milestone
    ]
    newly_unlocked_badges = [b for b in unlocked_badges if b not in previously_seen_badges]

    return {
        "current_streak": current_streak,
        "longest_streak": longest_streak,
        "studied_today": studied_today,
        "quiz_count": quiz_count,
        "chat_count": chat_count,
        "file_count": file_count,
        "badges": [
            {"id": b, **BADGE_DEFINITIONS[b]} for b in unlocked_badges
        ],
        "newly_unlocked": {
            "streaks": newly_unlocked_streaks,
            "badges": newly_unlocked_badges,
        },
    }


def ack_milestone(db: Session, user_id: str, milestone_type: str, milestone_id) -> None:
    """Records that a celebration was actually shown to the user, so it
    doesn't fire again on the next stats read. Called from the frontend only
    when the celebration UI renders and is dismissed — never from a GET."""
    stats_row = _get_or_create_stats_row(db, user_id)

    if milestone_type == "streak":
        try:
            milestone_days = int(milestone_id)
        except (TypeError, ValueError):
            logger.warning(f"Ignoring invalid streak milestone ack: {milestone_id!r}")
            return
        stats_row.last_seen_streak_milestone = max(stats_row.last_seen_streak_milestone, milestone_days)
    elif milestone_type == "badge":
        current = list(stats_row.last_seen_badge_ids or [])
        if milestone_id not in current:
            current.append(milestone_id)
            stats_row.last_seen_badge_ids = current
    else:
        logger.warning(f"Ignoring unknown milestone type: {milestone_type!r}")
        return

    db.commit()
